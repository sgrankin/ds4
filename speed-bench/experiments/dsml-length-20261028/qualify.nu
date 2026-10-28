#!/usr/bin/env nu
# Correctness only: one already-running, pressure-guarded server, normal reasoning.
use std/assert

def tool [name: string, payload: string, responses: bool] {
    let declaration = {
        name: $name, description: 'Echo the exact requested text.', strict: true,
        parameters: {type: object, properties: {text: {type: string, const: $payload}},
                     required: [text], additionalProperties: false}
    }
    if $responses { $declaration | insert type function } else { {type: function, function: $declaration} }
}

def body [model: string, family: string, api: string, cap: int, streaming: bool, payload: string] {
    let responses = $api == 'responses'
    let prompt = $'Call echo exactly once with text equal to this complete string. Do not summarize it: ($payload | to json -r)'
    let common = {
        model: $model, temperature: 0.0, top_p: 1.0,
        min_p: (if $family == 'deepseek' { 0.05 } else { 0.0 }),
        tools: [(tool echo $payload $responses)], stream: $streaming
    }
    if $responses {
        let request = $common | insert store true | insert input $prompt | insert max_output_tokens $cap
        if $family == 'deepseek' { $request | insert reasoning {effort: high} } else { $request }
    } else {
        let request = $common | insert messages [{role: user, content: $prompt}] | insert max_tokens $cap
        if $family == 'deepseek' { $request | insert reasoning_effort high } else { $request }
    }
}

# Reuse a known Responses request verbatim apart from model/cap/transport.
# Chat receives the same tool schema and one user input through its wire shape.
def from-fixture [source: record, model: string, api: string, cap: int, streaming: bool] {
    if $api == responses {
        $source | upsert model $model | upsert store true | upsert stream $streaming | upsert max_output_tokens $cap
    } else {
        assert equal ($source.input | describe) string
        let tools = $source.tools | each {|entry| {type: function, function: ($entry | reject type)} }
        let request = $source | reject --optional input max_output_tokens reasoning store |
            upsert model $model | upsert stream $streaming | upsert max_tokens $cap |
            upsert messages [{role: user, content: $source.input}] | upsert tools $tools
        let effort = $source | get -o reasoning.effort
        if $effort == null { $request } else { $request | upsert reasoning_effort $effort }
    }
}

def post [base: string, endpoint: string, request: record, output: path, label: string, --expected: int = 200] {
    $request | to json | save -f ($output | path join $'($label).request.json')
    let reply = http post --full --raw --allow-errors --max-time 180sec --content-type application/json $'($base)/v1/($endpoint)' $request
    $reply.body | save -f ($output | path join $'($label).body')
    assert equal $reply.status $expected
    $reply.body
}

def events [wire: string] {
    $wire | lines | where {|line| $line | str starts-with 'data: '} |
        each {|line| $line | str substring 6.. } | where $it != '[DONE]' | each { from json }
}

def check-responses [response: record, frames: list] {
    assert equal $response.status incomplete
    assert equal $response.incomplete_details.reason max_output_tokens
    assert equal ($response | get -o error) null
    for item in ($response.output | where type == message) {
        let text = $item.content | where type == output_text | get text | str join ''
        assert (not ($text | str contains '<｜DSML｜ calls>'))
        assert (not ($text | str contains '<tool_call>'))
    }
    let calls = $response.output | where type == function_call
    for call in $calls {
        assert ($call.status in [incomplete completed])
        if ($frames | is-not-empty) {
            let deltas = $frames | where type == response.function_call_arguments.delta | where item_id == $call.id
            assert equal ($deltas | get delta | str join '') $call.arguments
            let done = $frames | where type == response.function_call_arguments.done | where item_id == $call.id
            assert equal ($done | length) 1
            assert equal $done.0.arguments $call.arguments
            let ended = $frames | where type == response.output_item.done | where {|e| $e.item.id == $call.id }
            assert equal ($ended | length) 1
            assert equal $ended.0.item.call_id $call.call_id
            assert equal $ended.0.item.status $call.status
            assert equal $ended.0.item.arguments $call.arguments
        }
    }
    {quality: true, partial_calls: ($calls | where status == incomplete | length),
     id: $response.id, call_ids: ($calls | get call_id)}
}

def check-chat [wire: string, streaming: bool] {
    if not $streaming {
        let response = $wire | from json
        assert equal $response.choices.0.finish_reason length
        let visible = $response.choices.0.message | get -o content | default ''
        assert (not ($visible | str contains '<｜DSML｜ calls>'))
        assert (not ($visible | str contains '<tool_call>'))
        let calls = $response.choices.0.message | get -o tool_calls | default []
        for call in $calls { assert equal ($call.function.arguments | describe) string }
        {quality: true, partial_calls: ($calls | length), id: $response.id, call_ids: ($calls | get id)}
    } else {
        let frames = events $wire
        let endings = $frames | each {|frame| $frame.choices | where finish_reason == length } | flatten
        assert equal ($endings | length) 1
        let visible = $frames | each {|frame| $frame.choices | each {|choice| $choice.delta | get -o content | default '' } | str join '' } | str join ''
        assert (not ($visible | str contains '<｜DSML｜ calls>'))
        assert (not ($visible | str contains '<tool_call>'))
        let calls = $frames | each {|frame| $frame.choices | each {|choice| $choice.delta | get -o tool_calls | default [] } | flatten } | flatten
        let started = $calls | where {|call| ($call | get -o id) != null }
        for start in $started {
            let matching = $calls | where index == $start.index
            let fragments = $matching | each {|part| $part.function | get -o arguments | default '' } | str join ''
            assert ($fragments | str starts-with '{')
            # Literal marker spellings inside arguments are payload, not framing.
        }
        {quality: true, partial_calls: ($started | length), id: null, call_ids: ($started | get id)}
    }
}

def main [base: string, model: string, output: path, --family: string = 'deepseek',
          --caps: string = '64,256,512', --fixture: path, --dry-run] {
    assert ($family in [deepseek flint])
    mkdir $output
    if $fixture != null {
        {fixture: ($fixture | path expand)} | to json | save -f ($output | path join fixture-source.json)
    }
    let limits = $caps | split row ',' | each { into int }
    let payload = 1..64 | each { 'λ checkpoint literal </think> data & boundary; ' } | str join ''
    mut results = []
    mut parent: any = null
    for api in [responses chat] {
        for streaming in [false true] {
            for cap in $limits {
                let label = $'($api)-stream($streaming)-cap($cap)'
                let request = if $fixture == null {
                    body $model $family $api $cap $streaming $payload
                } else { from-fixture (open $fixture) $model $api $cap $streaming }
                if $dry_run {
                    $request | to json | save -f ($output | path join $'($label).request.json')
                    continue
                }
                let checked = try {
                    let wire = post $base (if $api == responses { 'responses' } else { 'chat/completions' }) $request $output $label
                    if $api == responses {
                        let frames = if $streaming { events $wire } else { [] }
                        let response = if $streaming {
                            assert equal ($frames | where type == response.failed | length) 0
                            let terminal = $frames | where type == response.incomplete
                            assert equal ($terminal | length) 1
                            $terminal.0.response
                        } else { $wire | from json }
                        check-responses $response $frames
                    } else { check-chat $wire $streaming }
                } catch {|failure| {quality: false, error: $failure.msg, partial_calls: 0, id: null, call_ids: []} }
                let result = $checked | insert label $label
                if $api == responses and $result.quality and $result.partial_calls > 0 { $parent = $result }
                $results = $results | append $result
                $results | to json | save -f ($output | path join results.json)
                print $result
            }
        }
    }
    if $dry_run { print 'Saved requests; no network/model activity.'; return }
    assert ($parent != null) 'No cap reached an actual partial call; retain results and retry a larger cap.'
    # Existing policy rejects every incomplete parent, including a turn cut off
    # during ordinary reasoning. Preserve that explicit boundary; retry from
    # safe client input, with a replacement tool set and no unfinished call.
    mut recovery: any = {
        model: $model, store: true, temperature: 0.0, top_p: 1.0,
        previous_response_id: $parent.id, max_output_tokens: 1024,
        tools: [(tool recover READY true)],
        input: 'Disregard the unfinished request. Call recover exactly once with text equal to READY.'
    }
    if $family == deepseek { $recovery = $recovery | insert reasoning {effort: high} }
    let rejected = post $base responses $recovery $output continuation --expected 400 | from json
    assert ($rejected.error.message | str contains 'previous response is incomplete')
    let safe_retry = $recovery | reject previous_response_id
    let continued = post $base responses $safe_retry $output safe-retry | from json
    assert equal $continued.status completed
    let calls = $continued.output | where type == function_call
    assert equal ($calls | length) 1
    assert equal $calls.0.name recover
    assert equal ($calls.0.arguments | from json) {text: READY}
    assert (not ($calls.0.call_id in $parent.call_ids))
    assert ($results | all {|result| $result.quality })
    print 'Typed truncation, streamed byte/ID consistency, explicit incomplete-parent rejection and safe changed-tool retry: PASS'
}
