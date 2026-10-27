#!/usr/bin/env nu
# One already-running guarded server; requests/results stay local and ignored.
use std/assert

def text-answer [response: record] {
    $response.output | where type == message | each {|item|
        $item.content | where type == output_text | get text | str join ''
    } | str join '' | str trim
}

def expected-value [] {
    {
        text: 'literal </think> </｜DSML｜ parameter> </arg_value> &lt; café'
        mixed: 1
        nullable: null
    }
}

def cases [model: string, family: string, temperature: float] {
    let expected = expected-value
    let schema = {
        type: object
        '$defs': {literal: {type: string, const: $expected.text}}
        properties: {
            text: {'$ref': '#/$defs/literal'}
            mixed: {enum: [1, 'x']}
            nullable: {type: [string, 'null']}
        }
        required: [text, mixed, nullable]
        additionalProperties: false
    }
    let common = {
        model: $model, store: false, temperature: $temperature,
        top_p: (if $temperature == 0 { 1.0 } else { 0.95 }),
        top_k: (if $temperature == 0 { 0 } else { 20 }),
        min_p: (if $family == 'deepseek' { 0.05 } else { 0.0 }),
        max_output_tokens: 2048
    }
    let normal = if $family == 'deepseek' {
        $common | insert reasoning {effort: high, summary: detailed}
    } else { $common | insert reasoning {summary: detailed} }
    let ordinary_expected = {numbers: (0..15 | each {|n| $n }), text: $expected.text}
    let ordinary = $normal | insert input $'Return this exact JSON value, without extra prose: ($ordinary_expected | to json -r)'
    let instruction = $'Use echo once with text equal to ($expected.text | to json -r), mixed equal to integer 1, and nullable equal to null.'
    let tool = $normal | insert tools [{
        type: function, name: echo, strict: true,
        description: 'Echo the supplied fields using their exact JSON types.', parameters: $schema
    }] | insert input $instruction
    let formatted = $normal | insert text {format: {type: json_schema, name: exact_value, schema: $schema, strict: true}} |
        insert input $'Return this exact JSON object: ($expected | to json -r)'
    [
        {name: ordinary, body: $ordinary, expected: $ordinary_expected}
        {name: strict-tool, body: $tool, expected: $expected}
        {name: schema-answer, body: $formatted, expected: $expected}
    ]
}

def validate [response: record, case: record] {
    assert equal $response.status completed
    let actual = if $case.name == 'strict-tool' {
        let calls = $response.output | where type == function_call
        assert equal ($calls | length) 1
        assert equal $calls.0.name echo
        $calls.0.arguments | from json
    } else { text-answer $response | from json }
    assert equal $actual $case.expected
}

def request [base: string, output: path, label: string, case: record, warmup: bool] {
    $case.body | to json | save -f ($output | path join $'($label).request.json')
    let started = date now
    try {
        let reply = http post --full --raw --allow-errors --max-time 90sec --content-type application/json $'($base)/v1/responses' $case.body
        let seconds = (((date now) - $started) | into int) / 1e9
        let response = $reply.body | from json
        $response | to json | save -f ($output | path join $'($label).response.json')
        let grade = try {
            assert equal $reply.status 200
            validate $response $case
            {quality: true, error: null}
        } catch {|failure| {quality: false, error: $failure.msg} }
        {
            name: $label, case: $case.name, temperature: $case.body.temperature,
            warmup: $warmup, seconds: $seconds, status: $reply.status,
            quality: $grade.quality, error: $grade.error,
            usage: ($response | get -o usage), response_id: ($response | get -o id),
            seed_policy: 'Responses has no implemented seed control; T0 is deterministic, T1 is stochastic'
        }
    } catch {|failure|
        {name: $label, case: $case.name, temperature: $case.body.temperature, warmup: $warmup,
         seconds: ((((date now) - $started) | into int) / 1e9), status: null,
         quality: false, error: $failure.msg, usage: null, response_id: null,
         seed_policy: 'Responses has no implemented seed control; T0 is deterministic, T1 is stochastic'}
    }
}

def stream-probe [base: string, output: path, case: record] {
    let body = $case.body | insert stream true
    $body | to json | save -f ($output | path join stream.request.json)
    let reply = http post --full --raw --allow-errors --max-time 90sec --content-type application/json $'($base)/v1/responses' $body
    $reply.body | save -f ($output | path join stream.sse)
    assert equal $reply.status 200
    # This server emits one JSON object per data line. Let the standard JSON
    # parser check each frame; event names/comments do not carry payload data.
    let events = $reply.body | lines | where {|line| $line | str starts-with 'data: '} |
        each {|line| $line | str substring 6.. } | where $it != '[DONE]' | each { from json }
    $events | to json | save -f ($output | path join stream.events.json)
    let completed = $events | where type == response.completed
    assert equal ($completed | length) 1
    validate $completed.0.response $case
    let call = $completed.0.response.output | where type == function_call | first
    let deltas = $events | where type == response.function_call_arguments.delta
    assert (($deltas | length) > 0)
    assert equal ($deltas | get delta | str join '') $call.arguments
    assert ($deltas | all {|event| $event.item_id == $call.id })
    let done = $events | where type == response.function_call_arguments.done
    assert equal ($done | length) 1
    assert equal $done.0.arguments $call.arguments
    {quality: true, delta_count: ($deltas | length), item_id: $call.id}
}

def main [base: string, model: string, output: path,
          --family: string = 'flint', --repeats: int = 2,
          --no-warmup, --stream, --dry-run] {
    assert ($family in [flint, deepseek])
    assert ($repeats > 0)
    mkdir $output
    let greedy = cases $model $family 0.0
    if $dry_run {
        for temperature in [0.0, 1.0] {
            for case in (cases $model $family $temperature) {
                $case.body | to json | save -f ($output | path join $'T($temperature)-($case.name).request.json')
            }
        }
        print 'Saved six request fixtures; no network or model activity.'
        return
    }
    mut results = []
    if not $no_warmup {
        for case in $greedy {
            $results = $results | append (request $base $output $'warm-($case.name)' $case true)
        }
    }
    for repeat in 0..<$repeats {
        # Alternate order so both temperatures see early/late runs.
        let temperatures = if ($repeat mod 2) == 0 { [0.0, 1.0] } else { [1.0, 0.0] }
        for temperature in $temperatures {
            for case in (cases $model $family $temperature) {
                # Greedy covers all three fixed cases. Stochastic sampling is
                # one strict-tool quality probe, not a seeded equality claim.
                if $temperature != 0 and $case.name != 'strict-tool' { continue }
                let label = $'r($repeat)-T($temperature)-($case.name)'
                let result = request $base $output $label $case false
                $results = $results | append $result
                $results | to json | save -f ($output | path join results.json)
                print ($result | select name seconds quality usage)
            }
        }
    }
    if $stream {
        let streamed = try { stream-probe $base $output ($greedy | where name == strict-tool | first) } catch {|failure| {quality: false, error: $failure.msg} }
        $streamed | to json | save -f ($output | path join stream.result.json)
    }
    $results | to json | save -f ($output | path join results.json)
    print ($results | where warmup == false | select name seconds quality)
}
