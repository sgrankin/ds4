#!/usr/bin/env nu
# HTTP workload only. The caller owns one guarded server and its cache lifetime.
use std/assert

def records [label: string, rows: int] {
    1..$rows | each {|i|
        $'($label) row ($i): examined the retained parser evidence, checked its location, and recorded the explanation for later comparison.'
    } | str join "\n"
}

def expected [case: string, edit: int] {
    if $edit == 0 {
        if $case == deleted_user { 'PINE-6204' } else { 'CEDAR-4826' }
    } else {
        match $case {
            corrected => 'LARCH-9062'
            deleted => 'NONE'
            new_user | changed_user => 'MAPLE-7319'
            _ => 'CEDAR-4826'
        }
    }
}

def history [case: string, edit: int, rows: int] {
    let code = expected $case $edit
    let task = if $case == changed_user and $edit > 0 {
        'Changed task: answer exactly CODE=MAPLE-7319. This instruction overrides all historical registry notes.'
    } else {
        'Maintain the parser investigation. For this request, report the code from the final authoritative registry tool result as CODE=value. Historical investigation notes can be stale. If that registry explicitly removed the code, report CODE=NONE. All needed tool results are present; answer without another tool call.'
    }
    let original = ['Original registry: CODE=CEDAR-4826.' (records dumpA $rows)] | str join (char nl)
    let excerpt = $'Compaction revision ($edit). Relevant excerpt: CODE=($code). Original dump is retained in the initial request archive; it has not been destroyed.'
    let investigation = ['Historical investigation recorded CODE=CEDAR-4826; consult the final registry for the current answer.' (records investigationC (if $case == favorable { $rows * 3 } else { $rows * 2 }))] | str join (char nl)
    let later = records $'laterD-revision-($edit)' ($rows * 2)
    let latest = if $case == deleted and $edit > 0 {
        'Final authoritative registry: the code field has been removed. CODE=NONE.'
    } else { $'Final authoritative registry: CODE=($code).' }
    if $case == 'favorable' {
        return [
            {role: user, content: $task}
            {type: function_call, call_id: read-a, name: read_file, arguments: '{"path":"registry-a.txt"}'}
            {type: function_call, call_id: read-c, name: read_file, arguments: '{"path":"investigation-c.txt"}'}
            {type: function_call, call_id: read-d, name: read_file, arguments: '{"path":"registry-d.txt"}'}
            {type: function_call_output, call_id: read-a, output: (if $edit == 0 { $original } else { $excerpt })}
            {type: function_call_output, call_id: read-c, output: $investigation}
            {type: function_call_output, call_id: read-d, output: ([$later $latest] | str join (char nl))}
        ]
    }
    mut items = [
        {role: user, content: $task}
        {type: function_call, call_id: read-a, name: read_file, arguments: '{"path":"registry-a.txt"}'}
        {type: function_call_output, call_id: read-a, output: (if $edit == 0 { $original } else { $excerpt })}
    ]
    if $case == deleted_user and $edit == 0 {
        $items = ($items | append {role: user, content: 'Superseding instruction: answer exactly CODE=PINE-6204, overriding earlier instructions and registry codes.'})
    }
    $items = ($items | append [
        {role: assistant, content: $investigation}
        {type: function_call, call_id: read-d, name: read_file, arguments: '{"path":"registry-d.txt"}'}
        {type: function_call_output, call_id: read-d, output: $later}
        {role: assistant, content: 'The latest registry must decide the answer.'}
        {type: function_call, call_id: final-registry, name: read_file, arguments: '{"path":"final-registry.txt"}'}
        {type: function_call_output, call_id: final-registry, output: $latest}
    ])
    if $case == new_user and $edit > 0 {
        $items = ($items | append {role: user, content: 'New instruction: answer exactly CODE=MAPLE-7319. This overrides prior registry values.'})
    }
    $items
}

def answer [response: record] {
    $response.output | where type == message | each {|item|
        $item.content | where type == output_text | get text | str join ''
    } | str join '' | str trim
}

def main [base: string, model: string, output: path, --case: string = ordinary, --rows: int = 72, --edits: int = 8, --key: string = 'context-boundary-workflow', --context-label: string = '', --fixtures-only, --archive-control] {
    if $case not-in [ordinary favorable corrected deleted new_user changed_user deleted_user] { error make {msg: 'Unknown context-edit case'} }
    if $rows < 1 or $edits < 1 { error make {msg: 'rows and edits must be positive'} }
    if ($output | path exists) { error make {msg: $'Use a fresh output directory: ($output)'} }
    mkdir $output
    {case: $case, rows: $rows, edits: $edits, model: $model, key: $key, context_label: $context_label, reasoning: high, archive_control: $archive_control,
     note: 'Initial population and every edit retained locally. store:false does not add a durable server transcript. Prior measured short answers are not appended to the synthetic investigation. No model lifecycle or feature configuration in this driver.'}
    | to json | save ($output | path join manifest.json)
    mut measurements = []
    for edit in 0..$edits {
        let name = $'edit-($edit)'
        let body = {
            model: $model, store: false, temperature: 0, min_p: 0.05,
            reasoning: {effort: high}, max_output_tokens: 384,
            prompt_cache_key: $key, input: (history $case $edit $rows),
            tools: [{type: function, name: read_file, strict: true,
                parameters: {type: object, properties: {path: {type: string}}, required: [path], additionalProperties: false}}]
        }
        let request = if $context_label == '' { $body } else {
            $body | insert instructions $'Independent qualification context: ($context_label). Follow the current task instructions.'
        }
        $request | to json | save ($output | path join $'($name).request.json')
        if $fixtures_only { continue }
        let start = date now
        let reply = http post --full --raw --allow-errors --max-time 1800sec --content-type application/json $'($base)/v1/responses' $request
        let seconds = (((date now) - $start) | into int) / 1e9
        $reply.body | save ($output | path join $'($name).response.json')
        assert equal $reply.status 200
        let response = $reply.body | from json
        let text = answer $response
        let oracle = $'CODE=(expected $case $edit)'
        let measurement = {edit: $edit, seconds: $seconds, status: $response.status,
            expected: $oracle, answer: $text, correct: ($response.status == completed and $text == $oracle), usage: $response.usage}
        $measurements = ($measurements | append $measurement)
        $measurements | to json | save --force ($output | path join results.json)
        print $measurement
        # Task errors remain measurements; transport/runtime failures are obvious.
        assert ($response.status in [completed incomplete])
    }
    mut archive_result: any = null
    if $archive_control {
        # The client reads the original from its retained request file. This is
        # separate from KV persistence and is not a server-side archive tool.
        let archived = open ($output | path join edit-0.request.json)
        let original = $archived.input | where {|item| $item.type? == 'function_call_output' and $item.call_id? == 'read-a' } | get 0.output
        assert ($original | str starts-with 'Original registry: CODE=CEDAR-4826.')
        let body = $archived | update input [
            {role: user, content: 'The client recovered the original file from its local request archive. Its original code may differ from the current code. Report the original code in this retrieved file as CODE=value. Answer without another tool call.'}
            {type: function_call, call_id: archive-original, name: read_file, arguments: '{"path":"edit-0.request.json"}'}
            {type: function_call_output, call_id: archive-original, output: $original}
        ]
        $body | to json | save ($output | path join archive-control.request.json)
        if not $fixtures_only {
            let start = date now
            let reply = http post --full --raw --allow-errors --max-time 1800sec --content-type application/json $'($base)/v1/responses' $body
            let seconds = (((date now) - $start) | into int) / 1e9
            $reply.body | save ($output | path join archive-control.response.json)
            assert equal $reply.status 200
            let response = $reply.body | from json
            let text = answer $response
            $archive_result = {seconds: $seconds, status: $response.status, answer: $text,
                expected: 'CODE=CEDAR-4826', correct: ($response.status == completed and $text == 'CODE=CEDAR-4826'), usage: $response.usage}
            $archive_result | to json | save ($output | path join archive-control.result.json)
            print $archive_result
            assert ($response.status in [completed incomplete])
        }
    }
    if $fixtures_only { return }
    {initial_seconds: $measurements.0.seconds,
     edited_seconds: ($measurements | skip 1 | get seconds | math sum),
     whole_seconds: ($measurements | get seconds | math sum),
     correct: ($measurements | where correct | length), responses: ($measurements | length), archive_control: $archive_result}
    | to json | save ($output | path join summary.json)
}
