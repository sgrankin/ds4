#!/usr/bin/env nu
# JSON-only diagnostic inputs; no model/server lifecycle.
use std/assert

def main [fixtures: path, output: path] {
    if ($output | path exists) { error make {msg: 'Use a fresh output directory'} }
    let source = open ($fixtures | path join edit-0.request.json)
    let edited = open ($fixtures | path join edit-1.request.json)
    let evidence = 1..96 | each {|i| $'Retained registry evidence ($i): the parser was inspected and the checked registry is authoritative.' } | str join (char nl)
    let registry = [$evidence 'Final authoritative registry: CODE=CEDAR-4826.'] | str join (char nl)
    let task = 'Two parallel read_file results are supplied. The registry-c.txt result is the final authoritative registry; registry-a.txt is historical. Report exactly CODE=value from the final authoritative registry. All required data are present; answer without another tool call.'
    let input = [
        {role: user, content: $task}
        {type: function_call, call_id: read-a, name: read_file, arguments: '{"path":"registry-a.txt"}'}
        {type: function_call, call_id: read-c, name: read_file, arguments: '{"path":"registry-c.txt"}'}
        {type: function_call_output, call_id: read-a, output: $source.input.2.output}
        {type: function_call_output, call_id: read-c, output: $registry}
    ]
    let initial = $source | update prompt_cache_key 'ongoing-parallel-smoke' | update input $input
    let next = $initial | update input.3.output $edited.input.2.output
    assert equal $initial.input.0 $next.input.0
    assert equal $initial.tools $next.tools
    assert equal $initial.input.4 $next.input.4
    assert ($initial.input.3.output != $next.input.3.output)
    mkdir $output
    $initial | to json | save --force ($output | path join initial.request.json)
    $next | to json | save --force ($output | path join edited.request.json)
    {expected: 'CODE=CEDAR-4826', reasoning: high,
        sole_input_edit: 'The first tool result is trimmed; real controls, tool declarations and the second result are unchanged.',
        expected_render: 'One User wrapper opens before both parallel tool results. The final Assistant cue follows the retained second result.',
        policy_check: 'With boundaries=1 fixed, ongoing=0 rejects post-wrapper C, ongoing=1 may import it; prove actual selection/fallback in logs. This is a diagnostic, not a five-sample latency comparison.'}
    | to json | save --force ($output | path join manifest.json)
}
