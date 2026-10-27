#!/usr/bin/env nu
# Exercise one already-running server; launch it under the existing pressure
# guard. Requests/results are local artifacts, not tracked snapshots.
use std/assert

def request [base: string, output: path, name: string, body: record, expected: int = 200] {
    $body | to json | save -f ($output | path join $'($name).request.json')
    let began = date now
    let reply = http post --full --raw --allow-errors --max-time 300sec --content-type application/json $'($base)/v1/responses' $body
    let seconds = (((date now) - $began) | into int) / 1e9
    let response = $reply.body | from json
    $response | to json | save -f ($output | path join $'($name).response.json')
    assert equal $reply.status $expected
    {name: $name, seconds: $seconds, response: $response}
}

def answer [response: record] {
    $response.output | where type == message | each {|item|
        $item.content | where type == output_text | get text | str join ''
    } | str join '' | str trim
}

def call-arguments [response: record] {
    let calls = $response.output | where type == function_call
    assert equal ($calls | length) 1
    assert equal $calls.0.name 'echo'
    $calls.0.arguments | from json
}

def main [base: string, model: string, output: path, --family: string = 'deepseek', --cache-only, --coercion-only] {
    mkdir $output
    let ordinary = {
        model: $model, store: false, temperature: 0, min_p: 0.05,
        max_output_tokens: 2048, input: 'Reply with exactly READY.'
    }
    # Keep DeepSeek thinking enabled for ordinary and positive quality cases.
    let normal = if $family == 'deepseek' {
        $ordinary | insert reasoning {effort: high}
    } else { $ordinary }
    let literal = 'literal </think> </｜DSML｜ parameter> <arg_value> &lt; café'
    let schema = {
        type: object, properties: {text: {type: string, enum: [$literal]}}
    }
    let tool = {type: function, name: echo, description: 'Echo the exact supplied text.', parameters: $schema, strict: true}
    let tools = $normal | insert tools [$tool] | update input $'Call echo with this exact text: ($literal)'
    mut measurements = []
    if $coercion_only {
        assert equal $family 'flint'
        let types = {
            type: object, '$defs': {truth: {const: 'true'}},
            properties: {
                ref: {'$ref': '#/$defs/truth'},
                mixed: {enum: [1, 'x']},
                nullable: {type: [string, 'null']}
            }
        }
        let typed = $normal | insert tools [{
            type: function, name: echo, strict: true, parameters: $types,
            description: 'Return the exact supplied values with their declared JSON types.'
        }] | insert tool_choice {type: function, name: echo} | update input 'Call echo with ref equal to the string "true", mixed equal to the integer 1, and nullable equal to null.'
        let result = request $base $output 'strict-xml-types' $typed
        assert equal $result.response.status completed
        assert equal (call-arguments $result.response) {ref: 'true', mixed: 1, nullable: null}
        let ambiguous = $typed | update tools.0.parameters {
            type: object, properties: {value: {type: [string, integer], pattern: '^x$'}}
        }
        request $base $output 'ambiguous-string-pattern-rejected' $ambiguous 400 | ignore
        [{name: $result.name, seconds: $result.seconds, quality: true, usage: $result.response.usage}] | to json | save -f ($output | path join results.json)
        print 'PASS Flint referenced string, mixed enum, nullable value and ambiguous-schema rejection'
        return
    }
    if $family == 'deepseek' {
        # Run first on a fresh server/store: no older authorized endpoint can
        # legitimately supply even a shared template prefix to this test.
        let cold = $normal | update input 'Uncached origin CACHE_ORIGIN_7319. Reply with READY.'
        let disabled = request $base $output 'fresh-explicit-no-marks' ($cold | insert prompt_cache_options {mode: explicit})
        assert equal $disabled.response.usage.input_tokens_details.cached_tokens 0
        assert equal $disabled.response.usage.input_tokens_details.cache_write_tokens 0
        let implicit = request $base $output 'implicit-after-no-marks' $cold
        assert equal $implicit.response.usage.input_tokens_details.cached_tokens 0
        let repeated = request $base $output 'implicit-repeat' $cold
        assert ($repeated.response.usage.input_tokens_details.cached_tokens > 0)
    }
    if $family == 'deepseek' {
        let stable = 0..47 | each {|n| $'Stable cache fixture note ($n): this prefix is unchanged. ' } | str join ''
        let block = {type: input_text, text: $stable, prompt_cache_breakpoint: {mode: explicit}}
        let marked = $normal | insert prompt_cache_options {mode: explicit} | update input [
            {role: developer, content: [$block]}, {role: user, content: 'Reply with READY.'}
        ]
        let first = request $base $output 'marked-cache-first' $marked
        let second = request $base $output 'marked-cache-repeat' ($marked | update input.1.content 'Reply with READY again.')
        assert ($second.response.usage.input_tokens_details.cached_tokens >= 512)
        let changed = $marked | update input.0.content.0.text $'Changed earlier prefix. ($stable)'
        let changed_result = request $base $output 'changed-prefix-miss' $changed
        assert equal $changed_result.response.usage.input_tokens_details.cached_tokens 0
        let unmarked = $marked | update input.0.content [{type: input_text, text: $stable}]
        let disabled = request $base $output 'explicit-no-marks' $unmarked
        assert equal $disabled.response.usage.input_tokens_details.cached_tokens 0
        assert equal $disabled.response.usage.input_tokens_details.cache_write_tokens 0
        assert equal $second.response.usage.input_tokens_details.cache_write_tokens 0
        $measurements = $measurements | append {
            name: $second.name, seconds: $second.seconds, quality: true, usage: $second.response.usage
        }
    }
    if $cache_only {
        $measurements | to json | save -f ($output | path join results.json)
        print $measurements
        return
    }
    for repeat in 0..1 {
        let result = request $base $output $'ordinary-($repeat)' $normal
        assert equal $result.response.status completed
        let text = answer $result.response
        $measurements = $measurements | append {
            name: $result.name, seconds: $result.seconds,
            quality: ($text == 'READY'), usage: $result.response.usage
        }
    }
    let result = request $base $output 'strict-tool' $tools
    assert equal $result.response.status completed
    let arguments = call-arguments $result.response
    assert equal $arguments.text $literal
    $measurements = $measurements | append {name: $result.name, seconds: $result.seconds, quality: true, usage: $result.response.usage}

    let format = {type: json_schema, name: exact_text, schema: $schema, strict: true}
    let formatted = $normal | insert text {format: $format} | update input $'Return a JSON object with text equal to: ($literal)'
    let result = request $base $output 'schema-answer' $formatted
    assert equal $result.response.status completed
    let decoded = answer $result.response | from json
    assert equal $decoded.text $literal
    $measurements = $measurements | append {name: $result.name, seconds: $result.seconds, quality: true, usage: $result.response.usage}

    if $family == 'deepseek' {
        request $base $output 'thinking-required-rejected' ($tools | insert tool_choice required) 400 | ignore
        let required = $tools | update reasoning {effort: none} | insert tool_choice {type: function, name: echo}
        # Explicit non-thinking is only an API conformance case, not the normal
        # quality/performance setting above.
        let result = request $base $output 'explicit-nonthinking-named' $required
        assert equal (call-arguments $result.response).text $literal
        $measurements = $measurements | append {name: $result.name, seconds: $result.seconds, quality: true, usage: $result.response.usage}
    }
    request $base $output 'strict-plus-format-rejected' ($tools | insert text {format: $format}) 400 | ignore
    request $base $output 'unknown-choice-rejected' ($normal | insert tool_choice impossible) 400 | ignore
    request $base $output 'background-rejected' ($normal | insert background true) 400 | ignore
    request $base $output 'automatic-truncation-rejected' ($normal | insert truncation auto) 400 | ignore
    request $base $output 'custom-declaration-rejected' ($normal | insert tools [{type: custom, name: raw_code}]) 400 | ignore
    request $base $output 'unsupported-format-rejected' ($normal | insert text {format: {type: magic}}) 400 | ignore
    $measurements | to json | save -f ($output | path join results.json)
    print $measurements
}
