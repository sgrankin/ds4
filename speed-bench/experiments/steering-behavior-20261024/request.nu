#!/usr/bin/env nu
# Run against one fixed steering condition; every request has fresh context.
const ROOT = path self ../../..

def main [
    base: string
    out: path
    family: string
    mode: string
    strength: float
    --cases: string = '' # Comma-separated zero-based prompt IDs; default all.
    --tokens: int = 0 # Override the family allowance for bounded slow-model checks.
] {
    cd $ROOT
    let out = $out | path expand
    mkdir $out
    mut model = ''
    for _ in 1..120 {
        let found = try { (http get --max-time 2sec $'($base)/v1/models').data.0.id } catch { null }
        if $found != null { $model = $found; break }
        sleep 250ms
    }
    if $model == '' { error make {msg: 'Server did not become ready'} }
    let prompts = open --raw 'speed-bench/experiments/steering-behavior-20261024/corpus/prompts.txt' | lines | enumerate
    let selected = if $cases == '' { $prompts } else {
        let ids = $cases | split row ',' | into int
        $prompts | where {|case| $case.index in $ids }
    }
    mut results = []
    for case in $selected {
        let request = {
            model: $model, input: $case.item, instructions: 'You are a helpful assistant.'
            store: false, temperature: 0
            max_output_tokens: (if $tokens > 0 { $tokens } else if $family == flint { 2048 } else { 512 })
            reasoning: (if $family == flint { {summary: detailed} } else { {effort: none} })
        }
        $request | to json | save --force ($out | path join $'($case.index).request.json')
        let start = date now
        let response = http post --max-time 240sec --content-type application/json $'($base)/v1/responses' $request
        $response | to json | save --force ($out | path join $'($case.index).response.json')
        let answer = $response.output | where type == message | each {|item|
            $item.content | where type == output_text | get text | str join (char nl)
        } | str join (char nl)
        let reasoning = $response.output | where type == reasoning | each {|item|
            $item.summary | get text | str join (char nl)
        } | str join (char nl)
        let record = {
            family: $family, mode: $mode, strength: $strength, case: $case.index
            prompt: $case.item, status: $response.status
            incomplete: ($response | get -o incomplete_details)
            answer: $answer, words: ($answer | str stats | get words)
            reasoning_words: ($reasoning | str stats | get words)
            usage: $response.usage, elapsed: ((date now) - $start | into string)
        }
        $results = $results | append $record
        $results | to json | save --force ($out | path join results.json)
        print ($record | select family mode strength case status words reasoning_words | to json --raw)
    }
}
