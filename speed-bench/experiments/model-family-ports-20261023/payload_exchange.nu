#!/usr/bin/env nu
# Run inside the existing pressure guard: one process owns one model at a time.
const ROOT = path self ../../..

# Exchange canonical payloads control -> candidate -> control, then compute a
# fresh teacher token after each import. Raw arrays stay in the chosen output.
def main [family: string, control: path, candidate: path, output: path, --model: path] {
    cd $ROOT
    let installed = match $family {
        qwen => 'gguf/Qwen3.8-Flash-Next-Q4.gguf'
        glm => 'gguf/GLM-5.3-Flash-Q2.gguf'
        _ => { error make {msg: 'family must be qwen or glm'} }
    }
    let weights = if $model == null { $installed | path expand } else { $model | path expand }
    let before = $control | path expand
    let after = $candidate | path expand
    let out = $output | path expand
    if ($out | path exists) { error make {msg: $'Output already exists: ($out)'} }
    mkdir $out
    for name in ($env | columns | where {|name| ($name | str starts-with DS4) or ($name | str starts-with PT_)}) { hide-env $name }
    mut results = []
    for item in [
        {name: control, binary: $before, source: null}
        {name: candidate, binary: $after, source: ($out | path join control)}
        {name: control-return, binary: $before, source: ($out | path join candidate)}
    ] {
        let folder = $out | path join $item.name
        mkdir $folder
        mut args = [--family $family --model $weights --out $folder]
        if $item.source != null { $args = $args | append [--source $item.source] }
        if $family == glm and $item.name == candidate { $args = $args | append [--check-early-invalidation] }
        let command = [$item.binary] | append $args
        $command | to json | save ($folder | path join command.json)
        let started = date now
        ^$item.binary ...$args out> ($folder | path join stdout.log) err> ($folder | path join stderr.log)
        let code = $env.LAST_EXIT_CODE
        let records = open --raw ($folder | path join results.jsonl) | lines | where {|line| $line != ''} | each {|line| $line | from json}
        let frontiers = $records | where kind == frontier | get position
        let passed = $code == 0 and ($records | last | get pass) and $frontiers == [1 3 4 5 129]
        $results = $results | append {name: $item.name, command: $command, returncode: $code,
            elapsed: ((date now) - $started | into string), binary_sha256: (open --raw $item.binary | hash sha256),
            pass: $passed, records: $records}
        {family: $family, measured_latency: false, runs: $results} | to json | save --force ($out | path join results.json)
        print $'($item.name): ($passed)'
        if not $passed { error make {msg: $'Payload exchange failed: ($item.name), see ($folder)'} }
    }
}
