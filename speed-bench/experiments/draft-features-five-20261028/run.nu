#!/usr/bin/env nu
# One pressure_run.py owns this driver and every sequential server child.
use std/assert
const SCRIPT = path self

def post [
    base: string
    dir: path
    name: string
    body: record
    case: string
    measured: bool
    capped: bool
] {
    $body | to json | save ($dir | path join $'($name).request.json')
    let start = date now
    let reply = http post --full --raw --allow-errors --max-time 180sec --content-type application/json $'($base)/v1/responses' $body
    let seconds = (((date now) - $start) | into int) / 1e9
    $reply.body | save ($dir | path join $'($name).response.json')
    assert equal $reply.status 200
    let response = $reply.body | from json
    let expected = {text: 'literal </think> </｜DSML｜ parameter> </arg_value> &lt; café', mixed: 1, nullable: null}
    let quality = if $capped {
        $response.status in [completed incomplete] and $response.usage.output_tokens == 32
    } else {
        try {
            assert equal $response.status completed
            if $case == strict-tool {
                let calls = $response.output | where type == function_call
                assert equal ($calls | length) 1
                assert equal $calls.0.name echo
                assert equal ($calls.0.arguments | from json) $expected
            } else {
                let text = $response.output | where type == message | each {|m| $m.content | where type == output_text | get text | str join '' } | str join ''
                let wanted = if $case == ordinary {
                    {
                        numbers: (0..15 | each {|n| $n})
                        text: $expected.text
                    }
                } else {
                    $expected
                }
                assert equal ($text | from json) $wanted
            }
            true
        } catch { false }
    }
    let row = {
        name: $name
        case: $case
        measured: $measured
        capped: $capped
        seconds: $seconds
        quality: $quality
        status: $response.status
        usage: $response.usage
    }
    print ($row | select name seconds quality usage)
    $row
}

def run-arm [
    binary: path
    shaders: path
    dir: path
    arm: string
    fixtures: path
    port: int
] {
    mkdir ($dir | path join cache) ($dir | path join responses)
    let base = $'http://127.0.0.1:($port)'
    let occupied = try {
        http get --max-time 1sec $'($base)/health' | ignore
        true
    } catch {
        false
    }
    assert (not $occupied)
    mut settings = {
        DS4_STATE_CACHE_DIR: ($dir | path join cache)
        DS4_STATE_CACHE_HOT_MB: '0'
        DS4_STATE_CACHE_SSD_MB: '512'
        DS4_SERVER_RESPONSE_STORE_DIR: ($dir | path join responses)
        DS4_LOCK_FILE: ($dir | path join server.lock)
        DS4_SERVER_CONTEXT_SUFFIX_REUSE: '0'
        DS4_SERVER_CONTEXT_BOUNDARIES: '0'
        DS4_V41_CHECKPOINT_DRAFT_FEATURES: (if $arm == features { '1' } else { '0' })
        DS4_V41_INDEX_FP4: '1'
        DS4_V41_INDEX_SIMD: '1'
        DS4_V41_CANDIDATE_INDEX: '0'
        DS4_V41_COMPRESSED_KV_F16: '0'
        DS4_V41_COMPRESSED_KV_FP4: '0'
        DS4_DSPARK_STATS: '1'
        DS4_DSPARK_SPEC_LOG: '1'
        DS4_METAL_V41_RESIDENT_ROUTE_GAP: '0'
        DS4_METAL_V41_RESIDENT_PREFILL_GAP: '0'
    }
    for source in (glob ($shaders | path join '*.metal')) {
        let key = $'DS4_METAL_($source | path parse | get stem | str uppercase)_SOURCE'
        $settings = ($settings | upsert $key $source)
    }
    let args = [
        --model
        gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
        --mtp-model
        gguf/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf
        --dspark
        --mtp-draft
        6
        --mtp-exact-sampling
        --host
        127.0.0.1
        --port
        $port
        --ctx
        8192
        --tokens
        2048
        --prefill-chunk
        512
        --ssd-streaming
        --ssd-streaming-cache-experts
        '32GB'
        --trace
        ($dir | path join trace.log)
    ]
    {binary: $binary, args: $args, settings: $settings} | to json | save ($dir | path join command.json)
    let frozen_settings = $settings
    let task = job spawn --description $'DSP1 ($arm)' { with-env $frozen_settings { ^$binary ...$args out> ($dir | path join server.stdout) err> ($dir | path join server.stderr) } }
    let result = try {
        mut ready = false
        for attempt in 1..240 {
            $ready = (
                try {
                    http get --max-time 1sec $'($base)/health' | ignore
                    true
                } catch {
                    false
                }
            )
            if $ready { break }
            sleep 500ms
        }
        assert $ready
        mut rows = []
        # Fill each durable endpoint; measured requests then switch contexts,
        # forcing HOT0 restore instead of retaining the previous live graph.
        for case in [ordinary strict-tool schema-answer] {
            let body = open ($fixtures | path join $'T0.0-($case).request.json')
            $rows = ($rows | append (post $base $dir $'seed-($case)' $body $case false false))
            $rows | to json | save --force ($dir | path join results.json)
        }
        for case in [ordinary strict-tool schema-answer ordinary-short] {
            let source = if $case == ordinary-short { 'ordinary' } else { $case }
            let body = open ($fixtures | path join $'T0.0-($source).request.json')
            let request = if $case == ordinary-short {
                $body | update max_output_tokens 32
            } else {
                $body
            }
            let row = post $base $dir $'measure-($case)' $request $case true ($case == ordinary-short)
            assert ($row.usage.input_tokens_details.cached_tokens > 0)
            $rows = ($rows | append $row)
            $rows | to json | save --force ($dir | path join results.json)
        }
        {ok: true}
    } catch {|failure| {ok: false, message: $failure.msg} }
    # Outside timed requests: allow terminal logs/store publication to finish.
    sleep 300ms
    try { job kill $task } catch { }
    mut closed = false
    for attempt in 1..20 {
        $closed = (
            try {
                http get --max-time 1sec $'($base)/health' | ignore
                false
            } catch {
                true
            }
        )
        if $closed { break }
        sleep 100ms
    }
    assert $closed 'Owned server did not stop; abort before loading another model.'
    sleep 300ms
    if not $result.ok { error make {msg: $result.message} }
}

def main [binary: path, output: path, --port: int = 18349] {
    let binary = $binary | path expand
    let output = $output | path expand
    let shaders = $binary | path dirname | path dirname | path join shaders
    assert (not ($output | path exists))
    assert ($shaders | path exists)
    mkdir $output
    cd (
        $SCRIPT
        | path dirname
        | path join ../../..
        | path expand
    )
    let fixtures = $output | path join fixtures
    nu speed-bench/experiments/constrained-packets-20261027/qualify.nu unused deepseek-v4.1-flash $fixtures --family deepseek --dry-run
    {
        pairs: 5
        order: [AB BA AB BA AB]
        control: base
        candidate: features
        confidence: 'Metal default 0.6, no override'
        draft_width: 6
        reasoning: high
        temperature: 0
        native_routes: true
        index_fp4: 1
        index_simd: 1
        main_storage: f32
        suffix_reuse: false
        cache: 'HOT0; fresh directory/server per arm; three seed requests excluded and retained'
        primary: 'HTTP time for each restored complete task; separate32-token capped continuation'
        only_changed_setting: DS4_V41_CHECKPOINT_DRAFT_FEATURES
    } | to json | save ($output | path join manifest.json)
    for pair in 1..5 {
        let arms = if $pair mod 2 == 1 { [base features] } else { [features base] }
        for arm in $arms {
            run-arm $binary $shaders ($output | path join $'pair-($pair)' $arm) $arm $fixtures $port
            print $'Completed pair ($pair) ($arm)'
        }
    }
}
