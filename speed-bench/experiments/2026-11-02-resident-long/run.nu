#!/usr/bin/env nu
# Quality diagnostics only: logit capture invalidates benchmark decode timing.
# Exactly 1024 appended tokens select a native full sweep. The corpus stages
# isolate decode approximation; replay and 512-token answer tails cover prefill.
def main [out: path, stage: string = 'quality'] {
    mkdir $out
    for key in ($env | columns | where {|k| $k | str starts-with 'DS4'}) { hide-env $key }
    let policies = [
        {name: native, cap: '3', mass: '.25', total: '.50', gap: '0', weights: '0'}
        {name: preset3, cap: '3', mass: '.25', total: '.50', gap: '.90', weights: '1'}
        {name: cap5, cap: '5', mass: '1', total: '1', gap: '.90', weights: '1'}
    ]
    let workloads = if $stage == 'answers' { [{name: facts, prompt: ''}] } else if $stage == 'replay' { [{name: agent, prompt: ''}] } else { [
        {name: prose, prompt: speed-bench/promessi_sposi.txt}
        {name: code, prompt: src/server/generation.zig}
    ] }
    $policies | to json | save ($out | path join 'policies.json')
    for workload in $workloads {
        let order = if $workload.name == code { $policies | reverse } else { $policies }
        for policy in $order {
            let dest = $out | path join $'($workload.name)-($policy.name)'
            mkdir ($dest | path join 'logits')
            let settings = {
                DS4_V41_COMPRESSED_KV_FP4: '1', DS4_V41_INDEX_FP4: '1', DS4_V41_INDEX_SIMD: '1'
                DS4_METAL_V41_SELECTED_TILE: '64', DS4_METAL_V41_SELECTED_LIMIT: '1024'
                DS4_METAL_V41_RESIDENT_ROUTE_GAP: $policy.gap
                DS4_METAL_V41_RESIDENT_PREFILL_GAP: $policy.gap
                DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS: $policy.weights
                DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS: $policy.cap
                DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK: '64'
                DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS: $policy.mass
                DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS: $policy.total
                DS4_METAL_V41_RESIDENT_ROUTE_POLICY: 'replace'
            }
            let config = if $stage == 'replay' { $settings | merge {
                DS4_REPLAY_CONTEXT: '16384', DS4_REPLAY_CACHE_GB: '48', DS4_REPLAY_PREFILL_CHUNK: '8192'
                DS4_REPLAY_TEACHER_DIR: ($dest | path join 'logits')
                DS4_REPLAY_SKIP_FINAL_SNAPSHOT: '1'
            } } else { $settings }
            $config | to json | save ($dest | path join 'environment.json')
            let args = if $stage == 'answers' { [
                --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
                --out $dest --rows 0 --tokens 8704 --chunk 8192 --ctx 16384
                --warm-tail 512 --case all --thinking 1 --max-new 512 --cache-gb 48 --resident-env 1
            ] } else if $stage == 'replay' { [
                gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
                speed-bench/agent-session/session-v1.txt
                ($dest | path join 'phase-logits.bin') ($dest | path join 'unused-snapshot.bin')
            ] } else { [
                --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
                --metal --ssd-streaming --ssd-streaming-cache-experts '48GB'
                --prompt-file $workload.prompt
                --ctx-start 8192 --ctx-max 9216 --step-incr 1024
                --ctx-alloc 16384 --prefill-chunk 8192 --gen-tokens 1024 --teacher-forced-decode
                --dump-decode-logits-dir ($dest | path join 'logits')
                --csv ($dest | path join 'diagnostic-timing.csv')
            ] }
            let binary = if $stage == 'answers' { './zig-out/bin/decoder-replay-bench' } else if $stage == 'replay' { './zig-out/bin/agent-session-replay' } else { './zig-out/bin/ds4-bench' }
            print $'start ($workload.name)/($policy.name) (date now)'
            with-env $config {
                ^$binary ...$args out> ($dest | path join 'stdout.log') err> ($dest | path join 'stderr.log')
                if $env.LAST_EXIT_CODE != 0 { error make {msg: $'model failed: ($dest)'} }
            }
            print $'done ($workload.name)/($policy.name) (date now)'
        }
    }
}
