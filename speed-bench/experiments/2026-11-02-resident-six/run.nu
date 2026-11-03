#!/usr/bin/env nu
# Run from the repository root, under speed-bench/ideas/pressure_run.py.
# Timing and logit capture are separate: dumping logits changes decode timing.
def main [out: path, stage: string = 'timing'] {
    mkdir $out
    for key in ($env | columns | where {|k| $k | str starts-with 'DS4'}) {
        hide-env $key
    }
    let policies = [
        {name: native, cap: 3, mass: '.25', total: '.50', gap: '0', weights: '0'}
        {name: preset3, cap: 3, mass: '.25', total: '.50', gap: '.90', weights: '1'}
        {name: cap6_mass50, cap: 6, mass: '.25', total: '.50', gap: '.90', weights: '1'}
        {name: cap4, cap: 4, mass: '1', total: '1', gap: '.90', weights: '1'}
        {name: cap5, cap: 5, mass: '1', total: '1', gap: '.90', weights: '1'}
        {name: cap6, cap: 6, mass: '1', total: '1', gap: '.90', weights: '1'}
        {name: cap6_gap10, cap: 6, mass: '1', total: '1', gap: '.10', weights: '1'}
    ]
    let selected = if $stage == 'smoke' { $policies | where name == cap6 } else { $policies }
    let repeats = if $stage == 'timing' { 5 } else { 1 }
    $policies | to json | save ($out | path join 'policies.json')
    for repeat in 1..$repeats {
        let order = if ($repeat mod 2) == 0 { $selected | reverse } else { $selected }
        for policy in $order {
            let dest = $out | path join $'($repeat)-($policy.name)'
            mkdir $dest
            let config = {
                DS4_V41_COMPRESSED_KV_FP4: '1'
                DS4_V41_INDEX_FP4: '1'
                DS4_V41_INDEX_SIMD: '1'
                DS4_METAL_V41_SELECTED_TILE: '64'
                DS4_METAL_V41_SELECTED_LIMIT: '1024'
                DS4_METAL_V41_RESIDENT_ROUTE_GAP: $policy.gap
                DS4_METAL_V41_RESIDENT_PREFILL_GAP: $policy.gap
                DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS: $policy.weights
                DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS: ($policy.cap | into string)
                DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK: '64'
                DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS: $policy.mass
                DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS: $policy.total
                DS4_METAL_V41_RESIDENT_ROUTE_POLICY: 'replace'
            }
            $config | to json | save ($dest | path join 'environment.json')
            mut args = [
                --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
                --metal --ssd-streaming --ssd-streaming-cache-experts '48GB'
                --prompt-file speed-bench/promessi_sposi.txt
                --ctx-start 2048 --ctx-max 2560 --step-incr 512
                --ctx-alloc 4096 --prefill-chunk 2048
                --gen-tokens 32 --teacher-forced-decode
                --csv ($dest | path join 'timings.csv')
            ]
            if $stage == 'quality' {
                mkdir ($dest | path join 'logits')
                $args = $args | append [--dump-decode-logits-dir ($dest | path join 'logits')]
            }
            let binary = if $stage == 'answers' {
                $args = [
                    --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
                    --out $dest --rows 0 --tokens 2048 --chunk 2048 --ctx 4096
                    --case all --thinking 1 --max-new 384 --cache-gb 48 --resident-env 1
                ]
                './zig-out/bin/decoder-replay-bench'
            } else { './zig-out/bin/ds4-bench' }
            print $'start ($repeat)/($repeats) ($policy.name) (date now)'
            let start = date now
            let command_args = $args
            with-env $config {
                ^$binary ...$command_args out> ($dest | path join 'stdout.log') err> ($dest | path join 'stderr.log')
                if $env.LAST_EXIT_CODE != 0 { error make {msg: $'model failed: ($dest)'} }
            }
            {repeat: $repeat, policy: $policy.name, wall_s: (((date now) - $start) / 1sec)} | to json | save ($dest | path join 'wall.json')
            print $'done ($policy.name) (((date now) - $start) / 1sec)s'
        }
    }
}
