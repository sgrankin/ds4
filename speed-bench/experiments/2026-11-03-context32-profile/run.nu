#!/usr/bin/env nu
# Run from the repository root under the existing pressure guard.
def main [out: path, stage: string = 'timing'] {
    mkdir $out
    for key in ($env | columns | where {|key| $key | str starts-with 'DS4'}) { hide-env $key }
    let common = {
        DS4_V41_COMPRESSED_KV_FP4: '1', DS4_V41_INDEX_FP4: '1', DS4_V41_INDEX_SIMD: '1'
        DS4_METAL_V41_SELECTED_TILE: '64', DS4_METAL_V41_SELECTED_LIMIT: '1024'
        DS4_METAL_V41_RESIDENT_ROUTE_GAP: '.90', DS4_METAL_V41_RESIDENT_PREFILL_GAP: '.90'
        DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS: '1'
        DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS: '5', DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK: '64'
        DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS: '1', DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS: '1'
        DS4_METAL_V41_RESIDENT_ROUTE_POLICY: 'replace'
    }
    let repeats = if $stage == 'timing' { 5 } else { 1 }
    for repeat in 1..$repeats {
        let chunks = if $stage != 'timing' { [8192] } else if ($repeat mod 2) == 0 { [8192 2048] } else { [2048 8192] }
        for chunk in $chunks {
            let dest = $out | path join $'($repeat)-chunk($chunk)'
            mkdir $dest
            let coarse = if $stage in ['coarse' 'profile'] { {
                DS4_METAL_GRAPH_PREFILL_PROFILE: '1'
                DS4_METAL_PIPELINE_CREATE_PROFILE: '1'
                DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE: '1'
                DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: '1'
                DS4_METAL_GPU_BUSY_PROFILE: '1'
            } } else { {} }
            let profile = if $stage == 'profile' { $coarse | merge {
                DS4_METAL_CB_TIMES: '1'
                DS4_METAL_ENCODER_TIMELINE: ($dest | path join 'timeline.txt')
            } } else { $coarse }
            let settings = $common | merge $profile
            $settings | to json | save ($dest | path join 'environment.json')
            let args = [
                --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
                --metal --ssd-streaming --ssd-streaming-cache-experts '64GB'
                --prompt-file speed-bench/promessi_sposi.txt
                --ctx-start 32768 --ctx-max 33280 --step-incr 512 --ctx-alloc 40960
                --prefill-chunk ($chunk | into string) --gen-tokens 128 --teacher-forced-decode
                --csv ($dest | path join 'timings.csv')
            ]
            $args | to json | save ($dest | path join 'arguments.json')
            print $'start ($repeat)/($repeats) chunk=($chunk) (date now)'
            let started = date now
            with-env $settings {
                ./zig-out/bin/ds4-bench ...$args out> ($dest | path join 'stdout.log') err> ($dest | path join 'stderr.log')
                if $env.LAST_EXIT_CODE != 0 { error make {msg: $'Benchmark failed: ($dest)'} }
            }
            {repeat: $repeat, chunk: $chunk, wall_s: (((date now) - $started) / 1sec)} | to json | save ($dest | path join 'wall.json')
            print $'done chunk=($chunk) (((date now) - $started) / 1sec)s'
        }
    }
}
