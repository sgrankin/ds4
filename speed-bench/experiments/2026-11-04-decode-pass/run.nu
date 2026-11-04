#!/usr/bin/env nu
# Run from the repository root under speed-bench/ideas/pressure_run.py.
# Compare one selected environment setting against the adopted decode baseline.
# Timing is profile-free; quality and profile stages are diagnostics only.
def main [out: path, stage: string = 'timing', --flag: string = 'DS4_METAL_V41_PACKED_F16_GATHER', --control: string = '0', --control-unset, --candidate: string = '1', --cache: string = '64GB', --frontier: int = 0, --prefill-gap: string = '.90', --only: string = 'both', --samples: int = 5, --tokens: int = 0] {
    if $stage not-in ['timing' 'smoke' 'quality' 'profile' 'timeline' 'cpu-profile'] {
        error make {msg: 'Stage must be timing, smoke, quality, profile, timeline or cpu-profile'}
    }
    if $only not-in ['both' 'baseline' 'candidate'] {
        error make {msg: '--only must be both, baseline or candidate'}
    }
    mkdir $out
    for key in ($env | columns | where {|key| $key | str starts-with 'DS4'}) { hide-env $key }
    let common = {
        DS4_V41_COMPRESSED_KV_FP4: '1', DS4_V41_INDEX_FP4: '1', DS4_V41_INDEX_SIMD: '1'
        DS4_METAL_V41_SELECTED_TILE: '64', DS4_METAL_V41_SELECTED_LIMIT: '1024'
        DS4_METAL_V41_RESIDENT_ROUTE_GAP: '.90', DS4_METAL_V41_RESIDENT_PREFILL_GAP: $prefill_gap
        DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS: '1'
        DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS: '5', DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK: '64'
        DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS: '1', DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS: '1'
        DS4_METAL_V41_RESIDENT_ROUTE_POLICY: 'replace'
    }
    let frontier = if $frontier > 0 { $frontier } else if $stage in ['smoke' 'quality'] { 8192 } else { 32768 }
    let tokens = if $tokens > 0 { $tokens } else if $stage == 'smoke' { 32 } else { 128 }
    let repeats = if $stage == 'timing' { $samples } else { 1 }
    for repeat in 1..$repeats {
        let arms = if ($repeat mod 2) == 0 { ['candidate' 'baseline'] } else { ['baseline' 'candidate'] }
        for arm in $arms {
            if $only != 'both' and $arm != $only { continue }
            let dest = $out | path join $'($repeat)-($arm)'
            mkdir $dest
            let diagnostic = if $stage in ['profile' 'timeline'] { {
                DS4_METAL_GRAPH_PREFILL_PROFILE: '1'
                DS4_METAL_PIPELINE_CREATE_PROFILE: '1'
                DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE: '1'
                DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: '1'
                DS4_METAL_GPU_BUSY_PROFILE: '1'
                DS4_METAL_V41_RESIDENT_ROUTE_PROFILE: '1'
                DS4_METAL_SELECTED_READBACK_PROFILE: '1'
            } } else if $stage == 'quality' { {DS4_BENCH_DECODE_LOGITS_HASH: '1'} } else { {} }
            let settings = $common | merge $diagnostic | merge {
                DS4_METAL_V41_ROUTE_SHARED_OVERLAP: '1'
                DS4_METAL_V41_ROUTE_SHARED_DOWN: '1'
                DS4_METAL_SELECTED_READBACK_SPIN_US: '2000'
                DS4_METAL_V41_STREAMING_ROPE_TABLE: '1'
                DS4_METAL_STREAMING_EXPERT_SLAB_MB: '4096'
                DS4_METAL_V41_SHORT_Q8_QUERY: '1'
                DS4_METAL_V41_SHORT_Q8_QUERY_TABLE: '1'
                DS4_METAL_V41_PACKED_F16_GATHER: '1'
                DS4_METAL_V41_REUSE_SELECTED_KV: '1'
                DS4_METAL_V41_DECODE_FLUSH_LAYERS: '1'
                DS4_METAL_V41_PREFILL_ATTN_RB4: '1'
            } | upsert $flag (if $arm == 'candidate' { $candidate } else { $control }) | merge (if $stage == 'timeline' { {
                DS4_METAL_CB_TIMES: '1'
                DS4_METAL_ENCODER_TIMELINE: ($dest | path join 'timeline.txt')
            } } else { {} })
            # Presence-based flags are enabled even by "0". Their control
            # must actually omit the variable; record the resulting map below.
            let settings = if $arm == 'baseline' and $control_unset {
                $settings | reject $flag
            } else { $settings }
            let quality_args = if $stage == 'quality' {
                let logits = $dest | path join 'logits'
                mkdir $logits
                [--dump-frontier-logits-dir $logits --dump-decode-logits-dir $logits]
            } else if $stage == 'cpu-profile' {
                # One frontier marker lets an external sampler start after
                # prefill; no per-token dumps or GPU profiling are enabled.
                let frontier_dir = $dest | path join 'frontier'
                mkdir $frontier_dir
                [--dump-frontier-logits-dir $frontier_dir]
            } else { [] }
            let args = [
                --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
                --metal --ssd-streaming --ssd-streaming-cache-experts $cache
                --prompt-file speed-bench/promessi_sposi.txt
                --ctx-start ($frontier | into string) --ctx-max ((if $stage == 'cpu-profile' { $frontier } else { $frontier + 512 }) | into string)
                --step-incr 512 --ctx-alloc 40960 --prefill-chunk 8192
                --gen-tokens ($tokens | into string) --teacher-forced-decode
                --csv ($dest | path join 'timings.csv')
                ...$quality_args
            ]
            $settings | to json | save ($dest | path join 'environment.json')
            $args | to json | save ($dest | path join 'arguments.json')
            print $'start ($repeat)/($repeats) ($arm) (date now)'
            let started = date now
            with-env $settings {
                /usr/bin/time -p ./zig-out/bin/ds4-bench ...$args out> ($dest | path join 'stdout.log') err> ($dest | path join 'stderr.log')
                if $env.LAST_EXIT_CODE != 0 { error make {msg: $'Benchmark failed: ($dest)'} }
            }
            {repeat: $repeat, arm: $arm, wall_s: (((date now) - $started) / 1sec)} | to json | save ($dest | path join 'wall.json')
            print $'done ($arm) (((date now) - $started) / 1sec)s'
        }
    }
}
