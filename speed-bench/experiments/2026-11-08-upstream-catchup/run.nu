#!/usr/bin/env nu
# Run under pressure_run.py after correctness checks, with no competing jobs.
# Each arm runs from its checkout so its own Metal source is compiled.
def main [
    out: path,
    baseline: path,
    candidate: path,
    --frontier: int = 32768,
    --stage: string = 'timing',
    --model: string = 'deepseek',
    --exact-routing
] {
    let out = $out | path expand
    let baseline = $baseline | path expand
    let candidate = $candidate | path expand
    let models = {
        deepseek: 'DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
        glm: 'GLM-5.3-Flash-Q2.gguf'
        qwen: 'Qwen3.8-Flash-Next-Q4.gguf'
    }
    if $stage not-in ['timing' 'quality'] { error make {msg: 'stage must be timing or quality'} }
    let checkpoint = $candidate | path join gguf ($models | get $model)
    let prompt = $candidate | path join speed-bench promessi_sposi.txt
    for key in ($env | columns | where {|key| ($key | str starts-with DS4) or ($key | str starts-with PT_)}) { hide-env $key }
    # The installed persistent-fast kernel/routing package, without unrelated
    # agent/server cache policy. Experimental GPU routing and pairing stay off.
    let common = if $model == 'deepseek' { {
        DS4_V41_COMPRESSED_KV_FP4: '1', DS4_V41_INDEX_FP4: '1', DS4_V41_INDEX_SIMD: '1'
        DS4_METAL_V41_SELECTED_TILE: '64', DS4_METAL_V41_SELECTED_LIMIT: '1024'
        DS4_METAL_V41_RESIDENT_ROUTE_GAP: '.90', DS4_METAL_V41_RESIDENT_PREFILL_GAP: '.90'
        DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS: '1'
        DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS: '5', DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK: '64'
        DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS: '1', DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS: '1'
        DS4_METAL_V41_RESIDENT_ROUTE_POLICY: 'replace'
        DS4_METAL_V41_ROUTE_SHARED_OVERLAP: '1', DS4_METAL_V41_ROUTE_SHARED_DOWN: '1'
        DS4_METAL_SELECTED_READBACK_SPIN_US: '2000', DS4_METAL_V41_STREAMING_ROPE_TABLE: '1'
        DS4_METAL_V41_SHORT_Q8_QUERY: '1', DS4_METAL_V41_SHORT_Q8_QUERY_TABLE: '1'
        DS4_METAL_V41_PACKED_F16_GATHER: '1', DS4_METAL_V41_REUSE_SELECTED_KV: '1'
        DS4_METAL_V41_DECODE_FLUSH_LAYERS: '1', DS4_METAL_V41_PREFILL_ATTN_RB4: '1'
        DS4_METAL_V41_ATTN_UNIT_SCALE: '1', DS4_METAL_V41_PREFILL_ROPE_BF16: '1'
        DS4_METAL_V41_PREFILL_INDEX_Q_MPP: '1'
    } } else if $model == 'glm' {
        {DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP: '1'}
    } else { {} }
    let storage = if $model == 'qwen' { [] } else {
        [--ssd-streaming --ssd-streaming-cache-experts '64GB']
    }
    let allocation = if $model == 'qwen' { '4096' } else { '40960' }
    let chunk = if $model == 'qwen' { '512' } else { '8192' }
    let chunk_args = if $model == 'glm' { [] } else { [--prefill-chunk $chunk] }
    mkdir $out
    let repeats = if $stage == 'timing' { 5 } else { 1 }
    for repeat in 1..$repeats {
        let arms = if ($repeat mod 2) == 1 { ['baseline' 'candidate'] } else { ['candidate' 'baseline'] }
        for arm in $arms {
            let checkout = if $arm == 'baseline' { $baseline } else { $candidate }
            let dest = $out | path join $'($repeat)-($arm)'
            mkdir $dest
            let routing = if $exact_routing and $model == 'deepseek' {
                {DS4_METAL_V41_RESIDENT_ROUTE_GAP: '0', DS4_METAL_V41_RESIDENT_PREFILL_GAP: '0',
                 DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS: '0'}
            } else { {} }
            let settings = $common | merge $routing | merge (if $stage == 'quality' { {DS4_BENCH_DECODE_LOGITS_HASH: '1'} } else { {} })
            let diagnostics = if $stage == 'quality' {
                let logits = $dest | path join logits
                mkdir $logits
                [--dump-frontier-logits-dir $logits --dump-decode-logits-dir $logits]
            } else { [] }
            let args = [
                --model $checkpoint --metal ...$storage
                --prompt-file $prompt --ctx-start ($frontier | into string)
                --ctx-max (($frontier + 512) | into string) --step-incr 512 --ctx-alloc $allocation
                ...$chunk_args --gen-tokens (if $stage == 'timing' and $model != 'glm' { '128' } else { '32' })
                --teacher-forced-decode --csv ($dest | path join timings.csv) ...$diagnostics
            ]
            $settings | to json | save -f ($dest | path join environment.json)
            $args | to json | save -f ($dest | path join arguments.json)
            print $'Starting ($repeat)/($repeats) ($arm) at (date now)'
            let started = date now
            cd $checkout
            with-env $settings {
                ^/usr/bin/time -p ./zig-out/bin/ds4-bench ...$args out> ($dest | path join stdout.log) err> ($dest | path join stderr.log)
                if $env.LAST_EXIT_CODE != 0 { error make {msg: $'Benchmark failed: ($dest)'} }
            }
            {repeat: $repeat, arm: $arm, wall_s: (((date now) - $started) / 1sec)} | to json | save -f ($dest | path join wall.json)
            print $'Finished ($arm) in (((date now) - $started) / 1sec) s'
        }
    }
}
