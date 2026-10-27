#!/usr/bin/env nu
# Run under pressure_run.py: five fresh pairs, one model process at a time.
def main [binary: path, output: path] {
    let binary = $binary | path expand
    let output = $output | path expand
    let shaders = $binary | path dirname | path dirname | path join shaders
    if ($output | path exists) { error make {msg: $'Use a fresh output directory: ($output)'} }
    if not ($shaders | path exists) { error make {msg: $'Missing frozen shaders: ($shaders)'} }
    mkdir $output
    mut common = {DS4_V41_INDEX_SIMD: '1', DS4_V41_CANDIDATE_INDEX: '0'}
    for source in (glob ($shaders | path join '*.metal')) {
        let key = $'DS4_METAL_($source | path parse | get stem | str uppercase)_SOURCE'
        $common = ($common | upsert $key $source)
    }
    let model = 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
    {
        binary: $binary
        model: $model
        pairs: 5
        order: [AB BA AB BA AB]
        A: f32
        B: fp4
        tokens: 32768
        context: 40960
        chunk: 8192
        cache_gib: 48
        reasoning: high
        case: favorable
        max_new: 384
        simd: 1
        candidate_index: 0
        primary_ms: 'greedy_sync_ms + greedy_ms'
        warmup_policy: 'No discarded run; record all five fresh processes per layout. Each existing probe retains its first teacher prefill and independent answer prefill.'
    } | to json | save ($output | path join manifest.json)
    for pair in 1..5 {
        let order = if $pair mod 2 == 1 { [f32 fp4] } else { [fp4 f32] }
        for arm in $order {
            let dir = $output | path join $'pair-($pair)' $arm
            mkdir $dir
            let settings = $common | upsert DS4_V41_INDEX_FP4 (if $arm == fp4 { '1' } else { '0' })
            let args = [
                --model
                $model
                --out
                $dir
                --rows
                0
                --tokens
                32768
                --chunk
                8192
                --ctx
                40960
                --case
                favorable
                --thinking
                1
                --max-new
                384
                --cache-gb
                48
                --candidate-sweep
                0
            ]
            with-env $settings {
                ^$binary ...$args out> ($dir | path join stdout.log) err> ($dir | path join stderr.log)
                if $env.LAST_EXIT_CODE != 0 { error make {msg: $'Pair ($pair) ($arm) failed; see ($dir)/stderr.log'} }
            }
            print $'Completed pair ($pair) ($arm)'
        }
    }
}
