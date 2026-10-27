#!/usr/bin/env nu
# One existing pressure guard owns these fresh model processes.
def main [binary: path, output: path, --case: string = 'all'] {
    let binary = $binary | path expand
    let output = $output | path expand
    let shaders = $binary | path dirname | path dirname | path join shaders
    if ($output | path exists) { error make {msg: $'Use a fresh output directory: ($output)'} }
    if not ($shaders | path exists) { error make {msg: $'Missing frozen shaders: ($shaders)'} }
    mkdir $output
    mut common = {
        DS4_V41_INDEX_FP4: '1'
        DS4_V41_INDEX_SIMD: '1'
        DS4_V41_CANDIDATE_INDEX: '0'
        DS4_METAL_DISABLE_V41_PACKED_M32N128: '1'
        DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: '1'
    }
    for source in (glob ($shaders | path join '*.metal')) {
        let key = $'DS4_METAL_($source | path parse | get stem | str uppercase)_SOURCE'
        $common = ($common | upsert $key $source)
    }
    let model = 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
    {
        binary: $binary
        model: $model
        pairs: 5
        A: control
        B: int8
        order: [AB BA AB BA AB]
        case: $case
        tokens: 8192
        context: 16384
        chunk: 8192
        cache_gib: 48
        reasoning: high
        native_expert_routing: true
        max_new: 384
        packed_index: 1
        simd_index: 1
        candidate_index: 0
        routed_compute_tile: M64N32
        disable_packed_m32n128: 1
        expert_io_summary: true
        changed_setting: DS4_METAL_ROUTED_MPP_I8
        primary_ms: 'greedy_sync_ms + greedy_ms'
        warmup_policy: 'All five fresh processes per arm retained; first teacher prefill and independent answer prefill both reported. Engine initialization excluded.'
    } | to json | save ($output | path join manifest.json)
    for pair in 1..5 {
        let order = if $pair mod 2 == 1 { [control int8] } else { [int8 control] }
        for arm in $order {
            let dir = $output | path join $'pair-($pair)' $arm
            mkdir $dir
            let settings = $common | upsert DS4_METAL_ROUTED_MPP_I8 (if $arm == int8 { '1' } else { '0' })
            let args = [
                --model
                $model
                --out
                $dir
                --rows
                0
                --tokens
                8192
                --chunk
                8192
                --ctx
                16384
                --case
                $case
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
            let invoked = open --raw ($dir | path join stderr.log) | str contains 'ds4: experimental INT8 IQ2 prefill'
            if $invoked != ($arm == int8) {
                error make {msg: $'Pair ($pair) ($arm) used the wrong compute path; see ($dir)/stderr.log'}
            }
            print $'Completed pair ($pair) ($arm): ($case)'
        }
    }
}
