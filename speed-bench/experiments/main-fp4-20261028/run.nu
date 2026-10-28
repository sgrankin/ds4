#!/usr/bin/env nu
# Run this whole sequence under the existing pressure guard, one model at a time.
def main [binary: path, output: path, --tokens: int = 8192, --case: string = 'all'] {
    let binary = $binary | path expand
    let output = $output | path expand
    let shaders = $binary | path dirname | path dirname | path join shaders
    let context = [16384 ($tokens + 8192)] | math max
    if ($output | path exists) { error make {msg: $'Use a fresh output directory: ($output)'} }
    if not ($shaders | path exists) { error make {msg: $'Missing frozen shaders: ($shaders)'} }
    mkdir $output
    mut common = {
        DS4_V41_INDEX_FP4: '1'
        DS4_V41_INDEX_SIMD: '1'
        DS4_V41_CANDIDATE_INDEX: '0'
        DS4_V41_COMPRESSED_KV_F16: '0'
        DS4_METAL_V41_FUSED_HC_NORM: '0'
        DS4_METAL_ROUTED_PACKED_ACT_I8: '0'
        DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: '1'
    }
    for source in (glob ($shaders | path join '*.metal')) {
        let key = $'DS4_METAL_($source | path parse | get stem | str uppercase)_SOURCE'
        $common = ($common | upsert $key $source)
    }
    let model = 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
    {
        binary: $binary
        source_revision: (open --raw ($binary | path dirname | path dirname | path join source-revision.txt) | str trim)
        shaders: $shaders
        model: $model
        pairs: 5
        A: control
        B: packed
        order: [AB BA AB BA AB]
        case: $case
        tokens: $tokens
        context: $context
        chunk: 8192
        cache_gib: 48
        reasoning: high
        native_expert_routing: true
        max_new: 384
        packed_index: 1
        simd_index: 1
        candidate_index: 0
        main_f16: 0
        hc_fusion: 0
        routed_compute_tile: 'normal serving defaults, identical in both arms'
        expert_io_summary: true
        changed_setting: DS4_V41_COMPRESSED_KV_FP4
        main_storage: 'F32 expanded mode3 values versus 288-byte native E2M1/E4M3 rows'
        teacher_rows_per_case: 'Expected-answer tokens plus EOS; actual counts in rows.csv'
        primary_ms: 'greedy_sync_ms + greedy_ms'
        warmup_policy: 'All five fresh processes per arm retained; teacher prefill and independent normal-reasoning answer prefill both reported. Engine initialization excluded.'
    } | to json | save ($output | path join manifest.json)
    for pair in 1..5 {
        let order = if $pair mod 2 == 1 { [control packed] } else { [packed control] }
        for arm in $order {
            let dir = $output | path join $'pair-($pair)' $arm
            mkdir $dir
            let settings = $common | upsert DS4_V41_COMPRESSED_KV_FP4 (if $arm == packed { '1' } else { '0' })
            let args = [
                --model
                $model
                --out
                $dir
                --rows
                0
                --tokens
                $tokens
                --chunk
                8192
                --ctx
                $context
                --compressed-kv-f16
                0
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
            let format = if $arm == packed { 'fp4' } else { 'f32' }
            let allocation = open --raw ($dir | path join stderr.log) | str contains $'KV allocated main=($format) '
            if not $allocation { error make {msg: $'Pair ($pair) ($arm) used the wrong main storage; see ($dir)/stderr.log'} }
            print $'Completed pair ($pair) ($arm): ($case), ($tokens) tokens'
        }
    }
}
