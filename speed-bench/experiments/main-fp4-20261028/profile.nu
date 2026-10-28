#!/usr/bin/env nu
# Matching 32K diagnostic only; run under the existing pressure guard.
def main [binary: path, output: path] {
    let binary = $binary | path expand
    let output = $output | path expand
    let prefix = $binary | path dirname | path dirname
    let shaders = $prefix | path join shaders
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
        DS4_METAL_PIPELINE_CREATE_PROFILE: '1'
        DS4_METAL_GRAPH_PREFILL_PROFILE: '1'
        DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: '1'
    }
    for source in (glob ($shaders | path join '*.metal')) {
        let key = $'DS4_METAL_($source | path parse | get stem | str uppercase)_SOURCE'
        $common = ($common | upsert $key $source)
    }
    {
        binary: $binary
        source_revision: (open --raw ($prefix | path join source-revision.txt) | str trim)
        shaders: $shaders
        model: 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
        A: control
        B: packed
        changed_setting: DS4_V41_COMPRESSED_KV_FP4
        tokens: 32768
        context: 40960
        chunk: 8192
        cache_gib: 48
        case: favorable
        reasoning: high
        max_new: 384
        packed_index: 1
        simd_index: 1
        candidate_index: 0
        main_f16: 0
        hc_fusion: 0
        routed_compute_tile: 'normal serving defaults, identical in both arms'
        comparative_performance: false
        note: 'Two fresh processes; both prefills and teacher/normal decode retained. Encoder timeline boundaries perturb scheduling; use for attribution, not latency claims.'
    } | to json | save ($output | path join manifest.json)
    for arm in [control packed] {
        let dir = $output | path join $arm
        mkdir $dir
        let settings = $common | merge {
            DS4_V41_COMPRESSED_KV_FP4: (if $arm == 'packed' { '1' } else { '0' })
            DS4_METAL_ENCODER_TIMELINE: ($dir | path join timeline.txt)
        }
        let args = [
            '--model' 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
            '--out' $dir '--rows' '0' '--tokens' '32768' '--chunk' '8192'
            '--ctx' '40960' '--compressed-kv-f16' '0' '--case' 'favorable'
            '--thinking' '1' '--max-new' '384' '--cache-gb' '48'
            '--candidate-sweep' '0'
        ]
        {binary: $binary, args: $args, settings: $settings} | to json | save ($dir | path join command.json)
        with-env $settings {
            ^$binary ...$args out> ($dir | path join stdout.log) err> ($dir | path join stderr.log)
            if $env.LAST_EXIT_CODE != 0 { error make {msg: $'Profile ($arm) failed; see ($dir)/stderr.log'} }
        }
        let format = if $arm == 'packed' { 'fp4' } else { 'f32' }
        let stderr = open --raw ($dir | path join stderr.log)
        if not ($stderr | str contains $'KV allocated main=($format) ') {
            error make {msg: $'Profile ($arm) used the wrong main storage'}
        }
        if not ($dir | path join timeline.txt | path exists) {
            error make {msg: $'Missing encoder timeline for ($arm)'}
        }
        print $'Completed profiled 32K favorable ($arm) with reasoning'
    }
}
