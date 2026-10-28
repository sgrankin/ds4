#!/usr/bin/env nu
# Foreground restart smoke; the caller owns the guard and process lifetime.
const LOADER = path self with-shaders.nu

def main [binary: path, output: path, --port: int = 18353, --main-fp4: int = 1, --restart] {
    let binary = $binary | path expand
    let output = $output | path expand
    if not $restart and ($output | path exists) { error make {msg: 'Use a fresh output directory'} }
    if $restart and not ($output | path join cache | path exists) {
        error make {msg: '--restart requires an existing cache'}
    }
    mkdir ($output | path join cache) ($output | path join responses)
    let name = if $restart { 'restart' } else { 'initial' }
    let settings = {
        DS4_STATE_CACHE_DIR: ($output | path join cache)
        DS4_STATE_CACHE_HOT_MB: '0'
        DS4_STATE_CACHE_SSD_MB: '512'
        DS4_SERVER_RESPONSE_STORE_DIR: ($output | path join responses)
        DS4_LOCK_FILE: ($output | path join server.lock)
        DS4_SERVER_CONTEXT_SUFFIX_REUSE: '0'
        DS4_SERVER_CONTEXT_BOUNDARIES: '0'
        DS4_V41_CHECKPOINT_DRAFT_FEATURES: '1'
        DS4_V41_INDEX_FP4: '1'
        DS4_V41_INDEX_SIMD: '1'
        DS4_V41_COMPRESSED_KV_F16: '0'
        DS4_V41_COMPRESSED_KV_FP4: ($main_fp4 | into string)
        DS4_METAL_V41_RESIDENT_ROUTE_GAP: '0'
        DS4_METAL_V41_RESIDENT_PREFILL_GAP: '0'
        DS4_DSPARK_STATS: '1'
        DS4_DSPARK_SPEC_LOG: '1'
    }
    let args = [
        '--model' 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
        '--mtp-model' 'gguf/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf'
        '--dspark' '--mtp-draft' '6' '--mtp-exact-sampling'
        '--host' '127.0.0.1' '--port' ($port | into string)
        '--ctx' '8192' '--tokens' '2048' '--prefill-chunk' '512'
        '--ssd-streaming' '--ssd-streaming-cache-experts' '32GB'
        '--trace' ($output | path join $'($name).trace.log')
    ]
    {binary: $binary, args: $args, settings: $settings,
     confidence: 'ordinary Metal default 0.6', comparative_performance: false}
    | to json | save ($output | path join $'($name).command.json')
    with-env $settings {
        nu $LOADER $binary ...$args out> ($output | path join $'($name).stdout') err> ($output | path join $'($name).stderr')
        exit $env.LAST_EXIT_CODE
    }
}
