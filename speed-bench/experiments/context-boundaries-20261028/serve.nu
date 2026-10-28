#!/usr/bin/env nu
# Foreground quality/restart fixture. The caller owns the guard and process.
def main [binary: path, output: path, --port: int = 18355, --ongoing: int = 1, --label: string = 'initial'] {
    let binary = $binary | path expand
    let output = $output | path expand
    let shaders = $binary | path dirname | path dirname | path join shaders
    mkdir ($output | path join cache) ($output | path join responses)
    let inherited = $env | columns | where {|name| $name | str starts-with 'DS4_' }
    hide-env --ignore-errors ...$inherited
    mut settings = {
        DS4_STATE_CACHE_DIR: ($output | path join cache)
        DS4_STATE_CACHE_HOT_MB: '512'
        DS4_STATE_CACHE_SSD_MB: '4096'
        DS4_SERVER_RESPONSE_STORE_DIR: ($output | path join responses)
        DS4_LOCK_FILE: ($output | path join server.lock)
        DS4_SERVER_CONTEXT_SUFFIX_REUSE: '1'
        DS4_SERVER_CONTEXT_SUFFIX_CANONICAL: '1'
        DS4_SERVER_CONTEXT_SUFFIX_PAD: '1'
        DS4_SERVER_CONTEXT_SUFFIX_RIGHT: '1'
        DS4_SERVER_CONTEXT_SUFFIX_FRESH_TAIL: '64'
        DS4_SERVER_CONTEXT_SUFFIX_STAGING_MB: '512'
        DS4_SERVER_CONTEXT_BOUNDARIES: '1'
        DS4_SERVER_CONTEXT_SUFFIX_ONGOING_USER: ($ongoing | into string)
        DS4_SERVER_CONTEXT_BOUNDARY_MIN_TOKENS: '512'
        DS4_SERVER_CONTEXT_BOUNDARY_LIMIT: '4'
        DS4_SERVER_CONTEXT_CAPTURE_PROFILE: '1'
        DS4_V41_INDEX_FP4: '1'
        DS4_V41_INDEX_SIMD: '1'
        DS4_V41_COMPRESSED_KV_FP4: '1'
        DS4_V41_COMPRESSED_KV_F16: '0'
        DS4_METAL_V41_FUSED_HC_NORM: '0'
        DS4_METAL_V41_RESIDENT_ROUTE_GAP: '0'
        DS4_METAL_V41_RESIDENT_PREFILL_GAP: '0'
    }
    for source in (glob ($shaders | path join '*.metal')) {
        let key = $'DS4_METAL_($source | path parse | get stem | str uppercase)_SOURCE'
        $settings = ($settings | upsert $key $source)
    }
    let args = [
        '--model' 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
        '--host' '127.0.0.1' '--port' ($port | into string)
        '--ctx' '32768' '--tokens' '512' '--prefill-chunk' '8192'
        '--batched-session' '2' '--ssd-streaming' '--ssd-streaming-cache-experts' '32GB'
        '--trace' ($output | path join $'($label).trace.log')
    ]
    {binary: $binary, args: $args, settings: $settings, pid: $nu.pid,
        comparative_performance: false, coordinator_quantum: 2048}
    | to json | save ($output | path join $'($label).command.json')
    load-env $settings
    exec $binary ...$args
}
