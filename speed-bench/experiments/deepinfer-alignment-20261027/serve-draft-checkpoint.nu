#!/usr/bin/env nu
# Foreground only; the caller owns the pressure guard and server lifetime.
const SCRIPT = path self
def main [binary: path, output: path, --features: int, --port: int] {
    if $features not-in [0 1] { error make {msg: '--features must be 0 or 1'} }
    if $port == null { error make {msg: '--port is required'} }
    let binary = ($binary | path expand)
    let output = ($output | path expand)
    if ($output | path exists) { error make {msg: $'Use a new output directory: ($output)'} }
    mkdir ($output | path join cache) ($output | path join responses)
    cd ($SCRIPT | path dirname | path join ../../.. | path expand)
    with-env {
        DS4_STATE_CACHE_DIR: ($output | path join cache), DS4_SERVER_RESPONSE_STORE_DIR: ($output | path join responses),
        DS4_STATE_CACHE_HOT_MB: '0', DS4_STATE_CACHE_SSD_MB: '512', DS4_LOCK_FILE: ($output | path join server.lock),
        DS4_SERVER_CONTEXT_SUFFIX_REUSE: '0', DS4_V41_CHECKPOINT_DRAFT_FEATURES: ($features | into string),
        DS4_V41_CANDIDATE_INDEX: '0', DS4_DSPARK_STATS: '1', DS4_DSPARK_SPEC_LOG: '1'
    } {
        (^$binary --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
            --mtp-model gguf/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf
            --dspark --dspark-confidence 0 --mtp-draft 6 --mtp-exact-sampling
            --host 127.0.0.1 --port $port --ctx 8192 --tokens 128 --prefill-chunk 512
            --ssd-streaming --ssd-streaming-cache-experts 32GB
            --trace ($output | path join trace.log)
            out> ($output | path join server.stdout) err> ($output | path join server.stderr))
        exit $env.LAST_EXIT_CODE
    }
}
