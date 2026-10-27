#!/usr/bin/env nu
# Run under the existing pressure guard; paired GPU anatomy, not quality/speed headlines.
def main [binary: path, output: path] {
    let binary = $binary | path expand
    let output = $output | path expand
    let shaders = $binary | path dirname | path dirname | path join shaders
    if ($output | path exists) { error make {msg: $'Use a fresh output directory: ($output)'} }
    if not ($shaders | path exists) { error make {msg: $'Missing frozen shaders: ($shaders)'} }
    mkdir $output
    let prompt = $output | path join prompt.txt
    1..2048 | each { "A small local model reads this repeated passage about trees, rivers, and clear summer skies.\n" } | str join | save $prompt
    for arm in [
        {name: f32, fp4: '0'}
        {name: fp4, fp4: '1'}
    ] {
        let dir = $output | path join $arm.name
        mkdir $dir
        mut settings = {
            DS4_V41_INDEX_SIMD: '1'
            DS4_V41_INDEX_FP4: $arm.fp4
            DS4_V41_CANDIDATE_INDEX: '0'
            DS4_V41_COMPRESSED_KV_F16: '0'
            DS4_METAL_V41_RESIDENT_ROUTE_GAP: '0'
            DS4_METAL_V41_RESIDENT_PREFILL_GAP: '0'
            DS4_METAL_ENCODER_TIMELINE: ($dir | path join timeline.txt)
            DS4_METAL_PIPELINE_CREATE_PROFILE: '1'
        }
        for source in (glob ($shaders | path join '*.metal')) {
            let key = $'DS4_METAL_($source | path parse | get stem | str uppercase)_SOURCE'
            $settings = ($settings | upsert $key $source)
        }
        let args = [
            --model
            gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
            --prompt-file
            $prompt
            --ctx-start
            8192
            --ctx-max
            8192
            --ctx-alloc
            16384
            --gen-tokens
            0
            --prefill-chunk
            8192
            --ssd-streaming
            --ssd-streaming-cache-experts
            '48GB'
            --csv
            ($dir | path join timings.csv)
        ]
        {binary: $binary, args: $args, settings: $settings} | to json | save ($dir | path join command.json)
        with-env $settings {
            ^$binary ...$args out> ($dir | path join stdout.log) err> ($dir | path join stderr.log)
            if $env.LAST_EXIT_CODE != 0 { error make {msg: $'($arm.name) failed; see ($dir)/stderr.log'} }
        }
        print $'Completed ($arm.name): one 8192-token prefill, no generation'
    }
}
