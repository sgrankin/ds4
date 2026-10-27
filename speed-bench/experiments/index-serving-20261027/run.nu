#!/usr/bin/env nu
# Run under the existing pressure guard. Root owns binary freezing and GPU jobs.
def main [
    baseline: path
    candidate: path
    output: path
    --tokens: int = 8192
    --case: string
    --max-new: int = 384
    --candidate-index: int = 0
    --packed: int = 0
    --hot-loop
    --simd-loop
] {
    if $candidate_index not-in [0 1] or $packed not-in [0 1] { error make {msg: 'Flags must be 0 or 1'} }
    if $hot_loop and $simd_loop { error make {msg: 'Choose one sweep axis'} }
    let output = $output | path expand
    if ($output | path exists) { error make {msg: $'Use a fresh output directory: ($output)'} }
    mkdir $output
    let selected = if $case != null { $case } else if $hot_loop or $simd_loop { 'favorable' } else { 'all' }
    let model = 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
    # Preserve directory names consumed by the existing fixed-teacher analysis.
    let arms = if $simd_loop {
        [
            {
                name: simd-sweep
                binary: ($candidate | path expand)
                simd: '0'
                fp4: ($packed | into string)
            }
        ]
    } else {
        [
            {
                name: dense
                binary: ($baseline | path expand)
                simd: '0'
                fp4: '0'
            }
            {
                name: candidates
                binary: ($candidate | path expand)
                simd: '1'
                fp4: ($packed | into string)
            }
            {
                name: dense-repeat
                binary: ($baseline | path expand)
                simd: '0'
                fp4: '0'
            }
        ]
    }
    {
        arms: $arms
        model: $model
        tokens: $tokens
        case: $selected
        reasoning: high
        candidate_index: $candidate_index
        hot_loop: $hot_loop
        simd_loop: $simd_loop
        hot_loop_policy: 'Candidate sweep alternates candidate0/1; SIMD sweep alternates SIMD0/1 on one exact trace. Repeat0 is retained warmup.'
    } | to json | save ($output | path join manifest.json)
    for arm in $arms {
        let dir = $output | path join $arm.name
        mkdir $dir
        let ordinary_args = [
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
            ($tokens + 8192)
            --case
            $selected
            --thinking
            1
            --max-new
            $max_new
            --cache-gb
            48
            --candidate-sweep
            (if $hot_loop { 1 } else { 0 })
        ]
        let args = if $simd_loop {
            $ordinary_args | append [--simd-sweep 1]
        } else {
            $ordinary_args
        }
        # These executables load Metal source at runtime. Freeze it beside
        # each build as well, so later edits cannot change the measured arm.
        let shaders = $arm.binary | path dirname | path dirname | path join shaders
        if not ($shaders | path exists) { error make {msg: $'Missing frozen shaders: ($shaders)'} }
        mut settings = {
            DS4_V41_INDEX_SIMD: $arm.simd
            DS4_V41_INDEX_FP4: $arm.fp4
            DS4_V41_CANDIDATE_INDEX: ($candidate_index | into string)
        }
        for source in (glob ($shaders | path join '*.metal')) {
            let key = $'DS4_METAL_($source | path parse | get stem | str uppercase)_SOURCE'
            $settings = ($settings | upsert $key $source)
        }
        with-env $settings {
            ^$arm.binary ...$args out> ($dir | path join stdout.log) err> ($dir | path join stderr.log)
            if $env.LAST_EXIT_CODE != 0 { error make {msg: $'($arm.name) failed; see ($dir)/stderr.log'} }
        }
        print $'Completed ($arm.name): ($tokens) tokens, SIMD=($arm.simd), FP4=($arm.fp4)'
    }
}
