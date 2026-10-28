#!/usr/bin/env nu
# Correctness only; run the entire sequence under the existing pressure guard.
const LOADER = path self with-shaders.nu

def fixture [binary: path, args: list<string>, log: path] {
    nu $LOADER $binary ...$args out+err> $log
    if $env.LAST_EXIT_CODE != 0 {
        error make {msg: $'Qualification failed; see ($log)'}
    }
}

def main [prefix: path, output: path, --skip: string = ''] {
    let prefix = $prefix | path expand
    let output = $output | path expand
    if ($output | path exists) { error make {msg: 'Use a fresh output directory'} }
    mkdir $output
    let model = 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
    let support = 'gguf/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf'
    let skipped = $skip | split row ','
    load-env {DS4_V41_INDEX_FP4: '1', DS4_V41_INDEX_SIMD: '1'}
    {revision: (open --raw ($prefix | path join source-revision.txt) | str trim),
     prefix: $prefix, model: $model, support: $support, skipped: $skipped, comparative_performance: false}
    | to json | save ($output | path join manifest.json)
    for item in [
        {name: main, option: '--main-packed-cross'}
        {name: both, option: '--all-packed-cross'}
    ] {
        if $item.name in $skipped { continue }
        fixture ($prefix | path join bin/test-dspark-checkpoint) [
            $model $support ($output | path join $'cache-($item.name)') $item.option
        ] ($output | path join $'checkpoint-($item.name).log')
        print $'PASS checkpoint ($item.name)'
    }
    for item in [
        {name: f32-fp4, source_f16: '0', source_fp4: '0', dest_f16: '0', dest_fp4: '1', chain: '1'}
        {name: fp4-f32, source_f16: '0', source_fp4: '1', dest_f16: '0', dest_fp4: '0', chain: '1'}
        {name: f16-fp4, source_f16: '1', source_fp4: '0', dest_f16: '0', dest_fp4: '1', chain: '1'}
        {name: fp4-f16, source_f16: '0', source_fp4: '1', dest_f16: '1', dest_fp4: '0', chain: '1'}
        {name: fp4-fixed, source_f16: '0', source_fp4: '1', dest_f16: '0', dest_fp4: '1', chain: '0'}
    ] {
        if $item.name in $skipped { continue }
        let dir = $output | path join $item.name
        mkdir $dir
        fixture ($prefix | path join bin/suffix-repeat) [
            '--model' $model '--out' $dir '--moves' '4' '--prerope' '1'
            '--ctx' '8192' '--chunk' '2048' '--cache-gb' '32'
            '--source-f16' $item.source_f16 '--source-fp4' $item.source_fp4
            '--destination-f16' $item.dest_f16 '--destination-fp4' $item.dest_fp4
            '--source-index-fp4' '1'
            '--destination-index-fp4' (if $item.name == 'fp4-f32' { '0' } else { '1' })
            '--chain' $item.chain
        ] ($dir | path join fixture.log)
        print $'PASS suffix ($item.name)'
    }
}
