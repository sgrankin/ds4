#!/usr/bin/env nu
# Run under pressure_run.py only if the unprofiled paired measurements warrant it.
# Same replay workload as run.nu; diagnostics remain inside each child process.
def main [binary: path, output: path] {
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
    }
    for source in (glob ($shaders | path join '*.metal')) {
        let key = $'DS4_METAL_($source | path parse | get stem | str uppercase)_SOURCE'
        $common = ($common | upsert $key $source)
    }
    let model = 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
    {
        binary: $binary
        model: $model
        tokens: 8192
        context: 16384
        chunk: 8192
        cache_gib: 48
        reasoning: high
        case: favorable
        max_new: 384
        simd: 1
        packed_index: 1
        candidate_index: 0
        routed_compute_tile: M64N32
        changed_setting: DS4_METAL_ROUTED_MPP_I8
        primary_ms: 'greedy_sync_ms + greedy_ms'
        note: 'Two fresh processes; both teacher and independent answer prefills retained. GPU timeline pass boundaries perturb scheduling.'
    } | to json | save ($output | path join manifest.json)
    for arm in [control int8] {
        let dir = $output | path join $arm
        mkdir $dir
        # Native prefill.zig/command_schedule.zig and expert_cache.zig consume
        # these; pipeline/timestamp capture stays in the actual Metal driver.
        let settings = $common | merge {
            DS4_METAL_ROUTED_MPP_I8: (if $arm == int8 { '1' } else { '0' }),
            DS4_METAL_ENCODER_TIMELINE: ($dir | path join timeline.txt),
            DS4_METAL_PIPELINE_CREATE_PROFILE: '1',
            DS4_METAL_GRAPH_PREFILL_PROFILE: '1',
            DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: '1'
        }
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
            favorable
            --thinking
            1
            --max-new
            384
            --cache-gb
            48
            --candidate-sweep
            0
            --simd-sweep
            0
        ]
        {binary: $binary, args: $args, settings: $settings} | to json | save ($dir | path join command.json)
        with-env $settings {
            ^$binary ...$args out> ($dir | path join stdout.log) err> ($dir | path join stderr.log)
            if $env.LAST_EXIT_CODE != 0 { error make {msg: $'($arm) failed; see ($dir)/stderr.log'} }
        }
        let invoked = open --raw ($dir | path join stderr.log) | str contains 'ds4: experimental INT8 IQ2 prefill'
        if $invoked != ($arm == int8) { error make {msg: $'Wrong compute path for ($arm)'} }
        print $'Completed profiled 8K ($arm) replay workload with reasoning'
    }
}
