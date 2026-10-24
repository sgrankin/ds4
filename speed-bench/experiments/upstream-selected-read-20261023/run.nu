#!/usr/bin/env nu
const ROOT = path self ../../..

def main [label: string, variant: string, workload: string, --profile] {
    cd $ROOT
    let work = $ROOT | path join .workspaces/upstream-selected-read-20261023
    let out = $work | path join $label
    mkdir ($out | path join logits)
    let binary = match $variant {
        A => ($ROOT | path join .workspaces/upstream-sync-20261023/baseline/ds4-bench)
        B => ($work | path join candidate-build/bin/ds4-bench)
        _ => { error make {msg: 'variant must be A or B'} }
    }
    let args = match $workload {
        ordinary => [--ctx-start 512 --ctx-max 768 --step-incr 128 --ctx-alloc 1024 --prefill-chunk 512 --ssd-streaming-cache-experts '64GB' --gen-tokens 0]
        miss => [--ctx-start 1 --ctx-max 257 --step-incr 128 --ctx-alloc 512 --prefill-chunk 128 --ssd-streaming-cache-experts '32GB' --gen-tokens 0]
        control => [--ctx-start 2048 --ctx-max 2048 --ctx-alloc 4096 --prefill-chunk 2048 --ssd-streaming-cache-experts '64GB' --gen-tokens 8 --teacher-forced-decode]
        _ => { error make {msg: 'workload must be ordinary, miss or control'} }
    }
    mut settings = {DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: '1'}
    if $variant == B { $settings.DS4_METAL_STREAMING_PREFILL_SORT_SELECTED_READS = '1' }
    if $workload == miss {
        $settings.DS4_METAL_V41_SELECTED_TILE = '128'
        $settings.DS4_METAL_V41_SELECTED_LIMIT = '1024'
    }
    if $profile {
        $settings.DS4_METAL_GRAPH_PREFILL_PROFILE = '1'
        $settings.DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE = '1'
        $settings.DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_PROFILE = '1'
    }
    for name in ($env | columns | where { |name| ($name | str starts-with DS4) or ($name | str starts-with PT_) }) {
        hide-env $name
    }
    with-env $settings {
        (^python3 -B speed-bench/ideas/pressure_run.py ($out | path join guard) --
            $binary
            --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
            --metal --ssd-streaming
            --prompt-file ($work | path join prompt.txt)
            ...$args
            --csv ($out | path join timings.csv)
            --dump-frontier-logits-dir ($out | path join logits)
            out> ($out | path join stdout.log) err> ($out | path join stderr.log)
        )
        if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
    }
}
