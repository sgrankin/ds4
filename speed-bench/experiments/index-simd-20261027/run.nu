#!/usr/bin/env nu
# Run this driver under speed-bench/ideas/pressure_run.py, one GPU job at a time.
def main [binary: path, output: path, --rows: int = 0, --gpu-times] {
    mkdir $output
    let args = if $rows == 0 { [--index-simd] } else { [--index-simd ($rows | into string)] }
    let executable = $binary | path expand
    if $gpu_times {
        with-env {DS4_METAL_CB_TIMES: '1'} {
            ^$executable ...$args out> ($output | path join timings.csv) err> ($output | path join checks.log)
            if $env.LAST_EXIT_CODE != 0 { error make {msg: $'GPU fixture failed; see ($output)/checks.log'} }
        }
    } else {
        ^$executable ...$args out> ($output | path join timings.csv) err> ($output | path join checks.log)
        if $env.LAST_EXIT_CODE != 0 { error make {msg: $'GPU fixture failed; see ($output)/checks.log'} }
    }
    print $'Completed index SIMD fixture: ($output)'
}
