#!/usr/bin/env nu
# Run from the repository root under the existing pressure guard. Correctness
# (--resident-router) is a separate process, preserving the CB diagnostic cap.
# GPU span is throughput for 64 independent dispatches, not layer latency.
def variation [values: list] {
    {
        mean: ($values | math avg)
        median: ($values | math median)
        min: ($values | math min)
        max: ($values | math max)
        stddev: ($values | math stddev --sample)
    }
}

def main [out: path, --binary: path = 'zig-out/bin/test-deepseek41-metal', --analyze-only] {
    mkdir $out
    let host_path = $out | path join 'host.csv'
    let stderr_path = $out | path join 'stderr.log'
    if not $analyze_only {
        for key in ($env | columns | where {|key| $key | str starts-with 'DS4'}) { hide-env $key }
        with-env {DS4_METAL_CB_TIMES: '1'} {
            ^$binary --resident-router-bench out> $host_path err> $stderr_path
            if $env.LAST_EXIT_CODE != 0 { error make {msg: $'Microbenchmark failed: ($stderr_path)'} }
        }
    }
    let host = open $host_path
    let gpu = open --raw $stderr_path | lines
        | parse -r 'ds4: cb command batch: driver (?<driver_us>[0-9-]+) us, queue-wait (?<queue_wait_us>[0-9-]+) us, gpu (?<gpu_batch_us>[0-9]+) us,'
        | update driver_us { into int }
        | update queue_wait_us { into int }
        | update gpu_batch_us { into int }
    if ($host | length) != 72 or ($gpu | length) != ($host | length) {
        error make {msg: $'Expected 72 matching host/GPU samples; found ($host | length)/($gpu | length). Inspect raw logs.'}
    }
    let samples = $host | enumerate | each {|row|
        let record = $row.item | merge ($gpu | get $row.index)
        $record | merge {
            gpu_us_per_dispatch: ($record.gpu_batch_us / $record.calls)
            cpu_encode_us_per_dispatch: ($record.cpu_encode_us / $record.calls)
            wall_us_per_dispatch: ($record.batch_wall_us / $record.calls)
        }
    }
    $samples | to csv | save --force ($out | path join 'samples.csv')
    let measured = $samples | where sample > 0
    let summary = $measured | group-by experts missing arm --to-table | each {|group|
        if ($group.items | length) != 5 { error make {msg: 'Expected five measured samples per arm and case'} }
        {
            experts: $group.experts, missing: $group.missing, arm: $group.arm, samples: 5
            gpu_batch_us: (variation ($group.items | get gpu_batch_us))
            gpu_us_per_dispatch: (variation ($group.items | get gpu_us_per_dispatch))
            cpu_encode_us_per_dispatch: (variation ($group.items | get cpu_encode_us_per_dispatch))
            wall_us_per_dispatch: (variation ($group.items | get wall_us_per_dispatch))
        }
    }
    {
        metric: 'GPU command-buffer span for 64 independent dispatches; per-dispatch value is batch throughput, not model layer latency'
        host_metric: 'CPU encode excludes native-ID reset; wall includes submit, GPU wait and CB diagnostic logging'
        mask: 'About half the experts are resident; all-hit, three-miss and six-miss native selections'
        order: 'One warmup pair, then five pairs alternating SIMD/scalar and scalar/SIMD'
        gpu_timestamp_rounding_us: 1
        summary: $summary
    } | to json | save --force ($out | path join 'summary.json')
    $summary | each {|row| {
        experts: $row.experts, missing: $row.missing, arm: $row.arm
        gpu_mean_us: $row.gpu_us_per_dispatch.mean
        gpu_min_us: $row.gpu_us_per_dispatch.min
        gpu_max_us: $row.gpu_us_per_dispatch.max
        gpu_stddev_us: $row.gpu_us_per_dispatch.stddev
        cpu_encode_mean_us: $row.cpu_encode_us_per_dispatch.mean
    } }
}
