#!/usr/bin/env nu
# One process per cell releases session allocations. OS cache and thermals persist.
def --wrapped main [csv: path, --bench: path = ./speed-bench/session_concurrency_bench, ...args: string] {
    let concurrency = (
        $env.CONCURRENCY?
        | default '1 2 4 8 16'
        | split row -r '\s+'
        | into int
    )
    let contexts = (
        $env.CONTEXTS?
        | default '0 4096 16384 32768 65536 131072'
        | split row -r '\s+'
        | into int
    )
    mut failed = false
    for ctx in $contexts {
        for count in $concurrency {
            print $'=== concurrency ($count), context ($ctx) ==='
            let result = do { ^$bench --concurrency $count --ctx $ctx --csv $csv ...$args } | complete
            print --no-newline $result.stdout
            print --stderr --no-newline $result.stderr
            let status = $result.exit_code
            if $status != 0 {
                print --stderr $'cell concurrency=($count) ctx=($ctx) failed with ($status)'
                $failed = true
            }
        }
    }
    print $'wrote ($csv)'
    exit (if $failed { 1 } else { 0 })
}
