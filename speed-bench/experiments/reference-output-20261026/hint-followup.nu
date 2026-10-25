#!/usr/bin/env nu
# Only reference feedback changed; keep the two failed 2048-byte tasks fixed.
const RUN = path self run.nu

def main [--output: path = '.workspaces/reference-output-20261026/hint-screen'] {
    for case in [favorable unfamiliar] {
        ^$nu.current-exe $RUN --case $case --inline-bytes 2048 --repeats 2 --output ($output | path join $case)
        if $env.LAST_EXIT_CODE != 0 { error make {msg: $'Wire failure in hint follow-up: ($case)'} }
    }
}
