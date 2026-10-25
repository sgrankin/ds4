#!/usr/bin/env nu
# Rotate the budgets across the same three cases; do not tune between samples.
const RUN = path self run.nu

def main [--output: path = '.workspaces/reference-output-20261026/screen'] {
    let cases = [ordinary favorable unfamiliar]
    let budgets = [0 512 2048]
    for round in 0..2 {
        for index in 0..2 {
            let case = $cases | get $index
            let budget = $budgets | get (($round + $index) mod 3)
            let destination = $output | path join $'($case)-($budget)'
            ^$nu.current-exe $RUN --case $case --inline-bytes $budget --repeats 1 --output $destination
            if $env.LAST_EXIT_CODE != 0 { error make {msg: $'Wire failure for ($case), inline ($budget); inspect artifacts.'} }
        }
    }
}
