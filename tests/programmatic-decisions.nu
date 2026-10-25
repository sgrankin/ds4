#!/usr/bin/env nu
# One loaded generator/helper pair; the Python leaf is wire/SDK and numerical I/O.
const FIXTURES = path self '../speed-bench/experiments/ptc-decisions-20261025/fixtures.nu'

def main [
    --url: string = 'http://127.0.0.1:19333',
    --model: string = 'flint',
    --helper: string = 'clef-flash',
    --output: path = '.workspaces/ptc-decisions-20261025/wire',
    --python: path = '.workspaces/responses-ws-venv/bin/python',
    --suite: string = 'qualify',
    --case: string = 'all',
    --arm: string = 'all',
    --repeats: int = 2
] {
    ^$nu.current-exe $FIXTURES $output
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
    ^$python tests/programmatic-decisions-wire.py $url $model $helper $output --suite $suite --case $case --arm $arm --repeats $repeats
    if $env.LAST_EXIT_CODE != 0 { error make {msg: 'Programmatic decisions checks failed; inspect result.json.'} }
}
