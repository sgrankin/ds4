#!/usr/bin/env nu
# Same wire leaf and tasks; root owns the already-running guarded model server.
const FIXTURES = path self 'fixtures.nu'

def main [
    --url: string = 'http://127.0.0.1:19333',
    --model: string = 'deepseek-v4.1-flash',
    --helper: string = 'clef-flash',
    --output: path = '.workspaces/planning-abc-20261026/a-candidate',
    --python: path = '.workspaces/responses-ws-venv/bin/python',
    --case: string = 'all',
    --arm: string = 'ptc',
    --repeats: int = 1
] {
    ^$nu.current-exe $FIXTURES $output
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
    ^$python tests/mcp-tools-wire.py $url $model $helper $output --suite scenarios --case $case --arm $arm --repeats $repeats
    if $env.LAST_EXIT_CODE != 0 { error make {msg: 'Planning experiment failed; inspect scenarios.json.'} }
}
