#!/usr/bin/env nu
# Reuse the frozen planning tasks; root owns the existing guarded model server.
const FIXTURES = path self '../planning-abc-20261026/fixtures.nu'

def main [
    --url: string = 'http://127.0.0.1:19333',
    --model: string = 'deepseek-v4.1-flash',
    --helper: string = 'clef-flash',
    --inline-bytes: int = 0,
    --max-tool-calls: int = 8,
    --output: path,
    --python: path = '.workspaces/responses-ws-venv/bin/python',
    --case: string = 'all',
    --repeats: int = 1
] {
    let destination = if $output == null {
        '.workspaces/reference-output-20261026' | path join $'inline-($inline_bytes)'
    } else { $output }
    ^$nu.current-exe $FIXTURES $destination
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
    ^$python tests/mcp-tools-wire.py $url $model $helper $destination --suite scenarios --case $case --arm ptc --repeats $repeats --inline-bytes $inline_bytes --max-tool-calls $max_tool_calls
    if $env.LAST_EXIT_CODE != 0 { error make {msg: 'Reference-output experiment failed; inspect scenarios.json.'} }
}
