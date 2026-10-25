#!/usr/bin/env nu
# Preserve the original two cases and add only a new layout for their same data.
const BASE = path self '../mcp-tools-20261025/fixtures.nu'

def main [output: path] {
    ^$nu.current-exe $BASE $output
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
    let path = $output | path join fixtures.json
    let original = open $path
    let favorable = $original | where name == favorable | first
    let unfamiliar = $favorable | merge {
        name: unfamiliar,
        fetch_tool: fetch_bundle,
        program_description: 'inputs.queue_name is only a lookup key; no source notes are bound. schemas.triage is the complete active-account-takeover choice template with active/other answers. The declared MCP tool returns the original evidence in an unfamiliar nested export. Use its advertised output schema or inspect keys and metadata as needed; do not assume fetch_queue field paths. tools.ds4_decide accepts questions and input text, returning answers. Question names label results; put each target ticket ID in its instructions. Emit only the evidence needed for the requested answer.'
    }
    $original | append $unfamiliar | to json --indent 2 | save --force $path
}
