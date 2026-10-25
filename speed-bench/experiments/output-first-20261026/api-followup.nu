#!/usr/bin/env nu
# Only the API recipe suffix changes; root owns the frozen, running C server.
const FIXTURES = path self '../planning-abc-20261026/fixtures.nu'
const RECIPE = path self 'api-recipe.txt'

def main [
    --url: string = 'http://127.0.0.1:19333',
    --model: string = 'deepseek-v4.1-flash',
    --helper: string = 'clef-flash',
    --output: path = '.workspaces/output-first-20261026/api-screen',
    --python: path = '.workspaces/responses-ws-venv/bin/python'
] {
    for case in [ordinary favorable unfamiliar] {
        let destination = $output | path join $case
        ^$nu.current-exe $FIXTURES $destination
        if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
        ^$python tests/mcp-tools-wire.py $url $model $helper $destination --suite scenarios --case $case --arm ptc --repeats 2 --program-recipe $RECIPE
        if $env.LAST_EXIT_CODE != 0 { error make {msg: $'API-recipe wire failure for ($case); inspect scenarios.json.'} }
    }
}
