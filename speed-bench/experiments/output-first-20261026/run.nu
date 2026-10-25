#!/usr/bin/env nu
# Description-only comparison against the frozen C server; no model is launched.
const FIXTURES = path self '../planning-abc-20261026/fixtures.nu'
const RECIPE = path self 'recipe.txt'

def main [
    --url: string = 'http://127.0.0.1:19333',
    --model: string = 'deepseek-v4.1-flash',
    --helper: string = 'clef-flash',
    --recipe,
    --output: path,
    --python: path = '.workspaces/responses-ws-venv/bin/python',
    --case: string = 'all',
    --repeats: int = 1
] {
    let variant = if $recipe { 'recipe' } else { 'control' }
    let destination = if $output == null {
        '.workspaces/output-first-20261026' | path join $variant
    } else { $output }
    let extra = if $recipe { ['--program-recipe' $RECIPE] } else { [] }
    ^$nu.current-exe $FIXTURES $destination
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
    ^$python tests/mcp-tools-wire.py $url $model $helper $destination --suite scenarios --case $case --arm ptc --repeats $repeats ...$extra
    if $env.LAST_EXIT_CODE != 0 { error make {msg: 'Output-first recipe request failed; inspect scenarios.json.'} }
}
