#!/usr/bin/env nu
# Run against an already loaded main engine plus --decision-model helper.
# The Python leaf reuses the established WS wire client and official SDK.
def main [
    --url: string = 'http://127.0.0.1:19285',
    --main-model: string = 'flint',
    --helper-model: string = 'clef-flash',
    --output: path = '.workspaces/clef-mixed-serving',
    --python: path = '.workspaces/responses-ws-venv/bin/python',
    --repeats: int = 3,
    --timing-only
    --mixed-timing-only
    --helper-repeat: int = 1
] {
    let extra = if $timing_only { [--timing-only] } else { [] }
    let mixed = if $mixed_timing_only { [--mixed-timing-only] } else { [] }
    ^$python tests/mixed-serving-wire.py $url $main_model $helper_model $output --repeats $repeats --helper-repeat $helper_repeat ...$extra ...$mixed
    if $env.LAST_EXIT_CODE != 0 { error make {msg: 'Mixed serving qualification failed; inspect result.json.'} }
}
