#!/usr/bin/env nu
# Protocol qualification against an already loaded generator and decision helper.
def main [
    --url: string = 'http://127.0.0.1:19332',
    --model: string = 'flint',
    --helper: string = 'clef-flash',
    --output: path = '.workspaces/hosted-decisions-20261025/wire',
    --python: path = '.workspaces/responses-ws-venv/bin/python',
    --after-restart,
    --edges,
    --media,
    --mixed-only
] {
    let extra = if $media { [--media] } else if $mixed_only { [--mixed-only] } else if $after_restart { [--after-restart] } else { [] }
    let leaf = if $edges or $media or $mixed_only { 'tests/hosted-decisions-edge-wire.py' } else { 'tests/hosted-decisions-wire.py' }
    ^$python $leaf $url $model $helper $output ...$extra
    if $env.LAST_EXIT_CODE != 0 { error make {msg: 'Hosted decision qualification failed; inspect result.json.'} }
}
