#!/usr/bin/env nu
# Twelve separate requests: control/recipe/recipe/control within each fixed case.
const RUN = path self 'run.nu'

def main [
    --url: string = 'http://127.0.0.1:19333',
    --model: string = 'deepseek-v4.1-flash',
    --helper: string = 'clef-flash',
    --output: path = '.workspaces/output-first-20261026/screen',
    --python: path = '.workspaces/responses-ws-venv/bin/python'
] {
    let variants = [control recipe recipe control]
    for case in [ordinary favorable unfamiliar] {
        for index in 0..3 {
            let variant = $variants | get $index
            let extra = if $variant == 'recipe' { ['--recipe'] } else { [] }
            let destination = $output | path join $'($case)-($index)-($variant)'
            ^$nu.current-exe $RUN --url $url --model $model --helper $helper --python $python --case $case --repeats 1 --output $destination ...$extra
            if $env.LAST_EXIT_CODE != 0 { error make {msg: $'Wire failure for ($case), ($index) ($variant); inspect artifacts.'} }
        }
    }
}
