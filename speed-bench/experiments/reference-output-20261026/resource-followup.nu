#!/usr/bin/env nu
# Only the shared call budget changes; root owns the already-running server.
const RUN = path self 'run.nu'

def main [
    --url: string = 'http://127.0.0.1:19333',
    --model: string = 'deepseek-v4.1-flash',
    --helper: string = 'clef-flash',
    --output: path = '.workspaces/reference-output-20261026/resource-screen',
    --python: path = '.workspaces/responses-ws-venv/bin/python'
] {
    for case in [favorable unfamiliar] {
        ^$nu.current-exe $RUN --url $url --model $model --helper $helper --python $python --inline-bytes 2048 --max-tool-calls 12 --case $case --repeats 2 --output ($output | path join $case)
        if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
    }
}
