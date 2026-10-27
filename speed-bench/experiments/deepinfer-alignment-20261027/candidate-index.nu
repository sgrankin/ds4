#!/usr/bin/env nu
# Run under speed-bench/ideas/pressure_run.py; one model process at a time.
def main [binary: path, output: path, --tokens: int = 32768, --case: string = 'all', --hot-loop] {
    mkdir $output
    let arms = if $hot_loop { [{name: dense, enabled: '0'}] } else { [{name: dense, enabled: '0'}, {name: candidates, enabled: '1'}, {name: dense-repeat, enabled: '0'}] }
    for arm in $arms {
        let dir = ($output | path join $arm.name)
        mkdir $dir
        with-env {DS4_V41_CANDIDATE_INDEX: $arm.enabled} {
            ^$binary --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf --out $dir --rows 0 --tokens $tokens --chunk 8192 --ctx ($tokens + 8192) --case $case --thinking 1 --max-new 384 --cache-gb 48 --candidate-sweep (if $hot_loop { 1 } else { 0 }) out> ($dir | path join stdout.log) err> ($dir | path join stderr.log)
            if $env.LAST_EXIT_CODE != 0 { error make {msg: $'($arm.name) failed; see ($dir)/stderr.log'} }
        }
        print $'Completed ($arm.name), ($tokens) prompt tokens'
    }
}
