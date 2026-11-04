#!/usr/bin/env nu
# Run from the repository root, under the existing pressure guard.
# These four independent tasks provide behavioral evidence, not task-speed data.
def main [output?: path] {
    let root = $env.PWD
    let report = $root | path join 'speed-bench/experiments/2026-11-04-resident-agent-qualification'
    let work = $output | default ($root | path join '.workspaces/hillclimb-20261104/13-resident-agent-qualification/agent-check') | path expand
    if ($work | path exists) { error make {msg: $'Output already exists: ($work)'} }
    for key in ($env | columns | where {|key| $key | str starts-with 'DS4'}) { hide-env $key }
    let background = open --raw ($root | path join 'speed-bench/promessi_sposi.txt') | str substring 0..<12000
    let tasks = [
        {name: 'intervals', prompt: 'Create intervals.py with a Python function merge_ranges(ranges). Inputs are pairs of integers representing inclusive intervals, each individually valid. Return a sorted list of tuples, merging overlapping or adjacent intervals, without changing the input. The input may be any iterable, including an empty generator. Handle empty input. No command-line interface. Verify your implementation with a few tests, then respond briefly.'}
        {name: 'layers', prompt: 'Create layers.py with a Python function topological_layers(nodes, edges). nodes is any iterable of distinct integers. edges is any iterable of pairs (u, v), meaning u must precede v; both endpoints are in nodes. Ignore duplicate edges. Return a list of lists in dependency order: each layer contains all remaining nodes whose predecessors were in earlier layers, sorted numerically within that layer. Include disconnected nodes. Return [] for no nodes. Raise ValueError if any cycle exists, including a self-loop or a cycle disconnected from otherwise valid nodes. Do not change either input. Both arguments may be lists, iterators, or generators, including empty generators. No command-line interface. Verify your implementation with a few tests, then respond briefly.'}
    ]
    for arm in ['baseline' 'candidate'] {
        let source = if $arm == 'baseline' { $root | path join '.workspaces/hillclimb-20261104/starting-364ceeab' } else { $root }
        let settings_file = if $arm == 'baseline' { $root | path join 'speed-bench/experiments/2026-11-04-decode-pass/final-settings.json' } else { $report | path join 'candidate-settings.json' }
        let settings = open $settings_file | transpose key value
            | where {|row| $row.key !~ '(PROFILE|TIMELINE|TRACE|CB_TIMES)' }
            | reduce --fold {} {|row, acc| $acc | insert $row.key $row.value }
        let binary = $source | path join 'zig-out/bin/ds4-agent'
        if not ($binary | path exists) { error make {msg: $'Missing agent binary: ($binary)'} }
        let binary_sha256 = ^shasum -a 256 $binary | split row ' ' | first
        for task in $tasks {
            let dest = $work | path join $arm $task.name
            mkdir $dest
            let prompt = 'The following reference text is unrelated background. Do not summarize it; complete only the coding task after the closing tag.' + (char nl) + '<background>' + (char nl) + $background + (char nl) + '</background>' + (char nl) + $task.prompt
            let arguments = [
                '--model' ($root | path join 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf')
                '--metal' '--ssd-streaming' '--ssd-streaming-cache-experts' '64GB'
                '--ctx' '16384' '--prefill-chunk' '8192' '--non-interactive'
                '--think' '--temp' '0' '--tokens' '1024' '--chdir' $dest '--prompt' $prompt
            ]
            $settings | to json | save ($dest | path join 'environment.json')
            {source_root: $source, settings_file: $settings_file, binary: $binary, binary_sha256: $binary_sha256, arguments: $arguments}
                | to json | save ($dest | path join 'arguments.json')
            print $'start agent ($arm) ($task.name) (date now)'
            with-env $settings {
                cd $source
                ^$binary ...$arguments out> ($dest | path join 'stdout.log') err> ($dest | path join 'stderr.log')
                {exit_code: $env.LAST_EXIT_CODE} | to json | save ($dest | path join 'exit.json')
            }
            print $'done agent ($arm) ($task.name) (date now)'
        }
    }
    python3 ($report | path join 'check-agent.py') $work
    if $env.LAST_EXIT_CODE != 0 { error make {msg: $'Behavior checks failed; see ($work)/oracle-result.json'} }
}
