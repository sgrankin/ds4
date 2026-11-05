#!/usr/bin/env nu
# Run under the pressure guard. This is behavior evidence, not task-speed data.
const ROOT = path self ../../..
const REPORT = path self .

def main [output?: path, --plain] {
    let work = $output | default ($ROOT | path join '.workspaces/hillclimb-20261104/14-resident-behavior/agent-check') | path expand
    let canonical = $ROOT | path join '.workspaces/hillclimb-20261104/14-resident-behavior/task'
    if ($work | path exists) { error make {msg: $'Output already exists: ($work)'} }
    if ($canonical | path exists) and not (ls -a $canonical | is-empty) {
        error make {msg: $'Canonical task directory is not empty: ($canonical)'}
    }
    for key in ($env | columns | where {|key| $key | str starts-with 'DS4'}) { hide-env $key }
    let published_file = $ROOT | path join 'speed-bench/experiments/2026-11-04-decode-pass/final-settings.json'
    let current_file = $ROOT | path join 'speed-bench/experiments/2026-11-04-resident-agent-qualification/candidate-settings.json'
    let published = open $published_file
    let current = open $current_file | transpose key value
        | where {|row| not ($row.key | str starts-with 'DS4_METAL_V41_RESIDENT_GPU') }
        | reduce --fold {} {|row, acc| $acc | insert $row.key $row.value }
    let background = open --raw ($ROOT | path join 'speed-bench/promessi_sposi.txt') | str substring 0..<12000
    let task = 'Create layers.py with a Python function topological_layers(nodes, edges). nodes is any iterable of distinct integers. edges is any iterable of pairs (u, v), meaning u must precede v; both endpoints are in nodes. Ignore duplicate edges. Return a list of lists in dependency order: each layer contains all remaining nodes whose predecessors were in earlier layers, sorted numerically within that layer. Include disconnected nodes. Return [] for no nodes. Raise ValueError if any cycle exists, including a self-loop or a cycle disconnected from otherwise valid nodes. Do not change either input. Both arguments may be lists, iterators, or generators, including empty generators. No command-line interface. Verify your implementation with a few tests, then respond briefly.'
    let prompt = if $plain { $task } else {
        'The following reference text is unrelated background. Do not summarize it; complete only the coding task after the closing tag.' + (char nl) + '<background>' + (char nl) + $background + (char nl) + '</background>' + (char nl) + $task
    }
    let arguments = [
        '--model' ($ROOT | path join 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf')
        '--metal' '--ssd-streaming' '--ssd-streaming-cache-experts' '64GB'
        '--ctx' '16384' '--prefill-chunk' '8192' '--non-interactive'
        '--think' '--temp' '0' '--tokens' '2048' '--chdir' $canonical '--prompt' $prompt
    ]
    mkdir $work $canonical
    $prompt | save ($work | path join prompt.txt)
    for arm in [published current_gpu_off gpu_refresh4 gpu_refresh8 gpu_refresh16] {
        let source = if $arm == published { $ROOT | path join '.workspaces/hillclimb-20261104/starting-364ceeab' } else { $ROOT }
        let settings_file = if $arm == published { $published_file } else { $current_file }
        mut settings = if $arm == published { $published } else { $current }
        if ($arm | str starts-with 'gpu_refresh') {
            $settings = ($settings | merge {
                DS4_METAL_V41_RESIDENT_GPU: '1'
                DS4_METAL_V41_RESIDENT_GPU_REUSE_ADDR: '1'
                DS4_METAL_V41_RESIDENT_GPU_PARALLEL: '1'
                DS4_METAL_V41_RESIDENT_GPU_REFRESH: ($arm | str replace 'gpu_refresh' '')
            })
        }
        $settings = ($settings | transpose key value
            | where {|row| $row.key !~ '(PROFILE|TIMELINE|TRACE|CB_TIMES)' }
            | reduce --fold {} {|row, acc| $acc | insert $row.key $row.value }
            # Freeze the injected datetime context at 2026-11-04 18:33:00 EST.
            | merge {
                DS4_AGENT_TEST_TIME: '1793835180'
                TZ: 'America/New_York'
                # System-prefix KV is automatically cached; keep arms isolated.
                DS4_AGENT_CACHE_DIR: ($work | path join $arm cache)
            })
        let binary = $source | path join zig-out/bin/ds4-agent
        if not ($binary | path exists) { error make {msg: $'Missing agent binary: ($binary)'} }
        let binary_sha256 = ^shasum -a 256 $binary | split row ' ' | first
        let dest = $work | path join $arm
        mkdir $dest ($dest | path join cache)
        $settings | to json | save ($dest | path join environment.json)
        {source_root: $source, settings_file: $settings_file, binary: $binary, binary_sha256: $binary_sha256, arguments: $arguments}
            | to json | save ($dest | path join arguments.json)
        print $'start behavior ($arm) (date now)'
        with-env $settings {
            cd $source
            ^$binary ...$arguments out> ($dest | path join stdout.log) err> ($dest | path join stderr.log)
            {exit_code: $env.LAST_EXIT_CODE} | to json | save ($dest | path join exit.json)
        }
        # Move the complete task directory, including hidden files. Never delete
        # a failed arm; a fresh empty directory presents the same next prompt.
        mv $canonical ($dest | path join workdir)
        mkdir $canonical
        print $'done behavior ($arm) (date now)'
    }
    ^python3 ($REPORT | path join check-layers.py) $work
    if $env.LAST_EXIT_CODE != 0 { error make {msg: $'Behavior checks failed; see ($work)/oracle-result.json'} }
}
