#!/usr/bin/env nu
# Run under speed-bench/ideas/pressure_run.py; one model process at a time.
const SCRIPT = path self
def main [
    binary: path,
    output: path,
    --model: path = 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf',
    --tokens: int = 8448,
    --case: string = 'all',
    --max-new: int = 384,
    --teacher-dir: path,
    --poison-check
] {
    if $poison_check and $teacher_dir != null { error make {msg: 'Teacher-trace and poison checks use separate runs'} }
    if ($output | path exists) { error make {msg: $'Use a new output directory: ($output)'} }
    mkdir $output
    let arms = if $teacher_dir != null {
        [{directory: exact, mode: exact, verify: 0}, {directory: recover, mode: recover, verify: 0}]
    } else if $poison_check {
        [{directory: recover, mode: recover, verify: 0}, {directory: poisoned, mode: recover, verify: 1}]
    } else {
        [{directory: exact, mode: exact, verify: 0}, {directory: recover, mode: recover, verify: 0}, {directory: fresh, mode: fresh, verify: 0}, {directory: exact-repeat, mode: exact, verify: 0}]
    }
    let rows = if $poison_check { 2304 } else { $tokens }
    let selected = if $poison_check { 'ordinary' } else { $case }
    let teacher_args = if $teacher_dir != null { [--teacher-dir $teacher_dir] } else { [] }
    mut results = []
    for arm in $arms {
        let dir = ($output | path join $arm.directory)
        mkdir $dir
        let args = [--model $model --out $dir --mode $arm.mode --verify $arm.verify --tokens $rows --chunk 8192 --ctx ($rows + 8192) --case $selected --thinking 1 --max-new $max_new --cache-gb 48] | append $teacher_args
        {command: ([$binary] | append $args), mode: $arm.mode, verify: $arm.verify} | to json | save ($dir | path join execution.json)
        let status = with-env {DS4_V41_CHECKPOINT_DRAFT_FEATURES: '0', DS4_V41_CANDIDATE_INDEX: '0'} {
            ^$binary ...$args out> ($dir | path join stdout.log) err> ($dir | path join stderr.log)
            $env.LAST_EXIT_CODE
        }
        $results = ($results | append ($arm | insert returncode $status))
        $results | to json | save --force ($output | path join result.json)
        if $status != 0 { error make {msg: $'($arm.directory) failed; see ($dir)/stderr.log'} }
        print $'Completed ($arm.directory), ($rows) tokens, high reasoning, 48GiB expert cache'
    }
    let compare = ($SCRIPT | path dirname | path join compare-swa.py)
    ^python3 $compare $output out> ($output | path join summary.json)
    if $env.LAST_EXIT_CODE != 0 { error make {msg: 'SWA comparison failed; raw results retained'} }
    open ($output | path join summary.json)
}
