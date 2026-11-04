#!/usr/bin/env nu
# Run under speed-bench/ideas/pressure_run.py; this driver does not build binaries.
# Supply the final settings as JSON so the adopted package is explicit.
# Each binary must run in its own source root: Metal sources load relative to cwd.
# Timing/quality outputs match run.nu. Replay is one teacher-logit diagnostic pair.
# Replay-timing compares stdout phase rows without teacher-logit capture. Its wall
# time also includes loading and the final snapshot required by the replay tool.
def main [
    out: path
    stage: string = 'timing'
    --final-settings: path
    --baseline-settings: path
    --baseline-root: path
    --candidate-root: path
    --baseline-revision: string = '8971dfbd'
    --candidate-revision: string = 'unrecorded'
    --model: path
    --prompt: path
    --replay: path
    --cache: string = '64GB'
    --frontier: int = 0
    --tokens: int = 128
    --samples: int = 5
    --only: string = 'both'
] {
    if $stage not-in ['timing' 'quality' 'replay' 'replay-timing'] {
        error make {msg: 'Stage must be timing, quality, replay or replay-timing'}
    }
    if $only not-in ['both' 'baseline' 'candidate'] {
        error make {msg: '--only must be both, baseline or candidate'}
    }
    if $final_settings == null {
        error make {msg: '--final-settings must name the adopted environment JSON'}
    }
    if $samples < 1 or $tokens < 1 or $frontier < 0 {
        error make {msg: 'Samples and tokens must be positive; frontier must be nonnegative'}
    }

    # Resolve all paths before entering either checkout. The workload is shared,
    # while executable and runtime shader sources belong to the selected arm.
    let repo = $env.FILE_PWD | path join '../../..' | path expand
    let out = $out | path expand
    let baseline_root = ($baseline_root | default ($repo | path join '.workspaces/overnight-decode-20261104/starting-8971dfbd')) | path expand
    let candidate_root = ($candidate_root | default $repo) | path expand
    let baseline_settings_path = ($baseline_settings | default ($repo | path join 'speed-bench/experiments/2026-11-04-decode-pass/starting-settings.json')) | path expand
    let final_settings_path = $final_settings | path expand
    let model = ($model | default ($repo | path join 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf')) | path expand
    let prompt = ($prompt | default ($repo | path join 'speed-bench/promessi_sposi.txt')) | path expand
    let replay = ($replay | default ($repo | path join 'speed-bench/agent-session/session-v1.txt')) | path expand
    let baseline_settings = open --raw $baseline_settings_path | from json
    let final_settings = open --raw $final_settings_path | from json
    let frontier = if $frontier > 0 { $frontier } else if $stage == 'quality' { 8192 } else { 32768 }
    let repeats = if $stage in ['timing' 'replay-timing'] { $samples } else { 1 }
    let binary_name = if $stage in ['replay' 'replay-timing'] { 'agent-session-replay' } else { 'ds4-bench' }
    let arms = {
        baseline: {cwd: $baseline_root, revision: $baseline_revision, settings: $baseline_settings, settings_path: $baseline_settings_path}
        candidate: {cwd: $candidate_root, revision: $candidate_revision, settings: $final_settings, settings_path: $final_settings_path}
    }
    mkdir $out
    for key in ($env | columns | where {|key| $key | str starts-with 'DS4'}) { hide-env $key }

    for repeat in 1..$repeats {
        let order = if ($repeat mod 2) == 0 { ['candidate' 'baseline'] } else { ['baseline' 'candidate'] }
        for arm in $order {
            if $only != 'both' and $arm != $only { continue }
            let config = $arms | get $arm
            let executable = $config.cwd | path join 'zig-out/bin' $binary_name | path expand --strict
            let executable_sha256 = ^shasum -a 256 $executable | split row ' ' | first
            if $env.LAST_EXIT_CODE != 0 { error make {msg: $'Cannot hash ($executable)'} }
            let dest = $out | path join $'($repeat)-($arm)'
            mkdir $dest
            let logits = $dest | path join 'logits'
            if $stage in ['quality' 'replay'] { mkdir $logits }
            let diagnostic = if $stage == 'quality' {
                {DS4_BENCH_DECODE_LOGITS_HASH: '1'}
            } else if $stage in ['replay' 'replay-timing'] {
                {
                    DS4_REPLAY_CONTEXT: '16384'
                    DS4_REPLAY_CACHE_GB: ($cache | str replace 'GB' '' | into int | into string)
                    DS4_REPLAY_PREFILL_CHUNK: '8192'
                } | merge (if $stage == 'replay' { {
                    DS4_REPLAY_TEACHER_DIR: $logits
                    DS4_REPLAY_SKIP_FINAL_SNAPSHOT: '1'
                } } else { {} })
            } else { {} }
            let settings = $config.settings | merge $diagnostic
            let args = if $stage in ['replay' 'replay-timing'] {
                let snapshot = if $stage == 'replay' { 'unused-snapshot.bin' } else { 'final-snapshot.bin' }
                [$model $replay ($dest | path join 'phase-logits.bin') ($dest | path join $snapshot)]
            } else {
                let quality_args = if $stage == 'quality' {
                    [--dump-frontier-logits-dir $logits --dump-decode-logits-dir $logits]
                } else { [] }
                [
                    --model $model --metal --ssd-streaming --ssd-streaming-cache-experts $cache
                    --prompt-file $prompt
                    --ctx-start ($frontier | into string) --ctx-max (($frontier + 512) | into string)
                    --step-incr 512 --ctx-alloc 40960 --prefill-chunk 8192
                    --gen-tokens ($tokens | into string) --teacher-forced-decode
                    --csv ($dest | path join 'timings.csv') ...$quality_args
                ]
            }
            $settings | to json | save ($dest | path join 'environment.json')
            $args | to json | save ($dest | path join 'arguments.json')
            {
                stage: $stage, repeat: $repeat, arm: $arm, order: $order
                cwd: $config.cwd, revision: $config.revision
                executable: $executable, executable_sha256: $executable_sha256
                settings_path: $config.settings_path, settings: $settings, arguments: $args
            } | to json | save ($dest | path join 'manifest.json')
            print $'start ($repeat)/($repeats) ($arm) (date now)'
            let started = date now
            let status = with-env $settings {
                cd $config.cwd
                /usr/bin/time -p $executable ...$args out> ($dest | path join 'stdout.log') err> ($dest | path join 'stderr.log')
                $env.LAST_EXIT_CODE
            }
            let wall_s = ((date now) - $started) / 1sec
            {repeat: $repeat, arm: $arm, wall_s: $wall_s, exit_code: $status} | to json | save ($dest | path join 'wall.json')
            if $status != 0 { error make {msg: $'Benchmark failed: ($dest)'} }
            print $'done ($arm) ($wall_s)s'
        }
    }
}
