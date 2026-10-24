#!/usr/bin/env nu
# Frozen/native target comparison. Execute only under the sole model-job owner.
const ROOT = path self ../../..

def main [
    candidate: string
    --model: string = 'gguf/Qwen3.8-Flash-Next-Q4.gguf'
    --out: string = '.workspaces/model-family-ports-20261023/qwen-depth3'
    --tokens: int = 48
    --dry-run
] {
    cd $ROOT
    let candidate = $candidate | path expand
    let model = $model | path expand
    let out = $out | path expand
    let baseline = $ROOT | path join .workspaces/model-family-ports-20261023/baseline/bin/ds4
    let cases = [
        {name: counting, prompt: 'Count from one to fifty, separated by commas. Output only the sequence.'}
        {name: prose, prompt: 'Continue this story for at least 150 words: Mira found a bicycle with a flat tire outside the library on a rainy morning.'}
    ]
    let settings = {
        DS4_METAL_NO_RESIDENCY: '1'
        DS4_METAL_NO_MODEL_WARMUP: '1'
        DS4_QWEN4_MTP_DEPTH: '3'
        DS4_QWEN4_SPEC_TRACE: '1'
    }
    mut commands = []
    for case in $cases {
        for arm in [{name: frozen, binary: $baseline} {name: native, binary: $candidate}] {
            let dir = $out | path join $case.name $arm.name
            # Logprob-dump diagnostics have their own scalar loop; use the real
            # CLI packet path with its17-token output buffer for this check.
            let argv = [$arm.binary --model $model --metal --ctx '512' --prefill-chunk '128' --nothink --temp '0' --tokens ($tokens | into string) --mtp --seed '12345' -p $case.prompt]
            $commands = $commands | append {case: $case.name, arm: $arm.name, directory: $dir, argv: $argv}
        }
    }
    if $dry_run {
        {baseline_revision: '030a040e', settings: $settings, commands: $commands} | to json | print
        return
    }
    mkdir $out
    {baseline_revision: '030a040e', settings: $settings, commands: $commands} | to json | save --force ($out | path join commands.json)
    for name in ($env | columns | where {|name| ($name | str starts-with DS4) or ($name | str starts-with PT_)}) {
        hide-env $name
    }
    for command in $commands {
        mkdir $command.directory
        let argv = $command.argv
        with-env $settings {
            ^python3 -B speed-bench/ideas/pressure_run.py ($command.directory | path join guard) -- ...$argv out> ($command.directory | path join stdout.log) err> ($command.directory | path join stderr.log)
            if $env.LAST_EXIT_CODE != 0 { error make {msg: $'guarded ($command.case)/($command.arm) run failed'} }
        }
    }
    mut summary = []
    for case in $cases {
        let frozen = $out | path join $case.name frozen
        let native = $out | path join $case.name native
        let frozen_trace = open --raw ($frozen | path join stderr.log) | lines | where {|line| $line | str starts-with 'ds4: spec pos '}
        let native_trace = open --raw ($native | path join stderr.log) | lines | where {|line| $line | str starts-with 'ds4: spec pos '}
        let outputs_equal = (open --raw ($frozen | path join stdout.log)) == (open --raw ($native | path join stdout.log))
        let traces_equal = $frozen_trace == $native_trace
        let three_rows = $native_trace | where {|line| ($line | str contains ' +accept2') or ($line | str contains ' +reject2')} | length
        let partial_rejections = $native_trace | where {|line| $line | str ends-with ' accept +reject2'} | length
        $summary = $summary | append {case: $case.name, outputs_equal: $outputs_equal, traces_equal: $traces_equal, verify_three_rows: $three_rows, partial_rejections: $partial_rejections}
    }
    $summary | to json | save --force ($out | path join summary.json)
    print $summary
    if ($summary | any {|row| not $row.outputs_equal or not $row.traces_equal}) {
        error make {msg: $'frozen/native mismatch; see ($out)'}
    }
    if ($summary.verify_three_rows | math sum) == 0 {
        error make {msg: 'three-row verification was not reached'}
    }
    if ($summary.partial_rejections | math sum) == 0 {
        error make {msg: 'outputs matched, but snap2 partial rejection was not reached; coverage incomplete'}
    }
}
