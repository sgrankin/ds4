#!/usr/bin/env nu
# Fixed-token screen. Run one family at a time; raw results belong in .workspaces/.
const ROOT = path self ../../..
const INPUT = 'speed-bench/experiments/zig-metal-f32-hour26-20261022/prompt.txt'

def summarize [runs: list] {
    mut comparison = []
    for workload in ($runs.workload | uniq) {
        let group = $runs | where workload == $workload
        let base = $group | where state == baseline-off
        let off = $group | where state == candidate-off
        for state in [candidate-off candidate-zero candidate-active] {
            let selected = $group | where state == $state
            let reference = if $state == candidate-off { $base } else { $off }
            for i in 0..(($selected | first | get samples | length) - 1) {
                let a = $reference | each {|r| $r.samples | get $i }
                let b = $selected | each {|r| $r.samples | get $i }
                $comparison = $comparison | append {
                    workload: $workload
                    frontier: ($b | first | get ctx_tokens)
                    state: $state
                    reference: (if $state == candidate-off { 'baseline-off' } else { 'candidate-off' })
                    prefill_time_change_pct: (100 * (($b.prefill_s | math avg) / ($a.prefill_s | math avg) - 1))
                    decode_time_change_pct: (if ($b | first | get gen_tokens) > 0 { 100 * (($b.decode_s | math avg) / ($a.decode_s | math avg) - 1) } else { null })
                    reference_prefill_s: $a.prefill_s
                    selected_prefill_s: $b.prefill_s
                    reference_decode_s: $a.decode_s
                    selected_decode_s: $b.decode_s
                    all_frontiers_exact: (($reference.frontier_sha256 | append $selected.frontier_sha256 | uniq | length) == 1)
                }
            }
        }
    }
    $comparison
}

# Reuse completed model runs when only analysis or presentation changes.
def "main summarize" [out: string] {
    let out = $out | path expand
    let comparison = summarize (open ($out | path join runs.json))
    $comparison | to json | save --force ($out | path join comparison.json)
    print $comparison
}

def main [
    family: string # ds41, flint, qwen or glm.
    direction: string # The family's residual-stream f32 bank.
    candidate: string # Candidate executable directory.
    out: string # Fresh ignored output directory.
    --baseline: string = '.workspaces/steering-20261024/baseline/bin'
    --model: string # Override the installed model path.
    --ffn: float = 0.1
    --attn: float = 0.0
    --mode: string = project # project or add.
    --tokens: int = 32 # Exact teacher-forced decode count.
    --repeat: int = 1 # Each cycle runs all four states in mirrored order.
    --workload: string = all # all, short, append or context.
    --dry-run # Print commands; do not write files or run models.
] {
    cd $ROOT
    let baseline = $baseline | path expand
    let candidate = $candidate | path expand
    let direction = $direction | path expand
    let out = $out | path expand
    if $tokens < 2 or $repeat < 1 { error make {msg: 'Use tokens >= 2 and repeat >= 1'} }
    if $ffn == 0 and $attn == 0 { error make {msg: 'The active state needs a nonzero scale'} }
    if $mode != project and $mode != add { error make {msg: '--mode must be project or add'} }
    let settings = match $family {
        ds41 => {
            model: 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
            flags: [--ssd-streaming --ssd-streaming-cache-experts '32GB' --prefill-chunk '2048']
            environment: {
                DS4_V41_DECODER_REPLAY_ROWS: '0'
                DS4_METAL_V41_RESIDENT_ROUTE_GAP: '0'
                DS4_METAL_V41_RESIDENT_PREFILL_GAP: '0'
                DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS: '0'
            }
        }
        flint => {
            model: '../flint/.local/share/rec/models/fdc85248ad77a0fa9fbf0abda8b1525bea8ee74c11ada2ec95eb66113353d677'
            flags: [--prefill-chunk '4096']
            environment: {}
        }
        qwen => {
            model: 'gguf/Qwen3.8-Flash-Next-Q4.gguf'
            flags: [--prefill-chunk '1024']
            environment: {DS4_METAL_NO_RESIDENCY: '1', DS4_METAL_NO_MODEL_WARMUP: '1'}
        }
        glm => {
            model: 'gguf/GLM-5.3-Flash-Q2.gguf'
            flags: [--ssd-streaming --ssd-streaming-cache-experts '24GB']
            environment: {DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP: '1'}
        }
        _ => { error make {msg: $'Unknown model family: ($family)'} }
    }
    let model = $model | default $settings.model
    let workloads = [
        {name: short, first: 256, last: 256, generated: $tokens}
        {name: append, first: 2047, last: 4095, generated: 0}
        {name: context, first: 4095, last: 4095, generated: $tokens}
    ] | where {|w| $workload == all or $workload == $w.name }
    if ($workloads | is-empty) { error make {msg: $'Unknown workload: ($workload)'} }
    # Do not inherit a shell's resident-expert, cache or replay preset.
    for name in ($env | columns | where {|name|
        ($name | str starts-with DS4) or ($name | str starts-with PT_)
    }) { hide-env $name }
    if not $dry_run {
        if ($out | path exists) { error make {msg: $'Use a fresh output directory: ($out)'} }
        mkdir $out
    }
    let states = [
        {name: baseline-off, bin: $baseline, flags: []}
        {name: candidate-off, bin: $candidate, flags: [--dir-steering-mode $mode]}
        {name: candidate-zero, bin: $candidate, flags: [--dir-steering-mode $mode --dir-steering-file $direction --dir-steering-ffn '0' --dir-steering-attn '0']}
        {name: candidate-active, bin: $candidate, flags: [--dir-steering-mode $mode --dir-steering-file $direction --dir-steering-ffn ($ffn | into string) --dir-steering-attn ($attn | into string)]}
    ]
    mut runs = []
    for w in $workloads {
        for cycle in 1..$repeat {
            for index in 0..7 {
                let state = $states | get ([0 1 2 3 3 2 1 0] | get $index)
                let folder = $out | path join $w.name $'($cycle)-($index)-($state.name)'
                let argv = [
                    ($state.bin | path join ds4-bench)
                    --model $model --metal --prompt-file $INPUT
                    --ctx-start ($w.first | into string) --ctx-max ($w.last | into string)
                    --step-incr '2048' --ctx-alloc '8192'
                    --gen-tokens ($w.generated | into string) --teacher-forced-decode
                    --csv ($folder | path join timings.csv)
                    --dump-frontier-logits-dir ($folder | path join logits)
                    ...$settings.flags ...$state.flags
                ]
                let command = {family: $family, mode: $mode, workload: $w.name, state: $state.name, cycle: $cycle, order: $index, argv: $argv, environment: $settings.environment}
                if $dry_run { print ($command | to json --raw); continue }
                mkdir ($folder | path join logits)
                $command | to json | save ($folder | path join command.json)
                let started = date now
                let code = with-env $settings.environment {
                    ^python3 -B speed-bench/ideas/pressure_run.py ($folder | path join guard) -- ...$argv out> ($folder | path join stdout.log) err> ($folder | path join stderr.log)
                    $env.LAST_EXIT_CODE
                }
                if $code != 0 { error make {msg: $'($state.name) failed ($code); see ($folder)'} }
                let samples = open ($folder | path join timings.csv)
                if ($samples.gen_tokens | any {|n| $n != $w.generated }) {
                    error make {msg: $'Decode stopped before the fixed count; see ($folder)'}
                }
                if $w.name == append and ($samples | last | get prefill_tokens) != 2048 {
                    error make {msg: $'Expected a 2048-token live append; see ($folder)'}
                }
                let hashes = $samples | each {|sample|
                    let position = $sample.ctx_tokens
                    let file = $folder | path join logits $'frontier_($position | fill --alignment right --character 0 --width 6).logits.json'
                    let hash = ^python3 -c 'import hashlib,json,struct,sys; x=json.load(open(sys.argv[1]))["logits"]; print(hashlib.sha256(struct.pack("<%df"%len(x),*x)).hexdigest())' $file
                    if $env.LAST_EXIT_CODE != 0 { error make {msg: $'Cannot hash frontier: ($file)'} }
                    {position: $position, sha256: ($hash | str trim)}
                }
                let row = $command | merge {
                    wall_s: (((date now) - $started) / 1sec)
                    samples: ($samples | each {|s| $s | merge {
                        prefill_s: ($s.prefill_tokens / $s.prefill_tps)
                        decode_s: (if $s.gen_tokens > 0 { $s.gen_tokens / $s.gen_tps } else { null })
                    } })
                    frontier_sha256: $hashes
                }
                $runs = $runs | append $row
                $runs | to json | save --force ($out | path join runs.json)
                print ($row | reject argv environment | to json --raw)
            }
        }
    }
    if $dry_run { return }
    let comparison = summarize $runs
    $comparison | to json | save ($out | path join comparison.json)
    print $comparison
}
