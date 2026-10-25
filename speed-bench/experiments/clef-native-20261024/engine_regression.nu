#!/usr/bin/env nu
# Reuse the existing benchmark and numerical comparator on fixed model inputs.
const ROOT = path self ../../..

def main [
    control: path, candidate: path,
    --family: string = all,
    --output: path = '.workspaces/clef-engine-regression',
    --flint-model: path = 'gguf/flint-int4'
] {
    cd $ROOT
    mkdir $output
    let prompt = $output | path join prompt.txt
    (1..30 | each {'Local model serving keeps each conversation and its reusable memory independent. The next request continues the same recorded sequence.'} | str join "\n") | save -f $prompt
    let cases = [
        {family: ds41, model: gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf, flags: [--ssd-streaming --ssd-streaming-cache-experts '32GB' --prefill-chunk '64']}
        {family: qwen, model: gguf/Qwen3.8-Flash-Next-Q4.gguf, flags: []}
        {family: glm, model: gguf/GLM-5.3-Flash-Q2.gguf, flags: [--ssd-streaming --ssd-streaming-cache-experts '24GB']}
        {family: flint, model: $flint_model, flags: []}
    ]
    # Keep resident substitutions and prefill experiments out of this arithmetic
    # control. These are the existing bounded GLM/Qwen qualification settings.
    let inherited = $env | columns | where {|key| ($key | str starts-with 'DS4_') or ($key | str starts-with 'PT_') }
    hide-env --ignore-errors ...$inherited
    $env.DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP = '1'
    for case in ($cases | where {|item| $family == all or $item.family == $family }) {
        for variant in [{name: control, binary: $control} {name: candidate, binary: $candidate}] {
            let dir = $output | path join $case.family $variant.name
            mkdir $dir
            let dumps = $dir | path join logits
            mkdir $dumps
            let args = [--model $case.model --prompt-file $prompt --ctx-start '32' --ctx-max '64' --step-incr '32' --ctx-alloc '256' --gen-tokens '4' --teacher-forced-decode --dump-decode-logits-dir $dumps --threads '8' --csv ($dir | path join timings.csv)] | append $case.flags
            {binary: $variant.binary, sha256: (open --raw $variant.binary | hash sha256), args: $args} | to json | save -f ($dir | path join command.json)
            (^python3 speed-bench/ideas/pressure_run.py ($dir | path join guard) -- $variant.binary ...$args
                out> ($dir | path join stdout.log) err> ($dir | path join run.log))
            if $env.LAST_EXIT_CODE != 0 { error make {msg: $'($case.family) ($variant.name) failed; inspect its run.log.'} }
        }
        for frontier in ['000032' '000064'] {
            let name = $'decode_($frontier).json'
            (^.workspaces/clef-artifacts-20261024/venv/bin/python speed-bench/experiments/steering-20261024/compare_decode.py
                ($output | path join $case.family control logits $name)
                ($output | path join $case.family candidate logits $name)
                ($output | path join $case.family $'comparison-($frontier)'))
            if $env.LAST_EXIT_CODE != 0 { error make {msg: 'Fixed-input logit comparison failed.'} }
        }
    }
}
