#!/usr/bin/env nu
# Behavioral sweep through the same public options used for ordinary serving.
const ROOT = path self ../..

def main [
    --ds4: string = './ds4'
    --model: string = 'ds4flash.gguf'
    --direction: string # Required flat residual f32 bank.
    --prompts: string # Required text file, one prompt per nonempty line.
    --mode: string = project # project or add.
    --scales: string = '-1,0,0.5,1,2'
    --tokens: int = 160
    --ctx: int = 4096
    --attn-scale: float = 0.0
    --nothink # Other models only; omit for Flint's automatic template.
    --dry-run
] {
    if $direction == null or $prompts == null { error make {msg: '--direction and --prompts are required'} }
    let binary = $ds4 | path expand
    let model = $model | path expand
    let direction = $direction | path expand
    if $mode != project and $mode != add { error make {msg: '--mode must be project or add'} }
    let prompts = open --raw $prompts
        | lines
        | str trim
        | where {|line| $line != '' and not ($line | str starts-with '#') }
    if ($prompts | is-empty) { error make {msg: 'Prompt file is empty'} }
    let strengths = $scales | split row ',' | into float
    cd $ROOT
    for prompt in $prompts {
        print $'PROMPT: ($prompt)'
        for strength in $strengths {
            let argv = [
                $binary --model $model --ctx ($ctx | into string)
                --tokens ($tokens | into string) --temp '0'
                --dir-steering-file $direction --dir-steering-mode $mode
                --dir-steering-ffn ($strength | into string)
                --dir-steering-attn ($attn_scale | into string)
                --prompt $prompt
                ...(if $nothink { [--nothink] } else { [] })
            ]
            print $'mode=($mode) FFN=($strength) attention=($attn_scale)'
            if $dry_run { print ($argv | to json --raw); continue }
            ^$binary ...($argv | skip 1)
            if $env.LAST_EXIT_CODE != 0 { error make {msg: $'Inference failed at strength ($strength)'} }
        }
    }
}
