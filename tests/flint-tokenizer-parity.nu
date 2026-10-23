#!/usr/bin/env nu
# Compare production native IDs and raw decoded bytes with a CPU HF reference.
# First generate the local reference with tests/flint_reference.py using Flint's
# installed Python. No inference/model allocation is involved in either step.
const ROOT = path self ..

def main [
    model_dir: string
    reference: string
    --fixture: string = '.workspaces/flint-build/bin/test-flint-tokenizer'
    --artifacts: string = '.workspaces/flint-tests'
] {
    cd $ROOT
    mkdir $artifacts
    let data = open $reference
    let cases = ($data.tokenizer | append ($data.prompts | each {|case| {text: $case.expected, ids: $case.expected_ids} }))
    let input = ($artifacts | path join 'input.txt')
    for case in ($cases | enumerate) {
        $case.item.text | save --raw --force $input
        let output = (^$fixture ($model_dir | path join 'tokenizer.json') $input | complete)
        if $output.exit_code != 0 {
            error make {msg: $'native tokenizer failed on case ($case.index): ($output.stderr)'}
        }
        let lines = ($output.stdout | lines)
        let ids = ($lines | first | from json)
        if $ids != $case.item.ids {
            error make {msg: $'token IDs differ on case ($case.index): native ($ids | to json --raw); HF ($case.item.ids | to json --raw)'}
        }
        let restored = ($lines | skip 1 | str join | decode hex | decode utf-8)
        if $restored != $case.item.text {
            error make {msg: $'decoded raw bytes differ on case ($case.index)'}
        }
    }
    print $'Flint native tokenizer: PASS — ($cases | length) HF cases, exact IDs and raw byte round trips'
}
