#!/usr/bin/env nu
# Metadata/tokenizer-only comparison; no weight upload or GPU inference.
use std/assert

def main [control: path, candidate: path, ...models: path, --output: path = '.workspaces/model-token-comparison'] {
    mkdir $output
    for model in ($models | enumerate) {
        let ordinary = [[--nothink] [--think] [--think-max]]
        let modes = if ($model.item | str contains 'V4.1') { $ordinary | append [[--think-level '25']] } else { $ordinary }
        for mode in ($modes | enumerate) {
            let old = ^$control --model $model.item --ctx 512 --dump-tokens -p 'Return JSON {"ok":true}.' ...$mode.item | complete
            let new = ^$candidate --model $model.item --ctx 512 --dump-tokens -p 'Return JSON {"ok":true}.' ...$mode.item | complete
            let stem = $'model-($model.index)-mode-($mode.index)'
            {model: $model.item, mode: $mode.item, control: $old, candidate: $new} | to json | save -f ($output | path join $'($stem).json')
            assert equal $old.exit_code 0
            assert equal $new.exit_code 0
            assert equal $new.stdout $old.stdout
        }
    }
    print 'Metadata-only model prompt token streams match the frozen control in every selected thinking mode.'
}
