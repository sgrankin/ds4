#!/usr/bin/env nu

# The separately pressure-guarded frozen server owns the only model lease.
const ROOT = path self ../../..

def --wrapped checked [command: string, ...args: string] {
    ^$command ...$args
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
}

def main [base_url: string, output: path, --skip-omni] {
    cd $ROOT
    mkdir $output
    let experiment = 'speed-bench/experiments/clef-native-20261024'
    let work = '.workspaces/clef-artifacts-20261024'
    let python = $work | path join venv/bin/python
    $env.PYTHONPATH = '.workspaces/clef-references/decision-index'
    let text_models = if $skip_omni { [clef] } else { [clef clef-omni] }
    for model in $text_models {
        let target = $output | path join $'($model)-text-100'
        (checked nu ($experiment | path join evaluate.nu) $base_url $model $target
            out+err> ($output | path join $'($model)-text-100.log'))
    }
    let suite = $work | path join vision-suite/subset-28
    let vision_models = if $skip_omni { [clef-flash clef] } else { [clef-flash clef clef-omni] }
    for model in $vision_models {
        let target = $output | path join $'($model)-vision-28'
        (checked $python -m decision_index vision run --dir $suite --no-verify --workers 1
            --engine decide --option protocol=decisions --option path=/v1/systemone
            --option $'base_url=($base_url)' --option $'model=($model)' --option retries=1
            --out $target out+err> ($output | path join $'($model)-vision-28.log'))
        (checked $python -m decision_index vision score --dir $suite
            --results ($target | path join results.jsonl) --out $target
            out+err> ($output | path join $'($model)-vision-28-score.log'))
    }
    let data = $work | path join decision-index
    (checked nu ($experiment | path join evaluate.nu) $base_url clef-flash
        ($output | path join clef-flash-text-2000)
        --rows ($data | path join sample-2000.jsonl.gz)
        --suite ($data | path join sample-2000-suite)
        out+err> ($output | path join clef-flash-text-2000.log))
}
