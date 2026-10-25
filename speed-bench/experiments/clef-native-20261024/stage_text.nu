#!/usr/bin/env nu

# Preserve the upstream sampler's complete case groups and task scorers.
const ROOT = path self ../../..

def --wrapped checked [command: string, ...args: string] {
    ^$command ...$args
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
}

def write_rows [rows: list, output: path] {
    let text = $rows | each {|row| $row | to json --raw } | str join "\n"
    $"($text)\n" | ^gzip -n | save --raw --force $output
}

def main [
    --count: int = 2000
    --seed: int = 20260919
    --work: path = '.workspaces/clef-artifacts-20261024/decision-index'
    --rebuild
] {
    cd $ROOT
    let kit = $ROOT | path join .workspaces/clef-references/decision-index
    let python = $ROOT | path join .workspaces/clef-artifacts-20261024/venv/bin/python
    $env.PYTHONPATH = $kit
    let partial = $work | path join partial-suite
    if $rebuild {
        (checked $python -m decision_index suite rebuild --edition 0.2.1
            --work $work --only 3 4 5 62)
        mkdir $partial
        let artifacts = $work | path join artifacts/benchmark-suite
        cp --force ($artifacts | path join release-v1-rebuilt/selected-rows.jsonl.gz) $partial
        cp --force ($artifacts | path join release-v2-rebuilt/added-rows.jsonl.gz) $partial
        cp --force ($kit | path join hub/excluded-questions.json) $partial
    }
    let sample = $work | path join $'sample-($count).jsonl.gz'
    (checked $python -m decision_index suite sample --edition 0.2.1
        --dir $partial --n ($count | into string) --seed ($seed | into string) --out $sample)
    let rows = ^gzip -dc $sample | from json --objects
    let suite = $work | path join $'sample-($count)-suite'
    mkdir $suite
    write_rows ($rows | where {|row| $row._evaluation.catalog_id != 62 }) ($suite | path join selected-rows.jsonl.gz)
    write_rows ($rows | where {|row| $row._evaluation.catalog_id == 62 }) ($suite | path join added-rows.jsonl.gz)
    cp --force ($partial | path join excluded-questions.json) $suite
    let counts = $rows | group-by _evaluation.catalog_id | transpose catalog_id rows | each {|item| {catalog_id: $item.catalog_id, requests: ($item.rows | length)} }
    {
        edition: '0.2.1', sample_only: true, seed: $seed,
        requested_rows: $count, actual_rows: ($rows | length), counts: $counts,
        note: 'Four sampled tasks; no full Decision Index score.'
    } | to json | save --force ($suite | path join manifest.json)
    print {rows: $sample, suite: $suite, counts: $counts}
}
