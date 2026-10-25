#!/usr/bin/env nu

# Use the upstream runner/scorers; keep benchmark rows and answers local.
const ROOT = path self ../../..

def --wrapped checked [command: string, ...args: string] {
    ^$command ...$args
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
}

def main [
    base_url: string
    model: string
    output: path
    --rows: path
    --suite: path
] {
    let work = $ROOT | path join .workspaces/clef-artifacts-20261024
    let kit = $ROOT | path join .workspaces/clef-references/decision-index
    let python = $work | path join venv/bin/python
    let rows = $rows | default ($work | path join decision-index/smoke-100.jsonl.gz)
    let suite = $suite | default ($work | path join decision-index/smoke-suite)
    $env.PYTHONPATH = $kit
    mkdir $output
    (checked $python -m decision_index run --edition 0.2.1 --engine http
        --option $"base_url=($base_url)" --option $"model=($model)"
        --rows $rows --out $output)
    (checked $python ($ROOT | path join speed-bench/experiments/clef-native-20261024/score_subset.py)
        $suite ($output | path join results.jsonl) ($output | path join benchmark-summary.json))
    print 'Read benchmark-summary.json for these sampled tasks; this is not a full Decision Index score.'
}
