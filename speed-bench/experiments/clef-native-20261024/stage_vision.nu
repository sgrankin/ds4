#!/usr/bin/env nu

const ROOT = path self ../../..

def main [--count: int = 28, --work: path = '.workspaces/clef-artifacts-20261024/vision-suite', --rebuild] {
    cd $ROOT
    $env.PYTHONPATH = $ROOT | path join .workspaces/clef-references/decision-index
    $env.HF_HOME = $work | path join hf-cache
    let options = if $rebuild { ['--rebuild'] } else { [] }
    (^.workspaces/clef-artifacts-20261024/venv/bin/python
        speed-bench/experiments/clef-native-20261024/stage_vision.py
        $work --count $count ...$options)
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
}
