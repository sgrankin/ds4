#!/usr/bin/env nu
# CPU-only host/shader layout and compact-dispatch oracle.
const ROOT = path self | path dirname | path dirname

def main [--output: path = ".workspaces/qwen-moe-mm-compact", --sanitize] {
    let output = $output | path expand
    mkdir $output
    let source = $output | path join test.cpp
    let binary = $output | path join test
    let extracted = (^python3 ($ROOT | path join tests test_qwen4_moe_mm_compact.py) --output $source | complete)
    if $extracted.exit_code != 0 { error make {msg: $extracted.stderr} }
    let flags = if $sanitize { ['-fsanitize=address,undefined' '-fno-omit-frame-pointer'] } else { [] }
    let compiled = (^clang++ -std=c++17 -O2 -Wall -Wextra ...$flags $source -o $binary | complete)
    if $compiled.exit_code != 0 { error make {msg: $compiled.stderr} }
    let result = (^$binary | complete)
    if $result.exit_code != 0 or $result.stderr != "" { error make {msg: $result.stderr} }
    let report = ($result.stdout | from json | merge ($extracted.stdout | from json) | insert sanitizers $sanitize)
    $report | to json | save -f ($output | path join result.json)
    $report
}
