#!/usr/bin/env nu
# Explicit qualification only. Each model process retains the admission lock
# and the existing pressure guard; outputs stay under the requested workspace.
const ROOT = path self | path dirname | path dirname

def main [
    --model: path
    --output: path = ".workspaces/qwen-mtp-limits"
    --binary: path = "zig-out/bin/ds4"
    --ssd-streaming
] {
    if $model == null { error make {msg: "--model is required"} }
    let output = $output | path expand
    mkdir $output
    let binary = $binary | path expand
    let model = $model | path expand
    let storage = if $ssd_streaming { [--ssd-streaming --ssd-streaming-cache-experts 1024] } else { [] }
    let base = [-m $model --ctx 256 --temp 0 --nothink -p "Count from one to ten." -n 24]
    for case in [{name: one-row, chunk: 1, unfused: false} {name: two-rows-depth-three, chunk: 2, unfused: false} {name: unfused, chunk: 128, unfused: true}] {
        mut outputs = []
        for mode in [plain mtp] {
            let stem = $"($case.name)-($mode)"
            let extra = if $mode == mtp { [--mtp] } else { [] }
            let result = do {
                hide-env -i DS4_QWEN4_SPEC_FORCE_ACCEPT DS4_QWEN4_NO_FUSE
                $env.DS4_QWEN4_MTP_DEPTH = "3"
                if $case.unfused { $env.DS4_QWEN4_NO_FUSE = "1" }
                ^python3 ($ROOT | path join speed-bench ideas pressure_run.py) ($output | path join $"($stem)-guard") -- $binary ...$base ...$storage --prefill-chunk ($case.chunk | into string) ...$extra | complete
            }
            $result.stdout | save -f ($output | path join $"($stem).stdout")
            $result.stderr | save -f ($output | path join $"($stem).stderr")
            if $result.exit_code != 0 or ($result.stderr | str downcase | str contains failed) {
                error make {msg: $"($stem) failed; see ($output)"}
            }
            $outputs = ($outputs | append $result.stdout)
        }
        if $outputs.0 != $outputs.1 { error make {msg: $"($case.name): MTP changed greedy continuation; see ($output)"} }
        print $"PASS ($case.name)"
    }
}
