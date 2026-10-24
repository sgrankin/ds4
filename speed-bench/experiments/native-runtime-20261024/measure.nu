#!/usr/bin/env nu
# Serial ABBA screen of the existing CLI and live-append workloads.
# Raw output and JSON stay in the caller's ignored workspace, never this folder.
const ROOT = path self ../../..
const PROMPT = 'Count from one to fifty, separated by commas. Output only the sequence.'
const INPUT = 'speed-bench/experiments/zig-metal-f32-hour26-20261022/prompt.txt'

def main [
    baseline: string # Frozen baseline executable directory.
    candidate: string # Final candidate executable directory.
    out: string # Fresh local output directory, normally under .workspaces/.
    --dry-run # Print the commands without running models or writing files.
] {
    cd $ROOT
    let baseline = $baseline | path expand
    let candidate = $candidate | path expand
    let out = $out | path expand
    # An interactive DS4 preset must not silently change either measurement.
    for name in ($env | columns | where {|name|
        ($name | str starts-with DS4) or ($name | str starts-with PT_)
    }) { hide-env $name }

    let families = [
        {
            name: ds41
            model: 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
            options: [--ssd-streaming --ssd-streaming-cache-experts '32GB']
            settings: {
                DS4_V41_DECODER_REPLAY_ROWS: '0'
                DS4_METAL_V41_RESIDENT_ROUTE_GAP: '0'
                DS4_METAL_V41_RESIDENT_PREFILL_GAP: '0'
                DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS: '0'
            }
        }
        {
            name: flint
            model: '../flint/.local/share/rec/models/fdc85248ad77a0fa9fbf0abda8b1525bea8ee74c11ada2ec95eb66113353d677'
            options: []
            settings: {}
        }
        {
            name: glm
            model: 'gguf/GLM-5.3-Flash-Q2.gguf'
            options: [--ssd-streaming --ssd-streaming-cache-experts '24GB']
            settings: {DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP: '1'}
        }
        {
            name: qwen
            model: 'gguf/Qwen3.8-Flash-Next-Q4.gguf'
            options: []
            settings: {DS4_METAL_NO_RESIDENCY: '1', DS4_METAL_NO_MODEL_WARMUP: '1'}
        }
    ]
    if not $dry_run {
        if ($out | path exists) { error make {msg: $'Use a fresh output directory: ($out)'} }
        mkdir $out
    }
    mut rows = []
    for family in $families {
        for arm in [
            {name: A1, role: baseline, bin: $baseline}
            {name: B1, role: candidate, bin: $candidate}
            {name: B2, role: candidate, bin: $candidate}
            {name: A2, role: baseline, bin: $baseline}
        ] {
            let folder = $out | path join $family.name $arm.name
            let argv = if $family.name == qwen {
                [
                    ($arm.bin | path join ds4-bench)
                    --model
                    $family.model
                    --metal
                    --prompt-file
                    $INPUT
                    --ctx-start
                    '2047'
                    --ctx-max
                    '4095'
                    --step-incr
                    '2048'
                    --ctx-alloc
                    '8192'
                    --prefill-chunk
                    '1024'
                    --gen-tokens
                    '8'
                    --teacher-forced-decode
                    --show-output
                    --csv
                    ($folder | path join timings.csv)
                    --dump-frontier-logits-dir
                    ($folder | path join logits)
                ]
            } else {
                [
                    ($arm.bin | path join ds4)
                    --model
                    $family.model
                    --metal
                    --prompt
                    $PROMPT
                    --ctx
                    '512'
                    --temp
                    '0'
                    --tokens
                    '32'
                    ...$family.options
                ]
            }
            let command = {
                family: $family.name
                arm: $arm.name
                role: $arm.role
                argv: $argv
                environment: $family.settings
            }
            if $dry_run {
                print ($command | to json --raw)
                continue
            }
            mkdir $folder
            if $family.name == qwen { mkdir ($folder | path join logits) }
            $command | to json | save ($folder | path join command.json)
            let started = date now
            let code = with-env $family.settings {
                ^python3 -B speed-bench/ideas/pressure_run.py ($folder | path join guard) -- ...$argv out> ($folder | path join stdout.log) err> ($folder | path join stderr.log)
                $env.LAST_EXIT_CODE
            }
            let wall_s = ((date now) - $started) / 1sec
            if $code != 0 { error make {msg: $'($family.name)/($arm.name) failed ($code); see ($folder)'} }
            let metrics = if $family.name == qwen {
                let samples = open ($folder | path join timings.csv)
                let append = $samples | where ctx_tokens == 4095 | first
                if $append.prefill_tokens != 2048 { error make {msg: 'Qwen benchmark did not measure the 2048-token live append'} }
                {
                    first_prefill_tps: (
                        $samples
                        | first
                        | get prefill_tps
                        | into float
                    )
                    prefill_tps: ($append.prefill_tps | into float)
                    prefill_s: (2048 / ($append.prefill_tps | into float))
                    decode_tps: ($append.gen_tps | into float)
                }
            } else {
                let timing = open --raw ($folder | path join stderr.log)
                | parse --regex 'prefill: (?P<prefill>[0-9.]+) t/s, generation: (?P<decode>[0-9.]+) t/s'
                | last
                {
                    first_prefill_tps: null
                    prefill_tps: ($timing.prefill | into float)
                    prefill_s: null
                    decode_tps: ($timing.decode | into float)
                }
            }
            # The benchmark emits nine-digit FP32 JSON, not decoded text on
            # stdout. Hash the reconstructed FP32 bytes at both real frontiers.
            let frontier_hashes = if $family.name == qwen {
                [2047 4095] | each {|position|
                    let file = $folder | path join logits $'frontier_($position | fill --alignment right --character 0 --width 6).logits.json'
                    let hash = ^python3 -c 'import hashlib,json,struct,sys; values=json.load(open(sys.argv[1]))["logits"]; print(hashlib.sha256(struct.pack("<%df" % len(values), *values)).hexdigest())' $file
                    if $env.LAST_EXIT_CODE != 0 { error make {msg: $'Cannot hash frontier: ($file)'} }
                    {position: $position, sha256: ($hash | str trim)}
                }
            } else { [] }
            let row = $command | merge $metrics | merge {
                wall_s: $wall_s
                stdout_sha256: (if $family.name == qwen { null } else { open --raw ($folder | path join stdout.log) | hash sha256 })
                frontier_sha256: $frontier_hashes
            }
            $rows = $rows | append $row
            $rows | to json | save --force ($out | path join runs.json)
            print ($row | reject argv environment | to json --raw)
        }
    }
    if $dry_run { return }
    mut comparisons = []
    for family in $families {
        let pair = $rows | where family == $family.name
        let a = $pair | where role == baseline
        let b = $pair | where role == candidate
        # Mean inverse throughput compares time for the same token count.
        # Keep the first cold process visible; ABBA is not a startup speed claim.
        let prefill_a = $a | each {|r| 1 / $r.prefill_tps } | math avg
        let prefill_b = $b | each {|r| 1 / $r.prefill_tps } | math avg
        let decode_a = $a | each {|r| 1 / $r.decode_tps } | math avg
        let decode_b = $b | each {|r| 1 / $r.decode_tps } | math avg
        $comparisons = $comparisons | append {
            family: $family.name
            prefill_time_change_pct: (100 * ($prefill_b / $prefill_a - 1))
            decode_time_change_pct: (100 * ($decode_b / $decode_a - 1))
            all_output_hashes_equal: (if $family.name == qwen { null } else { ($pair.stdout_sha256 | uniq | length) == 1 })
            all_frontier_hashes_equal: (if $family.name == qwen { ($pair.frontier_sha256 | uniq | length) == 1 } else { null })
        }
    }
    $comparisons | to json | save ($out | path join comparison.json)
    print $comparisons
}
