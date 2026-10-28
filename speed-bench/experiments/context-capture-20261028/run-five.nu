#!/usr/bin/env nu
# Compare checkpoint implementations beneath the existing pressure_run.py.
# Reuse the original HTTP workflows and analyzer; only per-arm settings differ.
use std/assert
const SCRIPT = path self

def run-workflows [base: string, model: string, output: path, pair: int, arm: string, cases: list<string>, rows: int, edits: int, dry_run: bool] {
    for case in $cases {
        let directory = $output | path join $case $'pair-($pair)' $arm
        mkdir ($directory | path dirname)
        let key = $'context-($case)-pair-($pair)-($arm)'
        let args = [
            speed-bench/experiments/context-boundaries-20261028/run.nu
            $base $model $directory
            --case $case --rows $rows --edits $edits --key $key
            --context-label $'boundary-qualification-($case)'
        ]
        let result = if $dry_run {
            nu ...$args --fixtures-only | complete
        } else {
            nu ...$args | complete
        }
        $result.stdout | save ($directory | path dirname | path join $'($arm).client.log')
        $result.stderr | save ($directory | path dirname | path join $'($arm).client.stderr')
        {exit_code: $result.exit_code} | to json | save ($directory | path dirname | path join $'($arm).client-exit.json')
        if $result.exit_code != 0 { error make {msg: $'Workflow failed: pair ($pair), ($arm), ($case), exit ($result.exit_code)'} }
    }
}

def run-arm [binary: path, shaders: path, overrides: record, model: path, api_model: string, output: path, pair: int, arm: string, cases: list<string>, rows: int, edits: int, spacing: int, limit: int, chunk: int, port: int, dry_run: bool] {
    let directory = $output | path join processes $'pair-($pair)' $arm
    mkdir ($directory | path join cache) ($directory | path join responses)
    let base = $'http://127.0.0.1:($port)'
    mut settings = {
        DS4_STATE_CACHE_DIR: ($directory | path join cache)
        DS4_STATE_CACHE_HOT_MB: '512'
        DS4_STATE_CACHE_SSD_MB: '4096'
        DS4_SERVER_RESPONSE_STORE_DIR: ($directory | path join responses)
        DS4_LOCK_FILE: ($directory | path join server.lock)
        DS4_SERVER_CONTEXT_SUFFIX_REUSE: '1'
        DS4_SERVER_CONTEXT_SUFFIX_CANONICAL: '1'
        DS4_SERVER_CONTEXT_SUFFIX_PAD: '1'
        DS4_SERVER_CONTEXT_SUFFIX_RIGHT: '1'
        DS4_SERVER_CONTEXT_SUFFIX_FRESH_TAIL: '64'
        DS4_SERVER_CONTEXT_SUFFIX_STAGING_MB: '512'
        DS4_SERVER_CONTEXT_CAPTURE_PROFILE: '1'
        DS4_SERVER_CONTEXT_BOUNDARIES: '1'
        DS4_SERVER_CONTEXT_SUFFIX_ONGOING_USER: '1'
        DS4_SERVER_CONTEXT_BOUNDARY_LIMIT: ($limit | into string)
        DS4_SERVER_CONTEXT_BOUNDARY_MIN_TOKENS: ($spacing | into string)
        DS4_V41_CHECKPOINT_DRAFT_FEATURES: '0'
        DS4_V41_INDEX_FP4: '1'
        DS4_V41_INDEX_SIMD: '1'
        DS4_V41_CANDIDATE_INDEX: '0'
        DS4_V41_COMPRESSED_KV_F16: '0'
        DS4_V41_COMPRESSED_KV_FP4: '1'
        DS4_METAL_V41_RESIDENT_ROUTE_GAP: '0'
        DS4_METAL_V41_RESIDENT_PREFILL_GAP: '0'
    }
    $settings = ($settings | merge $overrides)
    for source in (glob ($shaders | path join '*.metal')) {
        let key = $'DS4_METAL_($source | path parse | get stem | str uppercase)_SOURCE'
        $settings = ($settings | upsert $key $source)
    }
    let args = [
        '--model' $model
        '--host' '127.0.0.1' '--port' ($port | into string)
        '--ctx' '32768' '--tokens' '512' '--prefill-chunk' ($chunk | into string)
        '--batched-session' '2'
        '--ssd-streaming' '--ssd-streaming-cache-experts' '32GB'
        '--trace' ($directory | path join trace.log)
    ]
    {binary: $binary, args: $args, settings: $settings,
        note: 'Inherited DS4_* cleared before the server starts. Arm overrides are recorded verbatim and may disable semantic checkpoints. No MTP head is loaded.'}
    | to json | save ($directory | path join command.json)
    if $dry_run {
        run-workflows $base $api_model $output $pair $arm $cases $rows $edits true
        return
    }
    let occupied = try {
        http get --max-time 1sec $'($base)/health' | ignore
        true
    } catch { false }
    assert (not $occupied) 'The qualification port is already occupied.'
    let frozen_settings = $settings
    let started = date now
    let task = job spawn --description $'Context boundaries ($arm)' {
        with-env $frozen_settings { ^$binary ...$args out> ($directory | path join server.stdout) err> ($directory | path join server.stderr) }
    }
    let result = try {
        mut ready = false
        for attempt in 1..240 {
            $ready = (try {
                http get --max-time 1sec $'($base)/health' | ignore
                true
            } catch { false })
            if $ready { break }
            sleep 500ms
        }
        assert $ready 'The owned server did not become ready.'
        {startup_seconds: ((((date now) - $started) | into int) / 1e9), cases: $cases}
        | to json | save ($directory | path join startup.json)
        run-workflows $base $api_model $output $pair $arm $cases $rows $edits false
        {ok: true}
    } catch {|failure| {ok: false, message: $failure.msg} }
    # Outside all timed HTTP requests, retain terminal capture/store logs.
    sleep 300ms
    try { job kill $task } catch { }
    mut closed = false
    for attempt in 1..20 {
        $closed = (try {
            http get --max-time 1sec $'($base)/health' | ignore
            false
        } catch { true })
        if $closed { break }
        sleep 100ms
    }
    assert $closed 'Owned server did not stop; abort before loading another model.'
    $result | to json | save ($directory | path join result.json)
    if not $result.ok { error make {msg: $result.message} }
}

def main [binary: path, output: path, --control-env: path, --candidate-env: path, --control-binary: path, --model: path = 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf', --api-model: string = 'deepseek-v4.1-flash', --port: int = 18351, --rows: int = 24, --edits: int = 2, --pairs: int = 5, --spacing: int = 512, --limit: int = 4, --chunk: int = 8192, --case: string = '', --dry-run] {
    let binary = $binary | path expand
    let control_binary = if $control_binary == null { $binary } else { $control_binary | path expand }
    let output = $output | path expand
    let shaders = $binary | path dirname | path dirname | path join shaders
    let control_shaders = $control_binary | path dirname | path dirname | path join shaders
    assert ($binary | path exists)
    assert ($shaders | path exists)
    assert ($control_binary | path exists)
    assert ($control_shaders | path exists)
    let runtime_revision = open --raw ($shaders | path dirname | path join source-revision.txt) | str trim
    let control_revision = open --raw ($control_shaders | path dirname | path join source-revision.txt) | str trim
    assert ($control_env != null and $candidate_env != null) 'Supply two JSON records of DS4 environment overrides.'
    let control_settings = open $control_env
    let candidate_settings = open $candidate_env
    assert (($control_settings | describe | str starts-with 'record') and ($candidate_settings | describe | str starts-with 'record'))
    assert (not ($output | path exists)) 'Use a fresh output directory.'
    assert ($pairs > 0 and $rows > 0 and $edits >= 0 and $spacing >= 0 and $chunk > 0)
    assert ($limit > 0 and $limit <= 16)
    let cases = if $case == '' { ['ordinary' 'favorable'] } else { [$case] }
    # These settings are the complete DS4 environment for every server child.
    let inherited = $env | columns | where {|name| $name | str starts-with 'DS4_' }
    hide-env --ignore-errors ...$inherited
    cd ($SCRIPT | path dirname | path join ../../.. | path expand)
    let model = $model | path expand
    assert ($model | path exists)
    mkdir $output
    let forward = ['control' 'candidate']
    let order = 1..$pairs | each {|pair| if $pair mod 2 == 1 { $forward } else { $forward | reverse } }
    {pairs: $pairs, order: $order, cases: $cases, rows: $rows, edits: $edits,
        binary: $binary, runtime_revision: $runtime_revision, shaders: $shaders,
        control_binary: $control_binary, control_revision: $control_revision, control_shaders: $control_shaders,
        model: $model, reasoning: 'high', temperature: 0, min_p: 0.05,
        main_fp4: true, index_fp4: true, mtp: false, hot_mb: 512, boundary_limit: $limit, spacing: $spacing, prefill_chunk: $chunk,
        arms: $forward, control_overrides: $control_settings, candidate_overrides: $candidate_settings,
        batched_sessions: 2, ordinary_coordinator_quantum: 2048,
        changed_bundle: 'See the two override records and per-process command.json; all other driver settings are fixed.',
        context_policy: 'Fixed distinct instructions per case; unique cache key per pair/arm/case. Each case starts a cold context; the second case shares warm model weights.',
        timing: $'Startup retained separately. Initial population and all ($edits) edits retained per case; archive retrieval excluded.',
        dry_run: $dry_run, comparative_perf: ($pairs == 5 and not $dry_run)}
    | to json | save ($output | path join manifest.json)
    mut execution_order = []
    for pair in 1..$pairs {
        let arms = $order | get ($pair - 1)
        for arm in $arms {
            $execution_order = ($execution_order | append {event: 'started', pair: $pair, arm: $arm, at: (date now)})
            $execution_order | to json | save --force ($output | path join execution-order.json)
            let arm_binary = if $arm == control { $control_binary } else { $binary }
            let arm_shaders = if $arm == control { $control_shaders } else { $shaders }
            let overrides = if $arm == control { $control_settings } else { $candidate_settings }
            run-arm $arm_binary $arm_shaders $overrides $model $api_model $output $pair $arm $cases $rows $edits $spacing $limit $chunk $port $dry_run
            $execution_order = ($execution_order | append {event: 'finished', pair: $pair, arm: $arm, at: (date now)})
            $execution_order | to json | save --force ($output | path join execution-order.json)
            print $'Completed pair ($pair) ($arm)'
        }
    }
    if $pairs == 5 and not $dry_run {
        if $edits == 0 {
            let result = python3 speed-bench/experiments/context-capture-20261028/no-edits-report.py $output | complete
            print $result.stdout
            if $result.exit_code != 0 { error make {msg: $'No-edit analysis failed: ($result.stderr)'} }
            return
        }
        for case in $cases {
            let arguments = [speed-bench/experiments/context-boundaries-20261028/analyze.py ($output | path join $case) --processes ($output | path join processes)]
            let result = python3 ...$arguments | complete
            print $result.stdout
            if $result.exit_code != 0 { error make {msg: $'Analysis failed: ($case): ($result.stderr)'} }
        }
    }
}
