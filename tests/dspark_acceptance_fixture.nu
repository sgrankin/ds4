#!/usr/bin/env nu
# Fixed DSpark vs ordinary-generation fixture. Run explicitly under pressure_run.py.
use std/assert
const ROOT = path self | path dirname | path dirname

def last-value [text: string, pattern: string, fallback: string = 'n/a'] {
    let found = $text | parse -r $pattern
    if ($found | is-empty) {
        $fallback
    } else {
        $found | last | get value
    }
}

def logged [binary: string, args: list<string>, stem: path] {
    let result = do { ^$binary ...$args } | complete
    $result.stdout | save --raw $'($stem).out'
    $result.stderr | save --raw $'($stem).err'
    if $result.exit_code != 0 {
        print --stderr $result.stderr
        exit $result.exit_code
    }
    $result
}

def probe [command: string, args: list<string>] {
    let result = do { ^$command ...$args } | complete
    if $result.exit_code == 0 {
        $result.stdout | str trim
    } else {
        'unknown'
    }
}

def file-bytes [file: path] {
    ls ($file | path expand) | get size | first | into int
}

def main [] {
    let binary = $env.DS4_BIN? | default './ds4'
    let model = $env.DS4_DSPARK_MODEL? | default ($env.DS4_TEST_MODEL? | default './ds4flash.gguf')
    let support = $env.DS4_DSPARK_SUPPORT? | default 'gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf'
    let tokens = $env.DS4_DSPARK_FIXTURE_TOKENS? | default '32' | into int
    let partial_required = ($env.DS4_DSPARK_FIXTURE_REQUIRE_PARTIAL? | default '0') != '0'
    let direct_required = ($env.DS4_DSPARK_FIXTURE_REQUIRE_DIRECT_COMMIT? | default '1') != '0'
    let identical_required = ($env.DS4_DSPARK_FIXTURE_REQUIRE_IDENTICAL? | default '0') != '0'
    let quality = $env.DS4_DSPARK_FIXTURE_REQUIRE_PROPOSAL_QUALITY? | default 'auto'
    let minimum = $env.DS4_DSPARK_FIXTURE_C_ADD_MIN_ACCEPTED? | default '8' | into int
    assert ($minimum >= 0) 'DS4_DSPARK_FIXTURE_C_ADD_MIN_ACCEPTED must be nonnegative'
    let confidence = $env.DS4_DSPARK_FIXTURE_CONFIDENCE? | default ''
    let exact = ($env.DS4_DSPARK_FIXTURE_EXACT_SAMPLING? | default '0') != '0'
    let temperature = $env.DS4_DSPARK_FIXTURE_TEMPERATURE? | default '0'
    let top_p = $env.DS4_DSPARK_FIXTURE_TOP_P? | default '0.95'
    let min_p = $env.DS4_DSPARK_FIXTURE_MIN_P? | default '0.05'
    let seed = $env.DS4_DSPARK_FIXTURE_SEED? | default '12345'
    let guard = match $quality {
        '0' | 'false' | 'no' | 'off' => false
        '1' | 'true' | 'yes' | 'on' => true
        'auto' | '' => (not $partial_required and $confidence == '' and $tokens >= 32)
        _ => { error make {msg: $'invalid DS4_DSPARK_FIXTURE_REQUIRE_PROPOSAL_QUALITY=($quality)'} }
    }
    if $partial_required and $env.DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS? == null {
        $env.DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS = '0'
    }
    for entry in [
        {kind: executable, file: $binary}
        {kind: model, file: $model}
        {kind: 'DSpark support model', file: $support}
    ] {
        if not ($entry.file | path exists) {
            print --stderr $'dspark-fixture: skipped, missing ($entry.kind) ($entry.file)'
            return
        }
    }
    let parent = $ROOT | path join '.workspaces'
    mkdir $parent
    let output = (mktemp -d -p $parent 'dspark-fixture.XXXXXX')
    let git = do { ^git rev-parse --short HEAD } | complete
    let commit = if $git.exit_code == 0 {
        let dirty = do { ^git diff --quiet -- . } | complete
        ($git.stdout | str trim) + (if $dirty.exit_code == 0 { '' } else { '+dirty' })
    } else { 'unknown' }
    let flags = [
        '--tokens'
        ($tokens | into string)
        '--temp'
        $temperature
        '--top-p'
        $top_p
        '--min-p'
        $min_p
        '--seed'
        $seed
        '--nothink'
    ]
    let draft_flags = ([
        (if $exact { ['--mtp-exact-sampling'] } else { [] })
        (if $confidence != '' { ['--dspark-confidence' $confidence] } else { [] })
    ] | flatten)
    let hardware = {
        os: (probe 'uname' ['-sm'])
        model: (probe 'sysctl' ['-n' 'hw.model'])
        cpu: (probe 'sysctl' ['-n' 'machdep.cpu.brand_string'])
    }
    let settings = {
        tokens: $tokens
        ctx: default
        temperature: $temperature
        top_p: $top_p
        min_p: $min_p
        seed: $seed
        exact_sampling: $exact
        confidence: $confidence
        proposal_quality_guard: $quality
        proposal_quality_active: $guard
        c_add_min_accepted: $minimum
        require_direct: $direct_required
        require_identical: $identical_required
        scheduler: ($env.DS4_DSPARK_SCHEDULER? | default 'default')
        no_draft_skip: ($env.DS4_DSPARK_SCHEDULER_NO_DRAFT_SKIP? | default 'default')
        short_accept_no_draft_skip: ($env.DS4_DSPARK_SCHEDULER_SHORT_ACCEPT_NO_DRAFT_SKIP? | default 'default')
        cold_low_confidence_skip: ($env.DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_SKIP? | default 'default')
        cold_low_confidence_milli: ($env.DS4_DSPARK_SCHEDULER_COLD_LOW_CONFIDENCE_MILLI? | default 'default')
        tail_min_tokens: ($env.DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS? | default 'default')
    }
    {
        commit: $commit
        hardware: $hardware
        model: $model
        model_bytes: (file-bytes $model)
        support: $support
        support_bytes: (file-bytes $support)
        settings: $settings
    } | to json | save ($output | path join 'execution.json')
    print $'# commit=($commit)'
    print $'# hardware_os=($hardware.os) hardware_model=($hardware.model) hardware_cpu=($hardware.cpu)'
    print $'# model=($model) model_bytes=(file-bytes $model) support=($support) support_bytes=(file-bytes $support)'
    print $'# settings=($settings | to json --raw)'
    print $'# artifacts=($output)'
    print (['id' 'output_match' 'baseline_tps' 'dspark_tps' 'dspark_stats'] | str join (char tab))
    let cases = [
        {id: hello, prompt: Hello}
        {id: redis, prompt: 'Explain Redis in one sentence.'}
        {id: math, prompt: 'What is 17 times 23?'}
        {id: python_reverse, prompt: 'Write a Python function that reverses a string.'}
        {id: c_add, prompt: 'Complete this C function: int add(int a, int b) {'}
    ]
    mut partial_cases = 0
    mut direct_partial_cases = 0
    mut direct_commits = 0
    for case in $cases {
        let stem = $output | path join $case.id
        let baseline = logged $binary (['-m' $model] | append $flags | append ['-p' $case.prompt]) $'($stem).baseline'
        let draft = (with-env {DS4_DSPARK_STATS: '1'} {
            logged $binary (['--dspark'] | append $draft_flags | append ['-m' $model '--mtp-model' $support] | append $flags | append ['-p' $case.prompt]) $'($stem).dspark'
        })
        let same = (do { ^cmp -s $'($stem).baseline.out' $'($stem).dspark.out' } | complete | get exit_code) == 0
        if $identical_required { assert $same $'dspark-fixture: output mismatch for ($case.id); see ($stem).*' }
        let lines = $draft.stderr | lines | where {|line| $line | str contains 'DSpark stats' }
        assert (not ($lines | is-empty)) $'dspark-fixture: missing DSpark stats for ($case.id)'
        let line = $lines | last | str replace 'ds4: DSpark stats ' ''
        let stats = (
            $line
            | parse -r '(?<key>\w+)=(?<value>\d+)'
            | reduce --fold {} {|pair, values| $values | upsert $pair.key ($pair.value | into int) }
        )
        let partial = $stats.partial? | default 0
        let errors = $stats.errors? | default 0
        let accepted = $stats.accepted_draft? | default 0
        let proposed = $stats.proposed? | default 0
        let first_tokens = $stats.first_tokens? | default 0
        let seed_batches = $stats.seed_batches? | default 0
        let full = $stats.direct_full? | default 0
        let direct_partial = $stats.direct_partial? | default 0
        assert ($errors == 0) $'dspark-fixture: verifier errors for ($case.id): ($line)'
        assert ($accepted <= $proposed and $seed_batches <= $first_tokens) $'dspark-fixture: inconsistent seed/draft accounting for ($case.id): ($line)'
        if $guard and $case.id == 'c_add' {
            assert ($accepted >= $minimum) $'dspark-fixture: c_add accepted_draft ($accepted) below required ($minimum): ($line)'
        }
        if $partial_required and $partial > 0 { $partial_cases += 1 }
        if $direct_partial > 0 { $direct_partial_cases += 1 }
        $direct_commits += ($full + $direct_partial)
        let base_tps = (last-value $baseline.stderr 'generation: (?<value>[0-9.]+) t/s')
        let draft_tps = (last-value $draft.stderr 'generation: (?<value>[0-9.]+) t/s')
        print (
            [
                $case.id
                (if $same { 'output_match=1' } else { 'output_match=0' })
                $'baseline_tps=($base_tps)'
                $'dspark_tps=($draft_tps)'
                $line
            ]
            | str join (char tab)
        )
    }
    if $partial_required {
        assert ($partial_cases > 0) 'dspark-fixture: expected at least one partial accept case'
        assert ($direct_partial_cases > 0) 'dspark-fixture: expected at least one direct partial commit'
    }
    if $direct_required { assert ($direct_commits > 0) 'dspark-fixture: expected at least one direct verifier-state commit' }
}
