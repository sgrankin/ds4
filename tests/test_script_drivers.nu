#!/usr/bin/env nu
# Model-free orchestration contracts. Child processes below are tiny Nu fakes.
use std/assert
const ROOT = path self | path dirname | path dirname

def invoke-driver [script: path, args: list<string> = []] {
    do { ^$nu.current-exe $script ...$args } | complete
}

def main [] {
    cd $ROOT
    for script in [
        tests/dspark_acceptance_fixture.nu
        tests/glm_long_context_smoke.nu
        tests/test_gpu_args_cli.nu
        tests/vision-fixtures/jpeg/generate.nu
        speed-bench/concurrency_sweep.nu
        speed-bench/ideas/build_probe.nu
        run-nvidia-tp-server.nu
    ] {
        assert (nu-check --debug ($ROOT | path join $script)) $'($script) did not parse'
    }
    for script in [speed-bench/ideas/build_probe.nu run-nvidia-tp-server.nu] {
        let result = (invoke-driver $script ['--help'])
        assert equal $result.exit_code 0 $result.stderr
        assert ($result.stdout | str contains 'Usage:') $'($script) help was not shown'
    }
    let parent = $ROOT | path join '.workspaces'
    mkdir $parent
    let work = (mktemp -d -p $parent 'nu-driver-tests.XXXXXX')
    let child = $work | path join 'fake-model.nu'
    r##'#!/usr/bin/env nu
use std/assert
def --wrapped main [...args: string] {
    {args: $args, stats: $env.DS4_DSPARK_STATS?, tail: $env.DS4_DSPARK_SCHEDULER_TAIL_MIN_TOKENS?} | to json --raw | $"($in)\n" | save --append --raw $env.DRIVER_LOG
    match $env.DRIVER_MODE {
        dspark => {
            print 'same'
            print --stderr 'ds4: prefill: 10.0 t/s, generation: 22.5 t/s'
            if '--dspark' in $args {
                let errors = ($env.DRIVER_ERRORS? | default '0')
                let accepted = ($env.DRIVER_ACCEPTED? | default '16')
                print --stderr $'ds4: DSpark stats cycles=4 first_tokens=4 proposed=20 accepted_draft=($accepted) partial=1 errors=($errors) seed_batches=2 direct_full=1 direct_partial=1'
            }
        }
        glm => {
            let at = ($args | enumerate | where item == '--prompt-file' | first | get index)
            let input = ($args | get ($at + 1))
            assert ((open --raw $input) | str ends-with "#include <unistd.h\n")
            print 'ds4: GPU config: fake'
            print (if ($env.DRIVER_BAD? | default '0') == '1' { 'unistdFlush' } else { '> continued' })
            print --stderr 'processing 200 input tokens'
            print --stderr 'ds4: prefill: 12.5 t/s'
        }
        sweep => {
            let at = ($args | enumerate | where item == '--concurrency' | first | get index)
            if (($args | get ($at + 1)) == '2') and (($env.DRIVER_BAD? | default '0') == '1') { exit 7 }
            print 'cell'
        }
        cc => { exit ($env.DRIVER_EXIT | into int) }
    }
}
'## | save --raw $child
    ^chmod +x $child
    let model = $work | path join 'model.gguf'
    let support = $work | path join 'support.gguf'
    'fixture' | save $model
    'fixture' | save $support
    let log = $work | path join 'calls.jsonl'
    let settings = {
        DS4_BIN: $child
        DS4_DSPARK_MODEL: $model
        DS4_DSPARK_SUPPORT: $support
        DRIVER_LOG: $log
        DS4_DSPARK_FIXTURE_TOKENS: '32'
        DS4_DSPARK_FIXTURE_REQUIRE_PARTIAL: '1'
        DS4_DSPARK_FIXTURE_REQUIRE_DIRECT_COMMIT: '1'
        DS4_DSPARK_FIXTURE_REQUIRE_IDENTICAL: '1'
        DS4_DSPARK_FIXTURE_REQUIRE_PROPOSAL_QUALITY: '1'
        DS4_DSPARK_FIXTURE_C_ADD_MIN_ACCEPTED: '8'
        DS4_DSPARK_FIXTURE_CONFIDENCE: '0.75'
        DS4_DSPARK_FIXTURE_EXACT_SAMPLING: '1'
        DRIVER_MODE: 'dspark'
        DRIVER_ERRORS: '0'
        DRIVER_ACCEPTED: '16'
    }
    let result = (with-env $settings { invoke-driver tests/dspark_acceptance_fixture.nu })
    assert equal $result.exit_code 0 $result.stderr
    let calls = open --raw $log | from json --objects
    assert equal ($calls | length) 10
    let draft = $calls | where stats == '1'
    assert equal ($draft | length) 5
    assert ($draft | all {|call| '--mtp-exact-sampling' in $call.args and '--dspark-confidence' in $call.args })
    for change in [
        {DRIVER_ERRORS: '1'}
        {DRIVER_ACCEPTED: '21'}
        {DRIVER_ACCEPTED: '1'}
    ] {
        let result = with-env ($settings | merge $change) { invoke-driver tests/dspark_acceptance_fixture.nu }
        assert ($result.exit_code != 0) 'DSpark failure guard unexpectedly succeeded'
    }
    let glm = {
        DS4_BIN: $child
        DRIVER_LOG: $log
        DRIVER_MODE: 'glm'
        DRIVER_BAD: '0'
        DS4_GLM_LONG_CONTEXT_REPEATS: '2'
        DS4_GLM_EXTRA_ARGS: ''
        DS4_GLM_BACKEND: 'metal'
    }
    let result = (with-env $glm { invoke-driver tests/glm_long_context_smoke.nu [$model '--label' 'space value'] })
    assert equal $result.exit_code 0 $result.stderr
    let result = with-env ($glm | merge {DRIVER_BAD: '1'}) { invoke-driver tests/glm_long_context_smoke.nu [$model] }
    assert ($result.exit_code != 0) 'GLM corruption guard unexpectedly succeeded'
    for bad in ['0' '1'] {
        rm -f $log
        let result = (with-env {DRIVER_LOG: $log, DRIVER_MODE: 'sweep', DRIVER_BAD: $bad, CONCURRENCY: '1 2', CONTEXTS: '0 4096'} {
            invoke-driver speed-bench/concurrency_sweep.nu [($work | path join 'cells.csv') '--bench' $child '--label' 'space value']
        })
        assert equal $result.exit_code ($bad | into int) $result.stderr
        let calls = open --raw $log | from json --objects
        assert equal ($calls | length) 4 'sweep did not run every cell'
        assert ($calls | all {|call| ($call.args | last 2) == ['--label' 'space value'] }) 'sweep lost argument boundaries'
    }
    let bin = $work | path join bin
    mkdir $bin
    let compiler = $bin | path join cc
    cp $child $compiler
    let runtime = $work | path join 'C checkout'
    let output = $work | path join 'probe output'
    let source = $work | path join 'probe source.c'
    let core = $work | path join 'custom core.o'
    for status in ['0' '7'] {
        rm -f $log
        let result = (with-env {PATH: ($env.PATH | prepend $bin), DRIVER_LOG: $log, DRIVER_MODE: cc, DRIVER_EXIT: $status} {
            invoke-driver speed-bench/ideas/build_probe.nu [$runtime $output '-DA=1 -DB=2' $source $core]
        })
        assert equal $result.exit_code ($status | into int) $result.stderr
        let args = open --raw $log | from json | get args
        assert ('-DA=1' in $args and '-DB=2' in $args) 'C control policy flags were not split'
        assert ($source in $args and $core in $args) 'C control input paths lost their argument boundaries'
        assert equal ($args | last 2) ['-o' $output] 'C control output path lost its argument boundary'
        assert (($runtime | path join ds4_metal.o) in $args) 'C control object path was not preserved'
    }
    print $'PASS: Nu driver parsing/help, DSpark/GLM guards, sweep and C-control argv/exit contracts; artifacts ($work)'
}
