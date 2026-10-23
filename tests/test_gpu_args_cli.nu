#!/usr/bin/env nu
# Model-free frontend option smoke tests. /dev/null forces loading to fail.
# These exercise parsing, not CUDA or tensor-parallel hardware.
use std/assert
const ROOT = path self | path dirname | path dirname

def invoke [binary: path, args: list<string>] {
    let result = do { ^$binary ...$args } | complete
    $result | insert output ($result.stdout + $result.stderr)
}

def text-check [
    label: string
    result: record
    required: list<string>
    forbidden: list<string> = []
] {
    for text in $required {
        assert ($result.output | str contains $text) $'($label): missing ($text); output: ($result.output)'
    }
    for text in $forbidden {
        assert (not ($result.output | str contains $text)) $'($label): unexpected ($text); output: ($result.output)'
    }
    print $'ok ($label)'
}

def rejection [
    label: string
    result: record
    required: list<string> = []
    forbidden: list<string> = ['unknown option']
    code?: int
] {
    assert ($result.exit_code != 0) $'($label): unexpectedly succeeded'
    if $code != null { assert equal $result.exit_code $code $'($label): wrong exit code' }
    text-check $label $result $required $forbidden
}

def reaches-model [label: string, result: record] {
    rejection $label $result [] ['unknown option' 'requires --layers']
    assert ($result.output =~ 'model file is too small|another ds4 process is already running') $'($label): did not reach model loading: ($result.output)'
}

def main [--bin-dir: path = $ROOT] {
    let binaries = ['ds4' 'ds4-server' 'ds4-bench' 'ds4-agent'] | each {|name| {name: $name, binary: ($bin_dir | path join $name)} }
    let cli = $bin_dir | path join 'ds4'
    let agent = $bin_dir | path join 'ds4-agent'
    let bench = $bin_dir | path join 'ds4-bench'
    let eval = $bin_dir | path join 'ds4-eval'
    for entry in $binaries {
        assert ($entry.binary | path exists) $'($entry.name) not built'
        let help = (invoke $entry.binary ['--help'])
        if ($help.output | str contains '--rocm') {
            text-check $'($entry.name) help advertises ROCm' $help ['--rocm'] ['--cuda-tensor-parallel']
        } else {
            (text-check
                $'($entry.name) help advertises GPU layout flags'
                $help
                ['gpu-vram' 'gpu-devices' 'cuda-tensor-parallel']
            )
        }
        let runtime = (invoke $entry.binary ['--help' 'runtime'])
        if $entry.name == 'ds4-bench' {
            (text-check
                'ds4-bench runtime help'
                $runtime
                ['--mtp-model FILE' '--dspark' '--dspark-confidence']
            )
        } else {
            (text-check
                $'($entry.name) runtime help'
                $runtime
                ['--mtp' '--mtp-model' 'mtp-exact-sampling']
                ['--glm-mtp']
            )
        }
        let prompt = if $entry.name == 'ds4-bench' { ['--prompt-file' '/dev/null'] } else { [] }
        (rejection
            $'($entry.name) invalid GPU budget'
            (invoke $entry.binary (['--gpu-vram' 'abc' '-m' '/dev/null'] | append $prompt))
            ['--gpu-vram: not a number']
        )
        rejection $'($entry.name) GPU count mismatch' (invoke $entry.binary ([
            '--gpu-vram'
            '40,12'
            '--gpu-devices'
            '0'
            '-m'
            '/dev/null'
        ] | append $prompt)) ['--gpu-devices count (1) does not match --gpu-vram count (2)']
        let cuda_help = (invoke $entry.binary ['--cuda' '--help'])
        assert ($cuda_help.output =~ 'Usage:|usage:|--help') $'($entry.name) --cuda --help did not print help'
        print $'ok ($entry.name) --cuda --help'
    }
    (text-check
        'ds4 distributed help uses canonical spellings'
        (invoke $cli ['--help' 'distributed'])
        ['tensor-parallel-token-prefill']
        ['--tp-']
    )

    if ($eval | path exists) {
        text-check 'ds4-eval runtime help' (invoke $eval ['--help' 'runtime']) ['--mtp-model']
        rejection 'ds4-eval parses support model' (invoke $eval [
            '--mtp-model'
            '/dev/null'
            '-m'
            '/dev/null'
            '--questions'
            '1'
        ])
        rejection 'ds4-eval rejects obsolete --mtp FILE' (invoke $eval [
            '--mtp'
            '/dev/null'
            '-m'
            '/dev/null'
            '--questions'
            '1'
        ]) ['unknown option'] []
        reaches-model 'ds4-eval TP coordinator parses common options' (invoke $eval [
            '--metal' '--tensor-parallel' '--role' 'coordinator' '--listen' '127.0.0.1' '9911'
            '--transport' 'tcp' '--tensor-parallel-token-prefill' '--debug-hash' '2'
            '--rdma-device' 'rdma-test' '--rdma-gid-index' '0' '-m' '/dev/null' '--questions' '1'
        ])
        rejection 'ds4-eval worker directs users to ds4' (invoke $eval [
            '--metal' '--tensor-parallel' '--role' 'worker' '--coordinator' '127.0.0.1' '9911' '-m' '/dev/null'
        ]) ['serving mode; start workers with ./ds4'] []
    }
    rejection 'ds4-bench parses DSpark options' (
        invoke $bench [
            '--dspark'
            '--dspark-confidence'
            '0'
            '--mtp-model'
            '/dev/null'
            '-m'
            '/dev/null'
            '--prompt-file'
            '/dev/null'
        ]
    )
    (rejection
        'ds4-bench requires DSpark support'
        (invoke $bench ['--dspark' '-m' '/dev/null' '--prompt-file' '/dev/null'])
        ['--dspark requires --mtp-model FILE']
        []
        2
    )

    for binary in [
        $cli
        ($bin_dir | path join 'ds4-server')
        $agent
    ] {
        let name = $binary | path basename
        rejection $'($name) parses exact sampling' (invoke $binary ['--mtp-exact-sampling' '--mtp-model' '/dev/null' '-m' '/dev/null'])
        for flag in ['--mtp' '--mtp-timing'] {
            rejection $'($name) parses ($flag)' (invoke $binary [$flag '-m' '/dev/null'])
        }
        for flag in ['--glm-mtp' '--glm-mtp-timing' '--dspark-exact-sampling'] {
            (rejection
                $'($name) rejects obsolete ($flag)'
                (invoke $binary [$flag '-m' '/dev/null'])
                ['unknown option']
                []
            )
        }
    }

    let zero = (invoke $cli ['--gpu-vram' '0' '-m' '/dev/null'])
    rejection 'ds4 zero GPU budget short-circuits before layout' $zero [] ['GPU config:']
    rejection 'TP coordinator rejects layer slices' (invoke $cli [
        '--metal' '--tensor-parallel' '--role' 'coordinator' '--listen' '127.0.0.1' '9911' '--layers' '0:1' '-m' '/dev/null'
    ]) ['always uses one 50/50 worker'] []
    (rejection
        'TP worker requires coordinator address'
        (invoke $cli ['--tensor-parallel' '--role' 'worker' '-m' '/dev/null'])
        ['requires --coordinator HOST PORT']
        []
    )
    reaches-model 'TP common options reach loading' (invoke $cli [
        '--metal' '--tensor-parallel' '--role' 'coordinator' '--listen' '127.0.0.1' '9911'
        '--transport' 'tcp' '--tensor-parallel-token-prefill' '--debug-hash' '2'
        '--rdma-device' 'rdma-test' '--rdma-gid-index' '0' '--inspect' '-m' '/dev/null'
    ])
    for args in [
        ['--tp-coordinator' '9911']
        ['--tp-lead' '9911']
        ['--tp-coordinator-host' '127.0.0.1']
        ['--tp-lead-host' '127.0.0.1']
        ['--tp-worker' '127.0.0.1' '9911']
        ['--tp-transport' 'tcp']
        ['--tp-debug-hash' '2']
        ['--tp-token-prefill']
    ] {
        (rejection
            $'obsolete ($args.0) is rejected'
            (invoke $cli ($args | append ['-m' '/dev/null']))
            ['unknown option']
            []
        )
    }
    let agent_help = (invoke $agent ['--help'])
    let backend = if ($agent_help.output | str contains '--rocm') { 'rocm' } else { 'cuda' }
    rejection $'agent parses --($backend)' (invoke $agent [
        $'--($backend)'
        '--non-interactive'
        '-p'
        'test'
        '-m'
        '/dev/null'
    ])
    rejection $'agent parses --backend ($backend)' (invoke $agent [
        '--backend'
        $backend
        '--non-interactive'
        '-p'
        'test'
        '-m'
        '/dev/null'
    ]) [] ['unknown option' 'invalid backend']
    reaches-model 'agent TP coordinator parses common options' (invoke $agent [
        '--metal' '--tensor-parallel' '--role' 'coordinator' '--listen' '127.0.0.1' '9911' '--transport' 'tcp'
        '--tensor-parallel-token-prefill' '--debug-hash' '2' '--rdma-device' 'rdma-test' '--rdma-gid-index' '0'
        '--non-interactive' '-p' 'test' '-m' '/dev/null'
    ])
    rejection 'agent worker directs users to ds4' (invoke $agent [
        '--metal' '--tensor-parallel' '--role' 'worker' '--coordinator' '127.0.0.1' '9911' '-m' '/dev/null'
    ]) ['serving mode; start workers with ./ds4'] []
    let work = $ROOT | path join '.workspaces'
    mkdir $work
    let prompt = (mktemp -p $work 'gpu-args-prompt.XXXXXX')
    'prompt file parser smoke' | save --force $prompt
    reaches-model 'agent prompt-file parses' (invoke $agent [
        '--non-interactive'
        '--prompt-file'
        $prompt
        '--raw-prompt'
        '-m'
        '/dev/null'
    ])
    rejection 'agent rejects two initial prompt sources' (invoke $agent [
        '--non-interactive'
        '-p'
        'inline'
        '--prompt-file'
        $prompt
        '-m'
        '/dev/null'
    ]) ['specify only one'] []
    rm $prompt
    (text-check
        'ds4 prints requested GPU layout'
        (invoke $cli ['--gpu-vram' '40,12' '-m' '/dev/null'])
        ['GPU config: 2 devices [0,1] requested, budgets 40,12 GB']
    )
    print 'test_gpu_args_cli: PASS'
}
