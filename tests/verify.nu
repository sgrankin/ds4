#!/usr/bin/env nu
# Host and script checks only; model/GPU qualifications are separate commands.
const ROOT = path self ..

const HOST = [
    [step, programs];
    [sampling-test, [test-sampling]]
    [engram-test, [test-engram]]
    [image-test, [test-image-decode]]
    [vision-image-test, [test-deepseek4-vision-image]]
    [vocab-test, [test-vocab]]
    [tokenizer-test, [test-tokenizer-piece]]
    [pretokenizer-test, [test-pretokenizer]]
    [gguf-test, [test-gguf-cursor]]
    [ssd-test, [test-ssd-cache]]
    [prompt-prefix-test, [test-prompt-prefix]]
    [session-state-test, [test-session-state]]
    [payload-io-test, [test-payload-io]]
    [session-snapshot-test, [test-session-snapshot]]
    [stream-selected-test, [test-stream-selected]]
    [pagein-test, [test-pagein]]
    [model-spans-test, [test-model-spans]]
    [multimodal-test, [test-multimodal]]
    [glm53-storage-test, [test-glm53-storage]]
    [serving-boundary-test, [test-serving-boundary]]
    [websocket-frames-test, [test-websocket-frames]]
    [resident-route-test, [test-resident-route test-resident-three test-resident-drop]]
    [state-cache-test, []]
    [response-store-test, []]
]

def --wrapped checked [program: string, ...args: string] {
    ^$program ...$args
    if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
}

def script-tests [] {
    checked $nu.current-exe tests/test_script_launchers.nu
    checked python3 tests/test_model_download.py
    checked $nu.current-exe tests/test_script_drivers.nu
}

# Run only the model-free script contracts, without compiling host fixtures.
def 'main scripts' [] {
    cd $ROOT
    script-tests
}

# Additional arguments are forwarded to zig build, such as -Doptimize=Debug.
def --wrapped main [...zig_flags: string] {
    if '--help' in $zig_flags or '-h' in $zig_flags {
        print 'Usage: nu tests/verify.nu [zig build options...]
       nu tests/verify.nu scripts

Run model-free host and script checks. The scripts subcommand skips compilation.'
        return
    }
    cd $ROOT
    let steps = $HOST | get step
    checked ($ROOT | path join tools/zig-build) ...$zig_flags ds4-eval ...$steps server-test agent-test web-test
    let bin = $ROOT | path join zig-out/bin
    checked ($bin | path join ds4-eval) --validate-cases
    checked ($bin | path join ds4-eval) --self-test-extractors
    for fixture in $HOST {
        for program in $fixture.programs { checked ($bin | path join $program) }
    }
    let work = $ROOT | path join .workspaces $'make-tests.(random uuid)'
    mkdir $work
    checked ($bin | path join test-state-cache) ($work | path join state)
    checked ($bin | path join test-response-store) ($work | path join responses)
    checked ($bin | path join ds4_test) --server
    checked ($bin | path join ds4_agent_test)
    checked python3 tests/test_web_recovery.py ($bin | path join test-web-recovery)
    script-tests
}
