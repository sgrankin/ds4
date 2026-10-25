#!/usr/bin/env nu
# Model-free contracts for the preset/build entry points.
use std/assert
const ROOT = path self ..

# Capture only launcher settings, never unrelated user credentials.
const CHILD = '#!/usr/bin/env nu
const PROGRAM = path self | path basename
def --wrapped main [...args: string] {
    let keys = [DS4_METAL_V41_SELECTED_TILE DS4_METAL_V41_SELECTED_LIMIT
        DS4_METAL_V41_RESIDENT_ROUTE_GAP DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS
        DS4_METAL_V41_RESIDENT_PREFILL_GAP DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS
        DS4_AGENT_CONTEXT_SUFFIX_REUSE DS4_AGENT_CONTEXT_SUFFIX_PREROPE
        DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL DS4_AGENT_CONTEXT_SUFFIX_RIGHT
        DS4_SERVER_CONTEXT_SUFFIX_REUSE DS4_SERVER_CONTEXT_SUFFIX_FRESH_TAIL
        DS4_SERVER_CONTEXT_SUFFIX_RIGHT DS4_STATE_CACHE_DIR DS4_STATE_CACHE_HOT_MB
        DS4_STATE_CACHE_SSD_MB DS4_SERVER_RESPONSE_STORE_DIR
        DS4_METAL_DSV41_SOURCE DS4_V41_COMPRESSED_KV_F16
        ZIG_GLOBAL_CACHE_DIR ZIG_LOCAL_CACHE_DIR DS4_CLEF_PREFIX_CACHE]
    {program: $PROGRAM, args: $args, cwd: $env.PWD,
        env: ($env | select --optional ...$keys)} | to json | print
    exit ($env.DS4_TEST_EXIT? | default "0" | into int)
}
'

# Keep the exact process result: a launcher must preserve a failed child status.
def call [program: string, args: list<string>, settings: record = {}] {
    with-env $settings { ^$program ...$args | complete }
}

def main [] {
    let work = $ROOT | path join '.workspaces/nu-launchers' (random uuid)
    let bin = $work | path join 'bin with spaces'
    mkdir $bin
    for name in ['ds4-agent' 'ds4-server' 'zig'] {
        let file = $bin | path join $name
        $CHILD | save $file
        ^chmod +x $file
    }
    cd $work
    let launcher = $ROOT | path join 'ds4-local'
    let leading = ['--bin-dir' $bin]
    let defaults = [
        '--ssd-streaming'
        '--ssd-streaming-cache-experts'
        '64GB'
        '--ctx'
        '8192'
        '--prefill-chunk'
        '512'
    ]
    let requested = [
        '--model'
        'models with spaces/*.gguf'
        '--ctx'
        '256000'
        '--prefill-chunk'
        '2048'
        '-p'
        'say "hi"; $(touch never-created)'
        '--help'
    ]
    let agent_result = (call $launcher ($leading ++ ['persistent-fast'] ++ $requested) {
        DS4_STATE_CACHE_DIR: '' DS4_STATE_CACHE_HOT_MB: '' DS4_STATE_CACHE_SSD_MB: ''
        DS4_AGENT_CONTEXT_SUFFIX_PREROPE: '' DS4_AGENT_CONTEXT_SUFFIX_RIGHT: ''
        DS4_METAL_V41_RESIDENT_PREFILL_GAP: '' DS4_V41_COMPRESSED_KV_F16: '1'
    })
    assert equal $agent_result.exit_code 0 $agent_result.stderr
    let agent = $agent_result.stdout | from json
    assert equal $agent.program 'ds4-agent'
    assert equal $agent.args ($defaults ++ $requested)
    assert equal $agent.cwd $work
    assert equal $agent.env.DS4_STATE_CACHE_DIR ($ROOT | path join '.workspaces/context-cache')
    assert equal $agent.env.DS4_STATE_CACHE_HOT_MB '512'
    assert equal $agent.env.DS4_STATE_CACHE_SSD_MB '8192'
    assert equal $agent.env.DS4_AGENT_CONTEXT_SUFFIX_PREROPE '1'
    assert equal $agent.env.DS4_AGENT_CONTEXT_SUFFIX_RIGHT '1'
    assert equal $agent.env.DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL '128'
    assert equal $agent.env.DS4_METAL_V41_RESIDENT_ROUTE_GAP '.90'
    assert equal $agent.env.DS4_METAL_V41_RESIDENT_PREFILL_GAP '.90'
    assert equal $agent.env.DS4_V41_COMPRESSED_KV_F16 '1'
    assert equal $agent.env.DS4_METAL_DSV41_SOURCE ($ROOT | path join 'src/metal/dsv41.metal')
    assert not ('never-created' | path exists)

    let server_result = (call $launcher (['--server'] ++ $leading ++ ['--preset' 'persistent-fast' '--ctx' '1234']) {
        DS4_SERVER_CONTEXT_SUFFIX_FRESH_TAIL: '96' DS4_SERVER_CONTEXT_SUFFIX_RIGHT: '0'
        DS4_STATE_CACHE_DIR: ($work | path join 'custom cache/state')
        DS4_STATE_CACHE_HOT_MB: '' DS4_STATE_CACHE_SSD_MB: ''
        DS4_SERVER_RESPONSE_STORE_DIR: ($work | path join 'custom store/responses')
        DS4_METAL_DSV41_SOURCE: 'shader override with spaces.metal'
    })
    assert equal $server_result.exit_code 0 $server_result.stderr
    let server = $server_result.stdout | from json
    assert equal $server.program 'ds4-server'
    assert equal $server.args (['--host' '127.0.0.1' '--batched-session' '2'] ++ $defaults ++ ['--ctx' '1234'])
    assert equal $server.env.DS4_SERVER_CONTEXT_SUFFIX_FRESH_TAIL '96'
    assert equal $server.env.DS4_SERVER_CONTEXT_SUFFIX_RIGHT '0'
    assert equal $server.env.DS4_STATE_CACHE_HOT_MB '0'
    assert equal $server.env.DS4_STATE_CACHE_SSD_MB '4096'
    assert equal $server.env.DS4_METAL_DSV41_SOURCE 'shader override with spaces.metal'
    assert (($work | path join 'custom cache') | path exists)
    assert (($work | path join 'custom store') | path exists)

    let flint_args = ['--model' 'models with spaces/flint' '--ctx' '32768' '--prefill-chunk' '1024']
    let flint_result = (call $launcher (['--server'] ++ $leading ++ ['flint'] ++ $flint_args) {
        DS4_STATE_CACHE_DIR: ($work | path join 'incompatible state')
        DS4_STATE_CACHE_HOT_MB: '512' DS4_STATE_CACHE_SSD_MB: '8192'
        DS4_METAL_V41_SELECTED_TILE: '128' DS4_METAL_V41_RESIDENT_ROUTE_GAP: '.90'
        DS4_SERVER_CONTEXT_SUFFIX_REUSE: '1' DS4_AGENT_CONTEXT_SUFFIX_REUSE: '1'
        DS4_V41_COMPRESSED_KV_F16: '1'
        DS4_SERVER_RESPONSE_STORE_DIR: ($work | path join 'flint store/responses')
    })
    assert equal $flint_result.exit_code 0 $flint_result.stderr
    let flint = $flint_result.stdout | from json
    assert equal $flint.program 'ds4-server'
    assert equal $flint.args (['--host' '127.0.0.1' '--batched-session' '2' '--ctx' '262144' '--prefill-chunk' '4096'] ++ $flint_args)
    assert equal $flint.cwd $work
    assert equal $flint.env.DS4_SERVER_RESPONSE_STORE_DIR ($work | path join 'flint store/responses')
    for key in [DS4_STATE_CACHE_DIR DS4_STATE_CACHE_HOT_MB DS4_STATE_CACHE_SSD_MB
        DS4_METAL_V41_SELECTED_TILE DS4_METAL_V41_RESIDENT_ROUTE_GAP
        DS4_SERVER_CONTEXT_SUFFIX_REUSE DS4_AGENT_CONTEXT_SUFFIX_REUSE DS4_V41_COMPRESSED_KV_F16] {
        assert equal ($flint.env | get --optional $key) null $'Flint inherited ($key)'
    }
    let flint_agent = call $launcher ($leading ++ ['flint'] ++ $flint_args)
    assert equal $flint_agent.exit_code 2
    assert ($flint_agent.stderr | str contains '--server flint')

    let context = (call $launcher ($leading ++ ['context-fast'])).stdout | from json
    assert equal $context.env.DS4_METAL_V41_RESIDENT_ROUTE_GAP '0'
    assert equal $context.env.DS4_AGENT_CONTEXT_SUFFIX_REUSE '1'
    let resident = (call $launcher (['--server'] ++ $leading ++ ['resident-fast'])).stdout | from json
    assert equal $resident.env.DS4_METAL_V41_RESIDENT_ROUTE_GAP '.90'
    assert equal $resident.env.DS4_AGENT_CONTEXT_SUFFIX_REUSE '0'
    assert equal $resident.env.DS4_SERVER_CONTEXT_SUFFIX_REUSE '0'
    assert equal (call $launcher ($leading ++ ['context-fast']) {DS4_TEST_EXIT: '7'}).exit_code 7
    assert equal (call $launcher ['--help']).exit_code 0
    assert equal (call $launcher ['invalid-profile']).exit_code 2
    assert equal (call $launcher ['--bin-dir']).exit_code 2

    let builder = $ROOT | path join 'tools/zig-build'
    let build_args = ['--prefix' 'output path with spaces' 'server-test' '--help']
    let build_result = (call $builder $build_args {
        PATH: ([$bin] ++ $env.PATH)
    })
    assert equal $build_result.exit_code 0 $build_result.stderr
    let build = $build_result.stdout | from json
    assert equal $build.program 'zig'
    assert equal $build.args (['build'] ++ $build_args)
    assert equal $build.cwd $ROOT
    assert equal $build.env.ZIG_GLOBAL_CACHE_DIR ($ROOT | path join '.workspaces/zig-build/global-cache')
    assert equal $build.env.ZIG_LOCAL_CACHE_DIR ($ROOT | path join '.workspaces/zig-build/local-cache')
    assert equal (call $builder [] {
        PATH: ([$bin] ++ $env.PATH)
        DS4_TEST_EXIT: '9'
    }).exit_code 9
    for profile in [clef-flash clef clef-omni] {
        let result = call $launcher (['--server'] ++ $leading ++ [$profile '--port' '9901']) {
            DS4_STATE_CACHE_DIR: 'incompatible-cache'
            DS4_SERVER_CONTEXT_SUFFIX_REUSE: '1'
        }
        assert equal $result.exit_code 0 $result.stderr
        let child = $result.stdout | from json
        assert equal $child.program ds4-server
        assert equal $child.args [--host 127.0.0.1 --model ($ROOT | path join gguf $"($profile)-int4") --ctx (if $profile == clef-omni {'64000'} else {'16384'}) --port '9901']
        assert equal ($child.env | get --optional DS4_STATE_CACHE_DIR) null
        assert equal ($child.env | get --optional DS4_SERVER_CONTEXT_SUFFIX_REUSE) null
        assert equal $child.env.DS4_CLEF_PREFIX_CACHE '1'
    }
    let uncached = call $launcher (['--server'] ++ $leading ++ [clef-flash]) {DS4_CLEF_PREFIX_CACHE: '0'}
    assert equal $uncached.exit_code 0 $uncached.stderr
    assert equal ($uncached.stdout | from json | get env.DS4_CLEF_PREFIX_CACHE) '0'
    assert equal (call $launcher ($leading ++ [clef-flash])).exit_code 2
    print 'PASS: Nu launchers: profiles, quoted argv, overrides, shader paths, cache paths, help and child status'
}
