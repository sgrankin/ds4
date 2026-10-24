#!/usr/bin/env nu
# Run from the repository root after building the named fixture targets.
def check-case [out: path, name: string, args: list<string>] {
    print $'Checking ($name)'
    ^python3 -B speed-bench/ideas/pressure_run.py ($out | path join $'($name)-guard') -- ...$args out> ($out | path join $'($name).stdout.log') err> ($out | path join $'($name).stderr.log')
    if $env.LAST_EXIT_CODE != 0 { error make {msg: $'($name) failed; see ($out)'} }
}

def main [bin: path, out: path, --python: path = '.workspaces/responses-ws-venv/bin/python'] {
    mkdir $out
    let ds41 = 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
    let qwen = 'gguf/Qwen3.8-Flash-Next-Q4.gguf'
    let glm = 'gguf/GLM-5.3-Flash-Q2.gguf'
    let flint = '../flint/.local/share/rec/models/fdc85248ad77a0fa9fbf0abda8b1525bea8ee74c11ada2ec95eb66113353d677'
    check-case $out gpu-state [
        ($bin | path join test-session-state-gpu)
    ]
    check-case $out kernels [
        ($bin | path join ds4_test)
        --metal-kernels
    ]
    with-env {DS4_METAL_NO_RESIDENCY: '1', DS4_METAL_NO_MODEL_WARMUP: '1'} {
        check-case $out qwen-state [($bin | path join test-qwen4-ngram-state) $qwen]
    }
    with-env {
        DS4_TEST_MODEL: $glm
        DS4_TEST_SSD_STREAMING: '1'
        DS4_TEST_SSD_STREAMING_CACHE_GB: '24'
        DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP: '1'
        DS4_TEST_GLM_MTP: '1'
        DS4_TEST_SNAPSHOT_CTX: '256'
    } {
        check-case $out glm-state [($bin | path join ds4_test) --session-snapshot --glm53-prefill-cancel]
    }
    mkdir ($out | path join suffix-f16)
    with-env {
        DS4_V41_DECODER_REPLAY_ROWS: '0'
        DS4_METAL_V41_RESIDENT_ROUTE_GAP: '0'
        DS4_METAL_V41_RESIDENT_PREFILL_GAP: '0'
        DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS: '0'
    } {
        check-case $out suffix-f16 [($bin | path join suffix-repeat) --model $ds41 --out ($out | path join suffix-f16) --ctx 4096 --chunk 512 --cache-gb 32 --moves 4 --prerope 1 --source-f16 1 --destination-f16 1]
    }
    for case in [
        {name: ds41, family: deepseek, model: $ds41}
        {name: qwen, family: qwen, model: $qwen}
        {name: glm, family: glm, model: $glm}
        {name: flint, family: deepseek, model: $flint}
    ] {
        check-case $out $'($case.name)-ws' [
            $python
            -B
            tests/test_responses_websocket.py
            --family
            $case.family
            --model
            $case.model
            --server
            ($bin | path join ds4-server)
            --restart-server
            ($bin | path join ds4-server)
            --output
            ($out | path join $'($case.name)-responses')
        ]
    }
}
