# Clef state-prefix reuse — 2026-10-25

Implemented and qualified on full Clef and Flash on M5 Max 128 GB. A mixed
27-request replay takes **44% less time** with each model. All 36 decisions per
model, including their complete probability distributions, match the uncached
run exactly: KL and gold delta-NLL are zero. The ordinary kernel comparison is
independent of this item; both replay arms use the same optimized kernels.

## Results

Nine request shapes run three times, including first fills, changed questions,
early fact edits, short input between hits and alternating states. Each model
has 11 cache hits out of 27 requests and reuses 16,064 total tokens.

| Workload | Full uncached → cached | Flash uncached → cached |
| --- | ---: | ---: |
| Long state, changed questions (3,081 tokens) | 3,387 → 495 ms | 959 → 149 ms |
| Support history, grown schema (602 tokens) | 764 → 305 ms | 203 → 94 ms |
| Shrunk schema after unrelated short request (510 tokens) | 593 → 167 ms | 166 → 58 ms |
| Early fact change, cache miss (490 tokens) | 596 → 596 ms | 163 → 162 ms |
| Entire mixed replay | 39.07 → 21.92 s | 11.21 → 6.32 s |

Individual rows are medians of three observations; totals include every first
call and fill. These are synthetic qualification cases, not a general agent
speed guarantee. All 72 known task answers are correct. Early edits miss on
every repetition and reverse billing→technical as expected; unrelated short
requests bypass the cache without destroying reusable state.

Actual HTTP tests pass changed schemas/facts, five image/video checks, and
text reuse after media. Logs confirm the returning text reuses 2,752 tokens.
The composed Flint/Flash server also passes 18 existing HTTP/WebSocket,
concurrency, cancellation, ancestry and official-SDK checks. Its guard exits
normally after controlled shutdown. All model runs have normal pressure and
zero swap growth. Raw evidence is local in
`.workspaces/clef-performance-20261025/` (`prefix-*-control`, `prefix-*-cached`,
`prefix-http`, `mixed-serving` and `serving-guard`).

## What is reused

Named `ds4-local` profiles enable this for loaded dense Clef helpers; direct
server users set `DS4_CLEF_PREFIX_CACHE=1`. Set it to `0` to opt out. The engine
retains one in-memory text prefix before the question schema. A subsequent
request can ask different questions or supply different options over the same
state: only the remaining tokens run through the backbone, then the ordinary
learned head runs over the complete retained-plus-fresh hidden rows. This is
causal prefix reuse, rather than returning a previously cached answer.

[`record.zig`](../../../src/models/clef/record.zig) supplies `schema_start`.
[`prefix.zig`](../../../src/models/clef/prefix.zig) rounds that boundary down to
64 tokens and requires at least256 prefix tokens. Reuse checks actual leading
token IDs; an early fact edit misses and fills a new entry. An edit after the
boundary is processed afresh. The entry holds per-layer attention K/V, GDN
recurrent state, convolution tails and final normalized hidden prefix. Questions,
lexical rows, probabilities and the head remain fresh on every request.

The entry belongs to the loaded engine and its existing mutex. It has no session
or client-key lookup. It reserves512 extra token rows, aligned to64 and capped
at16384, so modest schema growth can reuse the same entry. Growth beyond capacity
replaces it; there is no checkpoint search for changed middle text. Short text,
media and Omni use their ordinary paths. Failure during prefix inference leaves
no usable partial entry. Prefix-aware attention follows credited
[Cleffa](https://github.com/zknpr/cleffa/tree/24fc91e28d154ff8dadfa58462a30b437413c3f4);
the MIT notice is in `src/metal/clef/LICENSE.cleffa`.

The performance candidate independently selects dequantized MPP for at least2048
processed dense rows and avoids the FFN's unnecessary BF16→FP32 widening pass.
Those kernel choices apply with caching off too. A cache hit has fewer processed
rows and can therefore select another GEMM shape; exact token identity does not
promise bit-identical rounding before qualification.

## Retained memory estimate

For capacity`C`, hidden width`H`, full-attention layers`A`, GDN layers`D`,
KV heads`K`, head dimension`h`, GDN heads/dimensions`Hv,dk,dv`, convolution
width`Cconv` and kernel length`k`, retained bytes are approximately:

```text
4 C (H + 1) + 24
+ A × 8 (K C + 64) h
+ D × 4 (Hv dk dv + (k − 1) Cconv)
```

This includes owned IDs, CPU hidden rows and GPU state, excluding allocation
metadata and temporary overlap during growth. At2881 input tokens, capacity3456
gives **321.26 MiB for Flash and 651.14 MiB for full Clef**. Actual cache logs
report 336,868,888 and 682,767,896 retained payload bytes respectively. These
exclude allocation metadata and ordinary model/head/scratch memory.

## Reproduce

The existing [`prefix_requests.nu`](prefix_requests.nu) generates nine requests
with separate expected answers: the chart input, a varied support narrative,
changed/grown/shrunken questions, an early changed fact, short-input bypass and
state replacement. [`compare_prefix.py`](compare_prefix.py) compares saved
answers, task choices, KL/delta-NLL, timing, reused rows and retained bytes.

```nu
let out = '.workspaces/clef-prefix'
nu speed-bench/experiments/clef-native-20261024/prefix_requests.nu ($out | path join 'inputs') --model clef --repeats 3
nu tools/zig-build clef-smoke -j2 --prefix ($out | path join 'build')
for cache in [0 1] {
    let arm = $out | path join $'cache-($cache)'
    mkdir $arm
    with-env {DS4_CLEF_PREFIX_CACHE: ($cache | into string), DS4_CLEF_TIMING: '1'} {
        python3 speed-bench/ideas/pressure_run.py ($arm | path join 'guard') -- ($out | path join 'build/bin/clef-smoke') gguf/clef-int4 ($out | path join 'inputs/requests.jsonl') out> ($arm | path join 'answers.jsonl') err> ($arm | path join 'run.log')
        if $env.LAST_EXIT_CODE != 0 { error make {msg: 'Guarded prefix run failed; inspect its logs.'} }
    }
}
python3 speed-bench/experiments/clef-native-20261024/compare_prefix.py ($out | path join 'cache-0') ($out | path join 'cache-1') ($out | path join 'inputs/expected.json')
```

Repeat with `--model clef-flash` and `gguf/clef-flash-int4`. Run one model job at
a time. First fills include allocation and shape warmup; inspect individual hit,
miss and bypass rows rather than treating this sequence as quiet steady-state
latency. Raw JSONL, distributions and logs remain ignored under `.workspaces/`.
