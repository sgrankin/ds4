# Native SIMD index scoring and physical FP4 caches

Completed 2026-10-27: native Metal inference now uses the new scalar reader,
physical index allocation/publication, batched prefill, canonical SSD format and
bounded suffix conversions. Qualification passes, all five executables are
installed, and main remains unchanged. The F32 item ran 16:47–17:19 UTC and packed
integration 17:19–18:13 UTC, each within its one-hour budget.
This follows the [SIMD kernel experiment](../index-simd-20261027/README.md) and
[DeepInfer alignment work](../deepinfer-alignment-20261027/README.md). The
[self-contained format recipe](../../../recipes/14-packed-index-cache.md) covers
replication in another harness.

The control is frozen parent `641cfaa5`, including every runtime-loaded Metal
source beside the executable. Useful raw artifacts remain ignored under
`.workspaces/index-serving-20261027/`. All model jobs run sequentially through
the existing pressure guard. The user-owned reference Flint server stays
resident and untouched. These are measurements on the installed 1017 IQ2/Q2
model and this M5 Max, not a cross-hardware speed claim.

## Implementation and settings

| Setting | Behavior |
| --- | --- |
| `DS4_V41_INDEX_SIMD=1` | One SIMDgroup owns a key; four keys share a 128-thread group. Register `float4` keys remove shared key tiles and threadgroup barriers. Plain-engine F32 remains opt-in; DeepSeek fast presets now default this flag to 1. |
| `DS4_V41_INDEX_FP4=1` | Allocate 68-byte index rows: 64 bytes of signed E2M1 codes and 4 E8M0 scales. Publication packs original post-RoPE keys; no persistent F32 owner shadow remains. |
| SIMD crossover | Final gate is 1,024 keys. The old layout wins at 128/256; measured 1,024/2,048/4,096 points favor the register layout. With physical FP4 enabled, SIMD is automatic above the gate unless explicitly disabled. |
| `DS4_V41_CANDIDATE_INDEX` | Existing inherited-block filtering remains a separate axis. Fresh task comparisons hold it at0; the older hot-loop CSV records separate0/1 strata. |
| `DS4_V41_CHECKPOINT_DRAFT_FEATURES=1` | Independent DSP1 history persistence; used in checkpoint/MTP qualification. |

Batched prefill decodes physical keys directly into the existing transient BF16
matrix tiles and retains causal/tie order. A SIMD fallback covers small batches
and devices without the matrix path. This removes the persistent F32 index
arrays; the context-sized BF16 prefill workspace and other graph buffers remain.
No saved bytes automatically enlarge the fixed expert-cache budget.

Physical graphs export canonical DSV4v3 with 68-byte index rows; F32 graphs export
v2 with 512-byte index rows. The 13-word header and token offsets stay unchanged.
Main KV wire values remain F32. New loaders accept both versions into either
physical graph layout using bounded conversion scratch; older binaries reject
v3. DSP1 follows either base at its actual endpoint. SSD validation, manifest
restart, page splitting and equal-prefix selection understand both versions.
Suffix conversion handles both layouts; approximate recaptures retain their
private v2 representation. These format changes are actual code, with the
qualification status below distinguishing what has been run.

## Measured inference results

Normal DeepSeek reasoning stays enabled. Task probes use 8,192-token prefill
chunks, 48 GiB expert cache, 384 maximum generated tokens and contexts sized to the
prompt plus 8,192. Fresh answers use independent prefills from the forced teachers.
The short forced teachers begin at a reasoning frontier; they are distribution
comparisons, not natural reasoning traces. Generated reasoning/answer IDs and
practical task answers provide a separate ordinary-use check.

The first 32K F32 bracket preserves all 8 teacher distributions bit for bit
(ΔNLL=KL=0), all 42 generated IDs and the correct favorable-case answer. Decode
takes 4.653/4.020/3.979seconds for control/SIMD/closing control. The 14.5% control
drift exceeds the apparent 6.9% improvement against averaged controls; SIMD is
1.0% slower than the closing control. Preserve this inconclusive bracket.

Same-process sweeps restore one prefix and replay exactly the same generated
IDs, alternating SIMD 0/1. Repeat 0 is retained warmup; seven measured pairs follow.

| Prompt | Trace tokens | Control mean | SIMD mean | Mean paired time reduction | Median paired reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
|32K |42 |1809.602ms |1800.123ms |0.523% |0.381% |
|64K |27 |1214.857ms |1191.326ms |1.936% |1.866% |

The geometric speedups are 1.00526× and 1.01975×. These support small warmed
scalar-decode gains, rather than transferring the isolated kernel's 30–40%
reduction to whole-model execution. The sweeps used the initial 8,192-key gate;
these context sizes exceed both that gate and the final 1,024-key gate.
Evidence: `f32-simd-loop32/paired-analysis.json` and
`f32-simd-loop64/paired-analysis.json` under the ignored artifact root.

The first packed 8K bracket also preserves all 22 teacher rows exactly
(ΔNLL=KL=0, no argmax changes), every generated token ID and all 3 task answers.
It used the initial 8,192-key gate, before the final crossover adjustment.

| Task | Generated tokens | Decode-time change vs bracketed control |
| --- | ---: | ---: |
| Ordinary |87 |1.54% less |
| Favorable |31 |4.58% less |
| Changed fact |26 |6.46% more |

Prefill and whole-task timings also vary. This is useful quality evidence and
mixed latency evidence; there is no general packed-serving speed claim. The
original final 1,024-gate 32K favorable bracket also passes all 8 teacher rows and all 42
generated IDs exactly. Averaged controls versus packed are 45.631→47.666 s for
prefill (+4.46% time), 4.011→3.863 s for the free answer (−3.68%), and
49.642→51.529 s for their sum (+3.80%). Warm fixed-trace loops with candidate
filtering held off take 1821.490→1850.182 ms (+1.58%). These controls are the
older frozen F32 binary with SIMD disabled; the packed candidate also enables
SIMD. Preserve this noisy package comparison, but do not treat it as an isolated,
established packing penalty. The follow-up below supersedes that interpretation.
Evidence: `packed-8k/comparison.json`, `packed-32k/comparison.json` and each
arm's timing/answer and candidate-sweep files.

### Five samples per layout, with only packing changed

The [follow-up](../index-five-samples-20261027/README.md) runs five fresh 32K
favorable processes per layout, alternating AB/BA/AB/BA/AB order. Both arms use
the same current binary and frozen shaders, SIMD=1 and candidate-only scoring=0;
only `DS4_V41_INDEX_FP4` changes. All ten runs are retained, with no discarded
warmup or outlier. The ordinary reasoning trace remains 42 tokens in every run.

| Timed scope | Mean F32 seconds | Mean packed seconds | Aggregate time change |
|---|---:|---:|---:|
| First teacher prefill |44.735417 |44.419541 |−0.7061% |
| Independent answer prefill |44.721956 |44.441046 |−0.6281% |
| Answer decode |4.008286 |3.920567 |−2.1884% |
| Answer prefill + decode |48.7302418 |48.3616122 |−0.7565% |

Aggregate change is the ratio of phase means minus one, not the mean of paired
ratios. The whole-reply paired median is −0.1221%, and packed is faster in 3/5
pairs. Pair directions follow order, and the first pair contributes appreciably
to the mean; neither a repeatable penalty nor a robust speedup is established.
Every aggregate phase mean, including the teacher step, is lower, so the
user-requested slowdown-triggered profile was not run. The answer prefill builds
fresh KV after the teacher pass has warmed experts; initialization is excluded.
These are fresh-process, warm-expert answer timings, not cold-start latency.

All ten answers are correct and stopped, every generated reasoning/answer ID
matches, and all 80 fixed-teacher rows are byte-exact (ΔNLL=KL=0, no argmax
changes). Raw results remain under `.workspaces/index-five-samples-20261027/`;
the linked report retains every pair and order stratum. The capacity saving
remains established independently of this unresolved small latency difference.

A separate serial diagnostic profiles one raw 8,192-token prefill per layout
with generation disabled. It measures 425.18→429.39 tokens/s and captured GPU
time 10,357.93→10,342.26 ms. Matrix preparation is 1.977→1.911 ms and matrix
scoring 24.745→23.263 ms. This does not reproduce the earlier whole-prefill
slowdown or establish an extra FP4-unpack bottleneck. Encoder-timestamp pass
boundaries perturb execution, and this is one raw-prompt pair; use it for
attribution only. No further optimization is justified by that hypothesis.
Evidence: `prefill-profile/summary.json`; reproduction uses `profile-prefill.nu`
followed by `python3 speed-bench/experiments/index-serving-20261027/analyze-prefill.py OUTPUT`.

## Allocation, SSD and compatibility

Actual `ds4_gpu_tensor_bytes` sums at context 2,048 are 2,623,488 B for F32 index
owners and 348,432 B for physical FP4: 2,275,056 B saved, 86.72% smaller. The static
context plans change 733.02→730.85 MiB; all other graph/drafter allocations remain.
This is real owner allocation evidence, not just the 68-byte row formula or RSS.

The packed checkpoint oracle passes wrapped 257-token history, v2→packed import,
v3→F32 import, old base and flag-off restores, HOT0 SSD close/reopen and immediate
positive-temperature constrained verification. The first packed/F32 teacher
comparison has max absolute logit error 0; later conversion cycles show 2.861e−6.
Keep that tiny measured drift rather than claiming every converted continuation
is bit-identical.

The repeated checkpoint test publishes actual same-prefix SSD twins at 263
tokens: v2 is 16,627,824 B and v3 is 16,336,560 B, both including the same
3,932,192 B DSP1 appendix. The 291,264 B reduction is exactly the packed index
row difference and reduces this complete short checkpoint by 1.752%. Raw rings,
main KV and DSP1 still account for most bytes. Manifest/header/chunk extents
agree, and v3 materialized bytes round-trip exactly after HOT0 restart. This is
now a measured size pair, replacing the earlier layout-only calculation.
Evidence: `packed-checkpoint-twin.stderr`, `packed-checkpoint-twin-cache/` and
`checkpoint-twin-analysis.json` (matching manifest prefix hashes).

The public suffix fixture also completes all four moves (−66, +128, −514 and
zero shift), retaining original pre-RoPE values through recapture. Source main
KV is F32, destination main KV is F16, and both index caches are physical FP4.
Source-state and original-tape hashes stay unchanged; capture-size estimates
match the actual payload. Sixteen teacher comparisons give mean ΔNLL +0.0000372
nats/token, mean KL 0.0000284 and no argmax flips. This qualifies format/relocation
compatibility in the existing case, not a new general suffix-quality or FP4
speed claim. Evidence: `packed-suffix/moves.csv`, `packed-suffix/rows.csv` and
`guard-packed-suffix-checkpoint/result.json` (exit 0).

GPU producer/unpack/import cases through 256K pass, as do all 32 batched index
cases covering tensor/fallback paths, ratio 1/2, causal bounds, query offsets and
partial tiles. Legacy C graph reference fixtures assume 512-byte index rows and
are not qualification for physical FP4. Use the native GPU batch fixture and
public checkpoint/suffix paths below.

Completed GPU, checkpoint, crossover, paired-sweep and closing-retry guards
exit 0 with pressure level 1 and unchanged 21.5136 GiB swap. The initial bracket
closing arm exited 1: a frozen executable read a live, incomplete shader during
parallel development. The failure remains in `f32-32k/dense-repeat-shader-failure/`;
the retry and later arms freeze all runtime Metal sources beside each binary.
Packed Responses HTTP checks pass 7/7 semantic probes with normal reasoning,
including T0/T1 and strict tool/schema outputs. All 27 streamed argument deltas
reconstruct the final typed arguments. After a clean server restart,
`previous_response_id` recalls ORCHID-5179 without the fact in new input; that
history-recovery request has cached_tokens=0. A separate repeated strict-tool
request restores 392/392 input tokens, returns the same typed arguments, and
completes 169 generated tokens including 82 reasoning tokens. Logical history
recovery and actual KV reuse are therefore separately demonstrated. Fresh and
restart servers exit 0; their guards also stay at pressure level 1 with stable
21.5076 GiB swap. Evidence: `packed-server/http/results.json`,
`packed-server/http/stream.result.json`, `packed-server/durable-after.log` and
`packed-server/restart-tool.response.json`.

Host CPU/server/agent/state checks pass, as do the Nushell/script and formatted
launcher contracts. The initial sandbox localhost web check failed; the same
fixture passed with local networking allowed. Both environment outcomes remain
in `host-verify.log` and `web-recovery-verify.log`. All five final executables
(`ds4`, `ds4-agent`, `ds4-server`, `ds4-bench`, `ds4-eval`) are installed at root
and `zig-out/bin`; their hashes match the qualified packed build and all help
commands exit 0. `installed.json` records those checks. Named-preset agent/server
help also exits 0. Integration and the selected qualification are complete;
there is no claim of a universal packing speedup or qualification on an M1 Max.

## Preset enablement

The launcher now has `persistent-packed`: the existing `persistent-fast`
behavior plus `DS4_V41_INDEX_FP4=1`, including compact canonical SSD checkpoints.
The other DeepSeek fast profiles keep their F32 index default and now enable
SIMD scoring. Explicit `DS4_V41_INDEX_FP4=0` or `DS4_V41_INDEX_SIMD=0` overrides
win; launcher tests cover the new preset and those overrides. This makes the
capacity/latency tradeoff selectable without making packing the default for
existing profiles. Qualified root binaries are installed.

Start the agent or server with:

```nu
./ds4-local persistent-packed --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
./ds4-local --server persistent-packed --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf
```

## Reproduction

`run.nu` reuses `decoder-replay-bench`; each three-arm run brackets a candidate
between frozen controls. `manifest.json` records binary paths and flags. The
legacy directory names `dense/candidates/dense-repeat` are consumed by the
existing numerical comparison script. Freeze all `src/metal/*.metal` files in
`BUILD_PREFIX/shaders/` beside each `BUILD_PREFIX/bin/` before running; the driver
requires those sources and sets their overrides explicitly.

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/repro-guard -- nu speed-bench/experiments/index-serving-20261027/run.nu BASELINE_PROBE CANDIDATE_PROBE .workspaces/repro-results --tokens 8192 --packed 1
python3 speed-bench/experiments/deepinfer-alignment-20261027/compare.py .workspaces/repro-results
```

Use `--hot-loop --tokens 32768` for the existing candidate 0/1 loop over one saved
trace. Compare matched candidate strata across arms. Use `--simd-loop` for the
cleaner one-process SIMD axis: it invokes only the new candidate probe, holds
candidate selection fixed and writes eight pairs including warmup 0. Analyze it
with the small stdlib numerical reducer:

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/repro-sweep-guard -- nu speed-bench/experiments/index-serving-20261027/run.nu BASELINE_PROBE NEW_PROBE .workspaces/repro-sweep --tokens 65536 --simd-loop
python3 speed-bench/experiments/index-serving-20261027/analyze-sweep.py .workspaces/repro-sweep/simd-sweep/simd-sweep.csv .workspaces/repro-sweep/paired-analysis.json
```

The installed-model numerical fixtures run separately under the same pressure
guard, one at a time:

```nu
NEW_BIN/test-deepseek41-metal --index-fp4
NEW_BIN/test-deepseek41-metal --index-fp4-batch
NEW_BIN/test-dspark-checkpoint gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf gguf/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf EMPTY_CACHE_DIR --packed-cross
```

Packed suffix qualification uses the existing public `suffix-repeat` fixture
with `DS4_V41_INDEX_FP4=1`, `--moves 4 --prerope 1 --chunk 512 --cache-gb 32`.
Responses qualification reuses the constrained HTTP/WS launcher with installed
DSpark support, normal reasoning, strict tool/schema outputs and streamed/final
argument checks. Keep raw logs, vocabulary dumps and generated checkpoints local
and ignored; commit concise reports and the reproduction drivers.
