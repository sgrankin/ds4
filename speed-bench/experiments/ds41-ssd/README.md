# DS4.1 SSD experiments on Apple M5 Max, 128 GiB

Base: bd66c402. Fixed input: `jj file show -r bd66c402 ds4.c > /tmp/ds41-prefill-input.c`
(SHA-256 `1776dbfed177ea14f3ce6cac1d8d0b1c1b44dfff2c2663769a9a5634aeec34e7`).
Model: gguf/DeepSeek-V4.1-Flash-Q2.gguf. SSD streaming is required.

## 01: baseline and vision-loaded text-only benchmark

Added `ds4-bench --vision FILE` to isolate encoder loading from agent prompt changes.
`make -j2 ds4-bench` passed. Initial 2048-token raw ds4.c prefix, ctx=2053,
4 decode tokens, default streaming cache: 130.59 tps without vision, 130.95
with vision. These single runs do not reproduce the reported agent slowdown.
Logs and CSVs are adjacent. GPU busy profile is cumulative and sampled, not
a precise per-stage breakdown. Baseline prefill wall time is 15.68 seconds;
GPU busy reaches approximately 6.1 seconds near prefill completion.

Command: `DS4_METAL_GRAPH_PREFILL_PROFILE=1 DS4_METAL_GPU_BUSY_PROFILE=1
./ds4-bench -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --ssd-streaming
--prompt-file ds4.c --ctx-start 2048 --ctx-max 2048 --gen-tokens 4`
Add `--vision gguf/DeepSeek-V4.1-Flash-Vision.gguf` for loaded vision.

`run_abba.py` compares controls and candidates in fresh processes, retains
commands and profiles, and requires identical final full-vocabulary logit
values (9 significant digit float serialization). Copy the baseline ds4.c to
/tmp/ds41-prefill-input.c before changing source so token inputs remain fixed.
Results with full logits belong in /tmp; retain summaries here.

## 02: one prefetch reader versus default eight (no default change)

`python3 speed-bench/experiments/ds41-ssd/run_abba.py /tmp/ds41-threads1
--candidate-env DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_THREADS=1`

ABBA, fresh processes, 2048 tokens, 8 teacher-forced decode tokens.
Control: 131.08, 134.16 tps. Candidate: 135.14, 136.48 tps (about 2.4%
faster by aggregate time). All full-vocabulary final logits match exactly.
Steady decode: 17.13/17.32 control, 16.82/17.34 candidate.
The improvement is too small relative to run variation to change a shared
backend default. Retain this as a documented negative/inconclusive experiment.

## 03: two-layer lookahead (rejected; opt-in only)

`run_abba.py /tmp/ds41-lookahead2 --candidate-env DS4_METAL_V41_PREFETCH_TWO_LAYERS=1`

Control 128.12/130.04 tps; two-layer lookahead 87.49/94.07 tps.
All final full-vocabulary logits match exactly. Explicit prefetch waits mostly
disappear but later GPU drains grow from ~0.1-0.2 seconds to ~1.2 seconds.
This suggests page residency/contention rather than a compute improvement.
Keep the default one-layer pipeline. The optional experiment adds pageable
file-cache pressure, not another pinned expert cache; it is not recommended.

Actual unmodified agent with vision, default 100000-token context, `-n 1
--non-interactive -p 'Reply with OK.' --trace /tmp/ds41-agent-vision.trace`:
1907-token system prefill = 14805.383 ms; system prompt KV save succeeded.
40-token append = 3814.777 ms. This reproduces slow short appends, not the
reported large-prefill vision slowdown. No image was supplied.

## 04: fold an unaligned tail into a wide sweep (rejected: logits differ)

`run_abba.py /tmp/ds41-whole-tail --candidate-env DS4_METAL_V41_PREFILL_WHOLE_TAIL=1
--tokens 5000 --ctx 100000`

Control: 67.61 tps, candidate: 257.66 tps. Aborted after the first pair because
all 129280 logits differed, max absolute difference 2.651666, RMSE 0.565702.
Argmax was 2701 in both. This is not an accepted numerical optimization.
The control executes 4096 batched tokens then 904 scalar steps; candidate
executes one 5000-token sweep. Arithmetic/batch partitions differ.
The rejected implementation is retained only as whole-tail.patch, not enabled
in source. Runner now accepts --ctx to match the agent's 100000-token allocation.

## 05: eliminate redundant scalar layer-end drains (accepted)

128-token appends after a 2048-token prefill; ctx=100000. ABBA control
15.48/15.13 tps, candidate 16.93/16.51 tps: about 9.2% faster by aggregate
time. Both frontier full-vocabulary logit rows match across all four runs.
Development command: `run_abba.py /tmp/ds41-scalar-queue
--candidate-env DS4_METAL_V41_SCALAR_QUEUE=1 --tokens 2176
--initial-tokens 2048 --ctx 100000`.

Enabled by default only for single-GPU SSD-streaming DS4.1 scalar execution,
excluding quality and imatrix modes. TP behavior is unchanged. Streaming
route/cache checks already submit work; retain explicit drains before
reusing the Engram input at layer 14 and at token completion.
The rollback is now `DS4_METAL_DISABLE_V41_SCALAR_QUEUE=1` (the development
opt-in was removed). It benefits short prefill appends, scalar tails and decode;
it does not change the layer-major batch kernels.

Regression: `make -j2 tests/test_deepseek41_scalar_queue`, then
`./tests/test_deepseek41_scalar_queue gguf/DeepSeek-V4.1-Flash-Q2.gguf
/tmp/ds41-prefill-input.c gguf/DeepSeek-V4.1-Flash-Vision.gguf`.
With vision loaded: 32 complete logit rows and the serialized continuation
snapshot are bit-identical between scalar schedules, including snapshot restore.
Final default-enabled build also passed the same 32-row/snapshot regression
without vision. `make -j2 ds4-bench ds4-agent tests/test_deepseek41_scalar_queue`
completed successfully.

## 06: longer vision-loaded text parity (verified)

`run_abba.py /tmp/ds41-vision8192 --candidate-env DS4_BENCH_VISION_COMPARISON=1
--candidate-vision gguf/DeepSeek-V4.1-Flash-Vision.gguf --tokens 8192 --ctx 100000`
The environment name is a no-op marker; candidate-vision controls encoder load.
Fresh processes in ABBA order, final scalar scheduling enabled, no image input.
Text-only: 448.78/433.10 tps. Vision-loaded: 447.46/436.92 tps.
All 129280 final logits match exactly across the four runs. This does not
reproduce a vision-specific text prefill regression. The actual agent test
also successfully saved its text-only system prompt with vision loaded.
Sessions containing images remain outside that text-only save path.

## First-round outcome

Six separate experiments, one accepted default performance change (scalar
submission, commit 24914d66), and rebuilt ds4-agent/ds4-bench. The rejected
whole-tail change is not in active source; deeper prefetch remains opt-in and
is not recommended. `git diff --check` passed. Both vision-loaded and unloaded
32-token full-logit/snapshot scalar scheduling regressions passed.

The strongest remaining lead is unaligned prompt tails. Folding them into
batches was dramatically faster but not numerically equivalent. Resolve that
arithmetic/state discrepancy before adopting different partitions; do not
interpret a matching argmax alone as correctness. Reproducing the originally
reported vision slowdown still needs its prompt/runtime conditions: these
controlled text-only tests already show parity with the encoder loaded.

## 07: locate existing scalar/batch divergence before changing tail scheduling

Added a diagnostic-only build (`make tests/trace_deepseek41_prefill`) that
captures the last row of intermediate tensors. Trace barriers and file I/O are
compiled out of normal builds. `trace_deepseek41_prefill MODEL PROMPT 32
scalar|batch OUTDIR` runs the existing scalar or batch path directly.
`compare_trace.py CONTROL_DIR CANDIDATE_DIR` compares every captured vector.

At layer 0 of a fresh 32-token prompt, normalized inputs are bit-identical.
Mixing coefficients already differ (max 2.33e-5). Q projection has 8208/32768
differences after BF16 rounding; normalized Q low-rank and K projections also
differ. This predates wide-tail scheduling and does not depend on long-context
cache state. The batch dense paths use different arithmetic/half intermediates.
Detailed stage comparisons are in trace32/scalar-vs-batch.jsonl.

## 08: isolate the arithmetic differences and recover exact layer parity

`DS4_METAL_V41_EXACT_DENSE_PREFILL=1` expands existing exact Q8/F16 row
projections to larger batches and retains scalar grouped attention output.
This removes layer-0 HC mixing and Q/K differences exactly. Attention heads
still differ in 7/32768 elements. Adding `DS4_METAL_DISABLE_V41_BATCH_CORE=1`
removes those differences; all inputs to the routed FFN then match exactly.
Router IDs/weights and shared-expert output match; routed output differs by
up to 4.22e-5 before BF16, producing 192 residual differences.

Adding `DS4_METAL_DISABLE_V41_BATCH_MOE=1` gives bit-identical final residuals
at all 40 layers in the 32-token trace. All other captured vectors also match.
The trace comparison reports absent intermediate captures for the unfused HC
path; the complete residual comparison is present for every layer. These
switches are diagnostic, not default performance changes. This establishes a
scalar-equivalent layer-major path worth testing for formerly scalar tails.

## 09: preserve the original prefix and batch only its scalar tail

Opt-in `DS4_METAL_V41_EXACT_SHORT_PREFILL=1` admits warm SSD tails of
256-1023 tokens to a separate layer-major sweep, using exact row projections,
scalar attention and scalar routed experts. The preceding batch partitions
stay unchanged. This targets single-GPU, non-quality, non-imatrix execution
with at least half the experts cached. The exact arithmetic scope is thread-local
and restored even if the sweep fails.

`run_abba.py /tmp/ds41-exact-short5000 --candidate-env
DS4_METAL_V41_EXACT_SHORT_PREFILL=1 --tokens 5000 --ctx 100000`:
representative control 70.65 tps, candidate 93.99/88.75 tps (26-33% faster).
All full-vocabulary frontier logits match exactly. The first control was an
outlier at 14.31 tps; do not use it to claim a sixfold improvement. No competing
model process or thermal warning was found, but its cause is not established.
Raw result rows, including the outlier, are in exact-short5000.json.

The sweep changes expert-cache residency: candidate first decode is about
1.2 seconds versus 58 ms for control, and the following seven tokens run at
12.9-13.2 versus 15.5 tps. This cost must be included when choosing a default
threshold. Smaller-tail timing and full serialized continuation-state checks
follow before enabling this path by default.

Regression `tests/test_deepseek41_exact_tail MODEL PROMPT` passed: 257- and
513-token tails after a restored 4096-token prefix, each followed by eight
teacher-forced decode steps, produce bit-identical complete serialized
snapshots (KV state, hidden state, history, and logits).

## 10: exact hyperconnection batching (opt-in; not a clear default win)

`DS4_METAL_V41_EXACT_BATCH_HC=1` allows batched HC mixing around scalar
attention and MoE in the exact-tail path. The exact dense arithmetic switch
also prevents the fused HC projection from changing precision. Scalar MoE
writes the batch block buffer, followed by one batched HC expansion.

Both 257- and 513-token restored-prefix continuation snapshots are bit-identical
with this switch. Their within-process ctx=8192 timings were worse than scalar
(24.0 vs 13.9 seconds and 38.6 vs 26.7 seconds), so this is not enabled by default.
The fresh-process ctx=100000 ABBA comparison is recorded in exact-hc5000.json.
These context allocations have different expert-cache budgets; results from
one configuration do not establish a win in the other.

The baseline exact-tail path (without HC batching) did improve 513-token
appends at ctx=100000 in all four separate-process runs: scalar 17.81/17.38 tps,
exact sweep 22.49/21.86 tps. All full-vocabulary frontier logits match;
see exact-short513.json.

The runner now copies the executable into each experiment directory before
starting, preventing a rebuild from changing the binary between A/B samples.
It also accepts --gen-tokens for longer continuation measurements.

## 11: reuse eight-row scalar-equivalent expert kernels

`DS4_METAL_V41_EXACT_MOE_TILES=1`, together with the exact-tail opt-in,
tiles routed/shared FFN work into at most eight rows. It reuses the backend's
existing DS4.1 decode-batch kernels, which retain F32 intermediate values and
scalar reduction order; it does not enable the large-batch grouped matmul.
Token-ID views advance with each tile. HC and attention remain scalar unless
separately requested. The final partial tile is covered by odd-length cases.

At ctx=8192, 257- and 513-token restored-prefix tails plus eight continuation
tokens have byte-identical complete snapshots. The test now checks that the
candidate actually uses a layer-major sweep and the control stays scalar.
Within-process timings (24.8 vs 14.0 seconds and 38.6 vs 25.9 seconds) again
argue against enabling short SSD sweeps at every tail length.
The ctx=100000, 5000-token fresh-process comparison includes 32 teacher-forced
decode tokens per run and is retained in exact-tiles5000.json.

Result: exact sweep control 89.36/94.57 tps; eight-row tiles 83.18/81.14 tps.
All final logits match. Tiles reduce first decode from 1.07-1.12 seconds to
193-205 ms, but the prefill regression is larger than that saving. Keep tiles
opt-in. The non-tiled sweep reaches 14.9-15.1 tps over the next 31 decode tokens.

## 12: enable the measured long-tail regime by default

The default now uses the exact layer-major sweep for warm text-only SSD tails
of 768-1023 tokens when the prefill buffer holds 8192 rows and at least half the
experts are cached. It remains single-GPU, non-quality and non-imatrix only.
This covers the agent's default 100000-token allocation. Shorter tails and
smaller buffer configurations retain the previous dispatch because the
measurements did not establish a consistent win there. Sessions containing
image spans are excluded pending the deferred vision work.

Rollback: `DS4_METAL_DISABLE_V41_EXACT_SHORT_PREFILL=1`.
The old `DS4_METAL_V41_EXACT_SHORT_PREFILL=1` remains a diagnostic force flag
for 256-1023-token comparisons; rollback takes precedence. HC batching and
eight-row expert tiles remain opt-in, independently of the accepted sweep.

Current-default comparison:
`run_abba.py /tmp/ds41-default-tail5000 --candidate-env
DS4_METAL_DISABLE_V41_EXACT_SHORT_PREFILL=1 --tokens 5000 --ctx 100000
--gen-tokens 32`. Here **control is the new default; candidate is rollback**.
Earlier opt-in A/B commands should be run at their experiment revisions,
because unsetting the force flag no longer disables the accepted default.

`make -j2 ds4-bench ds4-agent tests/test_deepseek41_exact_tail
 tests/test_deepseek41_scalar_queue` completed without warnings. The exact-tail
regression now checks the actual default against rollback at 768 and 1023
appended tokens, including explicit batch-dispatch activation and eight
continuation steps. Diagnostic build prerequisites and clean targets also
include the new test tools correctly.

Both default-boundary regressions passed with byte-identical full snapshots:
768-token tail 44.0 -> 35.2 seconds; 1023-token tail 60.7 -> 57.5 seconds.
These are within-process restored-prefix timings, not the separate-process
ABBA estimate. See default-tail-state.txt for the exact command and output.

Final review also excludes the optional resident-encoder mode: its short
chunks were already batched and must keep their existing arithmetic. The
normal wide-prefill path used by these measurements has no resident encoder.

Final ABBA: default **88.60/89.96 tps**, rollback **70.45/68.78 tps**:
**28.3% faster by aggregate prefill time** on this 5000-token prompt.
All full-vocabulary frontier logits match across the four fresh processes.
First decode after the sweep costs 1.00-1.20 seconds (rollback 57 ms);
the next 31 tokens run at 14.95-15.00 tps (rollback 15.52-15.61).
Including all 32 decoded tokens, average measured prefill-plus-decode time
falls from about 73.9 to 59.2 seconds, a 19.9% wall-time reduction.
See default-tail5000.json. The final resident-encoder exclusion does not
change this measured path: all profile rows report encoder_resident=0.

Second-round outcome: six more jj experiments, one new default admission
policy, exact logit and continuation-state checks, and rebuilt agent/benchmark.
The batching discrepancy is arithmetic: different dense/attention/expert
rounding, rather than a missing KV-cache update. The accepted path preserves
the old batch prefix and uses scalar-equivalent arithmetic for its long tail.
Vision reproduction remains deferred to the user's next keyboard session.

## 13: actual hello turns, fresh versus restored system KV

Added `DS4_AGENT_CACHE_DIR` as an optional agent cache-directory override.
The default remains ~/.ds4/kvcache. agent_probe.py snapshots the executable,
uses a private cache, sends three separate turns on the real noninteractive
readiness protocol, and limits output to one token per turn. This isolates
prefill latency; it is not a benchmark of complete natural-language answers.

`python3 speed-bench/experiments/ds41-ssd/agent_probe.py /tmp/ds41-agent-short-baseline2`
(no vision; default ctx=100000). System prompt 1859 tokens: 14.932 seconds.
After fresh system prefill, appends of 38/12/6 tokens took 3.527/0.981/0.379 s.
After loading saved system KV in a new process, the same appends took
4.954/1.203/0.509 s. The trace confirms a system KV hit. Repeated short turns
get faster as weights warm; restoring KV does not recreate that weight cache.
All KV files are isolated under the experiment directory. See agent-short-baseline.json.

## 14: smaller 64 GiB cache target (inconclusive; no default change)

`agent_probe.py /tmp/ds41-agent-cache64 --cache-gb 64` uses the existing explicit
cache budget. Fresh/restored 38-token hello took 3.435/4.659 seconds versus
baseline 3.527/4.954. Later turns were mixed (12 tokens 1.033/1.153 seconds;
6 tokens 0.455/0.482). This single comparison does not justify a cache-default
change; differences are small and need balanced repeats. See agent-cache64.json.

## 15: exact weighted normalization and BF16 fusion (opt-in)

`DS4_METAL_V41_FUSED_NORM=1` retains the original RMS reduction and learned
weight multiplication, rounding to BF16 in the same kernel. Forty synthetic
shape/alias cases match every output byte and buffer guards. The full-model
check (`test_deepseek41_ablation MODEL PROMPT ENV`) matches all 32 full-vocabulary
logit rows and the complete serialized continuation state.

Warmed synthetic 2000-call timing: separate ~9.96 ms, fused 6.79/6.83 ms; the
first separate pass was 19.96 ms and includes cold overhead (norm-kernel.txt).
The four-process 40-token append after 2048 tokens at ctx=100000 shows no
wall-time improvement: control 13.21/13.09 tps, fused 13.05/13.23. All frontier
logits match. Retained opt-in while measuring actual agent behavior; no default
change justified by this sample. See norm-abba40.json.

Actual agent probe with fusion: fresh/restored hello 3.614/5.027 seconds,
versus 3.527/4.954 in the initial baseline. No measured end-to-end gain.
The restored-process profile (three short turns plus three output tokens)
reports 2360 layer selections, 3.641 s selected synchronization, 1.674 s pread,
and 0.595 s read-ahead advice. Timing categories overlap. This points toward
submission/I/O scheduling rather than treating norm microbenchmark gains as
user-visible gains. See agent-norm.json.

## 16: start selected SSD loads before shared-expert GPU work (opt-in)

`DS4_METAL_V41_EARLY_EXPERT_LOAD=1` moves selected-ID synchronization before
shared-expert computation, starts the existing protected asynchronous cache
loader, submits shared kernels, then consumes the same IDs in routed MoE.
Only non-quality single-rank SSD streaming is changed. No routing prediction.

All 32 full logit rows and the complete serialized continuation match exactly.
Forty-token append ABBA at ctx=100000: control 13.08/12.93 tps; early load
13.52/13.24 (~2.9% mean throughput gain). All frontier logits match exactly.
Actual-agent balanced trials follow; this remains opt-in pending stronger
workload evidence. agent_abba.py fixes one executable across all eight agent
processes and tests both fresh and restored system KV in ABBA order.
See early-abba40.json.

Real-agent ABBA confirms early loading: mean fresh hello 3.498 to 3.369 s
(-3.7% latency), restored hello 5.005 to 4.709 s (-5.9%). Both candidates
beat both controls in each hello condition. Later turns are broadly improved,
with restored 12-token turns essentially flat. All runs use a fixed binary,
private KV stores, default ctx=100000, and one output token. See agent-early-abba.json.

## 17: omit selected-expert read-ahead advice (rejected)

Actual-agent ABBA with `DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD=1`:
mean fresh hello 3.585 to 3.633 s, restored hello 5.041 to 5.441 s (7.9% slower).
Both restored candidates were slower than both controls. Read-ahead overhead
is real, but omitting it makes subsequent reads sufficiently slower to lose
overall. Keep the existing default. See agent-noadvice-abba.json.

The benchmark harness now copies all external Metal sources and records their
SHA256 hashes, passing source overrides to every subprocess. This complements
the fixed executable and allows later development without changing a running
experiment's shaders.

## 18: exact Q8 matvec output and BF16 fusion (opt-in)

`DS4_METAL_V41_FUSED_Q8_BF16=1` rounds the final scalar Q8 reduction in its
output store, replacing a separate BF16 dispatch. The scalar reduction order
and explicit rounding are unchanged. Exact-row batches use the same epilogue;
large matrix batches keep their prior path.

All 48 synthetic shape cases match exact output bytes and output guards.
Warmed 1000-call 5120x5120 matrix timing: separate 33.241 ms, fused 31.624/31.680;
first separate pass 40.487 ms includes cold overhead. All 32 full-vocabulary
model logit rows and serialized continuation state match exactly.
Short-append ABBA pending; no default change yet. See q8-kernel.txt.

Q8 short-append ABBA is flat: control 13.22/12.80 tps, candidate 13.10/12.82,
all frontier logits identical. Keep the fusion opt-in; the microbenchmark
gain does not establish an end-to-end gain. See q8-abba40.json.

## 19: honor prefill chunk independently of maximum context

DS4.1 now honors explicit `--prefill-chunk` in graph allocation, admission,
and memory reporting, capped at the supported 8192 rows. The existing
context-derived default is unchanged. All call sites, including imatrix and
multiple-session admission, receive the engine setting.

At ctx=100000, chunk2048 reduces scratch from 8.01 to 3.76 GiB; the automatic
expert cache grows from 72.51 to 76.50 GiB. Total planned memory changes only
from 97.61 to 97.36 GiB, so this is primarily a budget redistribution.
Forty-token append ABBA: control 13.00/11.94 tps; chunk2048 13.11/12.75.
All full-vocabulary frontier logits match. Control drift makes the timing
inconclusive; actual-agent ABBA follows before considering a default change.
See chunk2048-abba40.json.

Actual-agent buffer ABBA shows no consistent short-turn gain: fresh hello
3.537 to 3.521 s, restored hello 5.038 to 5.101 s. Keep the default buffer cap;
the explicit chunk setting remains useful for controlling the memory budget.
See agent-chunk2048-abba.json.

## 20: tabulate Engram FP8/scale decoding (opt-in, tiny total savings)

`DS4_ENGRAM_DECODE_LUT=1` uses a process-wide 256 KiB table initialized once
from the original F32 scaling and BF16 rounding. Invalid encodings and
non-finite results retain EDOM behavior. All 65536 code/scale pairs match an
independent double-precision reference after the original rounding boundaries,
including signed zero and subnormals. The existing Engram suite passes.

Cached CPU row benchmark: 2000x24 rows takes 46.4-49.2 ms normally, 27.1-28.1 ms
with the table. That saves only about 0.02 ms per model token's two Engram
lookups. All 32 full model logit rows and serialized continuation bytes match.
Retain opt-in, with no claim of meaningful agent latency gain.
See engram-lut-cpu.txt and tests/test_engram_lut.c.

## 21: overlap selected-load preparation on the existing worker (opt-in)

`DS4_METAL_V41_ASYNC_EXPERT_LOAD=1` submits a routing-ready event and reuses
the established asynchronous selected-load worker while the main thread
encodes shared-expert kernels. It joins before routed MoE and retries on the
main thread when cache entries require waiting for in-flight GPU use. Error
paths join the worker before releasing session state. This takes precedence
over the earlier-load opt-in.

All 32 full model logit rows and the complete continuation snapshot match.
Actual-agent ABBA directly against early loading is pending.

Direct ABBA loses to early loading: mean fresh hello 3.355 to 3.422 s;
restored hello 4.684 to 4.893 s (4.5% slower). Keep the worker path opt-in.
The earlier simple loading schedule remains the default candidate.
See agent-async-vs-early.json.

## 22: small layer-major tiles with selected-expert streaming (opt-in)

`DS4_METAL_V41_SELECTED_SMALL_PREFILL=1` uses tiles of at most eight rows for
short warm appends. Dense projections and HC work are batched with exact
scalar reductions, while attention and routed/shared MoE retain scalar
arithmetic. The tile keeps selected-expert streaming and skips all full-layer
mapping, read-ahead, and cache seeding. It therefore avoids the full SSD sweep
that made earlier short-batch attempts unattractive.

Tails 6/8/17/40 after 512 tokens, each followed by eight decoded tokens, produce
bit-identical complete snapshots. Within-process 40-token timing improves
2.110 to 1.536 s, but cache warming favors the second run; balanced real-agent
ABBA against early loading alone follows. See selected-small-state.txt.

Actual-agent tiles versus early loading alone: fresh hello 3.346 to 3.425 s
(slightly worse), restored hello 4.691 to 4.513 s (3.8% better). Fresh/restored
12-token turns improve 0.922/1.177 to 0.787/1.033 s; 6-token turns improve
0.392/0.472 to 0.350/0.387 s. Tile size/policy needs care: initial cache misses
can outweigh the dense-kernel savings. See agent-small-vs-early.json.

The same four tail/continuation snapshot checks also pass after a 2048-token
prefix, exercising indexed attention. See selected-small-indexed-state.txt.

## 23: batch the small tile's selected experts too (opt-in)

`DS4_METAL_V41_SELECTED_SMALL_MOE=1`, with the small-prefill flag, enables
the existing exact <=8-row shared/routed kernels and passes force_resident=false
to the selected-address backend. Whole-layer prefills and multi-session callers
retain force_resident=true. Only actual selected weights enter the cache.

All four indexed-prefix tail/continuation snapshots match exactly. Warmed
40-token comparison: scalar 2.021 s, batched selected MoE 1.068 s (within-process
cache caveat applies). Actual-agent ABBA versus early loading alone follows.
See selected-moe-state.txt.

Actual-agent ABBA confirms selected MoE batching: mean fresh hello 3.438 to
2.873 s (16.4% lower latency than early loading alone), restored hello 4.717
to 4.465 s (5.3% lower). Later 12-token turns improve 0.889/1.138 to
0.696/0.952 s; 6-token turns 0.397/0.469 to 0.314/0.344 s. Both hello
candidates beat both controls. See agent-selected-moe.json.

## 24: accept early loading and exact selected-expert short batches

The defaults now start selected SSD loads before shared-expert GPU work and
use <=8-row exact tiles for short warm text appends. Selected expert batches
reuse the existing address kernels. A backend capability query checks expert
formats, cache capacity, pipeline availability, and backend ablation settings;
unsupported configurations retain scalar prefill. Quality, imatrix, image-bearing
and multi-rank sessions retain their previous path.

Rollback both changes with `DS4_METAL_DISABLE_V41_SHORT_OPTIMIZATIONS=1`.
`DS4_METAL_DISABLE_V41_SELECTED_SMALL_MOE=1` separately keeps scalar MoE
inside tiles for diagnosis. The earlier enable flags were experiment controls;
use the rollback flag to compare current defaults. Kernel norm/Q8 fusion,
Engram lookup conversion, and worker-based loads remain opt-in. Cache budget,
prefill-buffer defaults, and read-ahead advice remain unchanged.

Final real-agent ABBA (control=current default, candidate=rollback), fixed
binary and shaders, SSD streaming, ctx=100000, isolated caches, three short
turns with one output token each:

| Condition | Prior path | New default | Latency reduction |
| --- | ---: | ---: | ---: |
| Fresh hello, 38-token append | 3.566 s | 2.917 s | 18.2% |
| Restored system KV hello | 5.029 s | 4.417 s | 12.2% |
| Fresh 12-token turn | 0.962 s | 0.666 s | 30.8% |
| Restored 12-token turn | 1.209 s | 0.955 s | 21.0% |
| Fresh 6-token turn | 0.427 s | 0.323 s | 24.2% |
| Restored 6-token turn | 0.487 s | 0.347 s | 28.8% |

Both default samples beat both rollback samples in every short-turn condition.
Post-prefill to first output stays in the same ~60-110 ms range, with no hidden
second-scale first-decode penalty. System-prompt prefill itself is essentially
unchanged. See agent-final-abba.json and agent-final-metrics.json.

Final numerical regression: tails 2/3/4/5/6/7/8/9/17/40 after a 2048-token
prefix, followed by eight decoded tokens each, have bit-identical complete
continuation snapshots against rollback. See final-small-state.txt. The prior
768- and 1023-token tail sweeps, each followed by eight decoded tokens, also
remain byte-identical to scalar execution (final-long-tail-state.txt). Both
ds4-agent and ds4-bench are rebuilt; whitespace checks pass. The round ran from
16:12:48 to approximately 17:13 UTC on 2026-09-13, with one commit per experiment.

## 25: profile accepted defaults and expose bounded tile experiments

Round starts 2026-09-13 17:18 UTC. Real-agent profile at the accepted defaults
uses isolated fresh/restored system KV, one generated token per turn. Restored
hello is 4.394 s: 200 selected batch bindings total 2.250 s, including 1.809 s
of parallel pread. GPU busy accumulation reaches about 0.94 s before the next
ready marker. Fresh hello is 3.072 s with 0.594 s binding, including 0.551 s
pread. These categories are nested/overlapping, not additive independent stages.
See round3-profile.json; raw logs /tmp/ds41-round3-profile.

Add bounded diagnostic DS4_METAL_V41_SELECTED_TILE=2..8 and opt-in
DS4_METAL_V41_SELECTED_MEDIUM=1 to use exact small tiles below 768 remaining
tokens. Existing default stays below 256. The model-dependent snapshot test
accepts an optional single tail length for testing the medium gap. Benchmark
512-token appends at ctx=100000 after a 2048-token prefix next.

## 26: measure complete agent generations

Agent probes now accept --gen-tokens and --prompts-json, and record prefill-to-
first-output, total response, decode duration, generated count and output-ID
hash from the trace. Longer generations use seed 1234; the original one-token
probe behavior is preserved. ABBA forwards these settings and an optional fixed
expert-cache budget. Python compilation passes. Full-response trials follow
after the current medium-append benchmark; exact snapshots remain the stronger
numerical gate than matching sampled outputs.

## 27: exact eight-row tiles across medium appends

Balanced separate-process ABBA after a 2048-token prefix, ctx=100000, 512-token
append: controls 23.05/21.92 tps, selected-medium candidates 27.85/29.75 tps.
Mean latency falls from 22.785 to 17.797 s, about 21.9%; all full-vocabulary
frontier logits match exactly. See medium512-abba.json. The flag remains opt-in
pending full continuation snapshots and boundary tests.

## 28: larger exact selected-address tiles (opt-in)

Extend diagnostic DS4_METAL_V41_SELECTED_TILE to 2..32; default remains eight.
Selected-address routed kernels already preserve scalar arithmetic independently
of row count, and the exact-dense scope selects scalar-equivalent projections.
No grouped routed matmul is introduced. A 40-token tail with tile=32 after an
indexed 2048-token prefix, plus eight teacher-forced tokens, has a bit-identical
full continuation snapshot. Within-process scalar 3.068 s, tile32 0.959 s is a
correctness run with warm-cache bias, not an independent performance result.
Actual-agent ABBA is running at /tmp/ds41-agent-tile32.

Tile32 actual-agent ABBA: fresh hello 2.884 to 2.422 s (16.0% lower), restored
4.520 to 4.083 s (9.7% lower). Twelve-token turns 0.657/0.940 to 0.572/0.826 s.
Six-token turns use the same shape and are approximately flat amid run noise.
Both hello candidates beat both controls. See agent-tile32-abba.json. One-token
probes retain the old random seed behavior, so sampled output IDs are not a
correctness comparison; the expanded full-state test supplies that gate.

## 30: overlap selected-batch SSD binding with shared-expert GPU work

Opt-in DS4_METAL_V41_BATCH_SHARED_OVERLAP=1 completes the router first, then
queues shared-expert work. A scoped completed-router hint lets the selected
backend flush those commands without waiting, load selected expert bytes on
the CPU, and enqueue routed computation behind shared work. The hint is
consumed by the backend and cleared on every caller exit. Existing cache
in-flight lifetime protections remain active. No arithmetic or weight changes.

The expanded snapshot test accepts several explicit tail lengths in one process.
Combined tile32/medium/overlap checks have passed 12/16/24/31/32/33/64 plus eight
continuation tokens at a 2048-token indexed prefix; larger boundaries are still
running in /tmp/ds41-round3-expanded-state.txt. Performance gate follows.

## 31: parallel reads for small Engram batches (opt-in)

DS4_ENGRAM_SMALL_BATCH_PARALLEL=1 lowers the existing macOS concurrent-read
threshold from 256 requests to two tokens (48 requests). This reuses the same
16 readers, disjoint outputs, bounded allocation and error propagation. The
existing Engram hash/history/disk-row suite passes with the flag enabled.
No default change; benchmark after the expanded GPU snapshot test completes.

## 32: reuse Engram prefetch on small selected tiles (opt-in)

DS4_METAL_V41_SMALL_ENGRAM_PREFETCH=1 starts the existing table reader for
selected tiles too. Layer 0 overlaps table 0; layers 2-13 overlap table 1.
The existing allocated staging tensor, release/acquire row publication, thread
join, cancellation and copy path are reused. No additional admitted memory.
This is independent of the small-batch parallel-reader flag. Built successfully;
model snapshot and real-agent measurement follow the current expanded run.

Expanded tile32/medium/shared-overlap regression is complete: tails
12/16/24/31/32/33/64/127/128/255/256/511/512/767 after a 2048-token prefix,
each followed by eight decoded tokens, all have bit-identical complete snapshots.
See round3-expanded-state.txt. Timing within this single process is explicitly
not an ABBA performance claim.

## 34: probe selected tiles up to 128 rows and eviction pressure

Extend the diagnostic tile range to 2..128 while retaining the eight-row default.
The supported selected-address path has no grouped reduction switch at these
row counts. Full-model exactness still must be verified. The snapshot test now
accepts DS4_TEST_EXPERT_CACHE_GB=4..96 so the same continuation checks can force
cache reuse/eviction instead of only exercising the default 72.5 GiB cache.
No new runtime defaults. Build succeeds; tests are queued after overlap ABBA.
