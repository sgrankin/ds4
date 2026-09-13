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

## Outcome

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
