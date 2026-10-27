# DeepInfer convergence experiments, 2026-10-27

The user authorized four sequential, initially one-hour experiments, with
DeepInfer as the preferred design reference unless Apple/SSD measurements or
constraints justify a difference. Inventory: [49-entry comparison](../../../docs/DEEPINFER_ALIGNMENT.md).
Reference upstream `553741557a57df44a46fb3ab72b6aafeaf945337`; fixed local runtime
`e2a38f6e2bf4`, audit parent `aeac7ed091cf`. Main remains unchanged.
Raw outputs live in ignored `.workspaces/deepinfer-alignment-20261027/`.

## 1. Candidate-only scalar index scoring

Completed12:50–13:33UTC within the initial one-hour budget. Implemented behind
`DS4_V41_CANDIDATE_INDEX=1`; default remains dense during qualification.
Layer20 chooses its unchanged 2,048 eight-key blocks. Source layers24/28/32/36
score only those keys, scattering scores to their original global IDs. Existing
filter/top-k and tie ordering stay unchanged. The initial implementation cleared
scores on GPU (a CPU clear would race queued work). Inspection then removed that
redundant pass: the existing filter already writes `-inf` for every excluded row.
The scorer writes only selected rows, then its caller applies that filter.
At <=16,384 keys the runtime keeps the original dense path.

Production-linked Metal fixture passes exact score/ID comparisons for finite
values and exact ties at37,8192,32768,32771,65536,262144 rows, including reverse
candidate order, partial eight-key tails and output canaries. Unchanged layer20
selection is covered by its existing fixtures; these checks target its readers.
Initial warmed alternating score + filter + top-k timings, eight repeats per point:

| Keys | Scored candidate keys | Dense ms | Candidate ms | Time reduction |
|---:|---:|---:|---:|---:|
| 8,192 | 8,192 | 0.788 | 0.823 | -4.5% |
| 32,768 | 16,384 | 1.011 | 0.780 | 22.8% |
| 65,536 | 16,384 | 1.151 | 0.590 | 48.7% |
| 262,144 | 16,384 | 2.683 | 0.750 | 72.1% |

After removing the redundant clear, the same fixture still passes. New matched
kernel pairs are0.625/0.918ms at8K,1.025/0.710ms at32K,1.391/0.679ms at64K,
and2.969/0.781ms at256K (dense/candidate). Independent sessions/runs vary, so
compare each matched pair rather than attributing all between-run change to
the simplification.

These are isolated scoring/selection timings, not complete token latency. The
8K point intentionally forces the all-candidate kernel to demonstrate why the
runtime bypasses it at short contexts. Prefill and batched/MTP verification are
unchanged by this scalar experiment.

Full-model dense/candidate/dense-repeat comparison completed at32K with the
1017 IQ2_XXS/Q2_K-imatrix model, normal reasoning, chunk8192 and48GiB expert cache.
The reused teacher fixture measures common token distributions and separately
generates greedy answers for ordinary, recent-record and changed-fact cases.
All three tasks pass in every arm. All22 teacher distributions are byte-identical
and every generated token, including reasoning, matches: measured ΔNLL=KL=0.
Candidate decode time is2.8%,2.8%,4.3% higher than the bracket mean for ordinary,
favorable and changed-fact cases, respectively. Closing dense controls are
9.3%,11.3%,12.5% slower than opening controls. This noisy bracket establishes
correctness, not a whole-model speed win. A64K bracket with the simpler scorer
completed; preserve this first result rather than replacing it.
Teacher targets are fixed answer tokens at the same reasoning frontier; they
measure distribution changes, not a standalone task-quality score.
The initial64GiB-cache attempt was stopped by the unchanged pressure guard on
macOS pressurelevel2, before producing a result. Its process exited;48GiB retry
starts from pressurelevel1. Preserve the failed attempt, not a timing sample.

At64K with the simpler scorer, all8 teacher rows and all27 generated tokens
are byte-identical. Whole-answer decode is3.224/3.257/3.289s for the
dense/candidate/dense-repeat bracket: candidate is0.03% slower than the control
mean, effectively neutral here. Most whole-answer cost remains elsewhere.

A short loop then restores **one identical in-memory64K checkpoint**, replays
the same27 tokens in alternating dense/candidate order and excludes the first
pair as warmup. Four measured pairs average1.208s dense versus1.188s candidate,
**1.68% less token-evaluation time**, with each pair improving1.0–2.2%.
Restore and snapshot time are outside that window. Repetition warms the expert
cache too: do not compare its1.2s directly to the first-answer3.2s or call this
a cold-request benefit. The useful win is small and applies to cache-resident
decode; retain the optimization opt-in, with no preset/default promotion yet.
The ordinary8K runtime gate is also checked through this loop:87 tokens average
3.663s dense versus3.670s with the flag on (0.17% variation on the same dense
path). No default/preset or installed executable is changed in this item.

Reproduce under the pressure guard (one GPU/model job at a time):

```sh
nu tools/zig-build metal-test decoder-replay-bench --prefix .workspaces/alignment-build -j6
python3 speed-bench/ideas/pressure_run.py .workspaces/alignment-gpu-guard -- .workspaces/alignment-build/bin/test-deepseek41-metal --index-candidates
python3 speed-bench/ideas/pressure_run.py .workspaces/alignment-model-guard -- nu speed-bench/experiments/deepinfer-alignment-20261027/candidate-index.nu .workspaces/alignment-build/bin/decoder-replay-bench .workspaces/alignment-model --tokens 32768
python3 speed-bench/experiments/deepinfer-alignment-20261027/compare.py .workspaces/alignment-model
# The same loop used to reduce first-answer cache/order effects:
python3 speed-bench/ideas/pressure_run.py .workspaces/alignment-loop-guard -- nu speed-bench/experiments/deepinfer-alignment-20261027/candidate-index.nu .workspaces/alignment-build/bin/decoder-replay-bench .workspaces/alignment-loop --tokens 65536 --case favorable --hot-loop
```

## 2. Block-only checkpoint and bounded attention recovery

Completed13:33–14:13UTC within its initial budget, baseline parent`56ebae95`. A direct
diagnostic API loads a private SWB1 file and replays128 tokens at their original
absolute positions. Raw visibility is clipped to the replay start, old
compressed/index publication is suppressed, and Engram's preceding token
history is retained. This first scalar prototype rebuilds all40 raw rings;
it combines encoder/decoder recovery rather than isolating the documented
DeepInfer encoder-only policy or implementing an optimized cache controller.
It requires an endpoint>=256 aligned to128, native single-GPU text, no drafter,
steering, archive, images or decoder-replay policy. Ordinary paths/defaults stay
unchanged; the direct diagnostic API is its only activation.

SWB1 v1 reuses the13-u32 dimensions/token-header layout with a distinct tag,
then exact tokens and F32 main/index owner arrays. It omits all40 raw rings and
saved logits. The probe slices its own exact checkpoint; canonical DSV4 loaders
reject SWB1. No general SSD-cache kind/controller is added. Recovered state is
marked approximate and excluded from canonical/suffix/replay export. At even
aligned endpoints pending C2 carries are absent/unused; odd endpoints remain
unqualified.

The2048-prefix qualification verifies unchanged cumulative blocks and unused
carry scratch. Ordinary recovery versus NaN-poisoned missing raw rings gives
byte-identical fresh frontier logits and all shared continuation logits. This
checks actual reconstruction/masking, not imported saved logits. Both answer
correctly. Verification overhead is excluded from measured arms.

At8192 prefix, actual files are63,464,500B exact versus52,461,620B blocks:
**11,002,880B (17.34%) smaller**. Every arm first prepares identical prefix
files; that setup is excluded. Files are reread without flushing the OS cache,
so these are warm-file restore measurements, not cold physical-SSD bandwidth.
All cases append the same256 uncached tokens and keep normal reasoning.

| Case | Exact restore, bracket mean | Block recovery | Full miss | Answer grades |
|---|---:|---:|---:|---:|
| Ordinary |21.19s|26.54s|41.98s|All pass|
| Recent fact inside replay tail |22.80s|29.45s|43.94s|All pass|
| Correction outside replay tail |19.31s|26.72s|41.67s|All pass|

Times include restore/recovery/recompute, append and the first free answer.
Across these tasks recovery is35.2% faster than a full miss and30.7% slower
than full-state hits. A256-token append itself takes~17s here. Do not attribute
that shared cost to recovery alone. Cache tokens are identical in every arm;
the correction was already present in the cached prefix. Editing cached prefix
tokens must miss rather than reuse its old blocks.

Quality uses two explicitly retained measurements. The old short forced teacher
begins with answer tokens at a reasoning frontier:22 rows give meanΔNLL+.193,
KL.0101, with per-caseΔNLL−.0139/+.3422/+.2286. That is a valid distribution
probe but poorly matches the model's normal continuation. The second comparison
forces the exact control's **actual reasoning and answer tokens plus EOS**:

| Case | Trace tokens | MeanΔNLL (nats/token) | KL | Teacher perplexity change |
|---|---:|---:|---:|---:|
| Ordinary |39|−.00425|.00550|−.42%|
| Recent fact |79|−.00033|.00550|−.03%|
| Correction |27|+.00102|.00799|+.10%|

Weighted meanΔNLL−.00113 and KL.00596; all free answers remain correct.
Recovery changes reasoning lengths (38→29,78→63,26→26 generated tokens), so
whole-answer timing includes that behavioral difference. Fresh misses and
repeated exact controls have byte-identical distributions/outputs. These three
retrieval tasks do not establish broad coding/reasoning quality equivalence.

The32K-prefix recent-fact check also passes every arm. Whole-answer times are
21.95/29.16/75.37/21.58s (exact/recover/miss/exact-repeat): recovery is61.3%
faster than a miss and34.0% slower than the exact bracket mean. Full checkpoint
220,849,204B versus block209,846,324B, a4.98% reduction; the omitted11MB is
constant, so its percentage shrinks with context. Its short forced teacher
givesΔNLL−.05993/KL.00679. This is an additional longer-context case, not a
new baseline reasoning-trace quality suite.

Keep full SWA for recent/exact hits; this supports exploring smaller cold
checkpoints with bounded recovery. Batch the128-row reconstruction and measure
real cache-retention benefit before wiring ordinary cache policy. The first
driver finished all model arms then failed its postprocessing `path self`
runtime use; fixed to a top-level Nushell const and analyzed retained outputs.
The subsequent trace driver completes cleanly. No model rerun was required
to repair reporting.

Reproduction (all model commands under the existing pressure guard):

```sh
nu tools/zig-build swa-restore-probe --prefix .workspaces/swa-build -j6
python3 speed-bench/ideas/pressure_run.py .workspaces/swa-poison-guard -- nu speed-bench/experiments/deepinfer-alignment-20261027/swa-restore.nu .workspaces/swa-build/bin/swa-restore-probe .workspaces/swa-poison --poison-check
python3 speed-bench/ideas/pressure_run.py .workspaces/swa-guard -- nu speed-bench/experiments/deepinfer-alignment-20261027/swa-restore.nu .workspaces/swa-build/bin/swa-restore-probe .workspaces/swa-measured
nu speed-bench/experiments/deepinfer-alignment-20261027/make-teachers.nu .workspaces/swa-measured/exact/answers.jsonl .workspaces/swa-teachers
python3 speed-bench/ideas/pressure_run.py .workspaces/swa-trace-guard -- nu speed-bench/experiments/deepinfer-alignment-20261027/swa-restore.nu .workspaces/swa-build/bin/swa-restore-probe .workspaces/swa-trace --teacher-dir .workspaces/swa-teachers
python3 speed-bench/experiments/deepinfer-alignment-20261027/plot-swa.py .workspaces/swa-measured/summary.json .workspaces/swa-figures
```

The last command needs Matplotlib (the existing local
`.workspaces/bench-plots-venv/bin/python` has it). Generated figures/raw data
remain local and ignored. The plot shows measured points and exact-control
ranges, with no invented interpolation/frontier.

## 3. Physical index-FP4 producer and reader

Completed14:14–14:24UTC, parent`0363cb40`. One local row stores64 nibble bytes
and4 UE8M0 scale bytes,68B versus512B F32 (**86.72% smaller**). Even lanes use
low nibbles; sign bit3 retains negative zero. This interleaved row is a Metal
pilot, not byte-compatible with DeepInfer's block-wide data/scales layout.
It preserves the values of our existing mode2 quantizer; that is lossless
storage relative to today's quantized values, not lossless quantization of the
original activations.

The producer reads original post-RoPE values and retains the original group32
scale and ties-even E2M1 code. Scales are not reconstructed from rounded output:
the fixture checks a6.03125 transition where that could choose a different
encoding. Dense and candidate readers decode into the same128-float shared tile
and keep the old32-head dot/reduction and global-ID order. Candidate readers
write selected rows, then the existing filter initializes excluded rows.
Concretely: BF16-RNE inputs; take per32 absolute maximum; choose the power-of-two
ceiling of`max(amax,6*2^-126)*(1.0f/6.0f)` using the existing rounded-F32
operations; choose nearest magnitude in`{0,.5,1,1.5,2,3,4,6}` with ties to an even
code index. Store sign in bit3 and the scale's biased exponent in its UE8M0
byte. Decode the signed code times scale, then BF16-RNE. This specifies the
numerical boundary independently of our harness.

The production-linked GPU fixture passes bit-exact roundtrip against the old
GPU quantizer, signed codes/zero, ties, exponent boundaries/tiny inputs,
nonzero offsets/canaries, dense/candidate scores and selected IDs. Shapes37,
8192,32768,32771,65536,262144 include partial eight-key tails. Warmed rotating
read chains (score+filter+top-k) and separate whole-array publication timings:

| Keys | F32 / packed MiB | Pack ms | Dense F32 / FP4 ms | Candidates F32 / FP4 ms |
|---:|---:|---:|---:|---:|
|8,192|4 / .531|.490|.691 / .682|.894 / .830|
|32,768|16 /2.125|.237|.689 / .712|.468 / .478|
|65,536|32 /4.250|.281|.791 / .897|.475 / .453|
|262,144|128 /17.000|.504|1.957 /2.151|.570 / .558|

The8K point selects every key and is not a reason to enable the sparse path.
Dense reads cost~3–13% more at32K–256K in this sample; candidate reads are near
neutral. Packing cost is for the entire array, not a one-row decode-publication
measurement. Preserve this capacity result without a whole-model speed claim.

One bounded variant constructed normal FP4 values directly from their F32
sign/exponent/mantissa bits, retaining old arithmetic for exceptional ranges.
All GPU checks still pass, but timings show no useful improvement (e.g.256K
dense FP4 2.168ms versus F32 1.960ms; candidate.579/.579ms). Reverted it and kept
the simpler arithmetic reader. Original and variant raw evidence/patch remain
local. No further micro-tuning was attempted in this slice.

Only explicit GPU APIs/fixture are added: no graph cache allocation, batched
prefill, payload, SSD format, model-default or preset change. Whole-model
quality and speed remain unmeasured. A next integration needs a packed→BF16
prefill consumer and canonical import/export conversion; do not leave a full
F32 shadow cache and call it a memory saving.

```sh
nu tools/zig-build metal-test --prefix .workspaces/index-fp4-build -j6
python3 speed-bench/ideas/pressure_run.py .workspaces/index-fp4-guard -- .workspaces/index-fp4-build/bin/test-deepseek41-metal --index-fp4
```

## 4. Useful DSpark history across canonical and SSD restore

Completed14:26–15:08UTC, parent`ed9210a0`. Enable save/restore with
`DS4_V41_CHECKPOINT_DRAFT_FEATURES=1` on a DSpark-enabled engine. Normal/default
configuration is unchanged. The canonical DSV4v2 base remains intact, followed
by one optional DSP1v1 trailer:

```text
8 little-endian u32 words:
  magic="DSP1", version=1, absolute_position=P, rows=min(P,128),
  hidden_width=5120, target_layer_0, target_layer_1, target_layer_2
then three chronological [rows,5120] arrays of little-endian BF16 bits
```

The target features were already rounded to BF16 at capture; their packing is
lossless. The existing GPU compact-carry kernel converts two ring segments per
stage rather than a CPU element loop. Restore rebuilds absolute-position tags
and leaves projected draft KV invalid; the existing seed operation reconstructs
it on the very next draft. This uses more stored bytes than projected-only KV,
but preserves the current readiness/rollback contract and avoids tying cached
target features to projection-weight identity.

Maximum extra bytes are3,932,192 (3.75MiB plus32B header). Old base payloads
remain accepted. Flag-off, missing-drafter or different-layer receivers consume
the valid appendix and retain target-only fallback. Old binaries reject extended
payloads. Private suffix/DRP1/SWB1 formats remain base-only. SSD admission,
manifest discovery and segment splitting accept DSP1. At equal native-prefix
length, lookup prefers useful DSP1 over an old base entry regardless of list
or restart directory order; it does not change the target-cache profile.

The native Zig installed-model oracle uses two sessions and a wrapped257-token
frontier. Base12,657,208B becomes16,589,400B. It publishes DSP1 then base, proves
DSP1 wins that tie, closes/reopens HOT0 cache, checks exact materialized bytes,
then restores it. **The first T1 constrained cycle** performs10 matcher masks,
proposes5/accepts4, commits6 tokens including correction, and matches scalar
teacher logits exactly (max absolute error0). Old-base and flag-off cases
perform no draft masks. Pure wire2/2, existing cache regressions and the GPU
BF16 compact-carry bit/bounds fixture pass.

Matched HTTP servers use the same1017 trunk/support files,32GiB expert cache,
ctx8192/chunk512, HOT0, no suffix reuse, full five-proposal DSpark verification
(`--dspark-confidence 0 --mtp-draft 6 --mtp-exact-sampling`). This is an eager
verification experiment, not a measurement of the default confidence gate.
Both servers pass7/7 complete ordinary/tool/schema checks plus streaming;
T0 reasoning/output content and token counts match exactly after excluding IDs.

| Cached request | Tokens | Base seconds | DSP1 seconds | Time change |
|---|---:|---:|---:|---:|
| Ordinary complete |260|17.855|18.552|+3.9%|
| Strict tool complete |169|10.212|9.167|−10.2%|
| Schema answer complete |183|11.301|10.337|−8.5%|
| Ordinary first32 |32|2.251|1.890|−16.0%|
| Ordinary first96 |96|6.079|5.383|−11.4%|

The short-cap probes switch to a different context first, then restore the
97-token prefix from SSD; both end as`incomplete` at the exact token limit.
Actual q verification cycles change0→8 and0→25, proving restored execution,
not just cache credits or feature flags. Full streams retain27 argument deltas
with exact final accumulation and correct literal marker strings/types. T1
outputs both pass, but have different lengths (167/181); no stochastic whole-time
speed comparison is claimed. These few samples establish a practical small
benefit on short restored replies and mixed whole-request results, not a broad
throughput improvement.

The three prompt endpoints occupy34,721,352B versus45,566,376B physical cache
files, including manifests/chunks. Actual DSP1 headers and extents are verified
for97/172/392-token endpoints. Keep the feature opt-in; compare the normal
confidence policy and projected-state alternatives before default promotion.

All host/script regressions pass, all five final executables build and exit0
on help. Qualified builds stay in the ignored item4/final-build folder; root
executables and main are not advanced. Both owned benchmark servers exit0;
the user's separate reference Flint service is left running.

```sh
nu tools/zig-build ds4-server dspark-checkpoint-test draft-checkpoint-wire-test metal-test --prefix .workspaces/draft-build -j6
python3 speed-bench/ideas/pressure_run.py .workspaces/draft-oracle-guard -- .workspaces/draft-build/bin/test-dspark-checkpoint gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf gguf/DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf .workspaces/draft-oracle-cache
# Foreground server in one terminal; use a fresh output directory per arm:
python3 speed-bench/ideas/pressure_run.py .workspaces/draft-off-guard -- nu speed-bench/experiments/deepinfer-alignment-20261027/serve-draft-checkpoint.nu .workspaces/draft-build/bin/ds4-server .workspaces/draft-http/http-off --features 0 --port 18341
# In another terminal, with that server ready:
nu speed-bench/experiments/constrained-packets-20261027/qualify.nu http://127.0.0.1:18341 deepseek-v4.1-flash .workspaces/draft-http/http-off/http --family deepseek --repeats 1 --stream
nu speed-bench/experiments/deepinfer-alignment-20261027/cached-caps.nu http://127.0.0.1:18341 .workspaces/draft-http/http-off/http .workspaces/draft-http/http-off/caps-ordinary
```

Terminate the owned server cleanly before the second arm. Repeat with fresh
`http-on`, features1 and port18342, then
`python3 speed-bench/experiments/deepinfer-alignment-20261027/analyze-draft-http.py .workspaces/draft-http`.
The analysis joins actual requests/usage to verification phases and checks
native SSD trailer bytes. Raw output remains ignored.

## Concrete follow-ups

The original strict-tool96-token cap exposes a length-finalization issue:
cutting inside DSML returns`failed` instead of typed incomplete output. Its raw
request/response is preserved separately; it is not counted as a successful
cap measurement. Complete strict-tool responses pass. A coherent repair must
retain partial typed name/argument bytes and already-streamed IDs, keep finish
`length`, and avoid replaying/canonicalizing incomplete calls. Merely suppressing
the parser error would leak raw DSML as assistant text. Repair this separately
before returning to larger kernel/cache integrations.

Explicit SIMD remains an option, as the user reminded us. The discarded bit-only
FP4 variant did not change lane/work mapping; vectorized multi-code loads and
scale broadcasts remain untested. Current scorers already use float4 dots,
SIMD sums and shuffles. Do not interpret the rejected scalar arithmetic change
as evidence against SIMD layouts.
