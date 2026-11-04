# Physical FP4 storage for DeepSeek V4.1 main KV

Status: **qualified and enabled in the opt-in DeepSeek fast presets**. Both
five-pair campaigns and the matching diagnostic profile are complete. The 8K result is latency-neutral;
32K has a repeatable **+1.98% task-time cost**, with much smaller KV/payloads.
Across both campaigns, all 40 answers pass, generated IDs match, and all 150
matched teacher rows are byte-identical. See the [8K result](RESULTS_8K.md) and
[32K result](RESULTS_32K.md) for every sample, phase, order effect and byte count.
Canonical checkpoint/MTP and the forward suffix chain pass at
`907543ca`; reverse/F16/fixed-source suffix cases pass at `aee06ac9` after a
suffix-only command-batch fix. The real legacy conversion bug found after the
initial `c54f8a7f` leaf tests is fixed, with an expanded GPU fixture passing.
Actual v2↔v4/v5 roundtrips, scalar decode, speculative verification and HOT0
SSD reopen pass below. The measured 32K cost is accepted for capacity and
persistence. Ordinary engine defaults and main bookmark remain unchanged;
final installation completed with runtime `f74180e0` in the 2026-10-28
convergence batch. See the [project journal](../../../docs/lab/README.md) for that
completed installation and the [final runtime qualification](../context-capture-20261028/QUALIFICATION.md#fresh-rebuild-after-approximate-state).

## Mechanism and bytes

The established main compressed-KV path already quantizes each group of 16
values into signed E2M1 FP4 with an E4M3 scale, then stores the decoded values
as F32. This experiment retains those codes and scales physically. It changes
KV storage, rather than model weights or the intended quantization policy.

| One 512-value main row | Code bytes | Scale bytes | Total |
|---|---:|---:|---:|
| Expanded F32 | — | — | 2,048 |
| Existing F16 storage | — | — | 1,024 |
| Native FP4, 32 groups of 16 | 256 | 32 | 288 |

Main rows are **85.9375% smaller than F32** (1,760 bytes saved per row).
That percentage does not describe total process memory: raw rings, index KV,
pre-RoPE originals, attention scratch, model weights and expert cache are
separate owners. At ctx=2048, actual allocation logs show main owners
**10,493,952 → 1,475,712 bytes**, across 5,124 allocated rows including spare
capacity. These counters are from a correctness diagnostic, not a latency run.

Fresh scalar and batched publication preserve the original quantizer scale.
Attention reads codes directly into the required selected/tiled scratch; there
is **no persistent full-sized F32 shadow**. Default main-FP4 scalar attention
keeps the existing F32 scratch/attention algorithm. The half-scratch/M5 variant
remains a separate explicit `DS4_METAL_V41_PACKED_F16_GATHER=1` experiment;
existing F16 defaults remain unchanged.

The engine opt-in is exact `DS4_V41_COMPRESSED_KV_FP4=1`. DeepSeek fast presets
now supply 1 unless only `DS4_V41_COMPRESSED_KV_F16=1` is requested; explicit
main 0 disables packing and explicit main 1 wins over F16. Four fake-binary
wrapper checks cover these combinations. Source lives in the DeepSeek
[attention owner](../../../src/engine/deepseek41/attention.zig), batch
publication paths and [Metal KV kernels](../../../src/metal/dsv4_kv.metal).

## Persistence and suffix representation

The existing 13-word header, token offsets and numerical span order remain.
The [typed format](../../../src/engine/deepseek41/state_format.zig) adds two
versions, independently selecting main and index storage:

| Canonical version | Main rows | Index rows |
|---:|---|---|
| 2 | F32, 2,048 B | F32, 512 B |
| 3 | F32, 2,048 B | FP4, 68 B |
| 4, new | FP4, 288 B | F32, 512 B |
| 5, new | FP4, 288 B | FP4, 68 B |

New readers accept all four. Same-format saves copy native bytes; mismatched
F32/F16/FP4 owners convert through bounded tiles using the existing 8 MiB I/O
buffer. They do not create a second graph-sized expanded cache. SSD manifest
validation and chunk splitting understand versions 4/5; the outer SSD format
does not need replacement. Canonical DSP1 draft-feature appendices retain
their existing optional format.

[Suffix capture](../../../src/engine/deepseek41/suffix_transaction.zig) also
retains physical main/index rows, including private approximate DSSD captures.
The pre-RoPE BF16 original tape remains separate and immutable. Relocation
decodes a bounded tile, performs the existing rotary repair from originals
where available, and writes the destination representation. Approximate
captures retain their private provenance and cannot become canonical prefixes.
These existing suffix operations require text-only, non-TP sessions without
DSpark/MTP; canonical MTP checkpoint restore is a separate supported path.

| Qualification area | Current evidence |
|---|---|
| Physical owners; fresh pack/unpack; selected F32/F16 gathers | Post-fix expanded GPU fixture passes, matching established GPU mode3 values; signed zero, tails and offset guards covered |
| Typed v2/v3/v4/v5 dimensions, SSD segmentation and parser checks | Initial CPU qualification passes |
| Legacy decoded-row repacker | Post-fix GPU fixture passes the captured checkpoint corpus and all seven subnormal scales with max 4/max 6, including signed zero |
| Canonical F32 v2↔v4/v5 numerical roundtrip | Real model passes; pre-MTP roundtrip preserves every canonical byte |
| DSP1, first speculative verification, both-direction import, HOT0 SSD restart | Actual v2/v4 and v2/v5 matrices pass at `907543ca`; scalar one-token max 0, first verification max 0, reverse/import checks max 2.86e-6 |
| Responses HTTP restart across main storage formats | Actual server v5/FP4→fresh F32 server passes at `aee06ac9`; 392/392 input tokens cached, no rewrite, identical output/reasoning/tool arguments |
| Pre-RoPE suffix save/restart, left/right shifts and repeated recapture | All five main-format cases below pass all four moves; packed-index→F32 and private DSSD v2/v3/v5 covered |
| Three 8K cases and favorable 32K speed/quality | Five pairs each complete: 8K neutral latency; 32K +1.98% task time, slower all five; all 40 answers and 150 paired teacher rows exact |
| Long-source admission | Direct 32K inference passes; 32K/200K admission with pre-RoPE tape remains a separate pending smoke; arithmetic below is a size calculation |

## Diagnostic and correction history

The first GPU fixture found one CPU-reference versus established-GPU halfway
tie. The packed fresh path reproduced the established GPU bytes exactly.
That pre-existing rounding difference remains recorded; it did not justify
changing the baseline quantization math.

Real checkpoint testing exposed a different, actionable bug. An old decoded
group can have a coarse subnormal E4M3 scale and maximum FP4 code 4, not 6.
In owner 1, row 0, group 464, the scale was 2^-8. Reconstructing the scale from
`amax/6` changed **0.015625 → 0.01171875**. The synthetic fixture had missed
this group shape. Main-only, packed-index+main and forced-F32-gather diagnostics
all showed a first-cycle teacher difference around 0.537. A canonical
v2→v4→v2 comparison reproduced the byte difference **before any MTP cycle**,
locating the problem in legacy conversion rather than speculative decoding.
Fresh publication already retains its original scale and was not implicated.

The correction separates fresh quantization from legacy repacking. The latter
validates all 16 decoded values using the usual `amax/6` scale, then falls back
to `amax/4` for coarse subnormal groups. The expanded GPU fixture now passes
the captured checkpoint corpus plus all seven subnormal scale codes with
max 4/max 6 and signed zero. It preserves the decoded values needed for old
F32/F16 owners. The canonical matrix directly exercises F32 wire cross-import;
the separate suffix matrix also covers live F16 owners.

Post-fix actual model qualification at `907543ca` passes both v2↔v4 and
v2↔v5. In each, the pre-MTP roundtrip restores every canonical byte; a scalar
one-token decode has maximum logit difference 0. The first speculative cycle
also has maximum teacher difference 0, with ten masks and six committed tokens.
Reverse/import checks differ by at most 2.861e-6 and pass the existing oracle.
HOT0 cache publication, close/reopen and both-direction restore pass.

These same-prefix canonical sizes include DSP1 and other state, rather than
only main rows:

| Actual prefix 263 | Canonical bytes | Saved versus v2 |
|---|---:|---:|
| v2, expanded main/index | 16,627,824 | — |
| v4, packed main | 15,473,264 | 1,154,560 |
| v5, packed main/index | 15,182,000 | 1,445,824 |

Raw diagnostics and frozen artifacts remain ignored under
`.workspaces/main-fp4-20261028/`: notably `qualified-leaf.log`,
`checkpoint-roundtrip.log`, `checkpoint-main-only.log`,
`checkpoint-f32-gather.log`, `checkpoint-both.log` and the passing
`checkpoint-index-control.log`. Post-fix evidence is in
`legacy-fix-leaf.log`, `qualification/checkpoint-main.log`,
`qualification/checkpoint-both.log` and `qualification/manifest.json`.
Failed attempts are retained separately; correctness checks are not latency
samples and establish no speed result.

Suffix qualification subsequently found a separate integration bug. The
F32→FP4 chain passed all four moves, but the reverse FP4→F32 path reached a
tensor blit without an open caller command batch. The documented blit contract
requires one. A local begin/copy/end helper now covers both main and index
F32 receivers; it changes no codec or numerical policy. The rerun also selects
source/destination index formats explicitly, covering packed-index→F32 and
old private DSSD v2 output. The corrected reverse/F16/fixed-source rerun exits
successfully at `aee06ac9`; the earlier failed reverse attempt remains under
`qualification/fp4-f32/`, not part of the passing matrix.

## Qualified suffix matrix and teacher quality

The [correctness driver](qualify.nu) uses pre-RoPE originals and four moves
with requested offsets −66,+128,−514,0. Chained cases recapture each relocated
source for the next move; the fixed-source case always reuses its original
source. Logs verify immutable source state/originals, private capture byte
accounting and recapture provenance. Each case passes all four moves:

| Main source → destination | Source policy | Recaptured private DSSD version | Revision |
|---|---|---:|---|
| F32 → FP4 | Chained | 5 | `907543ca` |
| FP4 → F32, including index | Chained | 2 | `aee06ac9` |
| F16 → FP4 | Chained | 5 | `aee06ac9` |
| FP4 → F16 | Chained | 3 | `aee06ac9` |
| FP4 → FP4 | Fixed source | 5 | `aee06ac9` |

Each `rows.csv` contains four forced continuation positions per move, 16 per
case. NLL/ΔNLL are nats/token; KL is fresh recomputation‖suffix graft. These
are narrow suffix-quality probes, not a broad reasoning benchmark. They
include the approximation from reusing activations after a prefix edit,
rather than isolating storage packing alone. The four chained format cases
have identical recorded teacher rows; no extra numerical difference is
observed between their storage conversions.

| Main source → destination | Native NLL | Graft NLL | ΔNLL | Mean KL | Max KL | Argmax flips |
|---|---:|---:|---:|---:|---:|---:|
| F32 → FP4 | 7.083676e-5 | 1.270864e-4 | +5.624964e-5 | 3.242040e-5 | 2.185781e-4 | 0/16 |
| FP4 → F32 | 7.083676e-5 | 1.270864e-4 | +5.624964e-5 | 3.242040e-5 | 2.185781e-4 | 0/16 |
| F16 → FP4 | 7.083676e-5 | 1.270864e-4 | +5.624964e-5 | 3.242040e-5 | 2.185781e-4 | 0/16 |
| FP4 → F16 | 7.083676e-5 | 1.270864e-4 | +5.624964e-5 | 3.242040e-5 | 2.185781e-4 | 0/16 |
| FP4 fixed source | 7.083676e-5 | 1.282362e-4 | +5.739943e-5 | 3.089252e-5 | 2.185781e-4 | 0/16 |

All 80 teacher rows retain the native argmax. None has byte-identical logits
to fresh recomputation; that distinction remains recorded. Evidence is under
`qualification/f32-fp4/` and the four `qualification-rest/` case directories.
The successful rest guard records child exit 0. These fixture timings are not
used for comparative performance claims.

## Actual Responses restart

The serving smoke at `aee06ac9` uses ordinary Metal confidence 0.6 and DSpark,
HOT0, the same SSD directory and no suffix reuse. The first server has physical
main FP4 enabled and stores native v5 at the 392-token prompt frontier. It exits
cleanly; a new server starts with main FP4 disabled, with its allocation log
confirming `main=f32`. The repeated strict-tool request completes with
**392/392 cached input tokens and 0 cache-write tokens**. Both runs generate
169 output tokens, 82 reasoning tokens and the same `echo` arguments. All
reasoning and assistant/tool content match after excluding generated IDs.
Both owned servers stop, and the guard exits 0. This is restart correctness,
not a comparative latency measurement.

Evidence stays under `.workspaces/main-fp4-20261028/server-restart/`:
`initial.response.json`, `restart.response.json`, both command records,
stderr and trace logs, and the actual SSD cache. The foreground
[server helper](serve.nu) reproduces the two launch configurations. Run each
under the pressure guard, POST the same strict-tool Responses request while
it is running, and cleanly terminate that owned server before the next launch:

```nu
python3 speed-bench/ideas/pressure_run.py INITIAL_GUARD -- nu speed-bench/experiments/main-fp4-20261028/serve.nu FROZEN/bin/ds4-server FRESH_OUTPUT --main-fp4 1
python3 speed-bench/ideas/pressure_run.py RESTART_GUARD -- nu speed-bench/experiments/main-fp4-20261028/serve.nu FROZEN/bin/ds4-server FRESH_OUTPUT --main-fp4 0 --restart
```

Regenerate the strict-tool fixture once, then POST that same complete body
while each foreground server is running:

```nu
nu speed-bench/experiments/constrained-packets-20261027/qualify.nu unused deepseek-v4.1-flash FIXTURES --family deepseek --dry-run
http post --content-type application/json http://127.0.0.1:18353/v1/responses (open FIXTURES/T0.0-strict-tool.request.json)
```

## Large-source sizing calculation

For this 1017 model, even prefix length N≥128, packed index and no DSP1,
canonical base bytes are `11,002,932 + 894*N` with packed main, versus
`11,002,932 + 5,294*N` with expanded main. The pre-RoPE tape is
`675,852 + 257*sum(floor(N/r))`, for r=[2,2,2,1]. These follow the actual row
formats and include tokens/raw state, rather than multiplying the 85.94%
main-row reduction by the whole checkpoint.

At **N=200,000**, calculated canonical state plus original tape is
**304.2 MiB versus 1,143.4 MiB**. Packed storage therefore fits the existing
512 MiB source-admission budget with an 8 MiB staging buffer. This is size
arithmetic, not a 200K model qualification, cache-retention or performance result.

## Performance reproduction

The [driver](run.nu) changes only main physical storage, using five fresh
processes per side in AB/BA/AB/BA/AB order. Packed index/SIMD stay enabled;
candidate scoring, HC fusion and compressed activation experiments stay off.
Reasoning stays enabled. The primary task time is answer prefill plus normal
generation; teacher prefill and forced continuation are reported separately.
Answer prefill has fresh KV with experts warmed by the preceding teacher branch.

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/main-fp4-20261028/five-guard -- nu speed-bench/experiments/main-fp4-20261028/run.nu FROZEN/bin/decoder-replay-bench .workspaces/main-fp4-20261028/five-samples
python3 speed-bench/experiments/main-fp4-20261028/analyze.py .workspaces/main-fp4-20261028/five-samples
```

The [analyzer](analyze.py) retains all samples, order variation, generated IDs
and text, answer checks, fixed-teacher delta-NLL/KL and logit equality, actual
allocation counters, engine-bound planned memory and canonical payload sizes.
Teacher counts come from `rows.csv`, not a fixed 80-row assumption; these are
short expected-answer tokens plus EOS, not a broad reasoning-quality benchmark.
Canonical payload sizes are serializer byte counts, not measured SSD traffic.
The completed three-case 8K campaign is local at
`.workspaces/main-fp4-20261028/full-8k/` and summarized in
[RESULTS_8K.md](RESULTS_8K.md). Aggregate answer prefill+generation is
75.601→74.768 s (−1.10% of means), but paired changes are −5.438%, −1.280%,
+0.844%, +0.791%, +0.005%, with median +0.005%. Treat latency as neutral rather
than using the mean alone as evidence of acceleration. Actual total KV falls
97,165,584→25,068,944 bytes at ctx=16384; canonical 8192-token payloads fall
54,371,380→18,326,580 bytes.

The [completed 32K favorable result](RESULTS_32K.md) is a real small latency
tradeoff: 47.600→48.540 s (+1.976%), slower all five pairs, median paired
+2.368%. Teacher/answer prefill costs rise 1.813%/2.257%, while generation is
−1.234%. Actual KV at ctx=40960 falls 227,172,624→46,941,584 bytes; canonical
32768-token state falls 184,476,724→40,297,524 bytes. All 10 answers complete
with the same 42 generated IDs and 40 exact teacher comparisons.

The [matching 32K profile](RESULTS_32K.md#matching-diagnostic-attribution) locates
the added work in indexed mixed batch attention: 188 equal dispatches take
17.615→19.524 diagnostic GPU seconds. Standalone pack+gather totals only 56.986 ms;
quantization/copy work decreases and expert read counts/bytes match. This points
to decoding inside the attention consumer without proving a hardware cause.
Profiled wall times are not additional comparative samples.

**Decision: enable main FP4 through `context-fast`, `resident-fast`,
`persistent-fast` and the `persistent-packed` alias**, accepting the measured
~2% 32K task cost for smaller live/canonical/suffix-source state. Honor explicit 0
and F16-only preference; keep ordinary engine defaults and default F32 scratch.
Optimizing that attention reader is a separate bounded experiment.

## Next reader experiment — proposed

The matching profile attributes most added GPU time to the existing dual
16-head indexed-attention consumer, where packed rows decode inside its
selected-row loop; explicit publication/gather timing omits that inner work.
Source inspection shows one row is decoded again for each query/head group,
and four neighboring half4 loaders repeat the same per-16 scale calculation.
This supports an inner-reader hypothesis, without distinguishing ALU cost,
lookup latency or occupancy; no assembly/counter study has established which.
See [the measured profile attribution](RESULTS_32K.md).

A bounded next probe would change only
[`dsv41_main_fp4_half4`](../../../src/metal/dsv4_kv.metal): construct E2M1
half-vector bits directly, decode one E4M3 scale per four adjacent vector lanes
and broadcast it with SIMD shuffle, then multiply exact half vectors. Keep the
attention accumulation/order, physical row layout and allocation unchanged.
The existing all-scale/signed-zero GPU fixture and five alternating 32K pairs
would qualify it. This is unimplemented and has no measured speed benefit;
current packed storage is an accepted capacity/latency tradeoff.

The [November 4 follow-up](../2026-11-04-packed-attention-reader/README.md)
tested this reader proposal and retained the original implementation. Values
matched, but whole-model and attention-operation comparisons did not establish
a useful gain. It also added focused coverage for the half4 attention reader;
the earlier scalar unpack/gather fixture did not exercise that helper.
