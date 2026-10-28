# Supplied-PRE HC collapse and RMSNorm fusion

Status: **corrected GPU fixture and five-pair full-model campaign pass;
whole-task latency is neutral.** This experiment changes one adjacent numerical
boundary in the DeepSeek V4.1 Metal path. It remains opt-in through exact
`DS4_METAL_V41_FUSED_HC_NORM=1`. No useful whole-task speed or capacity gain
is established; keep the feature opt-in pending a different measured use case.

The baseline is the qualified physical main-FP4 source `aee06ac9`. Its cache
format, expert routing and reasoning remain fixed in both performance arms.
The first HC-only frozen build, `71fc3030c624d5d765db810b1edab465f7f6a25b`,
passed its first random scalar comparison but failed signed-zero collapse.
That failed build and raw evidence remain separate from corrected frozen
revision `b3b6f079`, whose complete focused GPU fixture passes. Its five
alternating 8K pairs run under
`.workspaces/hc-collapse-norm-20261028/full-8k/`; completed results are below.

## The boundary being fused

For each token, four residual streams `R[h,d]` are collapsed using supplied
PRE coefficients `p[h]`, followed by the existing weighted RMSNorm:

```text
x[d] = BF16(sum_h R[h,d] * p[h])
scale = 1 / sqrt(sum_d x[d]^2 / width + eps)
norm[d] = BF16((x[d] * scale) * weight[d])
```

The sum starts from positive zero and accumulates channels in order. The
first BF16 boundary occurs **before** the norm reduction; the second occurs
after weighting. The implementation retains the existing float4/thread and
SIMD reduction tree rather than changing the norm to a different formula or
reduction order. Both `x` and `norm` are stored because existing diagnostics,
projections and feature consumers may still need the collapsed row.

Previously this boundary launched HC weighted sum, BF16 quantization,
weighted RMSNorm and final BF16 quantization separately. The candidate uses
one dispatch, keeps collapsed BF16 values in registers for the norm, and
removes intermediate rereads/rounding passes. It does not fuse the HC mixer,
attention, expert execution or residual expansion. Fewer launches and bytes
are a mechanism to measure, not proof of a whole-task gain.

## PRE belongs to the preceding half-layer

V4.1 computes a new HC split while applying an already available PRE. The
supplied PRE is therefore independent of the newly computed mixer output:

| Use | Residual input | PRE applied |
|---|---|---|
| Scalar attention input | Current residual | Carried previous FFN PRE (`graph.pre`) |
| Batched attention input, first layer | Current residual | Initial plain PRE |
| Batched attention input, later layers | Current residual | Preceding FFN split |
| FFN input | Post-attention residual | Current attention split |
| Output head | Final residual | Final FFN PRE |

The new kernel accepts the PRE tensor and its row stride explicitly. Plain
PRE rows have four F32 values, stride 16 bytes. A split row contains 24 F32
values, stride 96 bytes; its first four values are PRE. The kernel reads those
four coefficients without generating a replacement mix. Existing legacy HC
split/sum/norm kernels generate and immediately apply their new PRE, so they
are not a drop-in replacement for this V4.1 boundary.

## Kernel and dispatch

[The Metal kernel](../../../src/metal/dsv41.metal) handles one token per
threadgroup, with HC=4 and width up to 8,192. The host uses the existing RMSNorm
thread selection: a power of two up to 1,024 threads. Each thread retains at
most two float4 values, and the reduction uses 128 bytes of shared memory.
Each token's outputs and reduction storage are disjoint.

The [native host owner](../../../src/metal/hc_control.zig) validates the actual
input/output extents and norm-weight range, selects the pipeline and constructs
the packet. The [fixed Metal binding leaf](../../../src/metal/hc_control.inc.m)
binds residual, supplied PRE, F32 norm weights and both outputs. It uses the
existing command owner and encoder lifetime. No new tensor runtime, owner
cache or context-sized intermediate is introduced.

The corrected packet carries a runtime channel count, set to four by the host.
The accumulation loop has unrolling disabled to retain the old positive-zero
seed and channel order. It does not clear all output zero signs: that would
also change negative zero arising from genuine rounded arithmetic. The
existing BF16 and RMSNorm boundaries otherwise remain unchanged.

## Integration and hooks

[Ops.hcNorm](../../../src/engine/deepseek41/ops.zig) selects the fused kernel
only for flag `1`, HC=4 and the supported width. Flag `0` or absence retains
the existing weighted-sum/BF16/norm/BF16 operations. Single-row and row-batch
norm wrappers use the same weighted-norm owner at rows=1.

The migrated adjacent pairs are:

- [Scalar executor](../../../src/engine/deepseek41/executor.zig): before
  attention, after attention and final output head.
- [Prefill and decoder preparation](../../../src/engine/deepseek41/prefill.zig):
  ordinary batch inputs, post-attention inputs and bounded decoder-preparation
  rows. Scalar fallbacks enter the same executor helper.
- [DSpark drafter](../../../src/engine/deepseek41/dspark.zig): stage attention,
  FFN and final head normalization.
- [Session batching](../../../src/engine/deepseek41/session_batch.zig): layer
  bodies through the prefill helpers, plus each session's final output head.

Engram updates and residual/steering capture retain their existing positions
before collapse; residual steering remains in expansion. HC mix generation
still runs before the supplied-PRE collapse and writes a different split
buffer. The next FFN PRE is carried forward as before. No cache publication,
suffix provenance, speculative rollback or sampling policy is changed.

DSpark's feature-capture mean is deliberately **not** fused: it computes an
unweighted residual mean with no following RMSNorm. The presence of another
HC weighted sum alone does not make it this same boundary.

## First failed GPU fixture and correction

The first `--hc-collapse-norm` run at `71fc3030` reported:

```text
width=5120 rows=1 stride=16 pattern=0: collapse and norm bit differences 0
width=5120 rows=1 stride=16 pattern=1: collapse bit differences 2560
first differing value: old +0, fused -0
```

The established weighted-sum kernel has a runtime channel loop seeded with
positive zero. The candidate initially spelled out four vector products and
additions; its all-negative-zero terms retained a negative-zero result. The
source correction restores the established runtime accumulation form, rather
than changing the fixture's equality requirement. A separate source review
also found and migrated the session-batch output-head pair before measurement.

The focused fixture compares the old GPU sequence against the new one for
widths 5,120/8,192, row counts 1/5/17, PRE strides 16/96, random BF16 inputs
and signed zeros. It checks finite/BF16 outputs, exact bits for both collapsed
and normalized rows, immutable inputs and output guards. The initial failed
log stays under `.workspaces/hc-collapse-norm-20261028/leaf.log`; the guard's
child exit is 1. It is not a latency sample. The corrected run at `b3b6f079`
passes every width/row/stride/random/signed-zero case with zero bit differences
in both outputs, immutable inputs and intact guards. Its separate
`leaf-fixed.log` and `leaf-fixed-guard/` retain child exit 0. This validates the
primitive boundary; the completed campaign below separately checks model
behavior and latency.

Fused DSpark/MTP correctness also passes at combined source `24f86d2d`:
pre-cycle v2→v5→v2 canonical bytes are exact, scalar one-token and first-cycle
teacher maximum differences are 0, and reverse/HOT0 teacher checks differ by at
most 2.861e-6. Both-direction MTP and HOT0 SSD cache close/reopen pass with
guard child 0.
The separate evidence is `.workspaces/hc-collapse-norm-20261028/mtp.log` and
`mtp-guard/`. This is correctness qualification, not another latency sample or
a reason to change the neutral-performance promotion decision.

Ignored patch snapshots are in `.workspaces/hc-collapse-norm-20261028/`:
`runtime.patch` and `fixture.patch` are refreshed against `aee06ac9`;
`signed-zero-fix.patch` and `session-batch-head.patch` are incremental fixes
for the first HC child. The original patches and pre-fix source are retained.
The root job owns actual builds, frozen shader snapshots and model scheduling.

## Reproduce qualification and the comparison

Build the chosen corrected source into `FROZEN` with
`nu tools/zig-build metal-test decoder-replay-bench -j3 --prefix FROZEN`.
Copy that revision's `src/metal/*.metal` into `FROZEN/shaders/` and record its
revision in `FROZEN/source-revision.txt`. Keep the failed first build separate.
Run the focused fixture through the existing frozen-shader loader and pressure
guard before the model campaign:

```nu
python3 speed-bench/ideas/pressure_run.py HC_LEAF_GUARD -- nu speed-bench/experiments/main-fp4-20261028/with-shaders.nu FROZEN/bin/test-deepseek41-metal --hc-collapse-norm
```

The [maintained driver](run.nu) then runs five fresh processes per side in
AB/BA/AB/BA/AB order, changing only `DS4_METAL_V41_FUSED_HC_NORM=0/1`:

```nu
python3 speed-bench/ideas/pressure_run.py HC_PERF_GUARD -- nu speed-bench/experiments/hc-collapse-norm-20261028/run.nu FROZEN/bin/decoder-replay-bench HC_PERF_OUTPUT
python3 speed-bench/experiments/main-fp4-20261028/analyze.py HC_PERF_OUTPUT
```

Packed main/index storage and SIMD scoring stay on; F16 main, candidate-only
scoring and compressed-activation experiments stay off. Routing is native,
reasoning remains high, expert cache is configured at 48 GiB, context is
16,384 and chunk size is 8,192. Ordinary, favorable and changed-fact cases
remain distinct. The primary metric is answer prefill plus normal generation;
teacher prefill/continuation are reported separately. Initialization is
excluded; answer KV is fresh, while its experts follow the teacher branch.

The [shared analyzer](../main-fp4-20261028/analyze.py) reads the candidate arm
from the manifest and retains every sample, order effect, variation, practical
answers, generated IDs and fixed-teacher ΔNLL/KL. Its answer-position probes
are not a broad reasoning benchmark. Output differences are evidence to
interpret alongside speed, rather than an automatic rejection threshold.
The complete five-pair result below retains every sample and shows exact
observed behavior with neutral whole-task latency. These answer-position
probes do not establish broad task-quality equivalence.

## Completed five-pair result

Frozen revision `b3b6f079`, normal high reasoning, physical main/index FP4 fixed
in both arms, native routes, context=16384, prompt=8192, chunk=8192 and 48 GiB
configured expert cache. The unchanged pressure guard exits 0. A is control,
B is fused; all ten fresh processes in AB/BA/AB/BA/AB order are retained.
Initialization is excluded; both prefills have fresh KV, with experts warmed
by the preceding teacher branch before answer prefill. OS caches are not flushed.

**Recommendation: retain opt-in; no fast-preset promotion from this result.**
The primary three-case task time averages **73.483→73.556 s (+0.100%)**;
paired median is +0.138%, range −0.876% to +1.214%, sample SD 0.760 percentage
points. Treat this as neutral rather than a demonstrated regression or gain.
AB averages −0.168%, BA +0.501%, with only three/two pairs respectively.
Fewer launches are implemented and qualified, but this workload shows no
useful whole-task or capacity benefit that warrants another serving default.

### All five primary samples

Seconds, answer prefill plus normal-reasoning generation:

| Case | Arm | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 |
|---|---|---:|---:|---:|---:|---:|
| ordinary | control | 25.307071 | 25.970218 | 25.869836 | 25.593771 | 25.555888 |
| ordinary | fused | 26.523268 | 26.341289 | 25.623810 | 25.362596 | 25.318251 |
| favorable | control | 24.902333 | 24.172700 | 24.393679 | 24.231093 | 24.006868 |
| favorable | fused | 24.251982 | 24.650423 | 24.424893 | 24.281409 | 24.398438 |
| changed | control | 23.582305 | 23.776057 | 23.577570 | 23.047518 | 23.426873 |
| changed | fused | 23.118412 | 23.824416 | 23.145742 | 23.073386 | 23.444014 |
| all_cases | control | 73.791709 | 73.918975 | 73.841085 | 72.872382 | 72.989629 |
| all_cases | fused | 73.893662 | 74.816128 | 73.194445 | 72.717391 | 73.160703 |

| Case | Mean control→fused (s) | Median control→fused (s) | Mean-time change | Paired median | AB mean paired | BA mean paired |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | 25.659→25.834 | 25.594→25.624 | +0.680% | -0.903% | +0.975% | +0.263% |
| favorable | 24.341→24.401 | 24.231→24.398 | +0.247% | +0.208% | -0.284% | +1.092% |
| changed | 23.482→23.321 | 23.578→23.146 | -0.685% | +0.073% | -1.242% | +0.158% |
| all_cases | 73.483→73.556 | 73.792→73.194 | +0.100% | +0.138% | -0.168% | +0.501% |

| Aggregate primary | Mean (s) | Median (s) | Range (s) | Sample SD (s) |
|---|---:|---:|---:|---:|
| control | 73.483 | 73.792 | 72.872–73.919 | 0.507 |
| fused | 73.556 | 73.194 | 72.717–74.816 | 0.820 |

All five aggregate paired changes are `+0.138%, +1.214%, −0.876%, −0.213%,
+0.234%`; three are slower and two faster. No first sample or outlier is
excluded. The ordinary-case first fused prefill is longer and remains included.

### Phase changes

Aggregate seconds across the three cases in each process. The local summary
also retains each phase's five samples, median/range/SD and order groups.

| Phase | Mean control (s) | Mean fused (s) | Mean-time change |
|---|---:|---:|---:|
| Teacher prefill | 60.277 | 60.954 | +1.123% |
| Forced teacher continuation | 2.600 | 2.655 | +2.120% |
| Answer prefill | 61.839 | 61.949 | +0.178% |
| Normal-reasoning generation | 11.644 | 11.608 | -0.312% |
| Answer prefill + generation | 73.483 | 73.556 | +0.100% |
| Teacher prefill + continuation | 62.877 | 63.609 | +1.164% |
| All measured phases | 136.360 | 137.165 | +0.591% |

Ordinary-case generation alone is faster in all five pairs, by 1.143% of means;
the aggregate normal-generation change is only −0.312%. That narrow signal does
not become a measurable primary whole-task win across the three workloads.
This experiment stops at its bounded result rather than adding tuning branches.

### Exactness and memory

All 30 answers complete correctly. Paired arms and repeated controls preserve
all generated token IDs and final texts: 87/31/26 tokens,
`CODE=EMBER-7319` / `CODE=CEDAR-4826` / `CODE=LARCH-9062`.
Every one of 110 matched teacher rows is byte-identical across the full
vocabulary: ΔNLL=KL=0, no argmax flips. There are 7/8/7 forced positions per
case, including EOS, as recorded by `rows.csv`; this is not a broad
reasoning benchmark. Mean teacher NLL remains 1.416736628 / 1.153435699 /
1.298985372 nats/token in both arms.

All ten processes report the same main/index/raw allocation totals:
11,797,632 / 2,785,552 / 10,485,760 bytes, total 25,068,944 bytes.
Canonical prompt payloads are 18,326,580 bytes for every case, both arms.
Scratch 267.59 MiB, static context buffers 6740.67 MiB and runtime tensor peak
6206.64 MiB also match. Expert live capacity 40.88 GiB, 50,690 hits, 14,198 misses
and 56.97 GiB pread are unchanged; these I/O counters span all branches/cases.
There is no KV, scratch, checkpoint or expert-I/O saving in this fusion test.

The source flag remains available for further experiments. Raw files, the
exact frozen manifest and all detailed statistics stay ignored under
`.workspaces/hc-collapse-norm-20261028/full-8k/`, including `summary.json`.
Re-run its analysis without loading a model:

```nu
python3 speed-bench/experiments/main-fp4-20261028/analyze.py .workspaces/hc-collapse-norm-20261028/full-8k --output .workspaces/hc-collapse-norm-20261028/full-8k/summary.json
```

## Reference and scope

This targets a local fusion opportunity identified while comparing with
[DeepInfer's lazy mHC](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/hc.py#L174-L314)
and direct producer-to-consumer layouts; see the
[alignment inventory](../../../docs/DEEPINFER_ALIGNMENT.md). It retains
[antirez/ds4](https://github.com/antirez/ds4)'s local weighted-sum/norm math
and the rewritten engine's V4.1 PRE convention. This is one bounded Metal
boundary, not a port of DeepInfer's whole mega-mHC/MoE execution region.
