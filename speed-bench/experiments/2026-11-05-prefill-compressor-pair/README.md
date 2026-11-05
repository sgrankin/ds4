# Batched compressor projection pairing

**Retained as an opt-in experiment.** This experiment asked whether pairing the ratio-two
compressor's two F16 projections improves long-context prefill while preserving
row-exact published keys. The focused projection fixture passed, and three fixed-history quality checks
produced identical measured logits and post-evaluation hashes. The diagnostic profile shows shorter local producer windows; profile-off
timings suggest a fresh-prefill benefit with substantial variation and no
reliable whole-task conclusion. The one-hour slice is November 5,
13:50–14:50 Eastern.

## Question and mechanism

During batched publication, the ratio-two compressor projects each normalized
input row twice: one set of F16 weights produces KV values and another produces
pooling scores. Pooling consumes both results along with previous compressor
state to form the latent rows used to publish attention keys. The two
projections read the same input but have separate weights and outputs. The
baseline issues separate row-exact projections; the candidate uses the existing
paired F16 reduction across multiple rows so that each pair can share activation
reads and a dispatch. Both independent dot products and their reductions must
retain the baseline's per-row numerical behavior.

The existing paired shader indexes input and output by row, but the baseline
host admits only one row and its launch uses a single row in the grid. This
experiment extends that path for a batch and selects it only for matching F16
shapes in streaming, single-host publication of 2–8192 rows. The exact-1
switch is `DS4_METAL_V41_PREFILL_COMPRESSOR_PAIR`. Quality, imatrix, image and
exact-dense-row modes retain the original calls, as do other declined cases;
a failure after pair admission propagates. The existing scalar pair policy
remains available. Multirow geometry follows the standalone row-exact F16 path,
including an NR1 specialization for M5.

Published keys are persistent state, not merely transient intermediates. Bulk
publication, scalar processing and suffix continuation must agree on each row's
keys: a difference can change subsequent attention and cache reuse even if the
immediate projections look close. Validation must therefore examine projection rows and continuation across
chunk and suffix boundaries. The focused fixture described below checks the
projection and pooling pieces; it does not establish full-model suffix coverage. The separate projection buffers and previous pooling state must
remain valid until their consumers finish; pairing does not change their
ownership or lifetime.

The selected baseline profile recorded 24 large standalone compressor KV and
score calls totaling 2.869 seconds in fresh prefill, across 12 publication
chunks. These profiled spans identify work to target; they do not imply that
2.869 seconds can be recovered. Weight reads and both reductions remain. The
already adopted index-query matrix path applies to transient queries, not these
published keys. The transient index-weight projection accounted for about 49 ms
in that profile; it is not this experiment's target. See the local
[initial target note](../../../.workspaces/hillclimb-20261104/38-prefill-target/initial-target-note.md)
and `baseline-phase-summary.json` beside it for profiling details.

## Setup

The pre-change source baseline is closure `d29c6bc7`, whose runtime source matches
`a31acf97`, with selected package 32. Deduplicated residency, unretained
command buffers and deferred set release are off. The restored baseline
benchmark binary has SHA-256
`7bb7e691e28dada2e49a48a5c33cf4c3d38186b6b5ad4ee24bbfbdbe48002b41`.
The candidate runtime revision is `5e8fafb5`. Its benchmark binary SHA-256 is
`f579099b36d498f1ed2cd71fa4634e8fc86e2cb4d992adbd55996b11ffaad881`
and fixture binary SHA-256 is
`4ff9c3be483b7a45e3bd7f90548de74842f92300c8d91b883fe3e65cc45f3b91`.
All measured comparisons use that same candidate build with the selector set
to `0` or `1`. The frozen pre-change binary is retained for provenance; it was
not used as a separate timing arm.
The workload uses DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix on an M5 Max
with 128 GiB memory and `speed-bench/promessi_sposi.txt`. The 64 GB expert-cache
request uses an 8192-token prefill chunk and 40960-token context allocation.
The saved `arguments.json` and `environment.json` for each run record the exact
benchmark arguments and settings. Fixed, diagnostic and
timeline settings and the baseline binary are retained locally under
`.workspaces/hillclimb-20261104/38-prefill-target/`.

The comparison uses five alternating profile-off samples per side at both 8K
and 32K. A separate intrusive profile examines the targeted projection work.
The existing pressure guard admits one heavy model job at a time. No preset or
installed binary change is implied by the experiment.

## Focused fixture

The guarded `test-deepseek41-metal --compressor-pair` fixture completed with
exit code 0. It compares both unrounded outputs with standalone row-exact
projections using two 5120 → 512 F16 weight banks at 1, 2, 7, 8, 31, 32, 33,
129 and 513 rows. Mixed partitions of 513 rows exercise nonzero tensor-view
offsets and transitions among scalar, few-row and larger batches. Output
sentinels and 72 bounded double-precision dot samples check writes and layout;
the command also runs a separate existing pooling check across eight chunk
sizes. It does not feed the paired outputs into pooling or compare published
cache state. The scalar comparison exercises the existing scalar pair policy, not a new scalar
selector or a cross-binary preservation claim. The fixture does not by itself
validate full-model publication or suffix continuation. The local
[source and fixture review](../../../.workspaces/hillclimb-20261104/38-prefill-target/source-review.md)
records the inspected gates, row geometry and independent buffer allocations.

```sh
./tools/zig-build ds4-bench metal-test -j4
./zig-out/bin/test-deepseek41-metal --compressor-pair
```

Build candidate revision `5e8fafb5` with its matching Metal sources, then run
the existing driver for five alternating pairs at each frontier. Both arms
use the same executable and sources:

```nu
let settings = 'speed-bench/experiments/2026-11-05-prefill-compressor-pair/candidate-settings.json'
for frontier in [8192 32768] {
    let out = $".workspaces/compressor-reproduction/($frontier)"
    python3 speed-bench/ideas/pressure_run.py $"($out)-guard" -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu $out timing --flag DS4_METAL_V41_PREFILL_COMPRESSOR_PAIR --frontier $frontier --tokens 128 --samples 5 --extra-settings $settings
}
```

The driver overrides the selected flag to `0` for baseline and `1` for candidate.
It clears inherited DS4 settings and records the resulting environment and
arguments. Use its `quality` stage at both frontiers for fixed-history checks,
plus `--cache 16GB` at 8K for cache stress; `timeline --tokens 17` reproduces
the separate intrusive 32K diagnostic. These stages are not timing samples.

## Fixed-history quality and resources

At 8K and 32K, each fixed-history comparison covers fresh prefill and a
restored 512-token append, and produced 256 byte-identical
full-vocabulary logit rows and 256 matching post-evaluation hashes between
baseline and candidate. A separate 8K case with a 16 GB expert cache produced
the same result. Delta-NLL, KL and top-1 changes were zero in all three cases.
These are fixed-history comparisons, not evidence of identical independently
generated trajectories or practical task performance. No new practical task
run has been completed. Fresh and append checks do not establish true suffix
relocation behavior.

Structural non-time counters matched between sides. The small-cache case
exercised 3,993 evictions and reuses, 163 mixed calls and five splits. Recorded
memory pressure was level 1 and swap remained at 29.003720703125 GiB. Raw
quality summaries and guard records are retained in the local experiment
folder.

Both timing guards completed without interruption. All 634 pressure samples at
32K were level 1, with swap ranging from 28.97021484375 to 29.00927734375 GiB.
All 370 samples at 8K were level 1, with swap stable at 28.970517578125 GiB.

## 32K profile-off timing

Five pairs alternate baseline-first and candidate-first. Each run includes a fresh 32K frontier, 128 teacher decode tokens, KV restore, 512-token append and 128 more teacher tokens. The expert cache survives restore; the OS cache was not flushed. Phase seconds are derived from two-decimal throughput and exclude loading and snapshot save/restore. Values are mean ± sample standard deviation.

| Phase | Baseline (s) | Candidate (s) |
| --- | ---: | ---: |
| Fresh prefill | 48.045 ± 4.028 | 47.069 ± 2.432 |
| Fresh decode | 3.805 ± 0.239 | 3.810 ± 0.168 |
| Warm append | 8.095 ± 0.723 | 8.008 ± 0.452 |
| Warm decode | 3.872 ± 0.185 | 4.006 ± 0.120 |
| Inference total | 63.816 ± 5.041 | 62.893 ± 3.057 |

All five phase samples follow as baseline / candidate seconds.

| Pair | First | Fresh prefill | Fresh decode | Warm append | Warm decode |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Baseline | 41.846 / 43.850 | 3.682 / 3.873 | 7.181 / 7.635 | 3.543 / 3.857 |
| 2 | Candidate | 47.465 / 46.243 | 3.666 / 3.706 | 7.952 / 7.754 | 3.959 / 3.959 |
| 3 | Baseline | 47.746 / 47.224 | 3.671 / 3.697 | 7.872 / 7.887 | 3.958 / 3.958 |
| 4 | Candidate | 52.298 / 47.429 | 4.224 / 3.697 | 9.164 / 7.981 | 3.968 / 4.110 |
| 5 | Baseline | 50.868 / 50.599 | 3.780 / 4.079 | 8.306 / 8.781 | 3.932 / 4.146 |

Fresh prefill was shorter in four pairs, but the unusually fast first baseline
and slow fourth baseline materially influence the mean. The first-arm split
also differs, so this is noisy evidence for the size of the gain. Warm decode
was slower or equal in every pair despite unchanged decode source; the timing
does not establish its cause. These measurements do not establish a general
end-to-end task speedup.

## 8K profile-off timing

The same five-pair procedure uses an 8K fresh frontier. Mean ± sample standard deviation and each paired phase sample are shown in seconds.

| Phase | Baseline (s) | Candidate (s) |
| --- | ---: | ---: |
| Fresh prefill | 20.373 ± 0.369 | 19.967 ± 0.391 |
| Fresh decode | 3.801 ± 0.055 | 3.889 ± 0.158 |
| Warm append | 7.968 ± 0.235 | 8.067 ± 0.433 |
| Warm decode | 3.862 ± 0.049 | 3.891 ± 0.107 |
| Inference total | 36.004 ± 0.224 | 35.814 ± 1.049 |

| Pair | First | Fresh prefill | Fresh decode | Warm append | Warm decode |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Baseline | 20.985 / 19.684 | 3.717 / 3.761 | 7.564 / 7.644 | 3.775 / 3.751 |
| 2 | Candidate | 20.020 / 19.443 | 3.778 / 3.733 | 7.965 / 7.699 | 3.879 / 3.858 |
| 3 | Baseline | 20.205 / 20.128 | 3.825 / 3.840 | 8.068 / 8.130 | 3.881 / 3.896 |
| 4 | Candidate | 20.244 / 20.186 | 3.858 / 4.023 | 8.128 / 8.142 | 3.894 / 3.896 |
| 5 | Baseline | 20.411 / 20.394 | 3.828 / 4.086 | 8.117 / 8.721 | 3.880 / 4.051 |

All five fresh prefills were shorter, although the first two dominate the
mean difference. Inference total improved in only two pairs and worsened in
the last three. Append and decode also show regressions even though the new
pair did not activate in those phases in the diagnostic trace. The comparison
does not establish their cause or a reliable whole-task benefit.

## Diagnostic profile

A separate intrusive 32K pair confirms activation: 24 standalone fresh
projection calls became 12 paired calls. Other normalized compute groups match;
fresh encoder records fell from 101,439 to 101,427 and command-buffer counts
were unchanged. The producer span sum fell from 2.727 to 0.984 s, but baseline
spans overlap. Summing each site's first-producer start through pool2 end gives
1.542 to 0.984 s, about 0.558 s less in this diagnostic, not an independent
five-sample timing claim. All 34 decode hashes matched.

The fresh GPU envelope was 44.891 to 43.441 s, while the append envelope was
9.839 to 9.057 s even though the new path did not activate there. This drift
limits attribution of whole-phase differences in the one profile. See the local
[timeline analysis](../../../.workspaces/hillclimb-20261104/38-prefill-target/timeline-analysis.md)
for scopes and remaining work.

## Findings and decision

Retain the exact paired path as an opt-in experimental setting, with the flag
enabled in [candidate settings](candidate-settings.json). The profile shows a
real local reduction and both context sizes favor fresh prefill on average,
while variable append and decode results limit the end-to-end conclusion.
Fixed-history checks found no measured drift; full suffix relocation and a new
practical task remain untested. Presets, installed frontends and main remain
unchanged.
