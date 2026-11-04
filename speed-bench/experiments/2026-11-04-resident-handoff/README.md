# Resident expert handoff experiment

The opt-in prototype is retained as groundwork for the next bounded experiment,
which will remove redundant address republishing while preserving cache pins and
fallback. It is not adopted in a fast preset, and the installed root binary was
not replaced. This experiment asks whether routing resident experts on
the GPU can remove the host handoff visible between routing and expert execution,
and whether the resulting approximation is useful at its measured quality cost.
The one-hour slice is November 4, 2026, 14:05–15:05 Eastern.

## Setup and mechanism

The runtime baseline is `02b7ae26`, after removal of the preceding experiment in
`caf67089`; the measured prototype is `6549a33b`. The candidate requires `DS4_METAL_V41_RESIDENT_GPU=1` together with a positive
resident route gap, native weights, apply mode and the streaming scalar-path
constraints. The flag alone does not activate it. The ordinary route and load path runs every eight
positions by default (`DS4_METAL_V41_RESIDENT_GPU_REFRESH=8`), refreshing the resident set.
Between refreshes the GPU uses existing resident cache addresses. CPU preflight
validates and pins those addresses for their use; this does not map the full model.

The GPU starts with the native router's expert IDs and weights. In native slot
order, it retains IDs that are resident and replaces each missing ID with the
highest bias-plus-affinity unused resident expert. It keeps the original native
slot weights. There is no cap, score-gap or rank restriction on substitutions.
Consequently this can execute different experts with the original weights: it is
an approximation, not merely a change in where the same route is executed.
The periodic ordinary route also makes the resident set evolve over time.

The source review found an exclusions mismatch, corrected before the measured
prototype; no further concrete safety blocker was identified in that review.
Ordinary CPU route statistics omit GPU-routed tokens, so those statistics alone
cannot establish how often the approximation occurred. The
`--resident-router` fixture passed. Raw artifacts are local under
`.workspaces/hillclimb-20261104/09-resident-handoff/`.

The timing runs used an M5 Max with 128 GiB memory and
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, SSD streaming with a
64 GB expert cache, 4,096 MB expert slabs and 8,192-token prefill chunks. Both
arms used the same candidate executable with the opt-in set to 0 or 1. The
profile-free comparison alternated five pairs: a fresh 32,768-token prefill and
128 teacher-forced decode tokens, then KV restore, a 512-token append and a
further 128 teacher-forced decode tokens. The expert cache survived restore; the
OS cache was not flushed. Phase times exclude loading and snapshot save/restore.

To reproduce, run the following under the existing `pressure_run.py` guard from
the repository root. The driver sets the other required route settings and
compares the same executable with the flag at 0 and 1. Use a new output directory
for each run; replace `timing` with `quality` or `timeline` for the diagnostic
stages, and use either frontier 8192 or 32768. The default is 128 decode tokens
per phase; steady decode excludes the first token.

```sh
nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/resident-repro timing --flag DS4_METAL_V41_RESIDENT_GPU --frontier 32768
```

## Findings so far

The fixed-history quality probe compared full-vocabulary temperature-one logits
on 256 teacher-token positions, 128 each at frontiers 8,192 and 8,704. The top
choice changed in 9 positions. Mean candidate-minus-baseline NLL was −0.000361
nats per token and mean KL(baseline || candidate) was 0.0224. The averages hide
larger changes: KL p95 was 0.0672, p99 0.424 and maximum 0.638; delta-NLL ranged
from −1.529 to +1.003 nats. These describe fixed histories, not practical task
correctness or what happens after sampled histories diverge.

At 32K, another 256-position fixed-history probe found 20 top-choice changes.
Mean delta-NLL was +0.083185 nats, corresponding to a 1.0867 teacher
perplexity ratio, and mean KL was 0.03696. At the 32,768 frontier the respective
means were +0.09535 and 0.05598, with 14 top-choice changes; at 33,280 they
were +0.07102 and 0.01794, with 6 changes. Across both frontiers delta-NLL
ranged from −0.902 to +3.261; KL p95 was 0.132, p99 0.451 and maximum 1.325.
These are distribution diagnostics rather than a task failure rate.

The five profile-free timing pairs give the following results. Values are mean ±
sample standard deviation; steady decode excludes the first token. The sample
lists are in pair order, with baseline first in pairs 1, 3 and 5.

| Measure | Baseline | Candidate |
| --- | --- | --- |
| Fresh steady decode, tokens/s | 34.668 ± 0.160 | 34.614 ± 0.142 |
| Warm steady decode, tokens/s | 33.060 ± 0.112 | 31.722 ± 0.152 |
| Fresh first decode token, ms | 826.308 ± 34.752 | 823.867 ± 11.380 |
| Warm first decode token, ms | 44.490 ± 18.100 | 38.184 ± 8.060 |
| Complete fresh decode, s | 4.490 ± 0.046 | 4.493 ± 0.010 |
| Complete warm decode, s | 3.886 ± 0.029 | 4.042 ± 0.018 |
| Combined inference phases, s | 64.498 ± 2.289 | 65.639 ± 0.500 |

Fresh steady samples were 34.50, 34.80, 34.49, 34.74, 34.81 versus 34.48,
34.55, 34.80, 34.73, 34.51 tokens/s. Warm steady samples were 32.99, 33.21,
32.94, 33.02, 33.14 versus 31.48, 31.67, 31.85, 31.82, 31.79 tokens/s. Warm
steady decode took 3.42–4.86% longer in every pair, across both run orders;
fresh steady decode was close. Combined inference samples were 60.442, 65.003,
65.527, 65.694, 65.825 versus 64.991, 65.235, 66.009, 65.822, 66.136 seconds.
These sums cover the fresh and warm prefill and decode phases, not process startup
or snapshot operations. Seconds are derived from two-decimal reported throughput.

The first baseline prefill was unusually fast at 44.123 seconds; later baseline
samples were 49.015–49.635 seconds, and candidate samples 48.873–49.738 seconds.
The experimental flag is inactive in fresh prefill, so the first-pair difference
should not be attributed to this change. All samples and phase details remain in
`timing32k/timing-samples.csv` under the local artifact directory.

A separate instrumented 32K pair shows a structural reduction in the handoff.
For steady tokens, the average router-to-expert-pair gap fell from 7.19 to
1.45 ms fresh and 7.08 to 1.45 ms warm; uncovered GPU time fell from 5.86 to
1.04 ms and 5.72 to 1.07 ms respectively. Instrumentation changes scheduling,
so these diagnostic timings do not override the profile-free comparison above.
The instrumented warm candidate first token took 360 ms versus 40 ms baseline,
a separate confound; both were ordinary refresh tokens.

Matched warm main-thread samples show synchronization moving from the selected
readback wait to end-of-token command-buffer completion: 1,259 of 1,348 baseline
samples were in the former path, versus 202 of 1,278 candidate samples; the
candidate had 822 in end-of-token completion. These are stack sample counts,
not elapsed milliseconds, and nested categories are not additive. Candidate
readiness preflight accounted for 87 samples, including 85 in redundant address
writes and their bridge calls. This identifies a concrete preparation hotspot,
but neither the samples nor the GPU gaps alone establish its latency cost.

A paired practical coding check asked each arm to implement `merge_ranges` for
any iterable, including an empty generator, after 12,000 characters of
unrelated background. Both used reasoning, temperature 0, a 16,384-token
context, 8,192-token chunks and a 1,024-token limit per call. Source inspection
preceded an independent oracle check: both implementations passed 207 cases in
list, iterator and generator forms (621 evaluations), returned lists of tuples
and left inputs unchanged. This is one task per arm, with different working
directory text, and is limited evidence of practical quality. No compaction
matches appeared in stdout or stderr. Candidate GPU activation appeared early in
the timeline (`E190`/`E191` `resident_route_one`), but later diagnostic tracing
repeatedly reported `Cannot allocate sample buffer`; the task completed with an
incomplete trace. The trace cannot support a task-performance conclusion.

The 8K profile-free five-pair comparison used the same procedure and shows a
small fresh difference and a noisier warm slowdown. Mean ± sample SD:

| Measure | Baseline | Candidate |
| --- | --- | --- |
| Fresh steady decode, tokens/s | 34.906 ± 0.662 | 35.012 ± 0.672 |
| Warm steady decode, tokens/s | 33.564 ± 1.208 | 32.954 ± 1.136 |
| Fresh first token, ms | 874.164 ± 48.370 | 844.578 ± 15.997 |
| Warm first token, ms | 33.281 ± 1.383 | 32.978 ± 0.651 |
| Complete fresh decode, s | 4.514 ± 0.070 | 4.474 ± 0.067 |
| Complete warm decode, s | 3.821 ± 0.136 | 3.890 ± 0.131 |
| Combined inference phases, s | 35.354 ± 0.963 | 35.432 ± 0.918 |

Fresh steady samples were 34.28, 34.09, 35.41, 35.37, 35.38 versus 34.53,
34.07, 35.58, 35.45, 35.43 tokens/s; warm samples were 35.38, 33.64, 33.84,
32.71, 32.25 versus 34.65, 33.53, 32.53, 32.21, 31.85. The complete sample
record, including first tokens and full windows, is in `timing8k/timing-samples.csv`
under the local artifact directory.

Every guard completed at pressure level 1 with zero bad samples and no
interruptions. Swap was 28.924 GiB for the fixture, 28.916 GiB for 8K quality,
28.877–28.916 GiB for 32K timing, 28.877 GiB for the 32K profile, quality, CPU
and task runs, and 28.854–28.877 GiB for 8K timing.

The ordinary timing results do not show a useful speedup, and the warm 32K case
is slower. The profile nevertheless shows the intended handoff reduction and a
concrete host preparation hotspot. Retaining this opt-in prototype enables the
selected address-republishing follow-up to measure the combined change against
the original baseline; these results do not justify preset promotion.
