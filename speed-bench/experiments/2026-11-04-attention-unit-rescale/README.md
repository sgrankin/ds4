# Skipping unit rescaling in indexed prefill attention

**Retain the candidate as an opt-in approximation; do not promote it to the fast presets yet.** This experiment asks whether skipping a redundant rescale in
online softmax reduces indexed prefill attention time. The one-hour window is
November 4, 2026, 11:22–12:22 Eastern. Baseline `e050fa0b` includes the
adopted four-row staging kernel; the candidate runtime is `8b8ae550`.

## Context and mechanism

The [four-row experiment](../2026-11-04-attention-four-row/README.md) found a
consistent isolated operation improvement, but did not establish a whole-model
speedup. A separate fresh 32K profile attributed 9.329778 seconds of GPU work
to the four-row attention kernel. That profiled observation identifies a target;
it is not a profile-free latency comparison.

For each row, online softmax maintains a running maximum `M`, a sum and a
weighted output. The baseline computes `new_m = max(M, score)` and rescales
the accumulated sum and output by `exp(M - new_m)` before adding the new row.
For finite values when `score <= M`, `new_m` is `M` and the old scale is
mathematically one. The candidate retains the sum and output directly and
adds the weighted row in that case; when the maximum increases it uses the
original rescaling path. Row consumption remains chronological.

The hypothesis is that avoiding the redundant exponential and multiplications
saves time often enough to outweigh the branch and any compiler or hardware
costs. The algebra alone does not prove that compiled instructions are removed,
that the branch is profitable, or that floating-point results match. Numerical
and practical quality must be measured rather than inferred from the formula.

The candidate is a separate `heads16_rb4_unit_scale` kernel specialization,
selected by numeric `DS4_METAL_V41_ATTN_UNIT_SCALE=1`; `0` selects the control.
Both arms force four-row staging, so the comparison isolates this change. The
candidate applies only to eligible four-row prefill dispatches; decode and other
attention paths are unchanged. It adds no persistent buffer or cache. See the
[Metal kernel](../../../src/metal/dsv4_misc.metal) and
[host dispatch](../../../src/metal/attention_batch.zig) for the implementation.

## Setup and planned checks

Reuse the six-shape expanded and packed attention fixture, including wrapped
rings, masked IDs and short windows, and five alternating operation pairs at
512, 2048 and 8192 rows. Record the candidate revision, commands, all samples,
variation, order effects and any failures below. If the operation results are
promising, run five profile-free full-model pairs at both 8K and 32K and
compare full-vocabulary quality on fixed histories. Record practical task
quality if tested; delta-NLL and KL describe distribution changes and are not
automatic rejection criteria.

The runner clears inherited DS4 settings and applies the fast preset settings,
including four-row staging in both arms. One heavy model job at a time remains
under the pressure guard. Raw
logs and samples are local and ignored under
`.workspaces/hillclimb-20261104/05-attention-unit-rescale/`.

## Numerical checks

The initial strict fixture stopped on the first active candidate shape: at two
rows and 64 heads, element 2 differed by five ULPs (`3fbaeccc` versus
`3fbaecc7`). Both preceding fallback shapes matched exactly. This was a
numerical mismatch, not a runtime crash; `fixture.log` retains the failure.

Fixture revision `5460b94c` keeps strict comparisons for the packed control
and fallback paths and measures finite errors for active candidate shapes. All
six packed controls and all three candidate fallback cases matched exactly.
The candidate is an approximation on the tested active inputs:

| Rows / heads | Changed elements | Maximum absolute error | RMS error |
| --- | ---: | ---: | ---: |
| 2 / 64 | 4,885 / 65,536 | 7.6294e-6 | 3.1587e-7 |
| 33 / 32 | 41,433 / 540,672 | 7.6294e-6 | 3.0020e-7 |
| 3 / 64, short-window tail | 7,872 / 98,304 | 2.9802e-8 | 2.1095e-9 |

The local `measured-fixture.log` records the full results. These inputs do not
establish full-model quality. The separate kernel specialization also does not
prove that the compiler generates identical control instructions.

## Focused operation comparison

Five alternating pairs used the warmed 64-head operation fixture with four
calls per sample. As in the previous experiment, the timed scope includes
sorting, encoding and GPU completion, but excludes model execution, expert
loading and allocation. At 512 rows, the baseline averaged 8.2886 ± 0.1180 ms
per call and the candidate 7.5255 ± 0.0173 ms, a 9.21% reduction between
means. At 2048 rows, the averages were 31.7256 ± 0.9211 and 28.4556 ±
0.0113 ms, a 10.31% reduction. Variation is sample standard deviation; every
pair favored the candidate. The first 2048-row baseline sample was 33.37275 ms,
while the other four ranged from 31.29475 to 31.35525 ms; it remains included.
These are operation timings, not whole-model speedups.

| Pair | First | 512 baseline / candidate (ms/call) | 2048 baseline / candidate (ms/call) |
| --- | --- | --- | --- |
| 1 | Baseline | 8.49850 / 7.53425 | 33.37275 / 28.44725 |
| 2 | Candidate | 8.23500 / 7.50400 | 31.30775 / 28.46125 |
| 3 | Baseline | 8.22300 / 7.51000 | 31.29725 / 28.47275 |
| 4 | Candidate | 8.25675 / 7.53575 | 31.35525 / 28.45050 |
| 5 | Baseline | 8.22975 / 7.54325 | 31.29475 / 28.44625 |

At 8192 rows, the baseline averaged 124.3048 ± 1.1178 ms and the
candidate 112.3009 ± 0.0393 ms, a 9.66% reduction; every pair favored the
candidate. The first baseline sample was 126.30350 ms, while the other four
ranged from 123.75450 to 123.83675 ms; it remains included.

| Pair | First | Baseline (ms/call) | Candidate (ms/call) |
| --- | --- | ---: | ---: |
| 1 | Baseline | 126.30350 | 112.32150 |
| 2 | Candidate | 123.83675 | 112.24775 |
| 3 | Baseline | 123.80300 | 112.34775 |
| 4 | Candidate | 123.75450 | 112.31150 |
| 5 | Baseline | 123.82600 | 112.27600 |

Local `operation512.csv`, `operation2048.csv` and `operation8192.csv` retain
the raw samples.

## 32K full-model timing

Five profile-free alternating pairs used the same M5 Max and DeepSeek V4.1
Flash 1017 IQ2_XXS/Q2_K imatrix checkpoint as the earlier report: SSD
streaming, a requested 64 GB expert cache, 8192-token prefill chunk and
40960-token context allocation. Each run includes 128 teacher tokens after
fresh prefill, KV restore, a 512-token append and another 128 teacher tokens.
The expert cache survives restore and the OS file cache is not flushed. Phase
seconds are derived from throughput rounded to two decimal places, exclude
loading and snapshot save/restore, and inference total sums the four phases.
Steady decode excludes the first token.

Fresh prefill averaged 48.678 ± 1.999 seconds for the baseline and
47.575 ± 1.772 seconds for the candidate, a 2.27% reduction between means;
the candidate was faster in all five pairs. Inference totals averaged
64.335 ± 2.273 and 63.369 ± 1.641 seconds, a 1.50% reduction. The first
candidate total was 0.280 seconds slower, while the other four were faster.
Wall time averaged 66.558 and 65.592 seconds. Sample standard deviations are
shown, and the first pair was faster for both arms.

| Pair | First | Fresh prefill baseline / candidate (s) | Inference total baseline / candidate (s) | Append baseline / candidate (s) |
| --- | --- | --- | --- | --- |
| 1 | Baseline | 45.219 / 44.557 | 60.458 / 60.738 | 7.043 / 7.742 |
| 2 | Candidate | 48.789 / 47.473 | 64.303 / 62.888 | 7.185 / 7.155 |
| 3 | Baseline | 49.381 / 48.257 | 65.038 / 63.900 | 7.300 / 7.333 |
| 4 | Candidate | 50.065 / 48.808 | 65.956 / 64.598 | 7.552 / 7.416 |
| 5 | Baseline | 49.937 / 48.781 | 65.917 / 64.721 | 7.596 / 7.523 |

Append means were 7.335 ± 0.237 and 7.434 ± 0.219 seconds. Fresh steady
decode averaged 34.770 ± 0.071 and 34.554 ± 0.475 tokens/s; warm steady
decode averaged 33.762 ± 0.686 and 33.540 ± 0.378 tokens/s. Although decode
arithmetic is unchanged, approximate prefill state can change later routing
and expert residency even with fixed teacher tokens. These results do not
establish a cause for the small decode differences. The pressure guard did not
interrupt the runs; reported swap was 24.971 GiB. Local `timing32k` retains
all samples and phase details.

## 8K full-model timing

With the same setup and scope, fresh prefill averaged 20.285 ± 0.166 seconds
for the baseline and 19.912 ± 0.141 seconds for the candidate, a 1.84%
reduction between means. Inference totals averaged 35.993 ± 0.159 and
35.586 ± 0.182 seconds, a 1.13% reduction; all five pairs favored the
candidate for both measures. Wall time averaged 38.095 ± 0.162 and
37.671 ± 0.179 seconds. Variation is sample standard deviation.

| Pair | First | Fresh prefill baseline / candidate (s) | Inference total baseline / candidate (s) | Append baseline / candidate (s) |
| --- | --- | --- | --- | --- |
| 1 | Baseline | 20.297 / 19.763 | 35.974 / 35.358 | 7.103 / 7.368 |
| 2 | Candidate | 20.078 / 19.994 | 35.828 / 35.577 | 7.508 / 7.353 |
| 3 | Baseline | 20.205 / 19.811 | 35.924 / 35.496 | 7.414 / 7.444 |
| 4 | Candidate | 20.530 / 20.109 | 36.256 / 35.845 | 7.481 / 7.452 |
| 5 | Baseline | 20.314 / 19.883 | 35.984 / 35.655 | 7.458 / 7.483 |

Append means were 7.393 ± 0.166 and 7.420 ± 0.056 seconds. Fresh steady
decode averaged 35.070 ± 0.662 and 35.458 ± 0.110 tokens/s; warm steady
decode averaged 33.470 ± 0.668 and 33.344 ± 0.265 tokens/s. The pressure
guard did not interrupt the runs. Local `timing8k` retains all samples and
phase details; routing effects on decode remain unresolved.

## Fixed-history full-model quality at 8K

At an 8192-token frontier and again after a 512-token append, the quality
probe compared 128 full-vocabulary, temperature-one teacher distributions per
frontier on identical histories. None of the 256 rows matched exactly, and
seven changed top-1 token. Small fixture errors propagate through the model;
these outputs are not near-exact.

| Frontier | Top-1 changes | Mean delta-NLL (nats/token) | Mean KL | Maximum delta-NLL |
| --- | ---: | ---: | ---: | ---: |
| 8192 | 3 / 128 | +0.0155165 | 0.014795 | +1.20123 |
| 8704, after append | 4 / 128 | -0.0100021 | 0.0110992 | +0.350322 |
| Combined | 7 / 256 | +0.0027572 | 0.0129471 | +1.20123 |

The combined teacher perplexity ratio is 1.002761, or approximately 0.276%
higher. The largest KL was 0.243893; the 95th percentile was 0.057809. These
statistics describe the tested teacher histories, not diverging generated
conversations or task success. Local `quality8k/quality-summary.json` retains
per-frontier and tail statistics.

## Fixed-history full-model quality at 32K

The same full-vocabulary diagnostic compared 128 teacher distributions at
each frontier. None of the 256 rows or post-evaluation hashes matched exactly.
Fourteen rows changed top-1 token.

| Frontier | Top-1 changes | Mean delta-NLL (nats/token) | Mean KL | Maximum delta-NLL | Maximum KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32768 | 9 / 128 | -0.01792275 | 0.02891299 | +1.077078 | 0.6427808 |
| 33280, after append | 5 / 128 | -0.03080894 | 0.01884333 | +0.8000994 | 0.2489017 |
| Combined | 14 / 256 | -0.02436584 | 0.02387816 | +1.077078 | 0.6427808 |

The candidate assigns higher mean likelihood to the teacher tokens in this
sample, but that does not establish better answers. The full-vocabulary KL and
individual changes show meaningful drift on fixed histories; they do not
measure task success or generated trajectories. Local
`quality32k/quality-summary.json` retains per-frontier and tail statistics.

## Practical task check

The first reasoning-enabled `merge_ranges` check completed for both arms, but
its traces showed no four-row kernel dispatch, so it does not qualify this
optimization. Its evidence is retained under `agent-smoke`. A longer repeat
prepended 12,000 characters of unrelated background, with an 8192-token context
and prefill chunk, temperature zero and a 1024-token generation limit. Both
generated implementations passed the reused independent 207-case oracle,
which also checked input immutability and a list-of-tuples result. The sources
differed. The baseline trace recorded 76 four-row dispatches; the candidate
recorded 228 unit-scale dispatches across initial prefill and compaction or
rebuild. The baseline had no reserve compaction, while the candidate had one
after an interrupted test call. The candidate initially wrote an incorrect
adjacency test expectation, then corrected it and completed the task. Local
`agent-smoke-long/oracle-result.json` and `activation.json` retain the checks.
This is one profiled task per arm, not a performance measurement or a general
quality claim.

## Diagnostic profile

A separate intrusive candidate profile confirms dispatch, but is not a paired
latency comparison. Fresh 32K prefill recorded 284 command buffers and 101542
encoders, with a 49.781183-second envelope, 37.553191 seconds of encoder union
and 12.227992 seconds uncovered. The unit-scale kernel accounted for
10.076695 seconds across 94 calls. Of these, 72 large `8192 × 4 × 1` calls
summed to 9.722263 seconds and ranged from 111.267 to 161.663 ms. No unit-scale
dispatch occurred in the warm append.

This unpaired profile was slower than the earlier four-row profile across
several kernels. It establishes activation and dispatch counts, not a kernel
speedup or regression. The comparative results above use profile-off pairs and
the focused operation runs. Local `timeline32k/1-candidate/decode-anatomy.json`
retains the profile.

## Decision

Retain the specialization as an opt-in approximation, without enabling it
in the fast presets yet. The candidate improved the measured isolated operation
and both tested full-model prefill and inference totals. The fixed-history
logits drift and one passing practical task define a limited quality picture,
not general agent qualification. The bit differences alone are not grounds
for rejection. A next bounded experiment will test whether preserving the
original contribution rounding while skipping the identity exponential offers
a different speed and quality tradeoff.

## Reproduction

Build the runtime at `8b8ae550` and the fixture at `5460b94c` with
`./tools/zig-build -Doptimize=ReleaseFast ds4-bench metal-test -j4`. Run the
fixture and each operation size under the pressure guard, changing 2048 to 512
and 8192 for the other sizes. The fixture measures active candidate error;
the packed control and fallback comparisons remain strict.

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/unit-fixture-guard \
  -- ./zig-out/bin/test-deepseek41-metal --fp4-attention
python3 speed-bench/ideas/pressure_run.py .workspaces/unit-op-guard \
  -- ./zig-out/bin/test-deepseek41-metal --fp4-attention-bench 2048
python3 speed-bench/ideas/pressure_run.py .workspaces/unit-timing-guard \
  -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu \
  .workspaces/unit-timing timing --flag DS4_METAL_V41_ATTN_UNIT_SCALE --frontier 32768
```

Use a separate output directory and frontier 8192 for the shorter comparison.
The runner's `quality` stage collects full-vocabulary diagnostics separately
from timing; preserve its recorded environment and prompt when reproducing a
historical result. See each raw run's `arguments.json` and `environment.json`
for exact settings.
