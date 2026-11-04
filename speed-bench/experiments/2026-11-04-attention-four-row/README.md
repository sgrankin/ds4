# Four-row staging for indexed prefill attention

**Retain the candidate and enable it in the resident-fast and persistent-fast
source presets; the ordinary default remains off.** The focused operation
comparison favored four-row staging by about 1.5% at all three tested sizes.
The full-model measurements do not establish a whole-task speedup or a
regression. This experiment asks whether staging four
key/value rows at a time reduces indexed prefill attention time while retaining
the baseline's reuse across 16 heads. The selected window is November 4, 2026,
10:45–11:45 Eastern.

## Context and mechanism

Indexed attention combines the visible raw window with selected compressed
key/value rows. The baseline `heads16_dual` kernel stages one 512-element row
as half-precision values in 1 KiB of threadgroup memory, uses that row for 16
heads, and synchronizes around each row. It processes the raw rows and then
the sorted compressed IDs in order, updating the online softmax for each row.
See the [Metal kernels](../../../src/metal/dsv4_misc.metal) and
[host dispatch](../../../src/metal/attention_batch.zig).

The candidate `kernel_dsv4_indexed_mixed_attention_heads16_rb4` stages up to
four rows in 4 KiB of threadgroup memory and consumes them sequentially. The
hypothesis is that fewer barriers per full block can outweigh the increased
threadgroup memory and any additional indexing cost. It retains 16-head reuse,
the half-precision staging boundary and chronological attention math. For compressed rows, invalid slots are staged as zero and skipped; a future
ID stops consumption after the group reaches its closing barrier. It adds
no persistent scratch buffer or dispatch and avoids a per-lane array of
compacted IDs. Preserving the order and staging boundary is intended to preserve
numerical behavior, but does not establish equivalence until tested. The larger
staging allocation could affect occupancy, and neither barrier savings nor an
end-to-end speedup is established by the design.

The earlier [16-row prefill trial](../2026-11-04-prefill-row-blocking/README.md)
regressed at both 8K and 32K and was removed. It changed both staging size and
head reuse, from 16 to eight heads per group, so it did not isolate the effect
of row blocking. The subsequent [cooperative packed reader](../2026-11-04-packed-attention-reader/README.md)
showed no clear payoff and was removed. Baseline `08711070` restores the
original reader and prefill behavior.

## Setup and planned checks

The candidate is opt-in through `DS4_METAL_V41_PREFILL_ATTN_RB4`. The intended
dispatch is limited to multirow, non-quality attention with Metal Performance
Primitives available, 32 or 64 heads, width 512, top-k 512 and window 128.
Both arms use runtime revision `52d5e7a1` in the same binary, with the flag
explicitly set to `0` or `1`. Fixture revision `578aa9a9` adds a tail case, and `b74e543c` adds an
optional operation row count; neither changes the runtime. ReleaseFast `ds4-bench` and Metal fixture builds
passed.

The existing [decode-pass runner](../2026-11-04-decode-pass/run.nu)
uses five alternating pairs per frontier with profiling disabled. The model setup
matches the preceding report: M5 Max with 128 GiB memory, DeepSeek V4.1 Flash 1017
IQ2_XXS/Q2_K imatrix checkpoint, SSD streaming, requested 64 GB expert cache,
8192-token prefill chunk and 40960-token context allocation. The workload uses
128 teacher-forced tokens before and after a 512-token append. The OS file
cache is not flushed, and the process-local expert cache survives KV restore.
The runner clears inherited DS4 settings. Phase times exclude loading and
snapshot save and restore; inference total sums fresh prefill, decode, append
and second decode. Seconds are derived from throughput rounded to two decimal
places; steady decode excludes the first token.

Raw output is local and ignored under
`.workspaces/hillclimb-20261104/04-attention-four-row`.

## Numerical checks

The guarded `--fp4-attention` fixture passed six shapes comparing expanded
and packed rows with the flag off and on. The original five include wrapped raw
rings and masked IDs; the added case covers query position zero with a wrapped raw ring and
successive windows of one, two and three rows. Single-row and quality cases
exercise fallback behavior rather than the new prefill dispatch. The large
2048-row operation input also matched the expanded reference exactly. Local
logs are `fixture.log`, `tail-fixture.log` and `operation-timing.log` under the
artifact directory. These tests establish agreement for their inputs, not all
possible histories.

At 8K, all 256 fixed-history full-vocabulary rows and all 256 post-step full-vocabulary logit
hashes matched byte for byte; delta-NLL and KL were zero. The local evidence is
`quality8k/quality-summary.json`. The 32K check also matched all 256 rows
and 256 post-step full-vocabulary logit hashes, with zero delta-NLL and KL.
Across both frontiers, 512 rows and 512 hashes matched. Its local evidence is
`quality32k/quality-summary.json`.

Practical task quality was not separately tested for this candidate. The fixed
teacher-history checks above establish agreement on those histories, not every
possible response or the contents of every cache buffer.

## Focused operation comparison

Five alternating pairs used a warmed 2048-row, 64-head shape with 512 valid
keys and 8192 compressed rows, four calls per sample. Timing included sorting,
encoding and GPU completion, but not model execution, SSD expert loading or
allocation. The baseline averaged 31.8172 ± 0.1052 ms per call and the
candidate 31.3174 ± 0.0261 ms, a 1.57% reduction between means; variation is
sample standard deviation. Every pair favored the candidate. This is a gain
for the isolated operation, not a whole-model speedup.

| Pair | First | Baseline (ms/call) | Candidate (ms/call) |
| --- | --- | ---: | ---: |
| 1 | Baseline | 32.00425 | 31.31275 |
| 2 | Candidate | 31.76050 | 31.27875 |
| 3 | Baseline | 31.78225 | 31.34800 |
| 4 | Candidate | 31.78075 | 31.33350 |
| 5 | Baseline | 31.75825 | 31.31375 |

The same comparison at 512 and 8192 rows also matched the expanded reference
and favored the candidate in every pair. At 512 rows the baseline averaged
8.3494 ± 0.0289 ms and the candidate 8.2222 ± 0.0220 ms, a 1.52% decrease;
at 8192 rows the averages were 125.7937 ± 0.0820 and 123.8070 ± 0.0167 ms,
a 1.58% decrease. The same timing scope includes overhead shared by both arms,
so these are operation-level changes rather than isolated kernel timings.

| Pair | First | 512 baseline / candidate (ms/call) | 8192 baseline / candidate (ms/call) |
| --- | --- | --- | --- |
| 1 | Baseline | 8.39950 / 8.24500 | 125.93875 / 123.79425 |
| 2 | Candidate | 8.34875 / 8.24750 | 125.74150 / 123.81175 |
| 3 | Baseline | 8.33375 / 8.20325 | 125.76375 / 123.81850 |
| 4 | Candidate | 8.33450 / 8.20750 | 125.75175 / 123.82500 |
| 5 | Baseline | 8.33050 / 8.20775 | 125.77250 / 123.78525 |

Local `operation-timing.csv`, `operation512.csv`, `operation8192.csv` and
their summaries retain the samples.

## 32K timing

The five profile-free pairs show no clear fresh-prefill change: baseline
48.709 ± 1.455 seconds versus candidate 48.633 ± 1.559 seconds, a 0.16%
decrease between means. Inference totals were 64.392 ± 1.570 and
64.521 ± 1.353 seconds, a 0.20% increase. Append was 7.302 ± 0.209 and
7.481 ± 0.261 seconds. These small differences do not establish a regression
or a whole-model benefit. The first pair was faster for both arms. The pressure
guard completed without interruption. Values use sample standard deviation.

| Pair | First | Fresh prefill baseline / candidate (s) | Inference total baseline / candidate (s) | Append baseline / candidate (s) |
| --- | --- | --- | --- | --- |
| 1 | Baseline | 46.205 / 45.917 | 61.771 / 62.304 | 7.055 / 7.828 |
| 2 | Candidate | 48.851 / 48.889 | 64.340 / 64.348 | 7.188 / 7.111 |
| 3 | Baseline | 49.047 / 49.217 | 64.654 / 64.990 | 7.224 / 7.405 |
| 4 | Candidate | 49.660 / 49.261 | 65.512 / 65.082 | 7.521 / 7.480 |
| 5 | Baseline | 49.784 / 49.878 | 65.683 / 65.880 | 7.519 / 7.580 |

Fresh steady decode averaged 34.372 ± 0.785 / 34.332 ± 0.655 tokens/s
and warm steady decode 33.838 ± 0.595 / 33.494 ± 0.267 tokens/s,
baseline / candidate. Local `timing32k` artifacts
retain unrounded samples and other phases.

## 8K timing

Fresh prefill averaged 20.040 ± 0.316 seconds for the baseline and
19.855 ± 0.426 seconds for the candidate, a 0.92% decrease between means.
Inference totals were 35.267 ± 0.655 and 35.298 ± 0.731 seconds, a 0.09%
increase. Append averaged 7.104 ± 0.327 and 7.271 ± 0.300 seconds. The
candidate was faster for fresh prefill in four of five pairs. Append means were
higher for the candidate at both frontiers, but in every pair at both frontiers
the arm run second had the slower append. The diagnostic profile below shows
that the measured append does not dispatch the candidate kernel. The separate
512-row operation comparison exercises it directly but does not reproduce the
whole append workload.

| Pair | First | Fresh prefill baseline / candidate (s) | Inference total baseline / candidate (s) | Append baseline / candidate (s) |
| --- | --- | --- | --- | --- |
| 1 | Baseline | 20.093 / 19.365 | 34.905 / 35.042 | 6.732 / 7.458 |
| 2 | Candidate | 19.579 / 19.416 | 34.492 / 34.142 | 6.862 / 6.747 |
| 3 | Baseline | 19.883 / 20.147 | 35.038 / 35.636 | 7.043 / 7.308 |
| 4 | Candidate | 20.349 / 20.136 | 35.976 / 35.688 | 7.438 / 7.383 |
| 5 | Baseline | 20.294 / 20.213 | 35.923 / 35.982 | 7.445 / 7.458 |

Fresh steady decode averaged 35.318 ± 0.519 / 35.216 ± 0.461 tokens/s and
warm steady decode 34.580 ± 0.816 / 34.398 ± 0.846 tokens/s, baseline /
candidate. Local `timing8k` artifacts retain calculated samples and other
phases. The timing scope and precision match the 32K comparison.

## Diagnostic profile

A separate candidate profile of fresh 32K prefill recorded 284 command buffers,
101542 encoders, a 45.238377-second envelope, 34.339142 seconds of encoder
union and 10.899235 seconds uncovered. The four-row kernel summed to 9.329778
seconds across 94 dispatches; 72 large `8192 × 4 × 1` dispatches totaled
8.943567 seconds, ranging from 117.694 to 140.820 ms each. All four-row kernel
work was in fresh prefill. The `warm_prefill_and_restore` phase used vector F16
attention (0.821957 seconds) and reduction (0.127524 seconds), with no four-row
kernel dispatch. Thus the measured 512-token append does not exercise the
candidate, although the focused 512-row operation fixture does. This is one
profiled observation, not a paired performance comparison. Local evidence is
`timeline32k` under the artifact directory.

## Decision

The consistent operation gain and absence of an established full-model
regression justify retaining the opt-in kernel and enabling it in the two fast
source presets. The default remains off. The measured full-model totals do
not establish a whole-task speedup, and the append ordering pattern prevents
attributing its higher candidate mean to the change. The separate profile confirms that the measured append uses a different
attention path.

## Reproduction

Build with `./tools/zig-build -Doptimize=ReleaseFast ds4-bench metal-test -j4`.
The focused benchmark accepts an optional row count at fixture revision
`b74e543c`; use 512, 2048 and 8192 to reproduce the three measured shapes.
Each invocation warms both arms and takes five alternating pairs.

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/rb4-fixture-guard \
  -- ./zig-out/bin/test-deepseek41-metal --fp4-attention
python3 speed-bench/ideas/pressure_run.py .workspaces/rb4-operation-guard \
  -- ./zig-out/bin/test-deepseek41-metal --fp4-attention-bench 2048
python3 speed-bench/ideas/pressure_run.py .workspaces/rb4-timing-guard \
  -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu \
  .workspaces/rb4-timing timing --flag DS4_METAL_V41_PREFILL_ATTN_RB4 --frontier 32768
```

Use a separate output directory and frontier 8192 for the shorter comparison.
The same runner's `quality` and `timeline` stages produce separate diagnostics;
the quality stage enables logit dumps, not the engine's quality mode that would
disable this candidate. Historical settings and the binary revision above own
this comparison; later runner defaults may include additional optimizations.
