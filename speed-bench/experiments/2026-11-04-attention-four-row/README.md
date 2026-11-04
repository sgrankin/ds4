# Four-row staging for indexed prefill attention

**The decision is pending.** A focused operation comparison favored the candidate by 1.57%, while the 32K full-model comparison shows no clear change in fresh-prefill time. This experiment asks whether staging four
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
explicitly set to `0` or `1`. Fixture revision `578aa9a9` adds a tail case
without changing the runtime. ReleaseFast `ds4-bench` and Metal fixture builds
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
`quality8k/quality-summary.json`. The 32K quality check is pending.

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

Local `operation-timing.csv` and `operation-summary.json` retain the samples.

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
higher for the candidate at both tested frontiers, but the magnitude and
cause remain uncertain; a focused 512-row operation check is underway.

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

## Decision

The focused append-shape check, 32K quality check, diagnostic profile and
retention decision remain pending.
