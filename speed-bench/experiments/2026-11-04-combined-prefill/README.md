# Combined indexed prefill qualification

**Adopt the unit-rescale shortcut in the resident-fast and persistent-fast
presets and their alias, with an explicit `0` override.** This experiment compares the published
`364ceeab` baseline with the combined four-row staging and unit-rescale
candidate at `b9c9e77b`, which has the same runtime as `1312c914`. The selected
window is November 4, 2026, 12:26–13:26 Eastern. The question
is how the combined package behaves against the published baseline, including
its practical quality; the separate experiments do not establish that result.

## Setup

The runs use an Apple M5 Max with 128 GiB memory and the
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf` checkpoint with SSD
streaming. The requested 64 GB expert cache comprises 56.88 GiB dynamic cache
and 7.12 GiB prefill reserve. Prefill chunks contain up to 8192 tokens; the
bench allocates a 40960-token context, and replay uses 16384 tokens. The bench
prompt is [the book](../../promessi_sposi.txt); replay uses the
[recorded session](../../agent-session/session-v1.txt).

The baseline uses the overnight [final settings](../2026-11-04-decode-pass/final-settings.json).
The [candidate settings](candidate-settings.json) preserve those settings and
enable `DS4_METAL_V41_PREFILL_ATTN_RB4=1` and
`DS4_METAL_V41_ATTN_UNIT_SCALE=1`. The candidate source also contains a narrow
batch-shared-overlap cleanup, but its flag is unset in both arms. Five
alternating timing pairs are complete for each of the 32K and recorded-session
replay workloads. Qualification also comprises one full-logit replay pair, one
32K full-logit pair, and one reasoning-enabled coding task per arm. The coding task
uses a 16,384-token context, 12,000 characters of unrelated book background,
and `merge_ranges`, checked with an independent 207-case oracle. The sections
below record the completed results and reproduction commands.
Raw output is local and ignored under
`.workspaces/hillclimb-20261104/07-combined-prefill`.

A source audit predicts 38 eligible indexed attention dispatches during the
replay's initial 1,859-token prefill; this prediction has not been verified by
a trace. Later prefill blocks contain 62–826 tokens and use selected 64-token
tiles with `batch_core=false`, so the target kernel directly affects only the
initial prefill. Approximate initial state may nevertheless change later routes
and cache behavior. The initial prefill represented about 8% of total replay
time in the prior package, which limits the opportunity for a whole-session
effect. This is context for the measurement, not a prediction of its result.
The audit follows [prefill policy](../../../src/engine/deepseek41/prefill_policy.zig#L86),
[prefill](../../../src/engine/deepseek41/prefill.zig#L219),
[batch attention](../../../src/engine/deepseek41/batch_attention.zig#L278),
[attention dispatch](../../../src/metal/attention_batch.zig#L533), and the
[Metal kernel](../../../src/metal/dsv4_misc.metal#L6004).

## Mechanism

Indexed prefill attention processes raw and selected compressed key/value rows
while maintaining an online softmax for each head. The baseline stages one
row at a time for reuse across 16 heads. Four-row staging instead loads up to
four rows into threadgroup memory and consumes them in order, reducing the
number of barriers per full block while increasing staging memory. Its
[separate experiment](../2026-11-04-attention-four-row/README.md) describes the
eligibility conditions and numerical checks.

For each row, online softmax rescales its accumulated sum and weighted output
when updating the running maximum. If a finite score does not increase that
maximum, the mathematical scale is one. The unit-rescale specialization skips
that rescaling in this case, and otherwise follows the original path. This is
an approximation in the implementation: the earlier
[unit-rescale experiment](../2026-11-04-attention-unit-rescale/README.md)
observed floating-point differences and full-model distribution drift. A
[subsequent contribution-rounding prototype](../2026-11-04-attention-contribution-rounding/README.md)
was removed; this qualification uses the original unit-scale mode 1.

## 32K timing

Five profile-free alternating pairs used fresh 32,768-token prefill followed
by decoding 128 fixed teacher tokens, restoring the key/value (KV) cache, a
512-token append and decoding another 128 teacher tokens. The expert cache
survives restore; the OS file cache was not
flushed. Phase seconds are derived from throughput rounded to two decimal
places and exclude loading and snapshot save and restore. Inference total sums
fresh prefill, the full first decode window, append and the full second decode
window. The separately recorded steady decode excludes only the first token;
that exclusion does not establish that initialization has finished.

Fresh prefill averaged 47.654 ± 0.805 seconds for the baseline and 46.394 ±
0.656 seconds for the candidate. Inference totals averaged 63.900 ± 0.962 and
62.412 ± 1.022 seconds; measured wall times averaged 66.638 ± 1.171 and
64.663 ± 0.941 seconds. Variation is sample standard deviation. The candidate
was faster on all three measures in every pair. Fresh prefill improved by
1.64–3.36% within pairs, with larger improvements in the two candidate-first
pairs. Both arms' fresh prefill times generally increased across the sequence; the first
baseline wall time was unusually high relative to its other samples.

| Pair | First | Fresh prefill baseline / candidate (s) | Inference total baseline / candidate (s) | Wall baseline / candidate (s) |
| --- | --- | --- | --- | --- |
| 1 | Baseline | 46.689 / 45.688 | 63.035 / 61.010 | 68.087 / 63.452 |
| 2 | Candidate | 47.179 / 45.696 | 63.152 / 61.661 | 65.297 / 63.863 |
| 3 | Baseline | 47.429 / 46.651 | 63.439 / 62.921 | 65.590 / 65.120 |
| 4 | Candidate | 48.511 / 46.880 | 65.054 / 63.096 | 67.259 / 65.341 |
| 5 | Baseline | 48.460 / 47.053 | 64.819 / 63.373 | 66.957 / 65.539 |

Fresh steady decode averaged 33.098 ± 0.755 and 33.614 ± 0.693 tokens/s,
baseline and candidate respectively; after the append it averaged 34.088 ±
0.544 and 34.066 ± 0.557 tokens/s. These steady rates exclude only the first
recorded token. The local `timing32k` summary and CSV retain all phases and
samples.

## Recorded-session replay timing

Five profile-free alternating pairs replayed recorded prefill and decode phases.
The inference time sums model calls, excluding loading, phase-logit writes,
the final snapshot, sampling and tools. It averaged 178.337 ± 3.252 seconds
for the baseline and 180.540 ± 6.239 seconds for the candidate, a 1.24%
increase between means; variation is sample standard deviation. Three of five
pairs favored the candidate, including both candidate-first pairs. The final
candidate sample was 191.695 seconds, compared with 177.442–177.978 seconds
for its other samples. It remains included. Its slowdown spans initial prefill
and multiple decode and append phases; the last decode phases converge. The
cause is unresolved, and the variation makes the replay comparison inconclusive.

| Pair | First | Inference baseline / candidate (s) |
| --- | --- | --- |
| 1 | Baseline | 173.358 / 177.442 |
| 2 | Candidate | 179.483 / 177.978 |
| 3 | Baseline | 178.410 / 177.854 |
| 4 | Candidate | 178.081 / 177.730 |
| 5 | Baseline | 182.351 / 191.695 |

The pressure guard completed without interruption, reporting pressure 1 and
swap of 24.974 GiB initially and 24.966 GiB at the end. A subsequent `pmset`
check reported no thermal or performance warnings; it cannot rule out a
transient interference during the run. Local `replay-timing/analysis.json`
retains phase and wall-time samples. No further timing measurements are planned
in this experiment window.

## Recorded-session fixed-history quality

The full-vocabulary replay compared identical recorded teacher histories. All
3,371 rows aligned and were finite; none matched exactly by bytes or values,
and 46 rows (1.365%) changed top-1 token. Mean candidate-minus-baseline
negative log likelihood (delta-NLL) was −0.005030 nats per token, and mean
KL(baseline || candidate) was 0.004832 nats. The negative mean delta-NLL
indicates higher likelihood for these teacher tokens, not better practical
answers.

The 95th and 99th percentile KL values were 0.026651 and 0.072658 nats,
with a maximum of 0.260582. Delta-NLL ranged from −1.290905 to +1.078453
nats per token, with a 95th percentile of +0.055862. Drift was not monotonic
across turns: phase 16 had the highest mean KL at 0.017932, phase 20 had
0.000030, and final phase 28 had 0.008280. These fixed histories do not
measure diverging generated trajectories or task success. Local
`replay-quality/analysis.json` and `rows.csv` retain phase and row details.

## 32K fixed-history quality

At the 32K frontier and after the 512-token append, all 256 teacher positions
aligned; none matched byte for byte and 14 changed top-1 token. Mean delta-NLL
was −0.024366 nats per token and mean KL was 0.023878 nats. The 95th
percentile and maximum KL were 0.080899 and 0.642781; for delta-NLL they were
+0.356526 and +1.077078 nats per token. None of the 256 post-step logit hashes
matched. Those hashes cover logits, not the entire model state. As with the
replay, these measurements describe fixed histories rather than practical task
success. Local `quality32k/quality-summary.json` retains the details.

## Practical coding check

One reasoning-enabled `merge_ranges` task per arm completed without compaction.
Both generated implementations passed all 207 cases in the existing independent
oracle, preserved their inputs and returned lists of tuples. Their sources
differed. The candidate trace contained 61 unit-scale kernel encoder records,
so activation was observed. These are single profiled tasks, not a timing
comparison.

An additional probe exposed a candidate weakness: its docstring promises an
iterable, but `merge_ranges(iter([]))` raises `IndexError`; the baseline
returns `[]`. The original task did not explicitly require iterator support,
and the existing oracle uses lists. One task run per arm does not establish
that the shortcut caused this behavioral difference. Local `agent-smoke-long/oracle-result.json`
and `extra-checks.json` retain the checks.

## Decision

Enable the approximate unit-rescale shortcut in resident-fast and
persistent-fast and their alias, while preserving an explicit `0` override.
The ordinary engine and context-fast settings remain unchanged. The 32K
comparison showed a repeatable prefill and inference benefit, with measured
fixed-history distribution drift. Replay timing was inconclusive, and the
limited practical check found the iterator failure described above. The
approximation and observed degradation are acceptable for this package, but
these results do not establish general agent speed or quality.

The launcher tests passed, including default selection, explicit zero overrides
and server propagation. The qualified agent, server and benchmark binaries were
installed locally; published main was not advanced.

## Reproduction

Build the corresponding binaries in each source root. The driver loads Metal
sources relative to each root and clears inherited DS4 settings. From the
candidate root, use a new output and guard directory for each run:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/combined-prefill-guard \
  -- nu speed-bench/experiments/2026-11-04-decode-pass/combined.nu \
  .workspaces/combined-prefill-timing timing \
  --baseline-root .workspaces/hillclimb-20261104/starting-364ceeab \
  --candidate-root . \
  --baseline-revision 364ceeab --candidate-revision b9c9e77b \
  --baseline-settings speed-bench/experiments/2026-11-04-decode-pass/final-settings.json \
  --final-settings speed-bench/experiments/2026-11-04-combined-prefill/candidate-settings.json \
  --frontier 32768
```

Change the stage and output to `replay-timing` for the five replay pairs,
`replay` for one full-logit replay pair, or `quality` with `--frontier 32768`
for the 32K full-logit pair. The per-arm local manifests record the exact
binaries, settings, arguments and source roots used for these runs.
