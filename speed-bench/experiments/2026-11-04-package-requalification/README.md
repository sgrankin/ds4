# Accumulated package: recorded agent qualification

**Retain the accumulated package as an explicit opt-in for continued
optimization.** The candidate had a favorable five-pair recorded-replay mean,
with uncertainty from an unusually fast first baseline, and failed an edge case
in one fresh coding task. This does not warrant promoting GPU routing to the
fast presets. The comparison is against published `364ceeab`, with fixed-history
distribution measurements and one fresh coding task.
The selected window is November 4, 2026, 22:27–23:27 Eastern; no runtime change
is selected for this experiment.

## Comparison

The baseline uses the preserved `364ceeab` checkout and the [original
settings](../2026-11-04-decode-pass/final-settings.json). The candidate uses
runtime `909faab5` and [candidate settings](candidate-settings.json). Both use
the same DeepSeek V4.1 Flash IQ2_XXS/Q2_K checkpoint, 64 GB expert cache and
8,192-token prefill chunks. Each binary runs in its matching source root, where
it loads its Metal sources. The driver clears inherited `DS4` settings and
records each arm's settings and binary hash.
The published baseline already uses quantized weights and cap-five resident
replacement. Measured drift is additional to that baseline, rather than a
comparison with unquantized weights or native routing.
Each arm starts a new replay process, engine and session: expert and KV state
start fresh and persist across that arm's recorded phases. The OS file cache
was neither flushed nor controlled.

The machine is an Apple M5 Max with 128 GiB memory. The rebuilt candidate
replay binary has SHA-256
`da8e8f5288cd6829775c37da05d2e986233e53e68175752363ff28cc90a5ae13`;
the baseline binary has
`bd36bcb5a8b528bef2e1b733f26a9f143c16f17fffb5b981456b7dae47bb37ab`.
The candidate manifest records checkout `b9bc4fb5`, which contains runtime
`909faab5` plus fixture and documentation changes. The replay started at
22:27:30 Eastern; its guarded command and manifests are in the raw root.

The candidate retains the four-row indexed prefill and unit-rescale settings,
GPU resident routing, address reuse, shared/routed expert concurrency and an
ordinary-route refresh every 16 positions from [experiment
13](../2026-11-04-resident-agent-qualification/README.md). Since that
qualification, it adds SIMD resident selection and router fusion, fused BF16
prefill RoPE, and a matrix projection for temporary indexer queries. The last
can change which compressed keys attention selects; its numerical and isolated
prefill evidence is in the [projection report](../2026-11-04-prefill-f16-projections/README.md).
This comparison measures the package together, not the sum of earlier isolated
gains. Experiment 13 did not establish a recorded-session inference gain, and
one candidate coding task did not complete within its configured budget.

The replay is not expected to dispatch the new indexer-query matrix projection.
It has 15 prefill phases totaling 5,339 tokens and 14 decode phases totaling
3,371 scalar evaluations. The initial 1,859-token prefill is not aligned to 32
rows; all later appends are below the 1,024-token selected limit and use the
selected-small path, which requires exact rows and bypasses the batch projection.
Cached prefixes match the preceding phase ends, so the inputs do not imply a
hidden rebuild; encoder residency also requires at least 16,384 remaining
tokens. This conclusion follows from the inputs and source, not a kernel trace:
the candidate log confirms TensorOps and the chunk size but does not count these
dispatches. The replay therefore qualifies accumulated package behavior on this
session, not the large-prefill projection benefit; experiment 18's large-prompt
evidence remains separate.

## Measurement scope

Five alternating, profile-free pairs replay the same recorded prefill and decode
histories. Inference totals include the first decode step but exclude loading,
phase-logit writes, the final snapshot, sampling and tools. Process wall includes
loading, phase-logit writes and the final snapshot. The fixed replay does not
measure autonomous tool choices or complete agent task speed. A separate
full-vocabulary pair compares teacher logits on the same histories; delta-NLL and
KL describe distribution differences rather than task accuracy.

One reasoning-enabled fresh interval coding task uses the independent oracle,
with matched working directory, prompt, time, sampling and fresh prefix-cache
conditions. The plain-prompt `merge_ranges` task uses seed 2, temperature 1,
top-p 1, min-p 0.05, reasoning enabled and a 2,048-token per-call limit. It is a
practical behavior check, not a task-latency comparison or a general accuracy
estimate. The earlier long-prompt task remains separate attribution evidence.

## Recorded-session timing

Five alternating pairs produced mean inference times of 179.111 ± 4.506 seconds
baseline and 175.888 ± 0.778 seconds candidate, a 1.800% difference between
means. The candidate was faster in four pairs. The table reports seconds and
sample standard deviations; sample lists are in pair order.

| Measure | Baseline mean ± SD | Candidate mean ± SD | Baseline samples | Candidate samples |
| --- | ---: | ---: | --- | --- |
| Initial prefill | 15.101 ± 0.407 | 14.816 ± 0.048 | 15.813, 14.832, 14.848, 14.984, 15.026 | 14.898, 14.800, 14.808, 14.774, 14.800 |
| Appended prefill | 57.434 ± 2.130 | 58.197 ± 0.161 | 53.627, 58.486, 58.432, 58.221, 58.403 | 58.104, 58.279, 58.310, 58.331, 57.958 |
| Decode | 106.577 ± 2.780 | 102.875 ± 0.724 | 101.612, 107.803, 107.996, 107.891, 107.582 | 101.927, 102.485, 103.299, 103.806, 102.861 |
| All inference phases | 179.111 ± 4.506 | 175.888 ± 0.778 | 171.053, 181.121, 181.276, 181.096, 181.011 | 174.929, 175.564, 176.417, 176.911, 175.620 |
| Process wall | 181.159 ± 4.502 | 177.965 ± 0.662 | 173.110, 183.228, 183.377, 183.052, 183.028 | 177.275, 177.633, 178.444, 178.866, 177.610 |

Paired inference changes were +2.266%, −3.068%, −2.680%, −2.311% and
−2.978%, with negative values faster. Baseline ran first in pairs 1, 3 and 5;
candidate ran first in pairs 2 and 4. The first baseline was unusually fast
overall and in appended prefill and decode, while its candidate was slower
overall. This retained first pair and the small number of pairs limit what can
be concluded about run order. Initial prefill favored the candidate in all five
pairs; decode favored it in four, and appended prefill was mixed. The local
`replay-timing/analysis.json` retains phase and first-step samples. These results
do not establish a general autonomous-agent speedup or attribute the change to
an individual optimization by comparison with experiment 13.

The timing guard completed successfully. Memory pressure remained at level 1,
swap decreased slightly from 28.908 to 28.900 GiB, and no adverse pressure
samples were recorded. The quality and practical guards also recorded level 1
throughout, with swap at 28.900 GiB and no adverse samples.

First-pair counters recorded 134,840 baseline and 8,440 candidate ordinary-route
layer calls, consistent with refresh every 16 positions. The counters omit GPU
substitutions, so they do not compare total replacements or bytes skipped.

## Fixed-history quality

The full-vocabulary pair produced 3,371 aligned, finite decode rows. Mean
candidate-minus-baseline negative log likelihood (delta-NLL) was +0.037790
nats per teacher token, and mean KL(baseline || candidate) was 0.030566 nats.
The highest-probability token changed in 126 rows (3.738%). Delta-NLL had a
95th percentile of +0.349430, a 99th percentile of +1.167536 and a maximum of
+6.585262 nats. KL had a 95th percentile of 0.159798, a 99th percentile of
0.440915 and a maximum of 3.051197 nats. These metrics describe differences on
fixed histories, not practical accuracy or diverging generated trajectories.

Direct binary comparison found both this baseline's logits and this
candidate's logits byte-identical to their respective experiment 13 captures;
each file contained 1,743,211,520 bytes. The local
`prior-capture-equality.json` records the comparison. Thus the changes since
experiment 13 preserved its measured distributions on this replay. The
indexer-query matrix projection is expected to be inactive here, so this does
not establish exactness for large prefills.

## Practical coding behavior

Both agents completed `intervals.py` and exited successfully. The independent
oracle tested 207 cases in list, iterator and generator forms. The baseline
passed all 621 evaluations; the candidate failed 30 evaluations across 10
cases. Its merge condition compared `start <= merged[-1][1]`, omitting the `+ 1`
needed to join consecutive integer ranges. For example, it left `(1, 2)` and
`(3, 4)` separate where the oracle expects `(1, 4)`. The candidate's reasoning
interpreted adjacency as touching endpoints, and its own passing tests missed
consecutive integers. The practical guard returned child code 1 because of
the oracle failure, not memory pressure. There was no retry in this slice.

The 30 failures are evaluations of one generated implementation, not 30
independent generations. This check neither establishes general accuracy nor
proves that GPU routing caused the error.

## Decision and reproduction

Keep the useful accumulated changes opt-in for continued optimization. The
candidate's favorable timing mean and lower observed variation are qualified by
the first-pair uncertainty, and the coding failure remains evidence against
promotion on this qualification. No presets, installed binaries or published
main changed. The next selected question should separately qualify the prefill
selectors from experiments 17 and 18 with the existing CPU-routing fast preset;
this replay does not exercise the indexer-query matrix path.

Raw evidence, including the practical task and driver, is local under
`.workspaces/hillclimb-20261104/19-package-requalification/`. With the recorded
binaries and checkpoint in place, use fresh output and guard directories for
each stage. The following reproduces the timing stage; replace `replay-timing`
with `replay` and use fresh directories for the full-vocabulary quality pair.
The per-arm manifests record the settings, arguments, source roots and hashes.

```sh
python3 speed-bench/ideas/pressure_run.py GUARD_DIR -- nu speed-bench/experiments/2026-11-04-decode-pass/combined.nu OUTPUT_DIR replay-timing \
  --baseline-root .workspaces/hillclimb-20261104/starting-364ceeab \
  --candidate-root . --baseline-revision 364ceeab --candidate-revision b9bc4fb5 \
  --baseline-settings speed-bench/experiments/2026-11-04-decode-pass/final-settings.json \
  --final-settings speed-bench/experiments/2026-11-04-package-requalification/candidate-settings.json
```

The driver does not build binaries. Keep each binary in its matching source root,
run one heavy model job under the pressure guard, and pause builds and unrelated
work during final timing.
