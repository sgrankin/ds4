# Accumulated package: long-session qualification

**Keep the selected GPU resident-routing package experimental.** Against published `364ceeab`, this
qualification measured a modest fixed-replay gain and no additional fixed-history
drift since experiment 19. The repeated interval failure and two fresh tasks do
not warrant preset promotion. The cached-refresh prototype remains off; no runtime
change is part of this comparison.

## Setup and scope

The November 5, 05:04–06:04 Eastern slice uses an M5 Max with 128 GiB and
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`. The baseline uses CPU
resident routing capped at five substitutions; the candidate uses GPU
resident-only routing with refresh every 16 positions, four-layer submission,
queue-attached slab residency with shared-event overlap off, and the existing
prefill changes. The exact [baseline](../2026-11-04-decode-pass/final-settings.json)
and [candidate](../2026-11-05-slab-event-interaction/candidate-settings.json)
settings define the comparison. It uses runtime `c1982bc8`, with closure
revision `5c87f08e`, and package 25's explicit candidate settings. The cached
refresh parallel prototype from experiment 27 is off. The baseline uses the
published `364ceeab` binary and settings, each binary running from its matching
source root because Metal sources resolve there. The candidate replay and agent SHA-256 hashes are
`a382578f28be0335589f8264797060c5ce08f8cbfae1026254c759efb920e77b` and
`bf3a031fe5b5f84ca114d4d7aeb08c663d860df17f72eef7fb20c3baa27d4bd8`;
the baseline hashes are
`bd36bcb5a8b528bef2e1b733f26a9f143c16f17fffb5b981456b7dae47bb37ab` and
`aafbcade3df6a99826e1b8fa96dd92acbabf107ca5c50246b09e5237824c227e`,
respectively. Raw evidence is retained under
`.workspaces/hillclimb-20261104/28-long-session/`.

The fixed recorded session has 15 prefill phases totaling 5,339 tokens and 14
decode phases totaling 3,371 teacher-forced evaluations. Five alternating
pairs measure initial and appended prefill, complete decode, per-phase first
steps, inference total and process wall. Inference phase timings exclude model
loading and process work; wall includes them. Fixed tokens and tool results make
this a replay measurement, not autonomous agent speed. The initial prefill is
1,859 tokens and later appends use the selected-small path; this is not a
large-prefill qualification and is not expected to exercise the index-query
matrix path.

A separate full-vocabulary comparison targets all 3,371 aligned rows. The
interval coding task provides a practical check with reasoning enabled. In the
[previous package qualification](../2026-11-04-package-requalification/README.md),
the baseline passed 621 evaluations and that candidate failed 30 adjacency
cases; this is context for the fresh trial, not a prediction. The long-background
layers task exercises a full prefill chunk. Each task is one trial per arm,
so neither establishes general autonomous task quality.

## Replay timing

The candidate reduced mean inference time by 1.71% and process wall by 1.46%
relative to the baseline means, with four of five pairs lower in each scope.
The first baseline run had slower initial prefill but unusually fast append and
decode; it is retained. Baseline ran first in pairs 1, 3 and 5. These variations
limit the precision of the modest measured benefit. The OS file cache was not
flushed or controlled.

Times below are seconds, baseline → candidate. The sample standard deviation
uses five observations per arm; inference is the sum of the displayed phases.

| Pair (first arm) | Initial prefill | Appended prefill | Complete decode | Inference | Process wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 (baseline) | 17.664 → 15.262 | 55.589 → 56.395 | 101.509 → 103.908 | 174.762 → 175.565 | 176.975 → 178.283 |
| 2 (candidate) | 14.891 → 14.846 | 58.261 → 56.552 | 106.569 → 104.340 | 179.721 → 175.737 | 181.757 → 178.140 |
| 3 (baseline) | 14.719 → 15.090 | 58.518 → 56.854 | 106.832 → 103.677 | 180.068 → 175.622 | 182.180 → 178.152 |
| 4 (candidate) | 14.781 → 14.942 | 58.259 → 56.866 | 106.322 → 104.096 | 179.362 → 175.904 | 181.392 → 178.240 |
| 5 (baseline) | 15.049 → 14.715 | 57.975 → 56.543 | 106.469 → 104.055 | 179.494 → 175.312 | 181.488 → 177.793 |
| Mean ± sample SD | 15.421 ± 1.260 → 14.971 ± 0.213 | 57.720 ± 1.207 → 56.642 ± 0.209 | 105.540 ± 2.261 → 104.015 ± 0.245 | 178.681 ± 2.207 → 175.628 ± 0.219 | 180.759 ± 2.137 → 178.122 ± 0.193 |

First decode steps below are milliseconds, with all five samples in pair order.
The context column is the end of each phase, after all its evaluations. The
candidate is faster on most first steps, but phase 10 is slower in every pair:
34.795 ± 1.815 → 57.790 ± 1.456 ms. The sum of all 14 first steps averages
569.282 ± 15.015 → 505.095 ± 7.809 ms, roughly 64 ms of the 1.525 s mean
complete-decode difference. These steps are distinct from complete decode and
startup. The earlier slab benchmark
saves an in-memory snapshot before fresh generation when another frontier
remains, then restores before warm append; this replay saves only after all 29
phases. Both exclude snapshot time from model phases. These different state
transitions do not establish a cause or imply the benchmark's first-step savings
must recur here.

| Phase (end context) | Baseline samples | Candidate samples | Mean ± sample SD, baseline → candidate |
| --- | --- | --- | ---: |
| 2 (2021) | 90.344, 86.704, 85.820, 92.303, 81.711 | 68.559, 68.934, 65.872, 66.369, 65.794 | 87.376 ± 4.125 → 67.106 ± 1.520 |
| 4 (2396) | 36.279, 35.637, 35.766, 35.722, 35.284 | 30.933, 31.117, 30.675, 30.144, 29.396 | 35.738 ± 0.357 → 30.453 ± 0.695 |
| 6 (2665) | 36.725, 38.411, 36.921, 37.725, 36.806 | 31.256, 31.540, 31.786, 31.755, 31.589 | 37.318 ± 0.730 → 31.585 ± 0.212 |
| 8 (3414) | 62.168, 48.208, 48.369, 48.138, 47.932 | 31.533, 31.393, 31.282, 30.732, 30.976 | 50.963 ± 6.266 → 31.183 ± 0.325 |
| 10 (5182) | 36.177, 32.816, 36.267, 35.906, 32.808 | 60.338, 57.138, 57.625, 57.061, 56.789 | 34.795 ± 1.815 → 57.790 ± 1.456 |
| 12 (5597) | 37.219, 39.027, 39.872, 35.421, 35.607 | 31.349, 31.360, 33.638, 35.038, 31.228 | 37.429 ± 1.995 → 32.523 ± 1.730 |
| 14 (5832) | 43.435, 41.875, 43.072, 42.263, 42.184 | 32.418, 31.976, 33.788, 34.789, 30.071 | 42.566 ± 0.657 → 32.608 ± 1.804 |
| 16 (6005) | 34.647, 34.015, 34.129, 34.605, 34.178 | 30.934, 31.340, 30.945, 31.621, 30.709 | 34.315 ± 0.291 → 31.110 ± 0.365 |
| 18 (7273) | 34.848, 34.341, 34.127, 34.880, 34.303 | 29.861, 30.847, 30.529, 30.975, 30.418 | 34.500 ± 0.342 → 30.526 ± 0.435 |
| 20 (7605) | 35.436, 34.814, 34.768, 34.930, 34.435 | 30.975, 31.435, 31.040, 31.139, 30.914 | 34.877 ± 0.363 → 31.101 ± 0.205 |
| 22 (7891) | 37.776, 36.985, 36.656, 33.883, 33.803 | 32.649, 31.320, 34.311, 34.600, 32.085 | 35.821 ± 1.851 → 32.993 ± 1.420 |
| 24 (8187) | 35.552, 33.833, 35.014, 35.518, 32.934 | 32.250, 35.739, 31.757, 31.308, 28.794 | 34.570 ± 1.149 → 31.970 ± 2.493 |
| 26 (8435) | 35.631, 34.067, 34.308, 34.733, 34.221 | 31.456, 31.438, 36.843, 31.617, 31.301 | 34.592 ± 0.631 → 32.531 ± 2.413 |
| 28 (8710) | 34.483, 33.173, 37.811, 33.579, 33.074 | 31.878, 31.848, 31.648, 31.180, 31.531 | 34.424 ± 1.973 → 31.617 ± 0.283 |

## Fixed-history quality

All 3,371 aligned, finite full-vocabulary rows were compared. Mean candidate
minus baseline NLL was +0.037790 nats per teacher token, corresponding to a
1.03851 perplexity ratio; mean KL(baseline || candidate) was 0.030566 nats.
The top token changed on 126 rows (3.74%); no rows were byte-identical between
arms. Delta-NLL ranged from −2.2404 to +6.5853 nats, with 1st and 99th
percentiles −0.4794 and +1.1675. KL had 99th percentile 0.4409 and maximum
3.0512 nats. The full per-row and per-phase results are in the local quality
analysis.

Both captures match their respective experiment 19 captures byte for byte
on all 3,371 rows with matching metadata. Later changes therefore add no
measured drift on this fixed replay; the difference from published remains.
These teacher-forced rows do not establish practical task quality.

## Practical checks and decision

The interval task baseline passed all 621 oracle evaluations. The candidate
failed 30 evaluations: ten distinct cases each tested as a list, iterator and
generator. Both agents exited successfully, but the candidate implementation
used `start <= merged[-1][1]`, omitting `+ 1` for adjacent inclusive integer
ranges despite its docstring claiming adjacency support. This reproduces the
earlier candidate failure; it is not evidence of a new scheduling regression
or an independent estimate of a broad failure rate. The practical guard recorded
child return code 1 for the oracle failure, with no pressure stop. Generated
code and transcripts are retained locally.

Both arms passed the long-background layers task: 174 of 174 evaluations over
58 graphs and three input forms, with successful agent exits. The first prompt
had 21,337 tokens, including 1,859 cached and 19,478 suffix tokens, exercising a
full prefill chunk. Baseline used three tool rounds and candidate four; their
generated outputs diverged. This is a behavior check, not a latency comparison.
The two fresh tasks do not establish general autonomous task quality.

## Reproduction and evidence

All four guarded stages completed without a pressure stop or adverse samples;
pressure stayed at level 1 and swap ranged from 29.00201 to 29.01690 GiB. The interval oracle
returned 1 for the failure described above; the other stages returned 0. Raw
manifests, all samples, pressure records, quality rows, generated implementations
and transcripts are in `.workspaces/hillclimb-20261104/28-long-session/`.
The baseline and candidate settings are respectively
[`final-settings.json`](../2026-11-04-decode-pass/final-settings.json) and
[package 25 settings](../2026-11-05-slab-event-interaction/candidate-settings.json).
The local candidate copy is equivalent; the cached-refresh parallel flag is unset.
The M5 Max has 128 GiB memory. The replay uses the existing checkpoint, 64 GB
expert-cache target and 8K chunks. Timing includes a final snapshot in process
wall; quality captures full vocabulary and skips that final snapshot.

With the recorded binaries and checkpoint in their matching roots, run the
following from the repository root with a fresh output root (change `out` if it already exists). The driver clears
inherited DS4 settings and alternates arm order; do not build or run another
heavy model process during timing.

```nu
let raw = '.workspaces/hillclimb-20261104/28-long-session'
let out = '.workspaces/hillclimb-20261104/28-long-session-repro'
python3 speed-bench/ideas/pressure_run.py ($out | path join guard-replay-timing) -- nu speed-bench/experiments/2026-11-04-decode-pass/combined.nu ($out | path join replay-timing) replay-timing --samples 5 --baseline-root .workspaces/hillclimb-20261104/starting-364ceeab --candidate-root . --baseline-revision 364ceeab --candidate-revision 5c87f08e --baseline-settings speed-bench/experiments/2026-11-04-decode-pass/final-settings.json --final-settings ($raw | path join candidate-settings.json)
python3 speed-bench/ideas/pressure_run.py ($out | path join guard-replay-quality) -- nu speed-bench/experiments/2026-11-04-decode-pass/combined.nu ($out | path join replay-quality) replay --baseline-root .workspaces/hillclimb-20261104/starting-364ceeab --candidate-root . --baseline-revision 364ceeab --candidate-revision 5c87f08e --baseline-settings speed-bench/experiments/2026-11-04-decode-pass/final-settings.json --final-settings ($raw | path join candidate-settings.json)
python3 .workspaces/overnight-decode-20261104/analyze_replay.py ($out | path join replay-timing) | save ($out | path join replay-timing/analysis.json)
python3 .workspaces/overnight-decode-20261104/compare_replay.py ($out | path join replay-quality)
python3 speed-bench/ideas/pressure_run.py ($out | path join guard-practical-intervals) -- nu ($raw | path join practical-check.nu) ($out | path join practical-intervals)
python3 speed-bench/ideas/pressure_run.py ($out | path join guard-practical-layers) -- nu ($raw | path join practical-layers.nu) ($out | path join practical-layers)
```

The practical drivers and oracles are local snapshots under the raw directory.
They use fresh per-arm prefix caches and the same canonical task directory per
arm. The interval trial uses seed 2, temperature 1 and a 2,048-token per-call
budget; the layers trial uses seed 1 and a 60,000-grapheme background. Quality
analysis used installed Python 3.15 and NumPy 2.5.4 after `uv` fetching failed.
