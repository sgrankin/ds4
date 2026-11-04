# Resident GPU package: agent qualification

**Keep the GPU resident package as an explicit opt-in.** Against published `364ceeab`, the recorded-session replay showed no established complete inference speed gain, and one candidate coding task did not finish within its initial token budget. Earlier fresh decode gains remain useful groundwork for a targeted follow-up. The selected window was November 4, 2026, 17:26–18:26 Eastern; evidence closed a few minutes later. The longer-budget retry also failed to produce the candidate layers implementation.

## Comparison

The baseline is the preserved `364ceeab` checkout with the [original final settings](../2026-11-04-decode-pass/final-settings.json). The candidate uses runtime `1b692d25` and the [candidate settings](candidate-settings.json). It combines the four-row indexed prefill and unit-rescale settings qualified in [experiment 7](../2026-11-04-combined-prefill/README.md) with GPU resident routing, address reuse, concurrent shared and routed experts, and an ordinary-route refresh every 16 positions. Both request the existing 64 GB expert cache. This comparison qualifies the package as a whole; the [resident refresh experiment](../2026-11-04-resident-refresh/README.md) provides the narrower attribution and found fresh decode gains, smaller and noisier warm gains, and fixed-history quality drift.

The runs use an Apple M5 Max with 128 GiB memory, the same DeepSeek V4.1 Flash IQ2_XXS/Q2_K checkpoint and recorded coding session for both arms. The OS file cache was neither flushed nor controlled. Each binary runs in its matching source root, where it loads its Metal sources. The candidate replay binary was rebuilt from the current runtime before measurement. The driver clears inherited `DS4` settings, records each arm's settings and binary hash, uses a 16,384-token context and an 8,192-token prefill chunk, and alternates baseline-first and candidate-first order. All monitored jobs recorded pressure level 1, constant swap at 28.8293457 GiB and zero bad pressure samples. Raw evidence is local and ignored under `.workspaces/hillclimb-20261104/13-resident-agent-qualification/`.

## Recorded-session timing

Five profile-free alternating pairs replayed the recorded prefill and decode phases. Total inference time averaged 180.487 ± 6.080 seconds for the baseline and 180.562 ± 1.458 seconds for the candidate; process wall averaged 182.540 ± 6.140 and 182.653 ± 1.370 seconds. These measurements do not establish a replay speed gain. Values below are seconds, with sample standard deviation; samples follow pair order and include the unusually fast first baseline.

| Measure | Baseline mean ± SD | Candidate mean ± SD | Baseline samples | Candidate samples |
| --- | --- | --- | --- | --- |
| Initial prefill | 15.099 ± 0.314 | 15.205 ± 0.291 | 15.560, 14.782, 15.062, 15.238, 14.853 | 14.979, 15.294, 15.291, 15.597, 14.862 |
| Appended prefill | 57.280 ± 2.663 | 58.170 ± 0.469 | 52.519, 58.586, 58.494, 58.327, 58.473 | 57.388, 58.239, 58.169, 58.589, 58.465 |
| Decode | 108.108 ± 3.717 | 107.188 ± 0.852 | 101.673, 109.996, 111.130, 108.893, 108.848 | 105.837, 107.056, 107.977, 107.853, 107.216 |
| All inference phases | 180.487 ± 6.080 | 180.562 ± 1.458 | 169.752, 183.364, 184.686, 182.458, 182.174 | 178.204, 180.589, 181.437, 182.038, 180.543 |
| Process wall | 182.540 ± 6.140 | 182.653 ± 1.370 | 171.715, 185.421, 186.870, 184.475, 184.219 | 180.496, 182.597, 183.555, 184.075, 182.541 |

The candidate's paired inference changes were +4.979%, −1.513%, −1.759%, −0.230% and −0.895%, where negative is faster. Baseline ran first in pairs 1, 3 and 5: their mean paired change was +0.775%; the two candidate-first pairs averaged −0.872%. These small groups and the fast first baseline do not establish an order effect. Decode averaged slightly less for the candidate, while appended prefill averaged more. The inference total includes every recorded phase, including the first decode step. The local `analysis.json` retains individual phase and first-step samples.

Inference phase totals cover model calls and exclude loading, phase-logit writes, the final snapshot, sampling and tools. Process wall includes loading, phase-logit writes and the final snapshot. A recorded replay uses fixed histories and excludes autonomous tool use and sampling; it does not measure complete agent task speed. Ordinary-route counters in the first pair recorded 134,840 baseline and 8,440 candidate layer calls, consistent with refresh every 16 positions and an active GPU route. Those counters omit GPU substitutions and do not compare total substitutions or bytes saved.

## Fixed-history quality

One full-vocabulary replay pair compared the same recorded histories. All 3,371 decode rows aligned and were finite. Mean candidate-minus-baseline negative log likelihood (delta-NLL) was +0.037790 nats per token, corresponding to 3.851% higher teacher-token perplexity; mean KL(baseline || candidate) was 0.030566 nats. The top-choice token changed in 126 rows (3.738%).

Delta-NLL had a 95th percentile of +0.34943, a 99th percentile of +1.16754 and a maximum of +6.58526 nats per token. KL had a 95th percentile of 0.15980, a 99th percentile of 0.44091 and a maximum of 3.05120 nats. Drift varied nonmonotonically across phases: phase 16 had the highest mean KL at 0.06950, while phase 20 had 0.000183. The local `replay-quality/analysis.json` and `rows.csv` retain the phase and row details. These fixed histories describe distribution differences, not practical accuracy or diverging generated trajectories.

## Practical coding behavior

Each reasoning-enabled agent received two independently checked coding tasks without an encoder timeline: no instrumentation split the normal encoders. The interval-merging oracle extends the prior list checks to iterators, generators and input mutation. The baseline completed both tasks and passed 621 interval and 174 layers checks. The candidate passed all 621 interval checks, but did not produce a file for the layers task within the configured 1,024-token call limit. Its output contained reasoning that repeatedly considered validation and the algorithm, with no tool call or implementation; the process exited successfully. This result concerns the test’s 1,024-token-per-call budget and does not establish that the agent normally fails the task. This is a failure to complete within the budget, not a test failure of a finished algorithm. The 795 oracle evaluations per arm describe coverage, not independent task trials.

A separate 2,048-token layers-only retry passed all 174 checks for the baseline. The candidate again exited successfully without creating a file; its output repeated “Need maybe if nodes is a generator that yields ints and edges generator yields pairs. Fine.” 21 times before truncation. The retry changed the working directory as well as the budget, so it is neither a controlled continuation nor a pure test of the budget. The original budgeted result remains part of the evidence. Both behavior guards returned code 1 because of oracle failures. The initial retry launch also failed before model work with a Nushell list-update type error and no pressure samples; the ignored driver was corrected before the retry. The local `agent-check/oracle-result.json` and `agent-check-long/oracle-result.json` retain the checks. These individual tasks assess behavior, not comparative task speed, and do not require identical outputs.

## Decision

Keep the GPU resident package explicit opt-in; do not change the resident fast presets or replace installed binaries. The earlier fresh decode gains warrant targeted GPU follow-up, but this qualification did not establish a complete recorded-session gain, and the candidate failed to complete one budgeted coding task. This decision does not reject the package merely because its fixed-history distributions differ. Runtime source was unchanged in this experiment; publication and advancing main are outside its scope.

A diagnostic CPU sampling pair used the original settings without an encoder timeline or added engine profiling callbacks. Each sample covered three seconds at 1 ms intervals, starting two seconds after phase 17 prefill during phase 18 decode. The first sampling attempt safely refused to proceed because its process filter matched both the timing wrapper and replay; the filter was corrected. The candidate sample came from the original diagnostic run and the baseline from a separate rerun. The categories below are mutually exclusive main-thread stack samples. Selected-ID commit/wait means submitting and waiting for GPU router IDs needed by the host; token completion means waiting for submitted token work. Candidate resource declarations appeared in 13 nested samples, and no address-publication samples appeared.

| Sample category | Baseline | Candidate |
| --- | ---: | ---: |
| Selected-ID commit/wait | 1,120 | 116 |
| Token completion | 58 | 978 |
| Readiness | 0 | 6 |
| Routed dispatch | 21 | 40 |
| Other | 75 | 133 |
| Total | 1,274 | 1,273 |

Host waits shifted under native grouping, which preserves the ordinary command-buffer and encoder boundaries. These counts do not measure milliseconds saved or identify a GPU kernel, and this diagnostic pair is not a comparative performance measurement. The next priority is a controlled same-working-directory comparison of routing policies to investigate the behavior. A useful subsequent GPU diagnostic would distinguish already-submitted GPU execution from queue gaps during phase 18; further CPU preparation cleanup is not established as the next bottleneck. Local `cpu-findings.txt` and `cpu-findings.json` retain the details.

## Reproduction

The replay uses the existing guarded driver from the candidate root, with a fresh output and guard directory. Replace `replay-timing` with `replay` and use fresh directories for the single full-vocabulary pair. The per-arm manifests record exact settings, arguments, roots and binary hashes.

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/hillclimb-20261104/13-resident-agent-qualification/replay-timing-guard \
  -- nu speed-bench/experiments/2026-11-04-decode-pass/combined.nu \
  .workspaces/hillclimb-20261104/13-resident-agent-qualification/replay-timing replay-timing \
  --baseline-root .workspaces/hillclimb-20261104/starting-364ceeab \
  --candidate-root . \
  --baseline-revision 364ceeab --candidate-revision 1b692d25 \
  --baseline-settings speed-bench/experiments/2026-11-04-decode-pass/final-settings.json \
  --final-settings speed-bench/experiments/2026-11-04-resident-agent-qualification/candidate-settings.json
```

The driver does not build binaries. Keep one heavy model job under the pressure guard and pause builds and unrelated work during final latency measurements.
