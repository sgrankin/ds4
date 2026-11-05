# Prefill options with CPU resident routing

**Adopt both prefill options in the resident-fast and persistent-fast presets**, retaining explicit zero overrides. Across the tested workloads, fresh prefill was faster in all 15 pairs and the inference phase sum was faster in 14. Fixed-history drift was nonzero and both coding implementations passed the oracle. Post-append decode was consistently slower in the gap-zero 8K comparison; its cause remains unresolved and is a tradeoff of this decision. Context-fast remains unchanged.

## Question and setup

The [earlier fusion experiment](../2026-11-04-prefill-followup/README.md) found bitwise equality on its fixtures and fixed histories, but no established whole-model gain. The [query projection experiment](../2026-11-04-prefill-f16-projections/README.md) measured faster prefill with numerical drift. Both held experimental GPU resident routing on. The subsequent [package replay](../2026-11-04-package-requalification/README.md) did not exercise the query matrix path and cannot settle this preset decision.

Both arms use runtime `909faab5` at checkout `7a43e7eb`, the same benchmark binary (SHA-256 `a8fad44806932fad5e041693b7001b913e147751e08e1790a58b3b067f7fb5e9`), the DeepSeek V4.1 Flash IQ2_XXS/Q2_K checkpoint, a 64 GB expert cache and 8,192-token prefill chunks. The machine is an Apple M5 Max with 128 GiB memory. The [baseline settings](../2026-11-04-combined-prefill/candidate-settings.json) retain CPU resident routing and the qualified RB4 attention and unit-rescale settings. The [candidate settings](candidate-settings.json) add only `DS4_METAL_V41_PREFILL_ROPE_BF16=1` and `DS4_METAL_V41_PREFILL_INDEX_Q_MPP=1`; GPU resident routing and dependent options remain off in both arms. The agent binary used for the practical task has SHA-256 `d06a2f32ee4304bb48605f38fa5139fc322e6cc98c8ee18cc80cb7a5d2c8eaa0`.

The fused path rounds each attention-output column to BF16 before inverse RoPE rotates the tail of each head, eliminating a separate pass while preserving that rounding boundary. The query option uses batched matrix multiplication for eligible F16 temporary indexer-query projections instead of row-exact projection. Changed queries can alter compressed-key selection and later state. Its activation conditions and numerical limits are described in the projection report.

Five profile-free pairs alternate order at each of 8,192 and 32,768 tokens. Persistent-fast sets the prefill resident gap to `.90`, whereas resident-fast leaves it unset, equivalent to `0`. Full layer sweeps keep native routes under either setting, but selected-small appends differ. To cover that distinction before deciding on both presets, another five 8,192-token pairs and a 32,768-token quality pair set `DS4_METAL_V41_RESIDENT_PREFILL_GAP=0` in both arms. Separate full-vocabulary quality pairs compare 768 total teacher-token rows on fixed histories: 512 across the main 8K and 32K pairs and 256 in the gap-zero 32K pair. One controlled, reasoning-enabled long-background coding task per arm checks practical behavior with matched task path, time, sampling and fresh prefix-cache conditions. Fixed-history delta-NLL and Kullback–Leibler (KL) divergence describe distribution drift, not practical accuracy or diverging generated histories. Raw artifacts and commands are retained locally under `.workspaces/hillclimb-20261104/20-prefill-preset-qualification/`.

## Fixed-history quality

The main CPU-routing quality pairs produced 128 full-vocabulary temperature-one teacher-token predictions starting at each frontier, the indicated prompt length. The second frontier follows a 512-token append. Delta-NLL is candidate minus baseline negative log likelihood in nats per teacher token; positive values indicate lower likelihood for the teacher token. KL is `KL(baseline || candidate)`. The table gives mean, 95th percentile, 99th percentile and maximum per prediction.

| Frontier | Delta-NLL mean / p95 / p99 / max | KL mean / p95 / p99 / max | Top-1 changed |
| ---: | ---: | ---: | ---: |
| 8,192 | -0.01215 / 0.15153 / 0.65269 / 0.81263 | 0.01244 / 0.05085 / 0.14428 / 0.36820 | 2 / 128 |
| 8,704 | -0.01649 / 0.09872 / 0.26968 / 0.80814 | 0.01603 / 0.06974 / 0.22062 / 0.40693 | 4 / 128 |
| 32,768 | 0.00303 / 0.34211 / 0.46454 / 0.51783 | 0.01612 / 0.07189 / 0.11016 / 0.14375 | 9 / 128 |
| 33,280 | 0.00189 / 0.26176 / 0.60437 / 1.09158 | 0.01242 / 0.04658 / 0.06166 / 0.09792 | 3 / 128 |

All 512 compared logit rows differed, and no post-evaluation hashes matched. Negative mean delta-NLL on these histories does not establish a general quality improvement. These measurements describe fixed-history distribution drift, not task accuracy or behavior after sampled histories diverge. With the prefill gap set to zero, the additional quality pair gave:

| Frontier | Delta-NLL mean / p95 / p99 / max | KL mean / p95 / p99 / max | Top-1 changed |
| ---: | ---: | ---: | ---: |
| 32,768 | 0.00303 / 0.34211 / 0.46454 / 0.51783 | 0.01612 / 0.07189 / 0.11016 / 0.14375 | 9 / 128 |
| 33,280 | -0.01583 / 0.32199 / 0.72055 / 0.79257 | 0.01371 / 0.05326 / 0.09461 / 0.09971 | 5 / 128 |

The initial 32,768-token results match the `.90` pair exactly, consistent with full sweeps retaining native routes under both settings. The append differs. The 768 paired predictions cover six condition/frontier cells; the initial 32K positions are repeated across settings, so they are not 768 independent unique positions. All three quality guards completed successfully.

## Profile-free timing

Five alternating pairs measured fresh prefill, 128 teacher-forced decode tokens, a 512-token append after KV restore and another 128 teacher-forced decode tokens. The inference sum excludes model loading and snapshot save and restore; process wall includes them. Expert state survives the KV restore and the OS file cache was not flushed. Seconds are derived from throughput rounded to two decimal places. Complete phase and first-token samples remain in the local CSV files.

| Context | Pair | First arm | Fresh prefill baseline → candidate (s) | Inference sum baseline → candidate (s) |
| --- | ---: | --- | ---: | ---: |
| 32K | 1 | Baseline | 47.912 → 45.910 | 64.117 → 61.182 |
| 32K | 2 | Candidate | 48.663 → 46.420 | 64.766 → 62.229 |
| 32K | 3 | Baseline | 49.277 → 47.989 | 65.424 → 64.224 |
| 32K | 4 | Candidate | 49.889 → 48.574 | 66.226 → 64.868 |
| 32K | 5 | Baseline | 49.920 → 48.761 | 66.231 → 65.103 |
| 8K | 1 | Baseline | 19.853 → 18.217 | 34.462 → 33.269 |
| 8K | 2 | Candidate | 19.041 → 18.425 | 34.712 → 33.790 |
| 8K | 3 | Baseline | 19.120 → 19.060 | 34.001 → 34.259 |
| 8K | 4 | Candidate | 19.827 → 19.352 | 35.504 → 34.859 |
| 8K | 5 | Baseline | 20.026 → 19.645 | 35.865 → 35.700 |
| Gap 0, 8K | 1 | Baseline | 20.092 → 18.900 | 40.006 → 39.148 |
| Gap 0, 8K | 2 | Candidate | 19.393 → 19.180 | 39.110 → 38.848 |
| Gap 0, 8K | 3 | Baseline | 19.824 → 19.629 | 39.950 → 39.838 |
| Gap 0, 8K | 4 | Candidate | 20.398 → 19.952 | 40.740 → 39.747 |
| Gap 0, 8K | 5 | Baseline | 20.371 → 19.990 | 40.880 → 40.078 |

At 32K, fresh prefill averaged 49.132 ± 0.855 s baseline and 47.531 ± 1.291 s candidate (sample standard deviation). The candidate was faster in every pair, by 4.178%, 4.610%, 2.614%, 2.636% and 2.321%, respectively. Inference sums averaged 65.353 ± 0.923 s and 63.521 ± 1.728 s, also favoring the candidate in all five pairs. Process wall averaged 67.523 and 65.694 s and favored the candidate in all five. Both arms slowed over the run, and the candidate's advantage was larger in the first two pairs; append and decode differences were less consistent.

At 8K, fresh prefill averaged 19.573 ± 0.457 s baseline and 18.940 ± 0.606 s candidate, faster in all five pairs. Inference sums averaged 34.909 ± 0.763 s and 34.375 ± 0.945 s, faster in four pairs. Append means were 7.130 and 7.249 s, with mixed, order-dependent results; decode differences were mixed and near neutral. With prefill gap zero at 8K, fresh prefill averaged 20.016 ± 0.419 s and 19.530 ± 0.479 s, and inference sums averaged 40.137 ± 0.711 s and 39.532 ± 0.513 s; the candidate was faster in all five pairs for both. Append means were 11.971 and 11.784 s, with mixed results. Post-append decode was slower in all five pairs: 3.702 ± 0.107 s versus 3.781 ± 0.097 s, with paired changes +5.923%, +0.085%, +3.385%, +0.571% and +0.930%. Mean steady throughput fell from 34.670 to 33.942 tokens/s. A bounded, instrumented gap-zero diagnostic completed successfully. During post-append decode, 1,767 of 5,120 acquired expert ID lists differed; cache misses changed from 56 to 57, binding time from 46.631 to 43.717 ms, and parallel pread batches remained zero in both arms. Complete 64-command-buffer GPU blocks strictly inside the phase took 3,462.9 versus 3,472.5 ms across the same 10,176 command buffers. Boundary blocks and the tail are omitted, so these are not total GPU time. These observations do not explain the roughly 80 ms mean decode difference: logging perturbs timing, and earlier router waits and selected loads are not fully included. The cause remains unresolved. These observations are limited to the measured workloads.

## Practical coding task

Both agents completed the long-background topological-layers task and exited successfully. An independent oracle checked each implementation on 58 graphs in list, iterator and generator forms; both passed all 174 evaluations. Each first prompt contained 21,337 tokens, with 1,859 cached and 19,478 suffix tokens, exercising a full prefill chunk. This is one generated task per arm, not 174 independent model trials or a task-latency comparison. The oracle results and complete task traces are in the local `practical-check` directory.

## Decision and reproduction

Enable the two selectors together in resident-fast and persistent-fast, preserving explicit `0` overrides. Keep context-fast unchanged. The timing benefit, measured drift and two passing coding implementations support adoption for these presets, with the gap-zero post-append decode regression retained as an unresolved limitation rather than attributed to the counters. The measurements do not isolate the individual selectors or establish general task accuracy.

The server build passed without warnings. The launcher test passed for enabled defaults, explicit zero overrides, unchanged context-fast and server settings. All three root binaries were installed, hash-verified and passed `--help`; the previous binaries are archived locally and `installed-binaries.json` records the installed hashes. Published main was not advanced. All eight guarded jobs exited successfully with pressure level 1 and no adverse samples; swap remained nonincreasing within 28.89246–28.90027 GiB. The local raw root retains the commands, manifests, complete samples and diagnostic summaries. The gap-zero arms use the linked settings with `DS4_METAL_V41_RESIDENT_PREFILL_GAP=0` in both arms. Run one model job at a time under the existing pressure guard and pause unrelated work during profile-free timing.

With the recorded binary and checkpoint in place, use fresh output and guard directories to reproduce the 32K timing comparison:

```sh
python3 speed-bench/ideas/pressure_run.py GUARD_DIR -- nu speed-bench/experiments/2026-11-04-decode-pass/combined.nu OUTPUT_DIR timing \
  --baseline-root . --candidate-root . --baseline-revision 7a43e7eb --candidate-revision 7a43e7eb \
  --baseline-settings speed-bench/experiments/2026-11-04-combined-prefill/candidate-settings.json \
  --final-settings speed-bench/experiments/2026-11-04-prefill-preset-qualification/candidate-settings.json \
  --frontier 32768 --samples 5 --tokens 128
```

Use `--frontier 8192` for 8K, or replace `timing` with `quality` for the fixed-history pair. For gap zero, copy both settings files into a fresh local workspace and set `DS4_METAL_V41_RESIDENT_PREFILL_GAP` to `"0"` in each, then point the command to those copies. The driver does not build binaries.
