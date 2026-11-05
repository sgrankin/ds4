# Profiling after SIMD resident selection

The profile identified substantial GPU work and host completion waiting, but did not establish GPU utilization or a SIMD speed difference. It motivated a small exact fusion of native selection and resident repair. Retain the fusion as an exact opt-in: fixtures and fixed-history quality checks pass, and isolated tests reduce GPU and CPU encoding work for missing experts. Model timings do not establish an end-to-end gain, so presets and installed binaries are unchanged.

## Question and setup

This November 4 experiment asked what limits decode after SIMD resident selection and whether one small change could shorten it. The one-hour slice ran from 20:17 to 21:17 Eastern. The [prior SIMD experiment](../2026-11-04-resident-simd/README.md) retained its selection kernel as an opt-in with mixed model timings.

The initial diagnostic used runtime `e5d12861`. Both measured fusion timing arms used the same `b006c4cf` binary, toggling fusion off and on; the archived `e5d12861` outputs were checked separately. Both timing arms used GPU resident routing, SIMD repair, refresh every sixteen tokens, address reuse, parallel FFN, the existing prefill settings and a 64 GB expert cache. The machine was an M5 Max with 128 GiB memory; the model was GGUF 1017 IQ2_XXS-Q2_K-imatrix, with context 40,960 and chunk size 8,192. The profile used a 32K workload. The benchmark driver is [run.nu](../2026-11-04-decode-pass/run.nu), with `fusion-settings.json` in the local experiment artifacts. The benchmark binary SHA-256 was `f0db01e1536640a03495c8ed810f3e9c07d89642a9ec5800f04dbb11fa3b3177`; the fixture binary SHA-256 was `67bdc003330c8985fd02d0faa9436da854e64df9528f5cae2b60836ee40c93a8`.

## What the profile shows

One instrumented scalar-versus-SIMD pair used native command grouping and CPU sampling. Both arms showed substantial submitted GPU spans and main-thread completion waiting. These measurements are diagnostic: cumulative spans are not GPU utilization, and host commit-to-completion includes waits beyond the terminal buffer whose GPU span is recorded. Subtracting those values would not establish idle time. The detailed command-buffer limit of 400 was exhausted during early fresh decode, so it does not cover warm decode. The relative GPU-start field had incompatible clock origins and was discarded.

The CPU sample watcher initially missed the relative benchmark path. There is no baseline fresh capture: the baseline file named `cpu-fresh.sample.txt` is actually a late warm sample, and the file named `cpu-warm.sample.txt` overlaps teardown and was excluded. Candidate fresh and warm captures were valid. Different capture offsets and sampling interference prevent a controlled comparison of their stack percentages.

A separate encoder timeline ranked operations but altered batching and added logging overhead. Counter allocation failures left some encoder records missing, so recorded category sums undercount. The complete command-buffer counts still show 40 command buffers per active token. Four parallel FFN kernels share an encoder, so the timeline cannot split their individual times. This evidence and source inspection supported testing the narrower selection/repair fusion; they did not justify broad FFN rescheduling.

Detailed profile notes, parsed JSON, CPU sampling metadata and timeline findings are retained locally under `.workspaces/hillclimb-20261104/16-post-simd-profile/`. Raw logs remain ignored. Neither this single pair nor the intrusive timeline supports a comparative performance claim.

## Candidate and fixture

The native router first chooses experts and their mixture weights. On eligible non-refresh resident decode, a second GPU dispatch repairs selected IDs that are not resident, retaining the native weights. The candidate combines those operations in one dispatch, using a full-threadgroup barrier before repair so the native results are available to all participating lanes. It preserves the native weights and the SIMD repair's selected-ID semantics. Implementation is in [dsv41.metal](../../../src/metal/dsv41.metal), [v41_dispatch.zig](../../../src/metal/v41_dispatch.zig) and [moe.zig](../../../src/engine/deepseek41/moe.zig).

`DS4_METAL_V41_RESIDENT_GPU_FUSED=1` opts in when GPU resident routing and SIMD repair are also enabled; the fused path applies only to eligible non-refresh resident work. The guarded `--resident-router` and `--resident-router-fused` fixtures passed. They checked exact native probabilities and weights and selected IDs, including total expert counts 6, 31, 33, 256, 257, 384 and 1024. The activation check observed the intended reduction from two dispatches to one.

## Model quality and timing

Fixed-history checks compared 128 full-vocabulary rows at each of positions 8,192, 8,704, 32,768 and 33,280. All 512 rows and all 512 post-evaluation hashes matched byte for byte between fused and unfused execution; delta-NLL and KL were zero on these histories. Pipeline-creation logs identify standalone SIMD repair for the baseline and `kernel_dsv41_router_resident_one` for the candidate. A separate comparison of this revision's unfused control against archived `e5d12861` SIMD outputs also matched all 512 rows and hashes, checking the helper extraction. Teacher metadata and inputs matched at both context sizes, including vocabulary, frontier, prompt IDs, steps, dtype and byte order. These checks found no extra drift from the fusion; the underlying GPU resident routing policy is already approximate relative to native/CPU routing.

Five alternating profile-free pairs at 32K completed. Each run evaluates the fresh frontier and 128 teacher decode tokens, restores KV, appends 512 tokens and evaluates another 128 teacher tokens. The expert cache survives restore; the OS cache was not flushed. Phase times exclude loading and snapshot save/restore. Steady decode excludes the first token; seconds are derived from two-decimal throughput values. The total sums fresh and warm prefill and decode, excluding startup and snapshot operations.

The table retains every pair in run order. B and C are baseline and candidate; first alternated B, C, B, C, B. First-token values are milliseconds, complete decode and total are seconds, and steady rates are tokens/s.

| Pair | Fresh first B / C | Fresh complete B / C | Fresh steady B / C | Warm first B / C | Warm complete B / C | Warm steady B / C | Total B / C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 865.261 / 845.124 | 4.479 / 4.498 | 35.140 / 34.770 | 32.719 / 33.680 | 3.646 / 3.707 | 35.150 / 34.570 | 62.494 / 62.491 |
| 2 | 848.621 / 823.477 | 4.378 / 4.339 | 35.990 / 36.130 | 33.828 / 33.278 | 3.780 / 3.719 | 33.900 / 34.460 | 63.682 / 62.608 |
| 3 | 827.208 / 821.885 | 4.360 / 4.329 | 35.960 / 36.220 | 34.376 / 34.517 | 3.865 / 3.937 | 33.150 / 32.540 | 65.011 / 65.386 |
| 4 | 811.627 / 787.600 | 4.342 / 4.301 | 35.970 / 36.140 | 34.036 / 36.345 | 3.873 / 3.949 | 33.080 / 32.450 | 65.851 / 65.737 |
| 5 | 802.908 / 807.879 | 4.310 / 4.308 | 36.210 / 36.280 | 33.847 / 35.014 | 3.959 / 3.984 | 32.350 / 32.170 | 66.143 / 66.146 |

Fresh steady throughput averaged 35.854 ± 0.412 versus 35.908 ± 0.639 tokens/s; paired time changed −0.141 ± 0.699%, with four of five pairs favorable. Warm steady averaged 33.526 ± 1.061 versus 33.238 ± 1.174 tokens/s; paired time changed +0.886 ± 1.511%, with four of five slower. Warm throughput declined substantially through run order in both arms. Complete fresh decode averaged 4.374 ± 0.064 versus 4.355 ± 0.081 seconds, warm decode 3.825 ± 0.118 versus 3.859 ± 0.135 seconds, and total inference 64.636 ± 1.532 versus 64.474 ± 1.777 seconds. Variation is sample standard deviation. These results do not establish a whole-model win or regression.

The same five-pair procedure at 8K produced the following samples, with the same units and alternating order:

| Pair | Fresh first B / C | Fresh complete B / C | Fresh steady B / C | Warm first B / C | Warm complete B / C | Warm steady B / C | Total B / C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 802.855 / 942.813 | 4.243 / 4.388 | 36.930 / 36.860 | 32.992 / 33.793 | 3.700 / 3.624 | 34.630 / 35.370 | 34.894 / 35.404 |
| 2 | 807.781 / 836.214 | 4.245 / 4.268 | 36.940 / 37.000 | 33.646 / 35.361 | 3.778 / 3.741 | 33.920 / 34.270 | 35.089 / 35.025 |
| 3 | 788.269 / 804.683 | 4.223 / 4.234 | 36.970 / 37.030 | 34.983 / 33.378 | 3.794 / 3.821 | 33.780 / 33.530 | 35.394 / 35.786 |
| 4 | 822.778 / 821.860 | 4.271 / 4.248 | 36.830 / 37.060 | 33.178 / 33.806 | 3.840 / 3.785 | 33.360 / 33.860 | 35.713 / 35.712 |
| 5 | 830.311 / 791.719 | 4.262 / 4.208 | 37.000 / 37.180 | 34.225 / 34.998 | 3.838 / 3.811 | 33.380 / 33.630 | 35.837 / 35.765 |

Fresh steady averaged 36.934 ± 0.064 versus 37.026 ± 0.115 tokens/s, with four of five pairs faster; warm steady averaged 33.814 ± 0.518 versus 34.132 ± 0.748, also four of five faster. Complete fresh decode averaged 4.249 ± 0.019 versus 4.269 ± 0.070 seconds, with three of five slower; the candidate's 942.813 ms first step is retained. Warm complete decode averaged 3.790 ± 0.057 versus 3.756 ± 0.080 seconds, with four of five faster. Total inference averaged 35.385 ± 0.400 versus 35.539 ± 0.326 seconds and is inconclusive.

## Isolated fusion cost

After model timing, a fixture-only rebuild used SHA-256 `d5e9c292dd63a7d2590d088995f1200016a4b516a7051e2d035c8afc21ffe1d5`; the benchmark binary was unchanged. The [resident SIMD microbenchmark](../2026-11-04-resident-simd/microbench.nu) ran with `--fused-router`. Each batch has 64 independent complete native-router and repair calls, with two dispatches per baseline call and one per fused call. GPU command-buffer span divided by 64 is throughput per complete call, not per dispatch or model-layer latency. CPU encoding excludes input reset. Each of two series used a warmup pair and five alternating measured pairs for each of zero, three and six missing selections out of six, among 384 experts with about half resident. The first measured all-hit pair was slow in both series and is retained.

| Series | Missing | GPU time per call, separate → fused, µs (mean ± SD) | Paired GPU time changes, % in run order | CPU encode per call, separate → fused, µs (mean ± SD) |
| --- | ---: | ---: | --- | ---: |
| microbench | 0 | 24.17 ± 1.22 → 27.39 ± 16.18 | +114.2, -14.5, -14.6, -14.0, -15.9 | 1.84 ± 0.05 → 0.97 ± 0.03 |
| microbench | 3 | 19.59 ± 2.16 → 16.04 ± 3.18 | -6.1, -25.5, -21.9, -16.3, -23.1 | 1.98 ± 0.10 → 1.07 ± 0.04 |
| microbench | 6 | 22.29 ± 0.86 → 18.50 ± 0.29 | -22.5, -12.5, -16.6, -14.7, -18.0 | 2.08 ± 0.09 → 1.12 ± 0.05 |
| microbench-repeat | 0 | 30.04 ± 14.24 → 27.14 ± 16.12 | +0.9, -20.0, -14.3, -15.0, -13.6 | 1.49 ± 0.05 → 0.83 ± 0.08 |
| microbench-repeat | 3 | 21.20 ± 4.71 → 17.44 ± 2.86 | -18.8, -30.0, -2.9, -16.3, -14.9 | 1.60 ± 0.07 → 0.88 ± 0.06 |
| microbench-repeat | 6 | 21.53 ± 0.39 → 18.19 ± 0.47 | -13.8, -17.6, -14.0, -15.8, -16.2 | 1.68 ± 0.07 → 0.92 ± 0.06 |

Both series favor fusion in all five pairs for three and six misses, and CPU encoding falls. All-hit means change direction between series because the first measured pair is slow; these results do not establish a universal all-hit win. Raw samples and the earlier warmup records are retained locally.

All nine guarded jobs completed at normal pressure level 1 without pressure interruptions. Recorded swap stayed between 28.898 and 28.907 GiB.

## Decision

Keep the fusion opt-in for its exact tested behavior and reduced isolated work. The model comparisons do not establish an end-to-end benefit, and no further unlocked improvement has yet been identified. No preset or installed binary is promoted on this evidence.

