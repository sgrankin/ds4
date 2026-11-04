# Clock checks during selected-event polling

The clock-stride change is unadopted. Five-pair comparisons did not establish a decode benefit, so the original stride 1 loop remains and the experimental runtime flag has been removed. At 32K, stride 64 was slower in every warm pair; this does not establish a universal regression.

## Diagnostic and change

A 3-second, 1 ms `/usr/bin/sample` trace of the main thread was triggered after 32K prefill and two seconds of decode. Of 1,154 main-thread samples, 1,082 were in selected-event polling, including 1,023 in the clock subtree; 55 were in the final command wait, eight in the Engram join, two in expert wait and seven elsewhere. These are sampled stack observations, not exact CPU time or an estimate of the achievable speedup. They did not identify a non-wait hotspot. The hypothesis is that fewer repeated clock reads allow faster observation of readiness, not that clock samples represent removable end-to-end work.

The candidate in `src/metal/tp_control.zig` checks the event on every poll but amortizes deadline checks over a configurable burst. `DS4_METAL_SELECTED_READBACK_CLOCK_STRIDE` defaults to 1 and is bounded from 1 to 64; stride 1 retains the original loop. The original 2 ms spin budget and 60-second fallback start remain, though the spin cutoff can be coarse by up to one burst. This changes neither model math nor cache retirement.

## Setup

The selected loop began at 03:32 Eastern on 2026-11-04 and closed at 04:18, within its 04:32 ceiling. Baseline revision `a65c4fa6` precedes candidate `61e0d05c`. The diagnostic used the existing [driver](run.nu), its `cpu-profile` stage and 1,024 tokens. Its local sample is `.workspaces/overnight-decode-20261104/cpu-host-profile/profile/1-baseline/cpu.sample.txt`; raw evidence for this experiment is under `.workspaces/overnight-decode-20261104/cpu-host-profile/`.

The profile-free comparisons used five alternating pairs of stride 1 versus 16 and stride 1 versus 64 at 32K, followed by stride 1 versus 16 at 8K. The [tradeoff curve](clock-stride-curve.svg) keeps the controls for the 32K batches separate.

## Results

At 32K, stride 1 versus 16 produced fresh steady decode of 33.754 ± 0.915 versus 33.762 ± 0.997 tokens/s, and warm steady decode of 33.826 ± 0.699 versus 33.318 ± 0.635 tokens/s (mean ± sample SD). The candidate won three fresh pairs and two warm pairs. Baseline ran first in pairs 1, 3 and 5, and candidate first in pairs 2 and 4. Inference phase totals averaged 64.499 ± 1.654 versus 64.755 ± 0.881 s, including prefill. The paired results are mixed.

| Pair | Fresh steady 1 → 16, tokens/s | Warm steady 1 → 16, tokens/s | Full fresh 128 steps, 1 → 16, s | Full warm 128 steps, 1 → 16, s |
| --- | --- | --- | --- | --- |
| 1 | 34.34 → 34.55 | 34.60 → 32.62 | 4.668 → 4.544 | 3.704 → 3.928 |
| 2 | 34.44 → 34.49 | 34.03 → 33.34 | 4.558 → 4.601 | 3.766 → 3.844 |
| 3 | 32.61 → 34.42 | 33.03 → 34.34 | 4.753 → 4.555 | 3.879 → 3.731 |
| 4 | 34.47 → 32.56 | 34.31 → 33.04 | 4.596 → 4.789 | 3.735 → 3.878 |
| 5 | 32.91 → 32.79 | 33.16 → 33.25 | 4.699 → 4.798 | 3.864 → 3.853 |

The benchmark labels decode after the first step as steady; excluding one step does not establish that initialization is finished. Full decode durations are approximate, derived from throughput rounded to two decimals. Each run includes fresh prefill and teacher-forced decode, then KV restore, a 512-token warm append and another decode. Phase timings exclude model loading and snapshot save/restore; the expert cache survives restore and the OS cache was not flushed.

At 8K, stride 64 produced 256 byte-identical full-vocabulary logit rows and 256 matching hashes against stride 1, with zero delta-NLL and KL on those fixed histories. This does not establish parity for other histories or practical tasks. Source review found no timeout or lifetime blocker.

At 32K, a separate stride 1 versus 64 batch measured fresh steady decode at 34.144 ± 0.483 versus 34.134 ± 0.681 tokens/s and warm steady decode at 34.078 ± 0.312 versus 33.582 ± 0.140 tokens/s. All five warm pairs favored stride 1; its lowest warm sample, 33.82 tokens/s, exceeded the candidate's highest, 33.74 tokens/s. These are local controls for this batch and are not pooled with the stride 16 controls.

| Pair | Fresh steady 1 → 64, tokens/s | Warm steady 1 → 64, tokens/s | Full fresh 128 steps, 1 → 64, s | Full warm 128 steps, 1 → 64, s |
| --- | --- | --- | --- | --- |
| 1 | 34.45 → 32.94 | 34.57 → 33.53 | 4.578 → 4.789 | 3.707 → 3.824 |
| 2 | 34.07 → 34.56 | 33.97 → 33.37 | 4.658 → 4.593 | 3.772 → 3.839 |
| 3 | 33.33 → 34.22 | 34.19 → 33.74 | 4.750 → 4.634 | 3.748 → 3.798 |
| 4 | 34.46 → 34.42 | 33.84 → 33.63 | 4.554 → 4.606 | 3.786 → 3.821 |
| 5 | 34.41 → 34.53 | 33.82 → 33.64 | 4.585 → 4.568 | 3.788 → 3.811 |

The same alternating order and timing scope apply as in the stride 16 batch.

At 8K, stride 1 versus 16 gave fresh steady decode of 35.238 ± 0.050 versus 34.896 ± 0.629 tokens/s and warm steady decode of 33.842 ± 0.993 versus 34.210 ± 1.111 tokens/s. Inference phase totals averaged 35.161 ± 0.873 versus 35.225 ± 1.105 s. Pair outcomes were mixed and do not establish a benefit.

| Pair | Fresh steady 1 → 16, tokens/s | Warm steady 1 → 16, tokens/s | Full fresh 128 steps, 1 → 16, s | Full warm 128 steps, 1 → 16, s |
| --- | --- | --- | --- | --- |
| 1 | 35.28 → 35.10 | 35.37 → 35.43 | 4.441 → 4.483 | 3.623 → 3.618 |
| 2 | 35.20 → 35.16 | 33.19 → 35.30 | 4.494 → 4.517 | 3.860 → 3.631 |
| 3 | 35.28 → 35.31 | 32.78 → 33.09 | 4.469 → 4.479 | 3.911 → 3.871 |
| 4 | 35.17 → 35.13 | 33.80 → 33.23 | 4.466 → 4.471 | 3.790 → 3.855 |
| 5 | 35.26 → 33.78 | 34.07 → 34.00 | 4.460 → 4.588 | 3.760 → 3.768 |


First-step latency remains visible below (mean ± sample SD, with range, in ms).

| Comparison | Fresh baseline → candidate | Warm baseline → candidate |
| --- | --- | --- |
| 32K, stride 1 → 16 | 890.0 ± 51.2 (840.6–969.3) → 893.1 ± 27.5 (866.5–924.5) | 33.4 ± 0.6 (32.8–34.2) → 34.0 ± 0.7 (33.1–35.0) |
| 32K, stride 1 → 64 | 904.1 ± 29.8 (867.1–938.8) → 915.7 ± 15.8 (890.2–933.2) | 33.5 ± 0.2 (33.2–33.7) → 36.7 ± 4.9 (33.3–45.2) |
| 8K, stride 1 → 16 | 862.3 ± 16.8 (841.9–886.9) → 866.5 ± 28.4 (828.4–904.2) | 33.8 ± 1.6 (32.0–36.3) → 33.1 ± 0.3 (32.7–33.5) |

No second CPU profile was run: the initial trace showed waiting rather than identifiable removable compute, and the timing comparisons did not establish a benefit. Practical task quality was not measured. Candidate `61e0d05c` remains available in history.

## Reproduction and resources

The runs used an M5 Max with 128 GiB memory and `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, SSD streaming and a requested 64 GB expert cache. The [driver](run.nu) clears inherited DS4 settings and records the common decode settings. All recorded pressure samples were level 1. Swap was flat in the profile, quality and 8K timing runs; it rose from 24.9556 to 24.9591 GiB during stride 16 at 32K and from 24.9591 to 24.9631 GiB during stride 64 at 32K.

To reproduce the 32K stride 16 timing, check out candidate `61e0d05c` in a separate workspace, build the benchmark, and use the current driver from that workspace:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/overnight-decode-20261104/cpu-host-profile/timing16-32k-guard -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/overnight-decode-20261104/cpu-host-profile/timing16-32k timing --flag DS4_METAL_SELECTED_READBACK_CLOCK_STRIDE --control 1 --candidate 16 --frontier 32768 --tokens 128
```

Use `--candidate 64` for the other 32K batch, or `--frontier 8192` for the 8K stride 16 batch, with distinct output and guard directories.
