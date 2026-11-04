# Raw clock for selected-event polling

The raw-clock change is unadopted. The 8K means favored the candidate, but all five 32K warm pairs favored the original clock. The large-context goal retains the original loop without a context-dependent switch. These results do not establish a universal regression.

## Question and mechanism

The earlier [clock-stride experiment](clock-stride.md) found many sampled main-thread stacks in the selected-event polling clock subtree, but reducing the frequency of clock checks did not establish a decode benefit. This experiment instead asks whether a different clock read can reduce polling overhead while retaining an event and deadline check on every poll. The trace showed where the thread waited, not the cost removable from a request.

Baseline `b4f823fa` includes the five previously adopted decode changes. Candidate `ca6e10c6` adds `DS4_METAL_SELECTED_READBACK_RAW_CLOCK=1` in `awaitSelected` (inspect with `git show ca6e10c6:src/metal/tp_control.zig`). The [active implementation](../../../src/metal/tp_control.zig) retains the original clock. When enabled with a nonzero spin budget, the candidate reads `clock_gettime_nsec_np(CLOCK_MONOTONIC_RAW)` for the spin interval; a zero timestamp skips or ends spinning and falls back to the bounded wait. The original monotonic start still determines the remaining 60-second timeout, and timestamps from the two clocks are never subtracted. The default path is unchanged.

The local SDK's `clock_gettime(3)` documents raw monotonic time as including sleep and a zero return on error for the nanosecond API. [Apple's Libc implementation](https://github.com/apple-oss-distributions/Libc/blob/main/gen/clock_gettime.c) shows a structural distinction between the ordinary monotonic clock's boot-time/gettimeofday path and raw time's continuous Mach clock path. This is a hypothesis about overhead, not a measurement of clock costs on this OS.

## Setup

The loop began at 04:50 Eastern on 2026-11-04, and finished at 05:13, before the planned 05:20 close and 05:50 one-hour ceiling. The M5 Max has 128 GiB memory; runs use `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, SSD streaming and a requested 64 GB expert cache. The [driver](run.nu) clears inherited DS4 settings, sets a 2,000 microsecond spin budget and uses the Promessi Sposi prompt; it records the common decode settings. Raw output is retained locally under `.workspaces/overnight-decode-20261104/raw-clock/`.

The profile-free comparison used five alternating pairs at each of 32K and 8K, with 128 teacher-forced generation steps. Each run performs fresh prefill and decode, then KV restore, a 512-token warm append and another decode. Phase timings exclude model loading and snapshot save/restore; the expert cache survives restore and the OS cache is not flushed. The benchmark's steady decode excludes the first step, which does not prove initialization has finished. Full generation and first-step timings are retained separately. An 8K fixed-history quality check and a CPU diagnostic supplement the latency runs.

## Results and decision

At 8K, the fixed-history comparison produced 256 byte-identical full-vocabulary logit rows and 256 matching post-evaluation hashes: 128 fresh rows at 8,192 tokens and 128 after the append at 8,704. Delta-NLL and KL were zero on those histories. This does not establish parity for other histories or practical tasks. The benchmark build passed. Source and SDK review found no timeout or clock-lifetime blocker; the spin setting is capped at 5,000 microseconds, so the conversion to nanoseconds is bounded.

At 32K, fresh steady decode was 34.568 ± 0.059 versus 34.656 ± 0.092 tokens/s and warm steady decode was 34.178 ± 0.337 versus 33.764 ± 0.206 tokens/s (mean ± sample SD, baseline versus candidate). All five warm pairs favored the baseline; its lowest sample, 34.00 tokens/s, exceeded the candidate’s highest, 33.97 tokens/s. Fresh outcomes were mixed. Inference phase totals averaged 64.091 ± 1.981 versus 64.843 ± 0.402 s, including prefill. The first baseline prefill took 45.289 s, while the other prefill samples were approximately 49 s; it is retained in the aggregate. These timings do not establish why the warm difference occurred.

| Pair (first arm) | Fresh steady, tokens/s | Warm steady, tokens/s | Fresh full, s | Warm full, s | Fresh first, ms | Warm first, ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (baseline) | 34.62 → 34.56 | 34.78 → 33.47 | 4.567 → 4.596 | 3.685 → 3.828 | 898.1 → 920.8 | 32.8 → 33.7 |
| 2 (candidate) | 34.64 → 34.66 | 34.06 → 33.83 | 4.544 → 4.589 | 3.785 → 3.787 | 877.4 → 925.5 | 55.7 → 33.6 |
| 3 (baseline) | 34.51 → 34.58 | 34.03 → 33.64 | 4.580 → 4.611 | 3.768 → 3.810 | 899.7 → 937.4 | 35.5 → 34.0 |
| 4 (candidate) | 34.52 → 34.79 | 34.02 → 33.97 | 4.580 → 4.536 | 3.769 → 3.772 | 900.8 → 884.7 | 35.2 → 33.2 |
| 5 (baseline) | 34.55 → 34.69 | 34.00 → 33.91 | 4.568 → 4.613 | 3.770 → 3.781 | 892.3 → 951.1 | 34.4 → 36.1 |

Values in each cell are baseline → candidate. Full durations are approximate, derived from throughput rounded to two decimal places.

At 8K, fresh steady decode was 34.170 ± 0.091 versus 34.512 ± 0.529 tokens/s and warm steady decode was 34.442 ± 0.246 versus 34.708 ± 0.497 tokens/s. The candidate won three of five pairs in each phase; pair 5 accounts for much of the mean advantage. Inference phase totals averaged 35.307 ± 0.209 versus 34.921 ± 0.528 s.

| Pair (first arm) | Fresh steady, tokens/s | Warm steady, tokens/s | Fresh full, s | Warm full, s | Fresh first, ms | Warm first, ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (baseline) | 34.26 → 34.61 | 34.79 → 34.90 | 4.580 → 4.523 | 3.682 → 3.671 | 872.4 → 854.5 | 32.6 → 32.0 |
| 2 (candidate) | 34.25 → 34.50 | 34.56 → 34.72 | 4.618 → 4.545 | 3.708 → 3.692 | 910.7 → 863.7 | 33.0 → 33.6 |
| 3 (baseline) | 34.19 → 34.11 | 34.41 → 34.28 | 4.591 → 4.616 | 3.724 → 3.737 | 876.4 → 892.8 | 32.6 → 32.5 |
| 4 (candidate) | 34.09 → 34.00 | 34.30 → 34.21 | 4.594 → 4.618 | 3.735 → 3.745 | 868.1 → 882.3 | 33.2 → 32.7 |
| 5 (baseline) | 34.06 → 35.34 | 34.15 → 35.43 | 4.557 → 4.409 | 3.753 → 3.617 | 828.7 → 815.9 | 33.5 → 32.0 |

Cells show baseline → candidate; full durations use rounded throughput, as above.

The candidate remains available at revision `ca6e10c6`; the experimental flag is removed from the active implementation. A separate candidate diagnostic sampled the main thread for 3 seconds at 1 ms intervals during warm 32K decode, after the 512-token append and a two-second delay. Of 1,389 observations, 1,306 were in `awaitSelected`, including 1,182 at the raw clock call site and 1,069 under `clock_gettime_nsec_np` → `mach_continuous_time`. The sampled warm phase generated 512 teacher-forced steps; the local sample is `.workspaces/overnight-decode-20261104/raw-clock/cpu-warm/1-candidate/cpu.sample.txt`. It confirms the new call path while the thread still spends most sampled observations waiting. These counts are neither milliseconds nor CPU utilization. The earlier baseline trace used a different phase and configuration, so it is a structural reference, not a paired CPU cost comparison. The cause of the warm timing difference remains unresolved.

All completed guarded runs recorded pressure level 1. Swap was flat at 24.9631 GiB in quality, rose from 24.9631 to 24.9683 GiB in 32K timing and from 24.9683 to 24.9723 GiB in 8K timing, and stayed flat at 24.9723 GiB during the CPU diagnostic. Practical task quality was not measured.

## Reproduction

Check out candidate `ca6e10c6` in a separate workspace and build the benchmark. From that workspace, use the [driver](run.nu) under the pressure guard, with distinct output directories for each run:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/raw-clock-32k-guard -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/raw-clock-32k timing --flag DS4_METAL_SELECTED_READBACK_RAW_CLOCK --control 0 --candidate 1 --frontier 32768 --tokens 128
```

Use `--frontier 8192` and separate output and guard directories for the 8K timing. For fixed-history logits, use stage `quality` with `--frontier 8192 --tokens 128` and distinct directories.
