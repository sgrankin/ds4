# Parallel shared and routed FFN on cached CPU refreshes

This experiment asks whether the existing parallel shared/routed feed-forward network (FFN) can reduce decode time when the CPU-selected experts are already in the cache. The parallel path activated and shortened its local FFN work in the sampled trace, but the timing comparisons did not establish an overall gain. Keep the selector off in the selected package and retain the opt-in prototype as groundwork for a bounded follow-up.

## Context

A mixture-of-experts layer evaluates one shared expert and six routed experts. On a CPU refresh, the baseline starts the selected expert loads, submits the shared work, and then executes the routed work. This can overlap shared GPU work with a cache miss. The [preceding experiment](../2026-11-05-refresh-shared-flush/README.md) removed that separate submission, but did not establish a speed gain or enter the parallel FFN path. Its diagnostic showed that the submission boundary also prevents the existing two-stage parallel join from remaining in one batch.

The hypothesis here is that an exact all-hit selection can use that join without giving up the baseline's serial schedule on misses. The existing GPU-resident parallel path has a different proof: it pins every possible selection. CPU refreshes must instead prove that the actual six selected entries are present before using their addresses.

## Setup

The one-hour slice is November 5, 2026, 04:09–05:09 Eastern. Baseline runtime source `8bf87bbd` was restored by closure `b479686a`; candidate runtime `c1982bc8` adds the selector. Both arms use the same new binary with `DS4_METAL_V41_CACHED_REFRESH_PARALLEL` set to `0` and `1`, respectively. The common experiment 25 settings are queue residency on, shared-event overlap off, GPU resident routing with refresh interval 16, four-layer submissions, a requested 64 GB expert cache, 4 GiB slabs and 8K prefill chunks.

The workload uses DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix on an M5 Max with 128 GiB memory. Using [`promessi_sposi.txt`](../../promessi_sposi.txt) and the [decode-pass driver](../2026-11-04-decode-pass/run.nu) with the [candidate settings](../2026-11-05-slab-event-interaction/candidate-settings.json), five alternating pairs at each of 8K and 32K measure fresh and warm phases with 128 teacher-forced tokens; warm restores KV and appends 512 tokens. The expert cache survives restore and the OS cache is not flushed. Inference excludes model loading and snapshot save/restore; process wall includes them. Complete decode and first-step latency are retained alongside steady throughput, which excludes the first token and does not establish that initialization has finished. Fixed-history full-vocabulary quality and a requested 16 GB cache diagnostic examine numerical behavior and miss fallback. Raw artifacts are local under `.workspaces/hillclimb-20261104/27-cached-refresh-parallel/`.

## Admission and numerical boundary

The opt-in path applies to supported scalar text SSD routing with shared overlap off, one tensor-parallel rank, the supported six-expert IQ2/Q2 shape and Q8 shared weights. It excludes deferred HC, predicted or oracle loading, asynchronous loads, quality and imatrix modes, images, and unsupported preparation or arithmetic modes. Ordinary GPU-resident routing keeps its existing path. See [`Moe.partial`](../../../src/engine/deepseek41/moe.zig) and [`Selection.cachedParallelReady`](../../../src/metal/routed_single_selected.zig).

Preflight checks the exact six CPU-selected identities with a cache snapshot, without recording a second cache hit or advancing use counters. It validates the entire set before pinning those six entries in the current command-buffer epoch. Normal selected preparation still validates and accounts for the entries, protects them during pruning, uploads the CPU-selected IDs and uses the native routing weights. It rechecks that all six are resident with none missing before allowing address preparation; compact, split and staged execution cannot use this proof. Actual GPU dispatch pins the selected entries again. A miss or declined preflight keeps the serial shared submission and routed load path; if parallel startup declines after the proof, the normal selected override restores that path. The layer-scoped proof is separate from the broader resident-only override.

On successful admission, shared and routed producers use the existing two-stage join without the intermediate flush. Shared down projection moves from the serial fused shared-down/HC path to the parallel shared Q8 BF16-input/output down projection, followed by the shared-first BF16 sum and serial HC split-add. The BF16 input, output, sum and HC-output rounding boundaries remain, but the kernels and their schedule differ. This source analysis does not prove numerical identity; the fixed-history comparison tests it.

## Quality, activation and fallback

All 768 paired full-vocabulary teacher-logit rows and post-evaluation hashes matched exactly: 256 each at 8K and 32K with the requested 64 GiB cache, and 256 at 8K with the requested 16 GiB cache. Delta NLL, KL and top-1 differences were zero. The smaller-cache case repeats the 8K histories, so these are not 768 unique positions. Fixed histories do not establish autonomous task quality.

The diagnostics recorded 640 CPU refresh layer executions in each case. At 64 GiB, 632 armed and joined and eight fell back on a miss in each 64 GiB case. At 16 GiB, 477 armed and joined and 163 fell back on a miss. Every armed execution joined successfully; there were no startup declines or failed joins. Both arms also completed the same 9,600 ordinary resident joins per case. These counts describe layer executions, not independent tasks.

Selected preparation and whole-workload cache counters matched between arms, as did selected scalar read sequences in the bounded intervals between consecutive same-frontier post-evaluation hashes. At 16 GiB, both arms retained five split layers, 958 final cache entries and 3,993 cumulative evictions. The requested budget provided 8.88 GiB of dynamic capacity after 7.12 GiB of prefill headroom, occupying three slabs totaling 8.89 GiB. The diagnostic establishes observed activation, fallback and accounting for these cases; source review explains the pinning invariant. It does not measure miss-path speed.

The candidate benchmark built without warnings (SHA-256 `01bca35010d5b8bc7e913cb03d8b84d7d17cab8e4f36917fcd26b657e56a142c`). The guarded router-policy and SSD cache fixtures passed; the router fixture covers cached-proof policy acceptance and rejection, but not full model admission geometry.

## Profile-off timing at 32K

Fresh and warm complete decode were slower with the selector in four of five pairs each. Mean inference rose 0.37%, but both improvements occurred when the candidate ran first and prefill time rose over the sequence. This order effect and variation limit interpretation. Phase seconds derive from throughput rounded to two decimals. Entries show selector off → on; SD is sample SD.

| Pair (first arm) | Fresh first (ms) | Fresh prefill (s) | Fresh decode (s) | Fresh steady (tokens/s) | Warm append (s) | Warm first (ms) | Warm decode (s) | Inference (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (baseline) | 185.681 → 222.931 | 43.043 → 42.983 | 3.795 → 3.898 | 35.190 → 34.560 | 7.256 → 7.322 | 44.537 → 42.926 | 3.587 → 3.666 | 57.682 → 57.868 | 60.262 → 60.400 |
| 2 (candidate) | 209.036 → 194.487 | 44.144 → 43.039 | 3.765 → 3.968 | 35.720 → 33.660 | 7.073 → 7.645 | 40.112 → 42.357 | 3.679 → 3.824 | 58.661 → 58.476 | 61.192 → 61.048 |
| 3 (baseline) | 173.294 → 177.839 | 44.845 → 45.521 | 3.743 → 3.739 | 35.580 → 35.660 | 7.386 → 7.611 | 44.756 → 45.112 | 3.893 → 3.936 | 59.867 → 60.807 | 62.478 → 63.349 |
| 4 (candidate) | 170.448 → 197.230 | 46.137 → 46.033 | 3.739 → 3.761 | 35.580 → 35.640 | 7.670 → 7.646 | 42.212 → 41.638 | 3.993 → 3.949 | 61.539 → 61.391 | 64.089 → 63.938 |
| 5 (baseline) | 204.258 → 214.105 | 46.464 → 46.743 | 3.767 → 3.785 | 35.650 → 35.570 | 7.735 → 7.720 | 41.959 → 45.803 | 3.976 → 4.003 | 61.942 → 62.250 | 64.576 → 64.840 |
| Mean ± SD | 188.543 ± 17.572 → 201.318 ± 17.644 | 44.927 ± 1.414 → 44.864 ± 1.746 | 3.762 ± 0.022 → 3.830 ± 0.098 | 35.544 ± 0.206 → 35.018 ± 0.888 | 7.424 ± 0.279 → 7.589 ± 0.155 | 42.715 ± 1.942 → 43.567 ± 1.802 | 3.826 ± 0.183 → 3.876 ± 0.134 | 59.938 ± 1.824 → 60.158 ± 1.897 | 62.519 ± 1.840 → 62.715 ± 1.907 |

## Profile-off timing at 8K

Fresh complete decode was slower with the selector in four of five pairs. Warm decode was longer in three pairs despite a lower candidate mean; inference was also shorter in two pairs. Variation and outliers are retained. Entries use the same timing scope and notation as the 32K table.

| Pair (first arm) | Fresh first (ms) | Fresh prefill (s) | Fresh decode (s) | Fresh steady (tokens/s) | Warm append (s) | Warm first (ms) | Warm decode (s) | Inference (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (baseline) | 168.329 → 228.521 | 18.115 → 18.515 | 3.651 → 3.716 | 36.460 → 36.420 | 6.789 → 6.809 | 43.145 → 38.734 | 3.500 → 3.531 | 32.055 → 32.571 | 34.555 → 35.237 |
| 2 (candidate) | 178.125 → 208.932 | 18.496 → 18.810 | 3.648 → 3.690 | 36.600 → 36.490 | 7.117 → 6.685 | 43.646 → 40.994 | 3.919 → 3.526 | 33.180 → 32.711 | 35.690 → 35.345 |
| 3 (baseline) | 182.010 → 205.100 | 19.470 → 19.668 | 3.659 → 3.686 | 36.520 → 36.490 | 7.720 → 7.715 | 42.988 → 41.905 | 3.857 → 3.864 | 34.706 → 34.933 | 37.255 → 37.454 |
| 4 (candidate) | 250.614 → 190.739 | 19.293 → 19.572 | 3.733 → 3.666 | 36.470 → 36.550 | 7.685 → 7.763 | 43.968 → 44.981 | 3.906 → 3.957 | 34.618 → 34.958 | 37.086 → 37.412 |
| 5 (baseline) | 168.464 → 205.983 | 19.255 → 19.184 | 3.651 → 3.697 | 36.470 → 36.370 | 7.627 → 7.644 | 41.937 → 44.317 | 3.888 → 3.876 | 34.421 → 34.402 | 36.850 → 36.852 |
| Mean ± SD | 189.508 ± 34.681 → 207.855 ± 13.530 | 18.926 ± 0.588 → 19.150 ± 0.492 | 3.668 ± 0.036 → 3.691 ± 0.018 | 36.504 ± 0.059 → 36.464 ± 0.070 | 7.388 ± 0.415 → 7.323 ± 0.530 | 43.137 ± 0.777 → 42.186 ± 2.538 | 3.814 ± 0.177 → 3.751 ± 0.206 | 33.796 ± 1.152 → 33.915 ± 1.185 | 36.287 ± 1.145 → 36.460 ± 1.094 |

## Profile-off timing at 8K with requested 16 GiB cache

Fresh complete decode was shorter in three pairs; warm decode and inference were shorter in two each. Both inference improvements occurred when the candidate ran first, so the lower mean does not establish a benefit. Entries use the same scope and notation as the 32K table.

| Pair (first arm) | Fresh first (ms) | Fresh prefill (s) | Fresh decode (s) | Fresh steady (tokens/s) | Warm append (s) | Warm first (ms) | Warm decode (s) | Inference (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (baseline) | 175.102 → 203.569 | 17.710 → 17.139 | 3.742 → 4.128 | 35.610 → 32.370 | 8.073 → 8.275 | 46.021 → 48.222 | 3.562 → 3.793 | 33.087 → 33.335 | 34.234 → 34.510 |
| 2 (candidate) | 194.878 → 206.064 | 18.212 → 17.942 | 4.216 → 3.777 | 31.580 → 35.570 | 8.808 → 8.133 | 50.807 → 48.562 | 3.888 → 3.576 | 35.124 → 33.429 | 36.333 → 34.580 |
| 3 (baseline) | 241.057 → 221.770 | 19.148 → 19.273 | 3.952 → 3.806 | 34.220 → 35.440 | 8.183 → 8.296 | 47.009 → 50.471 | 3.658 → 3.878 | 34.941 → 35.253 | 36.188 → 36.533 |
| 4 (candidate) | 206.539 → 195.615 | 19.654 → 19.449 | 3.776 → 3.767 | 35.580 → 35.570 | 8.634 → 8.427 | 53.971 → 53.108 | 3.948 → 3.894 | 36.012 → 35.536 | 37.306 → 36.773 |
| 5 (baseline) | 213.125 → 221.672 | 19.837 → 20.093 | 3.780 → 3.790 | 35.600 → 35.590 | 8.553 → 8.825 | 49.175 → 53.385 | 3.829 → 3.970 | 35.999 → 36.678 | 37.316 → 37.936 |
| Mean ± SD | 206.140 ± 24.281 → 209.738 ± 11.600 | 18.912 ± 0.921 → 18.779 ± 1.205 | 3.893 ± 0.198 → 3.854 ± 0.154 | 34.518 ± 1.747 → 34.908 ± 1.420 | 8.450 ± 0.311 → 8.391 ± 0.264 | 49.397 ± 3.165 → 50.750 ± 2.437 | 3.777 ± 0.162 → 3.822 ± 0.151 | 35.032 ± 1.193 → 34.846 ± 1.440 | 36.275 ± 1.257 → 36.066 ± 1.487 |

## Schedule diagnostic

An intrusive 32K timeline confirmed that successful cached refresh layers used the concurrent four-dispatch FFN encoder group followed by serial HC split-add. Fresh first and step 16 had 38 and 39 successful layers, reducing command buffers from 91 to 53 and 52; both sampled warm refreshes had 40 successes and 51 buffers. Three miss layers retained the serial fused tail. All 30 ordinary windows per arm retained 11 buffers and 40 resident parallel groups. The grouped trace does not resolve physical simultaneous execution of individual kernels.

On the sampled later refreshes, the sum of the 40 layer FFN sub-envelopes shortened from 9.927 to 9.690 ms fresh and 9.531 to 9.376 ms warm. Positive inter-buffer gaps immediately before the post-router FFN buffers grew from 10.508 to 16.087 ms fresh and 8.626 to 16.366 ms warm; all four sampled refresh windows were longer overall. These overlapping intervals are not an exact latency decomposition and do not identify a host, submission or driver cause. The baseline can submit shared work before CPU cache and address preparation finishes, while the candidate waits to submit the combined FFN after its proof and preparation. The trace locates a tradeoff consistent with that schedule, not a complete causal explanation.

## Per-token diagnostic

One separate 32K native-grouping pair measured the nonspeculative evaluation interval per token. Candidate minus baseline was +34.274 ms for the first fresh step, +8.685 ms across seven later refreshes, and −1.919 ms across 120 ordinary steps. The corresponding warm differences were +4.424 ms, +12.205 ms and +97.335 ms. This single diagnostic pair does not establish a comparative latency effect or explain the differences.

## Decision and reproduction

Do not select the flag in the experiment 25 package or presets. Retain runtime `c1982bc8` as an opt-in prototype: the trace establishes the intended concurrent schedule and a modest local FFN shortening, while larger preceding gaps and mixed end-to-end timings leave an overall benefit unproved. A bounded follow-up on CPU preparation and submission can be considered after the scheduled periodic qualification; this result does not establish the cause of the gaps.

All nine guarded jobs completed successfully, with no bad samples or pressure stops. Pressure remained at level 1 and swap ranged from 29.00201 to 29.00982 GiB. Reconstruct and build runtime `c1982bc8` with matching Metal sources and verify the benchmark hash above. For example, the profile-off 32K comparison uses the existing driver and a fresh output path:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/repro27-guard -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/repro27-timing timing --flag DS4_METAL_V41_CACHED_REFRESH_PARALLEL --control 0 --candidate 1 --extra-settings speed-bench/experiments/2026-11-05-slab-event-interaction/candidate-settings.json --frontier 32768 --tokens 128 --samples 5
```

Use `--frontier 8192` for 8K and add `--cache 16GB` for the smaller-cache comparison. Local raw artifacts retain commands, individual samples, profiles and diagnostic settings.
