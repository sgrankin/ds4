# Queue-attached slab residency and shared-expert scheduling

This experiment tests whether disabling shared-expert overlap changes the cost of queue-attached expert-slab residency. The combined candidate improves measured complete fresh decode and inference time at both tested frontiers, with lower fresh steady throughput. Retain it as an experimental latency candidate, not a general steady-throughput improvement.

## Context

The [queue-attached residency experiment](../2026-11-05-streaming-slab-residency/README.md) shortened the fresh first decode step, but complete fresh and warm decode slowed in every pair at both tested frontiers. A separate timeline showed new waits near shared-event boundaries. A [standing residency request](../2026-11-05-standing-slab-residency/README.md) avoided those repeated waits but restored the earlier first-step gaps. Neither observation establishes a driver cause.

The cache packs expert slots into slab buffers. Queue-attached residency registers whole slab allocations in a Metal residency set attached to the runtime queue. Existing resource declarations, cache pins and pressure handling remain in place. The first diagnostic holds residency enabled and changes shared overlap from `1` to `0`. On CPU-refresh layers this also moves shared gate/up work after CPU route and load startup, and moves shared down work to the matmul and HC-expand fusion. Ordinary GPU-resident steps remain unchanged. This is therefore a comparison of schedules, not a pure event toggle.

## Setup

The selected one-hour slice is November 5, 02:40–03:40 Eastern, on an M5 Max with 128 GiB memory. Both arms use the queue-attached runtime `5efd342f` and the same `ds4-bench` binary, SHA-256 `8b252cdb52306506644f58be8ac5761dddd41b035c27725c95fe9c7d49f7e52f`. The matching frozen workspace is `.workspaces/hillclimb-20261104/queue-residency-5efd`, an empty child `f8a219a1` of `5efd342f`. The historical runtime `d49d027a` used a standing request and is not the tested binary. Root runtime `8bf87bbd` restores the queue-attached implementation for the later native diagnostic.

The diagnostic compares queue residency on with overlap on against queue residency on with overlap off. Qualification compares the original package, queue residency off and overlap on, against the combined candidate, queue residency on and overlap off. Both qualification arms use the same `5efd342f` binary and hold the remaining experimental GPU package settings fixed, including GPU resident routing, refresh interval 16 and four-layer submissions. The [baseline](baseline-settings.json) and [candidate](candidate-settings.json) settings record the tested packages; raw artifacts and the runtime record are under `.workspaces/hillclimb-20261104/25-slab-event-interaction/`.

The workload uses `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, a 64 GB expert-cache target, 4 GiB slabs, 8K prefill chunks and a 40,960-token context allocation. Fresh and warm phases each evaluate 128 teacher-forced tokens; the warm phase restores KV state and appends 512 tokens. Qualification calls for fixed-history quality and five alternating pairs at each of 8K and 32K. It records fresh and warm first steps, complete decode windows, prefill and append, total inference and process wall. The expert cache survives KV restore, and the OS file cache was not flushed. Model startup and snapshot save/restore are excluded from phase timing but included in process wall. Diagnostic profiles are separate from profile-off timing.

## Findings and decision

A separate intrusive 32K timeline pair held queue residency on and changed overlap from `1` to `0`. The GPU embedding-to-head envelope for the fresh first step changed from 354.41 to 153.69 ms, and the next refresh from 192.50 to 37.83 ms. For the warm phase, the first step changed from 195.98 to 41.30 ms and the next refresh from 179.55 to 43.68 ms. Mean ordinary-step envelopes were 30.37 to 27.63 ms fresh and 29.89 to 27.41 ms warm. Each refresh figure is one observation; this profile changes encoder grouping and is not comparative latency evidence.

For the fresh next refresh, summed gaps inside command buffers changed from 160.05 to 0.03 ms while gaps between buffers changed from 2.81 to 10.39 ms. These intervals can overlap and do not prove a driver mechanism. The raw profile parser's host scope for overlap `0` refreshes covers only the final command scope, because this schedule ends and begins commands on each CPU-refresh layer. It must not be interpreted as whole-token time. The local `source-audit.md` and `overlap-diagnostic-summary.json` explain the scope and retain the observations.

The combined candidate and original baseline produced byte-identical full-vocabulary teacher-logit rows on all four fixed histories: 128 rows each at 8,192, 8,704, 32,768 and 33,280 tokens, or 512 of 512 rows. All 512 post-evaluation hashes also matched. Delta NLL and KL were zero throughout, with no top-1 mismatches. These checks use identical histories and do not measure autonomous task quality. Local `quality8k/quality-summary.json` and `quality32k/quality-summary.json` retain the per-frontier results.

Retain queue residency on with overlap off as an experimental latency candidate. It reduces complete fresh decode and inference time in all five pairs at both frontiers, while reducing fresh steady throughput in every pair. In the matched native-grouping diagnostic pair, ordinary steps also took longer in aggregate; one logged pair cannot establish a recurring effect or its size. CPU fast presets, installed binaries and main remain unchanged.

## Native-grouping diagnostic

A separate 32K pair used normal command grouping and logged the full nonspeculative `ds4_session_eval` interval per token, excluding row printing and token selection. For fresh decode, the seven later refresh steps summed to 207.341 → 257.008 ms and the 120 ordinary steps to 3,294.078 → 3,314.722 ms; all 127 steps after the first summed to 3,501.419 → 3,571.730 ms. In warm decode, later refresh steps summed to 204.357 → 269.681 ms and ordinary steps to 3,285.922 → 3,491.248 ms. The differences are not confined to refresh steps. This is one pair with per-token logging, not comparative performance evidence. The local `native-token32k/native-token-summary.json` retains the full measurements.

This diagnostic used root runtime `8bf87bbd`: its source matches `5efd342f` except for the five-line opt-in benchmark diagnostic. The build completed without warnings; benchmark SHA-256 was `50cbd0080e9ff72310000dc3dc4486c11f3611b72654ca49e6d473458138073f`. The built fixture SHA-256 `b52bf1d8f2363497d3796a0ee89a0da7dbb90276c9cc7b51bd873073d8d2190f` matches the fixture qualified in the queue-attached experiment; it was not rerun here.

All six guarded jobs completed without a pressure stop or bad sample; observed pressure was level 1 throughout. Swap ranged from 29.00293 to 29.01074 GiB across the jobs.

## Profile-off timing at 32K

Five alternating pairs favor the combined candidate for complete fresh decode, inference total and process wall in every pair. Fresh first-step savings outweigh slower fresh steady decode in this window; steady throughput is lower in every pair. Warm decode is slower in four pairs, warm first is slower in all five, and warm append is slower in four. Fresh prefill is shorter in all five pairs, but both arms drift upward over the run. Inference total covers prefill and decode, excluding model loading and snapshot save/restore; wall includes the process. Phase seconds are derived from throughput rounded to two decimal places. Steady throughput excludes the first token and does not establish that all initialization has finished.

| Pair (first arm) | Fresh first (ms) | Fresh prefill (s) | Fresh decode (s) | Fresh steady (tokens/s) | Warm append (s) | Warm first (ms) | Warm decode (s) | Inference (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (baseline) | 779.764 → 191.076 | 44.376 → 43.129 | 4.418 → 3.935 | 34.900 → 33.930 | 7.406 → 7.746 | 38.223 → 47.489 | 3.606 → 3.791 | 59.806 → 58.601 | 61.898 → 61.140 |
| 2 (candidate) | 834.858 → 199.482 | 45.498 → 43.949 | 4.345 → 3.774 | 36.180 → 35.530 | 7.288 → 7.067 | 35.156 → 40.797 | 3.831 → 3.693 | 60.963 → 58.483 | 63.068 → 61.017 |
| 3 (baseline) | 783.031 → 203.415 | 46.507 → 46.233 | 4.298 → 3.757 | 36.140 → 35.740 | 7.628 → 7.769 | 38.542 → 44.250 | 3.855 → 3.949 | 62.289 → 61.709 | 64.501 → 64.300 |
| 4 (candidate) | 859.276 → 193.193 | 47.141 → 46.346 | 4.363 → 3.758 | 36.250 → 35.620 | 7.645 → 7.918 | 35.859 → 44.907 | 3.919 → 3.969 | 63.068 → 61.991 | 65.289 → 64.558 |
| 5 (baseline) | 818.101 → 212.640 | 47.291 → 46.854 | 4.321 → 3.776 | 36.260 → 35.650 | 7.749 → 7.772 | 34.561 → 45.947 | 3.934 → 4.035 | 63.295 → 62.437 | 65.431 → 65.054 |
| Mean ± sample SD | 815.006 ± 34.014 → 199.961 ± 8.626 | 46.163 ± 1.223 → 45.302 ± 1.652 | 4.349 ± 0.046 → 3.800 ± 0.076 | 35.946 ± 0.587 → 35.294 ± 0.766 | 7.543 ± 0.190 → 7.654 ± 0.335 | 36.468 ± 1.810 → 44.678 ± 2.490 | 3.829 ± 0.132 → 3.888 ± 0.141 | 61.884 ± 1.477 → 60.644 ± 1.937 | 64.037 ± 1.520 → 63.214 ± 1.968 |

## Profile-off timing at 8K

Fresh first step, fresh prefill, complete fresh decode and inference total are shorter in all five pairs; wall is shorter in four. Fresh steady throughput is lower and warm first latency is higher in every pair. Warm append and complete warm decode are shorter in two pairs each. These use the same scopes and rounding limitations as the 32K results.

| Pair (first arm) | Fresh first (ms) | Fresh prefill (s) | Fresh decode (s) | Fresh steady (tokens/s) | Warm append (s) | Warm first (ms) | Warm decode (s) | Inference (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (baseline) | 808.095 → 182.730 | 19.549 → 19.288 | 4.231 → 3.651 | 37.090 → 36.620 | 7.188 → 7.137 | 34.936 → 39.958 | 3.812 → 3.748 | 34.781 → 33.824 | 36.850 → 36.278 |
| 2 (candidate) | 788.326 → 187.886 | 19.295 → 19.080 | 4.217 → 3.672 | 37.040 → 36.450 | 7.474 → 7.219 | 34.378 → 41.682 | 3.830 → 3.776 | 34.817 → 33.747 | 36.866 → 36.257 |
| 3 (baseline) | 819.368 → 215.211 | 19.230 → 19.222 | 4.250 → 3.689 | 37.030 → 36.560 | 7.516 → 7.604 | 34.297 → 45.684 | 3.798 → 3.886 | 34.794 → 34.401 | 36.870 → 36.889 |
| 4 (candidate) | 820.225 → 209.792 | 19.166 → 19.152 | 4.252 → 3.688 | 37.010 → 36.510 | 7.564 → 7.667 | 34.533 → 44.767 | 3.824 → 3.875 | 34.806 → 34.382 | 36.858 → 36.833 |
| 5 (baseline) | 780.968 → 177.166 | 19.522 → 19.452 | 4.202 → 3.655 | 37.120 → 36.510 | 7.563 → 7.652 | 34.453 → 44.277 | 3.831 → 3.911 | 35.119 → 34.670 | 37.170 → 37.106 |
| Mean ± sample SD | 803.396 ± 17.962 → 194.557 ± 16.923 | 19.352 ± 0.174 → 19.239 ± 0.142 | 4.231 ± 0.021 → 3.671 ± 0.018 | 37.058 ± 0.045 → 36.530 ± 0.064 | 7.461 ± 0.157 → 7.456 ± 0.256 | 34.519 ± 0.249 → 43.274 ± 2.376 | 3.819 ± 0.014 → 3.839 ± 0.072 | 34.863 ± 0.143 → 34.205 ± 0.400 | 36.923 ± 0.139 → 36.672 ± 0.384 |

## Reproduction

The maintained driver clears inherited `DS4` settings and runs each binary from its specified source root, because Metal sources load relative to the working directory. The committed settings files record the two tested packages; use fresh output paths and the guarded driver. This command runs the 32K timing comparison with the archived binary and matching sources on both sides:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/hillclimb-20261104/25-slab-event-interaction/repro-guard-timing32k -- nu speed-bench/experiments/2026-11-04-decode-pass/combined.nu .workspaces/hillclimb-20261104/25-slab-event-interaction/repro-timing32k timing --baseline-root .workspaces/hillclimb-20261104/queue-residency-5efd --candidate-root .workspaces/hillclimb-20261104/queue-residency-5efd --baseline-revision 5efd342f --candidate-revision 5efd342f --baseline-settings speed-bench/experiments/2026-11-05-slab-event-interaction/baseline-settings.json --final-settings speed-bench/experiments/2026-11-05-slab-event-interaction/candidate-settings.json --frontier 32768 --tokens 128 --samples 5
```

For 8K, change the frontier to `8192` and both output names to `8k`. For fixed-history logits, replace `timing` with `quality` and use fresh output names; the driver takes one pair per frontier. The frozen workspace is a local artifact; if unavailable, reconstruct and build revision `5efd342f` in a matching source root and verify the binary hash before reproducing the comparison.
