# Streaming expert slab residency

This experiment asks whether an opt-in Metal residency set for streaming expert slabs removes the roughly 0.8-second first scalar decode delay without moving the cost into prefill or later decode. At 8K and 32K, the fresh first step became much shorter, but complete fresh and warm decode were slower in every pair. Do not select this queue-attached form; retain the opt-in groundwork for a bounded follow-up. Resource admission remains a hypothesis. The retained groundwork is off by default; a request-only policy remains a possible follow-up, not an implemented result. Ordinary presets, installed binaries and main remain unchanged.

## Context and mechanism

The preceding [first-decode experiment](../2026-11-05-first-decode-admission/README.md) changed expert dispatch on the first scalar step but left large gaps before expert GPU work. Its 32K diagnostic attributed 666.487 ms to positive gaps between command-buffer spans in the address-dispatch arm. That intrusive trace does not establish a driver cause or associate a gap with a particular slab. Prefill already uses the expert cache through GPU blits, so this is not the first GPU use of those allocations.

The cache packs expert slots into slab buffers. When enabled, the candidate registers each actual slab allocation with an engine-owned Metal residency set before publishing its slots, commits the set and attaches it to the runtime queue. Existing encoder resource declarations and cache pins remain in place. A set covers whole slab buffers, including unused slots; it does not declare only the experts used by a layer. It does not request a warm-up or map the whole model. The implementation and fallback are in [`expert_storage.zig`](../../../src/metal/expert_storage.zig) and [`expert_cache.inc.m`](../../../src/metal/expert_cache.inc.m).

The set is detached and cleared before cache pressure relief, slot unlock or trimming, and during cache clearing and engine teardown. [`expert_cache.zig`](../../../src/metal/expert_cache.zig) owns these lifecycle transitions. Failure to create it falls back to ordinary declarations and disables further attempts until the cache is fully cleared. The Metal queue is shared with other runtime work, so behavior with co-resident helpers requires separate qualification.

## Setup

The selected one-hour slice is November 5, 01:27–02:27 Eastern on an M5 Max with 128 GiB memory. Both arms use candidate runtime `5efd342f`, whose parent is `7435eca0` (runtime `909faab5`), and the same `ds4-bench` binary (SHA-256 `8b252cdb52306506644f58be8ac5761dddd41b035c27725c95fe9c7d49f7e52f`), using the fixed experimental GPU package in [candidate-settings.json](../2026-11-05-first-decode-admission/candidate-settings.json). The package includes GPU resident routing with refresh cadence 16, four-layer submissions, early address dispatch and the two prefill options. The only new variable is `DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY=0` versus `1`; only the exact value `1` enables the set.

The workload uses `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, the `promessi_sposi.txt` prompt, a 64 GB expert cache, 4 GiB slabs, 8K prefill chunks and a 40,960-token context allocation. Each phase evaluates 128 teacher-forced tokens; the warm phase restores KV state and appends 512 tokens. The comparison covers fresh and warm phases at 8K and 32K, five alternating pairs per frontier, fixed-history logits and a separate dependency profile. It retains full prefill, the first scalar step, complete decode and process wall to detect shifted cost. The expert cache survives KV restore; the OS file cache was not flushed.

## Results and reproduction

`./tools/zig-build ds4-bench ssd-experts-test -j4` completed successfully. The guarded fixture passed its exact-output, cache-eviction and mapping checks. Its log records set addition and clearing on cache trim, plus ten seed/clear cycles with a stable 186 MiB footprint. The fixture binary SHA-256 was `b52bf1d8f2363497d3796a0ee89a0da7dbb90276c9cc7b51bd873073d8d2190f`.

A separate child process with `ulimit -l 0` exercised the mlock failure path: its log records one allocation added, then the set cleared for a slab mlock failure, with no subsequent re-add. The fixture itself exited 1 as expected because it could not lock any experts; the driver and pressure guard completed successfully. This is a fallback test, not a numerical pass. The ordinary fixture passed separately. These checks do not establish model performance. Raw artifacts are retained locally under `.workspaces/hillclimb-20261104/23-slab-residency/`. Activation must be checked from the diagnostic's residency-set add and clear records, not inferred from the flag or model-mapping residency messages. Profile timings and profile-off latency measurements are reported separately. All eight guard jobs completed without a pressure stop; all samples reported pressure level 1 and none were adverse. Swap ranged from 28.9829 to 28.9869 GiB, an increase of about 4.18 MiB.


## Profile-off timing at 32K

Across five alternating pairs, residency shortened the fresh first step in every pair, but fresh and warm complete decode and process wall were slower in every pair. Fresh prefill was shorter in four pairs, while warm append was slower in all five; inference total was slower in four. The unusually fast first baseline is retained. The inference total includes both prefill and decode phases, excluding model loading and snapshot save/restore; wall includes the process. Phase seconds are derived from throughput rounded to two decimal places.

| Pair (first arm) | Fresh first step (ms) | Fresh prefill (s) | Fresh decode (s) | Warm append (s) | Warm decode (s) | Inference (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (baseline) | 856.249 → 225.512 | 42.432 → 43.304 | 4.415 → 4.910 | 7.166 → 8.023 | 3.550 → 4.685 | 57.563 → 60.922 | 59.743 → 63.580 |
| 2 (candidate) | 815.882 → 234.304 | 46.842 → 45.300 | 4.321 → 4.852 | 7.446 → 7.703 | 3.859 → 4.801 | 62.468 → 62.656 | 64.634 → 65.283 |
| 3 (baseline) | 869.022 → 217.262 | 47.793 → 46.986 | 4.387 → 4.836 | 7.677 → 8.178 | 3.917 → 4.889 | 63.774 → 64.888 | 65.936 → 67.460 |
| 4 (candidate) | 824.168 → 252.322 | 47.371 → 46.628 | 4.323 → 4.884 | 7.849 → 8.117 | 3.920 → 4.861 | 63.463 → 64.489 | 65.600 → 67.035 |
| 5 (baseline) | 803.379 → 225.706 | 48.990 → 47.355 | 4.290 → 4.805 | 8.004 → 8.036 | 3.952 → 4.952 | 65.235 → 65.148 | 67.423 → 67.722 |
| Mean ± sample SD | 833.740 ± 27.766 → 231.021 ± 13.346 | 46.686 ± 2.506 → 45.914 ± 1.652 | 4.347 ± 0.052 → 4.857 ± 0.041 | 7.628 ± 0.331 → 8.011 ± 0.184 | 3.840 ± 0.165 → 4.838 ± 0.101 | 62.501 ± 2.933 → 63.621 ± 1.796 | 64.667 ± 2.929 → 66.216 ± 1.754 |

Warm first-step latency also increased in every pair, from 37.286 ± 2.288 to 141.738 ± 2.687 ms. All phase samples are retained locally in `timing32k/timing-samples.csv`.

## Profile-off timing at 8K

As at 32K, fresh first-step latency was lower in all five pairs, but complete fresh and warm decode, total inference and process wall were slower in all five. Warm first-step latency increased in every pair, from 35.929 to 208.736 ms on average. The same timing scopes and precision limits apply as at 32K.

| Pair (first arm) | Fresh first step (ms) | Fresh prefill (s) | Fresh decode (s) | Warm append (s) | Warm decode (s) | Inference (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (baseline) | 813.409 → 203.421 | 19.575 → 18.986 | 4.238 → 4.618 | 7.181 → 7.164 | 3.814 → 4.776 | 34.809 → 35.543 | 36.859 → 38.037 |
| 2 (candidate) | 792.758 → 211.938 | 19.006 → 18.894 | 4.227 → 4.648 | 7.389 → 7.235 | 3.781 → 4.798 | 34.404 → 35.574 | 36.522 → 37.995 |
| 3 (baseline) | 876.750 → 208.891 | 19.218 → 19.303 | 4.301 → 4.641 | 7.572 → 7.712 | 3.813 → 4.876 | 34.904 → 36.532 | 36.985 → 39.043 |
| 4 (candidate) | 819.673 → 210.980 | 19.262 → 19.273 | 4.261 → 4.670 | 7.649 → 7.636 | 3.831 → 4.887 | 35.003 → 36.467 | 37.150 → 38.974 |
| 5 (baseline) | 826.135 → 241.171 | 19.528 → 19.326 | 4.252 → 4.694 | 7.615 → 7.608 | 3.851 → 4.856 | 35.246 → 36.483 | 37.357 → 38.939 |
| Mean ± sample SD | 825.745 ± 31.140 → 215.280 ± 14.845 | 19.318 ± 0.235 → 19.156 ± 0.201 | 4.256 ± 0.028 → 4.654 ± 0.029 | 7.481 ± 0.195 → 7.471 ± 0.252 | 3.818 ± 0.026 → 4.839 ± 0.049 | 34.873 ± 0.309 → 36.120 ± 0.513 | 36.975 ± 0.314 → 38.598 ± 0.532 |

All phase samples are retained locally in `timing8k/timing-samples.csv`.

## Quality and diagnostic profile

At each 8K and 32K fresh frontier and its corresponding plus-512 warm frontier, 128 fixed-history full-vocabulary prediction rows and their post-evaluation hashes matched byte for byte: 512 of 512 rows and hashes in all. Delta NLL and KL were zero. This bounded numerical comparison does not measure autonomous task quality. The local raw `quality8k/quality-summary.json` and `quality32k/quality-summary.json` retain the results.

One intrusive 32K timeline pair measured the first scalar command scope at 822.8 → 316.9 ms with residency enabled, while the refresh scope increased from 41.0 → 176.5 ms fresh and 37.0 → 184.3 ms warm. These scopes include the wait after final commit but start after Engram reader startup; they differ from the benchmark first-step interval. Large gaps between command buffers shrink, while new gaps appear inside buffers near the selected-event boundary. Summed sampled encoder durations remain roughly unchanged; the intervals can overlap and are not a utilization breakdown. The trace does not establish the cause. A separate single native-grouping pair also showed the shift: first fresh scope 790.0 → 213.1 ms, fresh refresh 29.4 → 142.8 ms, warm first 34.7 → 136.3 ms and warm refresh 28.7 → 133.1 ms. Mean ordinary scopes changed from 27.544 → 29.803 ms fresh and 27.557 → 29.693 ms warm. These scopes also exclude Engram startup. Logging still perturbs execution, and one diagnostic pair is not a comparative latency claim.


## Reproduction

Run from the repository root, using a fresh output path for each guard. For the 8K timing and quality comparisons:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/hillclimb-20261104/23-slab-residency/repro-guard-timing8k -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/hillclimb-20261104/23-slab-residency/repro-timing8k timing --flag DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY --control 0 --candidate 1 --frontier 8192 --tokens 128 --samples 5 --extra-settings speed-bench/experiments/2026-11-05-first-decode-admission/candidate-settings.json
python3 speed-bench/ideas/pressure_run.py .workspaces/hillclimb-20261104/23-slab-residency/repro-guard-quality8k -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/hillclimb-20261104/23-slab-residency/repro-quality8k quality --flag DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY --control 0 --candidate 1 --frontier 8192 --tokens 128 --extra-settings speed-bench/experiments/2026-11-05-first-decode-admission/candidate-settings.json
```

For 32K, change `8192` to `32768` and the output names from `8k` to `32k`. The fixture command clears inherited `DS4` settings and enables residency and timing summaries:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/hillclimb-20261104/23-slab-residency/repro-guard-fixture -- nu -c 'for key in ($env | columns | where {|key| $key | str starts-with "DS4"}) { hide-env $key }; with-env {DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY: "1", DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: "1"} { ./zig-out/bin/test-metal-ssd-experts; if $env.LAST_EXIT_CODE != 0 { error make {msg: "Streaming expert fixture failed"} } }'
```
