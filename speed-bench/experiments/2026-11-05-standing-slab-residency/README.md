# Standing expert-slab residency request

**Do not select the standing request policy.** It avoided the queue-attached candidate’s repeated diagnostic refresh waits, but the original first-step gaps remained and five-pair timing did not establish an overall gain. The small timing differences are mixed and uncertain. Retain the infrastructure off for a bounded follow-up on queue and event interaction; presets, installed binaries and main remain unchanged.

## Context and mechanism

The [queue-attached experiment](../2026-11-05-streaming-slab-residency/README.md) shortened the fresh first step at both 8K and 32K, but complete fresh and warm decode slowed in all five pairs at each frontier. Its intrusive profile showed fewer gaps between command buffers and new waits near shared-event boundaries; normal encoder grouping also showed the shift. The driver mechanism remains unproven.

The expert cache packs slots into slab buffers. With the opt-in flag enabled, the candidate adds each actual slab allocation to a Metal residency set before publishing its slots. It commits the initial set and issues one standing `requestResidency`; later additions are committed to that set. The earlier candidate attached the set to the runtime command queue instead. A request and queue attachment are different Metal mechanisms, and neither is a guarantee of permanent residency under pressure. The set covers entire slab buffers, including unused slots.

Existing encoder `useResource` declarations, cache pins, selected-ID overlap and cache policy remain in place. The request ends and the set is cleared before pressure relief, slot unlock or trimming, and on cache clear or teardown. Failure falls back to ordinary declarations and disables further attempts until a full cache clear. See [`expert_cache.inc.m`](../../../src/metal/expert_cache.inc.m), [`expert_storage.zig`](../../../src/metal/expert_storage.zig) and [`expert_cache.zig`](../../../src/metal/expert_cache.zig).

## Setup

The one-hour slice is November 5, 02:04–03:04 Eastern on an M5 Max with 128 GiB memory. Both arms use the unchanged experimental GPU package in [candidate-settings.json](../2026-11-05-first-decode-admission/candidate-settings.json). At the new candidate revision, `DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY=0` disables the set and `1` selects the standing request; the same flag selected queue attachment at the prior revision. Both arms use runtime `d49d027a` (parent `e2f022cb`) and the same `ds4-bench` binary, SHA-256 `9c3bf7b342a9f49ec6e4795ef45f72350cfcd48a9d017516a665967c0ee3c7f5`.

The fixed workload uses `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, a 64 GB expert-cache target, 4 GiB slabs, 8K prefill chunks, four-layer submissions and a 40,960-token context allocation. Each fresh and warm phase evaluates 128 teacher-forced tokens; the warm phase restores KV state and appends 512 tokens. The comparison uses five alternating pairs at each of 8K and 32K, a bounded cache-lifetime fixture, fixed-history quality checks and a separate diagnostic profile. The expert cache survives KV restore; the OS file cache was not flushed.

Fresh and warm first-step latency and complete decode windows, fresh prefill, warm append, inference totals and process wall capture costs shifted between phases. The diagnostic profile is separate from profile-off latency measurements. Raw artifacts are retained under `.workspaces/hillclimb-20261104/24-standing-slab-residency/`.

## Results and decision

`./tools/zig-build ds4-bench ssd-experts-test -j4` completed. The guarded fixture passed exact-output, cache-eviction and mapping checks, with repeated seed/release cycles at approximately 186.3 MiB. Its lifecycle logs record `mode=request-only` additions and clearing, including trim and cache clear. The fixture binary SHA-256 was `046b03b976557436779cda5df1811299334dadc53dcb2a89799ba2689973192b`.

A separate guarded driver forced the child process's locked-memory limit to zero. Its log records one allocation added, then a clear for slab mlock failure, with no subsequent addition. The underlying fixture exited 1 as expected because it could not lock experts; the driver exited 0. This exercises fallback, not numerical correctness or model performance. Raw logs are retained under `.workspaces/hillclimb-20261104/24-standing-slab-residency/`.

At the 8K and 32K fresh frontiers and their corresponding plus-512 warm frontiers, all 512 fixed-history full-vocabulary prediction rows and post-evaluation hashes matched. Delta NLL and KL were zero. This bounded comparison does not establish autonomous task quality.

A separate 32K native-grouping diagnostic did not reproduce the earlier queue-attached candidate's fresh-step improvement: the first fresh command scope was 845.1 → 817.5 ms, fresh refresh 30.5 → 29.5 ms, warm first 34.8 → 34.5 ms, and warm refresh 28.8 → 28.8 ms, baseline to standing request. These scopes are diagnostic measurements, not the complete benchmark first-step interval or comparative latency evidence. The detailed trace recorded 14 slabs totaling 55.98 GiB in `request-only` mode and only the final clear. Both arms reported 6,014 entries against a 6,136 budget, 123,716 hits, 6,014 misses, 14 allocations and no evictions. This confirms activation and matching observed cache decisions in that diagnostic; it does not establish the driver cause of the earlier delays. The local `native32k/native-profile-summary.json` and `timeline32k/profile-summary.json` retain diagnostic details.

## Profile-off timing at 32K

Differences are small and mixed across five alternating pairs; these samples do not establish a speed benefit or slowdown. Inference total covers both prefill and decode phases, excluding model loading and snapshot save/restore; process wall includes the process. Phase seconds are derived from throughput rounded to two decimal places. The extra wall time can include startup, snapshot work or cleanup; its cause has not been attributed.

| Pair (first arm) | Fresh first (ms) | Fresh prefill (s) | Fresh decode (s) | Warm append (s) | Warm first (ms) | Warm decode (s) | Inference (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (baseline) | 831.796 → 800.516 | 44.605 → 44.280 | 4.518 → 4.537 | 7.591 → 7.760 | 37.723 → 34.568 | 3.682 → 3.807 | 60.397 → 60.385 | 62.499 → 62.954 |
| 2 (candidate) | 862.076 → 864.684 | 46.031 → 45.408 | 4.361 → 4.376 | 7.496 → 7.064 | 39.103 → 33.798 | 3.854 → 3.711 | 61.743 → 60.560 | 63.933 → 63.150 |
| 3 (baseline) | 849.126 → 823.133 | 46.801 → 47.096 | 4.355 → 4.321 | 7.637 → 7.700 | 38.722 → 34.095 | 3.905 → 3.911 | 62.698 → 63.029 | 64.818 → 65.607 |
| 4 (candidate) | 803.098 → 851.831 | 47.176 → 47.067 | 4.316 → 4.360 | 7.635 → 7.722 | 35.591 → 35.055 | 3.913 → 3.917 | 63.040 → 63.066 | 65.226 → 65.699 |
| 5 (baseline) | 814.186 → 821.736 | 47.398 → 47.607 | 4.336 → 4.327 | 7.738 → 7.735 | 34.767 → 35.902 | 3.942 → 3.967 | 63.414 → 63.636 | 65.550 → 66.193 |

Mean ± sample SD for fresh first was 832.056 ± 24.241 → 832.380 ± 25.671 ms, fresh decode 4.377 ± 0.081 → 4.384 ± 0.088 s, warm decode 3.859 ± 0.104 → 3.863 ± 0.103 s, inference 62.258 ± 1.212 → 62.135 ± 1.538 s and wall 64.405 ± 1.226 → 64.720 ± 1.541 s. Warm first averaged 37.181 → 34.684 ms, lower in four pairs. Fresh prefill averaged 46.402 ± 1.131 → 46.292 ± 1.397 s, lower in three pairs.

The separate timeline profile found first-step gaps between command buffers of 717.202 → 669.345 ms across the same 13 expert layers. The approximately 0.03 ms intra-buffer gaps at refresh step 16 were similar between arms; the much larger waits observed with queue attachment in the earlier experiment were absent here. These are separate diagnostic runs, not paired comparative timing, and do not establish the cause of either pattern. See local `profile-comparison-note.md` and `profile-comparison-summary.json`.

## Profile-off timing at 8K

The same timing scopes and precision limits apply as at 32K. Differences are mixed and do not establish a gain or slowdown.

| Pair (first arm) | Fresh first (ms) | Fresh prefill (s) | Fresh decode (s) | Warm append (s) | Warm first (ms) | Warm decode (s) | Inference (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (baseline) | 876.911 → 802.114 | 19.021 → 18.305 | 4.307 → 4.229 | 6.699 → 6.707 | 32.307 → 35.891 | 3.443 → 3.452 | 33.470 → 32.693 | 35.501 → 35.151 |
| 2 (candidate) | 810.813 → 823.808 | 18.203 → 18.309 | 4.414 → 4.348 | 7.208 → 6.959 | 33.152 → 33.354 | 3.646 → 3.516 | 33.471 → 33.133 | 35.517 → 35.618 |
| 3 (baseline) | 829.303 → 807.068 | 18.549 → 18.749 | 4.474 → 4.583 | 7.379 → 7.507 | 35.143 → 34.206 | 3.696 → 3.742 | 34.098 → 34.581 | 36.197 → 37.014 |
| 4 (candidate) | 862.190 → 806.165 | 18.884 → 18.604 | 4.284 → 4.233 | 6.986 → 6.708 | 34.360 → 32.447 | 3.730 → 3.628 | 33.883 → 33.173 | 35.936 → 35.664 |
| 5 (baseline) | 823.247 → 852.858 | 19.065 → 19.289 | 4.250 → 4.278 | 7.141 → 7.328 | 34.144 → 34.357 | 3.761 → 3.784 | 34.217 → 34.679 | 36.287 → 37.153 |

Mean ± sample SD: fresh first 840.493 ± 27.841 → 818.403 ± 20.976 ms; fresh decode 4.346 ± 0.094 → 4.334 ± 0.147 s; warm decode 3.655 ± 0.126 → 3.624 ± 0.142 s; inference 33.828 ± 0.348 → 33.652 ± 0.913 s; and wall 35.887 ± 0.369 → 36.120 ± 0.903 s. Fresh first, fresh decode and inference were lower in three pairs, warm decode and wall in two.

All eight guarded jobs completed without a pressure stop or bad sample; observed pressure was level 1 throughout. Swap ranged from 28.9869 to 29.0037 GiB, rising about 17.2 MiB.

## Reproduction

Run from the repository root with fresh output paths. For 32K replace `8192` with `32768` and the output names. The same guarded driver supplies the fixed settings for both arms.

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/hillclimb-20261104/24-standing-slab-residency/repro-guard-timing8k -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/hillclimb-20261104/24-standing-slab-residency/repro-timing8k timing --flag DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY --control 0 --candidate 1 --frontier 8192 --tokens 128 --samples 5 --extra-settings speed-bench/experiments/2026-11-05-first-decode-admission/candidate-settings.json
python3 speed-bench/ideas/pressure_run.py .workspaces/hillclimb-20261104/24-standing-slab-residency/repro-guard-quality8k -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/hillclimb-20261104/24-standing-slab-residency/repro-quality8k quality --flag DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY --control 0 --candidate 1 --frontier 8192 --tokens 128 --extra-settings speed-bench/experiments/2026-11-05-first-decode-admission/candidate-settings.json
python3 speed-bench/ideas/pressure_run.py .workspaces/hillclimb-20261104/24-standing-slab-residency/repro-guard-fixture -- nu -c 'for key in ($env | columns | where {|key| $key | str starts-with "DS4"}) { hide-env $key }; with-env {DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY: "1", DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: "1"} { ./zig-out/bin/test-metal-ssd-experts; if $env.LAST_EXIT_CODE != 0 { error make {msg: "Streaming expert fixture failed"} } }'
```
