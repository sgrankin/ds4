# Shared-expert submission on CPU refreshes

Skipping the separate shared-expert submission did not improve this workload: fresh complete decode was slower in nine of ten pairs, while whole-inference timing was mixed. Do not adopt the standalone selector. The diagnostic establishes a scheduling opportunity for a different, untested parallel path; it does not establish the cause of the slowdown.

## Context

A mixture-of-experts layer combines a shared expert, used for every token, with routed experts selected for that token. The shared gate and up projections need the layer input and static weights; the routed projections need the selected expert weights. On the overlap-off CPU-refresh path, the host first reads and applies the routing decision and starts loading the selected weights. It then encodes the shared projections and flushes them before entering routed execution. That submission can run shared GPU work while selected weights arrive from storage. If the selected weights are already cached, the separate submission may instead add overhead without hiding a meaningful load. This is the hypothesis under test; actual misses may favor the existing schedule.

The [previous experiment](../2026-11-05-slab-event-interaction/README.md) retained queue-attached slab residency with shared-event overlap off as an experimental latency candidate. Its single native-grouping diagnostic found additional time in later CPU refreshes, alongside variation in ordinary steps. This experiment holds that package fixed and changes only the separate shared-work flush.

## Setup

The selected one-hour slice is November 5, 2026, 03:26–04:26 Eastern. The baseline is runtime `8bf87bbd`, closure `d0af7acc`, with the experiment 25 candidate settings: queue residency on, shared-event overlap off, GPU resident routing, refresh interval 16, four-layer submissions, a 64 GB expert-cache target, 4 GiB slabs and 8K prefill chunks. Both arms use runtime `6f9a7e77` and the same `ds4-bench` binary (SHA-256 `ee4661ab7f7f262a0924479b06f35f17e090b24068ec95c69644f8c8a0ab3a66`), built without warnings, with `DS4_METAL_V41_SKIP_SHARED_LOAD_FLUSH` set to `0` and `1` respectively. The historical selector recognized the exact value `1`.

In the historical `6f9a7e77` implementation, the selector applied only to streaming, scalar CPU-refresh execution with shared overlap off, and excludes quality and imatrix modes, image input, tensor parallelism, disabled short optimizations, asynchronous expert loading, GPU-resident routing for the layer, deferred HC input, predicted loading and route-oracle paths. Selected reads still started before shared encoding, and routed execution still received the selected IDs. With the selector on, shared work stayed queued through the selected-load join and preceded the later shared/routed join; the change did not deliberately alter selections, numerical boundaries, cache pins or pressure relief. A cache miss can leave the GPU idle while the host waits for the selected weights, because shared work has not yet been submitted.

The comparison used fixed-history quality and five alternating timing pairs at each of 8K and 32K, recording fresh and warm first steps, complete decode, prefill or append, inference and process wall. The 512-row fixed-history quality check was the primary numerical comparison. A bounded 8K quality diagnostic requested a 16 GB cache to exercise misses; it was not a performance comparison. A separate per-token diagnostic distinguished refresh from ordinary steps without changing encoder grouping. Raw artifacts are retained locally under `.workspaces/hillclimb-20261104/26-refresh-shared-flush/`.

## Findings

The two arms produced byte-identical full-vocabulary teacher-logit rows on all four fixed histories: 128 rows each at 8,192, 8,704, 32,768 and 33,280 tokens, for 512 of 512 rows. All 512 post-evaluation hashes matched; delta NLL and KL were zero and there were no top-1 mismatches. This fixed-history check does not establish autonomous task quality.

The intrusive 17-token 32K diagnostic confirmed the intended structural change: refreshes used 91 command buffers with the flush and 51 without it, while all ordinary windows used 11. Shared gate/up and routed kernels moved into the same buffer at all 40 refresh layers; normalized model kernel-label counts matched across all 34 paired windows. The diagnostic is not comparative latency evidence. Its cache counters showed no evictions, though pending reads occurred; cumulative read bytes were 1.15 GiB in each arm. The local `timeline-diagnostic-note.md` records the trace boundary correction and scope.

## Profile-off timing at 32K

Fresh complete decode, fresh first step and steady throughput favored the flush in four of five pairs. Warm complete decode was mixed. Total inference and wall were shorter without the flush only in the two pairs in which the candidate ran first. Both arms drifted upward over the sequence, so order is a material limitation.

The workload uses DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix on an M5 Max with 128 GiB memory. Fresh and warm phases each evaluate 128 teacher-forced tokens; warm restores KV and appends 512 tokens. The expert cache survives restore and the OS cache was not flushed. Inference excludes model loading and snapshot save/restore; wall includes them. Phase seconds derive from throughput rounded to two decimals. Steady throughput excludes the first token and does not prove initialization is complete. Entries show flush kept → skipped; SD is sample SD.


| Pair (first arm) | Fresh first (ms) | Fresh prefill (s) | Fresh decode (s) | Fresh steady (tokens/s) | Warm append (s) | Warm first (ms) | Warm decode (s) | Inference (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (baseline) | 174.751 → 201.690 | 42.256 → 42.258 | 3.739 → 3.938 | 35.630 → 33.980 | 7.250 → 7.501 | 45.014 → 41.163 | 3.585 → 3.756 | 56.831 → 57.453 | 59.284 → 59.961 |
| 2 (candidate) | 215.419 → 210.176 | 45.007 → 43.860 | 3.797 → 3.771 | 35.460 → 35.670 | 7.555 → 7.063 | 40.970 → 42.011 | 3.738 → 3.651 | 60.098 → 58.346 | 62.772 → 60.914 |
| 3 (baseline) | 166.330 → 247.340 | 45.847 → 46.507 | 3.721 → 3.828 | 35.730 → 35.470 | 7.456 → 7.748 | 51.694 → 50.757 | 3.895 → 3.963 | 60.920 → 62.046 | 63.492 → 64.635 |
| 4 (candidate) | 184.784 → 210.916 | 46.887 → 46.328 | 3.750 → 3.777 | 35.620 → 35.610 | 7.837 → 7.729 | 45.537 → 47.489 | 4.011 → 3.983 | 62.486 → 61.817 | 65.035 → 64.468 |
| 5 (baseline) | 187.990 → 199.963 | 46.993 → 47.070 | 3.743 → 3.777 | 35.730 → 35.510 | 7.799 → 7.792 | 47.733 → 47.729 | 3.994 → 4.033 | 62.528 → 62.671 | 65.080 → 65.238 |
| Mean ± SD | 185.855 ± 18.603 → 214.017 ± 19.263 | 45.398 ± 1.936 → 45.205 ± 2.057 | 3.750 ± 0.028 → 3.818 ± 0.071 | 35.634 ± 0.111 → 35.248 ± 0.713 | 7.579 ± 0.244 → 7.567 ± 0.303 | 46.190 ± 3.929 → 45.830 ± 4.093 | 3.845 ± 0.181 → 3.877 ± 0.165 | 60.572 ± 2.336 → 60.467 ± 2.385 | 63.133 ± 2.371 → 63.043 ± 2.420 |

## Profile-off timing at 8K

Fresh complete decode was slower and fresh steady throughput lower without the flush in all five pairs. Warm decode and total inference were shorter without it in two pairs each; the first baseline warm decode is an outlier and remains included. The same timing scope and rounding limitations apply as at 32K.

| Pair (first arm) | Fresh first (ms) | Fresh prefill (s) | Fresh decode (s) | Fresh steady (tokens/s) | Warm append (s) | Warm first (ms) | Warm decode (s) | Inference (s) | Wall (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (baseline) | 177.229 → 197.492 | 19.523 → 19.342 | 3.658 → 3.691 | 36.480 → 36.350 | 7.324 → 7.406 | 108.906 → 46.408 | 4.157 → 3.898 | 34.662 → 34.337 | 37.160 → 36.770 |
| 2 (candidate) | 191.827 → 202.758 | 19.132 → 19.036 | 3.674 → 3.695 | 36.470 → 36.360 | 7.590 → 7.505 | 44.494 → 47.939 | 3.916 → 3.874 | 34.311 → 34.110 | 36.737 → 36.599 |
| 3 (baseline) | 171.385 → 202.011 | 19.172 → 19.175 | 3.651 → 3.693 | 36.500 → 36.380 | 7.675 → 7.711 | 44.968 → 45.840 | 3.901 → 3.924 | 34.399 → 34.503 | 36.796 → 36.910 |
| 4 (candidate) | 223.508 → 186.824 | 19.212 → 19.244 | 3.698 → 3.845 | 36.550 → 34.720 | 7.681 → 7.795 | 43.543 → 43.453 | 3.899 → 3.916 | 34.490 → 34.800 | 36.953 → 37.276 |
| 5 (baseline) | 209.448 → 208.650 | 19.157 → 20.054 | 3.688 → 3.739 | 36.510 → 35.970 | 7.726 → 7.512 | 45.862 → 46.281 | 3.899 → 3.911 | 34.470 → 35.216 | 36.906 → 37.684 |
| Mean ± SD | 194.679 ± 21.820 → 199.547 ± 8.145 | 19.239 ± 0.161 → 19.370 ± 0.398 | 3.674 ± 0.020 → 3.733 ± 0.066 | 36.502 ± 0.031 → 35.956 ± 0.712 | 7.599 ± 0.162 → 7.586 ± 0.161 | 57.555 ± 28.719 → 45.984 ± 1.621 | 3.954 ± 0.114 → 3.904 ± 0.019 | 34.466 ± 0.130 → 34.593 ± 0.429 | 36.911 ± 0.164 → 37.048 ± 0.434 |

## Miss coverage and native timing diagnostic

The 16 GB requested-cache quality pair added 256 byte-identical logit rows and 256 matching post-evaluation hashes, with zero delta NLL, KL and top-1 differences. Across both cache settings that is 768 paired rows; the 8K histories were repeated under the smaller cache, so these are not 768 unique positions. The actual dynamic cache was capped at 958 experts (8.88 GiB), with 7.12 GiB reserved for prefill headroom. Three slabs occupied 8.89 GiB. Across 14 later scalar refreshes, both arms loaded the same per-step and per-layer sequence of 171 experts through 513 tensor reads; none appeared in the 240 ordinary intervals. The workload accumulated 3,993 evictions; the logs do not establish when each eviction occurred. This checks miss-path behavior and quality, not miss-path speed.

One separate 32K native-grouping pair measured the full nonspeculative evaluation interval per token. After the first fresh step, the candidate added 86.588 ms over 127 steps: 15.790 ms across seven refreshes and 70.798 ms across 120 ordinary steps. The corresponding warm difference was 63.983 ms, comprising 10.184 ms on refreshes and 53.799 ms on ordinary steps. This single diagnostic pair does not attribute the difference or establish a comparative effect; it does show that its observed cost was not confined to refreshes.

The old flush can overlap shared GPU work with CPU cache and address preparation even on hits, as well as with I/O on misses. That opportunity is not proof of the regression's cause: the intrusive diagnostic's measured bind time was only about 2 ms across 160 selected calls and excludes other preparation. Removing the flush also removes a command-buffer reset that obstructs the existing parallel join, but this experiment did not arm that join. A future exact-selected all-hit path would need valid addresses and pins for those selected experts, normal cache accounting and a serial fallback for misses; widening the engine Boolean alone cannot establish the existing resident-only proof. Such a path would change the shared-down schedule and needs its own speed and quality qualification.

All seven guarded jobs completed with no bad samples or pressure stops. Observed pressure remained at level 1 and swap at 29.0029296875 GiB. The candidate selector was removed after the experiment; runtime `6f9a7e77` and the preserved candidate binary are historical evidence. Runtime source was restored to `8bf87bbd` and rebuilt without warnings; the restored benchmark hash is `c7b12c4f1222bbad37f25a754474af45233b5d32b3d12c336aae6c5fde61627d`. Installed frontends, presets and main were unchanged.

## Reproduction

Reconstruct and build source revision `6f9a7e77` in a matching checkout before reproducing this test: the current source no longer contains the selector, and Metal source files are loaded relative to the working directory. Verify the benchmark hash above. The local preserved binary is `.workspaces/hillclimb-20261104/26-refresh-shared-flush/ds4-bench-candidate`; it also requires matching sources. From that checkout, run the maintained driver under the pressure guard, using fresh output paths:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/repro26-guard -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/repro26-timing timing --flag DS4_METAL_V41_SKIP_SHARED_LOAD_FLUSH --control 0 --candidate 1 --extra-settings speed-bench/experiments/2026-11-05-slab-event-interaction/candidate-settings.json --frontier 32768 --tokens 128 --samples 5
```

Use frontier `8192` for 8K and `quality` in place of `timing` for a fixed-history pair. The raw local `runtime.json`, summaries and logs record the tested settings and individual runs.
