# Parallel resident experts

Parallel execution improved mean steady decode in the attribution comparisons; the combined
route remains an explicit opt-in rather than a preset. This experiment asks whether the existing parallel feed-forward
network (FFN) machinery improves GPU-resident routing by overlapping work from
its shared and routed experts. The one-hour slice is November 4, 2026,
15:40–16:40 Eastern. The fixed runtime baseline is `cc913524`, with GPU resident
routing and address reuse enabled and a refresh every eight positions. The
candidate revision is `1b692d25`.

The [address-reuse experiment](../2026-11-04-resident-address-reuse/README.md)
removed sampled host address-publication work but found neutral attribution
throughput; its host samples were dominated by GPU completion waits. Its
combined GPU route also retained a warm 32K slowdown. This experiment tests a
GPU dependency rather than assuming that more concurrency improves throughput:
shared and routed kernels may already saturate the same memory bandwidth.

## Mechanism and boundaries

Each FFN layer has a shared expert branch and a routed branch. Both consume the
normalized input independently. Each branch first computes gate and up
projections and an intermediate activation; its down projection then consumes
that intermediate. The prototype puts the two producers in a concurrent
encoder, places a barrier over both intermediate buffers, and then dispatches
the two down projections. It closes the encoder before the subsequent HC join.
The barrier protects the producer-to-consumer dependency; closing the encoder
preserves ordering before the join.

The new `DS4_METAL_V41_RESIDENT_GPU_PARALLEL=1` opt-in admits this route only
after resident preflight has pinned the possible expert selections. The policy
checks the layer-scoped resident override, single-host execution, six selected
experts, IQ2/Q2 expert weights and the exact 384-expert, 5120/2304 shape and
strides, along with the required kernels and streaming address path. It does
not use `force_resident` or full-model expert views. Refresh and fallback use
the ordinary serial route. The stage profiler is excluded from the concurrent
scope. The existing router policy fixture was extended for this admission and
its rejection cases; source safety review passed. The bench, agent and server
builds passed, as did the router fixture and the command-memory parallel FFN
fixture, which exercises the stages and abort and incomplete-join reset. An
initial build command named a nonexistent target; the corrected `metal-test`
target built successfully. Installed root executables were not replaced.

This is more than a scheduling-only change: the shared down projection uses
the existing separate BF16 down and join path instead of the late
matmul-expand path. The mathematical and kernel boundaries and fixed-history
outputs must be checked before attributing any difference solely to scheduling.
Optional `DS4_METAL_V41_RESIDENT_GPU_PARALLEL_PROFILE=1` reports successful
activation as `joined=1`; it is unset for timing.

## Setup

The intended attribution comparison holds GPU routing, address reuse and refresh
8 fixed and varies only the parallel flag, using `resident-settings.json`.
The combined comparison holds parallel, address reuse and refresh 8 fixed and
varies GPU routing against ordinary CPU resident routing, using
`parallel-settings.json`. The ordinary route permits five substitutions with a
0.90 score gap, rank limit 64 and native slot weights. GPU routing relaxes
those limits and is an approximation; parallel attribution instead compares
the same GPU route serially and concurrently. The raw local evidence is under
`.workspaces/hillclimb-20261104/11-resident-parallel/`.

The runs used an M5 Max with 128 GiB memory and
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, SSD streaming with a
64 GB expert cache, 4 GiB slabs and 8,192-token prefill chunks. The maintained
driver and settings can reproduce each comparison with a fresh output directory
under the pressure guard. For example, replace the frontier with `8192` for 8K:

```sh
nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/resident-parallel-attribution timing --flag DS4_METAL_V41_RESIDENT_GPU_PARALLEL --frontier 32768 --extra-settings speed-bench/experiments/2026-11-04-resident-parallel/resident-settings.json
nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/resident-parallel-combined timing --flag DS4_METAL_V41_RESIDENT_GPU --frontier 32768 --extra-settings speed-bench/experiments/2026-11-04-resident-parallel/parallel-settings.json
```

## Findings and decision

An activation smoke used 32 teacher tokens at each of the 8,192 and 8,704
frontiers with the activation profile enabled. It reported 2,240 successful
joins: all 40 layers at each of 56 non-refresh positions. No joins appeared at
positions divisible by eight. This establishes activation for that smoke, not
a latency benefit. The summary and candidate stderr are in the local raw
evidence directory.

At each of 8K and 32K, all 256 fixed-history full-vocabulary logit rows and
all 256 post-evaluation hashes were byte-exact with parallel off and on. These
checks hold GPU resident routing, address reuse and refresh interval 8 fixed;
they do not compare the GPU approximation with ordinary CPU cap-five routing.
That approximation's quality cost is recorded in the
[handoff report](../2026-11-04-resident-handoff/README.md). Across
both checks, 512 of 512 rows and hashes matched; delta-NLL and KL were zero.
This does not establish sampled task equivalence. The 32K diagnostic control
had slow prefill and append (575.76 and 36.22 tokens/s) and took 82.44 seconds
of process time versus 71.40 seconds for the candidate. These single diagnostic
runs are not a speed comparison. The guard was normal, and a process check
found no competing model. All 40 timing runs succeeded under the pressure guard. Across all guards,
pressure remained at level 1, no bad samples were recorded, and swap ranged
from 28.8469 to 28.8547 GiB.

### 32K attribution timing

Five profile-free pairs alternated order, with control first in pairs 1, 3 and
5. Each run used a fresh 32,768-token prefill and 128 teacher-forced decode
tokens, then KV restore, a 512-token append and another 128 decode tokens.
The expert cache survived restore; the OS cache was not flushed. Phase times
exclude loading and snapshot save/restore; steady decode excludes the first
token and complete decode includes it. Phase seconds are derived from
two-decimal throughput. Values are mean ± sample SD.

Both arms show much longer first-token latency after fresh prefill than after
warm append. This experiment did not isolate that cause or attribute it to
concurrency. Steady rates exclude that token; complete decode windows include it.

| Measure | Parallel off | Parallel on |
| --- | --- | --- |
| Fresh prefill, s | 48.956 ± 2.614 | 49.582 ± 1.348 |
| Warm append, s | 7.880 ± 0.177 | 7.790 ± 0.228 |
| Fresh steady decode, tokens/s | 34.366 ± 0.683 | 35.630 ± 0.069 |
| Warm steady decode, tokens/s | 31.896 ± 0.581 | 32.650 ± 0.117 |
| Fresh first token, ms | 837.655 ± 30.252 | 865.329 ± 45.231 |
| Warm first token, ms | 35.272 ± 1.583 | 34.695 ± 0.659 |
| Complete fresh decode, s | 4.535 ± 0.087 | 4.430 ± 0.051 |
| Complete warm decode, s | 4.018 ± 0.072 | 3.925 ± 0.014 |
| Inference phases, s | 65.389 ± 2.750 | 65.727 ± 1.606 |
| Wall, s | 67.601 ± 2.761 | 67.899 ± 1.625 |

All samples follow in pair order, parallel off then on.

| Measure | Parallel off | Parallel on |
| --- | --- | --- |
| Fresh prefill, s | 44.361, 49.532, 50.024, 50.876, 49.987 | 47.361, 49.590, 50.965, 49.799, 50.196 |
| Warm append, s | 7.620, 7.846, 8.112, 7.929, 7.891 | 7.400, 7.780, 7.966, 7.900, 7.904 |
| Fresh steady decode, tokens/s | 33.150, 34.670, 34.590, 34.780, 34.640 | 35.710, 35.620, 35.580, 35.550, 35.690 |
| Warm steady decode, tokens/s | 32.900, 31.670, 31.510, 31.520, 31.880 | 32.710, 32.520, 32.580, 32.620, 32.820 |
| Fresh first token, ms | 851.747, 843.517, 785.612, 843.593, 863.808 | 807.484, 921.863, 886.903, 877.228, 833.169 |
| Warm first token, ms | 33.044, 34.204, 35.932, 36.633, 36.548 | 35.813, 34.161, 34.745, 34.387, 34.370 |
| Complete fresh decode, s | 4.683, 4.507, 4.457, 4.496, 4.531 | 4.364, 4.488, 4.457, 4.451, 4.391 |
| Complete warm decode, s | 3.893, 4.044, 4.066, 4.066, 4.020 | 3.919, 3.940, 3.934, 3.928, 3.904 |
| Inference phases, s | 60.557, 65.929, 66.659, 67.368, 66.430 | 63.045, 65.798, 67.322, 66.078, 66.394 |
| Wall, s | 62.750, 68.122, 68.951, 69.547, 68.635 | 65.204, 67.930, 69.539, 68.209, 68.612 |

Fresh steady decode favored parallel in all five pairs; warm steady decode
favored it in four. The first control was unusually slow for fresh steady
decode (33.15 tokens/s) and unusually fast for warm steady decode (32.90
tokens/s). The first control prefill was also unusually fast at 44.361 s,
where the flag is inactive. Inference and wall times do not establish a
whole-workload gain, and prefill drift affects both.

### 32K combined timing

The same five-pair procedure compares ordinary CPU resident routing with the
combined GPU routing and parallel path. Values are mean ± sample SD.

| Measure | Ordinary route | Combined route |
| --- | --- | --- |
| Fresh prefill, s | 49.772 ± 0.481 | 49.613 ± 0.593 |
| Warm append, s | 7.760 ± 0.289 | 7.847 ± 0.117 |
| Fresh steady decode, tokens/s | 34.688 ± 0.049 | 35.554 ± 0.098 |
| Warm steady decode, tokens/s | 32.978 ± 0.222 | 32.940 ± 0.163 |
| Fresh first token, ms | 812.572 ± 17.348 | 836.115 ± 44.646 |
| Warm first token, ms | 41.315 ± 11.882 | 33.968 ± 0.226 |
| Complete fresh decode, s | 4.474 ± 0.015 | 4.408 ± 0.044 |
| Complete warm decode, s | 3.893 ± 0.031 | 3.889 ± 0.019 |
| Inference phases, s | 65.899 ± 0.799 | 65.757 ± 0.725 |
| Wall, s | 68.071 ± 0.804 | 67.897 ± 0.765 |

All samples follow in pair order, ordinary then combined.

| Measure | Ordinary route | Combined route |
| --- | --- | --- |
| Fresh prefill, s | 48.968, 49.730, 49.906, 50.082, 50.176 | 48.680, 49.405, 49.836, 50.178, 49.966 |
| Warm append, s | 7.244, 7.861, 7.883, 7.911, 7.901 | 7.668, 7.793, 7.955, 7.889, 7.928 |
| Fresh steady decode, tokens/s | 34.650, 34.740, 34.640, 34.740, 34.670 | 35.380, 35.610, 35.580, 35.600, 35.600 |
| Warm steady decode, tokens/s | 33.370, 32.890, 32.860, 32.840, 32.930 | 33.180, 33.020, 32.860, 32.880, 32.760 |
| Fresh first token, ms | 793.556, 803.426, 811.001, 839.969, 814.906 | 823.117, 814.432, 818.677, 808.920, 915.427 |
| Warm first token, ms | 35.926, 37.783, 62.458, 34.324, 36.086 | 33.628, 33.845, 34.115, 34.089, 34.163 |
| Complete fresh decode, s | 4.458, 4.460, 4.477, 4.496, 4.479 | 4.412, 4.381, 4.388, 4.376, 4.483 |
| Complete warm decode, s | 3.843, 3.899, 3.928, 3.901, 3.893 | 3.861, 3.880, 3.899, 3.896, 3.911 |
| Inference phases, s | 64.513, 65.950, 66.193, 66.390, 66.449 | 64.622, 65.459, 66.078, 66.339, 66.288 |
| Wall, s | 66.680, 68.117, 68.358, 68.565, 68.637 | 66.690, 67.607, 68.211, 68.498, 68.481 |

Every fresh steady decode pair favored the combined route, a roughly 2.5%
gain in mean throughput. Warm steady decode was close and the warm regression
in the prior experiment did not recur in this set. Inference and wall times
do not demonstrate a whole-workload gain. One ordinary-route warm first token
was 62.458 ms; that outlier is retained separately from steady decode.

### 8K combined timing

The same five-pair procedure used an 8,192-token fresh prefill. Values are
mean ± sample SD.

| Measure | Ordinary route | Combined route |
| --- | --- | --- |
| Fresh prefill, s | 20.234 ± 0.232 | 19.959 ± 0.395 |
| Warm append, s | 7.590 ± 0.368 | 7.635 ± 0.341 |
| Fresh steady decode, tokens/s | 35.000 ± 0.727 | 36.292 ± 0.329 |
| Warm steady decode, tokens/s | 32.866 ± 0.329 | 33.408 ± 0.547 |
| Fresh first token, ms | 815.691 ± 29.425 | 842.973 ± 19.418 |
| Warm first token, ms | 33.534 ± 0.369 | 47.878 ± 21.317 |
| Complete fresh decode, s | 4.446 ± 0.090 | 4.343 ± 0.030 |
| Complete warm decode, s | 3.898 ± 0.038 | 3.850 ± 0.059 |
| Inference phases, s | 36.168 ± 0.600 | 35.788 ± 0.727 |
| Wall, s | 38.239 ± 0.616 | 37.865 ± 0.737 |

All samples follow in pair order, ordinary then combined.

| Measure | Ordinary route | Combined route |
| --- | --- | --- |
| Fresh prefill, s | 20.262, 19.860, 20.199, 20.422, 20.428 | 19.484, 19.581, 20.160, 20.299, 20.273 |
| Warm append, s | 7.021, 7.467, 7.669, 7.959, 7.835 | 7.036, 7.727, 7.762, 7.762, 7.889 |
| Fresh steady decode, tokens/s | 35.260, 35.500, 35.420, 33.730, 35.090 | 36.490, 35.710, 36.480, 36.420, 36.360 |
| Warm steady decode, tokens/s | 33.400, 32.660, 32.970, 32.670, 32.630 | 34.070, 33.930, 33.130, 32.940, 32.970 |
| Fresh first token, ms | 772.458, 849.864, 811.745, 834.969, 809.421 | 869.554, 833.605, 857.138, 830.529, 824.041 |
| Warm first token, ms | 33.154, 33.256, 33.707, 33.481, 34.073 | 32.923, 80.731, 33.616, 33.720, 58.402 |
| Complete fresh decode, s | 4.375, 4.428, 4.397, 4.601, 4.429 | 4.351, 4.391, 4.338, 4.318, 4.317 |
| Complete warm decode, s | 3.836, 3.922, 3.886, 3.920, 3.926 | 3.761, 3.824, 3.867, 3.889, 3.910 |
| Inference phases, s | 35.494, 35.676, 36.151, 36.902, 36.618 | 34.632, 35.523, 36.127, 36.269, 36.388 |
| Wall, s | 37.549, 37.705, 38.261, 38.968, 38.714 | 36.684, 37.624, 38.183, 38.346, 38.489 |

Fresh and warm steady decode favored the combined route in every pair. One
fresh control was unusually slow at 33.73 tokens/s. Total inference and wall
times also include prefill drift, where the route is inactive; they cannot be
attributed wholly to the change.

The 8K combined candidate also had warm first-token outliers of 80.731 and
58.402 ms; steady decode excludes those tokens.

### 8K attribution timing

The same five-pair procedure compared the serial and parallel GPU routes at
an 8,192-token fresh frontier. Values are mean ± sample SD.

| Measure | Parallel off | Parallel on |
| --- | --- | --- |
| Fresh prefill, s | 20.433 ± 0.294 | 20.195 ± 0.243 |
| Warm append, s | 7.796 ± 0.303 | 7.706 ± 0.274 |
| Fresh steady decode, tokens/s | 35.270 ± 0.560 | 35.974 ± 0.717 |
| Warm steady decode, tokens/s | 31.770 ± 0.502 | 33.006 ± 0.141 |
| Fresh first token, ms | 815.757 ± 22.782 | 823.444 ± 24.691 |
| Warm first token, ms | 39.141 ± 10.391 | 44.316 ± 21.608 |
| Complete fresh decode, s | 4.417 ± 0.062 | 4.355 ± 0.086 |
| Complete warm decode, s | 4.037 ± 0.074 | 3.893 ± 0.022 |
| Inference phases, s | 36.683 ± 0.313 | 36.148 ± 0.494 |
| Wall, s | 38.780 ± 0.328 | 38.238 ± 0.505 |

All samples follow in pair order, parallel off then on.

| Measure | Parallel off | Parallel on |
| --- | --- | --- |
| Fresh prefill, s | 20.808, 20.694, 20.216, 20.243, 20.202 | 20.168, 19.801, 20.256, 20.293, 20.455 |
| Warm append, s | 7.458, 7.654, 7.765, 7.828, 8.277 | 7.290, 7.577, 7.798, 7.921, 7.945 |
| Fresh steady decode, tokens/s | 35.430, 35.510, 35.660, 35.470, 34.280 | 36.500, 36.350, 34.720, 36.120, 36.180 |
| Warm steady decode, tokens/s | 31.900, 32.120, 32.060, 31.880, 30.890 | 33.120, 33.160, 33.020, 32.830, 32.900 |
| Fresh first token, ms | 853.622, 803.594, 794.565, 817.696, 809.310 | 791.431, 840.900, 839.742, 843.091, 802.055 |
| Warm first token, ms | 33.698, 36.128, 34.380, 33.850, 57.649 | 82.814, 33.387, 33.676, 38.042, 33.661 |
| Complete fresh decode, s | 4.438, 4.381, 4.355, 4.399, 4.513 | 4.271, 4.335, 4.498, 4.358, 4.313 |
| Complete warm decode, s | 4.014, 3.990, 3.996, 4.018, 4.168 | 3.918, 3.864, 3.880, 3.907, 3.894 |
| Inference phases, s | 36.718, 36.719, 36.332, 36.486, 37.160 | 35.647, 35.577, 36.431, 36.479, 36.607 |
| Wall, s | 38.785, 38.824, 38.432, 38.565, 39.292 | 37.743, 37.636, 38.564, 38.557, 38.688 |

In this 8K attribution set, the retained warm first-token outliers were
82.814 ms with parallel execution and 57.649 ms with serial execution.
Complete warm decode includes those tokens.

Fresh steady decode time changed −2.93%, −2.31%, +2.71%, −1.80% and
−5.25% in pair order; warm steady time fell 2.89–6.11% in every pair.
Prefill drift affects the total inference and wall differences; the parallel
flag is inactive in fresh prefill.

## Decision

Retain concurrency as an explicit opt-in within the experimental GPU route;
do not promote it to a preset or installed root binary yet. The next bounded
question is the refresh 4/8/16 speed–quality tradeoff, using 512 teacher
tokens per phase before a preset choice.
