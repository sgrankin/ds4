# Resident address reuse

Removing redundant address publication did not demonstrate a throughput gain
in the tested attribution comparisons. The change is retained as an explicit
opt-in cleanup for the experimental GPU route; no preset or installed binary
is promoted. This experiment asks whether the GPU
resident routing prototype benefits from removing redundant host publication
of expert addresses. The one-hour slice is November 4, 2026, 14:48–15:48
Eastern. The preceding [resident handoff experiment](../2026-11-04-resident-handoff/README.md)
reduced the instrumented router-to-expert gap but did not establish a throughput
benefit; its CPU samples identified address publication as a preparation hotspot.
Those samples do not measure the latency saved by removing the writes.

The earlier detailed encoder timeline changes ordinary encoder grouping,
allocates sample storage per command buffer, and logs completion while holding
the mutex used for pipeline submission. More command buffers on the CPU resident-substitution
route can therefore incur asymmetric instrumentation overhead. No explicit
extra commits were found in that source review. The earlier instrumented gap
reduction cannot be treated as recoverable native latency; non-timeline CPU
samples remain useful for locating host work.

## Setup and mechanism

The baseline prototype is `6549a33b` and the address-reuse candidate is
`cc913524`. The runtime change is confined to `expert_tables.zig`. When streaming
address tables are enabled, cache installation publishes an expert's address and cache clear zeros it, so the candidate
reuses those published addresses instead of writing them again during resident
routing preflight. It retains the mask, identity checks and all cache pins.
The change is explicitly opt-in via `DS4_METAL_V41_RESIDENT_GPU_REUSE_ADDR=1`.
Streaming settings must remain stable: enabling address tables after warming a
cache with them disabled requires a cache reload and is outside this experiment.

The bench, agent and server builds passed; installed root executables were not
replaced.

The attribution comparison uses the same candidate executable with GPU resident
routing enabled and refresh interval 8, varying only the address-reuse flag
between 0 and 1. A combined comparison holds address reuse at 1 and varies GPU
resident routing between 0 and 1. The latter compares the combined behavior with
the ordinary CPU resident-substitution route, configured for up to five
replacements, a 0.90 score gap, rank limit 64 and native slot weights. GPU
resident routing relaxes those replacement limits and is a further
approximation, as explained in the preceding report. GPU routing is inactive
during fresh prefill. After the preceding decode, however, the combined arms
may have different expert-cache populations or hotness even after KV restore,
so warm-append differences can reflect the route.

The maintained driver is
[`run.nu`](../2026-11-04-decode-pass/run.nu). For attribution,
`gpu-settings.json` supplies GPU routing and refresh 8; for the combined
comparison, `reuse-settings.json` supplies address reuse and refresh 8. Use a
fresh output directory for each invocation, and run under the existing pressure
guard. For example:

```sh
nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/resident-address-attribution timing --flag DS4_METAL_V41_RESIDENT_GPU_REUSE_ADDR --frontier 32768 --extra-settings speed-bench/experiments/2026-11-04-resident-address-reuse/gpu-settings.json
nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/resident-address-combined timing --flag DS4_METAL_V41_RESIDENT_GPU --frontier 32768 --extra-settings speed-bench/experiments/2026-11-04-resident-address-reuse/reuse-settings.json
```

Raw local evidence is under `.workspaces/hillclimb-20261104/10-resident-address-reuse/`.
The runs used an M5 Max with 128 GiB memory and
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, SSD streaming with a
64 GB expert cache, 4,096 MB expert slabs and 8,192-token prefill chunks.

## Findings

At both the 8K and 32K quality probes, all 256 full-vocabulary logit rows
and all 256 post-evaluation hashes per probe were byte-exact between address
reuse and the old GPU route on the tested fixed histories. Across both probes,
512 of 512 rows and 512 of 512 hashes matched; delta-NLL and KL were zero.
This does not establish equivalence of sampled tasks. The timing comparisons and diagnostic profiles are described below.

### 32K attribution timing

Five profile-free pairs alternated order, with baseline first in pairs 1, 3 and
5. Each run used a fresh 32,768-token prefill and 128 teacher-forced decode
tokens, followed by KV restore, a 512-token append and another 128 decode
tokens. The expert cache survived restore and the OS cache was not flushed.
Phase times exclude loading and snapshot save/restore. Steady decode excludes
the first token; decode windows include it. Seconds for phases are derived from
two-decimal reported throughput. Values are mean ± sample SD.

| Measure | Reuse off | Reuse on |
| --- | --- | --- |
| Fresh steady decode, tokens/s | 34.622 ± 0.184 | 34.686 ± 0.078 |
| Warm steady decode, tokens/s | 32.046 ± 1.195 | 31.730 ± 0.314 |
| Fresh first token, ms | 835.170 ± 32.105 | 820.780 ± 20.037 |
| Warm first token, ms | 34.600 ± 1.239 | 35.240 ± 0.902 |
| Complete fresh decode, s | 4.504 ± 0.029 | 4.483 ± 0.026 |
| Complete warm decode, s | 4.002 ± 0.142 | 4.038 ± 0.038 |
| Inference phases, s | 66.321 ± 1.296 | 66.336 ± 1.937 |
| Wall, s | 68.482 ± 1.322 | 68.533 ± 1.956 |

All samples follow in pair order, reuse off then reuse on.

| Measure | Reuse off | Reuse on |
| --- | --- | --- |
| Fresh steady decode, tokens/s | 34.300, 34.740, 34.710, 34.720, 34.640 | 34.790, 34.630, 34.720, 34.700, 34.590 |
| Warm steady decode, tokens/s | 34.130, 31.630, 31.910, 31.280, 31.280 | 32.260, 31.760, 31.600, 31.470, 31.560 |
| Fresh first token, ms | 818.538, 872.168, 823.544, 864.743, 796.858 | 816.131, 850.685, 796.266, 813.749, 827.070 |
| Warm first token, ms | 33.888, 34.368, 33.285, 36.535, 34.922 | 36.788, 34.436, 35.032, 35.081, 34.862 |
| Complete fresh decode, s | 4.521, 4.528, 4.483, 4.523, 4.463 | 4.468, 4.518, 4.454, 4.474, 4.499 |
| Complete warm decode, s | 3.755, 4.049, 4.014, 4.097, 4.095 | 3.974, 4.033, 4.054, 4.070, 4.060 |
| Inference phases, s | 64.092, 66.470, 66.593, 67.321, 67.130 | 63.448, 65.413, 68.411, 67.206, 67.202 |
| Wall, s | 66.206, 68.662, 68.737, 69.504, 69.299 | 65.637, 67.551, 70.615, 69.369, 69.490 |

The first baseline warm steady sample was 34.13 tokens/s, versus 31.28–31.63
for the remaining baseline samples. Warm steady time changed by +5.80%,
−0.41%, +0.98%, −0.60% and −0.89% in pair order, so the mean is sensitive
to the first pair. Prefill drift also affected the total; the reuse flag is
inactive during prefill. The address-only change preserves route decisions and
cache marking, with no intended cache-policy change. This comparison does not yet show a convincing
attribution speedup.

### 32K combined timing

The combined comparison used the same five-pair procedure. Baseline is ordinary
routing with reuse enabled; candidate enables GPU routing as well. Values are
mean ± sample SD.

| Measure | Ordinary route | GPU route and reuse |
| --- | --- | --- |
| Fresh prefill, s | 49.997 ± 0.961 | 49.561 ± 0.729 |
| Warm append, s | 7.795 ± 0.392 | 7.852 ± 0.157 |
| Fresh steady decode, tokens/s | 34.670 ± 0.092 | 34.712 ± 0.089 |
| Warm steady decode, tokens/s | 32.918 ± 0.518 | 31.664 ± 0.233 |
| Fresh first token, ms | 824.447 ± 20.471 | 869.233 ± 45.215 |
| Warm first token, ms | 35.595 ± 1.357 | 34.075 ± 0.085 |
| Complete fresh decode, s | 4.487 ± 0.025 | 4.528 ± 0.048 |
| Complete warm decode, s | 3.895 ± 0.060 | 4.045 ± 0.029 |
| Inference phases, s | 66.174 ± 1.347 | 65.986 ± 0.879 |
| Wall, s | 68.349 ± 1.348 | 68.170 ± 0.897 |

All samples follow in pair order, ordinary route then GPU route and reuse.

| Measure | Ordinary route | GPU route and reuse |
| --- | --- | --- |
| Fresh prefill, s | 48.537, 50.016, 51.236, 50.035, 50.162 | 48.333, 49.445, 50.000, 49.970, 50.055 |
| Warm append, s | 7.096, 7.913, 7.986, 7.993, 7.985 | 7.573, 7.922, 7.901, 7.940, 7.926 |
| Fresh steady decode, tokens/s | 34.680, 34.700, 34.720, 34.740, 34.510 | 34.570, 34.720, 34.700, 34.770, 34.800 |
| Warm steady decode, tokens/s | 33.820, 32.730, 32.820, 32.720, 32.500 | 31.980, 31.770, 31.440, 31.700, 31.430 |
| Fresh first token, ms | 853.907, 812.173, 799.721, 825.229, 831.207 | 879.104, 928.836, 818.661, 889.711, 829.852 |
| Warm first token, ms | 35.409, 34.416, 36.491, 34.243, 37.415 | 33.964, 34.029, 34.126, 34.071, 34.183 |
| Complete fresh decode, s | 4.515, 4.472, 4.457, 4.480, 4.512 | 4.554, 4.588, 4.479, 4.542, 4.480 |
| Complete warm decode, s | 3.790, 3.914, 3.906, 3.917, 3.946 | 4.005, 4.031, 4.073, 4.040, 4.075 |
| Inference phases, s | 63.939, 66.316, 67.585, 66.425, 66.604 | 64.465, 65.986, 66.452, 66.493, 66.536 |
| Wall, s | 66.126, 68.486, 69.801, 68.620, 68.712 | 66.629, 68.123, 68.652, 68.703, 68.744 |

Fresh decode was close, while every candidate warm steady decode sample was
slower than its paired control: warm steady time rose 5.75%, 3.02%, 4.39%,
3.22% and 3.40% in pair order. Fresh prefill drift cannot be attributed to GPU
routing, which is inactive there; the warm append may reflect the cache state
left by the preceding decode. The combined change does not show an adoption
benefit at 32K.

### 8K combined timing

The same alternating five-pair procedure used an 8,192-token fresh prefill.
Values are mean ± sample SD.

| Measure | Ordinary route | GPU route and reuse |
| --- | --- | --- |
| Fresh prefill, s | 20.072 ± 0.177 | 20.157 ± 0.529 |
| Warm append, s | 7.513 ± 0.304 | 7.472 ± 0.373 |
| Fresh steady decode, tokens/s | 35.248 ± 0.091 | 35.558 ± 0.040 |
| Warm steady decode, tokens/s | 32.672 ± 0.782 | 32.446 ± 0.458 |
| Fresh first token, ms | 823.611 ± 20.010 | 851.038 ± 41.023 |
| Warm first token, ms | 43.100 ± 20.132 | 43.611 ± 21.424 |
| Complete fresh decode, s | 4.426 ± 0.019 | 4.423 ± 0.044 |
| Complete warm decode, s | 3.932 ± 0.097 | 3.959 ± 0.038 |
| Inference phases, s | 35.944 ± 0.371 | 36.011 ± 0.911 |
| Wall, s | 38.031 ± 0.394 | 38.111 ± 0.938 |

All samples follow in pair order, ordinary route then GPU route and reuse.

| Measure | Ordinary route | GPU route and reuse |
| --- | --- | --- |
| Fresh prefill, s | 20.149, 19.803, 19.983, 20.218, 20.207 | 19.330, 20.154, 20.802, 20.235, 20.265 |
| Warm append, s | 6.990, 7.525, 7.615, 7.687, 7.749 | 6.951, 7.200, 7.731, 7.739, 7.741 |
| Fresh steady decode, tokens/s | 35.280, 35.220, 35.230, 35.380, 35.130 | 35.550, 35.550, 35.560, 35.620, 35.510 |
| Warm steady decode, tokens/s | 33.530, 32.760, 31.390, 32.810, 32.870 | 33.160, 32.640, 32.220, 32.040, 32.170 |
| Fresh first token, ms | 850.582, 815.879, 837.311, 813.634, 800.651 | 805.250, 887.369, 854.718, 813.400, 894.454 |
| Warm first token, ms | 33.357, 79.086, 35.374, 33.592, 34.093 | 81.917, 33.484, 34.230, 35.061, 33.364 |
| Complete fresh decode, s | 4.451, 4.421, 4.441, 4.403, 4.415 | 4.378, 4.460, 4.426, 4.379, 4.471 |
| Complete warm decode, s | 3.821, 3.957, 4.082, 3.905, 3.898 | 3.912, 3.925, 3.976, 3.999, 3.981 |
| Inference phases, s | 35.410, 35.706, 36.120, 36.212, 36.270 | 34.570, 35.739, 36.935, 36.352, 36.459 |
| Wall, s | 37.445, 37.800, 38.328, 38.273, 38.311 | 36.653, 37.784, 39.083, 38.453, 38.581 |

Fresh steady decode time fell 0.76%, 0.93%, 0.93%, 0.67% and 1.07%
in pair order; this modest gain was consistent across both orders. Warm
steady time changed +1.12%, +0.37%, −2.58%, +2.40% and +2.18%, so those
results were mixed and noisy, and neither inference-phase nor wall time shows a whole-workload
gain.

### 8K attribution timing

This used the same alternating five-pair procedure with an 8,192-token fresh
prefill. Values are mean ± sample SD.

| Measure | Reuse off | Reuse on |
| --- | --- | --- |
| Fresh prefill, s | 20.185 ± 0.254 | 19.973 ± 0.147 |
| Warm append, s | 7.548 ± 0.162 | 7.510 ± 0.247 |
| Fresh steady decode, tokens/s | 35.498 ± 0.083 | 35.516 ± 0.108 |
| Warm steady decode, tokens/s | 32.198 ± 0.150 | 32.250 ± 0.273 |
| Fresh first token, ms | 813.379 ± 30.788 | 803.306 ± 36.007 |
| Warm first token, ms | 34.317 ± 1.111 | 41.105 ± 15.084 |
| Complete fresh decode, s | 4.391 ± 0.026 | 4.379 ± 0.042 |
| Complete warm decode, s | 3.978 ± 0.019 | 3.979 ± 0.027 |
| Inference phases, s | 36.103 ± 0.192 | 35.841 ± 0.412 |
| Wall, s | 38.194 ± 0.191 | 37.938 ± 0.413 |

All samples follow in pair order, reuse off then reuse on.

| Measure | Reuse off | Reuse on |
| --- | --- | --- |
| Fresh prefill, s | 20.574, 20.069, 19.891, 20.249, 20.141 | 19.880, 19.766, 20.102, 20.015, 20.100 |
| Warm append, s | 7.284, 7.523, 7.593, 7.634, 7.709 | 7.162, 7.347, 7.620, 7.678, 7.745 |
| Fresh steady decode, tokens/s | 35.550, 35.360, 35.560, 35.480, 35.540 | 35.460, 35.550, 35.520, 35.670, 35.380 |
| Warm steady decode, tokens/s | 32.240, 32.360, 32.280, 31.970, 32.140 | 32.510, 32.560, 31.940, 32.070, 32.170 |
| Fresh first token, ms | 789.763, 775.735, 853.149, 825.559, 822.690 | 798.913, 782.107, 858.757, 763.916, 812.837 |
| Warm first token, ms | 32.927, 35.294, 33.305, 35.088, 34.973 | 33.386, 67.934, 33.669, 37.186, 33.352 |
| Complete fresh decode, s | 4.361, 4.367, 4.424, 4.405, 4.397 | 4.381, 4.354, 4.434, 4.324, 4.402 |
| Complete warm decode, s | 3.973, 3.959, 3.967, 4.008, 3.986 | 3.940, 3.969, 4.010, 3.996, 3.981 |
| Inference phases, s | 36.192, 35.918, 35.875, 36.295, 36.233 | 35.362, 35.436, 36.166, 36.014, 36.228 |
| Wall, s | 38.249, 38.010, 37.982, 38.413, 38.317 | 37.437, 37.559, 38.293, 38.083, 38.316 |

Fresh steady time changed +0.25%, −0.53%, +0.11%, −0.53% and +0.45%
in pair order; warm steady time changed −0.83%, −0.61%, +1.06%, −0.31%
and −0.09%. Both were close. The lower mean inference and wall
times were mostly associated with fresh-prefill drift, where the address flag
is inactive, and cannot be attributed to removing the writes.

### Host and command-buffer diagnostics

A separate non-timeline diagnostic used the same GPU route with address reuse
0 and 1 and 512 decode tokens per phase. Each arm had one 3-second CPU
capture at a requested 1 ms interval, starting two seconds after the warm
33,280-token frontier marker. Command-buffer timing, accumulated GPU spans
and selected-readback counters were enabled; the encoder timeline was off.
Disjoint main-thread stack sample counts were 1,454 and 1,427 respectively: selected-readback wait 225 and 203,
end-of-token completion 1,026 and 1,061, resident readiness 59 and 5, routed
dispatch 35 and 34, and other 109 and 124. Within readiness, address-write call
sites fell from 22 samples at the `setAddress` call site and 23 at
`zeroAddress` to zero for each. These nested counts must not be added to readiness; they are samples,
not milliseconds. The targeted preparation work largely disappeared, while
waits remained dominant. This does not measure latency saved.

The first 400 detailed command-buffer rows ended shortly into cold decode,
before the warm marker, and cannot diagnose warm GPU gaps. A relative GPU-start
field had a large negative clock offset and was not interpreted.

Across all guards, pressure remained at level 1, swap ranged from 28.854 to
28.869 GiB (about 16 MiB), and no bad samples were recorded. All guarded
runs succeeded.

## Decision

Retain address reuse as an explicit opt-in preparation cleanup within the
experimental GPU route. The attribution tests did not demonstrate a throughput
gain. The combined route has a modest consistent fresh 8K decode benefit, but
no whole-workload gain there and a repeated warm 32K regression; it is not
ready for preset or installed-binary promotion. The selected bounded follow-up
is to test the existing concurrent shared and routed FFN machinery on this
already-resident path.
