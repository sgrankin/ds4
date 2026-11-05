# SIMD resident expert selection

Retain SIMD as an opt-in within the experimental GPU routing package. The 32-lane
search preserves scalar repair's tested choices and reduces batched cost when
experts are missing. Model timings are mixed: fresh 32K decode improves in every
attribution pair, while complete fresh 8K decode slows in four of five. The
combined package improves 8K decode in every pair; total-time results vary.

## Question and implementation

After removing the per-layer host routing handoff, the remaining GPU resident repair searches candidate experts on one lane. The earlier intrusive profile attributed 0.590 ms summed and 0.314 ms exclusive per fresh token to resident repair, and 0.541/0.293 ms per warm token. These spans identify a concrete target, not a causal critical-path saving. The [address reuse report](../2026-11-04-resident-address-reuse/README.md) also cautions that detailed encoder timing changes batching.

At runtime revision `e5d12861`, `DS4_METAL_V41_RESIDENT_GPU_SIMD=1` selects the candidate; the scalar search remains the default. Both repair six native selected IDs in slot order, retaining native mixture weights. For each missing ID, the SIMD group searches resident experts in parallel, excludes IDs already selected, and reduces to the highest score with lower expert ID breaking ties. It preserves the scalar rule for a first candidate with NaN and accumulates trace mass in scalar order. See [the kernels](../../../src/metal/dsv41.metal) and [dispatch](../../../src/metal/v41_dispatch.zig). GPU resident policy, refresh interval 16, address reuse, parallel FFN and the 64 GB expert cache were held fixed. The GPU resident policy is already approximate relative to native/CPU routing; SIMD preserves that policy. This changes kernel execution; neither presets nor installed binaries are promoted here.

## Correctness

The hardware `--resident-router` fixture passed for scalar and SIMD against an ordered CPU reference. It checks exact selected IDs and scalar-versus-SIMD trace, retained weights and probabilities, uniqueness and residency, with and without tracing. Counts include 6, 31, 33, 256, 257, 384 and 1024; cases include hits and misses, exactly six residents, ties, infinities, NaNs, and tiny or zero selected mass. The CPU trace mass comparison uses a tolerance; scalar and SIMD traces are compared exactly. See [the fixture](../../../tests/test_deepseek41_metal.c).

## Isolated GPU cost

The fixture measured 64 independent selected-ID buffers per command buffer, restoring the native IDs before each batch so repeated repairs did not turn into hits. About half of experts were resident. After one warmup pair, five measured pairs alternated order. The table gives mean ± sample standard deviation in microseconds for GPU command-buffer span divided by 64; this is **batch throughput per dispatch**, not single-layer latency. GPU timestamps are rounded to 1 µs per batch. Raw output and every sample, including warmups, remain locally in `.workspaces/hillclimb-20261104/15-resident-simd/microbench/`.

| Experts | Missing native IDs | Scalar, µs/dispatch | SIMD, µs/dispatch |
| --- | --- | ---: | ---: |
| 256 | 0 | 7.078 ± 0.097 | 7.816 ± 0.106 |
| 256 | 3 | 58.706 ± 4.254 | 8.528 ± 1.067 |
| 256 | 6 | 104.722 ± 0.562 | 10.084 ± 1.001 |
| 384 | 0 | 3.425 ± 0.039 | 3.531 ± 0.141 |
| 384 | 3 | 84.081 ± 0.949 | 10.503 ± 1.043 |
| 384 | 6 | 156.622 ± 0.807 | 13.338 ± 1.108 |

At 256 experts, SIMD was slower in every all-hit pair; the 384-expert all-hit difference was noisy. Case order was fixed, and the all-hit results differ notably between the two expert-count batches, so they are not evidence of scaling with expert count. These independent dispatches can overlap differently from a model's dependent layers. The retained `samples.csv` and `summary.json` also include host encode and wall timing; the latter includes submission, GPU wait and diagnostic logging. Reproduce with [microbench.nu](microbench.nu) under the existing pressure guard and the hardware fixture binary.

## Full-model qualification

Model runs use an M5 Max with 128 GiB RAM and `gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, context 40960, prefill chunk 8192 and 64 GB expert cache. The model comparison uses the existing [decode driver](../2026-11-04-decode-pass/run.nu), with 128 tokens per phase and five alternating pairs. Attribution compares scalar and SIMD GPU routing using [resident settings](resident-settings.json); the combined comparison changes ordinary CPU routing to SIMD GPU routing using [SIMD settings](simd-settings.json). Fixed-history quality checks covered 128 full-vocabulary rows at each frontier: 8192, 8704, 32768 and 33280. All 512 rows and post-evaluation hashes matched byte for byte; delta-NLL and KL were zero at every position. Diagnostic pipeline creation logs confirm the intended scalar and SIMD kernels were selected. This diagnostic profiling is excluded from final timing. These results establish no observed additional drift relative to GPU scalar repair; they do not erase the GPU policy’s differences from CPU routing. The benchmark binary SHA-256 was `bcd5281897d9f08e055cf38dfb1cd1462e158ff98d594b0a4c7e4f357cb8b52c`.

### 32K GPU scalar versus SIMD

The profile-free attribution run used five alternating pairs: scalar first in pairs 1, 3 and 5, SIMD first in pairs 2 and 4. Each run included a fresh 32K prefill and 128 teacher decode tokens, KV restore, a 512-token warm append and another 128 decode tokens. The expert cache survived restore; the OS cache was not flushed. Steady decode excludes the first token. Phase totals exclude model loading and snapshot save/restore; seconds are derived from two-decimal throughput values.

| Metric | Scalar mean ± SD | SIMD mean ± SD | Scalar samples, pair order | SIMD samples, pair order |
| --- | ---: | ---: | --- | --- |
| Fresh steady decode, tokens/s | 34.886 ± 1.072 | 35.964 ± 0.203 | 33.25, 35.55, 35.63, 34.33, 35.67 | 36.11, 35.85, 35.66, 36.12, 36.08 |
| Warm steady decode, tokens/s | 33.300 ± 0.526 | 34.134 ± 1.625 | 33.96, 33.62, 33.28, 33.05, 32.59 | 35.39, 33.67, 36.04, 33.65, 31.92 |
| Fresh complete decode, s | 4.463 ± 0.115 | 4.355 ± 0.034 | 4.641, 4.391, 4.382, 4.518, 4.384 | 4.324, 4.339, 4.373, 4.406, 4.335 |
| Warm complete decode, s | 3.850 ± 0.060 | 3.798 ± 0.142 | 3.774, 3.814, 3.852, 3.878, 3.931 | 3.621, 3.807, 3.743, 3.808, 4.013 |
| Total inference phases, s | 65.610 ± 0.857 | 66.368 ± 4.226 | 64.580, 65.047, 65.507, 66.692, 66.225 | 61.722, 63.906, 72.435, 65.097, 68.682 |

Fresh steady decode favored SIMD in all five pairs and warm in four. Pair 3's SIMD warm append took 14.504 s versus 7.697 s, and its first warm decode step took 218.749 ms versus 35.194 ms. This retained outlier contributes to the wide total variation. First fresh steps averaged 819.992 ± 1.664 ms scalar and 824.415 ± 37.334 ms SIMD; first warm steps averaged 34.893 ± 1.069 and 70.979 ± 82.612 ms. The first warm token at position 33280 is a refresh step, so its 218 ms is not a direct measure of repair-kernel execution. The data do not establish an overall inference-time improvement or the cause of the append outlier.

The driver was invoked under the pressure guard with:

```sh
nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/hillclimb-20261104/15-resident-simd/timing32k timing --flag DS4_METAL_V41_RESIDENT_GPU_SIMD --frontier 32768 --extra-settings speed-bench/experiments/2026-11-04-resident-simd/resident-settings.json
```

The full summary, phase samples, guard observations and logs are retained locally under `.workspaces/hillclimb-20261104/15-resident-simd/`.

### 32K CPU routing versus SIMD GPU routing

This separate five-pair batch compares the ordinary CPU route against the complete experimental GPU routing package with SIMD repair, keeping the other listed settings fixed. Its controls belong to this batch and are not pooled with the attribution controls. Scope, token counts and alternating order match the attribution run.

| Metric | CPU mean ± SD | GPU SIMD mean ± SD | CPU samples, pair order | GPU SIMD samples, pair order |
| --- | ---: | ---: | --- | --- |
| Fresh steady decode, tokens/s | 34.634 ± 0.111 | 35.366 ± 0.905 | 34.68, 34.59, 34.46, 34.72, 34.72 | 34.70, 34.11, 35.89, 36.06, 36.07 |
| Warm steady decode, tokens/s | 32.994 ± 1.348 | 33.688 ± 0.861 | 35.09, 33.52, 32.45, 32.22, 31.69 | 35.11, 33.39, 33.47, 33.68, 32.79 |
| Fresh complete decode, s | 4.519 ± 0.064 | 4.469 ± 0.151 | 4.476, 4.621, 4.544, 4.471, 4.482 | 4.526, 4.704, 4.406, 4.382, 4.324 |
| Warm complete decode, s | 3.967 ± 0.237 | 3.981 ± 0.163 | 3.682, 3.824, 4.311, 3.978, 4.042 | 4.125, 3.886, 4.180, 3.805, 3.907 |
| Total inference phases, s | 65.644 ± 4.631 | 67.483 ± 3.666 | 59.224, 64.977, 72.247, 65.442, 66.332 | 69.003, 63.881, 73.091, 65.087, 66.354 |

Both fresh and warm steady throughput favored GPU SIMD in four of five pairs; each complete decode favored it in three. Total inference phases do not establish a win. Candidate pair 1's warm append took 15.006 s and first warm decode step 506.936 ms. Pair 3 had slow warm appends on both arms, 14.832/14.700 s, and first warm steps of 396.877/386.674 ms. These first tokens are refresh steps; the cause of the slow appends is unestablished. All samples are retained. The pressure guard completed normally; a prior sandbox preflight failed before model launch and remains in `combined32k-guard`, while `combined32k-guard-retry` records the actual run.

The driver was invoked under the pressure guard with:

```sh
nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/hillclimb-20261104/15-resident-simd/combined32k timing --flag DS4_METAL_V41_RESIDENT_GPU --frontier 32768 --extra-settings speed-bench/experiments/2026-11-04-resident-simd/simd-settings.json
```

### 8K GPU scalar versus SIMD

The 8K attribution uses the same five-pair procedure and 128-token decode phases, with a fresh 8192-token frontier and 512-token warm append. The controls are specific to this batch.

| Metric | Scalar mean ± SD | SIMD mean ± SD | Scalar samples, pair order | SIMD samples, pair order |
| --- | ---: | ---: | --- | --- |
| Fresh steady decode, tokens/s | 36.576 ± 0.074 | 36.406 ± 0.532 | 36.55, 36.58, 36.67, 36.61, 36.47 | 36.96, 36.46, 35.74, 36.00, 36.87 |
| Warm steady decode, tokens/s | 34.068 ± 0.563 | 34.530 ± 0.813 | 34.85, 34.37, 33.99, 33.73, 33.40 | 35.53, 35.30, 33.97, 33.89, 33.96 |
| Fresh complete decode, s | 4.298 ± 0.029 | 4.366 ± 0.088 | 4.270, 4.346, 4.290, 4.285, 4.300 | 4.274, 4.454, 4.394, 4.437, 4.270 |
| Warm complete decode, s | 3.774 ± 0.075 | 3.720 ± 0.094 | 3.677, 3.729, 3.770, 3.857, 3.837 | 3.608, 3.631, 3.810, 3.780, 3.774 |
| Total inference phases, s | 35.666 ± 0.508 | 35.538 ± 0.864 | 35.190, 35.039, 36.055, 35.965, 36.083 | 34.303, 34.997, 36.285, 36.255, 35.849 |

Fresh steady throughput favored SIMD in two pairs, warm in four. Complete fresh decode was slower in four pairs; the first fresh step was longer in all five, averaging 825.736 ± 29.975 ms scalar versus 876.482 ± 61.691 ms SIMD. The first position 8192 uses the refresh path; diagnostic pipeline creation occurs between steps 0 and 1, so the first-step difference is not a direct SIMD-search measurement. Its cause is unestablished. Complete warm decode was faster in four pairs. Total inference phases are inconclusive. These results do not support a universal decode improvement.

Use the attribution driver command above with `--frontier 8192` and a distinct output and guard directory.

### 8K CPU routing versus SIMD GPU routing

This five-pair batch compares the complete GPU package with CPU routing under the same 8K phase scope. Its controls are separate from the attribution batch.

| Metric | CPU mean ± SD | GPU SIMD mean ± SD | CPU samples, pair order | GPU SIMD samples, pair order |
| --- | ---: | ---: | --- | --- |
| Fresh steady decode, tokens/s | 35.228 ± 0.165 | 36.954 ± 0.033 | 35.20, 35.37, 35.40, 35.18, 34.99 | 36.95, 36.95, 37.01, 36.93, 36.93 |
| Warm steady decode, tokens/s | 33.150 ± 0.399 | 34.576 ± 0.721 | 33.48, 33.54, 33.20, 32.96, 32.57 | 35.53, 35.09, 34.32, 34.20, 33.74 |
| Fresh complete decode, s | 4.430 ± 0.037 | 4.264 ± 0.017 | 4.405, 4.451, 4.382, 4.474, 4.440 | 4.267, 4.250, 4.244, 4.277, 4.282 |
| Warm complete decode, s | 3.865 ± 0.047 | 3.709 ± 0.076 | 3.827, 3.820, 3.859, 3.887, 3.934 | 3.609, 3.654, 3.734, 3.750, 3.797 |
| Total inference phases, s | 35.652 ± 0.459 | 35.151 ± 0.768 | 35.367, 35.111, 35.526, 36.103, 36.152 | 34.272, 34.363, 35.573, 35.695, 35.852 |

Both steady and complete decode favored the GPU package in all five pairs. Total inference favored it in four; paired change was −1.412 ± 1.241%, with prefill variation affecting the total. First fresh steps averaged 825.027 ± 34.287 ms CPU versus 827.396 ± 14.145 ms GPU; first warm steps averaged 33.720 ± 0.769 versus 34.475 ± 1.279 ms. Use the combined driver command above with `--frontier 8192` and distinct output and guard directories.

## Decision and limits

Retain the SIMD implementation as an opt-in: its tested choices are exact relative to scalar GPU repair, its missing-expert fixture cost falls, and 32K attribution shows faster fresh steady decode. The 8K attribution is mixed, including slower complete fresh decode in four pairs, and neither 32K batch established a total inference-time win. The combined 8K result concerns the full experimental GPU policy and does not attribute its gain to SIMD alone. The GPU package remains experimental; no preset or installed binary is promoted. No new dependency profile was taken in this slice; an updated profile is the next selected question.

All eight completed model and fixture guard runs reported pressure level 1 and zero adverse samples; swap ranged from 28.8298 to 28.9079 GiB. One sandbox preflight failed before model launch and produced no model samples; its evidence is retained alongside the successful retry. Raw logs, complete phase samples and guard records remain local and ignored under `.workspaces/hillclimb-20261104/15-resident-simd/`.
