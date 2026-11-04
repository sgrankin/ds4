# Half-precision gather for packed compressed KV

The resident-fast and persistent-fast presets now use half-precision gather for packed FP4 main KV. The existing path reduces intermediate traffic and enables fused staging, while measured latency was neutral to slightly favorable across the tested workloads. The fixed-history checks produced identical logits, but do not establish parity for every input or generated task.

## Question and mechanism

Decode attention combines a recent raw KV window with selected compressed rows. With packed FP4 main KV, the control gathers and expands the selected rows into float32 scratch; attention then converts them to half precision while staging them alongside the raw window. The candidate gathers directly into half-precision scratch. This changes staging eligibility and scheduling as well as the conversion: the half input can use fused staging and padding when their other conditions hold. Both the selected-row scratch and the attention staging scratch remain allocated. Stored KV and checkpoint representations do not change. See [`Attention.packedGather` and `run`](../../../src/engine/deepseek41/attention.zig) and [`stage` and `attend`](../../../src/metal/attention_decode.zig).

The alternate packed32 attention reduction also requires a separately deferred inverse RoPE operation. Native V4.1 does not arm that hook, so this flag retains vector attention and the ordinary or heads-transform reduction, including scalar prefill tails. The absence of an armed heads transform alone would not be enough to select packed32.

## Setup

The baseline was revision `8971dfbd`, using the existing binary from the preceding query-kernel experiment on an M5 Max with 128 GiB memory. The driver was committed as `0ed039be`; no kernel was added for this comparison. The only changed setting was `DS4_METAL_V41_PACKED_F16_GATHER=0` versus `1`, with packed FP4 main and index KV. Common settings included short Q8 query and streaming RoPE table, shared route overlap with a 2,000 µs poll, route cap 5 and gap 0.90, 4 GiB expert slabs, a requested 64 GB cache budget, and 8,192-token prefill chunks. The model was `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf` with SSD streaming. All completed guarded runs exited successfully; reported pressure was 1 and swap stayed at 24.944 GiB.

The [driver](run.nu) uses the same Promessi Sposi teacher-forced workload for both arms. The profile-free comparison uses five alternating pairs at each of two fresh frontiers: 8,192 and 32,768 tokens. Each run performs 128 decode steps, restores KV, appends 512 tokens, and performs another 128 decode steps. The expert cache survives restore; the OS cache was not flushed. Steady decode excludes the first step. Phase times exclude model loading and snapshot save and restore. The driver records exact arguments and environment with local raw results under `.workspaces/overnight-decode-20261104/f16-gather/`.

## Quality

The focused `--main-fp4` check passed. The 8K, 32K and scalar-tail 8,193 quality runs each matched all 256 full-vocabulary teacher-logit rows byte for byte and all 256 post-evaluation hashes: 768 of each in total. Mean delta-NLL and KL were zero and there were no top-1 mismatches. These are fixed-history numerical checks, not practical task tests or proof of parity for other contexts, hardware, or sampled continuations.

## Timing

At 32K, decode was neutral: fresh steady throughput was 30.972 ± 0.257 versus 30.966 ± 0.079 tokens/s, and warm steady throughput was 31.320 ± 0.248 versus 31.332 ± 0.124 tokens/s (mean ± sample SD). Complete fresh 128-step decode took 5.002 ± 0.026 versus 5.004 ± 0.020 s; warm decode took 4.093 ± 0.031 versus 4.090 ± 0.017 s. First steps averaged 900.7 ± 37.3 versus 902.1 ± 20.7 ms fresh, and 37.6 ± 2.9 versus 36.7 ± 1.2 ms warm.

Fresh prefill took 47.854 ± 0.668 versus 47.564 ± 0.600 s; the warm 512-token append took 7.559 ± 0.276 versus 7.690 ± 0.044 s. Total measured inference phases took 64.508 ± 0.459 versus 64.348 ± 0.605 s. Seconds are derived from throughput rounded to two decimal places. The first pair had the largest phase-time difference; paired differences change sign, so these samples do not establish a speed gain or a reliable order effect.

| Pair (first arm) | Fresh steady, baseline / candidate tokens/s | Warm steady, baseline / candidate tokens/s | Full fresh decode, baseline / candidate s | Full warm decode, baseline / candidate s | Phase total, baseline / candidate s |
|---|---:|---:|---:|---:|---:|
| 1 (baseline) | 31.38 / 30.95 | 31.67 / 31.18 | 5.010 / 4.984 | 4.047 / 4.110 | 65.079 / 63.379 |
| 2 (candidate) | 31.02 / 30.89 | 31.30 / 31.29 | 4.969 / 5.006 | 4.093 / 4.097 | 64.286 / 64.894 |
| 3 (baseline) | 30.95 / 30.95 | 31.45 / 31.47 | 4.996 / 4.992 | 4.080 / 4.071 | 63.904 / 64.418 |
| 4 (candidate) | 30.73 / 31.10 | 31.07 / 31.45 | 5.039 / 5.000 | 4.124 / 4.074 | 64.448 / 64.241 |
| 5 (baseline) | 30.78 / 30.94 | 31.11 / 31.27 | 4.994 / 5.035 | 4.118 / 4.097 | 64.823 / 64.807 |

At 8K the same five-pair procedure gave fresh steady throughput of 31.504 ± 0.386 versus 31.528 ± 0.313 tokens/s and warm steady throughput of 31.934 ± 0.293 versus 32.096 ± 0.362 tokens/s. Full fresh decode took 4.886 ± 0.063 versus 4.912 ± 0.045 s, and warm decode took 4.014 ± 0.035 versus 4.002 ± 0.053 s. First steps averaged 854.0 ± 19.0 versus 883.9 ± 21.3 ms fresh and 36.7 ± 1.7 versus 45.1 ± 17.9 ms warm; one candidate warm first step took 77.0 ms. Fresh prefill took 19.373 ± 0.165 versus 19.303 ± 0.160 s, the warm append 7.283 ± 0.147 versus 7.234 ± 0.176 s, and total inference phases 35.555 ± 0.202 versus 35.452 ± 0.350 s.

| Pair (first arm) | Fresh steady, baseline / candidate tokens/s | Warm steady, baseline / candidate tokens/s | Full fresh decode, baseline / candidate s | Full warm decode, baseline / candidate s | Phase total, baseline / candidate s |
|---|---:|---:|---:|---:|---:|
| 1 (baseline) | 32.00 / 31.88 | 32.36 / 32.63 | 4.819 / 4.884 | 3.963 / 3.929 | 35.401 / 34.974 |
| 2 (candidate) | 31.70 / 31.02 | 32.01 / 32.31 | 4.858 / 4.986 | 4.006 / 3.969 | 35.371 / 35.200 |
| 3 (baseline) | 30.96 / 31.61 | 31.98 / 31.87 | 4.988 / 4.923 | 4.006 / 4.023 | 35.659 / 35.617 |
| 4 (candidate) | 31.48 / 31.57 | 31.70 / 31.89 | 4.871 / 4.882 | 4.042 / 4.060 | 35.490 / 35.659 |
| 5 (baseline) | 31.38 / 31.56 | 31.62 / 31.78 | 4.893 / 4.885 | 4.052 / 4.031 | 35.856 / 35.809 |

Across these workloads latency is neutral to slightly favorable overall, with individual phase changes in both directions. The presets adopt this existing small path for its reduced intermediate traffic and staging work, with no meaningful measured regression and exact tested logits. Explicit `DS4_METAL_V41_PACKED_F16_GATHER=0` restores the float32 gather. Launcher tests passed. This result does not conclude the ongoing overnight session.

## Profile diagnostic

One intrusive timeline pair confirmed the FP4 half gather and fused staging paths. Encoders per token fell from 967.756 to 892.055; command buffers stayed at 61. Warm summed conversion time fell from 1.788 to 0.073 ms, while candidate fused staging took 0.819 ms and FP4 gather rose from 0.831 to 1.049 ms. Source extents imply 38 MiB less logical intermediate traffic per token; this is not a physical memory-traffic measurement, and both scratch allocations remain.

The instrumented token envelope instead increased from 35.102 to 36.673 ms warm and from 35.479 to 36.822 ms fresh. This one profiled pair conflicts with the neutral profile-off comparison; the component sums do not establish a speedup or explain the envelope difference. The adoption rests on the five-pair profile-off measurements and tested numerical behavior, not the instrumented timing.
