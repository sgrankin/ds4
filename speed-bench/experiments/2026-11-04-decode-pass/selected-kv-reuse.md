# Reusing selected compressed KV between layers

The resident fast presets adopt selected KV reuse as a small simplification of repeated work. Tested fixed-history outputs are exact, and timings are neutral to slightly favorable, though the comparisons do not establish a reliable throughput gain across both workloads. The ordinary runtime remains opt-in, and an explicit `0` disables the preset setting.

## Mechanism and hypothesis

Attention reads a recent raw window and a selection of older compressed key/value (KV) rows. An indexer chooses the compressed rows for a publisher layer; subsequent layers in its group use that selection and its shared compressed cache. With packed FP4 storage and packed F16 attention scratch, the baseline gathers and expands those selected rows into `selected_kv` again at each compressed layer. This scratch is separate from the raw window, which is still staged for each layer.

The candidate in [attention.zig](../../../src/engine/deepseek41/attention.zig) reuses that scratch for eligible nonpublisher layers. At a given token, the scalar path visits layers in order and aborts on failure. A publisher refreshes the selection and gathers its compressed rows before its consumers; within the group the cache owner, compression ratio, selected row count and format remain the same. The guard requires nonempty compressed and selected rows, matching owner and ratio with the previous layer, and neither KV nor index publication at the current layer. Each new token therefore refreshes scratch at its publisher before a consumer can reuse it. For this model, publishers at layers 2, 8, 14, 20, 24, 28, 32 and 36 reduce 38 compressed gathers to eight per token, as confirmed by the diagnostic profile.

The flag `DS4_METAL_V41_REUSE_SELECTED_KV=1` enables the change only for single-rank streaming scalar execution with FP4 compressed storage and packed F16 gather. Quality mode, tracing and projected execution exclude it. Scalar token-by-token prefill can qualify; layer-major and projected batch prefill do not. It introduces no cache fields, allocation or public API. The preceding profile's approximately 1.95 ms of gather duration is not a promised saving: most overlaps query work, and elapsed time depends on the exposed dependency and scheduling.

## Setup

The one-hour loop was 04:20–05:20 Eastern on 2026-11-04 and closed by 04:50. Baseline `81632aee` is runtime-equivalent to the adopted four-change configuration at `a65c4fa6`; candidate `c69e5ef4` adds this flag. The comparisons use the same candidate binary with the flag disabled or enabled. The [session driver](run.nu) supplies the common settings and clears inherited experimental settings.

The profile-off comparison used five alternating pairs at each of 32K and 8K, with 128 teacher-forced decode steps, then KV restore, a 512-token warm append and another 128 steps. Steady decode excludes only the first step, which does not prove initialization has finished; complete decode and first-step timings are retained separately. Phase timings exclude model loading and snapshot save/restore; the expert cache survives restore and the OS cache was not flushed. A separate diagnostic profile is not a latency comparison. Fixed-history quality checks cover aligned and unaligned frontiers; sampled practical task quality is not established by them.

The M5 Max has 128 GiB memory; the model is `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, using SSD streaming and a requested 64 GB expert cache. Raw logs and generated outputs remain local under `.workspaces/overnight-decode-20261104/selected-kv-reuse/`.

## Measurements and decision

For each starting frontier of 8,192, 32,768 and unaligned 8,193 tokens, the fixed-history check covered 128 fresh rows and 128 after a 512-token warm append. All 256 full-vocabulary rows and 256 post-evaluation hashes matched in each check: 768 of each in total, with zero delta-NLL and KL. This does not establish parity on other contexts or sampled tasks. The initial guard invocation failed a sandboxed `sysctl` before model launch; a second guarded invocation completed successfully. All completed timing, quality and profile runs passed the pressure guard at level 1, with swap stable at 24.963 GiB.

At 32K, fresh steady decode averaged 34.206 ± 0.463 versus 34.466 ± 0.525 tokens/s, and warm steady decode 33.750 ± 0.297 versus 33.824 ± 0.409 tokens/s (mean ± sample SD). The candidate was faster in all five fresh pairs and four warm pairs, but the absolute differences were small. Baseline ran first in pairs 1, 3 and 5; candidate ran first in pairs 2 and 4.

| Pair | Fresh steady, tokens/s | Warm steady, tokens/s | Full fresh 128 steps, s | Full warm 128 steps, s | First fresh step, ms | First warm step, ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 34.44 → 34.74 | 33.22 → 33.11 | 4.583 → 4.560 | 3.859 → 3.871 | 895.8 → 905.3 | 36.1 → 35.4 |
| 2 | 34.44 → 34.72 | 33.88 → 33.94 | 4.557 → 4.521 | 3.783 → 3.775 | 869.4 → 863.7 | 34.4 → 33.3 |
| 3 | 33.39 → 33.53 | 33.87 → 33.99 | 4.680 → 4.666 | 3.786 → 3.770 | 875.9 → 878.8 | 36.7 → 33.8 |
| 4 | 34.28 → 34.70 | 33.92 → 33.93 | 4.558 → 4.593 | 3.778 → 3.777 | 853.5 → 932.1 | 33.8 → 33.3 |
| 5 | 34.48 → 34.64 | 33.86 → 34.15 | 4.565 → 4.550 | 3.787 → 3.755 | 881.4 → 883.1 | 36.0 → 35.8 |

Each cell shows baseline → candidate. Full fresh decode averaged 4.589 ± 0.052 versus 4.578 ± 0.056 s and full warm decode 3.798 ± 0.034 versus 3.789 ± 0.046 s. First fresh steps averaged 875.2 ± 15.5 versus 892.6 ± 26.6 ms and first warm steps 35.4 ± 1.2 versus 34.3 ± 1.2 ms. Decode durations are approximate, derived from throughput rounded to two decimals.

Fresh prefill averaged 48.497 ± 1.781 versus 49.437 ± 0.224 s, and total inference phases 64.178 ± 1.560 versus 65.076 ± 0.241 s. The baseline's first fresh prefill, 45.31 s, is retained; the other baseline samples were about 49.25–49.34 s. The candidate does not change the layer-major prefill path, so these values do not establish a prefill effect.

At 8K, fresh steady decode averaged 34.312 ± 1.364 versus 34.600 ± 0.842 tokens/s, and warm steady decode 34.636 ± 0.710 versus 34.550 ± 0.334 tokens/s. Three fresh and three warm pairs favored the candidate. The order alternated in the same pattern as 32K; differences went in both directions.

| Pair | Fresh steady, tokens/s | Warm steady, tokens/s | Full fresh 128 steps, s | Full warm 128 steps, s | First fresh step, ms | First warm step, ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 35.26 → 34.15 | 35.40 → 34.53 | 4.440 → 4.586 | 3.620 → 3.712 | 837.8 → 867.6 | 32.6 → 34.1 |
| 2 | 32.27 → 33.91 | 34.27 → 34.34 | 4.807 → 4.623 | 3.739 → 3.732 | 872.0 → 877.2 | 33.2 → 33.6 |
| 3 | 35.21 → 33.91 | 35.30 → 34.12 | 4.485 → 4.616 | 3.630 → 3.755 | 877.9 → 870.4 | 32.6 → 32.8 |
| 4 | 33.53 → 35.58 | 33.72 → 34.88 | 4.655 → 4.431 | 3.798 → 3.675 | 866.1 → 860.3 | 32.7 → 33.4 |
| 5 | 35.29 → 35.45 | 34.49 → 34.88 | 4.448 → 4.428 | 3.717 → 3.674 | 847.8 → 845.1 | 34.1 → 32.8 |

Full fresh decode averaged 4.567 ± 0.160 versus 4.537 ± 0.099 s and warm decode 3.701 ± 0.075 versus 3.710 ± 0.035 s. First fresh steps averaged 860.3 ± 16.9 versus 864.1 ± 12.2 ms and first warm steps 33.0 ± 0.7 versus 33.4 ± 0.5 ms. Fresh prefill averaged 19.856 ± 0.323 versus 19.495 ± 0.063 s and inference phase totals 35.156 ± 0.526 versus 34.910 ± 0.472 s. The results do not establish a reliable throughput gain across both workloads.

## Diagnostic and decision

In the separate instrumented 32K pair, each 128-step fresh and warm decode phase contained 4,864 baseline gathers and 1,024 candidate gathers: 38 and eight per token respectively. Both arms used 80 command buffers per token; instrumented encoder counts fell from 932.055 to 902.055. Profiling forces separate encoders, so these are not production encoder counts. Summed gather durations fell from 1.82645 to 0.03852 ms per token fresh and 1.83992 to 0.03844 ms warm. Those durations overlap query work and are not fully recoverable latency. The token envelope was 33.6007 versus 33.4301 ms fresh and 33.4642 versus 33.5499 ms warm. This single intrusive pair confirms the dispatch change, not a comparative latency gain.

For a full 512-row selection, eliminating 30 gathers avoids a logical 4.21875 MiB of packed reads and 15 MiB of F16 scratch writes per token. These are derived operation sizes, not measured DRAM or SSD traffic; allocation size is unchanged.

The change is adopted for its simple reuse and exact tested output, with neutral to slightly favorable timing rather than a firm speed claim. The 32K fresh pairs all favor the candidate, but the 8K results are mixed. These measurements cover one model and machine and do not establish practical task quality.

To reproduce the 32K timing comparison from the candidate revision using the existing guarded driver:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/overnight-decode-20261104/selected-kv-reuse/timing32k-guard -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/overnight-decode-20261104/selected-kv-reuse/timing32k timing --flag DS4_METAL_V41_REUSE_SELECTED_KV --control 0 --candidate 1
```
