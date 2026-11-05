# Matrix projection for temporary indexer queries

Retain the matrix projection as a useful opt-in prefill approximation and include it in the next combined package qualification. The measured prefill improved on both tested frontiers, with nonzero numerical drift and a limited practical check. This does not promote the option to the fast preset, installed binary or main configuration.

## Question and mechanism

Can a matrix reduction accelerate the temporary indexer query projection during prefill at a useful numerical cost? The baseline projects each row with the existing F16 row-exact path. The candidate opts this query alone into batched matrix multiplication. The indexer uses the resulting queries to score and select compressed keys for attention, so a changed query can change selected rows and later model state.

The opt-in `DS4_METAL_V41_PREFILL_INDEX_Q_MPP`, set to a nonzero integer, applies only to F16 weights with dimensions 1280 → 4096, at least 32 rows aligned to 32, TensorOps available, one tensor-parallel rank, non-quality mode and no exact-dense-rows override. Otherwise the existing projection remains in use. The ordinary F16 matrix backend prefers a direct-RHS MPP kernel with F32 inputs; its generic matrix fallback rounds those inputs to half. See [`projectIndexQuery`](../../../src/engine/deepseek41/batch_attention.zig) and [`halfMode`](../../../src/metal/dense_matmul.zig).

The published-key projections remain row-exact: bulk publication and suffix replay must preserve their key projection contract across sweep sizes. That does not imply globally unchanged KV state, because changed selections can influence subsequent computation and state.

## Setup and numerical checks

Experiment 18 completed early within the November 4, 2026, 21:43–22:43 Eastern window on an Apple M5 Max with 128 GiB RAM, using `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`. Runtime revision was `909faab5`; the fixture was built at `c82b75cc`. Both timing arms used the same benchmark binary, SHA-256 `a8fad44806932fad5e041693b7001b913e147751e08e1790a58b3b067f7fb5e9`. The fixture binary SHA-256 was `783702d9f6eeda88867bf021c64556ccf7fb555b73cc8c8ba7d64ae4dff0b9e8`; the rebuilt agent binary for the practical check was `d06a2f32ee4304bb48605f38fa5139fc322e6cc98c8ee18cc80cb7a5d2c8eaa0`.

The usual driver used a 40,960-token context, 8,192-token prefill chunks and a 64 GB expert cache. The quality probe evaluated 128 teacher tokens per phase, with a 512-token append between phases. KV restore retains experts; the procedure did not flush the OS file cache. Shared experimental settings and raw evidence are retained locally under `.workspaces/hillclimb-20261104/18-prefill-f16-projections/`.

All seven guarded jobs completed successfully. Memory pressure stayed at level 1,
swap remained at 28.908 GiB and no adverse pressure samples were recorded.

The bounded fixture passed finite-output and sentinel checks and 32 CPU dot comparisons. Before BF16 output rounding, maximum absolute matrix-versus-row differences were `4.77e-6` at 32 rows and `8.11e-6` at 8,192 rows. After rounding, 36 of 131,072 and 8,046 of 33,554,432 values differed, respectively; maximum absolute differences were `0.015625` and `0.03125`. Its BF16 inputs are exactly representable as F16, so it does not exercise the additional input rounding or range issues possible in the generic matrix fallback; the profile below confirms direct-RHS dispatch in this run. These small fixture differences establish neither identical full-model behavior nor a failure by themselves.

Full-vocabulary, temperature-one teacher logits on identical histories show nonzero model drift. Each frontier has 128 predictions. ΔNLL is candidate minus baseline negative log likelihood in nats per teacher token; positive values mean the candidate assigned less likelihood to those tokens. KL is `KL(baseline || candidate)`. Percentiles and maxima describe individual predictions, and top-1 counts show changed highest-probability tokens.

| Frontier | ΔNLL mean / p95 / max | KL mean / p95 / max | Top-1 changed |
| ---: | ---: | ---: | ---: |
| 8,192 | -0.00134 / 0.18810 / 1.17616 | 0.01459 / 0.06352 / 0.50304 | 5 / 128 |
| 8,704 | 0.00555 / 0.16840 / 0.68153 | 0.01655 / 0.06128 / 0.82135 | 5 / 128 |
| 32,768 | 0.03217 / 0.40604 / 1.75620 | 0.02506 / 0.09399 / 0.35769 | 6 / 128 |
| 33,280 | -0.05058 / 0.19849 / 0.50856 | 0.01466 / 0.04535 / 0.30771 | 6 / 128 |

These fixed-history metrics describe distribution differences, not task accuracy or general improvement. They do not describe diverging sampled histories, and neither drift metric is an automatic rejection criterion. Practical task results remain necessary.

### Reproduction

The [driver](../2026-11-04-decode-pass/run.nu) defines the common baseline: FP4 compressed KV and index, SIMD indexing, selected-tile and resident-route settings, shared-route overlap, streaming RoPE and expert slabs, short Q8 queries, packed F16 gather, selected-KV reuse, decode layer flush, and RB4 attention with unit scale. The linked source records their exact values. It clears inherited `DS4` environment variables and alternates timing arms. The extra settings applied to both arms were:

```json
{
  "DS4_METAL_V41_RESIDENT_GPU": "1",
  "DS4_METAL_V41_RESIDENT_GPU_SIMD": "1",
  "DS4_METAL_V41_RESIDENT_GPU_REUSE_ADDR": "1",
  "DS4_METAL_V41_RESIDENT_GPU_PARALLEL": "1",
  "DS4_METAL_V41_RESIDENT_GPU_REFRESH": "16",
  "DS4_METAL_V41_RESIDENT_GPU_FUSED": "1",
  "DS4_METAL_V41_PREFILL_ROPE_BF16": "1"
}
```

Save this JSON as `settings.json` in a new local workspace. With the recorded binaries and model in place, run each command sequentially under the pressure guard, using a fresh output directory for each combination of stage and frontier:

```sh
python3 speed-bench/ideas/pressure_run.py GUARD_DIR -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu OUTPUT_DIR STAGE --flag DS4_METAL_V41_PREFILL_INDEX_Q_MPP --frontier FRONTIER --extra-settings SETTINGS_JSON
```

Use `quality` and `timing` for `STAGE`, and `8192` and `32768` for `FRONTIER`. The timing stage defaults to five pairs. For the fixture, build the `prefill-f16-query-test` target and run the resulting `./zig-out/bin/test-metal-prefill-f16-query` under the same guard with inherited `DS4` variables cleared. The recorded fixture invocation and guard results are in the local raw root.

## Profile and timing

One intrusive candidate trace compared with experiment 17's candidate trace confirms the intended activation. All 13 eligible fresh-prefill queries changed from F16 matvec to direct-RHS MPP. The 12 large query dispatches totaled 1,286.094 ms in the earlier trace and 23.302 ms in the candidate; one 2,048-row dispatch changed from 26.788 ms to 0.555 ms. Other F16 dispatch counts and geometries, including published-key controls, stayed unchanged; five irregular-row queries stayed on the fallback. Matching dispatches do not establish numerical key equality. Both fresh-prefill traces covered all 101,439 declared encoder records, with no sample-buffer allocation failures in that phase; failures occurred later in both whole runs.

For the 12 large queries, the summed interval from query start to indexer preparation start fell from 1,302.411 ms to 40.192 ms, consistent with removing the projection from that observed dependency chain. The 2,048-row candidate had an additional roughly 7 ms downstream gap of unknown cause, yet its query-start to first-score-start interval fell from 27.822 ms to 8.304 ms. Overlapping work means encoder gaps cannot be treated as GPU idle time. The whole fresh-prefill trace envelope increased elsewhere, from 43,844.475 ms to 44,697.678 ms. These single, intrusive traces establish local activation and dependency observations, not a global performance result. Detailed records are in `candidate-profile-summary.json` and `candidate-profile-notes.txt` under the local raw root.

The 32,768-token profile-free comparison used five alternating pairs. Times below are seconds, derived from throughput rounded to two decimal places. The phase total adds fresh prefill, 128-token teacher decode, the 512-token append after KV restore and another 128-token teacher decode; it excludes loading and snapshot save and restore.

| Pair | First arm | Fresh prefill baseline → candidate | Phase total baseline → candidate |
| ---: | --- | ---: | ---: |
| 1 | Baseline | 43.916 → 43.841 | 59.024 → 59.747 |
| 2 | Candidate | 48.883 → 46.326 | 64.750 → 62.002 |
| 3 | Baseline | 49.136 → 48.000 | 64.971 → 64.101 |
| 4 | Candidate | 50.289 → 48.224 | 66.538 → 64.391 |
| 5 | Baseline | 50.088 → 48.670 | 66.374 → 64.901 |

Fresh prefill averaged 48.462 ± 2.611 s baseline and 47.012 ± 1.983 s candidate (sample standard deviation), with the candidate faster in all five pairs. Phase totals averaged 64.331 ± 3.074 s and 63.028 ± 2.141 s, with the candidate faster in four pairs. Pair 1 was unusually fast for both arms, and its candidate phase total was slower despite slightly faster prefill. Both execution orders occur, but the candidate-first pairs show larger total gains; five pairs cannot resolve the order effect.

Mean fresh decode was 4.357 → 4.385 s, the append was 7.653 → 7.724 s, and post-append decode was 3.859 → 3.908 s. Those smaller phase differences vary across pairs and should not be attributed to the query projection alone. The raw CSV retains all phase and first-token samples.

The same procedure at an 8,192-token frontier produced the following paired seconds:

| Pair | First arm | Fresh prefill baseline → candidate | 512-token append baseline → candidate | Phase total baseline → candidate |
| ---: | --- | ---: | ---: | ---: |
| 1 | Baseline | 18.357 → 18.241 | 6.716 → 6.805 | 32.743 → 32.831 |
| 2 | Candidate | 20.329 → 19.139 | 8.461 → 7.525 | 37.383 → 34.813 |
| 3 | Baseline | 21.159 → 20.451 | 14.148 → 13.152 | 43.810 → 41.754 |
| 4 | Candidate | 20.248 → 20.083 | 7.842 → 7.978 | 36.268 → 36.283 |
| 5 | Baseline | 20.119 → 19.750 | 7.813 → 7.788 | 36.093 → 35.665 |

Fresh prefill averaged 20.042 ± 1.027 s baseline and 19.533 ± 0.868 s candidate, with the candidate faster in all five pairs. Append means were 8.996 ± 2.948 s and 8.650 ± 2.556 s; pair 3 was unusually slow in both arms and is retained. Phase totals were 37.259 ± 4.052 s and 36.269 ± 3.331 s, with the candidate faster in three of five pairs. The large variation limits inference about the total and append. Decode differences are inconclusive. Quality runs are diagnostic and do not support timing claims.

## Practical check and decision

Both agent runs completed a coding task to implement sorted topological layers, including duplicate edges, disconnected nodes, cycles and iterable inputs. An independent oracle evaluated each resulting implementation on 58 graphs in list, iterator and generator forms: 174 evaluations per arm, with zero failures and agent exit code 0 in both. This is one generated task and seed per arm, not 174 model trials or a latency comparison. The baseline first attempted to write `/layers.py`, encountered a permission failure, inspected its working directory and recovered. The candidate initially wrote an incorrect expected result in its own test, diagnosed it and passed the corrected test.

To reproduce the practical check, run the two agent arms sequentially under the pressure guard with the same long prompt and task, a canonical empty working directory and fresh per-arm caches. The prompt placed the coding task after 60,000 characters of unrelated `speed-bench/promessi_sposi.txt` background. Both first calls had 21,337 prompt tokens, 1,859 cached and 19,478 suffix tokens, exercising a full prefill chunk. Both used high reasoning, temperature 1, top-p 1, min-p 0.05, seed 1 and a 2,048-token limit per call, with fixed time `1793835180` and timezone `America/New_York`. The local raw root retains the complete prompt, invocations, traces and oracle inputs and results. These checks support keeping the option for combined qualification; they do not establish general task quality.
