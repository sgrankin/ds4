# Upstream catch-up: five-pair regression qualification

After correcting packed-cache publication, measured 32-token decode time falls from 2.474 to 2.298 seconds in this short test, with a saving in every pair. Prefill stays near 16.05 seconds and complete fresh-process wall time near 19.99 seconds. The small combined-inference saving does not establish a whole-agent or long-context speedup.

Those are the initial adaptation's results. The later [completion follow-up](#completion-follow-up)
compares the fully integrated port with that adaptation and finds effectively
neutral combined inference time. Its samples and validation are separate below;
the percentages from different baselines must not be added.

## Setup

- M5 Max, 128 GiB unified memory; native Metal, Zig 0.17 ReleaseFast.
- Baseline: `d41daf9f3e556eb565a70dd145e35f59f9719a5b`, extracted and built in its own checkout.
- Corrected candidate runtime: `d84cb67ca195a82906594af80e2c3831a02dbb09`. Later edits affect documentation, formatting and a fixture logging definition; runtime behavior is unchanged.
- Same installed `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf` and first 512 tokens of `speed-bench/promessi_sposi.txt`; 32 teacher-forced decode tokens, context allocation 4096, prefill chunk 2048 and a 32 GiB expert-cache target.
- Packed main and index KV plus SIMD index scoring on for both: `DS4_V41_COMPRESSED_KV_FP4=1`, `DS4_V41_INDEX_FP4=1`, `DS4_V41_INDEX_SIMD=1`. Other inherited `DS4*` variables were cleared. Resident-route approximation and DSpark were off.
- Five measured pairs per candidate, orders AB, BA, AB, BA, AB. Every invocation was a new process and ran from its own checkout to load that checkout’s shaders. No OS cache flush; all samples retained. The existing pressure guard enclosed each serial comparison; neither comparison stopped for pressure. No competing model job or build ran during measurement.

Prefill and decode durations below are reconstructed from the benchmark’s CSV counts and rates, which are rounded to two decimals. Inference windows exclude loading, frontier JSON export and teardown. Wall time includes those operations.

## Corrected candidate: all measured samples

| Pair/order | Baseline prefill s | Candidate prefill s | Baseline decode s | Candidate decode s | Baseline wall s | Candidate wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / AB | 15.846 | 16.085 | 2.627 | 2.265 | 19.913 | 19.995 |
| 2 / BA | 16.030 | 15.960 | 2.433 | 2.281 | 19.949 | 19.857 |
| 3 / AB | 16.055 | 16.106 | 2.415 | 2.343 | 19.924 | 20.126 |
| 4 / BA | 16.192 | 16.070 | 2.397 | 2.301 | 20.047 | 19.984 |
| 5 / AB | 16.167 | 16.020 | 2.498 | 2.301 | 20.122 | 19.954 |

| Metric | Baseline mean ± sample SD | Candidate mean ± sample SD | Change in mean |
| --- | ---: | ---: | ---: |
| Prefill, s | 16.058 ± 0.137 | 16.048 ± 0.059 | -0.06% |
| Decode, s | 2.474 ± 0.094 | 2.298 ± 0.029 | -7.13% |
| Decode, tokens/s | 12.948 ± 0.475 | 13.928 ± 0.176 | +7.57% |
| Prefill + decode, s | 18.532 ± 0.091 | 18.346 ± 0.075 | -1.00% |
| Process wall, s | 19.991 ± 0.090 | 19.983 ± 0.097 | -0.04% |

Decode time falls by 3.00–13.80% across pairs; the median paired reduction is 6.27%. The widest reduction includes the slowest baseline decode sample, which remains in the table. Prefill moves in both directions (−0.91% to +1.51%), as does process wall time (−0.83% to +1.01%). Neither shows a clear change in this screen. The bundle includes scheduling, kernel, weight-reader and quantizer changes. Changed quantization can also alter expert routing and SSD/cache I/O, so this comparison cannot attribute the decode saving solely to faster kernels.

All five corrected candidate frontier arrays have the same values across repetitions, as do the baseline arrays. Comparing candidate with baseline after prefill gives KL(baseline || candidate) = 0.006338, maximum absolute logit delta 1.24981 and the same argmax in all pairs. The imported exact-division quantizer changes the numerical trajectory; unchanged weights and identical teacher histories do not imply identical logits. This one frontier does not establish task quality or compare all subsequent decode distributions.

## Why the screen was repeated

The first five-pair screen used candidate `037816ecc12afa20f19e51a9e0ef8b600efea192`. It found a 4.09% reduction in decode time, neutral prefill (+0.016%) and a 0.129-second increase in process wall time. All five paired frontier arrays matched then. Those results are retained below, but this candidate is superseded by the publication fix.

The subsequent installed-model F32/packed continuation test exposed the mismatch. Upstream changed the FP4 quantizer to exact division; local packed publishers still used fast reciprocal multiplication. At BF16 input −37.5 and scale 7.5, the exact ratio has magnitude 5: the tie selects the even E2M1 code for 4, giving −30. Reciprocal error could select 6, giving −45. The model test observed a 0.223 maximum logit difference between cache layouts after one fresh step.

All equivalent main, index, fused-RoPE and suffix publishers now use exact division. The new midpoint corpus fails before the fix and passes afterward. The model comparison then has maximum delta 0 on that fresh step, and at most 3.815e-6 in subsequent constrained-draft cycles, including packed/F32 conversion and SSD restart. These checks compare packed storage with its canonical quantized F32 values, not with the original unquantized activations.

### Superseded candidate: all measured samples

| Pair/order | Baseline prefill s | Candidate prefill s | Baseline decode s | Candidate decode s | Baseline wall s | Candidate wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / AB | 16.295 | 16.075 | 2.408 | 2.292 | 20.122 | 20.221 |
| 2 / BA | 16.045 | 16.177 | 2.437 | 2.314 | 19.904 | 20.088 |
| 3 / AB | 15.975 | 16.060 | 2.421 | 2.331 | 19.852 | 20.010 |
| 4 / BA | 16.126 | 16.101 | 2.406 | 2.329 | 19.965 | 20.063 |
| 5 / AB | 16.244 | 16.285 | 2.441 | 2.351 | 20.180 | 20.289 |

One earlier diagnostic aborted after prefill because its logit directory had not been created. It is retained separately and is not a measured pair. Exact narrow decode-weight copies occupy 491.33 MiB with this checkpoint, charged once per engine. The process-wall remainder includes startup, JSON export and teardown; this screen does not separate their costs.

## Reproduce and limits

Build `ds4-bench` in each revision with `./tools/zig-build ds4-bench -j4`. Clear inherited DS4 settings, apply the three packed-cache settings above, create `LOGITS_DIR`, and run from the selected checkout. Use the same absolute model and prompt paths on both sides. Repeat in the five-pair order above under the [pressure guard](../../ideas/pressure_run.py).

```sh
./zig-out/bin/ds4-bench --model MODEL --metal --ssd-streaming \
  --ssd-streaming-cache-experts 32GB --prompt-file PROMPT \
  --ctx-start 512 --ctx-max 512 --ctx-alloc 4096 --prefill-chunk 2048 \
  --gen-tokens 32 --teacher-forced-decode \
  --dump-frontier-logits-dir LOGITS_DIR --csv RESULT.csv
```

The [catch-up journal](../../../docs/lab/journal/2026-11-02-upstream-catchup.md) records model continuation, reasoning and primitive qualification. This screen does not measure long-context throughput, persistent-agent task success, residency on larger Macs or physical tensor parallelism. Raw CSVs, logs, full frontier arrays, pressure samples and the short local Nushell loops remain ignored under `.workspaces/upstream-890898cf/`; the corrected cohort is in its `final-latency/` subdirectory.

## Completion follow-up

The full port adds the ordinary graph fusions, tiny-batch routed dispatch,
pre-M5 selectors, exact batched K-slice projections, index MMA selection and
two-Mac scheduling that the initial adaptation had omitted. This comparison
asks whether those integrations regress the same short single-Mac workload.
It does not exercise every new selector: the installed IQ2_XXS/Q2_K checkpoint,
for example, does not activate resident MXFP4 expert concurrency.

The baseline is `205962a8c4a4`, the validated initial adaptation, with its own
binary and matching shader tree. The candidate runtime is
`4e5e51390cdb96c5ead7c89ff1ac9b2c4ed538df`; subsequent edits only update
documentation and this report. Hardware, prompt, teacher tokens, cache settings,
allocation, command and timed scope are the same as above. Five fresh-process
pairs alternate AB/BA order, with no OS cache flush or competing model/build.
The pressure guard completed normally with no adverse samples.

| Pair/order | Baseline prefill s | Candidate prefill s | Baseline decode s | Candidate decode s | Baseline wall s | Candidate wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / AB | 15.152 | 15.812 | 2.337 | 2.370 | 21.904 | 19.949 |
| 2 / BA | 15.764 | 15.817 | 2.365 | 2.348 | 19.691 | 19.722 |
| 3 / AB | 15.866 | 15.896 | 2.538 | 2.395 | 19.933 | 19.824 |
| 4 / BA | 15.876 | 15.778 | 2.465 | 2.344 | 19.874 | 19.623 |
| 5 / AB | 15.915 | 15.817 | 2.367 | 2.390 | 19.847 | 19.757 |

| Metric | Baseline mean ± sample SD | Candidate mean ± sample SD | Change in mean |
| --- | ---: | ---: | ---: |
| Prefill, s | 15.715 ± 0.319 | 15.824 ± 0.043 | +0.70% |
| Decode, s | 2.414 ± 0.084 | 2.370 ± 0.023 | -1.86% |
| Decode, tokens/s | 13.266 ± 0.454 | 13.506 ± 0.133 | +1.81% |
| Prefill + decode, s | 18.129 ± 0.372 | 18.194 ± 0.063 | +0.36% |
| Process wall, s | 20.250 ± 0.929 | 19.775 ± 0.121 | -2.35% |

Combined inference is effectively neutral in this screen. Decode is lower in
three of five pairs; its small mean saving is not a firm speedup. The first
baseline has unusually fast prefill and slow process wall time, making it
especially important to distinguish these scopes. It remains in the table.
The process-wall reduction is dominated by that first baseline's time outside
the inference windows; this test does not separately time loading, JSON export
and teardown. No profile was needed to investigate a material inference
regression because the observed mean change is small relative to variation.

All five frontier arrays are identical within each side and between sides:
maximum logit delta and KL are zero, with the same argmax. This measures the
distribution immediately after the 512-token prompt, not every decode step or
task quality. Raw outputs are in the ignored `completion-latency/` directory
under the same workspace; `compare-completion.nu` records the serial loop.

### Completion correctness checks

The final build installs all five executables. `make test ZIG_FLAGS=-j4` passes,
as do the production-linked DeepSeek, dense/shared, expert-buffer, normalization,
Qwen, GLM and steering Metal fixtures. New checks cover exact K-slice rows and
tails, BF16 input, grouped low projection, F32 index MMA, MXFP4 deduplication,
Q4 small batches and concurrent shared FFN. Pre-M5 kernels and host admission
are exercised on M5; that does not establish their performance on an older GPU.

The first two new fixture failures were setup errors: a synthetic packet lacked
`nr0=2`, and an in-batch-only K-slice entry was called without an open command
batch. The fixtures now follow the production contracts. No production
admission check was weakened. TP GPU loopbacks verify inline/session gate
selection, payload order and bounded heap across 8,192 exchanges. Paired TCP
ranks pass all four optional full-residency offers and all three required-mode
mismatches. Initial sandbox socket denial was followed by a permitted passing
run; it was not treated as a transport failure.

The installed-model checkpoint test passes six endpoints, 18 teacher rows,
two relocations, immutable pre-RoPE archives and exact canonical round trips.
Collected final-prompt logits are byte-identical to ordinary sync. Endpoint
execution-shape differences reach KL 0.006321719 and absolute delta-NLL
0.052794232 nats/token; these are continuation diagnostics, not task scores.
The F32/packed DSpark test has zero difference on the fresh scalar step and
at most 3.8147e-6 on later constrained-draft comparisons, including SSD restart.
A reasoning-enabled CLI smoke test returns `{"sum": 42}`.

Reproduction commands for those model checks, each run separately beneath
`speed-bench/ideas/pressure_run.py`, are:

```nu
# Enable the three packed/SIMD settings from Setup for the capture test.
./zig-out/bin/test-checkpoint-capture MODEL --cache-gb 32
./zig-out/bin/test-dspark-checkpoint MODEL DSPARK EMPTY_SSD_DIR --all-packed-cross
./zig-out/bin/ds4 --model MODEL --metal --ssd-streaming --ssd-streaming-cache-experts 32GB --ctx 4096 --prefill-chunk 2048 --think --temp 0 --tokens 256 -p 'Calculate 17 + 25. Give the final answer as a JSON object with the single key sum.'
```

Physical two-Mac/RDMA execution, full resident 1017 MXFP4 inference, older-Mac
latency and the full-table Engram warmer remain unmeasured. The retained CUDA/HIP
header matches upstream; its previously removed build targets were not restored
or tested. These are qualification limits, not omitted upstream functionality.
