# Five fresh samples per index layout, 2026-10-27

The earlier **3.8% packed-index task slowdown did not reproduce**. Five fresh
processes per layout give 48.730 s F32 versus 48.362 s packed on average,
with a median paired change of −0.12%. Treat this as roughly speed-neutral;
the order effect and first pair do not support claiming a reliable speedup.

Runtime revision: `da31a6be3913`, Apple M5 Max, 128 GiB RAM. This follows the
[serving integration](../index-serving-20261027/README.md). Preparation, runs
and analysis fit the 18:59–19:59 UTC experiment budget. No runtime, preset,
installed executable or main-bookmark change was made in this follow-up.

## Results

Primary time is the fresh answer prefill plus the complete 42-token generated
reasoning/answer. A=F32, B=packed; negative change means packed took less time.
Every sample is retained, including the first pair.

| Pair | Order | F32, s | Packed, s | Packed time change |
| --- | --- | ---: | ---: | ---: |
| 1 | AB | 49.326 | 46.912 | −4.90% |
| 2 | BA | 48.397 | 48.874 | +0.98% |
| 3 | AB | 48.142 | 48.083 | −0.12% |
| 4 | BA | 48.873 | 49.094 | +0.45% |
| 5 | AB | 48.913 | 48.846 | −0.14% |
| Mean | | **48.730** | **48.362** | **−0.76%** |

The median paired change is −0.12%; packed wins three of five pairs. All
F32-first pairs favor packed, while both packed-first pairs favor F32. The
F32-first mean paired change is −1.72%, versus +0.72% when packed runs first.
F32 spans 48.142–49.326 s; packed spans 46.912–49.094 s. These measurements
remove the evidence for a repeatable 3.8% penalty on this task; they do not
establish equivalence across contexts, models or machines.

| Phase | F32 mean, s | Packed mean, s | Mean time change | Median paired change |
| --- | ---: | ---: | ---: | ---: |
| First teacher prefill | 44.735 | 44.420 | −0.71% | −0.57% |
| Independent answer prefill | 44.722 | 44.441 | −0.63% | −0.15% |
| Generated reasoning/answer | 4.008 | 3.921 | −2.19% | +0.05% |

The first F32 decode takes 4.413 s; subsequent F32 decodes take 3.881–3.926 s.
That affects the decode mean substantially. Three packed decodes are slower
by only 0.05–0.30%; the other two are faster. Do not advertise a 2.19% decode
improvement from this small, order-sensitive set.

All ten runs stop normally and return the correct answer with identical
42-token generated traces. All 80 fixed-teacher distributions match the first
F32 run byte for byte: ΔNLL=0, KL=0 and no argmax changes. This is one favorable
task, not an expanded capability evaluation.

The conditional profiling trigger was a positive aggregate slowdown or a
repeated meaningful phase cost. Neither occurred: every phase's packed mean
is lower, and the small mixed decode differences do not show a repeatable
penalty. **No additional profiling run was needed or performed.** The earlier
8K diagnostic remains separate evidence, not a profile of these 32K runs.

## Controls and reproduction

Both arms use the same binary, frozen adjacent shaders, current F32 SIMD
scoring (`DS4_V41_INDEX_SIMD=1`) and native candidate scoring
(`DS4_V41_CANDIDATE_INDEX=0`). Only `DS4_V41_INDEX_FP4=0/1` changes. The earlier
noisy bracket compared packed with the older SIMD-disabled F32 control;
this follow-up isolates packing against the current SIMD-enabled F32 path.

Model: `gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`.
Prompt 32,768 tokens, context 40,960, prefill chunk 8,192, expert cache 48 GiB,
`--case favorable --thinking 1 --max-new 384 --rows 0 --candidate-sweep 0`.
The existing replay probe disables approximate expert routing, main F16 and
replay. Each process performs a teacher prefill, teacher continuation, then
an independent fresh-KV answer prefill and free generation. The primary sum
excludes model initialization and teacher work. It measures a warm-model,
fresh-KV task, not cold process-start latency or the entire benchmark command.

```sh
python3 speed-bench/ideas/pressure_run.py \
  .workspaces/index-five-samples-20261027/guard -- \
  nu speed-bench/experiments/index-five-samples-20261027/run.nu \
  .workspaces/index-serving-20261027/packed/bin/decoder-replay-bench \
  .workspaces/index-five-samples-20261027/runs
python3 speed-bench/experiments/index-five-samples-20261027/analyze.py \
  .workspaces/index-five-samples-20261027/runs
```

Use new output/guard paths when repeating. The frozen executable SHA-256 is
`06709da551987839d052005152d2349055d96b774e79afe56252897be17b9a0a`.
The driver alternates AB/BA/AB/BA/AB, launches one model process at a time and
fails on a nonzero child exit. The analyzer reuses the existing teacher helper
and reads every expected sample. Raw logs, logits, traces and `summary.json`
remain ignored under `.workspaces/index-five-samples-20261027/`.

The unchanged pressure guard exits successfully; all recorded pressure levels
are normal. Swap grows 0.5 MiB over the campaign. The separate idle reference
Flint server was left untouched. Index capacity savings remain the separately
qualified **86.72% reduction in index-owner buffers**, not in total model RAM.
