# RoPE table for SSD streaming decode

The streaming table path is adopted in the `resident-fast` and `persistent-fast` presets, with `DS4_METAL_V41_STREAMING_ROPE_TABLE=0` as an override; ordinary runtime defaults remain off. At 32K it improved steady decode in all five pairs. At 8K fresh steady decode was neutral within variation, and neither workload establishes a whole-request gain. Fixed-history logits matched exactly within the measured scope.

## Question and mechanism

The [post-overlap profile](../2026-11-03-post-overlap-profile/README.md) identified repeated inverse rotary position encoding (RoPE) of attention heads as a possible decode cost. The baseline low projection rotates the shared heads as it loads them for output pairs. The candidate builds a graph-owned coefficient table once for the current token, then rounds and inverse-rotates each head once before the plain low projection reads it. The table also enables the existing fused index-query rotation and publication path. The low-projection matrix multiplication and its weight traffic still occur; the profiled low-projection interval is not a prediction of savings.

The sole runtime change extends the existing table path to eligible SSD streaming when `DS4_METAL_V41_STREAMING_ROPE_TABLE=1`. The [executor](../../../src/engine/deepseek41/executor.zig) retains the output, table, single-device, quality-mode, GPU feature and active-command checks, and the existing per-position validity and shape fallbacks. The baseline leaves this flag off. Prefill does not enter the eligible output-producing step. The table belongs to the graph and survives streaming command boundaries; no model weights are copied.

## Setup

The experiment started at 18:05:46 EST on 2026-11-03 with an initial one-hour budget. The baseline runtime is `a48a1c52`, with profiling infrastructure at `1ea86fe5`; the candidate adds the opt-in streaming gate. Both arms use the M5 Max and DeepSeek V4.1 Flash 1017 IQ2_XXS/Q2_K imatrix model with Metal SSD streaming, a 64 GiB expert-cache target, packed main and index KV, resident cap five, route/shared-expert overlap with 2,000 µs polling, 8,192-token prefill chunks and 40,960 allocated positions. The [driver](run.nu) clears inherited `DS4` settings and sets both arms identically apart from the candidate flag. Each process starts with an empty process-local expert cache; the OS file cache is not flushed. Expert residency persists across the KV restore before the append.

The 32K workload prefills 32,768 tokens, runs 128 teacher-forced decode steps, restores the KV state, appends 512 tokens and runs another 128 steps. Steady decode excludes the first step; total phase timing includes both complete decode phases and prefills, but excludes startup and snapshot save/restore. It is not a complete serving request: it excludes sampling, MTP and API overhead. Timing runs have profiling disabled and alternate arm order. Raw evidence remains local under `.workspaces/stream-rope-20261103/`.

```nu
./tools/zig-build ds4-bench -j4
python3 speed-bench/ideas/pressure_run.py .workspaces/stream-rope-repro-guard nu speed-bench/experiments/2026-11-03-stream-rope/run.nu .workspaces/stream-rope-repro timing
```

## 32K timing results

Five profile-off pairs show higher steady decode throughput in every pair at both frontiers, in both run orders. Mean ± sample standard deviation was 30.072 ± 0.534 to 30.958 ± 0.364 tokens/s after fresh prefill, and 30.648 ± 0.453 to 31.472 ± 0.358 tokens/s after the append. The corresponding mean steady-decode times fell 2.88% and 2.62%, respectively. These are per-phase comparisons, not whole-request gains.

Each cell lists baseline / candidate; prefill and combined seconds are derived from throughput rounded to two decimals. The first arm alternated as shown. Steady rates cover the last 127 of 128 decode steps.

| Pair (first arm) | Fresh prefill, s | Fresh steady, tok/s | Append, s | Warm steady, tok/s | Combined phases, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 (baseline) | 45.57 / 47.08 | 30.99 / 31.38 | 7.72 / 15.05 | 31.43 / 32.02 | 62.55 / 71.66 |
| 2 (candidate) | 48.88 / 50.80 | 29.75 / 31.33 | 15.32 / 7.73 | 30.34 / 31.62 | 74.04 / 67.67 |
| 3 (baseline) | 48.24 / 47.75 | 29.95 / 30.70 | 15.79 / 15.48 | 30.65 / 31.31 | 73.98 / 73.00 |
| 4 (candidate) | 47.89 / 47.81 | 30.02 / 30.65 | 8.16 / 8.09 | 30.44 / 31.31 | 65.53 / 65.13 |
| 5 (baseline) | 48.49 / 48.54 | 29.65 / 30.73 | 15.18 / 8.04 | 30.38 / 31.10 | 73.52 / 65.80 |

Combined phase time averaged 69.93 ± 5.48 s versus 68.65 ± 3.51 s. Append times again split between roughly 8 and 15–16 s for both arms, with unresolved cause. In pair 1 the candidate's combined time was 14.56% longer, dominated by its slow append. The variation does not support a whole-workload improvement claim. First-step decode latency after fresh prefill averaged 965 ± 40 ms baseline and 962 ± 27 ms candidate; after append it averaged 342 ± 231 ms and 278 ± 264 ms, respectively. These first steps are included in the combined phases, not steady rates.

The fresh control returned 30.07 tokens/s, compared with 28.50 tokens/s in the earlier noisy profile-off batch. That addresses the apparent 28 versus 30 discrepancy as variation between observations; these runs do not establish its cause or a code regression. The pressure guard completed without interruption, with pressure level 1 and swap stable at 24.974 GiB.

## 8K timing results

The same five-pair procedure at 8,192 tokens found fresh steady decode effectively unchanged: 32.032 ± 0.361 versus 32.112 ± 0.139 tokens/s, with the candidate faster in four of five pairs. Warm steady decode averaged 29.658 ± 2.097 versus 31.456 ± 0.769 tokens/s and improved in all five pairs; the 26.47 tokens/s control outlier contributes substantially to the apparent gain. Combined phases averaged 36.787 ± 1.195 versus 36.722 ± 0.918 s, essentially unchanged. This shorter case does not establish a general 6% speedup.

Each cell lists baseline / candidate. The first arm alternated; seconds use rounded throughput as above.

| Pair (first arm) | Fresh prefill, s | Fresh steady, tok/s | Append, s | Warm steady, tok/s | Combined phases, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 (baseline) | 19.48 / 19.89 | 31.85 / 31.97 | 6.66 / 6.79 | 32.26 / 32.31 | 35.05 / 35.60 |
| 2 (candidate) | 20.05 / 20.38 | 31.72 / 32.09 | 6.93 / 6.65 | 29.78 / 32.26 | 36.28 / 35.91 |
| 3 (baseline) | 21.13 / 20.95 | 32.06 / 32.24 | 7.35 / 7.27 | 29.37 / 30.70 | 37.74 / 37.34 |
| 4 (candidate) | 20.60 / 20.72 | 31.89 / 31.99 | 7.05 / 7.25 | 30.41 / 30.94 | 36.84 / 37.05 |
| 5 (baseline) | 21.04 / 21.26 | 32.64 / 32.27 | 7.22 / 7.37 | 26.47 / 31.07 | 38.02 / 37.71 |

The pressure guard completed without interruption at level 1; swap rose from 24.974 to 24.979 GiB, about 5.1 MiB.

## Activation and correctness

A separate instrumented candidate run at 8K confirmed the intended path: 256 coefficient-table kernels, 10,240 transformed-head reductions and 10,240 plain static low projections, corresponding to 256 decode steps across 40 layers. It recorded 2,048 fused index-query publications across eight index layers and no inline inverse-RoPE low-projection kernels. The additional generic low-projection kernels belong to prefill. Instrumentation changes timing, so this run is not a performance comparison.


The existing focused GPU tests comparing the head transform plus plain low projection with inline rotation, and fused index-query publication with separate operations, passed. In fixed-history full-model diagnostics at frontiers 8,192, 8,704, 32,768 and 33,280, all 512 full-vocabulary logit rows matched byte for byte and all 512 post-evaluation logit hashes matched. Mean and maximum delta-NLL and KL were zero, and there were no top-one changes. This establishes equality for those histories on one text and machine; it does not establish behavior across other models, devices or unsupported combinations.

A separate practical check ran the native `ds4-agent` once per arm, with thinking enabled, temperature zero and a 1,024-token generation limit. Each used its own workspace, 8K context and 2K prefill chunk. The task was to create `intervals.py` with `merge_ranges(ranges)`: sort valid inclusive integer intervals, merge overlap or adjacency, return tuples, preserve the input and handle empty input. Both agents created the file, ran their own tests and completed the task. An independent oracle expanded the intervals into integer sets and grouped consecutive values. Both implementations passed seven edge cases and 200 cases generated with seed zero and endpoints from −20 through 20; return types and input immutability also passed. The generated programs differed, and the workspace/tool histories were independent. This is one bounded coding task per arm, not broad agent qualification or a timing comparison. Local transcripts, programs and oracle results are under `agent-smoke/` in the experiment workspace.

The agent and server builds and the launcher default/override checks pass. All guarded stages completed; the last practical check remained at pressure level 1, with pre-existing swap at 24.983 GiB.

Reproduce the focused test and fixed-history diagnostic under the guard:

```nu
./tools/zig-build metal-test -j4
python3 speed-bench/ideas/pressure_run.py .workspaces/stream-rope-test-guard -- ./zig-out/bin/test-deepseek41-metal --rope-tab
python3 speed-bench/ideas/pressure_run.py .workspaces/stream-rope-quality-guard nu speed-bench/experiments/2026-11-03-stream-rope/run.nu .workspaces/stream-rope-quality quality
```

Use `--frontier 32768` on the quality driver to repeat the longer check. The driver stage named `quality` enables diagnostic dumps; it does not enable the runtime quality mode, which excludes the table path.
