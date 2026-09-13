# DS4.1 SSD optimization handoff

## Active round, 2026-09-13 17:18-18:18 UTC

Authorized next hour: profile accepted defaults, tune exact selected tile sizes
and the 256-767-token gap, investigate cold restored expert cache, then measure
full generated responses and correctness. One jj commit per experiment.
Fresh profile: /tmp/ds41-round3-profile. New experiment controls are opt-in until
measured. Keep vision deferred. No concurrent GPU experiments.


Round completed 2026-09-13, approximately 16:12:48-17:13 UTC. Final default
agent ABBA, every short-tile numerical check, and both long-tail regressions
passed. The agent and benchmark binaries are rebuilt. Use jj, one commit per
experiment. Vision reproduction remains deferred until the user is at keyboard.

## Current accepted behavior

- Early selected-expert SSD loading overlaps shared-expert GPU computation.
- Warm text appends below 256 remaining tokens use tiles of at most eight rows.
  Exact dense/HC work and selected expert kernels are batched; attention retains
  scalar causal order. These tiles never map or prefetch whole expert layers.
- A backend capability query checks expert formats, cache capacity, address
  kernels, and backend ablations. Unsupported cases retain scalar prefill.
- Roll back both new defaults with DS4_METAL_DISABLE_V41_SHORT_OPTIMIZATIONS=1.
- DS4_METAL_DISABLE_V41_SELECTED_SMALL_MOE=1 keeps scalar MoE inside tiles.
- Explicit --prefill-chunk now controls DS4.1 allocation, memory estimation,
  and admission independently of context, up to 8192 rows. Default unchanged.
- Older accepted 768-1023-token exact tail sweeps remain enabled. Their rollback
  is DS4_METAL_DISABLE_V41_EXACT_SHORT_PREFILL=1.

## Final measured result on this machine

M5 Max, 128 GiB, DeepSeek-V4.1-Flash-Q2.gguf, SSD streaming, agent ctx=100000.
Final ABBA uses control=current defaults and candidate=short-optimization rollback.
Fresh/restored system KV are separate processes. Three turns: hello, What is
2 + 2?, hello, with one output token each to isolate short-prefill latency.

| Condition | Prior path | New default |
| --- | ---: | ---: |
| Fresh hello, 38-token append | 3.566 s | 2.917 s |
| Restored system KV hello | 5.029 s | 4.417 s |
| Fresh/restored 12-token turn | 0.962/1.209 s | 0.666/0.955 s |
| Fresh/restored 6-token turn | 0.427/0.487 s | 0.323/0.347 s |

Both default samples beat both rollback samples in every condition. First
output after prefill remains about 60-110 ms, with no large decode penalty.
System prefill itself is unchanged. Restoring system KV does not restore the
expert-weight cache; this is why restored hello is slower than fresh-prefilled.
Evidence: README.md experiments 13-24, agent-final-abba.json,
agent-final-metrics.json, and final-small-state.txt.

## Validation and reproduction

Final tails 2/3/4/5/6/7/8/9/17/40 after a 2048-token indexed-attention prefix,
each followed by eight decoded tokens, have byte-identical full continuation
snapshots against rollback. Earlier kernel changes passed full per-step logits
and complete snapshots; synthetic normalization/Q8 shape tests and all 65536
Engram code/scale pairs passed. Tails 768 and 1023 plus eight decoded tokens
also preserve complete snapshots exactly; see final-long-tail-state.txt.

Commands from repo root:

    python3 speed-bench/experiments/ds41-ssd/agent_abba.py /tmp/NEW-DIR --candidate-env DS4_METAL_DISABLE_V41_SHORT_OPTIMIZATIONS=1
    ./tests/test_deepseek41_selected_small gguf/DeepSeek-V4.1-Flash-Q2.gguf /tmp/ds41-prefill-input.c 2048
    ./tests/test_deepseek41_exact_tail gguf/DeepSeek-V4.1-Flash-Q2.gguf /tmp/ds41-prefill-input.c

Do not run concurrent GPU benchmarks. Metal execution needs sandbox escalation,
not sudo. jj mutations need escalation for its backing Git object store.
agent_probe.py uses DS4_AGENT_CACHE_DIR to avoid the user's actual sessions.
The ready marker is on stderr, merged into stdout by the harness. Runners copy
the executable and all external Metal sources, recording shader hashes.
Fixed input /tmp/ds41-prefill-input.c was extracted from commit bd66c402 ds4.c;
SHA256 1776dbfed177ea14f3ce6cac1d8d0b1c1b44dfff2c2663769a9a5634aeec34e7.

## Experiments retained only as opt-ins

- DS4_METAL_V41_FUSED_NORM: exact and ~30% faster microbenchmark; no agent gain.
- DS4_METAL_V41_FUSED_Q8_BF16: exact, ~5% faster matrix microbenchmark; flat appends.
- DS4_ENGRAM_DECODE_LUT: exact 256 KiB lookup; saves only ~0.02 ms per token.
- DS4_METAL_V41_ASYNC_EXPERT_LOAD: worker-based loader slower than simple early load.
- Smaller 64 GiB expert cache: inconclusive. No default budget change.
- Smaller 2048-row scratch: 8.01 to 3.76 GiB, but auto expert cache grows from
  72.51 to 76.50 GiB. Actual short-agent latency flat/slightly worse.
- Disabling read-ahead advice: restored hello ~8% slower. Keep advice.
- Older deeper prefetch, full-tail HC and MoE tiling results are in README.md.

## Remaining ideas, not established speedups

1. Tune selected tile sizes and thresholds across cache budgets and contexts;
   measure full response latency as well as prefill. Preserve scalar reductions.
2. Further fuse shared gate/up and activation while preserving intermediate BF16
   boundaries. Norm and Q8 results warn against trusting microbenchmarks alone.
3. RoPE + quantization + direct KV cache writes, with block quantization rules intact.
4. HC fusion must consume the previous sublayer's mixer in DS4.1, not the newly
   computed mixer used by superficially similar graph helpers.
5. Demand-allocate scratch or explicitly reserve headroom instead of automatically
   giving every freed byte to expert cache. Explicit chunk control now works.
6. Saved system-KV expert hints could warm likely weights while the user is idle;
   include the warmup's startup/I/O cost, and never assume KV includes weights.
7. Profile larger/long-running conversations, cache eviction and pread concurrency.
8. Reproduce the user's original vision-enabled slowdown and session-save issue
   interactively next time. No vision experiment was run in this round.
