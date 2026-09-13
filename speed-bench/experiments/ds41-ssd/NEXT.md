# DS4.1 short-turn optimization notebook

Updated 2026-09-13. Current round started 16:12:48 UTC; aim to work until about
17:12:48 UTC, completing validation before stopping. Use jj; one commit per
experiment. No concurrent GPU benchmarks. Metal execution and jj snapshotting
need sandbox escalation, not sudo. User deferred vision reproduction.

## Actual user workload and current state

User's ad hoc ds4-agent test was "hello" and other short questions, with SSD
streaming required. Prior 5000-token gains target a 904-token tail and do not
address those short turns. Earlier scalar queue change helped 128-token appends
about 9%. Actual earlier agent trace: 1907-token system prompt ~14.8 seconds;
40-token message append ~3.8 seconds. A saved system KV avoids redoing the system
prompt but does not itself populate the expert-weight cache.

Current accepted default f4d56dec: exact layer-major sweeps for 768-1023-token
warm text tails with 8192-row prefill buffers, excluding resident-encoder mode,
quality, imatrix and TP. 5000-token ABBA improved 68.8-70.5 to 88.6-90.0 tps.
All logits and boundary continuation snapshots matched exactly. Rollback:
DS4_METAL_DISABLE_V41_EXACT_SHORT_PREFILL=1. See README.md for experiments 01-12.

## Working hypotheses, not established speedups

1. Short-turn benchmark first: measure actual agent "hello" after a fresh system
   prefill, after loading its saved KV, and after an active conversation. Record
   absolute latency as well as tps. Distinguish KV state from expert-weight cache.
2. Identify time spent in selected-expert SSD misses, GPU submission/drains,
   Engram reads, and GPU kernels. Partial GPU utilization alone is not evidence
   that fusion will improve wall time.
3. Simple exact fusion: RMS norm + BF16 rounding; matrix output + BF16 rounding;
   shared gate/up + activation + explicit rounding. Preserve every original
   rounding boundary and reduction order, including inside fused kernels.
4. KV publication: position rotation + quantization + direct cache write, removing
   intermediate reads/copies where per-block quantization dependencies permit it.
5. HC mixing/expansion/norm: other graph paths have fused kernels worth adapting,
   but DS4.1 consumes the previous sublayer's mixer and has distinct BF16 rules.
6. Small prefill batches that retain selected-expert streaming (not a complete
   ~150 GiB full-layer sweep). Reuse input/weights and amortize dispatch, while
   tracking how expert unions affect cache misses and causality.
7. Decouple max context, prefill scratch capacity, and expert-cache budget.
   Agent defaults to ctx=100000; DS4.1 picks prefill_cap from ctx (2048/4096/8192),
   rather than independently honoring --prefill-chunk. Explore demand allocation
   and explicit headroom; smaller buffers currently let the auto cache grow.

## Evidence and constraints

At ctx=100000, planned memory: KV/index 0.61 GiB, scratch 8.01 GiB, fixed weights
9.37 GiB, experts 72.51 GiB, prefill reserve 7.12 GiB; total 97.61 GiB.
Large-batch dense, attention and routed kernels differ numerically from scalar.
Exact row projections + scalar attention + scalar MoE recover exact parity.
Two-layer prefetch regressed; HC batching was context-dependent; eight-row MoE
tiles in a full tail sweep regressed overall despite improving first decode.
These opt-ins are not proof that all batching/fusion is unhelpful.

Use fixed input /tmp/ds41-prefill-input.c, from:
  jj file show -r bd66c402 ds4.c > /tmp/ds41-prefill-input.c
SHA256: 1776dbfed177ea14f3ce6cac1d8d0b1c1b44dfff2c2663769a9a5634aeec34e7
Model: gguf/DeepSeek-V4.1-Flash-Q2.gguf. Machine: M5 Max, 128 GiB.
run_abba.py snapshots the binary and requires matching full frontier logits.
Use full continuation snapshots and per-step logits for changed arithmetic.
Keep bulk logs under /tmp and durable summaries here. Update this notebook at
experiment boundaries so a compaction can resume without losing decisions.

13 complete: actual short-agent baseline recorded in agent-short-baseline.json.
Fresh hello 38 tokens=3.527s; restored system KV hello=4.954s; later 6-token
turn=0.379/0.509s. Next: exact RMS norm+BF16 fusion. Cache override is
DS4_AGENT_CACHE_DIR; marker protocol is on stderr, merged into stdout by probe.
