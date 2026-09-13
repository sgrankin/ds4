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

14 complete: cache64 first probe is inconclusive, no default change.
15 in progress: fused weighted RMS norm+BF16, synthetic shapes/alias guards
and full-model per-step logit/snapshot comparison before performance claims.

15: synthetic and model parity passed; 40-token ABBA is flat (~13.1 tps).
Actual agent norm probe in progress. Next experiment: move selected-expert
readback/load initiation before shared-expert work, then flush shared GPU work
while pending preads run. Reuse protected cache/pending-load APIs and consume
the exact same selected IDs through the existing override.

16 in progress: EARLY_EXPERT_LOAD starts exact selected reads before shared
work, flushes shared kernels while I/O runs. Model check running. Norm actual
agent probe also flat (3.614/5.027s hello); leave norm opt-in. Readahead advice
consumes 0.595s across 59 restored-process tokens; test disabling before pread.

16 40-token ABBA ~2.9% faster, all logits equal. Real-agent ABBA running
in /tmp/ds41-agent-early-abba. Keep shaders unchanged until it finishes.

16 real-agent ABBA confirmed 3.7% fresh / 5.9% restored hello latency gain.
Candidate for acceptance. 17 no-advice actual-agent ABBA running at
/tmp/ds41-agent-noadvice-abba (existing rollback flag disables F_RDADVISE).
18 Q8+BF16 kernel/API and synthetic test prepared; build running. Benchmark
runners now snapshot all external Metal sources and SHA256 hashes as well
as the executable, so later source edits cannot silently alter A/B subprocesses.

17 complete: no-advice rejected, restored hello 7.9% slower. Keep advice.
18 Q8 synthetic check running /tmp/ds41-q8-kernel.log; new code remains opt-in.

18 synthetic/model parity passed; short-append ABBA /tmp/ds41-q8-abba40
is running using frozen shaders. Next: honor explicit --prefill-chunk in
DS41 allocation/admission/memory reports, then test 2048-row buffers with
default 100000 context. Default cap should remain unchanged unless measured.

18 complete: Q8 short-append ABBA flat (12.96 vs13.01tps mean), leave opt-in.
19 --prefill-chunk wiring built; /tmp/ds41-chunk2048-abba40 running.
Possible 20: tabulate exact Engram FP8+scale -> BF16 conversion (256 KiB LUT),
exhaustively test all 65536 input pairs and rejection semantics, then profile.

19 40-token logits equal; perf drifty. Chunk2048 scratch3.76GiB vs8.01,
cache76.50GiB vs72.51, total97.36 vs97.61. Real-agent ABBA next.
20 Engram LUT opt-in written; exhaustive CPU test running.

19 complete: actual-agent chunk2048 flat fresh and slightly worse restored;
keep default cap. Explicit --prefill-chunk fix retained.
20 Engram LUT exhaustive 65536 cases passed, CPU row benchmark46.4ms->27.1ms
for2000x24 rows; only ~0.02ms saved per token, leave opt-in. Model check now.
21 ASYNC_EXPERT_LOAD reuses existing selected-event worker; committed router
event before worker, shared encode overlaps route/load prep, joins worker on
all exits, synchronous retry if cache entries require main-thread waits. Built.

20 model parity passed too; no end-to-end claim, retained opt-in.
21 async model check running /tmp/ds41-async-model-check.log.

21 model parity passed. Direct actual-agent ASYNC vs EARLY ABBA running
at /tmp/ds41-agent-async-vs-early, baseenv EARLY=1, candidate ASYNC=1.
~21min remain in hour at16:51:35. Final acceptance should enable whichever
loader wins, preserve a rollback, retest actual defaults and long-tail decode.

21 complete: worker slower than simple early, keep worker opt-in.
22 selected small tiles written: 2..8 rows, scalar attention+MoE arithmetic,
batched exact dense+HC, skip all whole-layer mapping/read-ahead/cache seeding,
row.streaming=true. Test tails6/8/17/40 +8decode snapshotparity beforebench.
~18min remain at16:54:48. Need accept EARLY loader unless tiles supersede it,
then validate final build/longtail/defaultactualagent.

22 parity passed tails6/8/17/40 +8continuation; 40token2.11->1.536s within
process only. Real-agent ABBA vsEARLY /tmp/ds41-agent-small-vs-early running.
16min remain at16:56. If wins accept small+early with rollback and run actual
defaults vs rollback; keep all other kernel/worker/cache tweaks opt-in.

22 actualagent mixed freshhello, later6/12tokens clear10-18% improvement.
23 selected small MoE extension prepared: explicit force_resident=false only
inside selectedsmalltiles uses existing selected-address batch kernels; prior
fullprefill and multi-session calls stay force_resident=true. Build running.
At16:59 ~14minremain. Need indexedprefix2048 statecheck, evaluate23, accept
conservative winning policy with rollback and verify actualdefaults.

22 indexedprefix2048 tests all passed. 23 selected MoE indexedcheck running
/tmp/ds41-selected-moe-check.log. ~12min remain at17:00; prioritize final
acceptance and verification after this last experiment.

23 indexedstate6/8/17/40 +8decodeall exact. Warm40t2.021->1.068s.
Actualagent /tmp/ds41-agent-selected-moe running with baseEARLY=1,
baseSMALL_MOE=1 (onlyaffectsselectedtiles), candidateSMALL_PREFILL=1.
At17:01~11minremain. Acceptanceifclearwin, defaultvsrollbackABBA, then
longtailnumericalregression andfinalnotes.
