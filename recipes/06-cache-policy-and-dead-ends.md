# Cache policy, workspace size and scheduling: keep the whole cost

## Route-frequency aging

Native expert eviction combines decaying route frequency with LRU ties and
excludes protected/in-flight entries. The established decay interval is 16
decoded tokens. `DS4_METAL_STREAM_CACHE_DECAY_TOKENS=256` slows decay; it does
not change the clock, prefill seeding or exclusion rules.

```text
on route: increment selected expert's hotness
every N decoded tokens: halve hotness counters
on capacity miss: choose eligible lowest hotness, then oldest use
```

In the [native policy study](../speed-bench/experiments/ds41-ssd/CACHE_POLICY_RUNTIME.md),
base `07e8e131`/upstream `0aaea5a2`, aging 256 reduced recorded tool-session
turn inference 3.66%. A separate attribution run reported 7.80% MORE logical
expert `pread`. The full replay forced 3371 generated tokens and real tool-input
boundaries, but excluded sampling/tool execution. All phase logits and final
states matched. Faster did not mean fewer aggregate misses/reads.

At a smaller 4086-slot cache, aging 64 increased reads 30.41% and combined time
2.77% on a343-token excerpt. This was 64, not 256. Do not infer a64/256
capacity rule from an offline simulator that omitted native prefill, seeding,
readiness and protected entries.

SLRU gave only .73% recorded-session improvement, then regressed actual-agent
response time 6.45% fresh and 1.77% restored. Runtime SLRU was removed in
`1f757780`; `DS4_METAL_STREAM_CACHE_SLRU` is not a current supported switch.
Historical reproduction needs experiment `89370fba`. The simulator remains
useful for hypotheses, not runtime performance claims.

## Workspace demand and safe cache trimming

Prefill scratch competes with resident expert capacity. The opt-in
`DS4_METAL_V41_DEMAND_WORKSPACE=1` starts an implicit workspace at 2048 rows,
growing to 4096/8192 at native large-input thresholds. Explicit `--prefill-chunk`
overrides this mechanism, so setting the flag before the chunk 512 preset does
not test demand allocation. The inspected flag is presence-based: unset it
for the off arm rather than assuming a string 0 disables it.

```text
before larger prefill:
    drain GPU users of old scratch/aliases
    admit new scratch plus retained KV
    trim expert cache to revised capacity, preserving surviving slots
    rebuild workspace tensors and aliases
```

An early full-cache reset leaked page locks; explicit slab unlock fixed that
failure. In-place trimming then preserved surviving warm experts rather than
resetting the whole cache. A partially retained slab's removed slots are
unlocked/ineligible even when the Metal allocation stays alive. Preserve those
ownership invariants when porting a resize; allocating the final correct tensor
size alone does not establish safe or fast growth.

[Combined workspace+aging256 validation](../speed-bench/experiments/ds41-ssd/COMBINED_CACHE_WORKSPACE.md),
built after SLRU removal, reduced full recorded-session turn inference 7.42%,
and a three-short-turn actual-agent ABBA5.17% fresh/2.51% restored. Its2048-row
workspace enlarged dynamic cache 7822→8253 experts (72.51→76.50 GiB); NO growth
occurred in that winning session. Later short→8192→16384→short tests regressed
63.10% and 25.23% in opposite-order pairs; workspace alone regressed 30.05%.
Outputs/state matched and growth completed without mlock failures. Cause remains
unresolved, so both settings stay opt-in. Do not attribute the whole slowdown
to extra reads or announce a proven leak.

## Adopted native-path work worth carrying forward

The [earlier native runtime journal](../speed-bench/experiments/ds41-ssd/README.md)
contains detailed dispatch guards, revisions and state checks. Useful patterns:

| Change | Mechanism and evidence |
|---|---|
| Redundant scalar drains (§5, adopted `24914d66`) | Existing route/cache checks already submit GPU work. Remove redundant layer-end drains, retaining Engram-input reuse and token-completion boundaries. About9.2% higher append throughput in its ABBA; matched frontier logits. Current rollback `DS4_METAL_DISABLE_V41_SCALAR_QUEUE=1`. |
| Early selected loads + small selected batches (§24) | Start missing-expert reads before shared-expert GPU work; use capability-checked short row tiles with existing kernels. Real-agent one-output-token short turns saved 12–31% latency, with no hidden second-scale first-decode cost. Rollback `DS4_METAL_DISABLE_V41_SHORT_OPTIMIZATIONS=1`. |
| Wider selected tiles (§47–48) | Below1024 remaining tokens and when at least half the model's experts fit, selected tiles amortize work without sweeping all weights. A904-token append dropped 36.90→25.66s; do not call it a whole 5000-token prompt gain. Current overrides can select tile 64/limit 1024 at the smaller preset cache. |

The [quiet growth recording](../speed-bench/experiments/native-resident-growth-six-20261003/README.md),
runtime `b9408c3b`, compared native defaults with tile 64/limit 1024 at 6136 slots,
ctx 32768/chunk 512. Startup improved 20.04%, but later turn-model time only 1.04%.
Within-policy repeats were exact; cross-policy vectors differed. Its resident
arm used EARLIER reweighted coefficients, not today's native-weight preset.
This is timing evidence with no generated-task quality oracle. Keep those
distinctions when porting batching or quoting gains.

Chunk512 was inherited by the named profiles from measured context experiments,
not established as optimal for a cold 2k system prompt. Larger batches can
amortize SSD sweeps; smaller scratch can retain more experts. Historical chunk 2048
studies compared with default 8192 under ctx 100000/automatic cache budgets,
not with today's fixed 64 GB/chunk 512 package. One small-input session improved
2.66%, whereas 16384-token input had about 7.5% higher prefill time and substantial
control drift. Measure bootstrap, small appends and growth separately.

## Other tested directions not promoted

* Two-layer expert-prefetch lookahead ran 87–94 tokens/s versus 128–130 controls,
  despite exact logits. Lower explicit read wait was offset by later GPU drain;
  page-residency/contention was a hypothesis, not a measured cause (§3).
* Prioritizing gate/up reads before down and publishing partial readiness gave
  about 1.93% initial short-replay improvement, but the ordinary warm path was
  .91% slower. Staged expert loading remains opt-in (§74–75).
* A learned route predictor's top-k accuracy is not the objective. Predict
  COLD demand early enough to meet a read deadline, include readback/model
  overhead and cache pollution, and allow abstention. An oracle with future
  routes showed a12.9% decode/8.7% session opportunity, but is an architectural
  upper bound, not a deployed predictor speedup (§63–81).
* A one-piece 5000-token sweep was much faster but changed every frontier logit
  against the then numerical reference (§4). It was rejected as a lossless
  scheduling optimization at that time. That result does not prohibit a
  separately labelled approximation with measured task quality; later project
  experiments explicitly allow output degradation.

## How to reproduce the decision, not just a number

Use the same native replay first, then actual sampled tools. Measure initial
system prefill separately from append+decode, and include first decode after a
batch. Sweep cache capacity and workspace growth on the same source revision.
Retain pressure, eviction, locked pages and logical I/O counters, but time
completion wall: fewer misses, lower wait or higher all-route prediction recall
can still make the application slower. Use exact checks for a claimed lossless
reschedule; use fixed-teacher and task-quality tradeoffs for an approximation.

The eviction/workspace experiments were local DS4 work.
[ds4-recipe](https://tangled.org/astrra.space/ds4-recipe), chapters 04–07, is a
useful comparison for memory/SSD scheduling, with a different AMD runtime and
data format. These notes do not transplant its throughput figures.
