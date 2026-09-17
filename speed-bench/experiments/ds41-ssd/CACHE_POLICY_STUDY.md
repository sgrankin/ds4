# Offline expert-cache eviction study

No inference or GPU execution was used. This is a demand-cache simulation of
recorded decode routes, not a throughput benchmark or a reconstruction of the
native session's cache.

## Findings

At 8,253 slots, aging over 256 decode tokens reduces misses relative to the
16-token proxy on both recordings:2.07%/1.94% with independent cold starts,
and 10.06% / 6.04% in the artificial decode-only carryover stream. These scenarios
are sensitivity checks, NOT lower/upper bounds on real session savings.
SLRU improves misses across both traces and all tested capacities. ARC offers
smaller improvements at large capacity, while this 2Q configuration regresses
throughout. Plain non-decaying LFU regresses on every carryover comparison.

Capacity changes the answer: at 4,096 slots, 256-token aging regresses, whereas
64-token aging improves misses by 2.75–6.17% across the two traces/scenarios.
Therefore do not simply replace the native 16-token constant with 256 globally.
The first runtime candidate should be capacity-aware decay; SLRU is the more
consistent policy candidate in this screen. ARC and 2Q are not first choices
on this evidence. Other parameters or workloads could change these rankings.

Future-aware eviction remains much better: at 8,253 slots it saves 9.23–14.09%
of misses with cold resets and 43.50–48.94% in decode-only carryover. That is
an optimistic scheduling-free reference, not a proven optimum for protected
multi-expert groups, and not an achievable latency prediction.

## Model and provenance

The coding recording contains 3,371 decoded tokens, 40 layers, 6 selected experts
per layer: 809,040 requests in 14 contiguous stretches. The independently recorded
TTL-cache task contains 3,136 decoded tokens: 752,640 requests in 6 stretches.
The latter's raw SHA256 matches `predictor-cache-ttl-capture.json` and
`predictor-ttl-holdout.json`; its compressed routes are now checked in as
`agent-session/predictor-ttl.routes.bin.gz`. No policy parameters were changed
between the coding and TTL evaluations; the TTL task is a second workload,
not a large representative evaluation corpus.

Each request key is (layer, expert). All six current selected experts are
protected throughout the request group, including hits accessed later in
recorded order. Earlier GPU work is assumed complete. Experts have equal size;
capacity is a global slot count. The runtime's current effective per-layer cap
is the complete expert count, so the simulation imposes no restrictive quotas.
There is no prefetch, slab fragmentation, mlock failure, I/O contention, or
policy implementation CPU/GPU handoff cost in the model. All demand misses
are admitted; bypass admission with temporary staging is not modeled.

Prefill is absent. `reset` clears residency AND policy history at every
position gap. `carry` retains both across gaps but omits the intervening
prefill altogether. Neither recreates actual cache state. Each JSON also
reports per-stretch results and excludes the first 64 tokens of EACH stretch
for a warmup sensitivity view; this exclusion does not guarantee steady state.
Global totals below include startup misses. The native cache additionally sees
prefill frequency/seeding and in-flight restrictions; `aging16` is a policy
proxy, not a calibrated reproduction of native counters.

The raw request hashes and simulator source hash are included in each JSON.
Slot capacities 4,096, 7,822, 8,253 represent roughly 37.97, 72.51, 76.50 GiB for the
recorded model's 9,953,280-byte logical expert payload (two IQ2_XXS 5120x2304
matrices plus one Q2_K 2304x5120 matrix). Miss counts therefore translate
linearly into logical expert bytes; they are NOT measured physical SSD traffic.

## Policies

- LRU: oldest request first.
- LFU: all-request frequency, including misses and evicted-key history, with
  recency tie-breaks. Saturation is irrelevant at these trace lengths.
- aging16/64/256: integer-halved LFU at those decode-token intervals. All six
  selected frequencies update before eviction; counters survive eviction.
- SLRU: 20% probationary / 80% protected target; promote probationary hits,
  demote protected overflow, and evict probationary entries first.
- Full 2Q: 25% FIFO target and 50% nonresident ghost history. Cold FIFO hits do
  not promote; ghost hits enter the hot LRU queue. Based on
  [Johnson and Shasha](https://www.cs.cmu.edu/~natassa/courses/15-721/papers/P439.PDF).
- ARC: four lists, adaptive recency target, integer target adjustments, based
  on [Megiddo and Modha](https://www.usenix.org/legacy/event/fast03/tech/full_papers/megiddo/megiddo.pdf).
  For both 2Q and ARC, protected current-group entries cannot be evicted; if
  the preferred resident queue is entirely protected, use the other queue.
  These are explicitly batch-protected adaptations of the original algorithms.
- future: evict the unprotected expert with the most distant next recorded
  use. Future information is used only by this reference policy. For reset,
  lookahead ends at the stretch boundary; carry sees the synthetic joined trace.

No parameter search or training was performed. TinyLFU admission and wider
queue-ratio sweeps remain future experiments, not evaluated policies here.

## Results

Negative changes mean fewer misses than aging16. All-hit is the fraction of
layer calls with all six selected experts already resident before loading.

### Coding session

Stretch lengths: 72, 74, 197, 64, 942, 54, 73, 89, 1176, 54, 72, 234, 123, 147.

**reset**

| Slots | Policy | Misses | Change vs aging16 | All-hit layers | Logical GiB read |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4096 | lru | 142,694 | -0.00% | 39.59% | 1322.73 |
| 4096 | lfu | 158,002 | +10.73% | 35.11% | 1464.63 |
| 4096 | aging16 | 142,695 | +0.00% | 39.59% | 1322.74 |
| 4096 | aging64 | 138,767 | -2.75% | 40.43% | 1286.33 |
| 4096 | aging256 | 149,970 | +5.10% | 37.07% | 1390.18 |
| 4096 | slru | 140,734 | -1.37% | 39.89% | 1304.56 |
| 4096 | twoq | 152,441 | +6.83% | 36.00% | 1413.08 |
| 4096 | arc | 142,672 | -0.02% | 39.44% | 1322.53 |
| 4096 | future | 100,659 | -29.46% | 55.40% | 933.08 |
| 7822 | lru | 91,977 | +0.00% | 59.71% | 852.60 |
| 7822 | lfu | 91,084 | -0.97% | 60.11% | 844.32 |
| 7822 | aging16 | 91,977 | +0.00% | 59.71% | 852.60 |
| 7822 | aging64 | 91,932 | -0.05% | 59.74% | 852.18 |
| 7822 | aging256 | 90,017 | -2.13% | 60.63% | 834.43 |
| 7822 | slru | 90,601 | -1.50% | 60.38% | 839.85 |
| 7822 | twoq | 97,634 | +6.15% | 56.73% | 905.04 |
| 7822 | arc | 91,007 | -1.05% | 60.20% | 843.61 |
| 7822 | future | 81,543 | -11.34% | 65.71% | 755.88 |
| 8253 | lru | 89,193 | +0.00% | 61.20% | 826.79 |
| 8253 | lfu | 88,084 | -1.24% | 61.80% | 816.51 |
| 8253 | aging16 | 89,193 | +0.00% | 61.20% | 826.79 |
| 8253 | aging64 | 89,187 | -0.01% | 61.21% | 826.74 |
| 8253 | aging256 | 87,344 | -2.07% | 62.19% | 809.65 |
| 8253 | slru | 88,090 | -1.24% | 61.80% | 816.57 |
| 8253 | twoq | 95,532 | +7.11% | 57.82% | 885.55 |
| 8253 | arc | 88,217 | -1.09% | 61.73% | 817.75 |
| 8253 | future | 80,958 | -9.23% | 66.10% | 750.46 |

**carry**

| Slots | Policy | Misses | Change vs aging16 | All-hit layers | Logical GiB read |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4096 | lru | 120,932 | +0.01% | 43.40% | 1121.01 |
| 4096 | lfu | 167,923 | +38.87% | 30.65% | 1556.60 |
| 4096 | aging16 | 120,920 | +0.00% | 43.41% | 1120.89 |
| 4096 | aging64 | 117,534 | -2.80% | 43.92% | 1089.51 |
| 4096 | aging256 | 141,007 | +16.61% | 36.70% | 1307.09 |
| 4096 | slru | 119,737 | -0.98% | 43.25% | 1109.93 |
| 4096 | twoq | 129,737 | +7.29% | 39.79% | 1202.62 |
| 4096 | arc | 124,378 | +2.86% | 42.22% | 1152.95 |
| 4096 | future | 59,142 | -51.09% | 66.55% | 548.23 |
| 7822 | lru | 44,512 | +0.00% | 73.77% | 412.61 |
| 7822 | lfu | 48,702 | +9.41% | 71.15% | 451.45 |
| 7822 | aging16 | 44,512 | +0.00% | 73.77% | 412.61 |
| 7822 | aging64 | 44,581 | +0.16% | 73.73% | 413.25 |
| 7822 | aging256 | 40,457 | -9.11% | 75.64% | 375.02 |
| 7822 | slru | 42,233 | -5.12% | 74.86% | 391.49 |
| 7822 | twoq | 49,503 | +11.21% | 71.01% | 458.88 |
| 7822 | arc | 43,842 | -1.51% | 74.08% | 406.40 |
| 7822 | future | 22,179 | -50.17% | 86.54% | 205.59 |
| 8253 | lru | 39,504 | +0.00% | 76.42% | 366.19 |
| 8253 | lfu | 41,609 | +5.33% | 74.93% | 385.70 |
| 8253 | aging16 | 39,504 | +0.00% | 76.42% | 366.19 |
| 8253 | aging64 | 39,476 | -0.07% | 76.43% | 365.93 |
| 8253 | aging256 | 35,530 | -10.06% | 78.39% | 329.35 |
| 8253 | slru | 37,418 | -5.28% | 77.47% | 346.85 |
| 8253 | twoq | 44,183 | +11.84% | 73.78% | 409.56 |
| 8253 | arc | 38,853 | -1.65% | 76.76% | 360.16 |
| 8253 | future | 20,169 | -48.94% | 87.79% | 186.96 |


### TTL-cache task

Stretch lengths: 53, 74, 1802, 720, 59, 428.

**reset**

| Slots | Policy | Misses | Change vs aging16 | All-hit layers | Logical GiB read |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4096 | lru | 116,386 | +0.00% | 42.92% | 1078.86 |
| 4096 | lfu | 127,663 | +9.69% | 39.17% | 1183.40 |
| 4096 | aging16 | 116,381 | +0.00% | 42.92% | 1078.82 |
| 4096 | aging64 | 109,606 | -5.82% | 45.07% | 1016.02 |
| 4096 | aging256 | 118,024 | +1.41% | 42.11% | 1094.05 |
| 4096 | slru | 110,254 | -5.26% | 44.86% | 1022.02 |
| 4096 | twoq | 122,554 | +5.30% | 39.99% | 1136.04 |
| 4096 | arc | 112,446 | -3.38% | 44.19% | 1042.34 |
| 4096 | future | 66,115 | -43.19% | 64.67% | 612.87 |
| 8253 | lru | 49,950 | +0.00% | 73.97% | 463.02 |
| 8253 | lfu | 49,611 | -0.68% | 74.14% | 459.88 |
| 8253 | aging16 | 49,950 | +0.00% | 73.97% | 463.02 |
| 8253 | aging64 | 49,952 | +0.00% | 73.97% | 463.04 |
| 8253 | aging256 | 48,982 | -1.94% | 74.52% | 454.05 |
| 8253 | slru | 49,340 | -1.22% | 74.34% | 457.37 |
| 8253 | twoq | 56,549 | +13.21% | 69.93% | 524.19 |
| 8253 | arc | 49,615 | -0.67% | 74.18% | 459.92 |
| 8253 | future | 42,912 | -14.09% | 78.63% | 397.78 |

**carry**

| Slots | Policy | Misses | Change vs aging16 | All-hit layers | Logical GiB read |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4096 | lru | 107,876 | +0.00% | 44.47% | 999.98 |
| 4096 | lfu | 140,882 | +30.60% | 33.20% | 1305.94 |
| 4096 | aging16 | 107,874 | +0.00% | 44.47% | 999.96 |
| 4096 | aging64 | 101,215 | -6.17% | 46.45% | 938.23 |
| 4096 | aging256 | 117,598 | +9.01% | 40.69% | 1090.10 |
| 4096 | slru | 102,904 | -4.61% | 45.85% | 953.89 |
| 4096 | twoq | 113,106 | +4.85% | 41.87% | 1048.46 |
| 4096 | arc | 104,720 | -2.92% | 45.34% | 970.72 |
| 4096 | future | 50,813 | -52.90% | 68.82% | 471.02 |
| 8253 | lru | 30,744 | +0.00% | 80.28% | 284.99 |
| 8253 | lfu | 32,513 | +5.75% | 78.95% | 301.39 |
| 8253 | aging16 | 30,744 | +0.00% | 80.28% | 284.99 |
| 8253 | aging64 | 30,743 | -0.00% | 80.28% | 284.98 |
| 8253 | aging256 | 28,886 | -6.04% | 81.35% | 267.76 |
| 8253 | slru | 29,737 | -3.28% | 80.90% | 275.65 |
| 8253 | twoq | 35,778 | +16.37% | 77.13% | 331.65 |
| 8253 | arc | 30,342 | -1.31% | 80.50% | 281.26 |
| 8253 | future | 17,369 | -43.50% | 89.10% | 161.01 |

## Reproduce and validate

```sh
python3 -m unittest discover -s speed-bench/agent-session -p 'test_cache_policy_study.py'
python3 speed-bench/agent-session/cache_policy_study.py --output /tmp/cache-session.json
python3 speed-bench/agent-session/cache_policy_study.py \
  --routes speed-bench/agent-session/predictor-ttl.routes.bin.gz \
  --capacities 4096,8253 --output /tmp/cache-ttl.json
```

The output path must not already exist. Python standard library only.
Ten tests cover an independent scan-based LFU implementation, known LRU and
ARC/2Q transitions, randomized protected-group and ghost-list invariants,
capacity bounds, recording-gap semantics, no future-information use by online
policies, and exhaustive optimality checks for the singleton future policy on
all length-six traces over three keys. These do not prove grouped optimality.

Raw results: `cache-policy-session-v1.json` and `cache-policy-ttl.json`.
No inference runtime or default eviction behavior was changed.
