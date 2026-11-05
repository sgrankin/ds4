# Resident expert routing without waiting for every ideal expert

## Algorithm to port

When a large MoE streams weights from SSD, the router's preferred experts may
be absent while slightly weaker experts are resident. Substitute a bounded
number of those misses. This is intentionally approximate: neighboring router
scores do not imply equivalent expert functions.

Our DeepSeek-specific selector starts from the six original selected experts,
unbiased affinities `p`, learned correction bias `b`, and native mixture weights.
The configurable substitution limit selects the one to six original slots with
the lowest unbiased affinities for consideration. A slot's original normalized
mass is its affinity divided by the sum of the six selected affinities. The
selector does not search stronger slots to make up for a
resident or otherwise ineligible choice. Original resident experts are preserved.
Each considered slot must satisfy the per-slot original normalized mass limit;
multi-slot plans also have a total replaced-mass limit. Setting that total limit
to 1 disables the cap, including a rounding edge case where the six normalized
float masses sum slightly above one. One-slot mode does not consult the total
limit. Choose the highest `p+b` resident candidate outside the six original IDs
and all earlier substitutes, within the configured corrected rank, with affinity
within `p_old*(1±gap)`, and
`(p_old+b_old)-(p_new+b_new) ≤ gap*p_old`. Ties use expert ID. The tested gap .90
is permissive; do not describe it as a near-identical function guarantee.

Keep each original slot's GPU mixture coefficient unchanged. Do not recompute
weights from substitute affinities. Retain six unique IDs and account mass from
the original selection, not an evolving replacement mixture.

```text
native_ids, native_weights = route(hidden)
resident = snapshot_cache()
plan = bounded_replacements(native_ids, affinities, bias, resident)
ids = apply_id_changes(native_ids, plan)
evaluate_experts(ids, coefficients=native_weights)
```

For selected-cache prefill, snapshot residency for ALL layers once at the start
of each selected-prefill call, then plan every row from that frozen mask. Loads
caused by early rows must not retroactively change the mask. Full layer sweeps
and scalar fallbacks retain native prefill routing. Initial-prefill selected
tails can still change, so this is not exclusively a warm-append intervention.

The assumptions are model-family routing semantics, unique resident experts and
an execution path that accepts changed IDs with preserved coefficients. Port
the policy at the routing boundary; do not copy DeepSeek bias/affinity formulas
unchanged into a different MoE. Sources:
[selector](../src/resident_route.zig), [runtime](../src/engine/deepseek41/route_services.zig),
[focused test](../tests/test_resident_route.c).

## Local knobs and measured points

The `resident-fast` and `persistent-fast` presets use at most the five weakest
original slots, protecting the strongest, with per-slot and total mass limits
of 1. Their decoder settings are
`DS4_METAL_V41_RESIDENT_ROUTE_GAP=.90`, `_MAX_SUBS=5`, `_MAX_RANK=64`,
`_MAX_MASS=1`, `_MAX_TOTAL_MASS=1`, `_POLICY=replace`, `_NATIVE_WEIGHTS=1`
(all latter names share `DS4_METAL_V41_RESIDENT_ROUTE`). Selected prefill adds
`DS4_METAL_V41_RESIDENT_PREFILL_GAP=.90`. `resident-fast` enables the decoder;
`persistent-fast` also enables prefill. Setting the prefill gap explicitly 0
disables its substitutions. Ordinary runtime behavior remains opt-in.

| Experiment | Scope/result | Qualification |
|---|---|---|
| [Item22](../speed-bench/experiments/resident-combined-hour22-20261004/README.md), `fe5d6e26` | Decoder cap 3 saved 9.70% task time; cap 1 saved 3.58% | Ledger/LRU, stable high-effort READ0/WRITE128/FINAL1, suffix off; all 8 first codes 161/161; native drift .33% |
| [Item37](../speed-bench/experiments/recipe-prefill-hour37-20261008/README.md), `9c6918f0` | Prefill cap 3 saved 10.64% warm append /9.32% versus faster control; coding pair 5.98% /2.86% | Q2, SSD64 GB, ctx 8192/chunk 512/tile 64/limit 1024; decoder off; two reused coding families; coding drift 6.22% |
| Item37 fixed teacher | cap 3 ΔNLL +.005043, KL .002259; cap 1 +.001723 / .000834 | One evolving trajectory, 343 teacher rows, not 343 independent tasks; cap 3 worst target ΔNLL +1.111 and KL .472 |

Item 37 initial prefill changed only .26%, establishing no initial-prefill gain.
Cap 1's warm mean saving 1.84% sat inside 22.39% control drift and was 9.15% slower
than its faster control. Its later coding arm was unbracketed. Keep those weak
points rather than selecting only a favorable denominator.

The [2026-11-02 sweep](../speed-bench/experiments/2026-11-02-resident-six/README.md)
extends the selector through six slots. With five timing samples per policy,
cap five without mass limits saves 7.16% on a 512-token warm append plus 32
decode steps relative to the then-current three-substitution preset. Its
64-position teacher delta-NLL is +0.0446 nats/token against native routing.
Cap six has effectively
the same mean latency and substantially more distribution drift. Four and the
tighter score-gap policy offer other tradeoffs. These short probes identify
candidate settings; they do not establish broad agent quality. The report owns
the full curve, samples and qualification limits.

The [longer quality follow-up](../speed-bench/experiments/2026-11-02-resident-long/README.md)
compares 4,096 corpus positions and 3,371 recorded assistant positions per
policy. Five retains a small average likelihood cost, with larger local changes
in final explanatory prose. All three policies pass the long-prompt retrieval
and correction checks. This supports the candidate tradeoff without establishing
autonomous coding accuracy. The five-substitution policy was adopted for both
presets on 2026-11-03.

The [32K dependency profile](../speed-bench/experiments/2026-11-03-context32-profile/README.md)
finds 99.33% of decode layer calls fully resident after substitution with a
64 GiB cache target on the measured prose workload. Avoiding expert reads does
not remove the per-layer GPU-to-CPU routing handoff: the host still selects
substitutes before routed computation can proceed. The
[overlap follow-up](../speed-bench/experiments/2026-11-03-route-overlap/README.md)
now queues independent shared-expert work across that decision, with bounded
event polling and cache-only retirement at the ready boundary. It preserves
the selector and native coefficients. Both resident presets enable this
scheduling package; its separate decode saving should not be added to the
routing-policy percentages above.
The [profile after overlap](../speed-bench/experiments/2026-11-03-post-overlap-profile/README.md)
still finds little exposed expert I/O in this workload. The subsequent
[streaming RoPE experiment](../speed-bench/experiments/2026-11-03-stream-rope/README.md)
enables attention rotation reuse in the resident presets without changing the
resident policy. That decode improvement has its own baseline and measurements.
The [next dependency profile](../speed-bench/experiments/2026-11-03-post-rope-profile/README.md)
finds that resident selected-prefill can still wait substantially before its
expert kernels start. The [slab-size comparison](../speed-bench/experiments/2026-11-03-expert-slabs/README.md)
preserved cache capacity and measured logits but did not establish an append
or whole-workload improvement. The selection policy and 4 GiB slab default
remain unchanged.

## Experimental GPU selection without the host handoff

The [resident-handoff prototype](../speed-bench/experiments/2026-11-04-resident-handoff/README.md)
repairs missing IDs on the GPU using only pinned resident experts, retaining
native slot weights but relaxing the preset's substitution limits. Periodic
ordinary routing refreshes the cache. It remains opt-in. The initial five-pair
tests found no speed gain and consistently slower warm 32K decode. Fixed-history
drift was measurable, while one coding task per arm passed the independent checks.

The dependency profile showed fewer routing handoffs, and CPU samples exposed
redundant address-table writes. The
[address-reuse follow-up](../speed-bench/experiments/2026-11-04-resident-address-reuse/README.md)
removes those writes behind a second opt-in. It preserves all tested logits but
does not establish an attribution speedup. The subsequent
[concurrent-branch experiment](../speed-bench/experiments/2026-11-04-resident-parallel/README.md)
uses the resident-only path to overlap shared and routed experts. Its combined
comparison improves fresh decode at 8K and 32K and warm decode at 8K, with neutral
warm 32K timing. The added concurrency preserves all tested serial-route logits;
the GPU substitution policy remains approximate. The
[refresh study](../speed-bench/experiments/2026-11-04-resident-refresh/README.md)
selects ordinary routing every sixteen positions for further agent-workload
qualification. It improves fresh decode at 32K and over longer 8K continuations;
warm gains are smaller and less certain. Fixed-history quality has a modest
average cost with larger local tails. The four-position option has lower
measured drift but unresolved timing because of a retained slow sample.
The [agent qualification](../speed-bench/experiments/2026-11-04-resident-agent-qualification/README.md)
keeps the route experimental: five recorded-session pairs establish no complete
inference gain, and one reasoning-enabled coding task produced repetitive thought
without an implementation at both tested budgets. Fixed-history quality alone
did not decide this outcome. Native CPU samples confirm that host waits shifted
away from selection, preserving useful groundwork. A controlled comparison of
routing policies must separate that behavioral failure from prompt differences.
The [controlled follow-up](../speed-bench/experiments/2026-11-04-resident-behavior/README.md)
finds that completion varied across prompt and sampling conditions: all three GPU
refresh configurations passed both controlled greedy prompts, while normal sampling retains
one GPU failure among five seeds versus none for current CPU routing. This
narrow evidence neither identifies the earlier failure's cause nor establishes
general coding accuracy. The policy remains experimental while exact kernel
optimization continues.
The [SIMD search follow-up](../speed-bench/experiments/2026-11-04-resident-simd/README.md)
preserves the GPU policy's tested choices while distributing each missing-expert
search across 32 lanes. It substantially reduces isolated repair cost and improves
fresh 32K decode in all five attribution pairs. Shorter-context results are mixed,
including slower complete fresh decode, and total inference remains inconclusive.
Keep this exact execution change behind its opt-in; it does not qualify or
promote the approximate GPU routing package.
The [native-router fusion](../speed-bench/experiments/2026-11-04-post-simd-profile/README.md)
combines selection and repair without additional drift on the tested histories.
Repeated isolated tests reduce work when experts are missing, but five-pair
model timings at both context sizes did not establish a whole-workload benefit. It
remains opt-in; this does not promote the GPU routing policy.

The [later package qualification](../speed-bench/experiments/2026-11-04-package-requalification/README.md)
finds a lower five-pair recorded-session mean and less variation, with the
candidate faster in four pairs. The fast first baseline remains in the result,
so the size and repeatability of the gain are uncertain. Recent exact changes
preserve the earlier candidate's full replay logits, while a fresh candidate
coding solution fails the oracle's integer-adjacency cases. Retain the useful
kernel work and the failed trace; keep the approximate policy opt-in rather
than treating either one task or mean drift as a general quality verdict.

The [submission-cadence follow-up](../speed-bench/experiments/2026-11-05-gpu-submit-cadence/README.md)
selects four-layer submission for the experimental package. Removing CPU
handoffs made it worth revisiting the earlier every-layer choice. Complete fresh
decode improved slightly in nine of ten pairs across 8K and 32K, and all 512
checked predictions matched. Warm results were mixed. The profile confirms fewer
command buffers but does not show lower native host latency; retain the small
measured gain without extrapolating it to every phase or promoting GPU routing.

The [first-decode address-path screen](../speed-bench/experiments/2026-11-05-first-decode-admission/README.md)
then forced existing address dispatch before the automatic four-token threshold.
All 512 tested predictions matched. Small timing differences favor keeping this
existing option in the experimental package, but results are mixed and do not
establish a general gain. The intended kernel switch occurred while the large
first-decode gaps remained. The cause of the delay remains unresolved; resource
admission is a hypothesis for the next investigation. No new warm-up phase,
kernel or cache representation was added.

The next [slab-residency screen](../speed-bench/experiments/2026-11-05-streaming-slab-residency/README.md)
registered the actual streaming buffers with a Metal residency set attached to
the command queue. It consistently shortened the first decode step after fresh
prefill, yet made complete fresh and warm decode slower in all five pairs at
both 8K and 32K.
Do not combine this form with the tested shared-event overlap schedule. Its
profile still supplies useful groundwork: gaps
between command buffers largely disappear, while new waits appear near the
router's shared-event boundary. Preserve that evidence for a bounded test of
residency policy instead of concluding that residency itself cannot help.
All 512 checked predictions matched, and existing cache-pressure relief remains
in force. The [storage manual](../docs/storage/expert-streaming.md#how-metal-stores-cached-experts)
owns the resource-lifetime explanation.

A [standing request without queue attachment](../speed-bench/experiments/2026-11-05-standing-slab-residency/README.md)
avoids those added refresh waits, but brings back the original gaps before
expert work. Five-pair comparisons at both context sizes are mixed and close
to baseline; they do not establish an overall gain. Leave this policy off too.
The useful distinction is between a request to prepare residency and associating
the whole set with command execution. The results support testing that
association's interaction with the existing shared-event handoff, without
claiming that the driver mechanism has been identified.

The [combined follow-up](../speed-bench/experiments/2026-11-05-slab-event-interaction/README.md)
retains queue residency with shared-event overlap disabled as an experimental
latency candidate. Fresh 128-token decode and total inference improved in all
five pairs at both 8K and 32K. The first-step saving outweighs a small recurring
cost in these workloads: throughput after the first token is lower, and warm
decode does not share the clear gain. All 512 fixed-history logit rows matched.
Disabling overlap also changes shared-expert scheduling and fusion, so this
does not isolate event signaling as the cause. CPU-routing presets remain
unchanged; the complete report owns the measured tradeoff and its limits.

The [next scheduling test](../speed-bench/experiments/2026-11-05-refresh-shared-flush/README.md)
removed the separate shared-work submission on CPU refreshes. It reduced
command-buffer count but did not establish an overall gain; complete fresh
decode was slower in nine of ten pairs. The standalone selector was removed.
Quality remained exact at both primary frontiers and with a smaller cache that
performed actual scalar expert reads. The useful follow-up is an explicit
all-hit admission path for parallel shared/routed work, with preserved pins
and serial miss fallback. The submission change alone does not establish that
capability or explain the measured slowdown.

The [cached-refresh follow-up](../speed-bench/experiments/2026-11-05-cached-refresh-parallel/README.md)
implements that exact-selection proof and exercises the concurrent FFN schedule
while preserving serial miss fallback. It remains an unselected, opt-in
prototype. Its sampled FFN sub-envelopes are slightly shorter, but larger GPU
inter-buffer gaps precede the FFN buffers and sampled refresh windows are longer.
These observations do not identify a complete cause. Keep this finding for
a bounded follow-up; the selected latency package still uses serial CPU
refreshes. The report separates five-pair latency evidence from the intrusive
diagnostics and records the exact quality and cache-accounting checks.

The [periodic session qualification](../speed-bench/experiments/2026-11-05-long-session-qualification/README.md)
then compared the selected package with published CPU routing. Its modest
five-pair mean saving and unchanged fixed-history logits relative to the earlier
GPU candidate support continuing the experiment. A fresh interval task again
missed adjacency, while both arms passed a separate long-background graph task.
The package remains experimental; fixed replay speed and these limited behavior
checks do not justify a general agent-speed claim or preset promotion.

The [command-buffer residency test](../speed-bench/experiments/2026-11-05-scoped-residency/README.md)
narrows slab association to buffers that read cached expert weights. Keep this
compute-only policy unselected: it restores the large fresh-start delay, while
steady decode is nearly unchanged. The trace confirms that it excludes pure
shared/router buffers, but also the blits that initially populate the cache.
The extra fresh-decode gap precedes the first associated expert buffer. Retain
the opt-in groundwork for a test that also covers seed writes; the current
result does not establish that narrower association cannot help.

The [seed-write extension](../speed-bench/experiments/2026-11-05-seed-residency/README.md)
adds association to those existing initialization blits. The large historical
first-compute gap is absent in this comparison, and all checked outputs match.
Five pairs at both contexts still show mixed phase costs without establishing
an overall gain.
That result retained extended mode `2` as groundwork while the selected package
stayed on queue mode `1`. The narrower association enabled the later scheduling
test below; initialization work did not become free.

The [scoped-overlap test](../speed-bench/experiments/2026-11-05-scoped-overlap/README.md)
then restored shared-expert overlap with command residency fixed. Large-cache
decode regressed in every pair at both contexts. The trace concentrates the
added event-adjacent gaps in buffers that also contain earlier expert work.
Smaller-cache warm decode benefits, so this is not a general rejection of
overlap; the whole-workload result there remained mixed. That test kept the
selected large-cache package unchanged and identified a separate submission
after CPU-expert work as the next question.

The [CPU-expert boundary](../speed-bench/experiments/2026-11-05-cpu-expert-boundary/README.md)
now separates those expert reads from the next router event without changing
ordinary GPU cadence. Select its mode-2/overlap/flush combination for the
experimental GPU package: direct five-pair comparisons show about 2% faster
fresh complete decode at both frontiers, with warm decode faster in nine of ten
pairs. The 32K append costs more and complete workload times remain mixed;
this is a decode tradeoff. The report retains all samples and the unchanged
fixed-history outputs. Normal presets and installed frontends are unchanged.

The [residency-call deduplication](../speed-bench/experiments/2026-11-05-residency-once/README.md)
prototype removes repeated association within an unchanged active batch. Its
trace reduces calls from 4,361 to 863 while preserving the same 850 covered
buffers. Ordinary retained-buffer timings remain mixed, including a small
decrease in fresh steady throughput at 8K and a favorable smaller-cache case.
Keep it unselected for a bounded
unretained-buffer test: that path also removes repeated completion captures.
This was a source-supported next question, not an established speedup or a newly
exposed GPU stall. The [unretained follow-up](../speed-bench/experiments/2026-11-05-unretained-residency/README.md)
repaired a scratch lifetime defect and restored exact checked outputs, but its
five-pair timings do not justify selection: 32K decode and append cost more,
while 8K total time remains mixed. Keep the correctness repair. Deduplication
and unretained buffers remain unselected pending a bounded test of the extra
completion-handler ownership work and direct qualification against the selected
boundary package.

Detailed encoder profiling changes batching and adds host overhead; its gap reduction is not a
predicted ordinary-runtime saving. The
[Apple Silicon manual](../docs/models/deepseek/apple-silicon.md#resident-expert-substitution)
owns activation and lifetime details; neither fast preset enables this prototype.

## What did not earn its complexity

The [functional-expert map study](../speed-bench/experiments/resident-functional-hour2-20261003/README.md)
did not establish an advantage over the simpler affinity selector. Early
native-weight controls matter: replacing IDs and reweighting is a different
intervention from preserving the native coefficients.

The [original-arrival prototype](../speed-bench/experiments/original-arrival-hour3-20261003/README.md)
used a bounded worker to fetch one displaced ideal expert, promoting it only if
READY at a nonwaiting deadline. Initial joins made strong variants 10–18% slower
than substitution alone. Joining only admitted tickets made retention-age 8
timing-neutral (−.21%, control spread .65%) with +.30% logical reads; age 1 remained
7.40% slower. It remains a private experiment, not part of the preset. Do not
claim that using resident substitutes implies asynchronous ideal-expert arrival
has been integrated or proved beneficial.

## Replication and credit

Freeze native before/after, sweep the chosen substitution caps, and log actual decoder
and prefill substitutions. Compare warm append AND cold bootstrap, fixed-teacher
ΔNLL/KL, and first-attempt complete tool tasks. Count requested reads separately
from physical I/O. Re-run a growth case with the same expert-cache capacity.

The broader cache-aware routing idea is credited to
[MoCCE](https://arxiv.org/abs/2412.00099). The local `ds4-recipe` chapter 06
at revision `0316b956` describes a different Cache-Prior policy: it adds
`λ` times a per-layer running mean of the corrected score range to resident
experts and the original top `keep`, then selects the top six again. It
renormalizes the selected experts' unbonused affinities into mixture weights.
That policy preserves resident original picks and protected top picks, but has
no 50% replaced-mass cap. Our preset bounds changes to the five weakest
original slots and retains their original coefficients.

That chapter recommends `λ = .09`, `keep = 3`; its reported speed and quality
study used `λ = .10`, `keep = 2`. Its AMD/HIP measurements are not measurements
of this Metal implementation. It also motivated our warm-prefill investigation;
see [ds4-recipe chapter 06](https://tangled.org/astrra.space/ds4-recipe/blob/0316b956/06-expert-cache-and-routing.md).
