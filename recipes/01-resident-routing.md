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
substitutes before routed computation can proceed. Overlapping that decision
with independent shared-expert work is a proposed follow-up, not part of this
policy's measured saving.

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
