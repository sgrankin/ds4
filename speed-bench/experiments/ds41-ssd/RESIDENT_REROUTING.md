# Bounded resident rerouting during scalar decode

This experiment deliberately changes the routed FFN computation to avoid a
small number of logical expert-cache misses. It remains disabled by default.
The existing prefix/sync/prefill path stays exact, including one-token tails.
Only scalar `ds4_session_eval` work with output logits enables the intervention.

Enable with `DS4_METAL_V41_RESIDENT_ROUTE_GAP=0.05`. Unset or `0` retains native
execution and tensors without re-normalizing any weights. The optional
`DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK` defaults to 12, and accepts 7 through 384.
Variant 2 also exposes `DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS`, defaulting to
1/6, for the original weakest expert's share of the six unbiased affinities.
It accepts 0 through 1; 0 abstains on every layer. Values above 1/6 provide no
extra allowance on the default one-substitution path, which always targets the
weakest contribution. In the two-substitution mode, the second weakest original
contribution can exceed 1/6, so the per-expert cap still applies independently.
Variant 4 adds `DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS=2` to allow up to two
substitutions. The default remains 1 and calls the unchanged variant 3 selector.
`DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS` defaults to 0.20 and limits the
sum of original replaced affinity shares when MAX_SUBS is 2; it accepts 0
through 1. The total budget is ignored by the one-substitution path.
Positive gaps require single-rank Metal SSD text decode, without quality,
imatrix, vision, route-oracle or gate/async-prefetch modes. Unsupported enabled
configurations fail explicitly.

For each layer, start with the native six IDs and unbiased affinities
`p = sqrt(softplus(router_logit))`. Find the weakest of the six contributions.
If that expert is already resident, do nothing. Otherwise select the best
resident expert outside the native set satisfying both bounds:

```
(old_p + old_bias) - (new_p + new_bias) <= GAP * old_p
(1 - GAP) * old_p <= new_p <= (1 + GAP) * old_p
old_p / sum(original six affinities) <= MAX_MASS
```

Reject it when its rank under the original corrected selection scores exceeds
MAX_RANK. Substitute only this one slot, preserving the other five original
expert identities. Recompute every mixture weight from the original unbiased
affinities of the final six unique experts, normalized to total scale 1.5.
The learned correction bias affects selection, not mixture coefficients.
The two-sided affinity guard prevents a negative correction bias from hiding
a much larger replacement coefficient. The mass guard independently limits
how much original contribution is changed; screening values are 0.10, 0.12 and
0.14. These bounds constrain coefficient perturbation, not expert output error.
No cache bonus changes the coefficients, and no original expert is loaded in
the background solely because it was replaced.

The two-substitution mode considers only the two weakest original native
contributions in ascending affinity order, preserving the strongest four IDs.
It skips resident originals and applies the per-expert affinity/rank/mass
limits plus the shared total mass budget. The second candidate excludes the
first fallback and all six original IDs; fallbacks never become replacement
targets. If only the second weak original is missing, it can be substituted
while the weakest original remains. The final six affinities are normalized
once, after both choices, and the loader protects both fallback IDs.

The host intervention runs after the existing routing command completion,
before selected demand loads and routed kernels. Both GPU selected IDs and
weights are rewritten together before submitting further work. The loader's
protected selected set includes the new resident expert, and existing buffer
in-flight tracking protects every referenced resource. There is no extra GPU
wait or CPU readback: the probability buffer already resides in shared memory.
Cache snapshots and CPU candidate/rank scans are additional host work and must
be charged to measured time.

Variant 3 checks residency for just the original six IDs first, using the same
read-only cache identity predicate. It counts all native misses, then skips the
384-entry snapshot when the weakest original expert is already resident or
fails MAX_MASS. Eligible layers retain the complete variant 2 selector. There
are no new GPU waits, readbacks, cache-hit/hotness updates or load operations.
The added `full_snapshots` exit counter reports how many layers needed the
complete scan. Eligible layers pay six extra identity checks; ineligible ones
replace 384 checks with six.
For MAX_SUBS=2, the cheap guard checks eligibility of both original weak slots.
The additional `double_layers` counter counts layers with two substitutions;
substitutions and logical skipped-load/byte counters count both loads.

The stderr `ds4: resident-route` exit summary reports layers inspected,
substitutions, native logical misses before substitution, skipped logical loads
and their expert bytes, and mean replaced mixture mass, normalized score gap,
affinity ratio and original candidate rank. These are logical cache metrics;
skipped_bytes does not prove equivalent physical SSD traffic savings. The
replacement can alter downstream routes and future cache residency.

## Verification and evaluation

`make -j2 tests/test_resident_route && ./tests/test_resident_route` passes focused
CPU tests for strict score/rank rejection, zero-mode abstention, unique routes,
protection by unbiased contribution rather than slot order, and normalized
unbiased weights despite learned correction bias. Warning-free Metal C build
and links for ds4, ds4-agent, replay and the evaluation dylib pass. Unchanged
objects were copied only after source byte-identity checks against the base
checkout; ds4.o was rebuilt from this experiment.

Variant 1 is frozen at change `rnoqswlpsrnu`, commit `4448f4626222`, and its
binary/object copies plus SHA-256 manifest live in `/tmp/ds4-resident-v1`.
Variant 2 is a separate child experiment adding the two-sided affinity and
maximum mass guards. Additional CPU tests cover both guards and the failure
case of corrected-score neighbors with very different unbiased affinity.
Its sources, objects, binaries and build evidence are frozen under
`/tmp/ds4-resident-v2`, from child commit `16d3e5d25e15`.
Variant 3 is another child, adding only the cheap guard and its cache snapshot
API. A 50,000-case differential CPU check against the frozen variant 2 selector
passes both with and without `-ffast-math`: decisions, change fields and final
weights match exactly. GPU output equality still requires the matched replay.
Variant 4 is a separate child of variant 3. Its CPU tests cover summed mass
budget rejection, unique fallbacks, protection by contribution under shuffled
slot order, prohibition of replacing an already chosen fallback, rank limits,
and final normalization. A 50,000-case differential check preserves variant 3
one-slot decisions/change fields/weights exactly, with and without fast math,
and checks two-slot uniqueness, original-target identity and mass invariants.
Its runtime still needs mode-off and MAX_SUBS=1 matched GPU parity checks.

The runtime lifecycle review confirmed protection during buffer allocation in
`ds4_gpu_stream_expert_cache_begin_selected_load`, and GPU in-flight resource
marking before the existing slot kernels. Prior pending loads finish at the
preceding routed operation; no speculative service worker runs in the supported
configuration. Prefix calls pass no output logits and reset the intervention.
An unchanged or abstaining layer never rewrites weights. Substituted layers
use the native mathematical weight formula with a serial CPU summation, whose
float rounding can differ from the GPU reduction tree.

Initial GPU validation is coordinated separately to avoid concurrent timings.
Use a mode-off comparison for exact prefix/logit/state parity, then teacher-forced
delta-NLL and KL on frozen continuations, held-out tasks, free generation and
complete tool workflows. Compare full-turn latency and device reads alongside
decode-only speed; quality outcomes must accompany approximate throughput.
Candidate threshold examples are 0.02, 0.05 and 0.10, with MAX_RANK 12. Parameters
selected on these tasks are exploratory until evaluated on fresh whole tasks.

Primary precedent: [Mixture of Cache-Conditional Experts for Efficient Mobile
Device Inference](https://arxiv.org/html/2412.00099v2). It supports investigating
careful low-contribution substitutions; its tested models do not establish
DeepSeek V4.1 Flash Q2 quality or Apple Metal throughput.
