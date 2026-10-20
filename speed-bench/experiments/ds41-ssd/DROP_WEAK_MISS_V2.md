# Cheap original-weakest miss dropping

Status: source/CPU validation experiment; GPU quality and native-mode parity
remain unvalidated until root's serialized guarded run. Frozen neutral control
de747c27 is unchanged; this is a separate child, not the old recovered prototype.

Set DS4_METAL_V41_RESIDENT_ROUTE_POLICY=drop with a positive ROUTE_GAP to activate
the ordinary scalar text SSD routing path. GAP is activation-only for this
policy; it does not choose a neighbor. The default drop MAX_MASS is .10 and
MAX_SUBS must be 1. MODE=native|lookup|noop|apply retains the same native controls:
native uses original routing, lookup checks six residency identities/eligibility,
noop computes the five-contribution plan and coefficients then discards it,
and apply writes the alias ID plus five positive/one zero coefficients.

Only the weakest ORIGINAL unbiased affinity is eligible, only when missing,
within the original mixture mass cap and with another selected expert resident.
Its zero-weight sixth ID aliases the first resident original slot. The five
stronger original IDs survive; their original unbiased affinity ratios normalize
once to 1.5. No full384 cache snapshot, candidate score/rank scan, original
prefetch, asynchronous cache mutation or extra demand fetch is introduced.
The generic summary labels the alias action as a substitution and also emits
drops; score-gap/ratio/rank are zero for drops because no neighbor is chosen.

The existing early loader explicitly deduplicates missing selected IDs. The
routed binder accepts repeated IDs, binds their shared weight buffers per slot,
and protects the complete final selected set against eviction. The kernels
consume per-slot routing coefficients, so the aliased slot contributes zero.
CPU tests check the selector/coefficients; the actual Metal zero contribution
and cache lifetime require model validation. Compute still dispatches six slots.

One cache-accounting consequence is deliberately documented: existing selected
hotness and cache-hit accounting are per slot, so the resident donor can receive
two notes/hits. This can affect subsequent eviction/cache divergence; complete
trajectory quality/I/O must be checked, not just immediate skipped bytes.
Default replace and mode-off remain unchanged.

Preregister the first quality bound at mass .10, then compare .08/.12 only as a
distinct calibration stage if useful. Record drop counts/original mass, all raw
distributions and complete-phase I/O. Do not equate a zero slot with a quality
win; this is an aggressive I/O/host-overhead bound for comparison with resident
neighbor replacement. Pure timing disables PROFILE/explicit I/O atomics.
