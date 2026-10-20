# Three original weak miss substitutions

Separate opt-in child of frozen bfcde97a. The default remains one substitution.
The existing one- and two-slot selector bodies, coefficient arithmetic, loader,
Metal kernels and cache ownership are unchanged. Backfill is not imported.
Drop remains restricted to one original weak slot.

Setting `DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS=3` considers exactly the three
weakest ORIGINAL unbiased affinities (stable native-slot tie order). It protects
the other three original IDs. Each substitution must be a native miss and a
resident candidate, obey the existing symmetric unbiased-affinity gap, learned
correction-score and global rank bounds, and meet both per-slot and cumulative
ORIGINAL mixture-mass budgets. All six original IDs and previous fallbacks are
excluded from candidate selection. A fallback is never substituted twice.
Final six IDs are unique and coefficients are normalized once from their
unmodified affinities to the native1.5 scale/floor. Strong coefficients can
change through that final normalization, as in the existing replacement policy.

Preregistered first study, fixed before any model outcome:

```
DS4_METAL_V41_RESIDENT_ROUTE_GAP=.30
DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS=.20
DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS=.40
DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK=48
DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS=3
```

Use the same frozen independent16-fixture/512-row raw reference as the broader
two-slot screen, matching its model/context/cache/chunk. Require native-before/
after raw byte parity, then measure full-distribution KL, target delta-NLL,
argmax flips, per-prompt/tail quality and actual demand requested/returned bytes.
CPU one/two parity is insufficient to establish native GPU byte parity. No
default promotion or pure speed claim follows from the initial quality screen.
Serial MAIN guard/model-slot authorization is required for every model job.

The aggregate resident report adds `triple_layers`; existing `double_layers`
retains its exact-count meaning. Committed single-layer count is
`substitutions - 2*double_layers - 3*triple_layers`, giving a zero/one/two/three
histogram together with `layers`. Existing phase-profile API ABI is unchanged.

CPU checks under actual Darwin production `-O3 -ffast-math -mcpu=native`:
100000 one/two selector comparisons against the intact frozen bfcde header,
including every changed field and normalized weight bits;50000 randomized
three-slot invariant cases; synthetic exactly-three, third-only miss eligibility,
sum/per-slot budget abstention, rank, duplicate/native-ID and tie-order cases.
The random run includes210 actual triple plans and5046 third-only first changes.
Meaningful CPU sanitizers and existing selector/drop tests are also run. Model
quality and throughput are UNVALIDATED until explicitly scheduled.
