# Resident routing beyond three experts

Allowing five resident substitutions is a promising extension of the existing
three-expert policy. Six reaches essentially the same measured mean latency
with substantially greater distribution drift. The existing presets remain
unchanged; the selector now accepts one through six for further qualification.

## Context and setup

The user asked whether three substitutions and a 50% replaced-mass ceiling were
necessary, and whether astrra's recipe offered another selection policy. Three
was a conservative choice, not an architectural limit. Astrra instead describes
a residency score bonus, a protected set of original experts, and recomputed
mixture weights. The [routing recipe](../../../recipes/01-resident-routing.md)
explains that distinction. This experiment retained our original coefficients
and varied the cap, mass limits and score-closeness constraint.

The one-hour slice compared seven policies on the M5 Max using the installed
1017 quant, a 48 GiB expert-cache target, a 2K seed and a 512-token warm append.
Each policy received five alternating timing samples. Separate fixed-history
logit captures measured distribution change; three reasoning-enabled retrieval
and correction checks per policy tested generated answers.

## Findings and decision

Cap five saved 7.16% on warm append plus decode relative to the existing preset,
with a saving in every pair. Complete-process savings were much smaller because
the initial full sweep remains. Six showed no reliable extra mean benefit in
this cohort, and a tighter score-gap constraint offered lower KL at higher
latency. All simple answer checks passed. Those checks exercise approximate
decode; factual quality after approximate prefill and broader agent quality
remain unqualified.

The [result report](../../../speed-bench/experiments/2026-11-02-resident-six/README.md)
owns the curve, all 35 timing samples, quality values, activation counters,
commands and variation. Keep the larger caps available, without promoting a
preset. A broader agent-append evaluation is the useful next qualification;
the full Cache-Prior algorithm remains an independent implementation choice.
