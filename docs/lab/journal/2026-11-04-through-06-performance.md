# Decode and prefill hill climbing

The user authorized sequential performance experiments from November 4 at
08:35 Eastern through November 6 at 18:31 Eastern. The published starting
runtime is `364ceeab`, which includes the five adopted overnight decode changes.
The outcome is lower decode and prefill time on this M5 Max, with measured
quality tradeoffs and no unrelated architecture work.

Each experiment gets an initial one-hour budget: profile, choose one target,
implement, compare at least five alternating samples per side, and decide.
Subagents prepare or review that same target; only one guarded heavy model
job runs at a time. Periodic comparisons against the starting runtime will
check the accumulated package. The final slice is reserved for combined
qualification before the deadline. Raw artifacts stay ignored under
`.workspaces/hillclimb-20261104/`.

## First question: shared work during selected prefill

The earlier 32K profile showed long waits around selected routed-expert work
in a 512-token append. The batch overlap experiment found only a small shared
branch available to hide, while large delays outside routed GPU execution
remained. Five-pair comparisons were slightly favorable at 8K and inconclusive
at 32K, where both arms varied substantially. One control was interrupted by
the pressure guard; its unfinished pair was retried after pressure recovered.

We retained a narrow cleanup in the existing opt-in path: resident prefill has
already made the selected IDs ready, so it need not submit and wait for another
empty command buffer. A diagnostic confirms 320 fewer empty submissions per
512-token append, with unchanged GPU operation counts. Ordinary, native-routing
and tight-cache quality checks matched exactly. The fast presets remain
unchanged; there is no claimed speedup for this cleanup alone. The
[result report](../../../speed-bench/experiments/2026-11-04-prefill-shared-overlap/README.md)
owns the measurements, failure record and qualifications. Main remains at the
starting revision.
