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
in a 512-token append. Scalar shared-expert overlap is already enabled, while
the analogous batch path has a separate opt-in setting. We are refreshing the
current timeline and checking its resource lifetimes before selecting a change.
The [earlier profile](2026-11-03-post-rope-profile.md) explains the observation;
it does not establish the cause or a speedup from this next experiment.

No new result has been adopted yet. Main remains at the starting revision.
