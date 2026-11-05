# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can SIMD shorten resident expert selection without changing its choices?**
One-hour slice: November 4, 19:23–20:23 Eastern. Keep runtime `1b692d25`'s
GPU resident policy, refresh sixteen, address reuse, parallel FFN and 64 GB
expert cache fixed. The [behavior qualification](../experiments/2026-11-04-resident-behavior/README.md)
retains a bounded completion failure and does not promote this package. This
item changes kernel execution, not that policy or sampling.

First measure the existing resident-repair kernel, which searches experts on
one GPU lane. Use existing Metal fixtures and bounded diagnostics. Implement an
explicit SIMD search preserving native slot weights, replacement order, unique
resident choices and lower-ID tie breaking. Check ordinary all-hit cases as
well as partial and all-missing cases, supported expert counts and numerical
boundaries. Repeated microbenchmarks must restore the original selected IDs;
otherwise the first repair makes later iterations trivial.

Compare scalar and SIMD operation cost with at least five alternating samples
per side. Then use five alternating profile-free 32K model pairs to separate
attribution within the GPU route from its combined result against ordinary CPU
routing. Include 8K if the slice permits, full-vocabulary fixed-history checks
and first-step/complete-phase timing. Keep every sample and variation. If the
model result is neutral, inspect whether the repair cost fell and exposed another
GPU dependency before deciding what to keep. Detailed encoder timelines alter
batching; do not treat their gaps as native latency savings.

Use a temporary opt-in for the candidate, with installed binaries and presets
unchanged until measured. Keep one guarded model job; pause builds during final
timing. Write one report under `2026-11-04-resident-simd/`, raw artifacts under
`.workspaces/hillclimb-20261104/15-resident-simd/`, and update the recipe conclusion
and one short journal entry. No unrelated quality-policy or loader work in this
slice. New experiments and publication remain separate decisions.

[Project direction and current status](../../docs/project.md) records the goals,
implemented features and a few possible next milestones. Choose an outcome
with the user before scheduling another experiment. Ideas are not work orders.

When work resumes, record only the selected question, fixed baseline, one-hour
budget and decision criterion here. Keep results in one linked report and move
the completed outcome into a short journal entry rather than growing this queue.
Performance comparisons still require at least five alternating samples per
side and the existing pressure guard.

The [lab journal](../../docs/lab/README.md) records prior experiments and their
outcomes. [Design ideas](../../docs/design/README.md) preserve unselected options;
neither is an active queue.
