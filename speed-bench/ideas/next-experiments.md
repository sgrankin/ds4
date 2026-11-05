# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Completed question: two SIMD groups for scalar Q8 output projection.**
The November 5, 13:10–14:10 Eastern budget closed early with no adoption.
The [report](../experiments/2026-11-05-output-q8/README.md) preserves the verified
activation, numerical drift, five-pair timing results and profiling limits.
The selector was removed; source matches `f7549a01`, which restores `a31acf97`.
Selected package 32 remains fixed, with deduplicated residency, unretained
command buffers and deferred set release off. No model job is running.

Review the prior row-layout experiments before selecting the next bounded
target. The timed authorization remains active; this closure does not pause it.
Presets, installed frontends and main remain unchanged.

[Project direction and current status](../../docs/project.md) records the goals,
implemented features and a few possible next milestones. The current timed
authorization permits choosing the next bounded performance question; outside
that authorization, choose an outcome with the user before starting new work.

When work resumes, record only the selected question, fixed baseline, one-hour
budget and decision criterion here. Keep results in one linked report and move
the completed outcome into a short journal entry rather than growing this queue.
Performance comparisons still require at least five alternating samples per
side and the existing pressure guard.

The [lab journal](../../docs/lab/README.md) records prior experiments and their
outcomes. [Design ideas](../../docs/design/README.md) preserve unselected options;
neither is an active queue.
