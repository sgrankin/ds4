# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Completed question: associate residency only with expert computation.**
The November 5, 06:00–07:00 Eastern budget produced runtime `cd42c5ac` and
the [command-buffer residency report](../experiments/2026-11-05-scoped-residency/README.md).
Keep the compute-only mode unselected. Five pairs at both context sizes show a
longer fresh first step and complete generation, with nearly unchanged steady
decode. All 768 paired quality rows match, including a smaller-cache eviction
case. The selected experimental package, presets and installed frontends remain
unchanged.

The trace confirms a useful scope difference: fresh-prefill seed blits lack
association, and the extra gap precedes the first associated expert buffer.
This supports a bounded follow-up that covers seed writes, without establishing
whether it will remove cost or move it into prefill. Select that next question
and its budget before changing runtime source. The current report owns the
complete evidence and causal limits; raw artifacts remain local under
`.workspaces/hillclimb-20261104/29-scoped-residency/`.

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
