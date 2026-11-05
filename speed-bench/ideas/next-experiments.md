# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Last closed question: deduplicate active-batch residency association.**
The November 5, 09:16–10:16 Eastern slice retained runtime `47e461b0` as an
opt-in, unselected prototype. The [report](../experiments/2026-11-05-residency-once/README.md)
records fewer actual calls with unchanged associated buffers and checked
outputs, but mixed primary and frozen-build timings. No consistent net gain
or newly exposed GPU stall was established.

The selected package remains
[experiment 32](../experiments/2026-11-05-cpu-expert-boundary/candidate-settings.json).
A bounded unretained-command comparison is the next candidate because that
path also removes repeated completion captures. It has not yet been selected
here; source lifetime checks must precede its model runs. Normal presets,
installed frontends and main are unchanged.

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
