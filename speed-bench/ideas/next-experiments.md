# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Last closed question: separate CPU expert work from the next router event.**
The November 5, 08:18–09:18 Eastern slice selected the combined experimental
GPU decode package in
[candidate-settings.json](../experiments/2026-11-05-cpu-expert-boundary/candidate-settings.json),
runtime `b7e7af9d`. The [report](../experiments/2026-11-05-cpu-expert-boundary/README.md)
records direct five-pair comparisons at 8K and 32K: about 2% faster fresh
complete decode, mixed whole-workload time, and a 32K append cost. All checked
outputs match. Ordinary presets, installed frontends and main are unchanged.

The next bounded target has not yet been selected. Inspect the remaining costs
in the changed profile before choosing it; the current improvement does not
establish a whole-task gain.

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
