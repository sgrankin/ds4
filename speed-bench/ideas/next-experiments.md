# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: which part of the resident GPU package causes the coding failure?**
One-hour slice: November 4, 18:33–19:33 Eastern. The
[agent qualification](../experiments/2026-11-04-resident-agent-qualification/README.md)
found repetitive reasoning without a layers implementation. Its arm-specific
working directories also changed the prompt, so it cannot isolate the cause.
Keep runtime `1b692d25`, the model and 64 GB expert cache fixed for current-code
arms. Preserve installed binaries and presets.

Run the original topology task with reasoning enabled, temperature zero and a
fixed 2,048-token call limit. Every arm uses the identical canonical working
directory and task prompt; archive each result before recreating an empty task
directory. Compare published `364ceeab` with its original settings, the current
qualified CPU-route settings, and GPU refresh intervals four, eight and sixteen.
Run each binary from its matching source root. Reuse the independent layers
oracle and record arguments, settings, output and binary hashes. These single
behavioral trials provide no comparative performance claim.

Identify a concrete policy or numerical cause if evidence supports one; do not
assume a missing file is an algorithm failure or disable reasoning to conceal
it. A failing CPU-route arm would motivate isolating the earlier prefill settings.
A refresh-dependent result would motivate a safer routing policy. Inspect the
source in parallel for actual weight, residency or lifetime bugs, separating
those from intentional approximation. Do not build asynchronous SSD machinery.
If the controlled results resolve the question early, use the remaining slice
for a narrow corroborating behavior check, not an unrelated experiment.

Write one report under `2026-11-04-resident-behavior/`; keep raw artifacts in
`.workspaces/hillclimb-20261104/14-resident-behavior/`. Update the recipe conclusion
and one short journal entry. The later GPU profiling question remains whether
token-completion waits reflect execution or queue gaps; neutral replay timing
does not erase the demonstrated reduction in host selection handoffs.

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
