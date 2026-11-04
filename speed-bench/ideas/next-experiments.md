# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can shared-expert overlap shorten selected prefill?**
First one-hour slice: November 4, 08:37–09:37 Eastern. Refresh the 32K profile,
inspect the routing/shared-work dependencies and existing batch-overlap path,
then compare one candidate at the ordinary 8K and larger 32K frontiers. Include
fresh prefill, a 512-token append, full decode windows and numerical checks.
Use five alternating timing pairs per side; keep profiling separate from timing.
The first profile uses the existing overnight driver with all adopted settings.
Raw evidence stays under `.workspaces/hillclimb-20261104/01-prefill-overlap/`.

Record one result report and a short
[session journal](../../docs/lab/journal/2026-11-04-through-06-performance.md)
entry, update the owning manual if adopted, then select the next target from
measured dependencies. Periodically compare the accumulated package against
its starting runtime; reserve the final session slice for combined qualification.

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
