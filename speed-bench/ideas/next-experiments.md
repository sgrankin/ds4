# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can cooperative FP4 decoding shorten indexed attention?**
One-hour slice: November 4, 10:00–11:00 Eastern. The large-prefill attention
profile and the earlier main-FP4 report identify the packed half4 reader as a
remaining candidate. Inspect its repeated scale decoding and exact half-value
construction, then test one bounded reader change while preserving the current
head groups and key order. Include a focused packed-versus-expanded fixture,
full-model numerical checks and five alternating timing pairs at 8K and 32K.
Keep full decode windows to detect costs outside prefill.

The previous row-blocking selector was removed after consistent regressions at
both frontiers. The starting runtime again has the published execution for
this workload, with the earlier readiness cleanup confined to batch overlap's
existing opt-in path. Raw evidence for this question belongs under
`.workspaces/hillclimb-20261104/03-packed-attention-reader/`; reuse the existing
runner and keep intrusive profiling separate from timing.

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
