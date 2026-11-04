# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can we shorten attention during large fresh prefill?**
One-hour slice: November 4, 09:30–10:30 Eastern. The current 32K diagnostic
attributes about 10 seconds to `indexed_mixed_attention_heads16_dual`, the
largest individual prefill kernel total. Inspect its data dependencies and
dispatch geometry, select one bounded change, and compare it at 8K and 32K.
Include numerical checks and full decode windows so a prefill gain does not
hide a later cost. Use at least five alternating timing samples per side.

The starting runtime is `689c2d51` with batch shared overlap unset, preserving
the published runtime's execution for this workload. The preceding overlap
experiment is closed: its readiness cleanup remains in the existing opt-in
path, with no preset promotion. Raw evidence for the new question belongs under
`.workspaces/hillclimb-20261104/02-prefill-attention/`; reuse the existing runner
and keep intrusive profiling separate from timing.

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
