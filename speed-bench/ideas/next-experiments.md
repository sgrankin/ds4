# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can the rescaling shortcut preserve the old rounding?**
One-hour slice: November 4, 12:09–13:09 Eastern. Baseline `1312c914` retains
the approximate shortcut as opt-in, outside the presets. It reduced measured
total inference time by 1.1–1.5%, with nonzero distribution drift and a passing
but eventful coding check. Removing an identity multiplication may have changed
which product the compiler contracts with an addition; that explanation remains
a source-level hypothesis, not a generated-code finding.

Try an explicit rounded KV contribution inside the shortcut, using the existing
block-local contraction controls and `fma(output, 1, contribution)`. Keep the
original four-row kernel and the first shortcut available as measured controls.
Start with the existing six-shape fixture and five alternating operation pairs
at small and large sizes. If useful, measure five full-model pairs at 8K and
32K, fixed-history quality and an activated reasoning-enabled coding task.
Use enough context capacity to distinguish the arithmetic change from the
previous small-context compaction event. Approximation remains acceptable;
select the useful speed/quality tradeoff rather than requiring perfect equality.

Raw evidence belongs under
`.workspaces/hillclimb-20261104/06-attention-contribution-rounding/`. Reuse the
runner and fixture. Four-row staging remains enabled in the local resident fast
source presets; the rescaling shortcut is not. Root compatibility binaries and
published main remain at the starting package until combined qualification.

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
