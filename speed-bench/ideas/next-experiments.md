# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: should the unit-rescale shortcut join the fast presets?**
One-hour slice: November 4, 12:26–13:26 Eastern. Compare the accumulated
candidate against published `364ceeab`, using its frozen checkout and original
settings. The candidate adds exact four-row attention staging and the approximate
unit-rescale shortcut. The selected-prefill wait cleanup remains behind its
existing opt-in flag and is not enabled in this comparison.

Use five alternating 32K timing pairs, full-vocabulary fixed-history quality,
and a reasoning-enabled coding check with enough context capacity to avoid the
previous reserve-triggered compaction. Inspect whether the recorded coding
replay exercises the new kernel before selecting it as the normal-case workload.
Use the existing combined runner, without changing the runtime during samples.
Decide whether the practical speed and quality tradeoff warrants enabling the
shortcut in the resident fast presets. Small numerical differences and noisy
neutral timings are not automatic rejection criteria.

Raw evidence belongs under
`.workspaces/hillclimb-20261104/07-combined-prefill/`. Root compatibility binaries
and published main remain at the starting package. Record one result report
and a short [session journal](../../docs/lab/journal/2026-11-04-through-06-performance.md)
entry; update the owning manual if promoted. Then return to the next measured
prefill or decode dependency. Reserve the final session slice for another
combined qualification.

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
