# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can resident GPU routing reuse maintained address tables?**
One-hour slice: November 4, 14:48–15:48 Eastern. The fixed local baseline is
resident-handoff prototype `6549a33b`, with GPU routing enabled and ordinary
refresh every eight positions. Its [report](../experiments/2026-11-04-resident-handoff/README.md)
records neutral fresh and slower warm 32K decode, alongside a substantial
reduction in the diagnostic routing handoff. CPU sampling found 85 of 1,278
main-thread observations in redundant address publication/clearing.

Remove only those repeated writes: cache installation and eviction already
maintain the address tables under the selected streaming configuration. Preserve
identity validation, resident masks, every in-flight pin, GPU route policy and
refresh cadence. Leave duplicate Metal resource declarations for a separate
question. First verify the cache lifetime invariant and fixed-history equality
against the original GPU prototype. Then use five alternating samples per side
for the preparation change and for the combined path against ordinary routing.
Keep first-token and full-window costs, inspect changed dependencies even if
total time is neutral, and qualify ordinary 8K as well as 32K work.

Raw evidence belongs under
`.workspaces/hillclimb-20261104/10-resident-address-reuse/`. Reuse the existing
driver with explicit recorded settings. Only preparation/review serving this
outcome may run in parallel; retain one guarded heavy job. Write one result
report and one short [journal](../../docs/lab/journal/2026-11-04-through-06-performance.md)
entry, keep raw output ignored, and do not publish.

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
