# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: which resident-routing refresh interval offers a useful tradeoff?**
One-hour slice: November 4, 16:33–17:33 Eastern. Keep runtime `1b692d25`,
GPU resident routing, address reuse and concurrent shared/routed execution fixed.
The [concurrency report](../experiments/2026-11-04-resident-parallel/README.md)
finds useful decode gains and matching logits against serial GPU routing. The
combined route improves fresh decode and is neutral in warm 32K timing, while
its substitution policy remains approximate. Presets and installed binaries
remain unchanged.

Compare ordinary CPU cap-five resident routing against GPU refresh intervals
4, 8 and 16. More frequent ordinary routing both executes its less permissive
policy and refreshes the expert cache; intervening GPU steps use only resident
experts. Keep the 64 GB cache and other settings fixed. Do not change runtime
code or mix in a cache-budget experiment.

Use five alternating performance samples per side at 32K for intervals 4 and
16. Reuse interval 8's just-completed five-pair comparison at the same runtime
and settings, clearly identifying that source; there is no new reason to repeat
it. Preserve first-token, complete-window, append and steady decode results,
all samples and order effects. Extend quality checks to 512 teacher tokens per
phase at 32K for all three intervals, against the same ordinary policy. Report
mean and tail delta-NLL, KL, top-choice changes and teacher perplexity ratio;
these are not task-accuracy scores or automatic rejection thresholds.

Plot a speed–quality tradeoff with separate fresh and warm cases, showing
variation and paired ordinary controls. If a point looks useful, qualify it at
8K with five timing pairs and a longer fixed-history quality check. Do not claim
general agent quality or promote a preset from teacher metrics alone. At the
hour boundary, choose whether to carry a point into combined package and task
qualification, rather than adding another tuning axis.

Raw evidence belongs under
`.workspaces/hillclimb-20261104/12-resident-refresh/`. Reuse the existing driver
with explicit settings and keep one guarded heavy job. Only preparation/review
serving this outcome may run in parallel. Write one report, update the recipe
conclusion and one short [journal](../../docs/lab/journal/2026-11-04-through-06-performance.md)
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
