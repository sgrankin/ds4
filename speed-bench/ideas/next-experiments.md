# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can resident expert addresses remove a routing handoff?**
One-hour slice: November 4, 14:05–15:05 Eastern. The local runtime baseline is
`02b7ae26`, the qualified prefill package. The four-row output-B experiment is
closed and its variant removed; its [report](../experiments/2026-11-04-output-b-rows/README.md)
records the neutral timing and unchanged dependency profile.

Recent profiles still show roughly 5–6 ms per token of uncovered time between
routing and routed experts. Earlier CPU samples were dominated by waiting, and
clock/polling changes did not help. This slice investigates avoiding the handoff
for resident work rather than tuning the wait itself. First trace existing GPU
address tables, cache pinning and route policy. Identify a bounded path that can
select and execute resident experts without host selection, with an explicit
account of cache misses, refresh and approximation. Do not map the whole model
or weaken cache lifetime rules as an incidental implementation shortcut.

If feasible within this slice, build the smallest opt-in prototype and measure
its activation, fixed-history quality and at least five alternating timing pairs.
Otherwise finish with a concrete feasibility result and a narrowly scoped next
step. A neutral total time still requires inspection of dependency changes before
adoption or rejection. Any further candidate must retain this baseline for a
combined comparison. Raw evidence belongs under
`.workspaces/hillclimb-20261104/09-resident-handoff/`.

Only parallel preparation and review serving this outcome may run. Keep one
heavy job under the pressure guard, write one result report and one short
[session journal](../../docs/lab/journal/2026-11-04-through-06-performance.md)
entry. Do not publish; continue sequential hill climbing toward the deadline.

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
