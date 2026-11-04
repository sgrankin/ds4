# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can resident-only execution overlap shared and routed experts?**
One-hour slice: November 4, 15:40–16:40 Eastern. The fixed runtime baseline is
`cc913524`, with GPU resident routing, address reuse and ordinary refresh every
eight positions enabled. The [address-reuse report](../experiments/2026-11-04-resident-address-reuse/README.md)
finds neutral attribution throughput despite removal of the sampled preparation
work. The host mostly waits for GPU completion. The combined route retains a
warm 32K slowdown, so neither experimental feature is promoted to presets.

The existing parallel FFN machinery can overlap routed gate/up with shared
gate/up, then their down projections after a producer barrier. Current gates
exclude the streamed IQ2/Q2 V4.1 shape. Admit only the already validated,
pinned resident-address path under a new explicit opt-in. Preserve address-table
ownership, all cache pins and resource declarations, distinct output storage,
BF16 boundaries, and finish/abort cleanup. Do not use `force_resident` to bypass
streaming checks or map full model weights. Ordinary refresh/fallback keeps its
existing route. No host readback, expert load/eviction or command flush may
interrupt the concurrent scope.

Verify activation and fixed-history numerical behavior first, including ordinary
8K and 32K contexts. Use five alternating samples per side to isolate the
concurrency change and compare the combined path against ordinary CPU resident
routing. Preserve first-token and full-window costs. If timing is neutral,
inspect whether intended overlap appeared before deciding. Detailed encoder
timelines alter grouping and add host overhead; use them for structure, not as
predictions of native latency. Measure quality differences rather than rejecting
rounding changes automatically.

Raw evidence belongs under
`.workspaces/hillclimb-20261104/11-resident-parallel/`. Reuse the existing driver
with explicit settings. Only preparation/review serving this outcome may run in
parallel; retain one guarded heavy job. Write one result report and one short
[journal](../../docs/lab/journal/2026-11-04-through-06-performance.md) entry, keep
raw output ignored, and do not publish.

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
