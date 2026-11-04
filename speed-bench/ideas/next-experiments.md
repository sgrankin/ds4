# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: does the resident GPU package help agent work enough to promote?**
One-hour slice: November 4, 17:26–18:26 Eastern. Keep runtime `1b692d25`
and the selected configuration fixed: GPU resident routing, address reuse,
concurrent shared/routed experts and an ordinary-route refresh every sixteen
positions, with the existing 64 GB cache. The
[refresh report](../experiments/2026-11-04-resident-refresh/README.md)
finds repeatable fresh decode gains, smaller and noisier warm gains, and
measurable fixed-history quality drift. Presets and installed binaries remain
unchanged until this qualification has an outcome.

Compare the complete candidate package against published starting revision
`364ceeab`, using its preserved checkout and original final settings. This is
also the periodic check against the published baseline; attribution to the
resident route remains in the preceding experiments. Build the replay binary
from the current runtime before measuring, and run each binary in its matching
source root so it loads the correct Metal code.

Use five alternating, profile-free pairs of the recorded coding-session replay.
Report prefill, decode, complete inference phases and process wall time with
all samples, variation and order effects. Run one full-vocabulary replay pair
for delta-NLL, KL and tails on its longer, changing history. Reuse the existing
replay driver and analysis. No new runtime optimization belongs in this slice.

Run the existing interval-merging agent task with reasoning enabled, including
its list, iterator, generator and input-mutation checks. Add one small,
independently checkable coding task only if it fits the hour without new harness
machinery. Use the published and candidate agents with explicit settings and
no encoder timeline. These are behavior checks, not comparative task-speed
claims. Keep tool choices, errors and retries visible; do not require identical
outputs or reject an approximation solely for nonzero divergence.

At the boundary, decide whether the measured tradeoff warrants enabling the
package in the resident fast presets, needs another specific qualification, or
should remain opt-in. If promoted, preserve explicit overrides, update the owning
manual/recipe and install only the qualified binaries. Do not publish or advance
main. Raw evidence belongs under
`.workspaces/hillclimb-20261104/13-resident-agent-qualification/`. Keep one guarded
heavy job and pause builds and unrelated analysis during latency measurements.
Only preparation or review for this outcome may run in parallel. Write one report
and one short [journal](../../docs/lab/journal/2026-11-04-through-06-performance.md)
entry, keeping generated output ignored.

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
