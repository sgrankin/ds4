# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can refreshes avoid a separate shared-expert submission?**
One-hour slice: November 5, 03:26 through 04:26 Eastern. The
[residency/schedule combination](../experiments/2026-11-05-slab-event-interaction/README.md)
reduces fresh first-step and complete-workload time but adds a small recurring
decode cost. A full-token diagnostic places most of the fresh phase's added
time in the seven later CPU refreshes; ordinary-step variation remains too.

The overlap-off refresh path submits shared gate/up work before entering routed
expert execution. Check the data and resource dependencies, then test skipping
that flush behind one opt-in selector. Selected expert I/O already starts before
shared encoding. Removing the flush may help cache hits while losing useful I/O
overlap on misses, so retain a constrained-cache or other actual-miss check.
Do not change selected experts, numerical boundaries, cache pins or pressure
relief. Keep the old path as the control; do not change other model families.

Baseline runtime is `8bf87bbd`, closure `d0af7acc`, using the experiment 25
candidate settings: queue residency on, shared-event overlap off, GPU resident
routing, refresh interval 16, four-layer submissions, 64 GB cache, 4 GiB slabs
and 8K chunks. Use the same new binary with the selector off/on. Inspect whether
refresh command-buffer count falls from 91 toward 51 while ordinary steps stay
at 11. The native per-token timer can distinguish refresh and ordinary cost
without changing encoder grouping; it remains diagnostic.

Use fixed-history quality and five alternating timing pairs at 8K and 32K,
retaining first-step, complete decode, prefill/append, inference and wall scopes.
If the main comparison is useful, include a bounded miss-heavy qualification;
any performance conclusion there still needs five pairs. One guarded heavy
job at a time, no builds during timing. Profile neutral results for shifted
dependencies before deciding. Reuse existing drivers and stop at an answer.

Keep raw artifacts under `.workspaces/hillclimb-20261104/26-refresh-shared-flush/`
and one concise report under `2026-11-05-refresh-shared-flush/`. Update the owning
manual, recipe conclusion and one short journal entry at closure. The selected
experimental package still needs a periodic long-session comparison against
published `364ceeab`; reserve a subsequent slice for that qualification.
No main advancement or push is authorized.

[Project direction and current status](../../docs/project.md) records the goals,
implemented features and a few possible next milestones. The current timed
authorization permits choosing the next bounded performance question; outside
that authorization, choose an outcome with the user before starting new work.

When work resumes, record only the selected question, fixed baseline, one-hour
budget and decision criterion here. Keep results in one linked report and move
the completed outcome into a short journal entry rather than growing this queue.
Performance comparisons still require at least five alternating samples per
side and the existing pressure guard.

The [lab journal](../../docs/lab/README.md) records prior experiments and their
outcomes. [Design ideas](../../docs/design/README.md) preserve unselected options;
neither is an active queue.
