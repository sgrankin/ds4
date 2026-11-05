# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can exact cached CPU selections use parallel shared/routed FFN?**
One-hour slice: November 5, 04:09 through 05:09 Eastern. The
[submission-removal experiment](../experiments/2026-11-05-refresh-shared-flush/README.md)
did not establish a speed gain and its standalone selector was removed. It did
identify a real batch-boundary obstacle to parallel FFN. Removing that boundary
alone is insufficient: the existing parallel path assumes a GPU-selected,
resident-only set whose possible entries are already pinned.

Implement one opt-in path for the normal CPU-selected six experts only when
all selected weights are already cached. Preserve the CPU routing policy,
selected-ID upload, native mixture weights, normal cache accounting and pins.
Establish the exact-selected proof before consuming cached addresses; do not
reuse the broader resident-only override or merely widen a Boolean. Misses,
split/staged loads and unsupported modes must keep the existing serial path.
Use the existing shared/routed two-stage join, without a separate flush that
would reset its state. Document the changed shared-down/HC join boundary and
qualify numerical behavior. Keep the interface narrow and the experiment small.

Baseline runtime source is `8bf87bbd`, restored by closure `b479686a`, with the
experiment 25 candidate settings: queue residency on, shared-event overlap off,
GPU resident routing, refresh interval 16, four-layer submissions, requested
64 GB cache, 4 GiB slabs and 8K chunks. Use the same new binary with the new
selector off/on. Prove activation and serial fallback in a bounded diagnostic;
do not infer either from the environment flag alone.

Use fixed-history quality and five alternating timing pairs at 8K and 32K,
retaining first-step, complete decode, prefill/append, inference and wall scopes.
Include the requested 16 GB cache case, which previously admitted 958 experts
and produced actual scalar reads. Verify miss fallback and pins there; any speed
claim still requires five pairs. One guarded heavy job at a time, no builds
during timing. Profile neutral results for shifted dependencies before deciding.
Reuse existing drivers and stop at an answer or the slice boundary.

Keep raw artifacts under `.workspaces/hillclimb-20261104/27-cached-refresh-parallel/`
and one concise report under `2026-11-05-cached-refresh-parallel/`. Update the
owning manual, recipe conclusion and one short journal entry at closure. The
next slice is reserved for periodic long-session qualification against published
`364ceeab`; do not keep adding experiments before that comparison.
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
