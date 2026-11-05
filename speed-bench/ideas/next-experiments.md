# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: what limits decode after SIMD resident selection, and can one small change shorten it?**
One-hour slice: November 4, 20:17–21:17 Eastern. Fix runtime `e5d12861`,
GPU resident routing with SIMD, refresh sixteen, address reuse, parallel FFN,
the existing prefill settings and 64 GB expert cache. The
[SIMD report](../experiments/2026-11-04-resident-simd/README.md) retains the exact
execution change as an opt-in, with mixed attribution timings and no GPU-policy
promotion. Keep policy and sampling fixed during this follow-up.

First inspect a bounded 32K scalar-versus-SIMD diagnostic with native command
grouping. Existing cumulative GPU spans and host completion records can separate
submitted GPU work from host waits, but neither is a GPU-utilization counter.
Detailed command-buffer records stop after 400 rows; do not claim they cover
warm decode if prefill consumed the limit. Use existing controls and a bounded
CPU stack sample. A separate encoder timeline may rank operations, with its
known batching and logging interference stated explicitly.

By about twenty minutes into the slice, choose one concrete kernel or dependency
change from that evidence. Parallel source inspection and fixture preparation
must serve that choice. Preserve exact routing; do not add a new cache policy,
loader or profiling framework. If no justified implementation emerges, record
the profile and a specific next question rather than inventing a change.

For a candidate, check activation and fixed-history logits, then measure at least
five alternating profile-free samples per side at 32K and, if time permits, 8K.
Retain first-token, complete-phase and steady timing, every sample and variation.
If the total is neutral, inspect whether the intended dependency changed before
deciding whether it supports a useful follow-on. A later combined comparison
must still establish benefit against the original baseline.

Keep one guarded model job, builds paused during final timing, and installed
binaries and presets unchanged. Write one result report under
`2026-11-04-post-simd-profile/`, raw artifacts under
`.workspaces/hillclimb-20261104/16-post-simd-profile/`, then update the owning
manual or recipe conclusion and one short journal entry. Publication remains a
separate decision.

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
