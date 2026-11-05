# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: what remaining prefill work or dependency can one small change shorten?**
One-hour slice: November 4, 21:05–22:05 Eastern. Fix runtime `b006c4cf`, the
qualified RB4/unit-rescale prefill settings, 8,192-token chunks and a 64 GB expert
cache. Keep the experimental GPU resident route, SIMD, address reuse, parallel
FFN and refresh sixteen fixed; fusion remains enabled in this experimental
package. None of those decode-only changes implies a prefill benefit.

Inspect existing profiles first and use one bounded fresh diagnostic only if
needed. By twenty minutes into the slice, select one prefill kernel or dependency
change, informed by the earlier rejected RB16 and shared-overlap attempts. If
no justified candidate emerges, retain the profile and a concrete next question.
Do not grow a profiler or change sampling and routing policy in this slice.

For a candidate, check activation and fixed-history logits, then run at least
five alternating profile-free pairs at 32K and 8K. Retain complete phases,
first-token latency, steady rates, all samples and variation. If total time is
neutral, inspect whether the intended work or dependency changed before deciding
what to retain. A later package comparison must still measure against published
`364ceeab`; no installed-binary or preset promotion is part of this slice.

Keep one guarded model job and pause builds during final timing. Record one
report under `2026-11-04-prefill-followup/`, with raw artifacts under
`.workspaces/hillclimb-20261104/17-prefill-followup/`, then update the owning
manual or recipe conclusion and one short journal entry. Publication remains
separate.

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
