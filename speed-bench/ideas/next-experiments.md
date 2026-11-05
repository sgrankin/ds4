# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can bounded streaming-slab residency remove first-decode gaps?**
One-hour slice: November 5, 01:27 through 02:27 Eastern. Start from runtime
`909faab5` at closure `7435eca0`, holding the selected experimental GPU package
from `2026-11-05-first-decode-admission/candidate-settings.json` fixed. The model,
64 GB expert cache, 4 GiB slabs, 8K chunks and four-layer submissions remain fixed.
Early address dispatch did not remove the roughly 0.8-second first scalar
step after prefill. Resource admission is a hypothesis, not an established cause.

Test one opt-in, engine-owned Metal residency set containing only the actual
streaming slab buffers. Retain existing resource declarations and cache pins.
Release the set before cache pressure relief or trimming so it cannot defeat
memory reclamation; clear it during engine teardown. Do not map the whole model,
add a warm-up kernel or change the resident-expert policy in this slice.
The shared Metal queue means co-resident helper behavior remains a qualification
limit. Measure full prefill, complete decode and process wall to catch shifted
cost, alongside the first-step interval and a separate dependency profile.

Use at least five alternating pairs at 8K and 32K, fresh and warm phases, plus
fixed-history logits. Neutral results can justify a bounded follow-up only when
the changed profile shows useful work or waiting removed. Preserve the admission
lock and pressure guard; run one heavy model job, with no unrelated builds during
timing. Keep runtime behavior opt-in, and close this question within the slice.

Keep raw artifacts under `.workspaces/hillclimb-20261104/23-slab-residency/`
and one concise report under `2026-11-05-streaming-slab-residency/`. Update the
owning manual, recipe conclusion and short journal entry. No main advancement
or push is authorized.

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
