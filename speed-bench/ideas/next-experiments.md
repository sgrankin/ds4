# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can we remove the first-decode admission delay?**
One-hour slice: November 5, 00:49 through 01:49 Eastern. Start from runtime
`909faab5` at closure `0dff6c76`, holding the selected experimental GPU package
(including four-layer submissions), model, 64 GB expert cache and 8K chunks fixed.
The latest normal-grouping trace still spends about 0.8 seconds on the first
scalar evaluation after fresh prefill. This is not request time to first token.

First identify the exposed delay using existing command, expert-cache and read
profiles. The earlier 4 GiB versus 512 MiB slab experiment already traded a
shorter first step for slower later decode; do not repeat it. Prefill already
seeds cache slots through GPU blits, so do not assume this is their first GPU use.
Inspect compute resource admission, cache lifetimes and the difference between
first, ordinary and refresh tokens. Keep timestamp-domain and profiler limits
explicit. Then select one small intervention supported by the evidence.

Qualify any candidate with at least five alternating pairs, fresh and warm
phases, fixed-history logits, total inference and process wall. An improvement
must remove or overlap cost, not merely move it into startup, prefill or later
steps. Neutral results can be useful if they remove a dependency that the changed
profile exposes; record that evidence before retaining or rejecting the change.
Keep runtime behavior opt-in and preserve expert-cache pins, the admission lock
and pressure guard. One heavy model job, no unrelated build during timing.

Keep raw artifacts under `.workspaces/hillclimb-20261104/22-first-decode-admission/`
and one concise report under `2026-11-05-first-decode-admission/`. Update the
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
