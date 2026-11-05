# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: should the new prefill options join the existing fast presets?**
One-hour slice: November 4, 23:13 through November 5, 00:13 Eastern. Both arms
use runtime `909faab5` at checkout `7a43e7eb`, the same current benchmark binary,
64 GB expert cache and 8,192-token chunks. Keep the ordinary CPU resident policy
and qualified RB4/unit-rescale settings fixed. GPU resident routing and its
dependent options remain off in both arms.

The candidate enables only `DS4_METAL_V41_PREFILL_ROPE_BF16=1` and
`DS4_METAL_V41_PREFILL_INDEX_Q_MPP=1`. The first preserves tested values and
removes a local transform pass; the second changes the temporary indexer-query
reduction and has measured numerical drift. Their prior timing held experimental
GPU routing enabled. The recent recorded-session comparison did not exercise the
query matrix path, so it cannot decide this separate preset choice.

Take five alternating profile-free pairs at 8K and 32K, separate full-vocabulary
quality pairs, and one controlled reasoning-enabled long-prompt coding pair.
The two presets differ in prefill substitution: `resident-fast` leaves it off,
while `persistent-fast` enables it. Add five 8K pairs and a 32K quality pair with
the prefill gap set to zero before deciding whether to enable both presets.
Reuse the existing drivers and oracle; match task path, time, sampling and fresh
prefix-cache conditions. Keep first-step and complete phase timings, all samples
and variation. Preserve any coding failure. Do not equate mean distribution
drift with task accuracy or reject an approximation merely for differing output.

The outcome is an explicit adoption decision for the two prefill selectors.
If the combined evidence supports them, enable them in the two resident fast
presets, preserve explicit zero overrides, build the normal binaries and install
the qualified local package. Otherwise leave the selectors available explicitly
and record why. No GPU policy promotion or published-main advancement is part
of this slice.

Keep one guarded model job and pause builds during final timing. Write one report
under `2026-11-04-prefill-preset-qualification/`, with raw artifacts under
`.workspaces/hillclimb-20261104/20-prefill-preset-qualification/`, then update the
owning manual, recipe conclusion and a short journal entry.

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
