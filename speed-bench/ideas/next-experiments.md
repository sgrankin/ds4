# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: does the accumulated package improve recorded agent work?**
One-hour slice: November 4, 22:27–23:27 Eastern. Compare published `364ceeab`
with runtime `909faab5`, using the newly rebuilt replay binary, the same model,
64 GB expert cache and 8,192-token prefill chunks. Keep each binary in its
matching source root. The candidate combines the qualified RB4/unit-rescale
settings with GPU resident routing, address reuse, concurrent shared/routed
experts, refresh sixteen, SIMD selection, router fusion, prefill BF16/RoPE fusion
and the temporary indexer-query matrix projection. No runtime change is selected
in this slice.

Use the existing recorded-session driver for five alternating profile-free
pairs and a separate full-vocabulary teacher-logit pair. Retain every phase,
first-step samples, outliers and order effects. Check one additional controlled
reasoning-enabled coding task with the existing independent oracle; match its
working directory, prompt, time, sampling and fresh prefix-cache conditions.
This behavior check is not a task-latency comparison or a general accuracy
estimate. The recent long-prompt task remains separate attribution evidence.

The outcome is a package qualification decision, not the sum of prior isolated
percentage gains. Numerical differences are measured tradeoffs. If replay is
neutral, retain useful local improvements and select the next dependency target
from the profile rather than declaring their work useless. Do not change presets,
installed binaries or published main during measurement; any promotion needs a
concrete conclusion in the report.

Keep one guarded model job and pause builds during final timing. Write one report
under `2026-11-04-package-requalification/`, with raw artifacts under
`.workspaces/hillclimb-20261104/19-package-requalification/`, then add one short
journal entry and update an owning manual only if the qualification changes it.

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
