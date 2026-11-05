# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Experiment 34 closed: unretained command buffers.**
The November 5, 10:16–11:16 Eastern slice retained the scratch lifetime repair
in `d31275e9`. All 768 repaired quality rows match; five-pair 8K and 32K timing
leaves the optional unretained/deduplication package unselected. The corrected
profile shows nearly unchanged ordinary GPU windows and changed host scopes,
without isolating callback cost. The [report](../experiments/2026-11-05-unretained-residency/README.md)
owns the failed and repaired checks, timing, resource audit and unrun comparisons.

The next bounded question is whether active cache ownership plus deferred
retirement can remove the extra per-command residency-set completion handlers.
No implementation of that follow-up has started. Main, installed frontends and
normal presets remain unchanged.

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
