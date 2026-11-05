# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Last closed question: shared-expert overlap with scoped slab residency.**
The November 5, 07:29–08:29 Eastern slice completed early. The
[report](../experiments/2026-11-05-scoped-overlap/README.md) retains the five-pair
large-cache regression, favorable smaller-cache decode, and matching quality
checks. The selected package remains queue residency mode 1 with overlap off.
The profiles localize large event-adjacent gaps to buffers that also contain
prior expert work. A checked outgoing boundary after actual CPU-routed experts
is the next candidate; it is not yet implemented or selected here.

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
