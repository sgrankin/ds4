# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Completed question: include cache-seed writes in command-buffer residency.**
The [seed-write comparison](../experiments/2026-11-05-seed-residency/README.md)
uses runtime `d4c0d297`. The new helper call covers all 40 recorded seed buffers;
the large historical first-compute gap is absent in this comparison. Five pairs
at both contexts still show mixed phase costs without establishing an overall
gain over queue residency. All 768 paired quality rows match, including a
smaller-cache eviction case. Retain the extended mode as unselected groundwork.
The experimental package stays on queue mode 1; presets leave slab residency
off. Installed frontends and main remain unchanged.

The narrowed association is now ready for a bounded test of shared-expert
overlap. Select that next question and its budget before starting another job.
Current evidence and scope limits belong to the report; local artifacts are in
`.workspaces/hillclimb-20261104/30-seed-residency/`.

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
