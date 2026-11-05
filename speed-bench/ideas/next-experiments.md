# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: does the accumulated selected package help a recorded agent session?**
One-hour slice: November 5, 05:04 through 06:04 Eastern. The
[cached-refresh experiment](../experiments/2026-11-05-cached-refresh-parallel/README.md)
retains its prototype opt-in but does not add it to the selected package.
Qualify the experiment 25 package against published `364ceeab` before another
runtime change. The current runtime is `c1982bc8`, recorded by closure
`5c87f08e`; leave cached-refresh parallelism off. Use the existing
[package settings](../experiments/2026-11-05-slab-event-interaction/candidate-settings.json)
and the published baseline settings from the overnight decode report.

Reuse the recorded `session-v1.txt` replay with five alternating timing pairs,
then compare all 3,371 aligned full-vocabulary teacher rows. Keep initial
prefill, appended prefill, complete decode, first-step behavior, total inference
and process wall separate. Report every sample, variation and order effects.
This trace fixes model tokens and tool results; it does not measure autonomous
agent speed or large-context prefill. The existing interval-coding oracle is
the primary practical check, preserving reasoning and the earlier adjacency
failure. Run the existing long-background dependency-layer task if the budget
allows, to exercise a full prefill chunk. Retain failures and noncompletion.

Rebuild only the candidate replay and agent binaries. Execute both arms from
their matching source roots, with explicit settings and recorded hashes. Keep
one guarded heavy job, no builds during timing, and existing pressure checks.
Reuse the prepared local drivers under
`.workspaces/hillclimb-20261104/28-long-session/`; do not add another harness.
Keep one report under `2026-11-05-long-session-qualification/`, with a short
journal and owning-manual update at closure. Useful follow-up on the cached
refresh preparation/submission dependency remains deferred until this result.
No preset change, installed frontend replacement, main advancement or push is
part of this qualification.

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
