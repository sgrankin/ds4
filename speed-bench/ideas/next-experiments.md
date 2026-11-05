# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can a standing slab-residency request preserve the fresh-step gain without refresh stalls?**
One-hour slice: November 5, 02:04 through 03:04 Eastern. The queue-attached
candidate at runtime `5efd342f` reduced the fresh first step but slowed complete
decode in every pair at 8K and 32K. Its profile replaced large gaps between
command buffers with repeated waits near shared-event boundaries. Normal encoder
grouping confirmed the shift; the driver mechanism remains unresolved.

Change only the residency policy: replace queue attachment with one standing
`requestResidency` after initial set creation. Later slab additions still commit
into that set. Keep existing resource declarations, pins, selected-ID overlap,
cache policy and pressure teardown. Remove the unused queue-attachment state
rather than adding another permanent mode. The existing flag remains opt-in;
its value `1` now selects the candidate policy at the new recorded revision.
Apple exposes standing requests separately from queue attachment; do not assume
that either guarantees residency under all memory pressure.

Use the unchanged experimental GPU settings from
`2026-11-05-first-decode-admission/candidate-settings.json`, with the new flag
`0` versus `1`. Hold the model, 64 GB cache, 4 GiB slabs, 8K chunks and four-layer
submissions fixed. Run the bounded cache-lifetime fixture, fixed-history quality,
and at least five alternating pairs each at 8K and 32K. Check fresh and warm
first steps, full decode, prefill, inference totals and process wall. Compare the
profile with the previous candidate to determine whether the new wait disappears
or simply moves. No unrelated builds during timing; one guarded heavy job.

Keep raw artifacts under `.workspaces/hillclimb-20261104/24-standing-slab-residency/`
and one concise report under `2026-11-05-standing-slab-residency/`. Update the
owning storage manual, recipe conclusion and short journal entry. No main
advancement, push or preset promotion is authorized by a diagnostic result alone.

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
