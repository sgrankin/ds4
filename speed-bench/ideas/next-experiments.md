# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: can queue-attached slab residency help when the shared-event handoff is disabled?**
One-hour slice: November 5, 02:40 through 03:40 Eastern. The queue-attached
experiment reduced the fresh first step but added repeated waits near shared
events; a standing request removed those extra waits but restored the original
gaps. Both results are retained. Now test the interaction using the existing
`DS4_METAL_V41_ROUTE_SHARED_OVERLAP` switch, without adding another mechanism.

Run the already-built queue-attached runtime `5efd342f` in a matching frozen
workspace, with its recorded binary hash. First take a diagnostic pair holding
queue residency on while changing shared overlap from `1` to `0`.
This switch also moves shared gate/up and down work on CPU-refresh layers;
interpret it as a schedule comparison, not a pure event toggle. Then qualify
the combined candidate (queue residency on, overlap off) against the selected
pre-residency package (residency off, overlap on), using the same binary on both
sides. Hold GPU resident routing, refresh interval 16, four-layer submissions,
model, 64 GB cache, 4 GiB slabs, 8K chunks and workload fixed.

Use fixed-history quality and at least five alternating pairs at 8K and 32K.
Inspect fresh and warm first steps, complete 128-token decode, prefill, total
inference and process wall. Separate diagnostic scope from comparative timing;
do not infer a driver mechanism from an event-adjacent gap alone. Preserve
existing cache pins, declarations, admission and pressure handling. One guarded
heavy job at a time; no builds during timing. Current source and installed
frontends need not change for this configuration experiment.

Keep raw artifacts under `.workspaces/hillclimb-20261104/25-slab-event-interaction/`
and one concise report under `2026-11-05-slab-event-interaction/`. Record which
comparison isolates overlap and which qualifies the combined package against
the original baseline. Update owning documentation and one journal entry when
closing. No main advancement or push is authorized.

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
