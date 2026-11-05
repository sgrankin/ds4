# Experiment queue

The user authorized sequential decode and prefill experiments through
2026-11-06 18:31 Eastern (23:31 UTC). Start from published `364ceeab` and
preserve a baseline checkout at `.workspaces/hillclimb-20261104/starting-364ceeab`.
This authorization includes choosing the next target after each bounded result;
it does not authorize unrelated features or another push to main.

**Current question: does GPU-resident decode benefit from fewer submissions?**
One-hour slice: November 5, 00:02 through 01:02 Eastern. Use runtime
`909faab5` at checkout `bdcf85a5`, the current benchmark binary and the full
experimental GPU-resident package from the package-requalification settings.
The previous every-layer submission choice was measured with CPU routing;
removing those CPU handoffs changes the dependency graph. Its recent diagnostic
still showed approximately forty command buffers per ordinary GPU-resident token.

Hold routing, refresh interval sixteen, parallel shared/routed FFN, cache size,
packing and prefill settings fixed. Compare the existing
`DS4_METAL_V41_DECODE_FLUSH_LAYERS=1` with `2` and `4`. No new runtime selector or
kernel is needed. First check fixed-history logits, then take five alternating
32K pairs for each candidate and five 8K pairs for a useful candidate. Inspect
command-buffer counts and dependencies separately from profile-free timing.
Keep first-token, steady and complete decode costs; preserve noisy or neutral
results and check whether a removed submission exposes another useful target.

The outcome is a scheduling choice for the experimental GPU package, not its
promotion into ordinary presets. A useful exact setting may be retained even
when a noisy workload is neutral; regressions and changed logits need explanation.
Use the existing driver and pressure guard, one model job at a time. Keep raw
artifacts under `.workspaces/hillclimb-20261104/21-gpu-submit-cadence/` and one
report under `2026-11-05-gpu-submit-cadence/`. Finish with the owning manual,
recipe conclusion and one short journal entry. Do not advance main or push.

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
