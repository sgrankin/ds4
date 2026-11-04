# Experiment queue

The user authorized a seven-hour decode optimization session, from
2026-11-03 23:55:26 to 2026-11-04 06:55:26 Eastern. Work proceeds through
one-hour profile, target, implementation and measurement loops. This authorization
includes choosing the next bottleneck after each result; it does not authorize
unrelated features or publishing main. Reserve the final slice for combined
qualification and a concise account of adopted changes and unsuccessful attempts.

**Current loop: direct F16 gather from packed main KV.** Baseline `8971dfbd`
includes the adopted short Q8 query kernel. The profile shows F32 gathers
followed by attention's F32-to-F16 conversion. Test the existing
`DS4_METAL_V41_PACKED_F16_GATHER=1` path, which also changes KV staging,
against explicit `0`, holding all other adopted settings fixed. Check focused
GPU correctness and full-model logits; measure five alternating pairs at 8K
and 32K and use a separate profile to verify the removed work. First loop
budget ends 00:55:26 Eastern. Raw evidence stays under
`.workspaces/overnight-decode-20261104/`.

[Project direction and current status](../../docs/project.md) records the goals,
implemented features and a few possible next milestones. Choose an outcome
with the user before scheduling another experiment. Ideas are not work orders.

When work resumes, record only the selected question, fixed baseline, one-hour
budget and decision criterion here. Keep results in one linked report and move
the completed outcome into a short journal entry rather than growing this queue.
Performance comparisons still require at least five alternating samples per
side and the existing pressure guard.

The [lab journal](../../docs/lab/README.md) records prior experiments and their
outcomes. [Design ideas](../../docs/design/README.md) preserve unselected options;
neither is an active queue.
