# Experiment queue

The user authorized a seven-hour decode optimization session, from
2026-11-03 23:55:26 to 2026-11-04 06:55:26 Eastern. Work proceeds through
one-hour profile, target, implementation and measurement loops. This authorization
includes choosing the next bottleneck after each result; it does not authorize
unrelated features or publishing main. Reserve the final slice for combined
qualification and a concise account of adopted changes and unsuccessful attempts.

**Current loop: decode submission cadence.** Started 2026-11-04 00:33 Eastern,
with a one-hour budget ending 01:33. Baseline `cc3b5bbc` includes direct F16
KV gather. The profile shows that routed work submitted at a periodic layer
flush starts sooner than work retained until the next router commit. Test
`DS4_METAL_V41_DECODE_FLUSH_LAYERS=2` against `1`, then `0` as an opposite
control. These settings predict 61, 80 and 41 command buffers per token; fewer
buffers are not themselves the objective. Keep every router readback, expert
retirement fence and the final head drain intact. Use five alternating 32K
pairs per comparison, qualify the selected point at 8K, check logits and an
unaligned tail, and inspect a separate profile. The same pressure guard and
one-heavy-job rule apply. Raw evidence lives under
`.workspaces/overnight-decode-20261104/flush-cadence/`.

The [first loop](../experiments/2026-11-04-decode-pass/f16-gather.md) is complete
and adopted. The [session index](../experiments/2026-11-04-decode-pass/README.md)
links completed results.

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
