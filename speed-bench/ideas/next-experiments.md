# Experiment queue

The user authorized a seven-hour decode optimization session, from
2026-11-03 23:55:26 to 2026-11-04 06:55:26 Eastern. Work proceeds through
one-hour profile, target, implementation and measurement loops. This authorization
includes choosing the next bottleneck after each result; it does not authorize
unrelated features or publishing main. Reserve the final slice for combined
qualification and a concise account of adopted changes and unsuccessful attempts.

**Current loop: overlap the shared down projection with host routing.**
Started 2026-11-04 01:21 Eastern, with a one-hour budget ending 02:21.
Baseline `1efa1175` includes direct F16 gather and every-layer submission.
The new cadence profile still leaves about 1.09 ms/token of exclusive coverage
in the shared-down/HC tail, with about 6.34 ms/token available after shared
gate/up and routing finish. These diagnostic intervals are an opportunity,
not predicted savings. Move the existing Q8 shared down projection into that
host wait, then use the existing add/BF16/HC join. Preserve rounding order,
resource lifetimes and unsupported-path fallbacks; add no kernels or queues.
Compare `DS4_METAL_V41_ROUTE_SHARED_DOWN=0` and `1`, with five alternating
pairs at 32K and 8K, numerical checks including an unaligned prefix and a
tighter cache, and a separate profile. Raw evidence lives under
`.workspaces/overnight-decode-20261104/shared-down/`.

The [gather](../experiments/2026-11-04-decode-pass/f16-gather.md) and
[cadence](../experiments/2026-11-04-decode-pass/flush-cadence.md) loops are complete
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
