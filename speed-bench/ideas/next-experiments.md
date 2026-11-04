# Experiment queue

The user authorized a seven-hour decode optimization session, from
2026-11-03 23:55:26 to 2026-11-04 06:55:26 Eastern. Work proceeds through
one-hour profile, target, implementation and measurement loops. This authorization
includes choosing the next bottleneck after each result; it does not authorize
unrelated features or publishing main. Reserve the final slice for combined
qualification and a concise account of adopted changes and unsuccessful attempts.

**Current loop: use SIMD exchanges in the router's sort.** Started
2026-11-04 02:00 Eastern, with a one-hour budget ending 03:00. Baseline
`3e5f39f8` includes F16 gather, every-layer submission and shared-down overlap.
The current profile records 40 router calls and about 1.49 ms of router
duration per token, immediately before the selected-readback signal. That sum
is not predicted savings. Adapt the existing hybrid SIMD bitonic sorter to
replace intra-SIMD barriers while preserving the network's stage/comparison
order, tied expert IDs, padding and weight normalization. Keep the original
kernel as fallback and initially select only the 384-expert, top-six scalar
shape with `DS4_METAL_V41_ROUTER_SIMD=1`.

A focused Zig GPU fixture compares probabilities, ordered IDs and weights,
including boundary ties and normalization-floor cases. Then use five
alternating pairs at 32K and 8K, full-model numerical checks and a separate
profile. Raw evidence lives under
`.workspaces/overnight-decode-20261104/router-simd/`.

The [gather](../experiments/2026-11-04-decode-pass/f16-gather.md),
[cadence](../experiments/2026-11-04-decode-pass/flush-cadence.md) and
[shared-down](../experiments/2026-11-04-decode-pass/shared-down.md) loops are
complete and adopted. The [session index](../experiments/2026-11-04-decode-pass/README.md)
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
