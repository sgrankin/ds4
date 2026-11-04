# Overnight decode optimization

This session runs from 2026-11-03 23:55 to 2026-11-04 06:55 Eastern on the
M5 Max. Each loop uses one target, fixed settings, at least five alternating
timing samples per side, numerical checks and a separate diagnostic profile.
The starting runtime is `8971dfbd`. Raw evidence stays local under
`.workspaces/overnight-decode-20261104/`; the [driver](run.nu) records each
run's settings and arguments.

| Experiment | Decision and limit |
| --- | --- |
| [Direct F16 gather from packed KV](f16-gather.md) | Adopted in the resident fast presets. Less intermediate traffic and staging work; timing is neutral to slightly favorable, with exact tested logits. |
| [Decode submission cadence](flush-cadence.md) | Adopt every-layer submission in the resident fast presets. Five pairs at each of 8K and 32K improve fresh and warm decode; tested logits remain exact, including a tighter cache. |
| [Shared down during host routing](shared-down.md) | Adopted in the resident fast presets. Splitting the down/HC fusion gives a modest decode gain, with exact tested logits and no new command buffer or allocation. |

The session is still active. Results from different baselines are not additive;
a final comparison will measure the adopted combination against the starting
configuration.
