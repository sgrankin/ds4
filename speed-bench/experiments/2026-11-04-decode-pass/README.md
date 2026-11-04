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
| [SIMD exchanges in router sorting](router-simd.md) | Unadopted. Three five-pair batches, including longer decode, show no consistent throughput gain. This is not a demonstrated regression; the prototype remains in its experiment commit. |
| [RoPE coefficients in the short query](query-rope-table.md) | Adopted as a small reuse change in the resident fast presets. Tested outputs are exact and timings are neutral to slightly favorable; the profile does not establish a shorter exposed query tail or a specific speed gain. |
| [Clock checks during event polling](clock-stride.md) | Unadopted. Fewer deadline checks did not improve decode; stride 64 was slower in all five warm 32K pairs. The original loop remains. |
| [Reuse selected KV within a layer group](selected-kv-reuse.md) | Adopted in the resident fast presets. Removes 30 duplicate gathers per token without new state; tested logits are exact, with small mixed timing differences. |

The session is still active. Results from different baselines are not additive;
a final comparison will measure the adopted combination against the starting
configuration.
