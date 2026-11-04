# Overnight decode optimization

This session used the authorized window from 2026-11-03 23:55 to
2026-11-04 06:55 Eastern on the M5 Max. Eight experiments are complete, with
five changes adopted. Each loop used one target, fixed settings, at least five
alternating timing samples per side, numerical checks and a separate diagnostic profile.
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
| [Raw clock for event polling](raw-clock.md) | Unadopted. The 8K means improve, but all five warm 32K pairs favor the original clock. The simpler existing path remains. |

The [combined qualification](combined.md) measures the adopted package against
the starting runtime. Five alternating pairs at 32K improve steady throughput
by about 7–8%; the recorded coding replay takes 4.2% less inference time.
All 4,651 checked full-vocabulary rows match, and both builds pass a fresh
reasoning-enabled coding smoke check. The report retains absolute timings,
variation, startup and prefill costs, and the limits of these quality checks.
Individual experiment percentages are not additive. Main remains unpublished.
