# Experiment queue

**Paused at the user's request on November 5, 2026.** The user asked to finish
the current experiment and report what the session achieved because the target
token window had been consumed. Do not resume automatically under the earlier
November 6 time allowance. No next experiment is selected.

The final completed question was whether pairing the ratio-two compressor's
F16 KV and score projections could shorten prefill without changing published
keys. The [result report](../experiments/2026-11-05-prefill-compressor-pair/README.md)
records the exact-output checks, five alternating pairs at 8K and 32K, profile,
variation and limits. The implementation is retained behind
`DS4_METAL_V41_PREFILL_COMPRESSOR_PAIR=1`; the report's candidate settings add it
to the experimental GPU-routing package. It is not enabled in installed fast
presets. Whole-workload improvement remains uncertain.

Published `main` remains `364ceeab`. The installed frontends and CPU-routing
fast presets retain the last qualified package; subsequent GPU-routing and
compressor changes remain local and experimental. The
[project page](../../docs/project.md) owns current goals and capability status,
and the [session journal](../../docs/lab/journal/2026-11-04-through-06-performance.md)
records adopted, unadopted and removed attempts. Historical reports and design
ideas are evidence, not an execution queue.

On an explicit resumption, choose one outcome, freeze its baseline and give it
a one-hour budget. Performance comparisons require at least five alternating
samples per side, all samples retained, and one heavy job under the existing
pressure guard. If total timing is neutral, inspect whether the profile shows
useful work reduction or an unlocked dependency before deciding what to keep.
