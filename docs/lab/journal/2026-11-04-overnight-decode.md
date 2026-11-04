# Overnight decode experiments

The seven-hour session completed eight experiments and retained five changes
in the resident fast presets. Its final comparison measures the whole package
against the starting runtime, with five alternating pairs per workload. The
[combined report](../../../speed-bench/experiments/2026-11-04-decode-pass/combined.md)
owns the complete results and reproduction commands. The sections below explain
what motivated each experiment and why we kept or declined it.

## Gather packed KV directly into half-precision scratch

The first comparison tested gathering selected packed FP4 main KV rows directly
to half scratch. This can reduce intermediate traffic and fuse staging without
changing stored KV or checkpoints. Five alternating pairs each at 8K and 32K
showed neutral to slightly favorable latency overall, with phase variation in
both directions. All 768 tested fixed-history logit rows and continuation
hashes matched; these checks do not establish practical task quality.

The resident-fast and persistent-fast presets adopt the existing path, with an
explicit `DS4_METAL_V41_PACKED_F16_GATHER=0` override. This FP4-owner result is
separate from earlier F16-storage qualification. The
[result report](../../../speed-bench/experiments/2026-11-04-decode-pass/f16-gather.md)
contains the setup, samples and limitations.

## Submit ready decode work sooner

The next profile showed a longer gap before routed expert work when the CPU
kept its commands queued until the next router submission. Testing the existing
flush interval confirmed that more frequent submission helps this workload:
every-layer submission improved fresh and warm decode in all five pairs at
both 8K and 32K. Disabling periodic submission slowed both decode phases in
all five 32K pairs. Whole-phase time was noisier because prefill dominates it.

The resident fast presets now select every-layer submission, with an explicit
`DS4_METAL_V41_DECODE_FLUSH_LAYERS=2` override for the ordinary cadence.
Router waits, expert-cache retirement and the final drain are unchanged.
Numerical checks matched at 8K, 32K, an unaligned prefix and a requested 16 GB cache
budget. The [cadence report](../../../speed-bench/experiments/2026-11-04-decode-pass/flush-cadence.md)
owns complete samples, variation and the tradeoff curve. Other hardware and
TP performance remain unmeasured. The next experiment used a new profile.

## Move shared down into the routing interval

The next profile still showed an exposed shared-down/HC tail after routed
expert work. Moving the existing down projection before the router-event wait
and using a separate late join preserved all tested logits, including the
unaligned prefix and tighter cache. Five-pair comparisons found a modest decode
gain at both 8K and 32K, with variation and one slower warm 32K pair.

The resident fast presets adopt this path with an explicit
`DS4_METAL_V41_ROUTE_SHARED_DOWN=0` override. It adds a dispatch but no command
buffer or allocation. Earlier host preparation consumes some potential overlap,
so the profile's exposed tail was not a promised saving. The
[result report](../../../speed-bench/experiments/2026-11-04-decode-pass/shared-down.md)
owns the measurements, rounding/lifetime explanation and diagnostic limits.

## Fewer router barriers did not establish a decode gain

Replacing the router's intra-SIMD sorting exchanges reduced threadgroup barriers
from 46 to 16 while preserving tested probabilities, tied expert order and
weights. The separate profile showed only about 0.05 ms less router time per
token. All 45 comparison stages remain, and shared expert work overlaps almost
the entire router interval; the selected-ID event still constrains host routing.

Five-pair comparisons at 8K and 32K, followed by five longer 32K decode pairs,
showed no consistent throughput gain. We left the change unadopted rather than
call the small mixed differences a regression. The prototype remains recoverable
from its experiment commit; the active implementation returns to the adopted
baseline. The [router report](../../../speed-bench/experiments/2026-11-04-decode-pass/router-simd.md)
records all samples, the longer workload's slow append and the diagnostic limits.

## Reuse the query's existing position coefficients

The short Q8 query can read forward RoPE coefficients from the graph's current
per-token table, preserving its projection, reduction and BF16 boundaries.
This removes repeated trigonometry without allocating storage or adding a
dispatch. The table is valid only within the scalar step that encoded it.

Five alternating pairs at both 8K and 32K were neutral to slightly favorable,
and all 768 tested full-model rows and continuation hashes matched. The separate
profile did not show less exposed query work. We adopted the small reuse change
in the resident fast presets with an explicit off switch, without attributing
a specific speed gain to it. The [coefficient report](../../../speed-bench/experiments/2026-11-04-decode-pass/query-rope-table.md)
owns the samples, numerical checks and limits. The next investigation examined
host work in the remaining routing gaps.

## Fewer deadline checks did not improve event polling

A short CPU sample mostly observed the main thread waiting for selected expert
IDs. Clock reads dominated those sampled stacks, but that is not removable
model work. Checking the deadline every 16 or 64 polls preserved the measured
logits yet failed to establish a decode benefit in five-pair comparisons.
Stride 64 was slower in all five warm 32K pairs. The original polling loop
remains; the [clock-stride report](../../../speed-bench/experiments/2026-11-04-decode-pass/clock-stride.md)
preserves the measurements and the candidate revision.

## Keep selected KV scratch until its next publisher

Several scalar layers use the same compressed KV selection. Reusing its
separate half-precision scratch removes 30 of 38 gathers per token while raw
window staging still runs per layer. The ordered executor and publisher
boundaries make the lifetime explicit without another cache or validity field.

Five pairs at each of 8K and 32K gave small, mixed timing differences; all five
fresh 32K pairs favored reuse. The 768 tested logit rows and continuation hashes
matched, including an odd prefix boundary. We adopted the simple reuse in the
resident fast presets with an off switch, without promising the overlapped
gather duration as latency savings. The [reuse report](../../../speed-bench/experiments/2026-11-04-decode-pass/selected-kv-reuse.md)
owns the measurements and remaining limits.

## A different polling clock trades short-context speed for warm 32K cost

Replacing only the spin deadline's clock retained every-poll readiness checks
and the existing overall timeout. Five-pair timings favored the candidate on
average at 8K, but every warm 32K pair favored the original clock. A separate
CPU sample confirmed the raw-clock path without establishing why the warm
difference occurred. We left it unadopted for the large-context goal, avoiding
an unsupported context-length switch. The [raw-clock report](../../../speed-bench/experiments/2026-11-04-decode-pass/raw-clock.md)
records this tradeoff and the retained prototype. We then froze runtime changes
for combined qualification of the five adopted improvements.

## Qualify the adopted combination

The final comparison used the original executable and shader sources in a
separate checkout. The package improved steady decode in every pair at both
8K and 32K; the longer 32K windows gained about 7–8% throughput. A recorded
coding session took 4.2% less inference time, including its prefill work.
This replay fixes tokens and tool outputs, so it does not establish a reduction
in autonomous task completion time.

All 4,651 checked full-vocabulary rows matched byte for byte across the text
and coding histories. Both builds also completed one fresh, reasoning-enabled
coding task; an independent oracle checked 207 inputs for each generated
function. These are bounded quality checks, not a broad coding evaluation.
All final runs completed under the pressure guard. The tested binaries are
installed locally, the manuals describe activation and fallback conditions,
and main remains unchanged. No further experiment starts automatically.
