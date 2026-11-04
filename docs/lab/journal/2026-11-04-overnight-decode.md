# Overnight decode experiments

The authorized decode session continues until 06:55 Eastern. Its first
completed comparison tested gathering selected packed FP4 main KV rows directly
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
TP performance remain unmeasured. The session continues with a new profile.

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
