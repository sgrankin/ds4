# Profiling after streaming RoPE reuse

The next recommended experiment is to reduce expert-cache slab size while
holding total cache capacity fixed. A slab is one Metal buffer containing many
expert slots. The slow warm append spends substantial time waiting for routed
expert GPU work to start. Long waits recur at separated layers; those layers
could correspond to switching between the current 4 GiB slabs, but that
relationship is unverified. This is a testable explanation, not a proven cause.

## Context and setup

After adopting streaming RoPE reuse, we profiled runtime revision `ab947bef` on
the M5 Max with Flash 1017 IQ2/Q2. The fixed workload prefills 32K tokens,
decodes 128 teacher-forced tokens, restores the prompt frontier, appends 512
tokens and decodes another 128. Separate coarse, detailed and host-sampled runs
retain packed KV, resident cap five, the 64 GiB cache target, 8K prefill chunks,
routing overlap and 2 ms polling. These are diagnostic runs, not a comparative
performance campaign.

## Findings and conclusion

Existing command timestamps place about 6.46 seconds of one slow append before
routed expert GPU execution. Earlier work on the same engine has already
finished. Source inspection shows that selected experts expose whole slab
buffers to Metal; the exact relationship between stalled layers and slot
allocations remains unverified. The post-execution wait also protects expert
slots from premature reuse, so removing it is not a simple scheduling fix.

Compare 4,096 and 512 MiB slabs with at least five alternating samples per side,
checking actual capacity, outputs, cache traffic, total append time and decode
time. The independent short-Q8 query kernel remains a promising later attempt
for steady decode TPS. No runtime setting changed in this profiling slice.
The [result report](../../../speed-bench/experiments/2026-11-03-post-rope-profile/README.md)
owns the timings, dependency analysis, reproduction commands and limitations.
