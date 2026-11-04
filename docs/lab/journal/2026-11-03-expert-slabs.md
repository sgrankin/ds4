# Expert-cache slab size and latency

Keep the 4 GiB slab default. Changing to 512 MiB did not establish an improvement
in append or complete workload time. It does offer a narrower tradeoff: a
shorter first decode step, with more time spent in the remaining generation
window. This does not justify calling the smaller buffers generally slower or
faster.

## Context and setup

The preceding profile found long waits before selected expert work began on the
GPU. Since the encoder declares whole expert-cache buffers, we tested whether
smaller buffers would reduce those waits. Runtime `ab947bef` on the M5 Max used
Flash 1017 IQ2/Q2 and the adopted routing/RoPE settings. Only slab size changed.
Five alternating pairs at each of 8K and 32K active context used the same cache
target, fresh prefill, 128 teacher-forced steps, a 512-token append and another
128 steps. Separate runs checked logits and collected command traces.

## Findings and decision

Whole inference time was effectively neutral at both contexts. At 32K, append
timings remained variable; at 8K, smaller buffers lengthened the append in all
five pairs. The first decode step consistently became shorter, so a throughput
metric that drops only that step gives an incomplete account of the tradeoff.
The diagnostic traces show substantial remaining gaps before GPU work starts,
even when the longest individual waits shrink. They do not establish the
driver mechanism responsible for the gaps.

All 512 full-vocabulary logit rows and 512 post-evaluation hashes matched.
The cache diagnostics retained the same capacity, populated entries and loader
traffic, with more buffer allocations. No runtime or preset changed. The
[result report](../../../speed-bench/experiments/2026-11-03-expert-slabs/README.md)
owns all samples, the tradeoff plot, resource counters and reproduction commands.
