# Profiling after routing overlap

The next recommended decode experiment is to enable the existing per-token
RoPE table and attention-head transform for eligible SSD streaming. The current
path repeats inverse rotation while loading the same heads for different
output pairs. The alternative performs that transform once and leaves the
matrix multiplication in a plain projection. This is a candidate, not an
implemented improvement.

## Context and setup

After adopting [routing overlap](2026-11-03-route-overlap.md), we repeated
profiling at runtime revision `a48a1c52` to identify what to try next. The M5 Max
used Flash 1017 IQ2/Q2, the same 64 GiB expert-cache target, resident cap five,
packed KV, 8K prefill chunks and 2 ms event polling. Five current-only runs
refreshed 32K prefill, decode and a 512-token append. Separate coarse and
detailed profiles examined dependencies; they were not throughput comparisons.

## Findings and conclusion

The detailed trace confirms fewer command buffers with overlap enabled than
disabled, but its intrusive timing
does not explain the earlier measured speedup: the uncovered router handoff
interval did not shrink in that trace. Expert-load waits remain small in this
mostly resident workload. The low projection with inverse RoPE is a meaningful
GPU cost, and the existing transform has focused numerical tests. Extending its
eligibility is a smaller next experiment than writing a new kernel.

The timing refresh was noisy. Warm appends split between two timing ranges,
and one run was stopped by the pressure guard before two replacement runs
completed the five-sample set. System activity was observed, but the cause of
the append variation is unresolved. All samples, the interruption, source
references, reproduction commands and the proposed qualification are in the
[result report](../../../speed-bench/experiments/2026-11-03-post-overlap-profile/README.md).
The [Apple Silicon manual](../../models/deepseek/apple-silicon.md) explains the
current gate and candidate. No runtime behavior or preset changed in this slice.
