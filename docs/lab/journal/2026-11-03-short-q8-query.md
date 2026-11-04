# Short Q8 query projection

We asked whether specializing the single-token Q8 query projection for the
DeepSeek V4.1 shape could improve decode while preserving the existing numerical
boundaries. The first version assigned one row per SIMD group; the second
paired adjacent rows so they could share input loads and perform RoPE without a
threadgroup barrier.

The focused kernel check and 8K and 32K full-model teacher-forced logits
matched exactly. Five alternating pairs at 8K were neutral, while the 32K
results were mixed and did not establish a robust whole-workload gain. We retain
the specialized path as opt-in, without preset adoption; its benefit is
unresolved. A single instrumented timeline showed lower summed query encoder
durations, but overlap and profiling limit what that observation says about
end-to-end latency.

The [experiment report](../../../speed-bench/experiments/2026-11-03-short-q8-query/README.md)
explains the mechanism, workload, all paired samples, and reproduction commands.
