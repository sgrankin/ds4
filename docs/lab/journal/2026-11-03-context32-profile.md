# Large-context throughput after resident-routing adoption

The current stack reaches about 700 tokens/s on fresh 32K prefill and 27
tokens/s in steady decode on the measured M5 Max workload. Separate profiles
identify the per-layer routing handoff as a more promising immediate decode
target than further expert-read prefetching in this warm-cache case.

## Context and setup

Upstream's recent gains motivated another look at our dependencies after
adopting five resident substitutions. This diagnostic slice fixed revision
`4a77c0f2`, the installed Flash 1017 IQ2_XXS/Q2_K model, packed KV and a 64 GiB
expert-cache target. Five alternating pairs compared 2K and 8K prefill chunks
over fresh 32K prose, a warm 512-token append and teacher-forced decode at both
frontiers. Separate coarse and detailed traces supplied attribution rather
than comparative throughput. The pressure guard remained active throughout.

## Findings and decision

The 8K chunk reduced mean fresh prefill time by 5.72%, winning all five pairs,
with 4.07 GiB more planned scratch space. It did not materially change warm append
time. The [result report](../../../speed-bench/experiments/2026-11-03-context32-profile/README.md)
owns all samples, variation, timing boundaries, source findings and reproduction
commands. This workload does not establish autonomous agent throughput or
quality, and its small-append path differs from large native prefill.

Almost all decode layer calls found their final selected experts resident.
The GPU still waits for the host to read routing results and choose substitutes
at each layer. Existing Metal event primitives suggest a small next experiment:
start independent shared-expert work after signaling routing readiness, then
let the host apply the unchanged resident policy while that work runs. Observed
gaps are not a speedup forecast; only a paired experiment can establish how much
can be hidden.

The first decode step after large prefill remains unusually slow, with its
precise cause unresolved. Detailed profiling also more than doubled warm-append
time, reinforcing the need to keep instrumentation out of comparative runs.
No runtime or preset changes were made in this slice. The proposed overlap
experiment remains a choice for the next slice.
