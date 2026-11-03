# Profiling after route overlap

The next experiment should enable the existing per-token RoPE table and
attention-head transform for eligible single-device SSD streaming decode. The
current source explicitly excludes streaming from that path. This is a
source-backed candidate, not a measured speedup: the current detailed profile's
3.86–4.08 ms/token low-projection category includes matrix multiplication and weight
traffic that will remain. The [route-overlap report](../2026-11-03-route-overlap/README.md)
owns the prior five-pair speedup; the five new profile-off samples measure the
current path only. The paired
instrumented runs below compare flags for diagnosis, not throughput.

## Setup and timing

The baseline is runtime revision `a48a1c52` on the M5 Max, using the installed
DeepSeek V4.1 Flash 1017 IQ2_XXS/Q2_K imatrix model with Metal SSD streaming,
a 64 GiB expert-cache target and route overlap with 2 ms polling. It retains
packed main/index KV, resident cap five, 8,192-token prefill chunks and 40,960
allocated positions. The workload
is the same 32,768-token prefill, 128 teacher-forced decode steps, 512-token
warm append and another 128 steps as the previous experiment. Fresh processes
start with empty process-local expert caches; the OS file cache is not flushed.
Expert residency persists across the KV restore before the append. Phase times
exclude startup and snapshot save/restore and do not represent complete serving
requests. Steady decode excludes the first of 128 steps; combined phases include
both full decode phases and both prefills. There is no MTP, sampling or API
overhead. Prefill seconds below are calculated from rounded throughput.

The reused [driver](../2026-11-03-route-overlap/run.nu) supports `--only overlap`,
`--samples 2` and the `timeline` stage. Raw evidence is local under
`.workspaces/post-overlap-profile-20261103/`. The first three completed samples
are in `timing/` and the last two in `timing-resume/`; profiling was off.
Reproduce with fresh output directories and one guarded job at a time:

```nu
./tools/zig-build ds4-bench -j4
python3 speed-bench/ideas/pressure_run.py .workspaces/post-overlap-repro-guard nu speed-bench/experiments/2026-11-03-route-overlap/run.nu .workspaces/post-overlap-repro timing --only overlap --spin-us 2000
python3 speed-bench/ideas/pressure_run.py .workspaces/post-overlap-timeline-guard nu speed-bench/experiments/2026-11-03-route-overlap/run.nu .workspaces/post-overlap-timeline timeline --spin-us 2000
```

| Sample | Fresh prefill, s | Fresh steady decode, tok/s | Warm append, s | Warm steady decode, tok/s | Combined phases, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 48.89 | 30.08 | 16.47 | 30.03 | 75.83 |
| 2 | 49.79 | 29.28 | 7.35 | 28.90 | 67.17 |
| 3 | 50.41 | 30.36 | 7.19 | 29.05 | 67.26 |
| 4 | 47.51 | 26.62 | 16.48 | 29.27 | 75.19 |
| 5 | 50.80 | 26.17 | 15.88 | 26.41 | 78.80 |

Fresh prefill averaged 49.48 ± 1.32 s and steady decode 28.50 ± 1.97 tok/s
(sample standard deviation). Warm append split between 7.19–7.35 s and
15.88–16.48 s; its cause is unresolved. Safari and WebKit were active, so
these are noisy diagnostic baseline observations, not a controlled comparison.
The original fourth run was interrupted by the pressure guard and is retained
locally. The guard stopped that run after two consecutive kernel pressure-level 2
(warning) samples; the level returned to 1 after the stop. Swap rose about
1.3 MiB without runaway growth.

## Profile anatomy

One instrumented current-path run attributed 46.59 s of its 57.23 s fresh
prefill to draining GPU work, 2.37 s to mapping, 1.43 s to Engram, 0.16 s
to encoding and 5.47 s to seeding. These host-side intervals describe the
instrumented run, not isolated GPU kernel costs. Mapping prepares a layer's
weights, including waiting for its reads; seeding populates the expert cache
for subsequent decode. Its 17.00 s warm append contained 16.60 s in encoding
and internal waits. Each measured interval processes one model layer for a
64-token tile; 52 of these 320 intervals exceeded 100 ms, totaling 7.33 s.
A previous diagnostic append
measured 7.12 s with no such intervals. This locates the observed slow append
in the host encoding interval, but does not identify its cause. Pipeline
creation was absent from the warm stage range in either diagnostic.

The coarse profile recorded 10,171 fully resident decode layer selections out
of 10,240 (99.33%). The 69 exposed expert-load waits totaled 38.01 ms across
256 decode steps, about 0.15 ms/token. This does not count all storage traffic;
it shows that exposed expert I/O is a small decode cost for this workload.

In the paired detailed decode timelines, overlap used about 61 command buffers
per token versus 101 with the flag off. The instrumented uncovered handoff
gaps were roughly 11 ms/token with overlap and 10 ms/token without it. This
is the time between the router finishing its selection and routed expert
execution starting that no logged encoder interval covers. It is not a direct
measurement of CPU execution time. This
intrusive diagnostic does not quantify the earlier measured gain or show an
improved gap. Current-path low projection with inverse RoPE summed to 3.86
ms/token during decode after fresh 32K prefill and 4.08 ms/token after the
512-token append. Query projection plus forward RoPE summed to 4.68 and
4.75 ms/token respectively and is outside the proposed change. These
are summed encoder intervals, not critical-path savings.

The detailed current-path timeline recorded a 57.31 s fresh-prefill device
envelope and 44.53 s union of encoder intervals. Its largest summed kernel
buckets were indexed mixed attention at 12.67 s, routed IQ2 gate/up at
7.93 s, dense Q8 at 5.11 s, exact F16 GEMV at 4.15 s, and Q2 down at
4.09 s. In the warm append, 20,480 exact low-output projection dispatches
summed to 0.96 s. These encoder durations can overlap, and instrumentation
distorts timing, especially for the many small warm-append dispatches; neither
the sums nor the device envelope predict an achievable saving.

## Next experiment

The low projection compresses the attention heads into the smaller intermediate
representation used by the final attention output projection. Before this
compression, the model undoes the heads' rotary position transform (inverse
RoPE). Multiple output pairs consume the same transformed heads.

Today the [executor](../../../src/engine/deepseek41/executor.zig) constructs
one coefficient table per token only when `!graph.streaming`, among other
eligibility checks. The [attention path](../../../src/engine/deepseek41/attention.zig)
can use that table to round and inverse-rotate each attention head once, before
a plain low projection reads it. The current low projection instead performs
that transformation while loading the shared heads for output pairs. The table
also permits fused index-query rotation and publication. The existing
[dispatch](../../../src/metal/v41_dispatch.zig) selects the plain projection
when the heads have already been transformed. The table is graph-owned and is
valid for the current position; this does not require copying model weights.

Test the narrow change at the executor's streaming gate, retaining its other
eligibility checks and existing fallback paths. Reuse the focused GPU checks in
[`test_deepseek41_metal.c`](../../../tests/test_deepseek41_metal.c), which compare
head transformation plus plain low against inline rotation and compare fused
index-query publication against separate operations. Follow with fixed-history
full-model checks and at least five alternating profile-off pairs under the
pressure guard. Measure ordinary and favorable cases as well as the 32K case;
inspect quality and resource behavior. The measured low-projection duration is an
opportunity indicator, not a prediction that all of it is removable or even
that trigonometry dominates it. Query projection and forward RoPE are separate.

For prefill, the existing exact low-projection row-grid operation may reduce
warm-append dispatch overhead, and narrowing inactive SIMD groups in exact F16
projection is another bounded candidate. Both remain alternatives, not part of
this proposed decode experiment.
