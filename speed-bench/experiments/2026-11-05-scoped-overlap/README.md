# Shared-expert overlap with scoped slab residency

This experiment asks whether shared-expert overlap helps when expert slabs use
extended command-buffer residency. The one-hour slice is November 5, 2026,
07:29–08:29 Eastern. Output checks agree for the tested histories; 8K and 32K timing favor overlap
off with a requested 64 GB cache. Smaller-cache decode favors overlap on, with
mixed whole-workload results. The selected experimental
package remains queue residency mode 1 with overlap off; ordinary and fast
presets leave residency off.

The [seed-write comparison](../2026-11-05-seed-residency/README.md) established
association on seed copies and found the large first-compute gap from the
[preceding compute-only comparison](../2026-11-05-scoped-residency/README.md)
absent in its trace. It did not establish a whole-workload gain. Earlier
queue-wide residency interacted with shared-event handoffs, so the selected
package disables overlap. Narrower association may change that interaction,
but mixed expert and router buffers still receive residency; the traces inspect actual association and event coverage.

## Comparison and workload

Both arms use runtime `d4c0d297`, closure `19a2106c`, and the same existing
`ds4-bench` binary, SHA-256
`1ba1aeed3bd3dfdb001a1bd4945071357a1dc4f6b8250835af7c523dc9c9a3a7`.
Extended command residency mode 2 is fixed; only
`DS4_METAL_V41_ROUTE_SHARED_OVERLAP` changes from 0 to 1. The remaining
package 25 settings stay fixed, including refresh 16 and submission cadence 4;
cached-refresh parallelism is off. The flag changes the shared-expert schedule
and fusion as well as the event handoff, so a difference cannot be attributed
to event signaling alone.

The machine is a 128 GiB M5 Max, using
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`. At each 8K or 32K
frontier, the runner prefills, evaluates 128 teacher tokens, saves and restores
a snapshot, appends 512 tokens and evaluates another 128 tokens. The expert
cache survives restore; the OS file cache is not flushed. Phase timers exclude
startup and snapshot operations; process wall includes them. Steady decode
excludes only the first token, so retain first-step and complete generation
latency as well as whole inference and process wall.

Compare fixed-history full-vocabulary outputs at both frontiers and at 8K with
a requested 16 GB cache to exercise misses. Run five profile-off,
alternating timing pairs at each primary frontier, retaining every sample and
reporting variation, paired differences and order effects. Separate profile
pairs inspected shifted gaps and resource retirement; profiling changes
timing. Fixed-history agreement or drift does not establish practical task
quality. The primary regressions do not warrant direct 32K qualification against the
selected queue-mode-1/overlap-0 package. Five additional timing pairs at 8K
with a requested 16 GB cache test whether the penalty persists with a smaller
set; the quality check compares overlap settings within each cache size, not
quality across cache sizes.

## Output and source checks

At 8K and 32K, and in the smaller-cache case repeating the 8K history, each
comparison produced 256 byte-identical full-vocabulary teacher-logit rows and
256 matching post-evaluation hashes. All 768 paired rows match; delta-NLL, KL
and top-1 changes are zero. This agreement covers the tested fixed histories,
not practical task quality or arbitrary histories.

The requested 16 GB cache yielded 958 slots and an 8.88 GiB expert target.
Both arms recorded 122,672 hits, 4,951 misses, 3,993 evictions and reuses,
37.52 GiB of cumulative logical preads and 21,282 explicit-use calls. Logical
preads need not be physical SSD traffic, and use calls are not unique command
buffers.

The local source audit found no concrete blocker in event or pin retirement.
It also confirms that a command buffer containing previous expert work and a
subsequent router/event remains associated with the residency set. Pure
router/shared buffers can be unassociated. This source review does not prove
runtime coverage or performance; the traces inspect it below.

## 32K timing

Five profile-off pairs compare overlap 0 with overlap 1 in mode 2. Overlap 0
ran first in pairs 1, 3 and 5. Both complete fresh and warm decode, inference
total and process wall were slower with overlap 1 in every pair. Fresh decode
increased by 0.936–1.327 seconds per pair; inference total increased by
1.929–3.801 seconds. The warm candidate in pair 2 was unusually slow and is
retained. Prefill and warm append also drifted across the sequence; the paired
results include both run orders. The smaller-cache comparison gives a different decode result; the traces
locate an associated timing difference without establishing its cause.

Values are overlap 0 → 1 in pair order; the final column is mean ± sample
standard deviation. Seconds derived from two-decimal throughput output are
approximate. Inference total sums prefill and decode phases.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 42.914 → 42.562 | 44.639 → 44.526 | 45.388 → 46.576 | 47.129 → 46.876 | 46.943 → 47.114 | 45.403 ± 1.741 → 45.531 ± 1.951 |
| Fresh first step, ms | 183.072 → 251.080 | 187.265 → 271.552 | 195.087 → 271.335 | 225.376 → 299.054 | 180.486 → 251.605 | 194.257 ± 18.253 → 268.925 ± 19.614 |
| Fresh complete decode, s | 3.746 → 5.016 | 3.985 → 4.921 | 3.744 → 5.018 | 3.785 → 4.994 | 3.748 → 5.075 | 3.802 ± 0.104 → 5.005 ± 0.056 |
| Fresh steady decode, tokens/s | 35.640 → 26.650 | 33.440 → 27.310 | 35.790 → 26.760 | 35.680 → 27.050 | 35.600 → 26.320 | 35.230 ± 1.003 → 26.818 ± 0.379 |
| Warm append, s | 7.138 → 7.651 | 7.107 → 7.786 | 7.581 → 7.937 | 7.802 → 7.791 | 7.819 → 7.843 | 7.489 ± 0.348 → 7.801 ± 0.104 |
| Warm first step, ms | 94.278 → 156.010 | 39.887 → 154.513 | 45.445 → 178.648 | 40.947 → 156.796 | 43.865 → 148.578 | 52.884 ± 23.246 → 158.909 ± 11.495 |
| Warm complete decode, s | 3.671 → 4.871 | 3.732 → 5.433 | 3.889 → 4.872 | 3.958 → 4.942 | 4.008 → 4.979 | 3.851 ± 0.145 → 5.019 ± 0.236 |
| Warm steady decode, tokens/s | 35.510 → 26.940 | 34.400 → 24.060 | 33.040 → 27.060 | 32.420 → 26.540 | 32.040 → 26.290 | 33.482 ± 1.446 → 26.178 ± 1.224 |
| Inference total, s | 57.469 → 60.099 | 59.463 → 62.666 | 60.602 → 64.403 | 62.674 → 64.603 | 62.518 → 65.011 | 60.545 ± 2.183 → 63.356 ± 2.029 |
| Process wall, s | 60.425 → 62.782 | 62.044 → 65.594 | 63.488 → 67.084 | 65.379 → 67.315 | 65.003 → 67.612 | 63.268 ± 2.067 → 66.077 ± 2.000 |

The independent quality and resource review found matching arguments and
non-time cache and routing counters. It found no event timeout, command error,
intermediate residency clear or residency fallback in the reviewed logs. The
selected-readback summaries were not emitted because the call count stayed
below their reporting threshold; their absence does not show that event waits
were absent. The traces inspect scheduling and association coverage below.

## 8K timing

The same five-pair profile-off procedure at 8K used the same arm order and
units. Complete fresh and warm decode, inference total and process wall were
slower with overlap 1 in every pair; prefill and warm append were mixed.
Values are overlap 0 → 1; the final column is mean ± sample standard deviation.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 18.683 → 18.199 | 18.690 → 18.333 | 19.204 → 19.326 | 20.095 → 19.382 | 19.212 → 19.232 | 19.177 ± 0.576 → 18.894 ± 0.578 |
| Fresh first step, ms | 179.989 → 224.846 | 189.401 → 226.570 | 220.738 → 226.514 | 196.068 → 230.164 | 199.595 → 223.126 | 197.158 ± 15.150 → 226.244 ± 2.609 |
| Fresh complete decode, s | 3.661 → 4.796 | 3.679 → 4.767 | 3.698 → 4.807 | 3.672 → 4.769 | 3.688 → 4.867 | 3.680 ± 0.014 → 4.801 ± 0.041 |
| Fresh steady decode, tokens/s | 36.490 → 27.780 | 36.390 → 27.970 | 36.520 → 27.730 | 36.530 → 27.980 | 36.400 → 27.350 | 36.466 ± 0.067 → 27.762 ± 0.256 |
| Warm append, s | 6.780 → 6.746 | 7.153 → 6.734 | 7.516 → 7.536 | 7.642 → 7.968 | 7.647 → 7.496 | 7.348 ± 0.376 → 7.296 ± 0.540 |
| Warm first step, ms | 39.921 → 151.074 | 44.157 → 220.465 | 43.244 → 208.081 | 43.641 → 227.437 | 43.392 → 156.346 | 42.871 ± 1.685 → 192.681 ± 36.292 |
| Warm complete decode, s | 3.497 → 4.672 | 3.878 → 4.692 | 3.904 → 4.917 | 3.820 → 5.176 | 3.851 → 4.876 | 3.790 ± 0.166 → 4.867 ± 0.204 |
| Warm steady decode, tokens/s | 36.730 → 28.090 | 33.130 → 28.400 | 32.900 → 26.970 | 33.630 → 25.660 | 33.360 → 26.900 | 33.950 ± 1.577 → 27.204 ± 1.089 |
| Inference total, s | 32.621 → 34.412 | 33.400 → 34.526 | 34.322 → 36.586 | 35.229 → 37.295 | 34.398 → 36.471 | 33.994 ± 1.004 → 35.858 ± 1.307 |
| Process wall, s | 35.016 → 36.823 | 35.885 → 37.040 | 36.771 → 39.099 | 37.653 → 40.243 | 36.891 → 38.882 | 36.443 ± 1.015 → 38.418 ± 1.454 |

## 8K timing with a smaller cache

Five profile-off alternating pairs requested a 16 GB cache, with overlap 0
first in pairs 1, 3 and 5. Fresh complete decode was lower with overlap 1 in
four pairs; warm first-step and complete decode were lower in all five. Whole
inference and process wall were lower in only two pairs each. This is a
favorable decode case for overlap, but it does not establish a whole-workload
gain. These arms compare overlap within this cache size; they do not establish
quality or performance equivalence across cache sizes. The large-cache primary
regressions do not support global adoption. A matched 32K trace with a requested
16 GB cache inspects event gaps alongside the large-cache trace; neither is a
comparative latency sample.

Values are overlap 0 → 1 in pair order; the final column is mean ± sample
standard deviation. Seconds derived from two-decimal throughput are approximate.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 16.413 → 16.907 | 19.051 → 17.867 | 19.667 → 20.046 | 20.100 → 20.161 | 20.139 → 20.063 | 19.074 ± 1.551 → 19.009 ± 1.519 |
| Fresh first step, ms | 215.109 → 167.745 | 190.408 → 191.559 | 178.872 → 180.260 | 191.518 → 189.335 | 195.186 → 212.865 | 194.219 ± 13.178 → 188.353 ± 16.600 |
| Fresh complete decode, s | 3.779 → 3.686 | 3.757 → 3.765 | 3.766 → 3.710 | 3.760 → 3.743 | 3.772 → 3.744 | 3.767 ± 0.009 → 3.729 ± 0.031 |
| Fresh steady decode, tokens/s | 35.640 → 36.100 | 35.610 → 35.540 | 35.410 → 35.980 | 35.590 → 35.740 | 35.500 → 35.970 | 35.550 ± 0.094 → 35.866 ± 0.224 |
| Warm append, s | 8.073 → 8.917 | 8.340 → 8.117 | 9.059 → 9.127 | 8.903 → 9.043 | 8.843 → 8.840 | 8.644 ± 0.417 → 8.809 ± 0.402 |
| Warm first step, ms | 45.617 → 40.922 | 50.599 → 42.360 | 49.848 → 44.686 | 53.004 → 46.371 | 51.684 → 44.447 | 50.150 ± 2.799 → 43.757 ± 2.131 |
| Warm complete decode, s | 3.560 → 3.530 | 3.963 → 3.537 | 3.998 → 3.956 | 3.963 → 3.956 | 3.956 → 3.885 | 3.888 ± 0.184 → 3.773 ± 0.220 |
| Warm steady decode, tokens/s | 36.140 → 36.400 | 32.460 → 36.340 | 32.180 → 32.470 | 32.480 → 32.490 | 32.530 → 33.070 | 33.158 ± 1.673 → 34.154 ± 2.037 |
| Inference total, s | 31.824 → 33.040 | 35.111 → 33.285 | 36.489 → 36.838 | 36.726 → 36.902 | 36.710 → 36.531 | 35.372 ± 2.093 → 35.319 ± 1.976 |
| Process wall, s | 32.997 → 34.286 | 36.337 → 34.543 | 37.767 → 38.077 | 37.922 → 38.080 | 37.939 → 37.850 | 36.592 ± 2.119 → 36.567 ± 1.969 |

## Large-cache trace

One intrusive 32K profile pair with a requested 64 GB cache recorded matching
hashes and cache counts. Refreshes used 91 → 51 command buffers. With overlap
on, 29 routers per refresh shared an associated command buffer with prior
expert work, while 11 occupied unassociated buffers without prior experts. In
four observed refresh windows, the positive router-end to shared-start gaps
totaled 93.890–116.083 ms in those 29 mixed buffers, versus 0.562–1.656 ms in
the 11 unassociated buffers. The event lies at this boundary according to the
source; it has no separate encoder row. This concentrates the observed delay at
mixed expert/router event buffers, but does not establish a driver cause or an
additive latency breakdown.

The profile also recorded the changed shared-down fusion and 40 additional
shader dispatches per refresh. Ordinary steps retained the same recorded kernel
groups and 11 associated buffers. This single profile pair locates differences;
it is not a comparative latency sample. The matched smaller-cache trace also had 29 associated mixed event buffers
per refresh. It did not show comparable large positive router-to-shared gaps
at the later sampled refreshes: their nonpositive counter intervals were clipped
to zero, which does not prove zero physical wait. All 34 hashes matched. Fresh
refreshes used 91 → 51 buffers; warm refreshes used 92 → 52 because both arms
had matching split layers, and ordinary windows used 11. The smaller cache also
changed cache contents, misses and reuse, so this contrast cannot be attributed
to cache bytes alone.

## Decision

Keep overlap off in the selected large-cache package. The large-cache regressions
are consistent across both primary contexts, while the smaller-cache results
preserve a favorable decode case and do not support rejecting overlap globally.
A concrete next question is whether a checked flush at actual CPU-routed expert
completion can keep that work out of the next router/event command buffer. This
has not been implemented, and the trace does not establish that it would remove
the total cost.

All eight guards completed with child return code 0, pressure level 1 only and
no adverse samples. Swap ranged from 29.009707 to 29.011777 GiB.

## Reproduction

Use the existing binary and checkpoint above. The driver clears inherited DS4
settings, alternates timing order and defaults to a 64 GB cache. Diagnostic
counters are enabled for quality checks, not timing. Run one guarded heavy
model job at a time, with no competing heavy model work or builds during timing,
and use fresh output directories. The commands below reproduce all three
timing configurations and their quality checks; the separate profiles require
their own recorded settings and output.

```nu
let out = '.workspaces/hillclimb-20261104/31-scoped-overlap-repro'
let package = 'speed-bench/experiments/2026-11-05-slab-event-interaction/candidate-settings.json'
mkdir $out
open $package | merge {DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY: '2'} | to json | save ($out | path join fixed.json)
open ($out | path join fixed.json) | merge {DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: '1', DS4_METAL_V41_RESIDENT_ROUTE_PROFILE: '1', DS4_METAL_SELECTED_READBACK_PROFILE: '1'} | to json | save ($out | path join diagnostic.json)
for frontier in [8192 32768] {
  python3 speed-bench/ideas/pressure_run.py ($out | path join $'guard-quality-($frontier)') -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu ($out | path join $'quality-($frontier)') quality --flag DS4_METAL_V41_ROUTE_SHARED_OVERLAP --control 0 --candidate 1 --frontier $frontier --tokens 128 --extra-settings ($out | path join diagnostic.json)
  python3 speed-bench/ideas/pressure_run.py ($out | path join $'guard-timing-($frontier)') -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu ($out | path join $'timing-($frontier)') timing --flag DS4_METAL_V41_ROUTE_SHARED_OVERLAP --control 0 --candidate 1 --frontier $frontier --tokens 128 --samples 5 --extra-settings ($out | path join fixed.json)
}
python3 speed-bench/ideas/pressure_run.py ($out | path join guard-quality-16gb) -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu ($out | path join quality-16gb) quality --flag DS4_METAL_V41_ROUTE_SHARED_OVERLAP --control 0 --candidate 1 --frontier 8192 --cache 16GB --tokens 128 --extra-settings ($out | path join diagnostic.json)
python3 speed-bench/ideas/pressure_run.py ($out | path join guard-timing-16gb) -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu ($out | path join timing-16gb) timing --flag DS4_METAL_V41_ROUTE_SHARED_OVERLAP --control 0 --candidate 1 --frontier 8192 --cache 16GB --tokens 128 --samples 5 --extra-settings ($out | path join fixed.json)
```

Local settings, guard records, raw logs and samples belong under
`.workspaces/hillclimb-20261104/31-scoped-overlap/`. The guard records retain pressure, swap and any interruptions.
