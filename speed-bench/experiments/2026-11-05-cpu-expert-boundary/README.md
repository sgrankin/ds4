# Boundary after CPU-routed expert work

This experiment tests whether submitting CPU-routed expert work before the next
router and shared-expert event improves the scoped residency and overlap
combination. The selected one-hour slice is November 5, 2026, 08:18–09:18
Eastern. Select the combined mode 2, overlap 1 and flush 1 settings for the
experimental GPU decode package: direct comparisons show a consistent small
decode gain, with mixed whole-workload results and a 32K append cost. Normal
and fast presets and installed frontends remain unchanged.

The [preceding overlap comparison](../2026-11-05-scoped-overlap/README.md)
found slower large-cache decode with overlap enabled, while the smaller-cache
case favored overlap for decode but had mixed whole-workload results. Its 32K
trace concentrated large router-to-shared gaps in command buffers containing
both earlier expert work and a later router/event. That observation identifies
a boundary to test; it does not establish the driver cause or predict the
whole-workload effect of a flush.

## Change and limits

Runtime `b7e7af9d` adds an opt-in checked flush in
[`Moe.partial`](../../../src/engine/deepseek41/moe.zig), after routed work and
any parallel join, before the caller combines shared and routed outputs with
the residual stream. Exactly
`DS4_METAL_V41_CPU_EXPERT_FLUSH=1` enables it, and only when the local `overlap_shared` condition selects the eligible CPU routing
and shared-expert event path. A failed flush propagates failure. Successful
resident-GPU routes do not enter that condition and retain their cadence of
four layers. The flush submits work; it does not itself wait for completion.

This separates outgoing CPU-routed expert work from the next router/event. It
cannot retroactively separate an incoming GPU-to-CPU readiness fallback from
expert work already encoded by earlier GPU-routed layers in the same command
buffer. The trace must establish actual association and event coverage; fewer
mixed buffers alone would not establish a latency gain.

## Comparison and workload

The previous runtime is `d4c0d297`; the prior closure is `81a8c17d`. Both
comparison arms use the same `b7e7af9d` binary, with SHA-256
`fd4596242336958607ae44f302e77bdc39e4a0ae1a28830df4d93da38ce88921`
and the new flag set to 0 or 1. Extended command residency mode 2, shared
overlap 1, refresh 16 and submission cadence 4 are fixed. The frozen prior
binary and local settings are under
`.workspaces/hillclimb-20261104/32-cpu-expert-boundary/`.

The machine is a 128 GiB M5 Max, using
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`. At each frontier the
runner prefills, evaluates 128 teacher tokens, saves and restores a snapshot,
appends 512 tokens and evaluates another 128 tokens. The expert cache survives
restore; the OS file cache is not flushed. Phase timers exclude startup and
snapshot operations; process wall includes them. Steady decode excludes only
the first token, so first-step and complete decode remain necessary.

The checks cover fixed-history quality and five alternating,
profile-off timing pairs at 8K and 32K with the requested 64 GB cache, and at
8K with a requested 16 GB cache. A separate 32K timeline examines the boundary.
Direct comparisons also test the combined candidate against the selected package.
Profiling is diagnostic and does not supply comparative latency samples.

## 32K timeline

One intrusive profile pair confirms the intended boundary for the sampled CPU
refreshes. Each refresh changed from 51 to 91 command buffers: the control
had 29 associated buffers mixing prior expert work with a router/event; the
candidate placed all 40 router/event/shared buffers outside the slab residency
set and put expert work in 40 separate associated buffers. Ordinary GPU
tokens retained 11 associated mixed buffers. All 34 sampled hashes matched.

Across fresh and warm step-16 refreshes, positive router-end to shared-start
gaps in the control’s 29 mixed buffers totaled 101.040 and 99.822 ms; the
candidate recorded no positive intervals across its 40 event buffers.
Nonpositive intervals were clipped to zero, so this does not prove zero
physical wait. The event has no separate encoder timestamp. The trace locates
the schedule change and associated gap reduction; it does not establish a
driver cause or comparative whole-workload latency. It also does not prove
isolation of every incoming GPU-to-CPU fallback.

## 32K timing

Five profile-off alternating pairs compare flush 0 to 1; control ran first in
pairs 1, 3 and 5. Both complete decode windows, first steps, inference total
and process wall were lower with flush 1 in every pair. Prefill and append
varied across the sequence. Values are flush 0 → 1; the final column is mean
± sample standard deviation. Seconds derived from rounded throughput are
approximate. Inference total sums prefill and decode phases.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 45.356 → 44.641 | 45.361 → 44.550 | 45.827 → 46.281 | 46.992 → 47.164 | 47.053 → 46.810 | 46.118 ± 0.848 → 45.889 ± 1.223 |
| Fresh first step, ms | 256.652 → 209.660 | 244.455 → 198.631 | 258.971 → 217.695 | 248.628 → 192.096 | 303.907 → 175.856 | 262.523 ± 23.871 → 198.788 ± 16.174 |
| Fresh complete decode, s | 5.063 → 3.912 | 4.912 → 3.720 | 4.936 → 3.876 | 4.843 → 3.689 | 5.548 → 3.678 | 5.061 ± 0.284 → 3.775 ± 0.111 |
| Fresh steady decode, tokens/s | 26.430 → 34.300 | 27.210 → 36.070 | 27.150 → 34.710 | 27.640 → 36.320 | 24.210 → 36.260 | 26.528 ± 1.367 → 35.532 ± 0.953 |
| Warm append, s | 7.830 → 7.791 | 7.430 → 7.056 | 7.616 → 8.100 | 7.950 → 7.743 | 8.104 → 7.877 | 7.786 ± 0.267 → 7.713 ± 0.392 |
| Warm first step, ms | 155.908 → 32.466 | 156.067 → 32.042 | 155.484 → 33.593 | 150.495 → 33.609 | 157.522 → 33.299 | 155.095 ± 2.684 → 33.002 ± 0.710 |
| Warm complete decode, s | 4.952 → 3.735 | 4.839 → 3.690 | 4.825 → 3.879 | 4.965 → 4.109 | 4.979 → 3.889 | 4.912 ± 0.074 → 3.860 ± 0.164 |
| Warm steady decode, tokens/s | 26.490 → 34.300 | 27.120 → 34.720 | 27.200 → 33.030 | 26.380 → 31.160 | 26.340 → 32.940 | 26.706 ± 0.419 → 33.230 ± 1.394 |
| Inference total, s | 63.200 → 60.079 | 62.542 → 59.016 | 63.204 → 62.136 | 64.750 → 62.706 | 65.684 → 62.255 | 63.876 ± 1.296 → 61.238 ± 1.603 |
| Process wall, s | 66.152 → 62.842 | 65.093 → 61.602 | 65.794 → 64.785 | 67.390 → 65.508 | 68.297 → 64.869 | 66.545 ± 1.285 → 63.921 ± 1.636 |

## 8K timing

Five profile-off alternating pairs used the same order and units as the 32K
comparison. Complete decode, both first steps, inference total and process
wall were lower with flush 1 in every pair; prefill and append were mixed.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 19.557 → 18.765 | 18.934 → 18.871 | 19.041 → 19.184 | 19.260 → 19.309 | 19.239 → 19.442 | 19.206 ± 0.239 → 19.114 ± 0.288 |
| Fresh first step, ms | 238.116 → 186.221 | 254.529 → 201.197 | 294.334 → 182.345 | 233.534 → 202.344 | 225.750 → 164.591 | 249.253 ± 27.311 → 187.340 ± 15.499 |
| Fresh complete decode, s | 4.865 → 3.604 | 4.885 → 3.645 | 4.836 → 3.594 | 4.796 → 3.613 | 4.830 → 3.575 | 4.842 ± 0.034 → 3.606 ± 0.026 |
| Fresh steady decode, tokens/s | 27.450 → 37.160 | 27.430 → 36.880 | 27.970 → 37.220 | 27.830 → 37.230 | 27.590 → 37.240 | 27.654 ± 0.238 → 37.146 ± 0.152 |
| Warm append, s | 6.765 → 6.936 | 7.100 → 7.082 | 7.253 → 7.539 | 7.542 → 7.675 | 7.261 → 7.427 | 7.184 ± 0.283 → 7.332 ± 0.312 |
| Warm first step, ms | 258.198 → 34.844 | 153.180 → 32.436 | 152.737 → 32.610 | 228.874 → 31.898 | 210.995 → 32.979 | 200.797 ± 46.809 → 32.953 ± 1.126 |
| Warm complete decode, s | 4.863 → 3.754 | 4.773 → 3.675 | 4.823 → 3.749 | 4.982 → 3.772 | 5.049 → 3.714 | 4.898 ± 0.115 → 3.733 ± 0.039 |
| Warm steady decode, tokens/s | 27.580 → 34.150 | 27.500 → 34.870 | 27.190 → 34.170 | 26.720 → 33.950 | 26.250 → 34.500 | 27.048 ± 0.559 → 34.328 ± 0.362 |
| Inference total, s | 36.051 → 33.058 | 35.692 → 33.272 | 35.953 → 34.067 | 36.580 → 34.369 | 36.380 → 34.159 | 36.131 ± 0.351 → 33.785 ± 0.581 |
| Process wall, s | 38.554 → 35.643 | 38.176 → 35.747 | 38.407 → 36.552 | 39.061 → 36.918 | 38.801 → 36.963 | 38.600 ± 0.343 → 36.364 ± 0.633 |

## 8K timing with a smaller cache

Five profile-off alternating pairs requested a 16 GB cache, with control first
in pairs 1, 3 and 5. Fresh complete decode was lower with flush 1 in four
pairs, warm complete decode in three. Inference total and wall were lower in
all five, but so were prefill and append on a source path unchanged by this
scalar-only flag. That drift prevents attributing the whole-workload change
to the flush. Units and derivations match the large-cache tables.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 18.264 → 18.093 | 18.992 → 18.658 | 20.909 → 19.521 | 19.753 → 19.625 | 20.265 → 19.764 | 19.637 ± 1.040 → 19.132 ± 0.724 |
| Fresh first step, ms | 165.678 → 182.693 | 184.773 → 183.956 | 198.733 → 190.434 | 191.888 → 215.798 | 180.295 → 196.758 | 184.273 ± 12.539 → 193.928 ± 13.455 |
| Fresh complete decode, s | 3.688 → 3.677 | 3.718 → 3.688 | 4.062 → 3.719 | 3.919 → 3.737 | 3.714 → 3.731 | 3.820 ± 0.164 → 3.710 ± 0.027 |
| Fresh steady decode, tokens/s | 36.060 → 36.340 | 35.950 → 36.250 | 32.870 → 36.000 | 34.070 → 36.070 | 35.930 → 35.940 | 34.976 ± 1.440 → 36.120 ± 0.169 |
| Warm append, s | 8.243 → 8.139 | 8.428 → 8.346 | 8.820 → 8.477 | 8.533 → 8.467 | 8.761 → 8.715 | 8.557 ± 0.238 → 8.429 ± 0.210 |
| Warm first step, ms | 41.188 → 39.105 | 44.469 → 40.684 | 42.231 → 43.251 | 44.695 → 41.529 | 44.635 → 42.000 | 43.444 ± 1.627 → 41.314 ± 1.545 |
| Warm complete decode, s | 3.760 → 3.697 | 3.862 → 3.806 | 3.861 → 3.911 | 3.804 → 3.873 | 3.871 → 3.868 | 3.832 ± 0.048 → 3.831 ± 0.084 |
| Warm steady decode, tokens/s | 34.150 → 34.720 | 33.270 → 33.730 | 33.250 → 32.840 | 33.780 → 33.150 | 33.190 → 33.190 | 33.528 ± 0.421 → 33.526 ± 0.740 |
| Inference total, s | 33.955 → 33.606 | 35.000 → 34.497 | 37.653 → 35.627 | 36.009 → 35.702 | 36.611 → 36.078 | 35.846 ± 1.429 → 35.102 ± 1.024 |
| Process wall, s | 35.166 → 34.782 | 36.208 → 35.740 | 38.938 → 36.872 | 37.260 → 36.872 | 37.900 → 37.307 | 37.094 ± 1.465 → 36.314 ± 1.035 |

## Direct comparison with the selected package at 32K

Five profile-off alternating pairs compare the selected mode 1, overlap 0,
flush 0 package with mode 2, overlap 1, flush 1 on the same runtime and
binary. This changes several settings together. Control ran first in pairs
1, 3 and 5. Complete fresh and warm decode were lower in all five pairs,
while warm append was slower in all five. Inference total and wall were lower
in three pairs; their means are strongly affected by the unusually slow
first control prefill. The decode gain therefore comes with an append tradeoff
and does not establish a consistent whole-workload gain. Units and derivations
match the preceding tables.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 50.275 → 46.395 | 46.749 → 46.693 | 47.477 → 47.249 | 47.150 → 47.621 | 47.275 → 47.797 | 47.785 ± 1.417 → 47.151 ± 0.598 |
| Fresh first step, ms | 265.311 → 161.756 | 192.415 → 201.234 | 194.896 → 194.328 | 206.849 → 163.602 | 194.750 → 162.282 | 210.844 ± 30.966 → 176.640 ± 19.464 |
| Fresh complete decode, s | 3.828 → 3.660 | 3.767 → 3.741 | 3.764 → 3.693 | 3.762 → 3.653 | 3.753 → 3.675 | 3.775 ± 0.030 → 3.684 ± 0.035 |
| Fresh steady decode, tokens/s | 35.650 → 36.300 | 35.530 → 35.890 | 35.580 → 36.300 | 35.720 → 36.400 | 35.690 → 36.160 | 35.634 ± 0.078 → 36.210 ± 0.198 |
| Warm append, s | 7.134 → 7.429 | 7.675 → 7.685 | 7.782 → 7.916 | 7.802 → 7.862 | 7.732 → 7.834 | 7.625 ± 0.279 → 7.745 ± 0.196 |
| Warm first step, ms | 44.464 → 33.720 | 42.695 → 33.802 | 47.877 → 33.516 | 48.998 → 37.162 | 47.446 → 33.768 | 46.296 ± 2.619 → 34.394 ± 1.552 |
| Warm complete decode, s | 3.916 → 3.858 | 4.006 → 3.902 | 4.025 → 3.920 | 4.037 → 3.953 | 4.008 → 3.948 | 3.998 ± 0.048 → 3.916 ± 0.039 |
| Warm steady decode, tokens/s | 32.800 → 33.220 | 32.040 → 32.820 | 31.930 → 32.680 | 31.850 → 32.430 | 32.070 → 32.440 | 32.138 ± 0.380 → 32.718 ± 0.326 |
| Inference total, s | 65.153 → 61.341 | 62.198 → 62.021 | 63.049 → 62.778 | 62.752 → 63.089 | 62.767 → 63.254 | 63.183 ± 1.143 → 62.497 ± 0.801 |
| Process wall, s | 68.453 → 64.120 | 64.801 → 64.648 | 65.627 → 65.400 | 65.345 → 65.580 | 65.315 → 65.839 | 65.908 ± 1.454 → 65.117 ± 0.712 |

## Direct comparison with the selected package at 8K

Five alternating profile-off pairs use the same package settings and order as
the direct 32K comparison. Fresh complete decode was lower in all five pairs
and warm complete decode in four. Inference total and wall were lower in two
pairs, so these results do not establish a consistent whole-workload gain.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 18.331 → 18.139 | 18.371 → 18.378 | 19.218 → 19.708 | 19.363 → 19.491 | 19.215 → 19.319 | 18.900 ± 0.505 → 19.007 ± 0.702 |
| Fresh first step, ms | 220.441 → 159.646 | 199.904 → 170.860 | 225.468 → 205.557 | 198.360 → 185.385 | 182.034 → 181.254 | 205.241 ± 17.710 → 180.540 ± 17.176 |
| Fresh complete decode, s | 3.708 → 3.564 | 3.682 → 3.591 | 3.707 → 3.635 | 3.676 → 3.605 | 3.666 → 3.594 | 3.688 ± 0.019 → 3.598 ± 0.026 |
| Fresh steady decode, tokens/s | 36.420 → 37.300 | 36.460 → 37.130 | 36.480 → 37.030 | 36.520 → 37.150 | 36.460 → 37.210 | 36.468 ± 0.036 → 37.164 ± 0.100 |
| Warm append, s | 6.790 → 6.725 | 7.210 → 6.712 | 7.672 → 7.960 | 7.706 → 7.875 | 7.674 → 7.738 | 7.410 ± 0.403 → 7.402 ± 0.629 |
| Warm first step, ms | 42.451 → 30.794 | 45.866 → 31.313 | 42.270 → 33.770 | 41.253 → 32.730 | 45.171 → 34.222 | 43.402 ± 2.000 → 32.566 ± 1.494 |
| Warm complete decode, s | 3.502 → 3.434 | 3.941 → 3.447 | 3.926 → 3.942 | 3.963 → 3.904 | 3.952 → 3.860 | 3.857 ± 0.199 → 3.718 ± 0.254 |
| Warm steady decode, tokens/s | 36.710 → 37.310 | 32.600 → 37.180 | 32.700 → 32.490 | 32.390 → 32.810 | 32.510 → 33.190 | 33.382 ± 1.864 → 34.596 ± 2.431 |
| Inference total, s | 32.331 → 31.863 | 33.205 → 32.129 | 34.523 → 35.246 | 34.708 → 34.874 | 34.506 → 34.511 | 33.855 ± 1.042 → 33.724 ± 1.602 |
| Process wall, s | 34.763 → 34.445 | 35.703 → 34.645 | 37.085 → 37.702 | 37.179 → 37.345 | 36.910 → 36.955 | 36.328 ± 1.058 → 36.218 ± 1.552 |

## Findings and decision

At 8K and 32K with the requested 64 GB cache, and at 8K with the requested
16 GB cache, all 768 paired full-vocabulary teacher-logit rows were
byte-identical and all 768 post-evaluation hashes matched. Delta-NLL, KL and
top-1 changes were zero for these fixed histories. The smaller-cache case
repeats the 8K history; it compares the flag within that cache size, not
quality across cache sizes. The requested 16 GB cache had an 8.88 GiB expert
target, 3,993 evictions and reuses, and five split layers in each arm. No
fresh practical-task checks were run in this experiment. A direct 32K quality
comparison of the two packages also produced 256
byte-identical rows and 256 matching post-evaluation hashes, with zero
delta-NLL, KL and top-1 changes. It repeats the 32K histories above; it is not
an additional set of independent histories.

Select the combined candidate for the experimental GPU decode package. The
direct comparisons show a roughly 2% fresh decode gain at both frontiers,
while whole-workload differences are mixed and 32K warm append is consistently
slower. Fixed-history agreement does not establish practical task quality.
Normal and fast presets and installed frontends remain unchanged. The selected
settings are in [candidate-settings.json](candidate-settings.json).

All ten guards completed with child return code 0, pressure level 1 only and
no adverse samples. Swap ranged from 29.011777 to 29.012207 GiB.

## Reproduction

Use `./zig-out/bin/ds4-bench` with matching `b7e7af9d` sources and the
recorded SHA-256 and checkpoint above. The drivers execute that binary, clear
inherited DS4 settings, and alternate timing order. Use fresh output directories,
run one guarded heavy model job at a time, and pause competing work during
timing. Diagnostic settings enable counters for quality, not timing.

```nu
let out = '.workspaces/hillclimb-20261104/32-cpu-expert-boundary-repro'
mkdir $out
let settings = $out
let selected = (open speed-bench/experiments/2026-11-05-slab-event-interaction/candidate-settings.json | merge {DS4_METAL_V41_CPU_EXPERT_FLUSH: '0'})
let combined = (open speed-bench/experiments/2026-11-05-cpu-expert-boundary/candidate-settings.json)
let fixed = ($combined | merge {DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY: '2', DS4_METAL_V41_ROUTE_SHARED_OVERLAP: '1'})
$selected | to json | save ($settings | path join selected-baseline.json)
$combined | to json | save ($settings | path join combined-candidate.json)
$fixed | to json | save ($settings | path join fixed.json)
$fixed | merge {DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: '1', DS4_METAL_V41_RESIDENT_ROUTE_PROFILE: '1', DS4_METAL_SELECTED_READBACK_PROFILE: '1'} | to json | save ($settings | path join diagnostic.json)
for case in [{name: '8k', frontier: 8192, cache: '64GB'} {name: '32k', frontier: 32768, cache: '64GB'} {name: '16gb', frontier: 8192, cache: '16GB'}] {
  for stage in ['quality' 'timing'] {
    let extra = if $stage == 'quality' { 'diagnostic.json' } else { 'fixed.json' }
    python3 speed-bench/ideas/pressure_run.py ($out | path join $'guard-($stage)-($case.name)') -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu ($out | path join $'($stage)-($case.name)') $stage --flag DS4_METAL_V41_CPU_EXPERT_FLUSH --control 0 --candidate 1 --frontier $case.frontier --cache $case.cache --tokens 128 --samples 5 --extra-settings ($settings | path join $extra)
  }
}
for frontier in [8192 32768] {
  python3 speed-bench/ideas/pressure_run.py ($out | path join $'guard-combined-($frontier)') -- nu speed-bench/experiments/2026-11-04-decode-pass/combined.nu ($out | path join $'combined-($frontier)') timing --baseline-root . --candidate-root . --baseline-revision b7e7af9d --candidate-revision b7e7af9d --baseline-settings ($settings | path join selected-baseline.json) --final-settings ($settings | path join combined-candidate.json) --frontier $frontier --tokens 128 --samples 5
}
python3 speed-bench/ideas/pressure_run.py ($out | path join guard-combined-quality-32768) -- nu speed-bench/experiments/2026-11-04-decode-pass/combined.nu ($out | path join combined-quality-32768) quality --baseline-root . --candidate-root . --baseline-revision b7e7af9d --candidate-revision b7e7af9d --baseline-settings ($settings | path join selected-baseline.json) --final-settings ($settings | path join combined-candidate.json) --frontier 32768 --tokens 128
```

Raw logs, settings, samples and guard records remain local and ignored under
the experiment workspace. The separate timeline uses its recorded diagnostic
settings; it is not a profile-off latency measurement.
