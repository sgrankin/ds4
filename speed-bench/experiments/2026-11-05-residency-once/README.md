# Associate expert residency once per batch

This experiment asks whether avoiding repeated association of the expert slab
residency set with the same active command buffer reduces decode or append
cost. The selected package's earlier profile recorded 4,361 explicit residency
use calls on 850 unique command buffers. That repetition identifies work to
investigate; call counts alone do not establish a latency gain. The one-hour
slice is November 5, 2026, 09:16–10:16 Eastern. Retain the prototype as opt-in but unselected: the ordinary retained-buffer
measurements do not establish a consistent net gain.

## Change under review

Runtime `47e461b0` enables deduplication only when
`DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY_ONCE=1`. An engine-owned byte tracks
whether the current batch has actually received the association. Creating a new
batch resets it, including scheduler and direct event paths; committing or
clearing the set also resets it conservatively. Standalone command buffers
retain their existing behavior. Conservative invalidation means a batch can receive another association after
a set commit; this is once per unchanged batch and set state, not an absolute
one-call limit per command buffer. The completion capture for an unretained
buffer remains paired with the actual association. The local source and API
review is retained in `.workspaces/hillclimb-20261104/33-residency-once/source-review.md`;
the final source review found no design blocker. Apple documents that an
association covers the command buffer and that committed changes apply to
the mutable set. Set changes do not synchronize CPU and GPU use or provide
hazard tracking. See [Metal residency sets](https://developer.apple.com/documentation/metal/mtlresidencyset?language=objc).

## Comparison and workload

The baseline is the [selected boundary package](../2026-11-05-cpu-expert-boundary/candidate-settings.json),
with runtime `b7e7af9d` and closure `fd1a2e74`. Both arms use the same new
binary, SHA-256 `d5995372a55dcb95a612fef4d0ed816b68606a7918e5f770b11bbea673de3c47`, with the flag set to 0 or 1 and all selected package settings fixed,
including residency mode 2, overlap, the CPU-expert boundary, refresh 16 and
cadence 4. This same-binary comparison isolates the flag. The frozen-build
comparison below checks the net effect against package 32.

The completed checks used bounded cache fixtures, fixed-history comparisons at
8K and 32K with a requested 64 GB cache and at 8K with a requested 16 GB cache,
then five alternating profile-off timing pairs for each configuration. The
workload and phase boundaries follow the [boundary experiment](../2026-11-05-cpu-expert-boundary/README.md):
prefill, 128 teacher tokens, snapshot and restore, a 512-token append, and
another 128 teacher tokens. The expert cache survives restore and the OS file
cache is not flushed. Phase timers exclude startup and snapshot operations;
process wall includes them. Steady decode excludes only the first token, so
first-step latency and the complete decode windows must also be inspected.

A separate 32K timeline compared association calls and associated
buffer coverage and inspected shifted CPU and GPU gaps. Profiling is
diagnostic and does not provide the profile-off latency comparison. Append
cost must be measured alongside decode. Fixed-history agreement does not by
itself establish practical task quality.

## 32K timeline

An intrusive profile pair reduced actual residency calls from 4,361 to 863
while preserving the same 850 associated command buffers and seed and event
coverage. Fresh seed writes accounted for 1,001 → 53 calls, 34 decode tokens
for 2,720 → 490, and the 512-token append for 640 → 320. Thirteen seed-growth
buffers had two calls after conservative invalidation; the other 837 had one.
All 34 output hashes and command counts matched, with 91 command buffers per
refresh and 11 per ordinary token. Encoder-group multisets matched for 33 of
34 tokens; one keepalive group was attributed differently.

Sampled GPU windows were largely unchanged. The profile has no dedicated API
CPU timer, and the candidate also printed 3,498 fewer trace lines, so it cannot
isolate API cost or establish a speed gain. The profile notes and raw traces
are retained locally under the experiment workspace.

## 32K timing

Five alternating profile-off pairs compare flag 0 → 1 on the same binary;
control ran first in pairs 1, 3 and 5. Phase seconds and inference totals are derived
from throughput rounded to two decimal places. The first control prefill was unusually fast and
the first candidate fresh decode was slower. The first arm had faster prefill
in every pair, indicating an order effect or drift without establishing its
cause. Fresh decode favored the
candidate in three pairs, warm decode in two, and inference total and wall in
two each. The mixed and variable results do not establish a consistent gain.
Values are flag 0 → 1; the final column gives mean ± sample standard deviation.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 42.835 → 43.641 | 46.590 → 45.889 | 46.717 → 47.124 | 47.452 → 47.101 | 47.588 → 48.046 | 46.236 ± 1.951 → 46.360 ± 1.702 |
| Fresh first step, ms | 193.935 → 232.904 | 195.747 → 182.390 | 191.894 → 170.817 | 190.743 → 173.155 | 170.312 → 171.923 | 188.526 ± 10.361 → 186.238 ± 26.488 |
| Fresh complete decode, s | 3.680 → 3.957 | 3.700 → 3.696 | 3.686 → 3.666 | 3.716 → 3.691 | 3.675 → 3.679 | 3.691 ± 0.017 → 3.738 ± 0.123 |
| Warm append, s | 7.178 → 7.733 | 7.669 → 7.435 | 7.763 → 7.780 | 7.889 → 7.767 | 7.569 → 7.739 | 7.614 ± 0.271 → 7.691 ± 0.144 |
| Warm first step, ms | 32.576 → 32.395 | 34.884 → 33.619 | 55.221 → 33.379 | 34.371 → 33.048 | 33.497 → 33.717 | 38.110 ± 9.606 → 33.232 ± 0.534 |
| Warm complete decode, s | 3.538 → 3.799 | 3.934 → 3.823 | 3.947 → 3.923 | 3.828 → 3.945 | 3.851 → 3.953 | 3.819 ± 0.166 → 3.889 ± 0.072 |
| Inference total, s | 57.231 → 59.130 | 61.893 → 60.844 | 62.113 → 62.492 | 62.884 → 62.503 | 62.683 → 63.417 | 61.361 ± 2.344 → 61.677 ± 1.700 |
| Process wall, s | 60.166 → 61.857 | 64.431 → 63.528 | 64.742 → 64.977 | 65.550 → 65.008 | 65.287 → 66.023 | 64.035 ± 2.208 → 64.278 ± 1.620 |

## 8K timing

Five alternating profile-off pairs compare flag 0 → 1 with the same order
and units as the 32K comparison. Fresh complete decode was lower in one pair;
fresh steady throughput was lower in all five. Warm complete decode, inference
total and wall were lower in two pairs each. The warm mean is affected by
variation and does not establish a consistent gain.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 18.097 → 18.099 | 18.440 → 18.867 | 19.377 → 19.849 | 19.267 → 19.500 | 19.329 → 19.262 | 18.902 ± 0.592 → 19.115 ± 0.671 |
| Fresh first step, ms | 178.814 → 202.653 | 179.130 → 173.355 | 187.305 → 210.652 | 207.514 → 180.976 | 186.826 → 220.954 | 187.918 ± 11.680 → 197.718 ± 20.034 |
| Fresh complete decode, s | 3.589 → 3.631 | 3.596 → 3.607 | 3.606 → 3.658 | 3.620 → 3.610 | 3.608 → 3.647 | 3.604 ± 0.012 → 3.630 ± 0.023 |
| Fresh steady decode, tokens/s | 37.230 → 37.040 | 37.180 → 36.990 | 37.150 → 36.840 | 37.220 → 37.040 | 37.130 → 37.080 | 37.182 ± 0.043 → 36.998 ± 0.094 |
| Warm append, s | 6.691 → 6.840 | 7.283 → 6.708 | 7.878 → 8.057 | 7.734 → 7.899 | 7.668 → 7.642 | 7.451 ± 0.478 → 7.429 ± 0.618 |
| Warm first step, ms | 31.320 → 31.716 | 32.603 → 31.366 | 32.945 → 36.548 | 32.975 → 33.090 | 32.832 → 32.602 | 32.535 ± 0.695 → 33.064 ± 2.065 |
| Warm complete decode, s | 3.429 → 3.474 | 3.848 → 3.463 | 3.920 → 3.930 | 3.873 → 3.894 | 3.859 → 3.825 | 3.786 ± 0.201 → 3.717 ± 0.230 |
| Inference total, s | 31.806 → 32.044 | 33.167 → 32.644 | 34.781 → 35.494 | 34.494 → 34.902 | 34.463 → 34.376 | 33.742 ± 1.249 → 33.892 ± 1.483 |
| Process wall, s | 34.238 → 34.608 | 35.650 → 35.149 | 37.319 → 37.947 | 37.000 → 37.403 | 36.972 → 36.814 | 36.236 ± 1.288 → 36.384 ± 1.445 |

## 8K timing with a smaller cache

Five alternating profile-off pairs requested a 16 GB cache, with the same
order and units as the other comparisons. Fresh complete decode was lower
with the flag in four pairs, warm complete decode in two. Inference total and
wall were lower in four pairs, amid prefill variation.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 18.185 → 18.193 | 18.918 → 18.396 | 19.318 → 19.605 | 19.909 → 19.732 | 20.095 → 19.936 | 19.285 ± 0.773 → 19.172 ± 0.813 |
| Fresh first step, ms | 212.421 → 190.683 | 175.700 → 179.235 | 209.697 → 195.114 | 201.693 → 203.954 | 184.587 → 198.589 | 196.820 ± 16.036 → 193.515 ± 9.344 |
| Fresh complete decode, s | 3.722 → 3.693 | 3.710 → 3.692 | 3.725 → 3.709 | 3.731 → 3.721 | 3.710 → 3.711 | 3.720 ± 0.009 → 3.705 ± 0.012 |
| Fresh steady decode, tokens/s | 36.190 → 36.260 | 35.930 → 36.160 | 36.130 → 36.140 | 35.990 → 36.110 | 36.020 → 36.150 | 36.052 ± 0.106 → 36.164 ± 0.057 |
| Warm append, s | 8.443 → 8.149 | 8.518 → 8.317 | 8.662 → 8.787 | 8.858 → 8.841 | 8.880 → 8.784 | 8.672 ± 0.196 → 8.576 ± 0.319 |
| Warm first step, ms | 42.257 → 39.599 | 42.142 → 40.227 | 40.794 → 41.434 | 44.056 → 43.648 | 41.496 → 41.406 | 42.149 ± 1.216 → 41.263 ± 1.548 |
| Warm complete decode, s | 3.805 → 3.780 | 3.836 → 3.793 | 3.867 → 3.886 | 3.867 → 3.888 | 3.871 → 3.892 | 3.849 ± 0.028 → 3.848 ± 0.056 |
| Inference total, s | 34.155 → 33.815 | 34.982 → 34.197 | 35.572 → 35.986 | 36.365 → 36.183 | 36.556 → 36.323 | 35.526 ± 0.992 → 35.301 ± 1.195 |
| Process wall, s | 35.312 → 35.003 | 36.218 → 35.397 | 36.730 → 37.232 | 37.588 → 37.461 | 37.819 → 37.548 | 36.733 ± 1.024 → 36.528 ± 1.226 |

## Frozen-build comparison at 32K

Five alternating pairs compare frozen `b7e7af9d` with `47e461b0` and the
flag enabled. The old binary SHA-256 is
`fd4596242336958607ae44f302e77bdc39e4a0ae1a28830df4d93da38ce88921`;
the selected settings are otherwise fixed. The old build ran first in pairs
1, 3 and 5. The first old-build run was unusually slow; its cause is unknown.
Fresh and warm complete decode, inference total and wall were lower with the
candidate in three pairs each. The first pair strongly affects the means, so
these results do not establish a consistent net gain. Units and derivations
match the preceding tables.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 48.499 → 45.658 | 47.218 → 46.485 | 47.392 → 47.562 | 47.631 → 48.864 | 47.770 → 47.748 | 47.702 ± 0.494 → 47.263 ± 1.232 |
| Fresh first step, ms | 196.702 → 190.321 | 198.735 → 216.965 | 197.026 → 152.376 | 180.403 → 216.960 | 221.824 → 192.233 | 198.938 ± 14.799 → 193.771 ± 26.474 |
| Fresh complete decode, s | 3.952 → 3.712 | 3.702 → 3.711 | 3.694 → 3.651 | 3.673 → 3.717 | 3.713 → 3.702 | 3.747 ± 0.116 → 3.699 ± 0.027 |
| Fresh steady decode, tokens/s | 33.820 → 36.060 | 36.260 → 36.350 | 36.310 → 36.300 | 36.360 → 36.290 | 36.380 → 36.190 | 35.826 ± 1.122 → 36.238 ± 0.115 |
| Warm append, s | 8.014 → 7.202 | 7.808 → 7.666 | 7.878 → 7.786 | 7.867 → 7.891 | 7.830 → 7.905 | 7.880 ± 0.080 → 7.690 ± 0.289 |
| Warm first step, ms | 34.416 → 33.473 | 33.798 → 50.918 | 33.477 → 33.169 | 36.907 → 33.874 | 36.392 → 36.645 | 34.998 ± 1.556 → 37.616 ± 7.564 |
| Warm complete decode, s | 3.823 → 3.852 | 3.935 → 3.908 | 3.945 → 3.864 | 3.960 → 3.959 | 3.948 → 3.949 | 3.922 ± 0.056 → 3.906 ± 0.049 |
| Inference total, s | 64.288 → 60.424 | 62.663 → 61.770 | 62.908 → 62.863 | 63.131 → 64.431 | 63.262 → 63.304 | 63.250 ± 0.623 → 62.558 ± 1.527 |
| Process wall, s | 71.039 → 63.146 | 65.254 → 64.491 | 65.548 → 65.508 | 65.730 → 66.991 | 65.855 → 65.946 | 66.685 ± 2.445 → 65.216 ± 1.465 |

## Results and decision

The build completed. Eight existing cache fixture variants returned their expected
exit codes, including the forced lock-limit case, which returned the expected exit code 1
rather than an output-check pass; their guard returned 0. In
the seed-release fixture, the enabled flag reduced actual residency use calls
from 43 to 1 per pass with an approximately 186 MiB footprint in both arms.
The fixture binary SHA-256 is
`c198e93e2f09cba86c8135fc0ce8ee732962b3ea8ed03172f6f5ee7039020453`.
At 8K and 32K with the requested 64 GB cache and at 8K with a requested
16 GB cache, each fixed-history comparison produced 256 byte-identical
full-vocabulary rows and matching hashes. The smaller-cache case had an
8.88 GiB effective expert target, 3,993 evictions and reuses, and five split
layers. The frozen-build comparison is reported above. Reusing the
32K histories from the preceding experiment and this one also yielded 256
identical rows and hashes across revisions; this is not a new quality run.
No new practical-task quality checks have been run.

Retain the prototype unselected for one bounded follow-up using unretained
command buffers. In that mode deduplication removes repeated completion-handler
captures as well as API calls. The current profile exposes no new stall and
does not establish a future speedup. Package 32 remains selected; presets,
installed frontends and main remain unchanged.

All nine guards returned child code 0, observed pressure level 1 only and
recorded no adverse samples. Swap ranged from 29.005488 to 29.013301 GiB.
Raw settings, the frozen prior binary, logs and samples are retained locally
under `.workspaces/hillclimb-20261104/33-residency-once/`.

## Reproduction

Use the recorded checkpoint and matching source revisions. Both drivers read
`zig-out/bin/ds4-bench` from their source roots: the current root must contain
the recorded `47e461b0` build. For the frozen-build comparison, prepare a
checkout of `b7e7af9d` at the baseline root shown below, with its recorded
binary at `zig-out/bin/ds4-bench`; rebuilding there with
`./tools/zig-build ds4-bench -j4` reproduces that revision. The drivers clear
inherited DS4 settings and alternate the arms. Use fresh output
directories, the pressure guard and one heavy job at a time; pause competing
work during profile-off timing. The local workspace retains the exact settings,
fixture driver, timeline configuration, commands and raw results. The quality
comparison across revisions reused existing 32K dumps, rather than running a
new independent history.

```nu
let out = '.workspaces/hillclimb-20261104/33-residency-once-repro'
mkdir $out
let fixed = (open speed-bench/experiments/2026-11-05-cpu-expert-boundary/candidate-settings.json)
$fixed | to json | save ($out | path join fixed.json)
$fixed | merge {DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: '1', DS4_METAL_V41_RESIDENT_ROUTE_PROFILE: '1', DS4_METAL_SELECTED_READBACK_PROFILE: '1'} | to json | save ($out | path join diagnostic.json)
$fixed | merge {DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY_ONCE: '1'} | to json | save ($out | path join candidate.json)
for case in [{name: '8k', frontier: 8192, cache: '64GB'} {name: '32k', frontier: 32768, cache: '64GB'} {name: '16gb', frontier: 8192, cache: '16GB'}] {
  for stage in ['quality' 'timing'] {
    let settings = if $stage == 'quality' { 'diagnostic.json' } else { 'fixed.json' }
    python3 speed-bench/ideas/pressure_run.py ($out | path join $'guard-($stage)-($case.name)') -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu ($out | path join $'($stage)-($case.name)') $stage --flag DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY_ONCE --control 0 --candidate 1 --frontier $case.frontier --cache $case.cache --tokens 128 --samples 5 --extra-settings ($out | path join $settings)
  }
}
python3 speed-bench/ideas/pressure_run.py ($out | path join guard-net32k) -- nu speed-bench/experiments/2026-11-04-decode-pass/combined.nu ($out | path join net32k) timing --baseline-root .workspaces/hillclimb-20261104/33-residency-once/before-root --candidate-root . --baseline-revision b7e7af9d --candidate-revision 47e461b0 --baseline-settings ($out | path join fixed.json) --final-settings ($out | path join candidate.json) --frontier 32768 --tokens 128 --samples 5
```

The separate 32K timeline uses the locally retained diagnostic settings and
is not a profile-off latency measurement.
