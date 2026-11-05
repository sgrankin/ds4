# Seed-write residency association

This experiment asks whether associating the existing expert slab residency set
with seed-copy command buffers changes the fresh-start delay observed with
compute-only command-buffer association. The extended policy remains unselected groundwork; package 25 retains queue
association. The one-hour slice is November 5,
2026, 06:50–07:50 Eastern.

The [preceding experiment](../2026-11-05-scoped-residency/README.md) compared
queue association with compute-only association at runtime revision `cd42c5ac`
and closed at `ee5ebdbc`. Its trace found that seed blits lacked explicit set
association and that the largest added gap preceded the first associated
expert buffer. This suggests deferred admission as a hypothesis; it does not
establish the driver cause. Adding association to seed writes could move cost
into prefill or the snapshot boundary rather than remove it.

## Change and comparison

The change calls the existing association helper on the actual seed-copy
command buffer after slab allocation and before the blit encoder. It adds no
queue, submission or warmup. Queue mode 1 remains the control; extended
command-buffer mode 2 is the candidate. Both arms use the same new binary and
package 25 settings: shared-event overlap 0, refresh 16, cadence 4 and cached
refresh parallelism off. The preceding compute-only mode 2 results are
historical evidence, not a concurrent third arm. The runtime is frozen at `d4c0d297`; the clean build used
`./tools/zig-build ds4-bench ssd-experts-test -j4`. Binary hashes are in the
local `runtime.json`.

The experimental package 25 uses queue mode 1. Ordinary and fast presets leave
residency off.

## Workload and timing scope

Runs use a 128 GiB M5 Max and
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`. At each frontier the
runner prefills, evaluates 128 teacher tokens, saves and restores a snapshot,
appends 512 tokens, and evaluates another 128 tokens. The expert cache survives
the snapshot and restore; the OS file cache was not flushed. Steady decode
excludes only the first token, so it does not by itself establish that all
initialization has finished. Whole prefill and complete generation retain those
costs. Inference total sums the reported prefill and generation phases; process
wall also includes startup, snapshot and other work outside those phase timers.
No separate practical task test was performed.

## Fixture checks

The existing queue, command and command with unretained command buffers
fixtures passed. In the seed/release fixture, command mode recorded 43 explicit
use calls per pass, compared with zero in experiment 29; memory remained stable
over ten passes. This confirms association on the exercised seed writes, not
its effect on model latency. The forced zero lock-limit fixture exited with
status 1 as expected: one add was followed by a clear on `mlock` failure, with
no re-add. The parent pressure guard exited successfully.

## Output checks

Fixed-history comparisons at 8K and 32K, and a 16 GB cache case repeating the
8K histories, each produced 256 byte-identical full-vocabulary teacher-logit
rows and 256 matching post-evaluation hashes between modes. All 768 row pairs
match; delta-NLL and KL are zero for these rows. These checks establish output
agreement for the exercised histories, not general quality equivalence.

The candidate recorded 22,116 explicit use calls at 8K, 22,121 at 32K and
21,282 in the smaller-cache case. These are calls, not unique command buffers.
The requested 16 GB case had an effective 8.88 GiB expert target after reserves;
both arms had 958 slots, 8.88 GiB of live cache and three slabs totaling 8.89 GiB. Each recorded
122,672 hits, 4,951 misses, 3,993 evictions and reuses, and 37.52 GiB of
cumulative logical preads. The set cleared only at final cleanup. An independent resource review found
matching arguments, routing and prefill
reports, no fallback or intermediate clear, and stable fixture memory near
186 MiB over ten seed/release passes in both retained and unretained modes.

## 32K timing

Five profile-off pairs compare queue mode 1 with extended command mode 2 on
the same binary. Mode 1 ran first in pairs 1, 3 and 5. All values below are
queue → command, and the final column gives mean ± sample standard deviation.
Times derived from reported throughput are approximate.

The first step was lower in all five pairs, so the large first-step regression
seen in the historical compute-only experiment is absent here. Complete fresh
decode was lower in two pairs; candidate pair 3 took 4.003 seconds with
33.24 steady tokens/s and is retained. Warm complete decode was lower in four
pairs. Whole inference was lower only in pair 2, when the candidate ran first;
pair 4 was effectively flat. Drift and order limit the comparison.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 42.408 → 43.179 | 44.673 → 43.806 | 45.546 → 46.210 | 46.428 → 46.534 | 46.583 → 47.022 | 45.128 ± 1.702 → 45.350 ± 1.734 |
| Fresh first step, ms | 202.588 → 166.838 | 207.361 → 205.935 | 191.767 → 182.438 | 197.380 → 176.719 | 187.465 → 186.531 | 197.312 ± 8.008 → 183.692 ± 14.466 |
| Fresh complete decode, s | 3.771 → 3.802 | 3.778 → 3.769 | 3.755 → 4.003 | 3.760 → 3.745 | 3.737 → 3.794 | 3.760 ± 0.016 → 3.822 ± 0.103 |
| Fresh steady decode, tokens/s | 35.590 → 34.940 | 35.570 → 35.650 | 35.650 → 33.240 | 35.650 → 35.600 | 35.780 → 35.200 | 35.648 ± 0.082 → 34.926 ± 0.987 |
| Warm append, s | 7.051 → 7.391 | 7.358 → 7.056 | 7.617 → 7.729 | 7.884 → 7.811 | 7.824 → 8.118 | 7.547 ± 0.345 → 7.621 ± 0.408 |
| Warm complete decode, s | 3.858 → 3.675 | 3.860 → 3.600 | 3.912 → 3.920 | 3.965 → 3.952 | 4.003 → 3.985 | 3.920 ± 0.064 → 3.826 ± 0.176 |
| Inference total, s | 57.089 → 58.047 | 59.670 → 58.231 | 60.829 → 61.862 | 62.038 → 62.041 | 62.147 → 62.918 | 60.355 ± 2.086 → 60.620 ± 2.301 |
| Process wall, s | 60.183 → 60.711 | 62.267 → 60.881 | 63.422 → 64.447 | 64.617 → 64.571 | 64.680 → 65.483 | 63.034 ± 1.876 → 63.218 ± 2.248 |

## 8K timing

The same five-pair procedure at 8K used the same arm order and units as the
32K table. First-step and complete fresh-decode times were lower in two pairs;
whole inference and process wall were lower in one pair each. Candidate timings
varied more, and all samples are retained.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 19.236 → 18.374 | 18.625 → 18.482 | 18.727 → 19.027 | 19.062 → 19.216 | 19.144 → 19.364 | 18.959 ± 0.268 → 18.893 ± 0.442 |
| Fresh first step, ms | 193.611 → 205.315 | 207.636 → 182.062 | 172.174 → 181.166 | 199.955 → 182.731 | 170.885 → 182.311 | 188.852 ± 16.581 → 186.717 ± 10.412 |
| Fresh complete decode, s | 3.681 → 3.941 | 3.689 → 3.664 | 3.668 → 3.712 | 3.691 → 3.673 | 3.653 → 3.664 | 3.676 ± 0.016 → 3.731 ± 0.119 |
| Fresh steady decode, tokens/s | 36.420 → 34.000 | 36.480 → 36.470 | 36.330 → 35.970 | 36.380 → 36.390 | 36.480 → 36.470 | 36.418 ± 0.065 → 35.860 ± 1.060 |
| Warm append, s | 6.707 → 7.438 | 7.151 → 6.741 | 7.267 → 7.767 | 7.441 → 7.626 | 7.594 → 7.621 | 7.232 ± 0.339 → 7.439 ± 0.407 |
| Warm complete decode, s | 3.604 → 3.767 | 3.796 → 3.719 | 3.833 → 3.931 | 3.882 → 3.900 | 3.872 → 3.867 | 3.797 ± 0.114 → 3.837 ± 0.090 |
| Inference total, s | 33.228 → 33.520 | 33.260 → 32.606 | 33.494 → 34.438 | 34.076 → 34.415 | 34.263 → 34.517 | 33.664 ± 0.477 → 33.899 ± 0.830 |
| Process wall, s | 35.649 → 36.088 | 35.657 → 35.063 | 35.973 → 36.910 | 36.465 → 36.902 | 36.700 → 36.927 | 36.089 ± 0.477 → 36.378 ± 0.817 |

## Diagnostic traces

In one intrusive 32K, 17-token timeline pair, all 40 recorded seed-fill command
buffers were associated in mode 2, with the first association at sequence 9.
All 34 output hashes matched; refresh and ordinary windows retained 91 and 11
command buffers respectively in both arms. The gap before the first expert
buffer was 0.642 → 0.658 ms. The roughly 815 ms gap seen at that boundary in
the historical compute-only trace was absent. This confirms coverage in the
trace but does not establish physical residency or a driver cause; the pairs
were separate runs and profiling changes timing.

A separate native 128-token diagnostic pair at 32K measured 270.089 ms more
fresh generation time and 224.769 ms more warm generation time in the
candidate. Most of the added recurring time appeared on ordinary tokens rather
than refresh tokens. This single diagnostic pair locates an observed cost; it
is not another comparative performance sample.

## Decision

Keep extended mode 2 as unselected groundwork; package 25 continues to use
queue mode 1. The large historical compute-only first-step regression is absent in the
current comparison, but the historical mode was not a concurrent arm, so this
does not isolate the cause. These comparisons do not establish a whole-workload
gain. The trace and native diagnostic locate differences but do not change
that decision.

## Reproduction

Build `zig-out/bin/ds4-bench` at `d4c0d297` and use the same checkpoint and
[package 25 settings](../2026-11-05-slab-event-interaction/candidate-settings.json).
The driver clears inherited DS4 settings, alternates order and defaults to a
64 GB cache. Quality runs add diagnostic counters; timings do not. Run guarded
jobs sequentially, without competing heavy model jobs or builds during timing.
Use fresh output directories.

```nu
let out = '.workspaces/hillclimb-20261104/30-seed-residency-repro'
let fixed = 'speed-bench/experiments/2026-11-05-slab-event-interaction/candidate-settings.json'
mkdir $out
open $fixed | merge {DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: '1', DS4_METAL_V41_RESIDENT_ROUTE_PROFILE: '1', DS4_METAL_SELECTED_READBACK_PROFILE: '1'} | to json | save ($out | path join diagnostic.json)
for frontier in [8192 32768] {
  python3 speed-bench/ideas/pressure_run.py ($out | path join $'guard-timing-($frontier)') -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu ($out | path join $'timing-($frontier)') timing --flag DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY --control 1 --candidate 2 --frontier $frontier --tokens 128 --samples 5 --extra-settings $fixed
  python3 speed-bench/ideas/pressure_run.py ($out | path join $'guard-quality-($frontier)') -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu ($out | path join $'quality-($frontier)') quality --flag DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY --control 1 --candidate 2 --frontier $frontier --tokens 128 --extra-settings ($out | path join diagnostic.json)
}
python3 speed-bench/ideas/pressure_run.py ($out | path join guard-quality-16gb) -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu ($out | path join quality-16gb) quality --flag DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY --control 1 --candidate 2 --frontier 8192 --cache 16GB --tokens 128 --extra-settings ($out | path join diagnostic.json)
```

All eight guarded jobs completed with child return code 0, pressure level 1
only and no adverse samples. Swap ranged from 29.0034 to 29.0112 GiB. Raw
artifacts, guard records and all samples are retained locally under
`.workspaces/hillclimb-20261104/30-seed-residency/`.
