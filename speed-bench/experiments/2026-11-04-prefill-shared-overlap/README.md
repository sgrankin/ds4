# Shared-expert overlap during selected prefill

This experiment found no reliable benefit sufficient to promote batch shared
overlap into the fast presets. The redundant-wait removal is retained in the
existing opt-in path; it does not establish a default-path speedup. The first
one-hour slice
runs on November 4, 2026, from 08:37 to 09:37 Eastern.

The [previous dependency profile](../2026-11-03-post-rope-profile/README.md)
found long delays before routed-expert GPU commands in a warm append. With
batch overlap off, the selected backend waits for shared-expert work before
preparing selected cache entries. The hypothesis is that preparing those
entries while shared work runs can hide some of that preparation. The earlier
[SSD experiment](../ds41-ssd/README.md#30-overlap-selected-batch-ssd-binding-with-shared-expert-gpu-work)
introduced this path as an opt-in; its historical snapshot checks do not
establish performance for the current baseline.

In [`Moe.batch`](../../../src/engine/deepseek41/moe.zig), the eligible streaming
path completes the router and marks its selected IDs ready before queuing
shared GPU work. The selected backend can then flush the shared work and prepare
expert cache entries on the host while it runs, before submitting routed work
behind it. The final wait protecting shared and routed work remains. This is
cache preparation overlap, distinct from the CPU resident-routing substitution
performed earlier by [`route_services.prefill`](../../../src/engine/deepseek41/route_services.zig).
The feature applies only when the streaming selected-batch backend supports
the shape and the call does not force resident execution.

## Setup

Both arms use published runtime revision `364ceeab`; no runtime change is
required for this first comparison. The baseline omits
`DS4_METAL_V41_BATCH_SHARED_OVERLAP`, while the candidate sets it to `1`.
The flag is presence-based, so setting it to `0` would also enable it. Driver
revision `36f8d24b` extends the existing
[decode-pass runner](../2026-11-04-decode-pass/run.nu) with `--control-unset`
to remove it from the control environment. The runner clears inherited DS4
settings and records the effective environment and arguments for each run.

The workload uses an M5 Max with 128 GiB RAM and the Flash 1017 IQ2_XXS/Q2_K
checkpoint, a 64 GB expert-cache request, 8,192-token prefill chunks and the
adopted routing and decode settings in the runner. It compares fresh prefill,
a 512-token append after restoring the prompt frontier, and complete
teacher-forced decode windows at 8K and 32K frontiers. Fresh processes start
with empty process-local expert caches, which survive the KV frontier restore;
the OS file cache is not flushed.
Profile-free timing uses five alternating pairs per comparison under the pressure
guard. The original flag is compared at 32K; a variant that avoids a redundant
drain was evaluated at 32K and 8K. Raw logs and outputs are retained locally under
`.workspaces/hillclimb-20261104/01-prefill-overlap/`.

## Observations

The 8K quality comparison completed: all 256 paired logit rows were
byte-identical, as were all 256 post-evaluation hashes; delta-NLL and KL were
zero for every compared row. This checks the teacher-forced histories in that
run, not practical task quality or other workloads.

A single intrusive timeline diagnostic per arm recorded 288,572 encoders for
each warm append. Command buffers rose from 1,609 to 1,929, including 320
additional empty readiness drains. The append envelope was 16.737 s versus
16.829 s, and the union of encoder intervals was 6.385 s versus 6.374 s.
Instrumentation perturbs these runs; they locate behavior and are not a
performance comparison. In the baseline append, 320 shared-command waits
totaled 365.1 ms, including 179.6 ms GPU time. Routed-command waits totaled
9,199.9 ms, including 2,655.0 ms GPU time; 23 exceeded 100 ms. In the original
candidate, routed waits totaled 9,640.5 ms, including 2,675.0 ms GPU time;
32 exceeded 100 ms. Thus the observed shared stage provides only a few tenths
of a second of overlap opportunity in this trace, while multi-second routed
wait gaps remain. The trace does not establish the cause of those gaps.

## Original flag: 32K timing

Five profile-free pairs show no reliable prefill gain from the existing flag.
The candidate append was faster in three pairs and slower in two. Its slower
mean is dominated by pair 2, when the candidate ran first and took 13.789 s;
the remaining candidate appends were 7.426–7.596 s. This variation does not
establish a regression. Times below are seconds, in pair order, with mean ±
sample standard deviation. Phase times exclude model loading and snapshot
save/restore and are derived from throughput rounded to two decimal places.
Each decode window contains 128 tokens; steady decode excludes the first token.

| Phase | Baseline samples | Candidate samples | Baseline mean ± SD | Candidate mean ± SD |
| --- | --- | --- | ---: | ---: |
| Fresh prefill | 46.137, 48.782, 47.893, 50.336, 48.355 | 47.112, 49.677, 48.088, 48.146, 48.291 | 48.301 ± 1.518 | 48.263 ± 0.918 |
| Warm 512-token append | 7.029, 7.581, 7.641, 7.577, 7.518 | 7.426, 13.789, 7.596, 7.576, 7.426 | 7.469 ± 0.250 | 8.763 ± 2.811 |
| Fresh full decode | 4.633, 4.748, 4.697, 4.744, 4.769 | 4.778, 5.000, 4.722, 4.778, 4.690 | 4.718 ± 0.055 | 4.794 ± 0.121 |
| Warm full decode | 3.677, 3.807, 3.833, 3.879, 3.830 | 4.267, 4.476, 3.843, 3.829, 3.812 | 3.805 ± 0.076 | 4.045 ± 0.307 |
| Fresh steady decode | 3.711, 3.785, 3.783, 3.854, 3.824 | 3.818, 4.051, 3.808, 3.840, 3.816 | 3.792 ± 0.054 | 3.867 ± 0.104 |
| Warm steady decode | 3.644, 3.727, 3.763, 3.800, 3.797 | 4.201, 4.025, 3.751, 3.766, 3.778 | 3.746 ± 0.064 | 3.904 ± 0.201 |
| All inference phases | 61.476, 64.917, 64.065, 66.536, 64.473 | 63.583, 72.942, 64.248, 64.329, 64.219 | 64.293 ± 1.833 | 65.864 ± 3.968 |

The flag is read only in the batch path; scalar decode uses a separate overlap
setting and one-row batches do not support this batch path. The diagnostic
traces have identical command-buffer, encoder and router counts per decode
token and the same kernel set in both phases. These observations do not explain
the early decode differences in the timing samples.

## Guarded variant

The diagnostic's 320 extra empty drains follow a resident-prefill operation
that has already completed the selected IDs. The guarded variant at prototype revision `689c2d51` skips
that second drain only when resident prefill is active with a positive gap. It
retains the readiness wait for the native path and the final cache-lifetime
wait. Five-pair 32K and 8K comparisons and quality checks cover this variant. The
driver adds `--prefill-gap` to exercise the native fallback with gap `0`. The
8K quality run completed with 256 byte-identical logit rows and 256
matching post-evaluation hashes. The native-routing fallback with prefill gap
zero also produced 256 byte-identical full-vocabulary rows and 256 matching
post-evaluation hashes, with zero delta-NLL and KL. The 16 GB cache request
likewise produced 256 byte-identical rows and 256 matching hashes, with zero
delta-NLL and KL. Across the three optimized cases, all 768 rows and 768
hashes matched. An independent source audit reported no
findings.

A separate intrusive candidate timeline confirms that the guarded variant
removes the 320 extra empty command buffers: warm append and restore used
1,609 command buffers, matching the original control, with 288,572 encoders
unchanged. Its 8.216 s envelope comprised a 6.433 s encoder union and 1.783 s
uncovered. The original diagnostic envelope was approximately 16.8 s, but
these separate instrumented runs and their large variation do not establish a
speed difference.

The 32K timing run was interrupted during the fifth baseline at 09:08:51–52,
after the pressure guard observed level 2 twice. Four pairs completed; no
fifth candidate ran. Wired memory rose from approximately 49 to 82 GiB and
compressed memory from 18 to 26 GiB immediately before the stop; swap stayed
approximately 24.965 GiB. The guard reported that child PID 41531 was not
promptly reaped, but subsequent process inspection found no remaining
`ds4-bench` processes. Earlier checks found no competing heavy jobs or thermal
warnings. Pressure subsequently returned to level 1 with no `ds4-bench` process alive.
The fifth pair was retried with the same source, settings and order under
a new guard and raw directory, `reuse-timing32k-retry`; the interrupted attempt
remains intact. The symlink-only `reuse-timing32k-completed` analysis directory
combines the first four completed pairs with the retry as pair 5, recording
the mapping in `sources.json`.

The optimized timing comparisons use flag-off control versus flag-on with the
guard; they do not isolate the speed effect of the guard itself. The completed
32K comparison shows no reliable benefit. Both arms varied
substantially; these samples do not identify the cause of the stalls or
establish a regression. Seconds below are derived from rounded throughput, in
pair order; mean ± sample standard deviation is shown. Decode windows contain
128 tokens; steady decode excludes the first token. All inference phases
exclude loading and snapshot save/restore.

| Phase | Baseline samples | Candidate samples | Baseline mean ± SD | Candidate mean ± SD |
| --- | --- | --- | ---: | ---: |
| Fresh prefill | 48.819, 60.918, 55.481, 56.492, 48.756 | 53.622, 57.479, 52.664, 53.287, 47.336 | 54.093 ± 5.257 | 52.878 ± 3.628 |
| Warm 512-token append | 7.399, 8.436, 10.822, 8.016, 7.110 | 16.005, 15.648, 7.680, 8.076, 7.547 | 8.357 ± 1.472 | 10.991 ± 4.420 |
| Fresh full decode | 6.578, 4.780, 5.102, 4.931, 5.108 | 5.498, 5.163, 4.810, 4.823, 4.984 | 5.299 ± 0.727 | 5.056 ± 0.286 |
| Warm full decode | 3.823, 4.434, 4.646, 4.159, 3.822 | 4.639, 4.284, 4.016, 4.487, 3.940 | 4.177 ± 0.367 | 4.273 ± 0.299 |
| Fresh steady decode | 4.598, 3.750, 3.752, 3.890, 3.780 | 4.279, 4.107, 3.829, 3.822, 3.872 | 3.954 ± 0.365 | 3.982 ± 0.203 |
| Warm steady decode | 3.780, 4.292, 3.963, 4.073, 3.762 | 4.056, 3.818, 3.954, 4.312, 3.860 | 3.974 ± 0.220 | 4.000 ± 0.197 |
| All inference phases | 66.619, 78.568, 76.051, 73.597, 64.796 | 79.765, 82.574, 69.170, 70.672, 63.807 | 71.926 ± 5.978 | 73.198 ± 7.775 |

The five-pair 8K comparison was completed without the 32K interruption. Its
append mean was 1.1% lower and its total inference mean 0.6% lower with the
candidate; both changes are small relative to the observed variation. Times
are seconds in pair order, with mean ± sample standard deviation.

| Phase | Baseline samples | Candidate samples | Baseline mean ± SD | Candidate mean ± SD |
| --- | --- | --- | ---: | ---: |
| Fresh prefill | 21.098, 19.621, 19.669, 20.177, 20.222 | 19.925, 19.830, 20.114, 20.252, 20.236 | 20.157 ± 0.595 | 20.071 ± 0.188 |
| Warm 512-token append | 7.220, 6.625, 7.338, 6.973, 7.053 | 7.091, 7.147, 6.739, 6.812, 7.035 | 7.042 ± 0.273 | 6.965 ± 0.179 |
| Fresh full decode | 4.720, 4.391, 4.663, 4.648, 4.446 | 4.583, 4.616, 4.491, 4.463, 4.455 | 4.574 ± 0.145 | 4.522 ± 0.073 |
| Warm full decode | 3.949, 3.660, 3.798, 3.755, 3.769 | 3.722, 3.797, 3.830, 3.796, 3.798 | 3.786 ± 0.105 | 3.789 ± 0.040 |
| Fresh steady decode | 3.808, 3.562, 3.767, 3.587, 3.586 | 3.721, 3.754, 3.595, 3.582, 3.586 | 3.662 ± 0.116 | 3.647 ± 0.083 |
| Warm steady decode | 3.915, 3.610, 3.764, 3.722, 3.736 | 3.690, 3.764, 3.797, 3.763, 3.746 | 3.750 ± 0.109 | 3.752 ± 0.039 |
| All inference phases | 36.988, 34.297, 35.469, 35.553, 35.490 | 35.321, 35.389, 35.174, 35.323, 35.525 | 35.559 ± 0.955 | 35.347 ± 0.127 |

The decision is to keep the redundant-wait removal within the existing opt-in
path, without promoting batch overlap into the fast presets; the 32K stalls
remain unexplained.

## Reproduction

Build the desired runtime revision and run one guarded process at a time. The
original flag comparison uses runtime `364ceeab`; the guarded variant uses
`689c2d51`. Use the current driver, including `--control-unset` and
`--prefill-gap`. These examples retain results in separate local directories:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/overlap-32k-guard nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/overlap-32k timing --flag DS4_METAL_V41_BATCH_SHARED_OVERLAP --control-unset --frontier 32768
python3 speed-bench/ideas/pressure_run.py .workspaces/overlap-8k-guard nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/overlap-8k timing --flag DS4_METAL_V41_BATCH_SHARED_OVERLAP --control-unset --frontier 8192
python3 speed-bench/ideas/pressure_run.py .workspaces/overlap-native-guard nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/overlap-native quality --flag DS4_METAL_V41_BATCH_SHARED_OVERLAP --control-unset --prefill-gap 0
python3 speed-bench/ideas/pressure_run.py .workspaces/overlap-16g-guard nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/overlap-16g quality --flag DS4_METAL_V41_BATCH_SHARED_OVERLAP --control-unset --cache 16GB
```

The quality stage defaults to the 8K frontier; omitting `--prefill-gap 0` and
`--cache 16GB` reproduces the ordinary 64 GB quality case. The runner records
settings and arguments; its quality outputs require the existing paired analysis
workflow to compute full-vocabulary delta-NLL and KL.
