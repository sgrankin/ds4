# Deferred residency-set release

This experiment tests whether moving residency-set ownership from per-command
completion captures to the existing full completion drain reduces host work.
The November 5, 2026 slice is 11:16–12:16 Eastern. Keep selected package 32 unchanged: the combined settings did not demonstrate
a consistent net benefit across the measured workloads. Retain the opt-in
prototype and the earlier scratch correctness repair; no further completion-handler
follow-up is selected.

## Context and hypothesis

The preceding [unretained residency experiment](../2026-11-05-unretained-residency/README.md)
repaired a scratch lifetime defect and found byte-exact fixed-history results,
but the combined unretained and residency-deduplication settings remained
unselected: 32K timing worsened and 8K was mixed. Its trace showed nearly
unchanged ordinary GPU windows and more host encoding work. The source review
found an additional completion capture for each residency association in the
unretained arm, but did not measure its individual cost. Removing those captures
may reduce host work; the effect on latency is a hypothesis.

The active expert cache already owns its residency set. The implemented candidate retains
that ownership while the set is active, then transfers a cleared set to a
separate per-context retired-set list until an existing full completion drain.
This keeps the set object alive; it does not replace buffer ownership, pins,
resource declarations or allocation membership. Successful event handoffs do
not themselves establish full completion. The source and runtime checks below
must establish the relevant paths before any safety or performance conclusion.

## Comparison and procedure

The parent source baseline is runtime `d31275e9`, closure `a8e51c8a`. Both
primary comparison arms use candidate runtime `1e0e5841` and the same benchmark
binary, with [selected package 32](../2026-11-05-cpu-expert-boundary/candidate-settings.json)
plus `DS4_METAL_UNRETAINED_COMMAND_BUFFERS=1` and
`DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY_ONCE=1` fixed. The one changed setting is
`DS4_METAL_STREAMING_EXPERT_DEFER_SET_RELEASE=0` versus `1`, in fresh processes.
The clean build produced `ds4-bench`
SHA-256 `aa74f9055beeef4d8bd5933e95c4d6e3ec5f285e9ca3d86b5a480d273abd0c40`
and `test-metal-ssd-experts` SHA-256
`bae85ea8a14a33ce633bcfdc518aece52d36c054348a94cb20514659cbf8dbd4`.
The machine is an M5 Max with 128 GiB memory. The model is
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, with context 40,960 and
prefill chunks of 8,192. The requested 64 GB cache uses 55.98 GiB of slabs;
the requested 16 GB miss case has an effective 8.88 GiB expert cache. Raw artifacts remain local under
`.workspaces/hillclimb-20261104/35-deferred-set-release/`.

The checks cover bounded fixtures and fixed-history quality at 8K and
32K with a requested 64 GB cache, plus an 8K requested 16 GB miss case. Five
profile-off timing pairs at each of 8K and 32K alternate order. A separate
trace investigates changed work and gaps; it is diagnostic rather than a
latency measurement. A direct five-pair comparison against selected package 32
is required before selecting the combined settings. A smaller-cache timing
comparison is conditional on the slice budget.

## Source and correctness evidence

Independent source review found no blocker in the scoped change. The deferred
policy is fixed when each set is published; only exact value `1` enables it,
and only when the retirement array exists. With the flag absent or `0`, the
existing unretained-command completion capture remains. Successful flushes and
event handoffs preserve retired sets; full drains wait for pending and current
commands before releasing them. The review also covered standalone commands,
seed blits, error drains, pressure clear, synchronization and teardown. These
are source conclusions, not a proof of every runtime path. The separate list
does not alter the typed transient-buffer collection or allocation ownership.

The guarded SSD fixture ran both flag values with retained, unretained and
queue modes; all six exited 0. An early-clear check clears residency after seed
encoding but before submission, verifies that slab and entry counts remain
unchanged, then checks that the deferred owner holds the set until the full
drain and is empty afterward. The candidate unretained branch passed with
activation confirmed. The check uses a weak reference and does not require
immediate object destruction after the drain. In two additional runs, forcing
`ulimit -l 0` produced the expected fixture exit 1; the enclosing guard exited
0 after verifying those expected failures. These fixtures do not exercise every
reviewed event or error path.

## Quality and resource observations

All three fixed-history comparisons matched: 256 of 256 full-vocabulary rows
were byte-exact in each case, with matching post-evaluation hashes and zero
top-1 differences, delta-NLL, KL and maximum logit difference. The 8K and
32K cases use a requested 64 GB cache; the requested 16 GB case repeats the
8K history under cache pressure. Each consists of 128 teacher tokens after
prefill and 128 after a 512-token append. Hashes overlap the dumped distributions
and are not independent additional quality samples. These checks compare the
deferred owner with the existing capture within the unretained and deduplicated
package; they do not establish practical task quality or compare that package
with selected package 32 by itself. Within each pair, cache, route, slab and
residency-association counts matched. The smaller-cache case exercised five
split layers and 3,993 evictions and buffer reuses, with an effective 8.88 GiB
expert cache. No intermediate set clear occurred in these model runs. All three
guards completed with child code 0, pressure level 1, unchanged 28.998 GiB swap
and no adverse samples; logs showed no unexpected errors or timeouts.

A separate reuse-only comparison checks the 32K candidate against the actual
selected package 32 retained arm at runtime `b7e7af9d`: all 256 rows and
post-evaluation hashes were exact, with zero delta-NLL and KL. The selected arm
comes from experiment 32's `combined-quality32k/1-candidate`, resolved through
the experiment 33 prior symlink; provenance and analyzer inputs are recorded in
the local `selected-cross-quality32k` directory. This repeats the same 32K
history, so it adds neither an independent task nor a performance comparison.

## Profile-off performance

### 32K timing

Five profile-off pairs alternate order, with the control first in pairs 1, 3
and 5. Values are control → candidate. Each run prefills the 32K frontier,
decodes 128 tokens, restores a KV snapshot, appends 512 tokens and decodes
another 128. The expert cache survives restore; the OS file cache is not
flushed. Phase timers exclude startup and snapshot operations; process wall
includes them. Seconds for phases and inference totals are derived from
throughput rounded to two decimals. Steady decode excludes only the first token,
so it does not establish that initialization is complete.

Inference total was neutral at 61.410 ± 2.392 → 61.340 ± 2.249 s. Prefill and
totals increased over the sequence, and paired changes follow run order in
several cases. Candidate fresh steady decode has conspicuously slow samples in
pairs 1 and 3, while the control is stable. The cause is not established.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 43.074 → 43.150 | 46.092 → 44.806 | 47.020 → 47.143 | 47.543 → 47.324 | 47.802 → 47.794 | 46.306 ± 1.922 → 46.044 ± 1.988 |
| Fresh first step, ms | 184.252 → 213.530 | 198.118 → 163.890 | 189.450 → 197.131 | 186.454 → 168.475 | 189.564 → 214.443 | 189.568 ± 5.270 → 191.494 ± 24.165 |
| Fresh complete decode, s | 3.682 → 3.893 | 3.703 → 3.676 | 3.703 → 3.943 | 3.696 → 3.672 | 3.698 → 3.732 | 3.696 ± 0.008 → 3.783 ± 0.127 |
| Fresh steady decode, tokens/s | 36.310 → 34.520 | 36.240 → 36.160 | 36.150 → 33.900 | 36.190 → 36.250 | 36.190 → 36.110 | 36.216 ± 0.061 → 35.388 ± 1.099 |
| Warm append, s | 7.087 → 7.534 | 7.572 → 7.245 | 7.543 → 7.704 | 7.832 → 7.889 | 7.766 → 7.762 | 7.560 ± 0.292 → 7.627 ± 0.249 |
| Warm first step, ms | 32.320 → 33.454 | 35.524 → 32.894 | 33.487 → 33.684 | 37.065 → 33.486 | 33.603 → 33.786 | 34.400 ± 1.882 → 33.461 ± 0.346 |
| Warm complete decode, s | 3.523 → 3.726 | 3.891 → 3.839 | 3.898 → 3.940 | 3.952 → 3.963 | 3.974 → 3.967 | 3.847 ± 0.185 → 3.887 ± 0.104 |
| Warm steady decode, tokens/s | 36.380 → 34.390 | 32.940 → 33.360 | 32.870 → 32.510 | 32.450 → 32.320 | 32.230 → 32.300 | 33.374 ± 1.706 → 32.976 ± 0.902 |
| Inference total, s | 57.366 → 58.304 | 61.257 → 59.566 | 62.163 → 62.730 | 63.023 → 62.848 | 63.240 → 63.255 | 61.410 ± 2.392 → 61.340 ± 2.249 |
| Process wall, s | 60.338 → 60.953 | 63.892 → 62.193 | 64.725 → 65.334 | 65.638 → 65.467 | 65.936 → 65.886 | 64.106 ± 2.254 → 63.967 ± 2.238 |

### 8K timing

Five profile-off pairs use the same order, procedure and units as the 32K
comparison, at an 8K frontier. The candidate's final run is slow across
prefill, append and decode. Its fresh steady decode includes a 34.66 tokens/s
sample; retain this variation rather than treating it as representative of
every run. Inference total means are 33.850 → 34.216 s. All three control-first
pairs favor the control on total time; the two candidate-first pairs split.
This association does not establish a cause.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 18.239 → 18.423 | 18.350 → 18.683 | 19.200 → 19.528 | 19.406 → 19.562 | 19.541 → 19.745 | 18.947 ± 0.609 → 19.188 ± 0.593 |
| Fresh first step, ms | 190.616 → 221.465 | 214.916 → 163.063 | 207.862 → 188.807 | 185.378 → 185.407 | 169.504 → 220.062 | 193.655 ± 18.131 → 195.761 ± 24.879 |
| Fresh complete decode, s | 3.614 → 3.639 | 3.637 → 3.586 | 3.648 → 3.632 | 3.610 → 3.608 | 3.594 → 3.885 | 3.621 ± 0.022 → 3.670 ± 0.122 |
| Fresh steady decode, tokens/s | 37.100 → 37.150 | 37.110 → 37.100 | 36.920 → 36.880 | 37.090 → 37.110 | 37.080 → 34.660 | 37.060 ± 0.079 → 36.580 ± 1.078 |
| Warm append, s | 6.702 → 6.917 | 7.349 → 6.777 | 7.927 → 7.938 | 7.763 → 7.843 | 7.718 → 8.522 | 7.492 ± 0.490 → 7.599 ± 0.736 |
| Warm first step, ms | 34.698 → 31.377 | 32.074 → 34.290 | 34.240 → 33.180 | 32.868 → 32.857 | 32.972 → 38.201 | 33.370 ± 1.074 → 33.981 ± 2.578 |
| Warm complete decode, s | 3.460 → 3.464 | 3.838 → 3.440 | 3.912 → 3.908 | 3.873 → 3.894 | 3.868 → 4.086 | 3.790 ± 0.186 → 3.758 ± 0.290 |
| Warm steady decode, tokens/s | 37.080 → 36.990 | 33.370 → 37.290 | 32.750 → 32.780 | 33.070 → 32.900 | 33.110 → 31.380 | 33.876 ± 1.805 → 34.268 ± 2.691 |
| Inference total, s | 32.015 → 32.444 | 33.175 → 32.486 | 34.687 → 35.007 | 34.652 → 34.907 | 34.722 → 36.237 | 33.850 ± 1.217 → 34.216 ± 1.682 |
| Process wall, s | 34.661 → 35.087 | 35.651 → 35.157 | 37.229 → 37.427 | 37.158 → 37.397 | 37.289 → 38.766 | 36.397 ± 1.187 → 36.767 ± 1.600 |

These pairs compare the candidate only with the unselected unretained and
deduplicated control; they do not measure net performance against selected
package 32.

### Direct 32K comparison with selected package 32

Five profile-off pairs compare frozen `b7e7af9d` package 32 with `1e0e5841`
and the three added flags. The manifest records benchmark SHA-256
`fd4596242336958607ae44f302e77bdc39e4a0ae1a28830df4d93da38ce88921` →
`aa74f9055beeef4d8bd5933e95c4d6e3ec5f285e9ca3d86b5a480d273abd0c40`.
The control runs first in pairs 1, 3 and 5; scopes and cache conditions are as
above. Inference total is 60.434 ± 2.855 → 60.619 ± 3.236 s, with a slow
candidate final prefill of 49.130 s retained. These samples do not demonstrate
a net benefit at 32K.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 41.843 → 42.136 | 44.673 → 42.836 | 45.869 → 46.309 | 46.958 → 46.271 | 47.709 → 49.130 | 45.411 ± 2.300 → 45.336 ± 2.859 |
| Fresh first step, ms | 197.050 → 187.005 | 192.667 → 195.269 | 168.211 → 180.376 | 182.829 → 217.393 | 187.721 → 200.047 | 185.696 ± 11.131 → 196.018 ± 14.139 |
| Fresh complete decode, s | 3.693 → 3.737 | 3.696 → 3.893 | 3.653 → 3.671 | 3.693 → 3.725 | 3.683 → 3.758 | 3.684 ± 0.018 → 3.757 ± 0.083 |
| Fresh steady decode, tokens/s | 36.330 → 35.770 | 36.250 → 34.350 | 36.450 → 36.390 | 36.180 → 36.200 | 36.340 → 35.700 | 36.310 ± 0.102 → 35.682 ± 0.799 |
| Warm append, s | 7.190 → 7.543 | 7.095 → 7.593 | 7.490 → 7.642 | 7.885 → 7.627 | 8.034 → 8.340 | 7.539 ± 0.414 → 7.749 ± 0.333 |
| Warm first step, ms | 32.203 → 32.168 | 32.214 → 32.943 | 33.518 → 32.677 | 33.507 → 33.292 | 33.386 → 33.719 | 32.966 ± 0.693 → 32.960 ± 0.590 |
| Warm complete decode, s | 3.514 → 3.520 | 3.683 → 3.757 | 3.854 → 3.797 | 3.952 → 3.901 | 4.001 → 3.910 | 3.801 ± 0.201 → 3.777 ± 0.158 |
| Warm steady decode, tokens/s | 36.480 → 36.410 | 34.780 → 34.110 | 33.240 → 33.740 | 32.410 → 32.830 | 32.000 → 32.770 | 33.782 ± 1.846 → 33.972 ± 1.480 |
| Inference total, s | 56.240 → 56.936 | 59.148 → 58.079 | 60.866 → 61.418 | 62.489 → 61.525 | 63.428 → 65.137 | 60.434 ± 2.855 → 60.619 ± 3.236 |
| Process wall, s | 58.811 → 59.537 | 61.870 → 60.699 | 63.469 → 64.058 | 65.072 → 64.135 | 65.985 → 67.804 | 63.041 ± 2.839 → 63.247 ± 3.258 |

### Direct 8K comparison with selected package 32

Five profile-off pairs use the same builds, settings and alternating order as
the direct 32K comparison. Inference total averaged 34.709 → 34.476 s, a small
mixed improvement; fresh steady decode averaged 36.906 → 36.752 tokens/s and
warm steady decode 34.124 → 34.198 tokens/s. This does not establish a
consistent decode gain, and the 32K comparison did not show a net benefit.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 19.856 → 18.922 | 19.415 → 19.226 | 20.029 → 20.001 | 19.760 → 19.724 | 19.864 → 19.603 | 19.785 ± 0.228 → 19.495 ± 0.425 |
| Fresh first step, ms | 194.615 → 173.204 | 203.816 → 176.126 | 175.977 → 185.845 | 193.057 → 193.183 | 168.398 → 187.987 | 187.173 ± 14.533 → 183.269 ± 8.359 |
| Fresh complete decode, s | 3.623 → 3.582 | 3.628 → 3.708 | 3.626 → 3.626 | 3.642 → 3.639 | 3.624 → 3.643 | 3.629 ± 0.008 → 3.640 ± 0.045 |
| Fresh steady decode, tokens/s | 37.050 → 37.250 | 37.090 → 35.960 | 36.820 → 36.920 | 36.820 → 36.860 | 36.750 → 36.770 | 36.906 ± 0.153 → 36.752 ± 0.478 |
| Warm append, s | 6.903 → 6.933 | 7.348 → 7.449 | 7.647 → 7.735 | 7.791 → 7.785 | 7.973 → 8.011 | 7.532 ± 0.419 → 7.583 ± 0.415 |
| Warm first step, ms | 32.711 → 53.951 | 64.799 → 48.346 | 33.116 → 33.608 | 32.978 → 33.723 | 33.475 → 33.295 | 39.416 ± 14.192 → 40.585 ± 9.846 |
| Warm complete decode, s | 3.642 → 3.652 | 3.711 → 3.649 | 3.744 → 3.772 | 3.812 → 3.778 | 3.908 → 3.937 | 3.763 ± 0.102 → 3.758 ± 0.118 |
| Warm steady decode, tokens/s | 35.190 → 35.300 | 34.820 → 35.280 | 34.230 → 33.960 | 33.610 → 33.920 | 32.770 → 32.530 | 34.124 ± 0.966 → 34.198 ± 1.151 |
| Inference total, s | 34.023 → 33.090 | 34.102 → 34.032 | 35.046 → 35.135 | 35.004 → 34.926 | 35.369 → 35.194 | 34.709 ± 0.607 → 34.476 ± 0.905 |
| Process wall, s | 36.556 → 35.673 | 36.824 → 36.520 | 37.695 → 37.699 | 37.494 → 37.522 | 37.894 → 37.656 | 37.292 ± 0.576 → 37.014 ± 0.892 |

## Diagnostic trace

One intrusive 32K diagnostic pair matched all 34 decode hashes, cache
accounting, submission structure and normalized kernel groups. Both arms had
2,587 command buffers, 417,062 encoder records and 863 actual associations
across 850 command buffers, including all 40 seed buffers. Based on the
reviewed source and activation logs, those associations register 863 set
completion captures in the control and none in the candidate. This is a
source-derived registration count, not a callback execution measurement. Both
arms still install 2,587 profiling completion callbacks; this trace cannot estimate
native callback overhead. No intermediate set clear occurred in the trace, so
the early-clear fixture supplies separate evidence for that ownership path.

For 15 ordinary tokens, mean embedding-to-head GPU windows were 28.424 →
27.414 ms fresh and 27.513 → 27.294 ms warm. Fresh step 1 is an outlier in
each arm at 40.249 → 31.488 ms; corresponding ordinary medians are 27.584 →
27.125 ms. No new recurring large inter-command-buffer stall appeared. GPU
windows exclude CPU callback execution, but profiling callbacks can change
submission timing. These measurements do not isolate saved registration time
or explain the intermittent native timing samples.
A separate native 128-token diagnostic pair recorded full session-evaluation
intervals without timeline callbacks. It showed no consistent benefit: fresh
total decode was 3,670.097 → 3,702.128 ms, with 27.823 ms of the 32.031 ms
difference in the first token. Warm total was 3,695.946 → 3,733.172 ms;
ordinary tokens accounted for 35.237 ms of the 37.226 ms difference, including
26.086 ms across steps 65–79. The pair does not reproduce the large fresh
steady penalty or establish a cause.

## Decision and reproduction

The combined ONCE, unretained and deferred settings remain unselected. Retain
the opt-in prototype as recorded evidence and the experiment 34 scratch lifetime
repair independently. No smaller-cache timing or fresh practical-task check was
run. All ten experiment guards completed with child code 0, pressure level 1
and no adverse samples; swap ranged from 28.997676 to 29.003418 GiB. The direct combined comparison uses the frozen
package 32 runtime `b7e7af9d` as baseline and `1e0e5841` with all three flags
as candidate; both binary hashes were verified before its 32K run. The fixed comparison settings can be reconstructed
from the committed package, rather than depending on the ignored raw JSON.
The following commands run the planned quality and timing comparisons one
model job at a time under the pressure guard. The driver starts each arm in a
fresh process and varies the deferred flag between `0` and `1`. Pause unrelated
work during timing. Diagnostic settings apply only to quality; they are not
used for profile-off timing. Record actual completed commands and any changes
when the measurements finish.

```nu
let out = '.workspaces/hillclimb-20261104/35-deferred-set-release-repro'
mkdir $out
let package = (open speed-bench/experiments/2026-11-05-cpu-expert-boundary/candidate-settings.json)
let fixed = ($package | merge {DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY_ONCE: '1', DS4_METAL_UNRETAINED_COMMAND_BUFFERS: '1'})
$fixed | to json | save ($out | path join fixed.json)
$fixed | merge {DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: '1', DS4_METAL_V41_RESIDENT_ROUTE_PROFILE: '1', DS4_METAL_SELECTED_READBACK_PROFILE: '1'} | to json | save ($out | path join diagnostic.json)
for case in [{name: '8k', frontier: 8192, cache: '64GB'} {name: '32k', frontier: 32768, cache: '64GB'} {name: '16gb', frontier: 8192, cache: '16GB'}] {
  let stages = if $case.name == '16gb' { ['quality'] } else { ['quality' 'timing'] }
  for stage in $stages {
    let settings = if $stage == 'quality' { 'diagnostic.json' } else { 'fixed.json' }
    python3 speed-bench/ideas/pressure_run.py ($out | path join $'guard-($stage)-($case.name)') -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu ($out | path join $'($stage)-($case.name)') $stage --flag DS4_METAL_STREAMING_EXPERT_DEFER_SET_RELEASE --control 0 --candidate 1 --frontier $case.frontier --cache $case.cache --tokens 128 --samples 5 --extra-settings ($out | path join $settings)
  }
}
```

The direct net comparison uses the same guarded five-pair driver, with frozen
baseline and candidate roots. This command is the 32K run; change the frontier
to `8192` and the output names for 8K. The candidate settings add the three
flags explicitly to package 32:

```nu
$fixed | merge {DS4_METAL_STREAMING_EXPERT_DEFER_SET_RELEASE: '1'} | to json | save ($out | path join combined-candidate.json)
python3 speed-bench/ideas/pressure_run.py ($out | path join guard-combined32k) -- nu speed-bench/experiments/2026-11-04-decode-pass/combined.nu ($out | path join combined32k) timing --baseline-root .workspaces/hillclimb-20261104/33-residency-once/before-root --candidate-root . --baseline-revision b7e7af9d --candidate-revision 1e0e5841 --baseline-settings speed-bench/experiments/2026-11-05-cpu-expert-boundary/candidate-settings.json --final-settings ($out | path join combined-candidate.json) --frontier 32768 --tokens 128 --samples 5
```
