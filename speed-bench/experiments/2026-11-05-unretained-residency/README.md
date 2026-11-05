# Unretained command buffers with deduplicated residency

This experiment asks whether unretained Metal command buffers improve the
selected package when expert residency association is deduplicated. The
November 5, 2026 slice ran from 10:16 to 11:16 Eastern. Retain the scratch
lifetime repair as a correctness fix. The combined unretained and deduplicated
residency settings remain unselected: 32K timing worsened, 8K was mixed, and
the direct comparison with selected package 32 was not run.

The preceding [residency experiment](../2026-11-05-residency-once/README.md)
reduced repeated associations but found mixed timing with retained command
buffers. Its prototype remains unselected. Unretained command buffers also
avoid normal Metal resource retains, so callers must keep resources alive until
GPU work completes. Deduplication removes repeated completion-handler captures
in this mode as well as repeated association calls. A speedup is a hypothesis,
not an established result.

## Comparison and procedure

The initial comparison used runtime `47e461b0`, closure `7fea79d0`, and the same benchmark
binary in both arms, SHA-256
`d5995372a55dcb95a612fef4d0ed816b68606a7918e5f770b11bbea673de3c47`.
The experiment 32 package and
`DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY_ONCE=1` are fixed. The control leaves
`DS4_METAL_UNRETAINED_COMMAND_BUFFERS` absent; the candidate sets it to `1`.
The one changed setting isolates the unretained mode within the unselected
residency-deduplication package. Local settings and raw artifacts are under
`.workspaces/hillclimb-20261104/34-unretained-residency/`.

Before model runs, the source audit covers scheduling and event handoffs as
well as graph, static/shared view, transient and expert-set lifetimes across
flushes. A selected event completing does not necessarily mean all shared work
has finished. The scheduling source review found no blocker in the selected
package: successful flushes and event handoffs keep transient buffers and exact
model views until full completion, including failure-path drains. Early
expert-read retirement does not release pending command buffers or those views.
The initial resource-owner review found no demonstrated blocker in the selected
graph, static, transient and expert paths, but a follow-up found the active
scratch lifetime defect described below. Graph and model-view owners persist,
transient arrays retain temporary buffers, and expert entries and slab arrays
own their buffers. The residency set has an explicit completion capture, but
that capture does not replace ownership of the buffers themselves.

A follow-up source review found an active scratch lifetime defect. The generic
scratch-growth helper replaces its global buffer without adding the old
allocation to transients. V4.1 raw attention encodes 2,048-row pieces in the
same layer command buffer. The second piece grows the key count from 2,048 to
2,175, growing KV scratch from 2,097,152 to 2,227,200 bytes and its block map
from 8,192 to 8,704 bytes. The previous helper's local owners have gone away,
so replacement can release those earlier buffers before GPU execution. This
source-supported defect prompted a
narrow repair in `d31275e9` that preserves the old scratch allocations through the
completion boundary. Independent source review passed; `ds4-bench` and
`ssd-experts-test` built successfully. The repaired benchmark SHA-256 is
`745a802a828d61ef5682e8ae1a1c7aca55013424749420d14d5f2ecb954e41a9`;
the fixture SHA-256 is
`c08f7dfe3e034cf3ba480aa913002d367ebec7b9ed9ac4a392b7a7a4ec4668c6`.
The repaired quality results are reported below. Detailed source reviews are retained
in the local experiment workspace. The control must omit the
unretained variable: its presence enables the mode even if its value is `0`,
and the policy is cached per process; each arm runs in a fresh process.

The checks use prefill, 128 teacher tokens, snapshot and restore, a 512-token
append, and another 128 teacher tokens. The expert cache survives restore and
the OS file cache is not flushed. Phase timers exclude startup and snapshot
operations; process wall includes them. Steady decode excludes only the first
token, so the complete decode windows and first-step latencies matter too. The
fixed histories use 8K and 32K frontiers with a requested 64 GB cache and an
8K frontier with a requested 16 GB cache. The plan calls for five alternating profile-off timing pairs per configuration. A separate 32K timeline examines changes
to work and gaps; its profiling is diagnostic, not a latency measurement.
Completion, errors, cache retirement and resource use also require inspection.
If the result is promising, five alternating pairs must compare the combined
settings directly with selected package 32 before any selection. A gain over
the unselected control alone is insufficient.

## Findings and decision

The initial 8K fixed-history comparison found differences in all 256
full-vocabulary rows: 16 top-1 mismatches, mean delta-NLL of +0.0236695 nats
per token, mean KL of 0.0242765 and maximum absolute logit difference of
10.73814. The first row already differed at the end of prefill, before the
first teacher decode step. The guard returned child code 0 and the logs showed
no command errors or event timeouts. Route and cache counts also differed.
The local `quality-resource-check.md` records per-frontier results and counters.

After the scratch repair, all three comparisons—8K and 32K with the requested
64 GB cache and 8K with a requested 16 GB cache—produced 768 of 768 byte-exact
full-vocabulary rows and matching post-evaluation hashes, with zero delta-NLL
and KL. The first prefill row also matched. The repair removed the observed
8K drift. A separate comparison reused the original and repaired retained-arm
8K dumps and found all 256 rows exact across revisions. These fixed-history
checks do not establish practical task quality. The smaller-cache case exercised five split layers and 3,993 evictions and
reuses with an effective 8.88 GiB expert cache. All eight experiment guards returned child code 0, observed only pressure
level 1 and no adverse samples; swap ranged from 28.997676 to 29.005488 GiB. The logs showed no command errors or event timeouts. The local
`fixed-resource-check.md` records resource observations.

The original intrusive 32K timeline had the same 2,587 command buffers,
417,062 encoder rows, 863 residency associations and normalized kernel-label
multisets in both arms, but all 34 decode hashes differed. The matching recorded
structure does not establish matching resource contents or lifetimes; this
trace is diagnostic, not a performance comparison. The repaired 32K timeline
matched all 34 decode hashes and recorded the same 2,587 command buffers,
417,062 encoder rows, normalized kernel groups, 850 associated command
buffers and 863 associations in each arm. All 40 seed command buffers were
covered, cache counts matched, and refresh and ordinary tokens retained 91 and
11 command buffers respectively. This intrusive trace adds completion callbacks
of its own and is not a performance estimate.

For ordinary tokens, the following are means of 15 tokens, in milliseconds,
retained → unretained. The GPU window spans embedding through output head.
Host encode runs from command begin through final commit; final wait runs from
that commit until all pending and current commands complete. Together these
host scopes cover the GPU transaction but exclude host preparation before
begin, so they are not full session evaluation time.

| Frontier | GPU window | Host encode | Host final wait |
| --- | ---: | ---: | ---: |
| Fresh ordinary | 27.380 → 27.490 | 5.140 → 5.887 | 22.833 → 22.293 |
| Warm ordinary | 27.204 → 27.241 | 5.760 → 6.213 | 22.080 → 21.733 |

The ordinary GPU windows were nearly unchanged while host encoding increased
and the final wait decreased. This is a redistribution of recorded scopes, not
an isolated measurement of completion-callback cost. The profile found no new
recurring large inter-command-buffer or event gap. A possible bounded follow-up
would retain cleared sets through the existing full-drain boundary instead of
capturing one per command buffer; that ownership change is not implemented and
its effect on performance is unproven.

## Repaired 32K timing

Five profile-off pairs alternate order; the retained control ran first in pairs
1, 3 and 5. Values are retained → unretained. Phase and inference-total seconds are derived from throughput rounded to two
decimal places; process wall is measured directly. Fresh complete
decode was slower in all five pairs; inference total and wall were slower in
four. Warm complete decode was slower in three pairs and warm append in all
five. Prefill and totals drifted upward during the sequence, so pair order and
variation matter. The repaired timeline above locates changed host scopes without isolating
the cause of these timings.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 43.656 → 43.880 | 44.958 → 44.815 | 45.521 → 46.131 | 47.051 → 46.959 | 47.105 → 47.279 | 45.658 ± 1.462 → 45.813 ± 1.440 |
| Fresh first step, ms | 184.018 → 196.098 | 178.602 → 205.033 | 180.423 → 217.356 | 189.471 → 168.804 | 167.644 → 159.399 | 180.032 ± 8.072 → 189.338 ± 24.470 |
| Fresh complete decode, s | 3.718 → 3.799 | 3.674 → 3.926 | 3.681 → 3.862 | 3.688 → 3.708 | 3.671 → 3.696 | 3.686 ± 0.019 → 3.798 ± 0.099 |
| Fresh steady decode, tokens/s | 35.940 → 35.240 | 36.330 → 34.130 | 36.280 → 34.840 | 36.300 → 35.880 | 36.250 → 35.910 | 36.220 ± 0.159 → 35.200 ± 0.749 |
| Warm append, s | 7.170 → 7.358 | 7.099 → 7.725 | 7.473 → 8.077 | 7.760 → 7.804 | 7.768 → 8.174 | 7.454 ± 0.316 → 7.828 ± 0.321 |
| Warm first step, ms | 33.319 → 32.673 | 33.187 → 33.024 | 33.474 → 33.214 | 33.614 → 32.921 | 33.585 → 37.156 | 33.436 ± 0.181 → 33.798 ± 1.888 |
| Warm complete decode, s | 3.526 → 3.673 | 3.753 → 3.786 | 3.854 → 3.895 | 3.926 → 3.919 | 3.946 → 3.924 | 3.801 ± 0.171 → 3.839 ± 0.109 |
| Warm steady decode, tokens/s | 36.360 → 34.890 | 34.140 → 33.840 | 33.240 → 32.880 | 32.630 → 32.680 | 32.470 → 32.680 | 33.768 ± 1.590 → 33.394 ± 0.964 |
| Inference total, s | 58.069 → 58.711 | 59.484 → 60.252 | 60.530 → 61.966 | 62.425 → 62.390 | 62.490 → 63.073 | 60.600 ± 1.908 → 61.278 ± 1.773 |
| Process wall, s | 61.069 → 61.391 | 62.034 → 62.820 | 63.132 → 64.540 | 64.990 → 64.914 | 65.045 → 65.625 | 63.254 ± 1.768 → 63.858 ± 1.722 |

## Repaired 8K timing

Five alternating profile-off pairs use the same order and units as the 32K
comparison. Inference totals are mixed, and a low candidate fresh steady-decode
sample affects its mean. The complete samples show the variation.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 19.048 → 18.022 | 18.337 → 18.316 | 18.679 → 18.987 | 19.405 → 19.239 | 19.333 → 19.429 | 18.960 ± 0.450 → 18.799 ± 0.605 |
| Fresh first step, ms | 194.172 → 169.031 | 184.553 → 176.504 | 210.340 → 185.175 | 178.058 → 179.174 | 165.943 → 207.140 | 186.613 ± 16.769 → 183.405 ± 14.479 |
| Fresh complete decode, s | 3.632 → 3.783 | 3.600 → 3.586 | 3.622 → 3.611 | 3.608 → 3.597 | 3.592 → 3.644 | 3.611 ± 0.016 → 3.644 ± 0.080 |
| Fresh steady decode, tokens/s | 36.940 → 35.150 | 37.190 → 37.240 | 37.220 → 37.070 | 37.030 → 37.170 | 37.070 → 36.960 | 37.090 ± 0.116 → 36.718 ± 0.883 |
| Warm append, s | 6.765 → 7.235 | 6.891 → 6.724 | 7.234 → 7.478 | 7.722 → 7.634 | 7.743 → 7.795 | 7.271 ± 0.455 → 7.373 ± 0.418 |
| Warm first step, ms | 35.364 → 32.739 | 32.041 → 30.931 | 32.268 → 33.399 | 32.999 → 32.670 | 34.289 → 32.750 | 33.392 ± 1.409 → 32.498 ± 0.924 |
| Warm complete decode, s | 3.434 → 3.648 | 3.657 → 3.459 | 3.748 → 3.797 | 3.832 → 3.788 | 3.839 → 3.813 | 3.702 ± 0.167 → 3.701 ± 0.150 |
| Warm steady decode, tokens/s | 37.360 → 35.130 | 35.040 → 37.050 | 34.170 → 33.740 | 33.430 → 33.820 | 33.370 → 33.600 | 34.674 ± 1.647 → 34.668 ± 1.467 |
| Inference total, s | 32.880 → 32.687 | 32.485 → 32.085 | 33.283 → 33.873 | 34.567 → 34.258 | 34.508 → 34.681 | 33.544 ± 0.950 → 33.517 ± 1.092 |
| Process wall, s | 35.361 → 35.118 | 35.010 → 34.571 | 35.753 → 36.404 | 37.091 → 36.762 | 36.995 → 37.130 | 36.042 ± 0.952 → 35.997 ± 1.100 |

The smaller-cache timing comparison and direct selected-package comparison
were not run within the slice. No new practical-task check was run. Direct
comparison against selected package 32 remains necessary before selection.
The scratch lifetime repair is retained as a correctness fix; the optional
unretained package remains unselected.

## Reproduction

Use a build of repair `d31275e9` with the recorded benchmark hash. The driver
clears inherited DS4 settings and starts each arm in a separate process. The
control must omit the presence-based flag. The following commands reproduce the
three planned fixed-history and profile-off comparisons with fresh output
directories, one guarded model job at a time. Quality uses diagnostic settings;
timing uses the same package without profiling. Pause unrelated work for timing.
The smaller-cache timing and direct comparison with selected package 32 were not run.
The repaired timeline uses a separate diagnostic run:

```nu
let out = '.workspaces/hillclimb-20261104/34-unretained-residency-repro'
mkdir $out
let package = (open speed-bench/experiments/2026-11-05-cpu-expert-boundary/candidate-settings.json)
let fixed = ($package | merge {DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY_ONCE: '1'})
$fixed | to json | save ($out | path join fixed.json)
$fixed | merge {DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: '1', DS4_METAL_V41_RESIDENT_ROUTE_PROFILE: '1', DS4_METAL_SELECTED_READBACK_PROFILE: '1'} | to json | save ($out | path join diagnostic.json)
for case in [{name: '8k', frontier: 8192, cache: '64GB'} {name: '32k', frontier: 32768, cache: '64GB'} {name: '16gb', frontier: 8192, cache: '16GB'}] {
  let stages = if $case.name == '16gb' { ['quality'] } else { ['quality' 'timing'] }
  for stage in $stages {
    let settings = if $stage == 'quality' { 'diagnostic.json' } else { 'fixed.json' }
    python3 speed-bench/ideas/pressure_run.py ($out | path join $'guard-($stage)-($case.name)') -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu ($out | path join $'($stage)-($case.name)') $stage --flag DS4_METAL_UNRETAINED_COMMAND_BUFFERS --control-unset --candidate 1 --frontier $case.frontier --cache $case.cache --tokens 128 --samples 5 --extra-settings ($out | path join $settings)
  }
}
let timeline = ($fixed | merge {DS4_BENCH_DECODE_LOGITS_HASH: '1'})
$timeline | to json | save ($out | path join timeline.json)
python3 speed-bench/ideas/pressure_run.py ($out | path join guard-timeline32k) -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu ($out | path join timeline32k) timeline --flag DS4_METAL_UNRETAINED_COMMAND_BUFFERS --control-unset --candidate 1 --frontier 32768 --cache 64GB --tokens 17 --extra-settings ($out | path join timeline.json)
```
