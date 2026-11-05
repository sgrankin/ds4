# Command-buffer expert-slab residency

**Retain command-buffer residency as opt-in groundwork; keep experimental
package 25 on queue residency.** Ordinary and fast presets leave slab residency
off. The compute-only policy adds a substantial fresh
first-step delay at both tested frontiers, with little change in steady decode.
A trace locates the extra delay before the first associated expert command
buffer, but does not establish its driver-level cause. The allotted November 5 slice
was 06:00 to 07:00 Eastern.

## Question and setup

The [selected package](../2026-11-05-slab-event-interaction/README.md) uses
queue-attached slab residency with shared-event overlap disabled. Earlier
profiles found that residency removes fresh-start gaps but interacts with
shared-event handoffs; disabling overlap avoids those gaps while adding a
recurring CPU-refresh cost. This experiment tests whether narrower association
changes that cost while preserving the selected overlap setting.

The frozen pre-experiment reference is runtime `c1982bc8`, closure `5840e835`.
The machine is an M5 Max with 128 GiB memory, using
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`. Both measured arms use
the same `cd42c5ac` binary; mode 1 preserves the
original queue behavior. Both use package 25 settings: shared-event overlap 0, refresh
every 16 positions and submission cadence 4. Slab residency mode 1 attaches
the set to the queue; mode 2 associates it with command buffers at actual
expert encoding. Mode 0 disables slab residency. Cached-refresh parallelism
from experiment 27 is off in both primary arms.
Pure router or shared-work buffers can omit the association; a buffer containing
both expert work and the next router remains covered. This does not exclude
every router handoff from residency.

Metal provides `MTLCommandBuffer` `useResidencySet` on macOS 15 and later;
the runtime already uses it for other table paths. The cache owns one set of
its slab allocations, which are committed before publication. Mode 2 associates
that set at native and GLM expert-encoding leaves, without a new command split.
Creation or unsupported-command failure falls back to ordinary resource
declarations and prevents recreation until full cache clear. Existing resource
declarations and cache pins remain; pressure relief, unlock, lock failure and
trim clear the set before unlocking. For unretained command buffers a completion
block retains the set through completion; that retains the object, not a snapshot
of its allocations, and does not replace existing owners or pins. Clear and
trim keep their existing drained-caller contract.

The candidate built cleanly with `./tools/zig-build ds4-bench ssd-experts-test
-j4`. Its bench and fixture SHA-256 hashes are
`bd131580e7d738b22a5bde87a57467ee74d75ed78adbdc644c105d544a2ca742` and
`f29ad2e27024b78b0eb70de37fca762a585729a5af95d1df25663addb9c2b68b`,
respectively. The source and API audit and build record are retained in the raw
directory.

## Checks so far

The queue, command-buffer and unretained command-buffer fixture runs all
exited 0. A process-local lock-limit-zero failure injection exited 1 as expected: its log
shows exactly one add, then clear with reason `slab mlock failure`, and no
re-add. The fixture could not establish its required locked cache, so this is
not a successful output check. Its pressure guard exited 0.

At both 8K and 32K, all 256 fixed-history full-vocabulary rows and all
post-evaluation hashes matched between arms in each case. Both recorded 14 slab
allocations totaling 55.98 GiB, with matching cache and routing counters. The
candidate recorded 21,120 association hook calls in each case; that count is
not a count of unique command buffers. These fixed-history checks do not
establish practical task quality. A third check repeats the 8K history with a requested 16 GB expert cache;
its 256 rows and post-evaluation hashes also match exactly, for 768 paired rows
in total. Delta-NLL, KL and top-token changes were zero. The smaller request
produced 958 slots (8.88 GiB) in three slabs (8.89 GiB). Both arms recorded
4,951 misses, 122,672 hits, 3,993 evictions and reuses, and 37.52 GiB of
cumulative logical preads; these are not necessarily physical SSD reads. Of
640 selected calls, 477 were all-hit and 163 mixed, with five split layers.
The candidate recorded 21,125 use calls and only a final clear, with no
reported failures.

## 32K timing

Five profile-free alternating pairs used a fresh 32K frontier, 128 teacher
forced decode tokens, KV restore, a 512-token warm append and another 128-token
decode. The expert cache survives restore. Phase times exclude loading and
snapshot save/restore; process wall includes them. The OS file cache was not
flushed. Seconds are derived from two-decimal throughput output. Steady decode
excludes the first token only.

Mode 2 increased the first fresh decode step and complete fresh decode in all
five pairs, while steady decode throughput was nearly unchanged. Whole inference
time was higher in four pairs. The first mode 1 run had unusually fast prefill;
it is retained, and order and drift limit the precision of the total comparison.
The cause of the fresh-step difference remains under investigation.

All values are mode 1 → mode 2 in pair order; the final column is mean ± sample
standard deviation. Mode 1 ran first in pairs 1, 3 and 5.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 41.737 → 43.789 | 45.870 → 45.524 | 46.367 → 46.567 | 46.279 → 46.512 | 46.423 → 47.092 | 45.335 ± 2.023 → 45.897 ± 1.308 |
| Fresh first step, ms | 167.922 → 728.170 | 208.673 → 717.199 | 181.957 → 780.680 | 205.286 → 729.622 | 180.143 → 766.643 | 188.796 ± 17.496 → 744.463 ± 27.535 |
| Fresh complete decode, s | 3.732 → 4.300 | 3.759 → 4.275 | 3.764 → 4.351 | 3.774 → 4.288 | 3.749 → 4.333 | 3.755 ± 0.016 → 4.309 ± 0.032 |
| Fresh steady decode, tokens/s | 35.630 → 35.560 | 35.770 → 35.690 | 35.460 → 35.570 | 35.590 → 35.700 | 35.580 → 35.610 | 35.606 ± 0.111 → 35.626 ± 0.066 |
| Warm append, s | 7.111 → 7.166 | 7.449 → 7.207 | 7.494 → 7.720 | 7.724 → 7.500 | 7.736 → 7.635 | 7.503 ± 0.255 → 7.446 ± 0.250 |
| Warm complete decode, s | 3.601 → 3.842 | 3.938 → 3.931 | 3.959 → 3.931 | 3.935 → 3.925 | 3.937 → 3.970 | 3.874 ± 0.153 → 3.920 ± 0.047 |
| Inference total, s | 56.180 → 59.096 | 61.018 → 60.937 | 61.584 → 62.569 | 61.711 → 62.225 | 61.846 → 63.030 | 60.468 ± 2.417 → 61.571 ± 1.587 |
| Process wall, s | 58.849 → 61.738 | 63.634 → 63.562 | 64.220 → 65.141 | 64.313 → 64.793 | 64.395 → 65.555 | 63.082 ± 2.385 → 64.158 ± 1.544 |

## 8K timing

The same five-pair procedure at 8K also made the first fresh step and complete
fresh decode slower in all five pairs. Steady throughput was nearly unchanged;
whole inference time was higher in four pairs, and warm outcomes were mixed.
Mode 1 ran first in pairs 1, 3 and 5. Values and units follow the 32K table.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 18.053 → 18.242 | 18.749 → 18.889 | 19.549 → 19.698 | 19.502 → 19.576 | 19.337 → 19.316 | 19.038 ± 0.636 → 19.144 ± 0.592 |
| Fresh first step, ms | 178.377 → 818.288 | 211.194 → 805.415 | 184.915 → 741.918 | 177.319 → 706.065 | 234.496 → 711.266 | 197.260 ± 24.936 → 756.590 ± 52.471 |
| Fresh complete decode, s | 3.656 → 4.294 | 3.698 → 4.287 | 3.695 → 4.217 | 3.651 → 4.201 | 3.713 → 4.194 | 3.683 ± 0.028 → 4.239 ± 0.048 |
| Fresh steady decode, tokens/s | 36.520 → 36.550 | 36.420 → 36.470 | 36.180 → 36.540 | 36.560 → 36.350 | 36.510 → 36.470 | 36.438 ± 0.153 → 36.476 ± 0.080 |
| Warm append, s | 6.789 → 6.717 | 7.401 → 6.725 | 8.008 → 7.876 | 7.718 → 7.923 | 7.688 → 7.554 | 7.521 ± 0.462 → 7.359 ± 0.600 |
| Warm complete decode, s | 3.501 → 3.524 | 3.970 → 3.522 | 4.048 → 4.006 | 3.895 → 4.003 | 3.936 → 3.925 | 3.870 ± 0.214 → 3.796 ± 0.251 |
| Inference total, s | 31.999 → 32.776 | 33.819 → 33.423 | 35.299 → 35.797 | 34.766 → 35.702 | 34.674 → 34.989 | 34.112 ± 1.295 → 34.538 ± 1.368 |
| Process wall, s | 34.412 → 35.340 | 36.338 → 36.035 | 37.818 → 38.277 | 37.241 → 38.209 | 37.199 → 37.467 | 36.602 ± 1.333 → 37.066 ± 1.321 |

## Coverage investigation

Source inspection shows that the measured mode 2 covers cached expert compute,
including the selected parallel routed path, but does not associate the set with
the separate GPU seed-copy command buffers used during fresh prefill. Queue
mode covers those buffers. Normal blit bindings remain in place. The snapshot
synchronization buffers before generation are another association-scope
difference; snapshot save is outside the printed prefill and generation timing
windows. These differences offer possible initialization explanations, but
source inspection alone does not establish the cause of the first-step delay.
One intrusive 32K timeline pair, with 17 tokens per generation, locates the
shift without proving its cause. All 34 compared hashes match, and both arms execute the same 91 refresh and 11
ordinary command buffers. Mode 2 has no explicit uses before fresh decode: all
40 seed command buffers, containing 1,001 sampled fill encoders, lack the
association. The first GPU window increases from 153.054 to 970.795 ms,
with the positive gap before the first associated expert buffer (sequence 287)
rising from 0.669 to 814.940 ms; summed command-buffer spans are 117.945 and 117.786 ms. These
are profiled observations, not the profile-free timing samples.

For a refresh, mode 2 associates 40 buffers: 29 contain both expert and router
work and 11 contain expert work without router work. It omits 11 router-only
and 40 shared-only buffers. All 11 ordinary buffers remain associated. The
trace therefore confirms narrower coverage but not exclusion of every handoff,
and it cannot establish why the driver or GPU waited. Ordinary-token coverage
is not narrowed. The trace shows no changed expert kernel or submission
topology; two of 34 encoder-group multisets differ only in a keepalive combined
with a router encoder.

In this single profile pair, reported fresh prefill was approximately 46.646 →
46.466 seconds and fresh 17-token generation 0.635 → 1.460 seconds. Snapshot
save lies between those scopes. The trace does not show a compensating prefill
saving, and adding seed association could move cost into prefill or the snapshot
boundary rather than remove it.


## Decision

The compute-only policy remains available as opt-in groundwork, but is not
selected in package 25 or any preset. A separate bounded question will test
association on the existing seed blits; that policy has not been implemented.
Any follow-up must account for whether initialization cost moves into prefill
and for the whole workload, not only the first decode step.

## Reproduction

The committed [package 25 settings](../2026-11-05-slab-event-interaction/candidate-settings.json)
are identical to the fixed timing map. With the checkpoint and `zig-out/bin/ds4-bench` built at `cd42c5ac`, the following uses fresh output directories.
Run each guarded job sequentially, with no competing heavy model work or
builds during timing. The driver clears inherited DS4 settings, changes the
specified mode between arms and alternates timing order. Its default cache is
64 GB. The diagnostic settings add counters for quality runs; those counters
are not enabled for the timing comparison.

```nu
let out = '.workspaces/hillclimb-20261104/29-scoped-residency-repro'
let fixed = 'speed-bench/experiments/2026-11-05-slab-event-interaction/candidate-settings.json'
mkdir $out
open $fixed | merge {DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: '1', DS4_METAL_V41_RESIDENT_ROUTE_PROFILE: '1', DS4_METAL_SELECTED_READBACK_PROFILE: '1'} | to json | save ($out | path join diagnostic.json)
for frontier in [8192 32768] {
  python3 speed-bench/ideas/pressure_run.py ($out | path join $'guard-timing-($frontier)') -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu ($out | path join $'timing-($frontier)') timing --flag DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY --control 1 --candidate 2 --frontier $frontier --tokens 128 --samples 5 --extra-settings $fixed
  python3 speed-bench/ideas/pressure_run.py ($out | path join $'guard-quality-($frontier)') -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu ($out | path join $'quality-($frontier)') quality --flag DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY --control 1 --candidate 2 --frontier $frontier --tokens 128 --extra-settings ($out | path join diagnostic.json)
}
python3 speed-bench/ideas/pressure_run.py ($out | path join guard-quality-16gb) -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu ($out | path join quality-16gb) quality --flag DS4_METAL_STREAMING_EXPERT_SLAB_RESIDENCY --control 1 --candidate 2 --frontier 8192 --cache 16GB --tokens 128 --extra-settings ($out | path join diagnostic.json)
```

Raw logs, settings, guard records and all samples are retained under
`.workspaces/hillclimb-20261104/29-scoped-residency/`. All seven guards completed with child return code 0, pressure level 1 only and
no adverse samples. Swap ranged from 29.016904 to 29.019043 GiB.
