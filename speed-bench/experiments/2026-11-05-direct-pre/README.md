# Direct PRE source in scalar decode

The direct PRE prototype removed 39 copies per scalar token and preserved the tested outputs, but did not establish a reliable decode benefit. **The prototype was not adopted and has been removed.** Its measured runtime `03aaf051` remains in history; selected package 32 remains unchanged. The November 5, 2026 budget was 12:16–13:16 Eastern, and work closed before 13:16.

## The boundary

DeepSeek V4.1 uses hyper-connections (HC) to combine residual streams before attention and the feed-forward network (FFN). PRE is the vector of weights used for that input combination. In scalar decode, the FFN HC operation produces the next PRE values in `ffn_split`. The baseline copies its first four F32 values, 16 bytes, into the persistent `graph.pre` tensor after each layer. The next layer's attention HC reads `graph.pre`. This copy is small, but the Metal blit closes the compute encoder and the following operation opens another one.

The candidate lets attention HC in layers 1–39 read the preceding layer's `ffn_split` directly. Layer 0 still reads the initialized `graph.pre`; the final layer still copies its PRE to that persistent tensor for the output head and later state consumers. Attention writes `attn_split`, so it does not overwrite the PRE it reads. The subsequent FFN HC reads `attn_split` and then overwrites `ffn_split` for the current layer. Serial dependencies and the parallel FFN join must continue to establish this order. The tensor owner remains `graph.pre`; selecting another input does not replace or alias that handle. Batched prefill already reads the preceding split directly, but that precedent does not validate the scalar selector.

The candidate gate, `DS4_METAL_V41_DIRECT_PRE=1`, is limited to queued single-host scalar streaming with four HC streams. It excludes image input, steering or steering capture, DSpark and verification, a nonzero decoder replay floor, prediction and oracle activity, graph trace capture, graph quality mode, and imatrix collection. Alternate executors and batched suffix or replay execution retain their existing behavior because the selector is local to the scalar step. The gate does not exclude `graph.prerope` or an ordinary scalar continuation after a suffix graft. Benchmark logits dumps, hash logging and Metal timeline profiling do not themselves disable it; these are distinct from graph trace capture and graph quality mode. Both the fused HC input and its fallback receive the same selected source.

The final publication preserves the value needed by the head, deferred logits, scalar prefill checkpoints and accepted-state copies. The candidate does not change the HC arithmetic or precision. Source review passed; evidence for the final PRE value must come from logits and separate-head checks, because canonical serialized snapshots omit PRE and residual state.

## Motivation and setup

The preceding [ownership experiment](../2026-11-05-deferred-set-release/README.md) found no consistent net benefit. Its source and timeline review identified repeated `tensor_copy` blit to HC input edges in an earlier 32K profile. Twenty-eight such adjacent edges summed to 6.968 ms per warm ordinary token in that diagnostic trace; a separate deferred-release trace showed 6.954 ms. The sums locate observed intervals and are neither additive critical-path costs nor an estimate of recoverable time. The instrumentation itself imposes boundaries, so the final latency comparison must run with profiling off.

The starting source was runtime `1e0e5841`, closure `a31acf97`, with [selected package 32 settings](../2026-11-05-cpu-expert-boundary/candidate-settings.json). The candidate runtime is `03aaf051`; its benchmark binary SHA-256 is `977263e83915211583f9151db11e9dd622c1cc25e730d2dc3d100fda09c80076`, and the graph fixture SHA-256 is `5d6d0a1f6e0fb29c5b51aaa54fcfa5923b0fd3ad979db5eec75d44555c537c40`. Residency deduplication, unretained buffers and deferred set release are off. The primary comparison used the candidate binary with the new flag off and on, holding the rest fixed. Fixed-history checks cover 8K, 32K and a requested 16 GB miss case. Profile-off latency used five alternating pairs at 8K and 32K. A timeline checked copy and compute/blit boundaries independently of latency; an additional 8K comparison used the frozen selected binary.

Local raw inputs and outputs are under `.workspaces/hillclimb-20261104/36-direct-pre/`. The settings files there record the actual comparison. The existing pressure guard and single heavy-job rule apply.

The machine is an M5 Max with 128 GiB memory, using `gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, an 8,192-token prefill chunk and context allocation of 40,960. Each timing process prefills to its frontier, takes a KV snapshot before decoding 128 teacher tokens, restores that snapshot, appends 512 tokens and decodes another 128. The expert cache survives the restore; the OS file cache is not flushed. Phase timers exclude startup and snapshot operations; process wall time includes them. The 64 GB and 16 GB cache figures are requests, not the effective dynamic expert-cache sizes.

## Findings

Three fixed-history checks compared 256 full-vocabulary rows each: 8K and 32K frontiers with a requested 64 GB cache, and 8K with a requested 16 GB cache. Across them, **768 of 768 rows were byte-exact and all 768 post-evaluation hashes matched**; measured delta-NLL and KL were zero. Each pair used the same arguments and environment apart from the direct PRE flag and output paths. Checked non-time route, cache and slab counts matched. The smaller-cache case exercised 3,993 evictions and reuses, 163 mixed selections and five split layers. These checks cover the recorded histories; they are not an autonomous task-quality evaluation.

Real-model session fixtures passed with the flag both off and on, with ordinary fused HC input and with that fusion disabled to exercise the fallback. They checked continuation at positions 1, 2, 3, 127, 128 and 129, separate-final-head parity, and public-session reuse, cancellation and bounds. The separate-head check provides evidence for the final PRE consumer; snapshot byte equality alone does not, because those snapshots omit PRE and residual state.

All five qualification guards and all three timing guards completed with child exit zero and pressure level 1 throughout. Swap remained at its pre-existing 29.0034 GiB; no lock failures or unexpected errors were found.

An intrusive 32K timeline pair covers 17 teacher tokens at each of two frontiers. All 34 hashes and normalized per-token recorded compute-group multisets matched. It recorded exactly 39 fewer copy blits per token, while command-buffer and compute-encoder counts were unchanged. The timeline forces per-operation compute passes, so these counts do not show how long an unprofiled compute encoder remains open. Concurrent group labels are truncated, limiting the kernel-level comparison.

The old 28 adjacent copy-to-HC edges per ordinary token became expansion-to-HC edges. Their positive sampled gap sums disappeared in that category, but ordinary embedding-to-head windows remained near 27.4 ms in both arms. Sampled encoder spans overlap and already exceed those windows; changing endpoints also changes the gap being measured. The old gap sums therefore do not establish recovered idle time or a driver cause. This single profiled pair establishes the structural change, not a latency benefit; the profile-off comparisons follow.

### Profile-off 32K timing

Five alternating pairs show no clear overall benefit: mean inference time was 62.038 ± 3.208 seconds with the flag off and 62.133 ± 2.159 seconds on. Each cell is off → on; the final column is mean ± sample standard deviation. Baseline ran first in pairs 1, 3 and 5. Complete decode covers all 128 tokens; steady decode excludes only the first token, so it does not by itself establish completed initialization. Seconds derived from throughput have the precision limits of the two-decimal throughput output.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 41.964 → 43.544 | 47.699 → 46.003 | 47.999 → 48.015 | 48.007 → 47.793 | 48.081 → 48.428 | 46.750 ± 2.679 → 46.756 ± 2.021 |
| Fresh first, ms | 180.643 → 173.255 | 220.912 → 169.759 | 188.479 → 145.905 | 163.634 → 231.741 | 192.914 → 174.004 | 189.316 ± 20.897 → 178.933 ± 31.703 |
| Fresh complete decode, s | 3.732 → 3.836 | 3.745 → 3.674 | 3.696 → 3.647 | 3.679 → 3.726 | 3.691 → 3.679 | 3.709 ± 0.028 → 3.712 ± 0.075 |
| Fresh steady decode, tok/s | 35.770 → 34.680 | 36.040 → 36.240 | 36.210 → 36.270 | 36.130 → 36.350 | 36.310 → 36.230 | 36.092 ± 0.206 → 35.954 ± 0.714 |
| Warm append, s | 7.079 → 7.668 | 7.785 → 7.485 | 7.788 → 7.841 | 7.902 → 7.877 | 7.996 → 7.912 | 7.710 ± 0.364 → 7.757 ± 0.178 |
| Warm first, ms | 34.100 → 35.955 | 33.868 → 36.886 | 33.781 → 33.119 | 33.402 → 36.164 | 33.566 → 74.475 | 33.743 ± 0.270 → 43.320 ± 17.475 |
| Warm complete decode, s | 3.536 → 3.778 | 3.951 → 3.914 | 3.937 → 3.936 | 3.963 → 3.943 | 3.958 → 3.965 | 3.869 ± 0.186 → 3.907 ± 0.075 |
| Warm steady decode, tok/s | 36.270 → 33.940 | 32.420 → 32.750 | 32.540 → 32.540 | 32.320 → 32.500 | 32.360 → 32.640 | 33.182 ± 1.728 → 32.874 ± 0.604 |
| Inference total, s | 56.310 → 58.826 | 63.179 → 61.077 | 63.421 → 63.438 | 63.551 → 63.339 | 63.727 → 63.985 | 62.038 ± 3.208 → 62.133 ± 2.159 |
| Process wall, s | 58.875 → 61.536 | 65.790 → 63.661 | 65.964 → 66.084 | 66.144 → 65.914 | 66.273 → 66.785 | 64.609 ± 3.211 → 64.796 ± 2.167 |

The first baseline prefill was unusually fast at 41.964 seconds versus 43.544 seconds for the candidate, and its warm steady decode was 36.27 versus 33.94 tokens/s. The candidate's first fresh steady result was 34.68 tokens/s; later fresh steady differences were small and mixed. In the final pair candidate warm first-token latency reached 74.475 ms. The first two pairs have larger opposite differences and prefill drifts across the sequence, so neither arm order nor those outliers should be ignored. The 8K same-binary comparison is reported below.

### Profile-off 8K timing

The same-binary flag comparison at 8K produced mean inference time of 34.667 ± 0.416 seconds off and 34.011 ± 0.714 seconds on. Much of the difference is fresh prefill, 19.606 → 19.109 seconds; this scalar decode selector does not establish the cause of that prefill difference. The table uses the same conventions and alternating order as the 32K comparison.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 19.520 → 18.800 | 20.252 → 18.831 | 19.247 → 19.325 | 19.607 → 19.213 | 19.403 → 19.375 | 19.606 ± 0.386 → 19.109 ± 0.274 |
| Fresh first, ms | 236.927 → 204.118 | 204.753 → 191.069 | 163.734 → 169.416 | 186.463 → 190.551 | 169.519 → 170.736 | 192.279 ± 29.647 → 185.178 ± 14.827 |
| Fresh complete decode, s | 3.661 → 3.606 | 3.644 → 3.616 | 3.596 → 3.604 | 3.649 → 3.622 | 3.601 → 3.619 | 3.630 ± 0.030 → 3.613 ± 0.008 |
| Fresh steady decode, tok/s | 37.090 → 37.330 | 36.930 → 37.090 | 37.010 → 36.980 | 36.680 → 37.010 | 37.010 → 36.830 | 36.944 ± 0.158 → 37.048 ± 0.184 |
| Warm append, s | 7.264 → 7.091 | 7.575 → 7.115 | 7.714 → 7.768 | 7.836 → 7.806 | 7.793 → 7.800 | 7.637 ± 0.230 → 7.516 ± 0.377 |
| Warm first, ms | 32.715 → 32.602 | 33.579 → 33.769 | 32.656 → 32.878 | 33.093 → 32.658 | 32.674 → 33.075 | 32.943 ± 0.398 → 32.996 ± 0.471 |
| Warm complete decode, s | 3.775 → 3.691 | 3.778 → 3.714 | 3.798 → 3.820 | 3.805 → 3.820 | 3.817 → 3.821 | 3.795 ± 0.018 → 3.773 ± 0.065 |
| Warm steady decode, tok/s | 33.940 → 34.710 | 33.920 → 34.500 | 33.730 → 33.540 | 33.670 → 33.530 | 33.560 → 33.520 | 33.764 ± 0.163 → 33.960 ± 0.594 |
| Inference total, s | 34.220 → 33.188 | 35.249 → 33.276 | 34.355 → 34.517 | 34.897 → 34.461 | 34.614 → 34.615 | 34.667 ± 0.416 → 34.011 ± 0.714 |
| Process wall, s | 36.655 → 35.669 | 37.745 → 35.892 | 36.811 → 36.942 | 37.362 → 36.950 | 37.002 → 37.041 | 37.115 ± 0.440 → 36.499 ± 0.661 |

Fresh steady decode averaged 36.944 → 37.048 tokens/s. Warm steady averaged 33.764 → 33.960 tokens/s, driven by the first two pairs; the candidate was slower in each of the last three. These mixed pairs do not establish a reliable decode gain. The direct comparison against the frozen selected package is reported below.

The source follow-up found no simple additional fusion unlocked by direct PRE. The existing next-HC input already combines several operations, and a fusion with the preceding expansion would require a new cross-residual dependency and preserve the existing BF16 boundaries and previous PRE values. The recorded gap changes do not establish a hidden gain that requires retaining this selector.

### Direct 8K comparison with selected package 32

This five-pair comparison uses the frozen selected `b7e7af9d` binary (SHA-256 `fd4596242336958607ae44f302e77bdc39e4a0ae1a28830df4d93da38ce88921`) against `03aaf051` (SHA-256 `977263e83915211583f9151db11e9dd622c1cc25e730d2dc3d100fda09c80076`) with direct PRE enabled. Both use package 32 settings; the candidate adds only `DS4_METAL_V41_DIRECT_PRE=1`. Baseline ran first in pairs 1, 3 and 5. Cells show selected → candidate, and the final column reports mean ± sample SD, with the timing and rounding conventions above.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 18.032 → 18.382 | 18.523 → 18.869 | 19.273 → 19.474 | 19.489 → 19.531 | 19.452 → 19.360 | 18.954 ± 0.647 → 19.123 ± 0.490 |
| Fresh first, ms | 163.466 → 206.384 | 195.919 → 172.707 | 179.536 → 187.439 | 187.416 → 171.614 | 169.591 → 190.138 | 179.186 ± 13.106 → 185.656 ± 14.298 |
| Fresh complete decode, s | 3.590 → 3.633 | 3.612 → 3.588 | 3.612 → 3.625 | 3.603 → 3.694 | 3.580 → 3.605 | 3.599 ± 0.014 → 3.629 ± 0.040 |
| Fresh steady decode, tok/s | 37.050 → 37.060 | 37.180 → 37.180 | 37.010 → 36.950 | 37.190 → 36.050 | 37.240 → 37.190 | 37.134 ± 0.099 → 36.886 ± 0.478 |
| Warm append, s | 6.721 → 6.829 | 7.217 → 6.738 | 7.820 → 7.986 | 7.778 → 8.131 | 7.775 → 7.719 | 7.462 ± 0.484 → 7.481 ± 0.654 |
| Warm first, ms | 30.831 → 31.355 | 32.219 → 31.381 | 33.128 → 32.770 | 33.140 → 36.274 | 33.045 → 32.992 | 32.473 ± 0.995 → 32.954 ± 2.006 |
| Warm complete decode, s | 3.430 → 3.467 | 3.747 → 3.438 | 3.912 → 3.924 | 3.902 → 3.910 | 3.864 → 3.839 | 3.771 ± 0.202 → 3.716 ± 0.242 |
| Warm steady decode, tok/s | 37.360 → 36.970 | 34.190 → 37.280 | 32.740 → 32.640 | 32.820 → 32.790 | 33.160 → 33.360 | 34.054 ± 1.936 → 34.608 ± 2.316 |
| Inference total, s | 31.773 → 32.312 | 33.099 → 32.633 | 34.617 → 35.009 | 34.771 → 35.265 | 34.672 → 34.523 | 33.786 ± 1.320 → 33.948 ± 1.378 |
| Process wall, s | 34.157 → 34.951 | 35.644 → 35.237 | 37.050 → 37.496 | 37.262 → 37.720 | 37.168 → 36.963 | 36.256 ± 1.347 → 36.473 ± 1.293 |

Mean inference time was 33.786 ± 1.320 → 33.948 ± 1.378 seconds; this variation does not establish a meaningful slowdown or confirm the small same-binary 8K benefit. Decode results are mixed: candidate fresh complete decode was slower in four of five pairs, while its warm mean benefits from pair 2. The alternating order and individual samples remain visible above. No direct selected-binary comparison at 32K, 16 GB timing, or fresh practical-task check was run.

## Reproduction

The following reconstructs the fixed and diagnostic settings from the committed package in a new output directory. Run from a checkout of historical runtime `03aaf051` with its candidate binary built; the prototype has been removed from the current runtime. The quality stage records logits and diagnostic counters; the timing stage is profile-off and alternates five pairs. Run each guarded invocation sequentially and pause unrelated work during timing.

```nu
let out = '.workspaces/hillclimb-20261104/36-direct-pre-repro'
mkdir $out
let fixed = (open speed-bench/experiments/2026-11-05-cpu-expert-boundary/candidate-settings.json)
$fixed | to json | save ($out | path join fixed.json)
$fixed | merge {DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: '1', DS4_METAL_V41_RESIDENT_ROUTE_PROFILE: '1', DS4_METAL_SELECTED_READBACK_PROFILE: '1'} | to json | save ($out | path join diagnostic.json)
for case in [{name: '8k', frontier: 8192, cache: '64GB'} {name: '32k', frontier: 32768, cache: '64GB'} {name: '16gb', frontier: 8192, cache: '16GB'}] {
    let stages = if $case.name == '16gb' { ['quality'] } else { ['quality' 'timing'] }
    for stage in $stages {
        let settings = if $stage == 'quality' { 'diagnostic.json' } else { 'fixed.json' }
        python3 speed-bench/ideas/pressure_run.py ($out | path join $'guard-($stage)-($case.name)') -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu ($out | path join $'($stage)-($case.name)') $stage --flag DS4_METAL_V41_DIRECT_PRE --control 0 --candidate 1 --frontier $case.frontier --cache $case.cache --tokens 128 --samples 5 --extra-settings ($out | path join $settings)
    }
}
```

## Decision and limits

Source review passed and the timeline confirms the expected copy reduction. The paired timing evidence does not establish a useful speed gain, and source review found no concrete simple fusion unlocked by retaining the selector. The prototype was removed. The restored `src`, `build.zig` and `tools/zig-build` match closure `a31acf97` by source diff, and the rebuilt benchmark SHA-256 is `fb1118935b27bae8f745e6b6b3ecb6f356e14b65d073f3cfaa561aa0565ce0cd`. The historical candidate binary is archived locally as `.workspaces/hillclimb-20261104/36-direct-pre/ds4-bench-candidate`.
