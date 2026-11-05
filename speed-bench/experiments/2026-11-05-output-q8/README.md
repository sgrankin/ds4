# Scalar Q8 attention output projection

The two-SIMD-group candidate did not show a consistent gain across contexts and was not retained. Practical task quality was not evaluated. The experiment asked whether using two SIMD groups instead of four for the scalar attention output-B projection improves decode while keeping selected package 32 fixed. The November 5, 2026 budget was 13:10–14:10 Eastern; the experiment closed within that window.

## Target and mechanism

The latest [diagnostic profile](../2026-11-05-direct-pre/README.md) recorded about 3.619 ms of summed spans per token for the static Q8 output-B projection with fused hyper-connection (HC) expansion. Those spans identify a target; they are not an additive critical-path cost or a guaranteed latency saving. Final timings run without profiling.

The projection maps 8,192 input values to 5,120 output values. The existing [Metal kernel](../../../src/metal/dense.metal) assigns two output rows to a threadgroup. Each SIMD group works on interleaved Q8 weight blocks, accumulating scaled dot products for both rows. It reduces within each SIMD group, writes the partial sums to threadgroup memory, and reduces those partials before rounding the projection to BF16. The fused HC expansion then combines that value with four residual streams and rounds each output to BF16.

The candidate uses the same static kernel with two SIMD groups per threadgroup instead of four. Each group therefore traverses more blocks and the partial dot products are grouped differently before reduction; floating-point outputs may differ. The two-row layout (`NR0=2`), 256-byte scratch allocation, BF16 boundaries and HC arithmetic remain unchanged. The local selector `DS4_METAL_V41_OUTPUT_Q8_NSG2=1` is intended only for the literal one-row, no-add, tensor-parallel-one path and only when no global SIMD-group override is set. One-row prefill or session remainders can also be eligible; the gate is not confined to a decoder entry point. Other shapes, batched prefill and explicit global overrides retain their existing paths. The timeline checked which pipeline actually ran.

## Setup

Closure `f7549a01` removed the previous direct-PRE prototype and restored runtime source to `a31acf97`. Its rebuilt benchmark SHA-256 is `fb1118935b27bae8f745e6b6b3ecb6f356e14b65d073f3cfaa561aa0565ce0cd`. The primary comparison used one candidate binary with the selector off and on. [Selected package 32](../2026-11-05-cpu-expert-boundary/candidate-settings.json) stays fixed; residency deduplication, unretained command buffers and deferred set release remain off. The candidate runtime revision is `4a09de67`; its benchmark SHA-256 is `210b18295b87aa2ab3105f19d26ff7e31e4d6a564cb828b394d775b709cc062b`, and the fixture SHA-256 is `93c68c457012c7a30929467f9c3a68433d4f47b9c71b9bf1c364590a9dd6311b`.

The measured workload used an M5 Max with 128 GiB memory and `gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, an 8,192-token prefill chunk and context allocation of 40,960. Each timing process prefills to its frontier, snapshots KV before decoding 128 teacher tokens, restores it, appends 512 tokens and decodes another 128. Expert cache survives the restore; the OS file cache is not flushed. Phase timers exclude startup and snapshot operations; process wall time includes them. Cache sizes are requests, not effective dynamic cache sizes.

Fixed-history checks covered 8K and 32K with requested 64 GB cache, and at 8K with requested 16 GB cache. Five alternating profile-off pairs ran at both frontiers, and a separate diagnostic timeline inspected the kernel and dependencies before final timing. Numerical drift is a measured tradeoff; fixed-history metrics do not establish practical task quality. Model jobs ran sequentially under the pressure guard, with unrelated work paused during final timing.

Raw settings, logs and outputs are local and ignored under `.workspaces/hillclimb-20261104/37-output-q8/`. The committed settings file and existing driver reconstruct the quality and timing setup:

```nu
let out = '.workspaces/hillclimb-20261104/37-output-q8-repro'
mkdir $out
let fixed = (open speed-bench/experiments/2026-11-05-cpu-expert-boundary/candidate-settings.json)
$fixed | to json | save ($out | path join fixed.json)
$fixed | merge {DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY: '1', DS4_METAL_V41_RESIDENT_ROUTE_PROFILE: '1', DS4_METAL_SELECTED_READBACK_PROFILE: '1'} | to json | save ($out | path join diagnostic.json)
for case in [{name: '8k', frontier: 8192, cache: '64GB'} {name: '32k', frontier: 32768, cache: '64GB'} {name: '16gb', frontier: 8192, cache: '16GB'}] {
    let stages = if $case.name == '16gb' { ['quality'] } else { ['quality' 'timing'] }
    for stage in $stages {
        let settings = if $stage == 'quality' { 'diagnostic.json' } else { 'fixed.json' }
        python3 speed-bench/ideas/pressure_run.py ($out | path join $'guard-($stage)-($case.name)') -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu ($out | path join $'($stage)-($case.name)') $stage --flag DS4_METAL_V41_OUTPUT_Q8_NSG2 --control 0 --candidate 1 --frontier $case.frontier --cache $case.cache --tokens 128 --samples 5 --extra-settings ($out | path join $settings)
    }
}
```

Run from the measured candidate revision after building its benchmark. Record the actual commands, candidate identity, timeline setup, pressure and any deviations with the results.

## Findings and decision

Independent source review passed. The existing `test-deepseek41-metal --literal-shapes` fixture passed with explicit global NSG 4 and NSG 2. This checks static and generic agreement with matched reduction order; it deliberately bypasses the local selector. Source establishes eligibility; fixed-history drift measures consequences but does not by itself prove selector activation. The pipeline timeline establishes the selected pipeline and geometry. Fixed-history quality shows numerical drift, as expected from reassociation. Each frontier compares 128 full-vocabulary rows on the same teacher history. Delta-NLL is candidate minus baseline; positive values indicate lower likelihood for the teacher tokens. KL compares the full distributions on those fixed histories. Neither metric determines practical task quality or automatically rejects the candidate.

| Requested cache | Frontier | Mean delta-NLL, nats/token | Mean KL | Top-1 changes |
| --- | ---: | ---: | ---: | ---: |
| 64 GB | 8,192 | +0.00329807 | 0.00473518 | 6 |
| 64 GB | 8,704 | -0.00852180 | 0.00707649 | 2 |
| 64 GB | 32,768 | +0.00900439 | 0.01676381 | 7 |
| 64 GB | 33,280 | +0.01138597 | 0.00986103 | 2 |
| 16 GB | 8,192 | -0.04567971 | 0.09840794 | 12 |
| 16 GB | 8,704 | +0.05566131 | 0.14907282 | 19 |

At 32K these mean delta-NLL values correspond to approximately 1% higher teacher perplexity. At 8K, 2 of 256 rows are byte-exact and no post-evaluation hashes match; maximum KL is 0.10814548 and the per-row delta-NLL range is -0.27619606 to +0.46907450. Route substitutions change from 226 to 232 and native misses from 236 to 242, while final cache counts match. The 8K guard completed with exit zero, pressure level 1 in all 67 samples and unchanged pre-existing swap of 29.0034 GiB. At 32K only the initial fresh row is byte-exact, no post-evaluation hashes match, and maximum KL is 0.18520415. Cache entries and misses change from 6,011 to 6,012 and mixed selections from 8 to 9; these counts do not establish equal cached expert identities. The 32K guard also exited zero with pressure level 1 throughout and unchanged swap. The requested 16 GB cache-capacity stress case used an effective 958-expert, 8.88 GiB cache and showed much larger drift: maximum KL 2.72945921, KL p99 1.45135787, per-row delta-NLL from -2.60597581 to +3.34039434, and maximum absolute logit difference 12.23213744. Only its initial fresh row was byte-exact and no post-evaluation hashes matched. Misses changed from 4,951 to 4,987, evictions and reuses from 3,993 to 4,029, mixed selections from 163 to 164, and split layers from 5 to 4. The changed route and cache work accompanies the larger drift, but its causal contribution is uncertain. Both arms used three slabs and locked 8.89 GiB without lock failures. Its guard exited zero with pressure level 1 throughout and unchanged swap. Profile-off timings are reported below.


### Diagnostic timeline

One intrusive 32K pair recorded 17 teacher tokens at each frontier. Pipeline creation keys changed from `_static_nsg=4` to `_static_nsg=2`, and target geometry from 2,560 threadgroups of 32 × 4 threads to 2,560 groups of 32 × 2. Each of 34 scalar windows recorded 40 target calls. One further target call appeared outside those windows in each trace, consistent with the one-row gate extending beyond scalar decode.

The sums of 40 target spans for ordinary tokens were 3.664 → 3.675 ms fresh and 3.600 → 3.687 ms warm; corresponding complete embedding-to-head windows were 27.765 → 27.423 ms and 27.286 → 27.381 ms. These are diagnostic observations, not profile-off comparative performance or additive critical-path costs. First-window changes cannot be attributed to the target: its warm first span sum increased 3.576 → 3.641 ms while the whole window fell 50.221 → 39.761 ms.

All 34 recorded post-evaluation hashes differ. Normalized recorded compute-group multisets and command-buffer, encoder and blit counts match after excluding keepalive annotations, but truncated concurrent labels limit interpretation. Resource structure matches while cache and load counters differ; the records do not prove identical expert identities. Existing broad dependency categories remain, and this trace does not establish a new serialization dependency. Both guards completed with child exit zero. Practical answer and tool checks have not yet been run.


### Profile-off 32K timing

Five alternating pairs used the same binary, selector off → on. Baseline ran first in pairs 1, 3 and 5. Cells show off → on; the last column is mean ± sample standard deviation. Complete decode includes all 128 tokens; steady decode excludes only the first, and does not establish that initialization is complete. Seconds derived from throughput inherit the precision of two-decimal throughput output.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 44.116 → 43.232 | 46.847 → 45.700 | 47.470 → 47.638 | 47.711 → 47.618 | 47.975 → 48.122 | 46.824 ± 1.570 → 46.462 ± 2.030 |
| Fresh first, ms | 189.784 → 195.813 | 198.862 → 195.184 | 191.368 → 187.086 | 172.532 → 220.390 | 192.002 → 174.438 | 188.910 ± 9.793 → 194.582 ± 16.807 |
| Fresh complete decode, s | 3.702 → 3.888 | 3.697 → 3.711 | 3.719 → 3.693 | 3.682 → 3.722 | 3.693 → 3.674 | 3.699 ± 0.013 → 3.738 ± 0.086 |
| Fresh steady decode, tok/s | 36.160 → 34.390 | 36.300 → 36.130 | 36.010 → 36.220 | 36.190 → 36.270 | 36.280 → 36.290 | 36.188 ± 0.116 → 35.860 ± 0.824 |
| Warm append, s | 7.229 → 7.758 | 7.702 → 7.593 | 7.902 → 7.865 | 8.003 → 7.943 | 7.922 → 7.807 | 7.751 ± 0.313 → 7.793 ± 0.131 |
| Warm first, ms | 35.694 → 35.567 | 33.605 → 36.190 | 37.185 → 33.628 | 33.480 → 37.345 | 33.228 → 36.836 | 34.638 ± 1.732 → 35.913 ± 1.442 |
| Warm complete decode, s | 3.534 → 3.789 | 3.947 → 3.926 | 3.956 → 3.967 | 3.913 → 3.968 | 3.947 → 3.958 | 3.859 ± 0.183 → 3.922 ± 0.076 |
| Warm steady decode, tok/s | 36.300 → 33.840 | 32.460 → 32.650 | 32.410 → 32.300 | 32.730 → 32.320 | 32.450 → 32.390 | 33.270 ± 1.699 → 32.700 ± 0.652 |
| Inference total, s | 58.580 → 58.667 | 62.193 → 60.931 | 63.047 → 63.162 | 63.309 → 63.251 | 63.537 → 63.561 | 62.133 ± 2.050 → 61.914 ± 2.096 |
| Process wall, s | 61.358 → 61.397 | 64.787 → 63.505 | 65.708 → 65.792 | 65.909 → 65.803 | 66.005 → 66.146 | 64.753 ± 1.958 → 64.529 ± 2.043 |

Mean inference time was 62.133 → 61.914 seconds, but all three baseline-first pairs were slower with the candidate and both candidate-first pairs were faster; pair 2 drives the mean difference. Fresh steady decode averaged 36.188 → 35.860 tokens/s, including a 34.39 candidate result in pair 1; later pairs are mixed. Warm steady averaged 33.270 → 32.700 tokens/s, including 36.30 → 33.84 in pair 1, with mostly small later differences. Warm complete decode was slower in four of five pairs. Prefill drift and order effects limit attribution, and the mean alone does not establish a benefit. The 32K guard exited zero; all 621 pressure samples were level 1 and swap remained at its initial 29.0037 GiB. Saved arguments matched after output paths were normalized, and only the selector differed between arms. The 8K comparison is reported below.


### Profile-off 8K timing

The same five-pair procedure and table conventions apply at 8K.

| Scope | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh prefill, s | 19.512 → 18.661 | 18.920 → 18.880 | 19.074 → 19.213 | 19.597 → 19.339 | 19.365 → 19.605 | 19.293 ± 0.288 → 19.140 ± 0.374 |
| Fresh first, ms | 189.018 → 198.231 | 205.458 → 184.084 | 177.666 → 221.020 | 178.896 → 188.760 | 180.932 → 184.448 | 186.394 ± 11.539 → 195.309 ± 15.463 |
| Fresh complete decode, s | 3.607 → 3.622 | 3.612 → 3.622 | 3.603 → 3.648 | 3.623 → 3.630 | 3.612 → 3.676 | 3.611 ± 0.008 → 3.640 ± 0.023 |
| Fresh steady decode, tok/s | 37.160 → 37.090 | 37.280 → 36.940 | 37.080 → 37.070 | 36.880 → 36.900 | 37.020 → 36.370 | 37.084 ± 0.150 → 36.874 ± 0.293 |
| Warm append, s | 7.023 → 7.047 | 7.438 → 7.103 | 7.638 → 7.811 | 7.612 → 7.732 | 7.719 → 7.748 | 7.486 ± 0.278 → 7.488 ± 0.379 |
| Warm first, ms | 33.721 → 32.532 | 32.427 → 32.327 | 32.873 → 36.505 | 32.511 → 34.589 | 33.118 → 32.354 | 32.930 ± 0.523 → 33.661 ± 1.851 |
| Warm complete decode, s | 3.768 → 3.721 | 3.786 → 3.739 | 3.796 → 3.816 | 3.846 → 3.806 | 3.830 → 3.795 | 3.805 ± 0.032 → 3.776 ± 0.043 |
| Warm steady decode, tok/s | 34.010 → 34.430 | 33.840 → 34.260 | 33.750 → 33.600 | 33.300 → 33.670 | 33.450 → 33.750 | 33.670 ± 0.290 → 33.942 ± 0.377 |
| Inference total, s | 33.910 → 33.051 | 33.755 → 33.345 | 34.111 → 34.488 | 34.678 → 34.508 | 34.526 → 34.824 | 34.196 ± 0.395 → 34.043 ± 0.790 |
| Process wall, s | 36.354 → 35.496 | 36.207 → 35.777 | 36.615 → 36.882 | 37.170 → 37.052 | 36.998 → 37.529 | 36.669 ± 0.411 → 36.547 ± 0.870 |

Fresh complete decode was slower in all five pairs, while warm complete decode was faster in four of five; the two decode windows largely cancel. Mean inference time was 34.196 → 34.043 seconds, with most of the difference in fresh prefill, 19.293 → 19.140 seconds. This does not establish a consistent gain from the selector. The guard exited zero; all 351 pressure samples were level 1 and swap remained at its initial 29.0037 GiB. Saved arguments matched after normalizing output paths, and only the selector differed between arms.

## Decision and limits

The local NSG2 selector was removed and the restored build passed. Runtime source under `src`, `build.zig` and `tools/zig-build` matches `f7549a01` by source diff. The restored benchmark SHA-256 is `7bb7e691e28dada2e49a48a5c33cf4c3d38186b6b5ad4ee24bbfbdbe48002b41`; the tested candidate binary remains locally at `.workspaces/hillclimb-20261104/37-output-q8/ds4-bench-candidate`. The 8K decode windows move in opposite directions, the 32K pairs show order effects and mixed decode results, and the diagnostic target spans do not improve. The timeline did not establish a new dependency to pursue. Numerical drift is a measured cost, including larger tails under cache-capacity stress; imperfect replication is not an automatic rejection criterion. No direct comparison with the frozen selected binary, 16 GB timing, or fresh practical answer or tool check was run. The tested candidate remains inspectable at `4a09de67`.
