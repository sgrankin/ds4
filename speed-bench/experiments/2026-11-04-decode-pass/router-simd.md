# SIMD exchanges in the decode router

The SIMD router remains unadopted: three five-pair comparisons did not show a consistent decode benefit. The results do not prove a regression; changes are small relative to observed variation. The focused fixture and tested full-model fixed histories matched exactly.

## Hypothesis and mechanism

For each token, the router transforms expert logits into probabilities, ranks the biased scores and normalizes the selected experts' unbiased probabilities into weights. The baseline uses a bitonic sorting network with threadgroup-memory exchanges and barriers. For 384 experts it pads the network to 512 positions; padding must sort after real experts, and strict score comparisons determine the existing order of ties.

The candidate keeps that comparator network but exchanges indices and scores in registers for strides smaller than 32, using SIMD shuffles. Larger strides still use threadgroup memory; the candidate reloads and writes indices at the boundaries between the two kinds of exchange. This reduces the network's threadgroup barriers from 46 to 16. The hypothesis is that less synchronization reduces decode latency without changing probabilities, ordered expert IDs or weights. The candidate kernel retains the baseline's normalization and rounding boundaries. It still performs all 45 comparator stages, including 35 stages with two shuffles each.

Candidate dispatch selects the candidate only when `DS4_METAL_V41_ROUTER_SIMD=1`, with 384 experts, top 6 and one row. Other shapes and settings use the baseline kernel.

## Setup

The one-hour budget was 02:00–03:00 Eastern on 2026-11-04. Baseline revision `3e5f39f8` precedes the candidate `b4f026c0`. Both arms use the candidate binary, with `DS4_METAL_V41_ROUTER_SIMD=0` and `1` respectively. The [driver](run.nu) clears inherited DS4 settings and holds the adopted decode settings constant.

The model-free fixture (`tests/test_metal_router_simd.zig` at candidate revision `b4f026c0`) compares all probability and weight bits and ordered expert IDs for varied scores, numerical branch boundaries, ties, normalization floor and fallback shapes. It passed 21 comparisons. This establishes parity for those inputs, not all possible inputs or full-model behavior.

The profile-free comparison used five alternating pairs at 32K and 8K frontiers, with 128 teacher-forced decode steps per frontier and a 512-token warm append after restoring KV. The append length stays 512 in the longer decode comparison too. A separate profile is diagnostic. Both 128-step timing comparisons are complete. Raw evidence is retained locally under `.workspaces/overnight-decode-20261104/router-simd/`.

## Findings and decision

At each of the 8K, 32K and unaligned 8,193-token frontiers, the full-model fixed-history check produced 256 byte-identical full-vocabulary logit rows and 256 matching hashes, with zero delta-NLL and KL: 768 rows and 768 hashes in total. This does not establish parity on other contexts or sampled tasks. The benchmark and router fixture builds passed.

At 32K, fresh steady decode averaged 33.736 ± 1.015 versus 34.090 ± 0.825 tokens/s (mean ± sample SD), while warm steady decode averaged 33.812 ± 0.688 versus 33.408 ± 0.793 tokens/s. The candidate won three fresh pairs and two warm pairs. Baseline ran first in pairs 1, 3 and 5; candidate ran first in pairs 2 and 4. Fresh decode favored the second arm in every pair, an order effect that limits interpretation.

| Pair | Fresh steady baseline → candidate, tokens/s | Warm steady baseline → candidate, tokens/s | Full fresh 128 steps, baseline → candidate, s | Full warm 128 steps, baseline → candidate, s |
| --- | --- | --- | --- | --- |
| 1 | 34.42 → 34.54 | 34.61 → 32.22 | 4.641 → 4.581 | 3.703 → 3.979 |
| 2 | 34.54 → 34.48 | 34.01 → 33.39 | 4.555 → 4.613 | 3.767 → 3.837 |
| 3 | 32.63 → 34.47 | 33.09 → 34.02 | 4.796 → 4.593 | 3.873 → 3.766 |
| 4 | 34.47 → 32.62 | 34.25 → 33.18 | 4.560 → 4.801 | 3.741 → 3.866 |
| 5 | 32.62 → 34.34 | 33.10 → 34.23 | 4.801 → 4.626 | 3.871 → 3.744 |

Complete fresh decode averaged 4.671 ± 0.122 versus 4.643 ± 0.090 s; warm decode averaged 3.791 ± 0.077 versus 3.838 ± 0.093 s. Steady decode excludes the first step, which averaged 903.2 ± 30.6 versus 915.5 ± 11.9 ms fresh and 33.4 ± 0.6 versus 35.1 ± 2.6 ms warm. The inference phase total averaged 64.412 ± 1.282 versus 65.408 ± 1.127 s and is dominated by variable prefill. Decode durations are approximate, derived from throughput rounded to two decimals. Loading and snapshot save/restore are excluded; the expert cache survives restore and the OS cache is not flushed.

At 8K, fresh steady decode averaged 34.556 ± 0.932 versus 34.430 ± 0.843 tokens/s, and warm steady decode 34.568 ± 0.885 versus 34.296 ± 0.279 tokens/s. The candidate won one fresh and three warm pairs, with the same alternating order as 32K.

| Pair | Fresh steady baseline → candidate, tokens/s | Warm steady baseline → candidate, tokens/s | Full fresh 128 steps, baseline → candidate, s | Full warm 128 steps, baseline → candidate, s |
| --- | --- | --- | --- | --- |
| 1 | 34.99 → 34.14 | 35.47 → 34.16 | 4.502 → 4.593 | 3.613 → 3.750 |
| 2 | 34.05 → 33.91 | 33.98 → 34.37 | 4.604 → 4.608 | 3.770 → 3.760 |
| 3 | 35.23 → 33.47 | 35.43 → 34.05 | 4.438 → 4.594 | 3.617 → 3.764 |
| 4 | 35.35 → 35.28 | 34.50 → 34.75 | 4.437 → 4.429 | 3.716 → 3.687 |
| 5 | 33.16 → 35.35 | 33.46 → 34.15 | 4.706 → 4.471 | 3.829 → 3.754 |

Complete fresh decode averaged 4.538 ± 0.116 versus 4.539 ± 0.083 s, and warm decode 3.709 ± 0.095 versus 3.743 ± 0.032 s. First steps averaged 860.6 ± 20.3 versus 848.8 ± 33.1 ms fresh and 33.1 ± 0.9 versus 39.5 ± 14.4 ms warm; one candidate warm first step took 65.3 ms. Inference phase totals were 34.937 ± 0.612 versus 34.986 ± 0.365 s.

The two workloads show no consistent throughput gain. A longer 32K comparison used five alternating pairs of 512 steps, with the same order and scope. Fresh steady decode averaged 33.680 ± 0.173 versus 33.740 ± 0.373 tokens/s and warm steady decode 31.868 ± 0.604 versus 31.728 ± 0.383 tokens/s.

| Pair | Fresh steady baseline → candidate, tokens/s | Warm steady baseline → candidate, tokens/s | Full fresh 512 steps, baseline → candidate, s | Full warm 512 steps, baseline → candidate, s |
| --- | --- | --- | --- | --- |
| 1 | 33.93 → 33.26 | 32.79 → 32.18 | 15.950 → 16.223 | 15.619 → 15.915 |
| 2 | 33.69 → 34.26 | 32.03 → 32.11 | 16.000 → 15.802 | 15.990 → 15.950 |
| 3 | 33.73 → 33.91 | 31.28 → 31.44 | 16.030 → 15.955 | 16.754 → 16.285 |
| 4 | 33.47 → 33.59 | 31.38 → 31.50 | 16.157 → 16.096 | 16.316 → 16.259 |
| 5 | 33.58 → 33.68 | 31.86 → 31.41 | 16.131 → 16.096 | 16.096 → 16.306 |

Complete fresh decode averaged 16.054 ± 0.088 versus 16.034 ± 0.161 s and warm decode 16.155 ± 0.419 versus 16.143 ± 0.193 s. Baseline pair 3 retained a 13.506 s warm append and a 416.8 ms first warm step; this outlier is included. These results also do not establish a consistent benefit.

## Diagnostic timeline

A separate instrumented 32K pair used 80 command buffers and 932.055 encoders per token in each arm. Summed router time fell from 1.474 to 1.426 ms per token fresh and from 1.473 to 1.428 ms warm, about 0.05 ms per token. Warm uncovered router-to-routed time rose from 5.654 to 5.757 ms and the token envelope from 33.385 to 33.818 ms. This single intrusive pair helps locate work but does not establish a throughput change; the mixed scheduling observations do not demonstrate that the router saving reaches end-to-end decode. Shared gate/up work covered 5,077 of 5,080 router intervals in the audit, and the change does not remove the host event dependency. These observations can explain why a small router saving is difficult to see in decode.

## Reproduction and limits

The runs used an M5 Max with 128 GiB memory and `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, SSD streaming and a requested 64 GB expert cache. The [driver](run.nu) records the common settings, including packed FP4 KV and index, short Q8 query, 4 GiB expert slabs, shared route overlap and down projection, cadence 1 and 8,192-token prefill chunks. All timing, quality and timeline runs passed the pressure guard at level 1 with swap flat at 24.947265625 GiB.

The candidate implementation and fixture are preserved in revision `b4f026c0`; inspect `src/metal/dsv41.metal`, `src/metal/v41_dispatch.zig` and `tests/test_metal_router_simd.zig` at that revision. To reproduce, check out that revision in a separate workspace, build the benchmark, and copy the current `run.nu` into that workspace for its `--tokens` support. Then run from the workspace root:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/overnight-decode-20261104/router-simd/timing32k-long-guard -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/overnight-decode-20261104/router-simd/timing32k-long timing --flag DS4_METAL_V41_ROUTER_SIMD --control 0 --candidate 1 --tokens 512
```

Fixed-history equality does not demonstrate practical task quality, which was not measured. The candidate remains available in its revision for further investigation, but is not adopted based on these results.
