# Reusing RoPE coefficients in the short query

The existing per-token rotary position encoding (RoPE) coefficients are reused
in the short Q8 query in `resident-fast` and `persistent-fast` (also named
`persistent-packed`). The five-pair timings are neutral to slightly favorable;
the diagnostic profile did not show an exposed-query reduction. Adoption rests
on reuse without an allocation or extra dispatch and exact output in the tested
cases, not a demonstrated kernel or throughput gain. Set
`DS4_METAL_V41_SHORT_Q8_QUERY_TABLE=0` to restore inline coefficients. Ordinary
and `context-fast` defaults are unchanged.

## Context and candidate

The [short Q8 query experiment](../2026-11-03-short-q8-query/README.md)
specialized the query projection while preserving its measured numerical
behavior. A diagnostic timeline from the subsequent
[router experiment](router-simd.md) showed roughly 0.9 ms of query work per
token extending beyond compressed KV staging. That exposed interval is not a
prediction of the saving available from this change.

The candidate uses the graph-owned table of 64 `float2` cosine and sine pairs
already produced for each eligible decode token. The first 32 entries hold raw
frequencies and the remaining 32 hold compressed frequencies. The table stores
positive-angle coefficients; consumers choose the forward or inverse rotation
formula. The candidate reads those coefficients in the existing short Q8
query path instead of recalculating them for each query head, preserving the
projection, reduction, BF16 boundaries and rotation FMA order. It introduces no
table allocation. `DS4_METAL_V41_SHORT_Q8_QUERY_TABLE=1` selects the experiment
only when the table is valid for the current scalar position; other cases retain
their ordinary dispatch and fallbacks. The [dispatch](../../../src/metal/v41_dispatch.zig)
selects the table kernel only for the existing eligible short-query shape and
uses the raw or compressed half of the table. Both kernels share the same
[projection and rotation body](../../../src/metal/dsv41.metal).

The graph owns one reusable coefficient buffer across streaming command
boundaries. The scalar executor fills it for the current position before running
the layers and clears its validity at the step boundary. It is not a table of
all context positions.

## Setup

The selected loop runs from 02:52 to 03:52 Eastern on 2026-11-04. Baseline
`a1900948` includes the adopted F16 gather, flush cadence 1 and early shared
down projection; the router SIMD candidate was not adopted. The candidate is
`a316c220`. Both arms use the candidate binary with the table flag set to 0 or 1.

The comparison uses the M5 Max with 128 GiB memory and
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf` with SSD streaming. The
[driver](run.nu) clears inherited DS4 settings and holds common settings constant:
packed FP4 main and index KV, short Q8 query, streaming RoPE table, shared route
overlap and down projection, cadence 1, route cap 5 and gap 0.90, 4 GiB expert
slabs, a requested 64 GB expert cache and 8,192-token prefill chunks. Five
alternating profile-free pairs at 32K and 8K measure decode; a separate
instrumented profile investigates whether any kernel change reaches the
exposed query tail. A focused Zig fixture checks raw and compressed frequencies,
position changes and scalar tails; fixed-history full-model checks compare logits.

Raw evidence is retained under `.workspaces/overnight-decode-20261104/query-rope-table/`.
All guarded runs reported pressure level 1. Swap rose from 24.947265625 GiB to
24.950986328125 GiB during 32K timing and to 24.95556640625 GiB during 8K
timing, then remained flat during later quality and timeline runs: about 8.5 MiB
total drift. Benchmark, fixture, agent and server builds passed, as did launcher
tests for preset defaults and explicit overrides.

## Findings and decision

The guarded [fixture](../../../tests/test_metal_query_rope_table.zig) passed 18 exact
comparisons of 32,768 outputs each, covering both frequency modes at positions
0, 1, 7, 8,193, 32,768 and 1,048,575, then 8,193, 1 and 0 again. It also
checked kernel dispatch counts and the forward rotation sign. This establishes
parity for those fixture inputs, not all possible inputs. At each of 8K, 32K
and 8,193 tokens, fixed-history full-model quality produced 256 byte-identical
full-vocabulary logit rows and 256 matching hashes: 768 of each in total, with
zero delta-NLL and KL. Practical task quality was not measured.

At 32K, fresh steady decode averaged 33.394 ± 0.717 versus 33.832 ± 0.846
tokens/s, and warm steady decode 33.430 ± 0.380 versus 33.394 ± 0.473
tokens/s (mean ± sample SD). The candidate was faster in four fresh pairs and
one warm pair. Baseline ran first in pairs 1, 3 and 5; the candidate ran first
in pairs 2 and 4. The fresh advantage was larger in baseline-first pairs, so
order may contribute to the difference; the warm pairs do not show the same
pattern.

| Pair | Fresh steady baseline → candidate, tokens/s | Warm steady baseline → candidate, tokens/s | Full fresh 128 steps, baseline → candidate, s | Full warm 128 steps, baseline → candidate, s |
| --- | --- | --- | --- | --- |
| 1 | 34.27 → 34.60 | 34.03 → 32.99 | 4.697 → 4.570 | 3.765 → 3.883 |
| 2 | 34.05 → 34.27 | 33.38 → 33.37 | 4.682 → 4.639 | 3.839 → 3.840 |
| 3 | 32.68 → 34.44 | 32.98 → 34.19 | 4.758 → 4.537 | 3.885 → 3.749 |
| 4 | 32.90 → 32.74 | 33.32 → 33.09 | 4.739 → 4.765 | 3.846 → 3.872 |
| 5 | 33.07 → 33.11 | 33.44 → 33.33 | 4.769 → 4.765 | 3.832 → 3.846 |

Full fresh decode averaged 4.729 ± 0.038 versus 4.655 ± 0.107 s; full warm
decode averaged 3.833 ± 0.043 versus 3.838 ± 0.053 s. Steady decode excludes
the first step, which averaged 924.4 ± 50.1 versus 900.3 ± 34.3 ms fresh and
34.1 ± 1.0 versus 34.5 ± 0.7 ms warm. Inference phase totals averaged
64.706 ± 1.653 versus 64.771 ± 1.024 s, dominated by prefill. The runs include
fresh prefill and decode, then a 512-token warm append and warm decode. KV restore
occurs between those phases, but its duration, snapshot saving and model loading
are excluded from inference totals. The expert cache survives restore and the
OS cache was not flushed. Durations are derived from throughput
rounded to two decimals.

At 8K, fresh steady decode averaged 35.216 ± 0.090 versus 35.128 ± 0.133
tokens/s; warm steady decode averaged 33.976 ± 1.011 versus 34.286 ± 0.862
tokens/s. The candidate was slower in all five fresh pairs and faster in two
warm pairs. Run order alternated as at 32K; the larger warm differences appear
in both directions among baseline-first pairs.

| Pair | Fresh steady baseline → candidate, tokens/s | Warm steady baseline → candidate, tokens/s | Full fresh 128 steps, baseline → candidate, s | Full warm 128 steps, baseline → candidate, s |
| --- | --- | --- | --- | --- |
| 1 | 35.16 → 35.10 | 35.53 → 34.98 | 4.488 → 4.501 | 3.607 → 3.664 |
| 2 | 35.17 → 35.00 | 33.08 → 35.37 | 4.460 → 4.494 | 3.873 → 3.623 |
| 3 | 35.34 → 35.32 | 33.06 → 33.28 | 4.466 → 4.443 | 3.875 → 3.852 |
| 4 | 35.13 → 35.02 | 34.07 → 33.86 | 4.488 → 4.469 | 3.760 → 3.787 |
| 5 | 35.28 → 35.20 | 34.14 → 33.94 | 4.446 → 4.446 | 3.753 → 3.783 |

Full fresh decode averaged 4.470 ± 0.018 versus 4.471 ± 0.027 s and full warm
decode 3.774 ± 0.110 versus 3.742 ± 0.095 s. First steps averaged 863.3 ± 14.4
versus 854.9 ± 18.3 ms fresh and 33.0 ± 0.8 versus 35.6 ± 3.3 ms warm.
Inference phase totals averaged 35.017 ± 0.911 versus 35.213 ± 0.820 s. The
combined results show small mixed differences rather than a major gain.

## Diagnostic profile

The separate instrumented 32K pair recorded 40 query calls per token, 80 command
buffers and 932.055 encoders per token in each arm. Summed query time was
3.321794 → 3.320301 ms/token fresh and 3.314718 → 3.319888 ms/token warm.
The query tail beyond compressed KV staging was 0.855323 → 0.868829 ms/token
fresh and 0.851241 → 0.885732 ms/token warm. This single intrusive profile
shows no reduction in that exposed interval; it is not a throughput comparison.

## Reproduction

Build the benchmark and fixture at `a316c220`; run one guarded job at a time:

```sh
./tools/zig-build ds4-bench query-rope-table-test
python3 speed-bench/ideas/pressure_run.py .workspaces/overnight-decode-20261104/query-rope-table/repro-fixture-guard -- ./zig-out/bin/test-metal-query-rope-table
```

The 32K timing comparison is:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/overnight-decode-20261104/query-rope-table/repro32k-guard -- nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/overnight-decode-20261104/query-rope-table/repro32k timing --flag DS4_METAL_V41_SHORT_Q8_QUERY_TABLE --control 0 --candidate 1 --frontier 32768
```

Use `--frontier 8192` for the 8K comparison. Substitute `quality` or `timeline`
for `timing` for the separate diagnostics; use frontier 8193 for the unaligned
quality check. The local raw artifacts retain the complete outputs and settings.
