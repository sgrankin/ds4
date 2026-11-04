# Short Q8 query projection

The first specialized kernel produced identical logits in the measured 8K
teacher-forced workload, but its five 32K timing pairs did not establish a
whole-workload speedup. A second version also produced identical logits in that
workload. Its 8K timing was neutral and its 32K timing was mixed. The resident
fast presets now enable the specialized kernel; the size of any benefit remains
uncertain.

## What changed

During decode, the query projection expands a 1,280-element input into 32,768
values for the attention heads. The Q8 weight matrix supplies one output row
per value. The projection rounds to BF16 and applies rotary position embedding
(RoPE) to the last 64 values of each 512-value head, so the attention query
encodes its position. The baseline combines the generic Q8 matrix-vector
projection and forward RoPE in one kernel.

V1, measured at revision `1550221e`, assigned one output row to each SIMD group
and used a threadgroup barrier to bring adjacent values together for RoPE. V2,
revision `5f34699b`, assigns adjacent rows to one SIMD group, reuses input
loads for both rows, and needs no barrier for the pair. It retains the generic
four-slot reduction and BF16 rounding boundaries. The [kernel](../../../src/metal/dsv41.metal)
and [dispatch](../../../src/metal/v41_dispatch.zig) show the implementation:
the specialized path is selected with `DS4_METAL_V41_SHORT_Q8_QUERY=1`, limited
to this shape and single-rank attention, and yields to an explicit Q8 NSG
override. The resident fast presets set this flag by default, while an explicit
`0` disables it; ordinary runtime and `context-fast` defaults are unchanged. The [focused test](../../../tests/test_deepseek41_metal.c) compares
the two kernels bitwise at four positions in both frequency modes and checks
dispatch and the override; run it with `--short-query`.
The focused test passed for V2.

## V1 measurements

On an Apple M5 Max with 128 GiB of memory, the [runner](run.nu) uses
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, the Promessi Sposi
prompt, a 64 GB expert cache, 4 GiB expert slabs, an 8,192-token prefill
chunk, and 128 teacher-forced decode tokens. Resident routing has a five
substitution cap; streaming RoPE tables and route/shared overlap are enabled.
Each profile-free sample starts with a fresh 32,768-token prefill and decode,
restores the KV snapshot, appends 512 tokens and decodes again. The expert
cache survives the restore and the OS cache was not flushed. Phase times exclude
model loading and snapshot save/restore. Seconds below are derived from CSV
throughput rounded to two decimal places; total inference includes both prefill
and decode phases. Baseline and V1 order alternated, starting with baseline.

| Pair | First | Fresh decode, baseline / V1 (s) | Restored decode, baseline / V1 (s) | Total inference, baseline / V1 (s) |
|---|---|---:|---:|---:|
| 1 | Baseline | 4.927 / 4.900 | 4.028 / 4.075 | 62.570 / 62.010 |
| 2 | V1 | 4.977 / 4.950 | 4.120 / 4.073 | 63.851 / 63.574 |
| 3 | Baseline | 5.020 / 5.098 | 4.121 / 4.133 | 63.960 / 64.363 |
| 4 | V1 | 5.041 / 4.988 | 4.178 / 4.132 | 65.291 / 64.837 |
| 5 | Baseline | 4.942 / 4.902 | 4.091 / 4.099 | 64.619 / 65.038 |

Mean total inference was 64.058 ± 1.013 s for baseline and 63.964 ± 1.229 s
for V1 (sample standard deviation); the
paired changes ranged from -0.89% to +0.65%. Mean restored decode was 4.107 s
and 4.102 s respectively, with paired changes from -1.15% to +1.18% and an
order effect: V1 was faster in both pairs where it ran first and slower in all
three where baseline ran first. These differences are small relative to the
observed variation. A separate instrumented timeline summed query-projection
encoder durations across layers to 3.93 ms per restored decode token for the
baseline and 3.38 ms for V1. Encoders can overlap, and profiling can alter
timing, so these sums are neither standalone operation latency nor a measured
critical-path gain. This single profiled observation is diagnostic rather than
a final latency comparison.

At the 8,192 and 8,704 frontiers, all 256 full-vocabulary teacher-forced logit rows
were byte-identical, with zero delta NLL, zero KL and no top-1 changes. All 256
post-evaluation hashes also matched. This establishes parity for those sampled
histories, not all possible inputs. Raw timing samples and summary, quality
results, and the diagnostic timeline are retained locally under
`.workspaces/short-q8-query-20261103/` in `timing32k`, `quality8k`, and
`timeline-v1` respectively.

## V2 measurements

The runtime baseline is `ab947bef` (documentation through `05e78fdf`). Both
comparison arms use the same binary built at `5f34699b`, with the flag set to
zero or one to isolate the kernel. The 8K and 32K quality runs together
produced 512 byte-identical full-vocabulary logit rows and 512 matching
post-evaluation hashes, with zero delta NLL and KL. Their evidence is in the
local `paired-quality8k/quality-summary.json` and
`paired-quality32k/quality-summary.json`.
No fresh task or autonomous-generation quality test was run in this round;
quality evidence consists of focused kernel tests and full-model teacher-forced
parity.

Five profile-free 32K pairs used the same workload and phase boundaries as V1.
Steady decode excludes the first token only. The table shows baseline / V2
throughput and total inference time; each pair retains its actual run order.

| Pair | First | Fresh steady decode (tokens/s) | Restored steady decode (tokens/s) | Total inference (s) |
|---|---|---:|---:|---:|
| 1 | Baseline | 29.47 / 31.69 | 30.53 / 31.86 | 70.256 / 64.605 |
| 2 | V2 | 31.29 / 30.51 | 31.22 / 30.80 | 65.050 / 65.257 |
| 3 | Baseline | 31.52 / 31.48 | 29.27 / 31.03 | 65.271 / 65.462 |
| 4 | V2 | 31.45 / 31.57 | 30.51 / 30.89 | 66.084 / 65.921 |
| 5 | Baseline | 31.59 / 31.63 | 30.45 / 30.82 | 66.453 / 66.130 |

Fresh steady throughput averaged 31.064 ± 0.898 versus 31.376 ± 0.490
tokens/s, and restored steady throughput 30.396 ± 0.704 versus 31.080 ±
0.445 tokens/s (mean ± sample standard deviation). V2 was faster in three of
five fresh pairs and four of five restored pairs; the exception for restored
decode was a V2-first pair. Mean total inference was 66.623 ± 2.111 versus
65.475 ± 0.599 s, with V2 lower in three pairs and higher in two. The first
baseline append took 15.35 s, strongly influencing that total; these results
do not establish a robust whole-workload gain. All ten guards completed with
code 0, reported pressure level 1, and swap remained
flat at 24.94421875 GiB. Raw samples and summary are in the local
`paired-timing32k` directory.

The same five-pair comparison at an 8,192-token frontier gave the following
results. As above, throughput excludes only the first decode token and total
inference includes both prefill and decode phases.

| Pair | First | Fresh steady decode (tokens/s) | Restored steady decode (tokens/s) | Total inference (s) |
|---|---|---:|---:|---:|
| 1 | Baseline | 31.27 / 31.28 | 31.72 / 31.37 | 36.315 / 36.116 |
| 2 | V2 | 32.07 / 32.21 | 31.79 / 32.10 | 35.369 / 35.007 |
| 3 | Baseline | 32.13 / 32.23 | 31.41 / 31.48 | 35.810 / 35.776 |
| 4 | V2 | 32.13 / 32.20 | 31.00 / 31.23 | 36.162 / 36.120 |
| 5 | Baseline | 32.14 / 32.06 | 31.04 / 31.19 | 36.223 / 36.200 |

Fresh steady throughput averaged 31.948 ± 0.380 versus 31.996 ± 0.406
tokens/s and restored throughput 31.392 ± 0.369 versus 31.474 ± 0.368
tokens/s. Total inference averaged 35.9758 ± 0.3896 versus 35.8437 ± 0.4955 s.
These small differences are neutral within the observed variation. Prefill
changed by about 1% even though the candidate affects decode, reinforcing the
need for caution in attributing the total difference. Samples and summary are
in the local `paired-timing8k` directory.

The full 128-token decode windows include the first token omitted from steady
throughput. These phase times are mean ± sample standard deviation in seconds;
prefill and decode times are derived from rounded CSV throughput.

| Frontier | Phase | Baseline (s) | V2 (s) |
|---|---|---:|---:|
| 8K | Fresh prefill | 20.045 ± 0.380 | 19.788 ± 0.220 |
| 8K | Fresh full decode | 4.800 ± 0.048 | 4.802 ± 0.065 |
| 8K | Restored prefill | 7.049 ± 0.259 | 7.168 ± 0.302 |
| 8K | Restored full decode | 4.082 ± 0.048 | 4.087 ± 0.062 |
| 32K | Fresh prefill | 48.388 ± 1.845 | 48.931 ± 0.596 |
| 32K | Fresh full decode | 4.964 ± 0.143 | 4.935 ± 0.058 |
| 32K | Restored prefill | 8.985 ± 3.563 | 7.474 ± 0.299 |
| 32K | Restored full decode | 4.286 ± 0.166 | 4.136 ± 0.064 |

At 8K, both full decode means were slightly slower for V2 within the observed
variation, despite higher steady throughput. At 32K, the first restored decode
token in pair 1 took 338.730 ms for baseline versus 35.035 ms for V2; it
materially influences the full-window mean.

One intrusive paired timeline confirmed that the new kernel replaced the
generic query kernel. Summed query encoder durations across layers per token
were 3.930 / 3.337 ms fresh and 3.928 / 3.345 ms restored, baseline / V2.
Encoder union was 27.784 / 27.693 ms fresh and 27.758 / 27.693 ms restored;
the corresponding envelopes were 35.729 / 35.614 ms and 35.589 / 35.518 ms.
Both arms used 61 command buffers and about 967.76 encoders per token. Some
other encoder durations increased, and overlap and profiling overhead prevent
assigning the difference in query sums directly to a causal latency saving.
The paired-row refinement was not clearly faster than V1's separate profile
observation of approximately 3.33–3.38 ms. The local `timeline-paired` files
retain this diagnostic; it is not a comparative performance measurement.

We initially left the small specialized kernel outside presets because its
benefit was unresolved. On reassessment, we adopted it in `resident-fast` and
`persistent-fast` (`persistent-packed` is an alias): exact tested logits, no
meaningful measured regression and a small implementation support a low-risk
engineering choice despite the uncertain gain. This decision uses the same
evidence, not a new benchmark. The `ds4-agent` and `ds4-server` builds passed.

## Reproduction

Run from the repository root with the intended binary built and no competing
heavy model process. Choose fresh output paths; the guard serializes jobs and
monitors pressure. The runner records the exact environment and arguments for
each arm, clears inherited DS4 settings, and alternates the timing order.

```nu
./tools/zig-build ds4-bench metal-test -j4
python3 speed-bench/ideas/pressure_run.py .workspaces/short-q8-query-repro-32k-guard -- nu speed-bench/experiments/2026-11-03-short-q8-query/run.nu .workspaces/short-q8-query-repro-32k timing --frontier 32768
python3 speed-bench/ideas/pressure_run.py .workspaces/short-q8-query-repro-8k-guard -- nu speed-bench/experiments/2026-11-03-short-q8-query/run.nu .workspaces/short-q8-query-repro-8k timing --frontier 8192
python3 speed-bench/ideas/pressure_run.py .workspaces/short-q8-query-repro-quality8k-guard -- nu speed-bench/experiments/2026-11-03-short-q8-query/run.nu .workspaces/short-q8-query-repro-quality8k quality --frontier 8192
python3 speed-bench/ideas/pressure_run.py .workspaces/short-q8-query-repro-quality32k-guard -- nu speed-bench/experiments/2026-11-03-short-q8-query/run.nu .workspaces/short-q8-query-repro-quality32k quality --frontier 32768
python3 speed-bench/ideas/pressure_run.py .workspaces/short-q8-query-repro-timeline-guard -- nu speed-bench/experiments/2026-11-03-short-q8-query/run.nu .workspaces/short-q8-query-repro-timeline timeline --frontier 32768
```
