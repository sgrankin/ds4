# Row blocking for indexed prefill attention

Row blocking increased fresh-prefill time in every pair at both 8K and 32K,
by 2.71% and 7.94% between means respectively. The candidate is rejected and
its experimental prefill selector has been removed; presets do not change. The
existing decode use of the row-blocked kernel remains unchanged.

## Context and mechanism

The published baseline `364ceeab` includes the five overnight decode
optimizations. A separate 32K diagnostic at runtime revision `689c2d51`, with
batch shared overlap enabled, attributed about 10.67 seconds to `heads16_dual`
indexed attention. This workload forces resident execution during fresh prefill,
so it does not enter the overlap branch; its fresh-prefill path is the same as
the published baseline. Most of that attention time comprised 72 dispatches
of `8192 × 4 × 1`, each about 117–167 ms. Profiling locates the work but is
not a profile-free latency comparison. Its local evidence is
`.workspaces/hillclimb-20261104/01-prefill-overlap/reuse-profile/1-candidate/decode-anatomy.json`.

The baseline kernel stages one key/value row in threadgroup memory and uses it
for 16 heads, with barriers around each row. Candidate revision `a2198630`
reuses the existing `heads8_rb16` kernel for eligible multirow prefill. It
stages up to 16 rows at a time and consumes them sequentially, reducing barriers
per full block by a factor of 16. This increases threadgroup memory from 1 KiB
to 16 KiB and reduces heads per group from 16 to 8: the candidate loads the rows twice
across head groups but carries less per-group head state. Twice as many
cooperative loads do not establish twice the physical memory traffic; hardware
caching was not measured. Both paths use the
same row attention calculation and half-precision staging boundaries; sorting
remains in place to preserve chronological order. The candidate adds no Metal
shader or scratch buffer. See the [dispatch](../../../src/metal/attention_batch.zig)
and [kernels](../../../src/metal/dsv4_misc.metal).

At measured revision `a2198630`, `DS4_METAL_V41_PREFILL_ATTN_RB16=1` selects the candidate only for multirow,
non-quality attention with Metal Performance Primitives available, 32 or 64
heads, width 512, top-k 512 and window 128. Setting it to `0` keeps the
baseline path. Decode splitting is unchanged. The row-blocked kernel previously
served decode; this is a prefill trial, not evidence of an established prefill
speedup.

## Setup and measurements

Both arms use `a2198630` on an M5 Max with 128 GiB memory, with the flag
explicitly set to `0` or `1` and all other settings held constant. The existing
[driver](../2026-11-04-decode-pass/run.nu) clears inherited `DS4` settings and
records its effective settings and arguments. Its timing stage alternates arm
order over five pairs with profiling disabled. It uses the DeepSeek V4.1 Flash
1017 IQ2_XXS/Q2_K imatrix checkpoint and SSD streaming. The requested 64 GB
expert cache provides 56.88 GiB dynamic capacity plus a 7.12 GiB reserve. The
prefill chunk is 8192 tokens and the context allocation is 40960 tokens. The
workload uses 128 teacher-forced tokens before and after a 512-token append.
The OS file cache was not flushed; the process-local expert cache starts empty
and survives KV restore. Raw artifacts are kept locally under
`.workspaces/hillclimb-20261104/02-prefill-attention`.

The candidate passed the bounded Metal attention fixture under the pressure
guard. Its command was:

```sh
python3 speed-bench/ideas/pressure_run.py OUT -- env DS4_METAL_V41_PREFILL_ATTN_RB16=1 ./zig-out/bin/test-deepseek41-metal --tp-attention
```

Here `OUT` is the local `fixture-guard` directory alongside `fixture.log`.
The fixture's incidental timing lines are not a performance comparison.

At 8K, the fixed-history quality check found all 256 full-vocabulary rows
byte-identical between arms, as well as all 256 post-evaluation hashes. Both
delta-NLL and KL were zero. The local evidence is
`quality8k/quality-summary.json` under the artifact directory. This establishes
agreement for those tested histories, not general equivalence. A broader 32K
quality check was skipped because the candidate was rejected for performance,
not because a numerical mismatch was observed.

## 32K timing

Five profile-free pairs alternated order. The pressure guard completed at level
1 with stable swap and no interruptions. Phase times exclude loading and
snapshot save and restore; the inference total sums fresh prefill, decode,
append and the second decode. Seconds are derived from throughput rounded to
two decimal places. Means below use sample standard deviation.

| Measure | Baseline | Candidate |
| --- | ---: | ---: |
| Fresh prefill (s) | 49.336 ± 2.034 | 53.251 ± 0.475 |
| Inference total (s) | 65.039 ± 2.232 | 69.088 ± 0.455 |
| 512-token append (s) | 7.309 ± 0.184 | 7.414 ± 0.133 |
| Fresh steady decode (tokens/s) | 34.522 ± 0.417 | 34.678 ± 0.090 |
| Warm steady decode (tokens/s) | 33.646 ± 0.674 | 33.180 ± 0.232 |

Steady decode excludes the first token. The inference total rose 6.23% between
means. Decode timing differences do not establish a change to the decode path.
The first baseline run was particularly fast, but the candidate was slower in
every fresh-prefill pair regardless of order. All samples follow, with each
cell showing baseline / candidate; the first arm is indicated separately.

| Pair | First | Fresh prefill (s) | Inference total (s) | Append (s) | Fresh steady (tokens/s) | Warm steady (tokens/s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Baseline | 45.861 / 52.415 | 61.232 / 68.323 | 7.006 / 7.457 | 33.78 / 34.74 | 34.85 / 32.81 |
| 2 | Candidate | 49.808 / 53.610 | 65.514 / 69.398 | 7.299 / 7.395 | 34.65 / 34.75 | 33.38 / 33.09 |
| 3 | Baseline | 49.864 / 53.389 | 65.649 / 69.137 | 7.349 / 7.345 | 34.72 / 34.74 | 33.30 / 33.32 |
| 4 | Candidate | 49.896 / 53.403 | 65.647 / 69.119 | 7.392 / 7.258 | 34.77 / 34.58 | 33.33 / 33.33 |
| 5 | Baseline | 51.250 / 53.439 | 67.152 / 69.466 | 7.497 / 7.613 | 34.69 / 34.58 | 33.37 / 33.35 |

The local `timing32k/timing-samples.csv` and `timing-summary.json` retain
unrounded calculated values and additional phase and wall measurements.

## 8K timing

The same alternating, profile-free procedure gives fresh prefill of
20.161 ± 0.172 seconds for the baseline and 20.707 ± 0.327 seconds for the
candidate. Inference totals were 35.469 ± 0.406 and 36.055 ± 0.675 seconds,
respectively, an increase of 1.65% between means. Append times were
7.126 ± 0.251 and 7.195 ± 0.305 seconds. Each pair below is baseline /
candidate, and every fresh-prefill pair favors the baseline.

| Pair | First | Fresh prefill (s) | Inference total (s) | Append (s) | Fresh steady (tokens/s) | Warm steady (tokens/s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Baseline | 20.203 / 20.314 | 35.107 / 35.295 | 6.743 / 6.862 | 35.49 / 35.33 | 34.47 / 34.72 |
| 2 | Candidate | 19.887 / 20.401 | 35.011 / 35.343 | 7.074 / 6.868 | 35.63 / 35.39 | 34.22 / 34.78 |
| 3 | Baseline | 20.168 / 20.858 | 35.492 / 36.460 | 7.164 / 7.378 | 35.35 / 35.28 | 34.25 / 33.90 |
| 4 | Candidate | 20.364 / 21.034 | 35.848 / 36.566 | 7.219 / 7.370 | 35.41 / 35.50 | 33.41 / 33.95 |
| 5 | Baseline | 20.182 / 20.929 | 35.887 / 36.614 | 7.431 / 7.497 | 35.35 / 35.62 | 33.88 / 33.88 |

Unrounded calculated values and other phase measurements remain in the local
`timing8k` artifacts. Reproduction requires building revision `a2198630` and running the
linked driver under the pressure guard with `--flag
DS4_METAL_V41_PREFILL_ATTN_RB16`, selecting the desired frontier and timing or
quality stage. The selector was experimental and was removed after this
measurement; it is not an option in the resulting runtime.

## Diagnostic profile

The separate candidate profile suggests that attention accounts for the added cost. Fresh
prefill used 284 command buffers and 101542 encoders in both profiles. The
candidate's envelope was 52.7362 seconds, with 42.3287 seconds of encoder union
and 10.4075 seconds uncovered. Its row-blocked attention kernel summed to
14.5473 seconds; 72 large dispatches of `8192 × 8 × 1` accounted for
14.0245 seconds, at 159.973–244.336 ms each.

The earlier `689c2d51` diagnostic had a 48.8262-second envelope, 38.0983-second
encoder union and 10.7278 seconds uncovered; its dual-head kernel summed to
10.6686 seconds. These are single profiled observations, not a five-sample
kernel comparison. They support attention as the source of the extra cost but do not establish
whether occupancy, loads or another mechanism caused it. The candidate evidence
is local at `candidate-profile/1-candidate/decode-anatomy.json` under the
artifact directory.
