# Cooperative packed FP4 attention reader

**The cooperative reader was not retained.** The 8K comparison has a small
favorable mean, while 32K and a focused operation comparison show no clear
benefit. That is insufficient payoff for the added code and reader metadata;
the measurements do not establish a regression.

This experiment asked whether sharing the scale decode and constructing half-precision values directly can reduce the cost
of reading packed FP4 rows in indexed attention. It retained the existing head
groups, row staging and key order.

## Context and mechanism

Indexed attention reads selected compressed key/value rows and stages them as
half-precision vectors for attention. A packed row holds 512 E2M1 four-bit
values in 256 bytes, followed by 32 E4M3 scale bytes. Each scale applies to
16 values, or four consecutive `half4` vectors; the complete row occupies
288 bytes. The original reader decodes each vector's four nibbles and its
scale separately, so neighboring vectors repeat the scale decode.

The tested opt-in cooperative reader constructed the exact E2M1 half bits
and shared one decoded scale across four adjacent SIMD lanes, each responsible for one
consecutive `half4`. This required those four participating lanes to address
the same row and the same 16-value scale group; the shuffle had to execute with
the source lane active. The scaled vectors fed the existing attention
calculation. Internal packet mode 3 selected this reader while retaining the
same packed storage. It added no buffers or dispatches. The runtime changes
were reverted to the parent; the original reader remains. The tested source was in
[`dsv4_kv.metal`](../../../src/metal/dsv4_kv.metal),
[`dsv4_misc.metal`](../../../src/metal/dsv4_misc.metal), and
[`attention_batch.zig`](../../../src/metal/attention_batch.zig).

The [earlier packed-storage profile](../main-fp4-20261028/RESULTS_32K.md#matching-diagnostic-attribution)
found extra time in indexed attention compared with expanded storage and
motivated this reader experiment. That was a different comparison and does not
establish a speedup for this change. The immediately preceding
[row-blocked prefill experiment](../2026-11-04-prefill-row-blocking/README.md)
regressed at both tested frontiers and was removed; this experiment keeps the
original staging geometry.

## Setup

The selected experiment window is November 4, 2026, 10:00–11:00 Eastern.
The parent baseline is `f965f844` and the prototype revision is `d2957118`.
Both measured arms used the same new binary with
`DS4_METAL_V41_COOPERATIVE_FP4` explicitly set to `0` or `1`. ReleaseFast
builds of `ds4-bench` and `metal-test` passed. The runner and fixture commands
below describe the historical prototype; the cooperative flag no longer
selects a reader in the reverted runtime.

The workload uses an M5 Max with 128 GiB memory and the DeepSeek V4.1 Flash
1017 IQ2_XXS/Q2_K imatrix checkpoint. It requests a 64 GB expert cache, with
an 8192-token prefill chunk and 40960-token context allocation. At frontiers
of 8192 and 32768 tokens, it runs 128 teacher-forced tokens, appends 512 tokens,
and runs another 128 teacher-forced tokens. The OS file cache is not flushed;
the process-local expert cache survives KV restore. The existing
[runner](../2026-11-04-decode-pass/run.nu) took five alternating timing
pairs at each frontier, with fixed-history full-vocabulary comparisons. Raw
output is local and ignored under
`.workspaces/hillclimb-20261104/03-packed-attention-reader`.

A host enumeration matched 4096 scale-byte and nibble combinations and all
128 lane mappings. The local script and output are `host_reader_equivalence.py`
and `host_reader_equivalence.log` under the artifact directory; this does not
establish GPU equivalence.

The GPU fixture passed under the pressure guard:

```sh
python3 speed-bench/ideas/pressure_run.py OUT -- ./zig-out/bin/test-deepseek41-metal --fp4-attention
```

Here `OUT` is the local `fixture-guard` directory, with stdout and stderr in
`fixture.log`. The fixture compares both packed readers with independently
expanded rows across five shapes, legal scale bytes 0 through 126, signed
nibbles, a wrapped raw ring and masked IDs. All five shapes produced
byte-identical outputs across expanded, original packed and cooperative packed
readers. The local evidence is `fixture.log` and `fixture-guard/result.json`.
This establishes agreement for the tested inputs. An independent runtime review
found no concrete issue with lane alignment, leader participation, signed zero,
stride, storage or lifetimes.

At 8K, the fixed-history quality check found all 256 full-vocabulary rows
byte-identical and all 256 post-evaluation hashes matching. Delta-NLL and KL
were zero. Local evidence is `quality8k/quality-summary.json`. At 32K, another
256 full-vocabulary rows and 256 post-evaluation hashes also matched exactly, with zero delta-NLL and KL; see
`quality32k/quality-summary.json`. Across both frontiers this is 512 rows and
512 hashes. This checks the selected histories, not all possible inputs.

## 32K timing

Five profile-free pairs show no clear benefit at 32K. Fresh prefill averaged
49.030 ± 1.839 seconds with the original reader and 49.292 ± 1.511 seconds
with the cooperative reader; inference totals averaged 64.710 ± 2.064 and
65.047 ± 1.676 seconds. The differences between means, +0.53% and +0.52%, are
small relative to the observed variation and do not establish a regression.
Append averaged 7.370 ± 0.190 and 7.405 ± 0.155 seconds. Variation is sample
standard deviation.

The phase times exclude loading and snapshot save and restore. Inference total
sums fresh prefill, decode, append and the second decode. Seconds are derived
from throughput rounded to two decimal places. Steady decode excludes the
first token. Each cell below is original / cooperative; the first arm alternates.

| Pair | First | Fresh prefill (s) | Inference total (s) | Append (s) | Fresh steady (tokens/s) | Warm steady (tokens/s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Original | 45.764 / 46.620 | 61.039 / 62.090 | 7.043 / 7.136 | 34.82 / 34.78 | 34.85 / 33.81 |
| 2 | Cooperative | 50.011 / 50.209 | 65.749 / 66.117 | 7.392 / 7.519 | 34.73 / 34.82 | 33.53 / 32.93 |
| 3 | Original | 49.510 / 49.636 | 65.265 / 65.376 | 7.410 / 7.416 | 34.76 / 34.68 | 33.64 / 33.48 |
| 4 | Cooperative | 49.808 / 50.128 | 65.629 / 65.938 | 7.511 / 7.466 | 34.82 / 34.74 | 33.51 / 33.44 |
| 5 | Original | 50.058 / 49.867 | 65.868 / 65.712 | 7.494 / 7.490 | 34.75 / 34.75 | 33.39 / 33.30 |

The first pair was faster than the remaining pairs for both arms. The local
`timing32k/timing-samples.csv` and `timing-summary.json` retain unrounded
calculations and other phases. The diagnostic profile below is a separate observation. Source analysis suggests possible explanations for a neutral
result: each 32-lane group still addresses eight unique scale bytes, leader
masking might leave issued decode instructions unchanged while adding a shuffle,
and the compiler's lowering of magnitude lookup versus bit construction is
unknown. These are hypotheses; no instruction or hardware-counter evidence
establishes the cause.


## 8K timing

Fresh prefill averaged 20.143 ± 0.148 seconds with the original reader and
19.994 ± 0.139 seconds with the cooperative reader, a 0.74% decrease between
means. Inference totals averaged 35.488 ± 0.240 and 35.357 ± 0.410 seconds,
a 0.37% decrease; append averaged 7.198 ± 0.255 and 7.194 ± 0.219 seconds.
The candidate was faster for fresh prefill in four of five pairs; the size and
consistency of any benefit remain uncertain. These use the same timing scope
and sample standard deviation as the 32K comparison.

| Pair | First | Fresh prefill (s) | Inference total (s) | Append (s) | Fresh steady (tokens/s) | Warm steady (tokens/s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Original | 20.366 / 19.774 | 35.332 / 34.714 | 6.798 / 6.866 | 35.46 / 35.52 | 34.24 / 34.84 |
| 2 | Cooperative | 19.968 / 19.958 | 35.170 / 35.206 | 7.109 / 7.076 | 35.54 / 35.32 | 34.46 / 34.62 |
| 3 | Original | 20.073 / 20.046 | 35.507 / 35.553 | 7.288 / 7.294 | 35.28 / 35.61 | 34.13 / 33.93 |
| 4 | Cooperative | 20.183 / 20.050 | 35.710 / 35.563 | 7.350 / 7.357 | 35.54 / 35.69 | 34.01 / 33.97 |
| 5 | Original | 20.123 / 20.143 | 35.723 / 35.750 | 7.444 / 7.378 | 35.46 / 35.35 | 33.91 / 33.80 |

The local `timing8k/timing-samples.csv` and `timing-summary.json` retain
unrounded calculations and other phases. Both timing pressure guards completed
normally.


## Focused operation comparison and diagnostic

The temporary operation benchmark at revision `b470caea` measured five
alternating pairs in one process, with four complete calls per sample. It used
a fixed warmed shape of 2048 rows, 64 heads, 512 valid keys and 8192
compressed rows. Timing included sort, encoding and GPU completion, but no
model, SSD expert loading or allocation. The original averaged 32.0187 ±
0.0898 ms per call and the candidate 32.0472 ± 0.0137 ms, a 0.089% increase
between means. This focused comparison does not show a useful reader payoff
for this shape. The fixture also verified byte-identical expanded, original
and cooperative results at this large shape.

| Pair | First | Original (ms/call) | Cooperative (ms/call) |
| --- | --- | ---: | ---: |
| 1 | Original | 32.17700 | 32.04550 |
| 2 | Cooperative | 32.00350 | 32.06750 |
| 3 | Original | 31.96900 | 32.03525 |
| 4 | Cooperative | 31.96300 | 32.05325 |
| 5 | Original | 31.98100 | 32.03450 |

Local `operation-timing.csv` and `operation-timing-summary.json` retain all
samples. At `b470caea`, the historical command was
`./zig-out/bin/test-deepseek41-metal --fp4-attention-bench` under the pressure
guard. The temporary benchmark was removed along with the experimental reader;
the retained `--fp4-attention` fixture covers packed versus expanded results.

The separate candidate profile covers fresh 32K prefill with the same model
and settings as the timing comparison. Its local evidence is
`candidate-profile/1-candidate/decode-anatomy.json` under the artifact directory.
It recorded 284 command buffers and 101542 encoders, a 45.180823-second
envelope, 34.323765 seconds of encoder union and
10.857059 seconds uncovered. The dual-attention kernel summed to 9.532333
seconds; 72 large `8192 × 4 × 1` dispatches totaled 9.135597 seconds, ranging
from 119.778 to 147.947 ms each. The
[earlier diagnostic](../2026-11-04-prefill-row-blocking/README.md) at `689c2d51`
recorded 10.6686 kernel seconds for fresh 32K prefill. These unpaired profiles are attribution observations,
not evidence of a speedup. The paired operation check above is the stronger
comparison for its fixed shape.
