# Cooperative packed FP4 attention reader

**The 8K comparison has a small favorable mean; 32K shows no clear benefit.**
The decision remains pending the 32K quality check and diagnostic profile.
This experiment asks whether sharing the
scale decode and constructing half-precision values directly can reduce the cost
of reading packed FP4 rows in indexed attention. It retains the existing head
groups, row staging and key order.

## Context and mechanism

Indexed attention reads selected compressed key/value rows and stages them as
half-precision vectors for attention. A packed row holds 512 E2M1 four-bit
values in 256 bytes, followed by 32 E4M3 scale bytes. Each scale applies to
16 values, or four consecutive `half4` vectors; the complete row occupies
288 bytes. The original reader decodes each vector's four nibbles and its
scale separately, so neighboring vectors repeat the scale decode.

The opt-in cooperative reader constructs the exact E2M1 half bits and shares
one decoded scale across four adjacent SIMD lanes, each responsible for one
consecutive `half4`. This requires those four participating lanes to address
the same row and the same 16-value scale group; the shuffle must execute with
the source lane active. The scaled vectors feed the existing attention
calculation. Internal packet mode 3 selects this reader while retaining the
same packed storage; the original reader remains available for comparison.
No new buffers or dispatches are planned. The implementation is in
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
Both measured arms will use the same new binary with
`DS4_METAL_V41_COOPERATIVE_FP4` explicitly set to `0` or `1`. ReleaseFast
builds of `ds4-bench` and `metal-test` passed. Timing commands are pending.

The workload uses an M5 Max with 128 GiB memory and the DeepSeek V4.1 Flash
1017 IQ2_XXS/Q2_K imatrix checkpoint. It requests a 64 GB expert cache, with
an 8192-token prefill chunk and 40960-token context allocation. At frontiers
of 8192 and 32768 tokens, it runs 128 teacher-forced tokens, appends 512 tokens,
and runs another 128 teacher-forced tokens. The OS file cache is not flushed;
the process-local expert cache survives KV restore. The existing
[runner](../2026-11-04-decode-pass/run.nu) will take five alternating timing
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
were zero. Local evidence is `quality8k/quality-summary.json`. This checks the
selected histories, not all possible inputs. Both timing comparisons are complete below; 32K quality is in progress.

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
calculations and other phases. A candidate profile is planned. Source analysis suggests possible explanations for a neutral
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
