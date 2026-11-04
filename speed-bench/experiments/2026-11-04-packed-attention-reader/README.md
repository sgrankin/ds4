# Cooperative packed FP4 attention reader

**Results and decision are pending.** This experiment asks whether sharing the
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
The parent baseline is `f965f844`; both measured arms will use the same new
binary with `DS4_METAL_V41_COOPERATIVE_FP4` explicitly set to `0` or `1`.
The candidate revision and exact commands are pending.

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

A bounded Metal fixture is being extended to compare packed and expanded
indexed attention. Fixture, numerical, timing and pressure-guard outcomes are
pending; no performance or equivalence claim is made yet.
