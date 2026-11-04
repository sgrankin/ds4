# Four-row staging for indexed prefill attention

**Results and decision are pending.** This experiment asks whether staging four
key/value rows at a time reduces indexed prefill attention time while retaining
the baseline's reuse across 16 heads. The selected window is November 4, 2026,
10:45–11:45 Eastern.

## Context and mechanism

Indexed attention combines the visible raw window with selected compressed
key/value rows. The baseline `heads16_dual` kernel stages one 512-element row
as half-precision values in 1 KiB of threadgroup memory, uses that row for 16
heads, and synchronizes around each row. It processes the raw rows and then
the sorted compressed IDs in order, updating the online softmax for each row.
See the [Metal kernels](../../../src/metal/dsv4_misc.metal) and
[host dispatch](../../../src/metal/attention_batch.zig).

The candidate `kernel_dsv4_indexed_mixed_attention_heads16_rb4` stages up to
four rows in 4 KiB of threadgroup memory and consumes them sequentially. The
hypothesis is that fewer barriers per full block can outweigh the increased
threadgroup memory and any additional indexing cost. It retains 16-head reuse,
the half-precision staging boundary and chronological attention math. It adds
no persistent scratch buffer or dispatch and avoids a per-lane array of
compacted IDs. Preserving the order and staging boundary is intended to preserve
numerical behavior, but does not establish equivalence until tested. The larger
staging allocation could affect occupancy, and neither barrier savings nor an
end-to-end speedup is established by the design.

The earlier [16-row prefill trial](../2026-11-04-prefill-row-blocking/README.md)
regressed at both 8K and 32K and was removed. It changed both staging size and
head reuse, from 16 to eight heads per group, so it did not isolate the effect
of row blocking. The subsequent [cooperative packed reader](../2026-11-04-packed-attention-reader/README.md)
showed no clear payoff and was removed. Baseline `08711070` restores the
original reader and prefill behavior.

## Setup and planned checks

The candidate is opt-in through `DS4_METAL_V41_PREFILL_ATTN_RB4`. The intended
dispatch is limited to multirow, non-quality attention with Metal Performance
Primitives available, 32 or 64 heads, width 512, top-k 512 and window 128.
Candidate revision and measured build: **pending**.

The bounded packed-versus-expanded attention fixture covers five shapes and is
being extended to compare the candidate off and on. Agreement would cover only
the tested rows, masks and shapes. A focused operation comparison is planned
with five alternating samples per arm, four calls per sample, at 2048 rows,
64 heads, 512 valid keys and 8192 compressed rows. Its scope and actual samples
will be recorded with the results; operation timing alone cannot establish
whole-model performance.

If the operation comparison is promising, the existing
[decode-pass runner](../2026-11-04-decode-pass/run.nu) will take five alternating
pairs at 8K and 32K and run quality checks. The planned model setup matches the
preceding report: M5 Max with 128 GiB memory, DeepSeek V4.1 Flash 1017
IQ2_XXS/Q2_K imatrix checkpoint, SSD streaming, requested 64 GB expert cache,
8192-token prefill chunk and 40960-token context allocation. The workload uses
128 teacher-forced tokens before and after a 512-token append. The OS file
cache is not flushed, and the process-local expert cache survives KV restore.
Actual commands, settings, timing scope, pressure conditions and quality
coverage remain to be recorded.

Raw output is local and ignored under
`.workspaces/hillclimb-20261104/04-attention-four-row`.

## Findings and decision

Fixture results, operation samples, any full-model measurements and the
retention decision are pending. No performance or quality claim is made yet.
