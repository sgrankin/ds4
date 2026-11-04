# Skipping unit rescaling in indexed prefill attention

**Result pending.** This experiment asks whether skipping a redundant rescale in
online softmax reduces indexed prefill attention time. The one-hour window is
November 4, 2026, 11:22–12:22 Eastern. Baseline `e050fa0b` includes the
adopted four-row staging kernel; the candidate revision and measurements will
be recorded here when available.

## Context and mechanism

The [four-row experiment](../2026-11-04-attention-four-row/README.md) found a
consistent isolated operation improvement, but did not establish a whole-model
speedup. A separate fresh 32K profile attributed 9.329778 seconds of GPU work
to the four-row attention kernel. That profiled observation identifies a target;
it is not a profile-free latency comparison.

For each row, online softmax maintains a running maximum `M`, a sum and a
weighted output. The baseline computes `new_m = max(M, score)` and rescales
the accumulated sum and output by `exp(M - new_m)` before adding the new row.
For finite values when `score <= M`, `new_m` is `M` and the old scale is
mathematically one. The candidate retains the sum and output directly and
adds the weighted row in that case; when the maximum increases it uses the
original rescaling path. Row consumption remains chronological.

The hypothesis is that avoiding the redundant exponential and multiplications
saves time often enough to outweigh the branch and any compiler or hardware
costs. The algebra alone does not prove that compiled instructions are removed,
that the branch is profitable, or that floating-point results match. Numerical
and practical quality must be measured rather than inferred from the formula.

The candidate is a separate `heads16_rb4_unit_scale` kernel specialization,
selected by numeric `DS4_METAL_V41_ATTN_UNIT_SCALE=1`; `0` selects the control.
Both arms force four-row staging, so the comparison isolates this change. The
candidate applies only to eligible four-row prefill dispatches; decode and other
attention paths are unchanged. It adds no persistent buffer or cache. See the
[Metal kernel](../../../src/metal/dsv4_misc.metal) and
[host dispatch](../../../src/metal/attention_batch.zig) for the implementation.

## Setup and planned checks

Reuse the six-shape expanded and packed attention fixture, including wrapped
rings, masked IDs and short windows, and five alternating operation pairs at
512, 2048 and 8192 rows. Record the candidate revision, commands, all samples,
variation, order effects and any failures below. If the operation results are
promising, run five profile-free full-model pairs at both 8K and 32K and
compare full-vocabulary quality on fixed histories. Record practical task
quality if tested; delta-NLL and KL describe distribution changes and are not
automatic rejection criteria.

The earlier report describes the M5 Max model setup and timing scope. Actual
settings and cache conditions for this comparison must be recorded with the
results. One heavy model job at a time remains under the pressure guard. Raw
logs and samples are local and ignored under
`.workspaces/hillclimb-20261104/05-attention-unit-rescale/`.

## Findings and decision

Pending fixture, timing and quality results. No performance or quality claim,
nor adoption decision, has yet been established.
