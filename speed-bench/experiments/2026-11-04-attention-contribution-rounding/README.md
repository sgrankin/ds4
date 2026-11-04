# Preserving contribution rounding in indexed prefill attention

**Result pending.** This experiment tests whether a different rounding sequence
can retain the speed benefit of skipping unit rescaling with less model drift.
The one-hour window is November 4, 2026, 12:09–13:09 Eastern. The baseline is
`1312c914`; the candidate revision and measurements are pending.

## Question and mechanism

The [unit-rescale experiment](../2026-11-04-attention-unit-rescale/README.md)
skipped `exp(M - M)` and multiplication of the accumulated output by one
when an attention row did not change the finite running maximum. It reduced
isolated operation time by about 9–10% and full-model inference total by about
1.1–1.5% on its measured workloads. Small kernel differences propagated to
model logits, so that version remains opt-in and is not enabled by the fast
presets.

The new candidate tests a possible explanation: removing the identity multiply
may change floating-point contraction and the rounding of the weighted KV
contribution. It explicitly rounds that contribution before using
`fma(output, 1, contribution)`, with local `#pragma clang fp contract(off)`
and `reassociate(off)`. This is a hypothesis about contraction, not evidence
of the instructions emitted for the baseline. The fixture and model checks must
establish the actual numerical effect and whether any speed benefit survives.

The planned `kernel_dsv4_indexed_mixed_attention_heads16_rb4_unit_scale_rounded`
specialization is selected by `DS4_METAL_V41_ATTN_UNIT_SCALE=2`; `1` selects
the prior approximation and `0` selects the original path. Four-row staging
must be enabled with `DS4_METAL_V41_PREFILL_ATTN_RB4=1`. The change is limited to
eligible indexed prefill attention and changes no packet or storage format.
Implementation and activation are subject to verification against the finished
source.

## Measurement plan

Compare mode 2 with mode 0 using the existing six-shape expanded and packed
attention fixture, then five alternating operation pairs at 512, 2048 and 8192
rows. Retain all samples, identify timing scope and report variation and order
effects. If the operation result is promising, run five profile-free full-model
pairs at both 8K and 32K and measure fixed-history full-vocabulary quality.
Report delta-NLL and KL as descriptions of distribution drift alongside practical
quality, not as automatic rejection criteria.

A reasoning-enabled long-context coding task will check practical behavior,
with a 16384-token context to avoid the reserve-compaction confound in the
prior task. Record actual dispatch and task outcomes. Run heavy model work one
job at a time under the existing pressure guard. Local raw artifacts belong in
`.workspaces/hillclimb-20261104/06-attention-contribution-rounding/`.

## Results and decision

Pending. Record the candidate revision, exact commands, fixture results,
operation samples, any full-model timing and quality results, resource
conditions, failures and the resulting decision here.
