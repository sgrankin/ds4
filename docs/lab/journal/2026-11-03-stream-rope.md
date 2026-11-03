# RoPE reuse in SSD streaming

The resident fast presets now enable the existing per-token RoPE table and
attention-head transform in eligible SSD streaming decode. Five alternating
pairs at 32K reduced steady decode time by 2.88% after fresh prefill and 2.62%
after an append. All 512 compared full-vocabulary rows and 512 continuation
hashes match exactly across the 8K and 32K checks.
Both native-agent runs also completed a small interval-merging coding task;
each generated implementation passed 207 independent oracle cases.

## Context and setup

The [profile after overlap](2026-11-03-post-overlap-profile.md) identified
repeated inverse rotation inside the low attention output projection. The
alternative already existed for non-streaming execution: transform each head
once, then reuse it in a plain projection. This experiment changes its streaming
eligibility and preserves the existing numerical and fallback paths.

The one-hour slice began at 18:05:46 Eastern on 2026-11-03. It used the M5 Max,
Flash 1017 IQ2/Q2, packed KV, resident cap five, a 64 GiB expert-cache target,
8K prefill chunks and the adopted routing overlap. Fresh controls matter here:
the previous noisy refresh averaged 28.5 tokens/s, whereas this experiment's
32K controls returned to 30.07 tokens/s. The cause of that earlier variation
remains unresolved.

## Findings and decision

At 32K, the candidate wins every decode pair in both run orders. At 8K, fresh
decode is effectively neutral within variation; decode after the append is
better, but its controls include a slow outlier. Neither context establishes a
whole-task improvement: the 32K append timing still varies widely on both arms.
The [result report](../../../speed-bench/experiments/2026-11-03-stream-rope/README.md)
owns all samples, scopes, commands and qualification details.

Adopt the opt-in feature through `resident-fast` and `persistent-fast`, with
an explicit `DS4_METAL_V41_STREAMING_ROPE_TABLE=0` override. The ordinary runtime
default remains off. The [Apple Silicon manual](../../models/deepseek/apple-silicon.md)
explains the table's lifetime, retained rounding and remaining matrix work.
Prefill arithmetic and the expert selection policy are unchanged.
