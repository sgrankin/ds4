# Native Flint qualification — 2026-10-23

Native revision: the commit containing this report (`flint-metal` bookmark).
Reference: `recursive-oss/rec`
`e53663b262af4cb1ca498e596ca8b1a2d79e0800`, MLX 0.32.0, original group-64
affine INT4 checkpoint with INT8 embeddings. Hardware: Apple M5 Max/128 GiB.
Native serving uses Zig and Metal, with source-only MLX matrix/attention kernels;
it does not load Python or the MLX runtime.

## Fixed comparison

Two identical tokenized prompts: **short 21 tokens** and **context 798 tokens**.
Both runtimes use context 262144, prefill chunks 4096, and 128 generated tokens.
Reference runs set `MLX_METAL_FAST_SYNCH=1` before import. Native Metal uses safe
math and up-to-eight-token resident packets. Each arm ran three fresh-prefix
repeats; the table is the arithmetic mean of warm repeats 1/2. Loading and cold
pipeline setup are excluded. Decode rate is `(generated_tokens−1)/decode_seconds`.
Jobs ran sequentially under the existing pressure guard.

| Case | Prefill ms, reference → native | Decode t/s, reference → native | Request ms, reference → native |
| --- | ---: | ---: | ---: |
| Short, greedy | 43.52 → 37.36 | 246.10 → 249.75 | 560.19 → 545.87 |
| Short, default sampling | 44.05 → 37.12 | 224.94 → 247.14 | 609.76 → 551.02 |
| Context, greedy | 227.28 → 216.96 | 239.75 → 235.29 | 757.69 → 756.75 |
| Context, default sampling | 226.58 → 216.78 | 212.61 → 225.94 | 825.20 → 779.12 |

Default sampling is temperature 1, top-k 20, top-p 0.95, min-p 0. The reference also
uses its empty-tool grammar; native serving parses tool calls but has no grammar
masking and uses a different random stream. The 6–10% native default decode gain
is a whole-harness comparison with that capability difference. It does not
isolate a kernel speedup. Context greedy decode is about 1.9% slower, while its
total request time is effectively neutral. Two warm repeats do not establish
universal non-regression or a broad throughput claim.

The short-prefill iteration stayed focused on one bottleneck:

| Implementation | Warm 21-token prefill |
| --- | ---: |
| Initial 64-row NAX tiles, including sparse short expert batches | 202.51 ms |
| SIMD QMV with BF16-dequantized weights for small batches | 61.45 ms |
| Selected packed-affine SIMD QMV, matching MLX's small-vector arithmetic | 37.36 ms |

The selected implementation keeps NAX for larger batches. Dense batches with
at most 8 rows and routed batches with at most 768 rows use packed QMV; routed
row count is six times prompt length. The discarded BF16 QMV candidate was
removed from maintained source.

## Numerical and serving checks

Eight teacher-forced prefixes use reference greedy positions 16/32/64/96 from
each prompt, comparing all 150144 vocabulary logits at identical token IDs:

- Mean delta-NLL: **+0.013817 nats/token**.
- Mean KL(reference‖native): **0.00100151 nats**.
- Top-1 agrees at **8/8** positions.
- `exp(mean delta-NLL)=1.013913`: **1.39% perplexity increase** on these targets.
- Worst sampled position: context 64, delta-NLL +0.092365 and KL 0.004182.

This is a focused numerical probe of two continuations, not a coding benchmark,
confidence interval, or long-context quality guarantee. Free-running replies
can diverge through BF16 reduction order and sampling.

Final WebSocket run **websocket4 passes 29/29 protocol checks and 17/17 answer
probes**. It covers independent concurrent conversations, ephemeral parents,
function/tool-result continuation, the official SDK, disconnect/cancellation
recovery, durable response/conversation restart, and clean shutdown. Its long
parent contains 4855 input tokens, crossing the 4096-token prefill boundary;
the next turn contains 5011 tokens after decoding and correctly credits 4855
cached tokens. An earlier run reported 4998 by crediting decoded tail tokens
that native sync actually recomputed. The final run verifies the corrected
prefill-base accounting.

Additional qualification:

- The full Release host/script suite passes (`host-suite4.log`); final launcher
  fixtures pass 11 cases. Existing server/session/tokenizer, persistence and
  sampling fixtures remain covered. Earlier runs caught a compatibility-path
  vocabulary-size regression (fixed) and needed localhost sandbox permission.
- Debug server/session checks pass; native tokenizer matches 47 Hugging Face
  fixtures exactly, including token IDs and raw-byte round trips.
- Real Metal KV fixture passes a 65536-token bank crossing and wrapped local-ring
  assembly/publication using actual GPU buffers. This fixture runs without a
  loaded model. GPU sampler ordering, token handoff and EOS fixtures also pass.
- Final guarded DeepSeek before/after smoke: selected IDs and every reported
  top-logprob/logit match exactly. Both use the installed 1017 IQ2 checkpoint,
  24 input tokens, context 512, prefill chunk 512, greedy decode and a 32GB
  expert cache. The 8-token allowance stops after two reported steps.

The supported serving boundary is **M5 Max text Responses HTTP/WebSocket** with
model-controlled reasoning and independent session state. Flint tensor-state
SSD caching, suffix relocation, vision, grammar masking and the built-in
`ds4-agent` are unsupported. Durable Responses history is separate from tensor
cache persistence. A full 262144-token model conversation has not been tested.

## Reproduction and local evidence

See [Flint setup and reproduction commands](../../../docs/FLINT.md), the
[portable original-harness driver](../../ideas/flint_reference.py), and
[prefill source/attribution notes](../../../src/metal/flint/prefill/README.md).
The native command is `flint-smoke MODEL PROMPT_JSON OUTPREFIX [default|quality]`.
All raw outputs remain ignored and local; this report records the selected point.

Evidence under `.workspaces/flint-baseline-20261023/`:

- Reference: `{short,context}-{greedy,default}-{1,2}.json` and prompt fixtures.
- Selected native: `{short,context}-native-{packed,default}-{1,2}.json`.
- Iterations: `native-short2.log`, `native-short-qmv.log`,
  `native-short-packed.log`; frozen binaries retain the compared arms.
- Numerical: `teacher-suite.json`, `teacher-reference-*.f32`,
  `teacher-native-*.f32`; recompute with `tools/flint-quality.py`.
- Serving: `websocket4/result.json` and `websocket4.log`.
- GPU boundaries: `kv-fixture.log` and its pressure-guard result.

Host/Debug evidence is under `.workspaces/flint-tests/`:
`host-suite4.log`, `launchers-final.log`, `debug-server.log`,
`debug-session.log`, and `debug-tokenizer.log`.
