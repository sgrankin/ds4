# Speculative Decoding

[Serving](overview.md)

Speculation proposes future tokens with a smaller draft block, then checks
them with the main model. An accepted prefix advances generation by several
tokens in one verification pass. It does not accelerate prefill.

It is opt-in. Gains depend on the prompt, model, backend, and context length;
poor acceptance can make it slower. Measure your workload rather than assuming
that a draft model always helps.

## DeepSeek V4.1 Flash: DSpark

V4.1 has its own 8.04 GiB support GGUF. It retains the native MXFP4 experts
and uses a different draft graph from V4 Flash. Do not interchange the files.
See [conversion](../../gguf-tools/README.md#convert-v41-dspark) to build it.

```sh
./ds4 -m gguf/DeepSeek-V4.1-Flash-Q2.gguf \
  --ssd-streaming --dspark \
  --mtp-model gguf/DeepSeek-V4.1-Flash-DSpark.gguf
```

The same options work with `ds4-agent` and `ds4-server`. The supported macOS Metal build supports single-host execution with either resident target weights or SSD streaming. CUDA and ROCm paths remain in source but are not newly qualified here. The drafter stays resident, so leave another 8 GiB for it.
Native Metal and retained CUDA code implement two-rank V4.1 tensor parallel
speculation. Add the same `--dspark
--mtp-model FILE` options to both the coordinator and worker. Each peer keeps
the drafter resident; the target and its verification run across both GPUs.
This source support is distinct from completed physical-pair qualification;
consult the DSpark QA below. ROCm remains single-host.

Up to five drafts are checked together. The sampling choices below apply;
`--mtp-exact-sampling` uses sampled proposals and the full acceptance/correction
rule for V4.1. Without compatible saved drafter features, restored sessions warm
the draft window with 128 ordinary tokens before speculation resumes. Supported
canonical checkpoints can retain those features; see [drafter-history persistence](../models/deepseek/deepinfer/cache-and-attention.md#restoring-useful-drafter-history-c12)
for activation conditions and the stored representation.
Measure speed on your prompts: acceptance and SSD
cache size both matter. V4.1 DSpark can still be slower than ordinary decoding,
including in TP mode.

The V4.1 cycle also uses the shared acceptance scheduler. If proposals produce
few accepted tokens or repeatedly stop before proposing anything, it pauses
drafting for a bounded number of cycles and then retries. Paused cycles use
ordinary target decoding. The decision uses acceptance counts, with a default
four-cycle observation window; it does not use wall-clock timing. This scheduler
is enabled when DSpark is selected; `DS4_DSPARK_SCHEDULER=0` disables the pause
policy. The window, minimum average acceptance and retry interval can be changed
with the corresponding `DS4_DSPARK_SCHEDULER_*` settings in
[session controls](../../src/engine/session_control.zig).

The maintained native Metal regression fixture is built with
`./tools/zig-build dspark-native-oracle`; see the
[DSpark qualification commands](../development/qualification/deepseek.md#dspark). It checks
trained draft tensors and actual runtime commits without relying on the
retired original-Flash acceptance driver's log counters.

## GLM: built-in MTP

GLM's draft block is already in its main GGUF:

```sh
./ds4 -m gguf/GLM-5.3-Flash-Q2.gguf --mtp
```

`--mtp-timing` also enables it and prints acceptance and timing counters.
The current GLM cycle commits up to two tokens. No external support file is
needed, and ordinary decode remains the default.

## Qwen3.8: built-in MTP

Both Qwen downloads include MTP and native BF16 n-grams in the main GGUF:

```sh
./download-model.nu qwen38-q4k
./ds4 --mtp
```

Ordinary decode uses the same file with `--mtp` omitted. For non-zero
temperature, add `--mtp-exact-sampling` to preserve the target sampling
distribution. See [Qwen setup](../models/qwen38-flash-next/usage.md) for Metal and CUDA.

Metal SSD streaming supports MTP with the built-in predictor weights. Prompt
prefill prepares its K/V and indexer history from trunk activations and the next
prompt token, without running its attention output, experts or vocabulary head.
The last activation waits until the actual next token is known. Rollback and
checkpoint snapshots retain that pending row. The
[native predictor](../../src/engine/qwen4/mtp.zig) owns this preparation.

`DS4_QWEN4_MTP_PREFILL` controls preparation of newly prefilled history:

| Value | Predictor history |
| --- | --- |
| `auto` (default) | Prepare it with SSD streaming; initialize historical cache rows to zero with resident weights |
| `on` or `1` | Prepare it with either weight placement |
| `off` or `0` | Initialize historical cache rows to zero |

All three settings keep drafts and target verification enabled when `--mtp` is
set. Zero history can change drafts and acceptance; preparing history costs
prefill time, so compare complete request time for the intended workload.
Decode prepares real predictor rows, and each prompt chunk retains its final
activation for preparation at the next boundary. Reused checkpoints retain
their saved history. Without `--mtp`, this setting has no effect.

Qwen payload version 3 stores the initialized predictor prefix and pending row.
Version 2 and plain checkpoints restore target state directly; when MTP needs
missing predictor history, the session replays the saved tokens once before
continuing. [Payload ownership](../../src/engine/qwen4/payload.zig) and
[session restore](../../src/engine/qwen4/session.zig) define those boundaries.

The cycle drafts one token ahead by default and engages a **second, chained
draft** (one extra nextn-layer step conditioned on the predictor's own
stream, verified in a 3-row pass) while recent first-draft acceptance is
perfect, disengaging after repeated second-draft rejections.
`DS4_QWEN4_MTP_DEPTH=2` or `=3` fixes the depth;
`0` (default) is the adaptive policy.

## Sampling and reproducibility

At temperature zero, accepted drafts must match the target's greedy
continuation. At non-zero temperature, the default mode is opportunistic:
ordinary tokens use the requested sampling settings, but matching greedy
drafts are accepted directly. Sampling resumes when the proposed suffix does
not match. This is deliberately more deterministic than ordinary sampling.

Use `--mtp-exact-sampling` to preserve the ordinary target sampling
distribution. Greedy proposals are accepted with their target probability;
sampled proposals use the target-to-draft probability ratio. Rejections sample
from the remaining target probability mass.

When a verified block crosses a tool sampling-mode boundary (for example
entering tool-call syntax during server decoding), the server rewinds to the
block start and re-evaluates the boundary token so the next sample uses the
new mode. Under exact sampling that rewind restores a pre-verify snapshot of
the recurrent state instead of resetting the graph, so long retained
contexts are not replayed at every boundary.

Accepted tokens keep the state produced by the batched verifier. Floating-point
reduction order can differ from one-token decode, so long greedy continuations
need not be byte-identical. For DeepSeek comparisons against the ordinary
target-only path, use `--quality` or `--dspark-strict`; these disable the
speculative acceptance path. They do not promise identical output across
different hardware or execution configurations.

Native session batches generally use ordinary target decoding. Resident
Qwen3.8 on Metal has a batched MTP path; SSD streaming uses ordered speculative
cycles. DS41 constrained requests can take bounded
exclusive speculative turns. See [serving](overview.md#multiple-sessions) for
eligibility, exact-sampling behavior and fallbacks.

## Constrained output and client tools

DeepSeek4.1 and Flint use a native XGrammar compiler and per-generation matcher
for declared function-tool framing. `strict:true` additionally enforces the
supported argument schema; omitted strict and `strict:false` leave argument
schemas non-strict. Strict normalization lists every property as required and
forbids extra object properties. Prompting and grammar use the same normalized
schema. Explicit namespaces map to `namespace::name` for model output while the
Responses wire retains separate namespace and local name.

Responses `text.format` and Chat `response_format` accept JSON objects and JSON
schemas. Chat also accepts regex. Reasoning remains separate from the constrained
answer. Unsupported schema constraints or combinations return an error before
prefill; this is XGrammar's supported subset, not every JSON Schema feature.
Required/named tool choices in DeepSeek thinking mode are rejected; clients can
explicitly choose non-thinking for those requests. `parallel_tool_calls:false`
limits the grammar to one call. Custom stop strings cannot combine with a
constraint because they could cut a valid structure partway through.
Flint's XML argument syntax cannot reliably distinguish a constrained string
from a non-null JSON value in some union schemas. Strict parameters combining
those types with pattern, format or string-length constraints are rejected;
nullable strings and finite enum/const type selection remain supported.

Grammar constraints preserve native decode acceleration: Flint encodes up to
eight sequential forwards per packet, overlapping CPU mask preparation with the
following GPU forward. With DS41 DSpark support loaded, draft and target masks
apply before sampling filters and speculative verification; tentative matcher
advances roll back before accepted tokens are committed. Unsupported Flint
packet filters retain constrained scalar sampling. Explicit reasoning budgets
and ignore-EOS requests retain their existing scalar path. Setting
`DS4_MTP_SPEC_DISABLE=1` selects the scalar control for either model.

Custom tool declarations, background Responses, automatic input truncation and
nonzero presence/frequency penalties or top_logprobs are unsupported and return
clear errors. Ordinary functions and client-executed tool_search remain usable.
Responses, Chat and Anthropic stream DS41/Flint tool names and argument fragments;
completed arguments preserve the same bytes as the accumulated deltas.
