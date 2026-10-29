# DeepSeek V4.1 Flash implementation

The runtime turns a rendered conversation into token IDs, synchronizes a session to those IDs, and returns logits for the next token. An engine owns the model and shared execution services; each session owns its committed token frontier and the model state needed to continue from it. The ordinary path evaluates every token after a matching prefix. Optional checkpoint and suffix mechanisms can avoid some evaluation, but have different correctness and eligibility boundaries.

This page maps those ownership and transaction boundaries. The [reference manual](reference-manual.md) describes model mathematics and the released reference; the [Apple Silicon manual](apple-silicon.md) describes physical layouts, weight movement and kernels. The shared [architecture](../../architecture.md) explains server and scheduler boundaries.

## Model and session ownership

[Model.init](../../../src/engine/model.zig) opens the GGUF, reads its metadata and tensor directories, and maps the model for the selected backend. Engram tables have special mapping treatment because their rows are read on demand. The engine retains the model and weights beyond any one request. [ModelConfig.deepseek41](../../../src/engine/config.zig) validates the expected metadata, sharing topology and Engram tensors against the supported local profile; a GGUF merely identifying itself as DeepSeek is not enough. Weight variants and their qualification are described in the [model guide](../../MODELS.md).

A session's [graph resources](../../../src/engine/deepseek41/resources.zig) include recent attention rings, compressed main and index owners, compressor carries and workspace. Its common [frontier](../../../src/engine/session_frontier.zig) records committed tokens, logits and validity, with approximation flags where applicable. The graph position and token count must agree. The engine's model and execution services can be shared, but one session's attention history is not another's. Scratch capacity can grow for prefill; the shared resource pool and GPU admission coordinate its use across sessions. The server can interleave bounded prefill and decode work, but a scheduler quantum is not the model's prefill capacity. See [session resources](../../../src/engine/session_resources.zig), [session batching](../../../src/engine/session_batch.zig) and the [serving architecture](../../architecture.md).

## From request to next token

```mermaid
flowchart LR
    R[Request messages and tools] -->|canonical values| M[Materialize and render]
    M -->|text and control boundaries| T[Tokenizer]
    T -->|token IDs| S[Session sync]
    S -->|unprocessed IDs| P[Prefill or scalar step]
    P -->|mutate history and frontier| V[Live session graph]
    P -->|last activation| H[Output head]
    H -->|next-token logits| G[Sample and emit]
    G -->|sampled token ID| D[Decode step]
    D -->|mutate history and frontier| V
    D -->|new logits| G
    V -->|capture| C[Stored checkpoint]
    C -->|restore| V
    V -->|existing frontier| S
```

The server [materializes](../../../src/server/materialize.zig) request and stored conversation inputs into canonical values. The DeepSeek [renderer](../../../src/models/deepseek41/prompt.zig) applies message, reasoning and tool framing before tokenization. Generated tool-call syntax is output for the caller to execute; it does not execute a tool. Rendered text and token IDs are distinct: token boundaries and sampled IDs matter when a continuation is reused. The server's [reuse probe](../../../src/server/reuse.zig) can identify a live continuation and materialize its visible suffix without retokenizing the already sampled prefix.

[Runtime.sync](../../../src/engine/deepseek41/session.zig) requires a nonempty prompt with room for at least one generated token. If the valid live checkpoint is an exact token prefix and its graph position agrees, sync evaluates the remainder; otherwise it resets the graph and starts again. It schedules multi-token portions through layer-major [prefill](../../../src/engine/deepseek41/prefill.zig), or a scalar step for a single row. After the evaluated portion it runs the output head once, commits matching logits and marks the checkpoint valid. Cancellation can leave a valid evaluated prefix if its head completes; failures invalidate state that cannot safely continue.

For example, a valid session at `[A, B]` synchronized to `[A, B, C, D]` evaluates `C, D` and produces logits predicting the token after `D`. If the caller samples `E`, [Runtime.eval](../../../src/engine/deepseek41/session.zig) evaluates `E`, advances both the graph and committed token list, and replaces the logits with those predicting the next token. A request `[A, X]` cannot simply append to the `[A, B, C, D, E]` graph: it needs a suitable earlier checkpoint or fresh evaluation. Sampling and output parsing belong to serving, not the model head.

The scalar [executor](../../../src/engine/deepseek41/executor.zig) embeds the token, runs the layer sequence, and applies the output head. Within a layer, attention reads the recent ring and selected compressed history, while publishing owners update their rows and carries; the feed-forward path routes experts and combines their outputs. [Attention](../../../src/engine/deepseek41/attention.zig) owns the publication and selection calls. The [reference manual](reference-manual.md) explains what those operations compute, and the [backend manual](apple-silicon.md) explains how prefill batches and decode move their data. Optional steering is applied at the residual expansion hook in the executor, so it changes model activations rather than prompt text; changing steering requires treating continuation state accordingly. See [session steering](../../../src/engine/session_steering.zig).

## Continuation and storage

A live continuation is the in-memory frontier plus its graph, not just the conversation text. Saving a usable checkpoint must preserve tokens, logits, the recent windows, published compressed and index rows, and incomplete compressor carries at the same position. [State payload](../../../src/engine/deepseek41/state_payload.zig) serializes and restores these components; [state format](../../../src/engine/deepseek41/state_format.zig) identifies versions 2–5 for expanded and packed main/index combinations. Supported layout conversions allow a checkpoint to be loaded into a differently configured graph. Restore validates the payload and must leave a coherent frontier before continuation.

The [state cache](../../../recipes/03-state-cache.md) holds disposable inference checkpoints in RAM and optionally SSD. It is separate from the expert-weight cache and SSD reads used to execute the model. The older server KV store is another checkpoint persistence mechanism, configured separately from the state cache; neither is the authoritative conversation record. Durable [conversation records](../../../src/server/conversations.zig) contain logical items and outlive slots and response files. [Publication](../../../src/server/publication.zig) checks the conversation version before committing generated items. A retained conversation can be rendered and evaluated again even when no inference checkpoint remains.

An exact prefix checkpoint represents the same preceding token sequence and execution state and can be followed by ordinary evaluation. Server [persistence](../../../src/server/persist.zig) checks cache identity and attempts restoration; a missing, incompatible or failed load falls back to evaluation. A suffix-grafted lineage remains approximate and is excluded from exact-prefix publication and reuse; it can have a separate approximate identity. Live reuse also has request-specific paths for visible output and tool results, so an apparent text prefix alone should not be confused with a universally reusable state.

## Edited contexts and semantic endpoints

Suffix reuse addresses a different case: a middle portion of a conversation changes while a later span remains. After evaluating the changed prefix, the runtime can import an eligible unchanged span from a stored source and recompute a fresh tail. The source activations were computed with the old left context, so this is an **approximation**, even when the imported token IDs match. [Suffix transactions](../../../src/engine/deepseek41/suffix_transaction.zig) and the [suffix recipe](../../../recipes/02-suffix-reuse.md) describe the transaction and rollback. The archive retains normalized BF16 pre-RoPE originals; relocation rotates them at the destination and repacks them, rather than repeatedly rotating already quantized rows. Position, alignment, source availability and fresh-tail requirements constrain the operation; failure falls back to exact evaluation.

Semantic endpoints make useful source checkpoints at message, tool-call and
result boundaries. The renderer maps ends to completed token positions and
selection bounds their count and spacing. Consider a changed tool result followed
by a long unchanged span: the runtime evaluates the changed prefix and the
ordered system, developer and user controls and rendered tool declarations.
Those controls form the admission identity used to find a compatible source.
Only after evaluating them can it graft an eligible matching span, then evaluate
the fresh tail; absent a compatible source or valid graft it evaluates normally.
Changing or deleting a control changes that identity, even if later text matches.
By default the latest actual user message also stays fresh. An optional ongoing-task
policy changes that cutoff under its own conditions. The [semantic checkpoint
recipe](../../../recipes/16-semantic-context-checkpoints.md) explains the boundary
and examples.

Ordinary capture can stop at chosen endpoints. The optional [in-sweep collector](../../../src/engine/deepseek41/prefill_checkpoints.zig)
captures endpoint windows, compressed prefixes and odd carries during an existing
prefill without installing an earlier endpoint into the live session. Unsupported
collection falls back to ordinary capture. The independent policy and eligibility
choices are:

- Runtime defaults choose recent boundaries and leave in-sweep capture off;
  server `persistent-fast` selects anchor endpoints and enables collection.
- Suffix reuse requires its configured cache and compatible native DeepSeek Metal
  mode. Steering, tensor parallelism, DSpark/MTP and decoder replay are excluded
  from semantic grafts. The recipes detail the remaining gates and overrides.

## Draft proposals and verification

DSpark is an optional trained draft model, separate from the target. It uses captured target features and draft history to propose tokens; the target verifies proposals and commits only the accepted prefix. [DSpark verification and commit](../../../src/engine/deepseek41/dspark.zig) must keep the target frontier, attention state and draft history consistent after rejection or rewind. Proposals are not target outputs until verified. Session-batched serving uses ordinary target decoding; DSpark does not accelerate prompt prefill. The [speculative decoding guide](../../SPECULATIVE_DECODING.md) describes activation and limitations, including the distinct MTP path.

A restored target checkpoint can continue without a saved draft history, but the drafter then needs to rebuild its window. The optional [DSP1 appendix](../../../src/engine/deepseek41/draft_checkpoint.zig) stores target feature rows for that rebuild. It is off in ordinary engine defaults and enabled by fast presets when DSpark is used. The appendix is an aid to draft restoration, not a substitute for the target checkpoint or proof that speculative decoding will help a particular workload.
