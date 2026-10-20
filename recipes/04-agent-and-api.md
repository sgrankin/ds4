# Context editing, archival and a harness-neutral server

## Two layers to implement independently

CLM-style behavior lets the model edit its own context as a file. Suffix reuse
accelerates evaluation after such an edit. Either can exist without the other.
The CLM paper's unrestricted context-file concept is broader than our guarded
DS4 mirror: we preserve a protected prefix and native role boundaries, and use
ordinary file/bash tools rather than a new context-management API.

Direct-agent use:

```sh
./ds4-local persistent-fast --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf \
  --live-context-file .workspaces/agent-context.txt
```

The agent exports its live transcript before tools, including reasoning, and
imports accepted edits after tools or while idle before the next user turn.
The model already sees that conversation. Instructions therefore forbid dumping
the full mirror or pasting old text into edit arguments, which otherwise copies
history back into context and can create cleanup loops.

Give short recipes: list only headers; inspect a bounded old span if necessary;
archive needed evidence; delete/shorten selected spans in the file itself;
print only a small receipt; then continue the task. Batch related edits once.
Do not keep searching for every mention of a topic: cleanup commands, requests
and search output themselves mention it.

## Opaque handles, raw bodies

```text
[[DS4_TURN a3c91e704b82d6f0 role=assistant]]
<｜Assistant｜>...native conversation span...
```

Each real user/assistant span gets a random nonzero 64-bit ID displayed as 16
lowercase hex characters. Treat IDs as strings; file order determines conversation
order. Keep the ID with a retained, rewritten or moved span. New turns receive
fresh IDs. Identity is worker-local, with no restart persistence promise.

Headers are file metadata and never enter model tokens/KV. Literal body lines
that resemble a header acquire one extra backslash display escape; preserve it
when retaining text. Tool calls/results may live inside an assistant span.
Validate roles and preserve native chat delimiters. Reuse original token IDs
for unchanged bodies: rendered text is not an injective token encoding, so
ordinary text spelling a control marker must not become a role boundary.

```text
export:
    preserve complete unlabelled prefix
    emit stable metadata handle + escaped raw body per real role span
import:
    validate complete prefix and known unique handles
    unescape display-only lines, strip handles
    copy unchanged body's original tokens; tokenize only changed text
    commit transcript, then reuse eligible KV or recompute
```

Protect the COMPLETE prefix, including session-date text between static system
tokens and the first user, not just the generated system-prompt token count.
Preserve the latest request and useful evidence. See `agent_live_context_*` and
the injected editing recipes in [ds4_agent.c](../ds4_agent.c).

Built-in automatic compaction and `/compact` instead generate a task-state
summary and retain recent verbatim history. The pre-RoPE path can reuse an
eligible older retained turn without a mirror file. It does not itself write
a permanent fact ledger; file archival/retrieval remains agent policy.

## What was qualified

[Item41](../speed-bench/experiments/agent-package-hour41-20261009/README.md),
runtime `e203b806`, exercised actual CLI left/right SSD imports and fresh
inventory/archive reads. A favorable warmed edit-heavy workflow saved 46.5%
launcher wall versus native brackets (drift .70%). Its separate real-summary
compaction reduced 3537→3021 tokens, imported 696 and then returned EMBER-7319;
the 90.16s includes summary generation and is not a comparative speed result.

[Item42](../speed-bench/experiments/agent-context-structure-hour42-20261009/README.md),
`d2952200`, let the MODEL choose, archive byte-for-byte and delete an old user
turn, then retrieve EMBER-7319 from disk while retaining COVE-3917. There was no
full mirror dump, cleanup loop or automatic compaction. This96.47s four-turn
screen used ordinary recomputation for an odd 431-token deletion: archival
quality evidence, not another suffix speedup. It used the earlier decimal IDs.
The first real run caught the incomplete protected-prefix bug described above;
failed records remain. The later
[opaque-ID correction](../speed-bench/experiments/agent-context-opaque-ids-20261009/README.md)
added reorder/edit round-trip and exact-recipe CPU checks, not a new model-quality
or timing result. Distinguish these scopes when reproducing it.

## Use a different agent harness

The server exposes the reuse machinery; it is not tied to `ds4-agent`:

```text
send stable system instructions/tools + full actual conversation
execute returned function call; append its real output and call ID
edit older client history when appropriate
send a fresh latest User with the same stable cache identity
inspect actual graft/fallback and grade first result
```

`/v1/responses` supports text/function-tool continuation and streaming. A full
history can use `store:false` and `prompt_cache_key`. With
`DS4_SERVER_RESPONSE_STORE_DIR`, stored Responses support GET/DELETE and
`previous_response_id` continuation/forks, preserving private native reasoning
and tool replay. Current top-level instructions are request-local, not inherited
from a parent. A durable parent is append/fork history, not a generic edit API;
the explicit suffix-edit path is full replay or Conversation edits.

Conversations support create/get/delete and text/function/reasoning item
append/list/get/delete, with optimistic version checks. Editing items discards
private replay; suffix states may still retain original conditioning. History
storage survives KV eviction, but persistence does not promise exact hidden
state after restart. Response-file and Conversation publication are separate
atomic writes, not one cross-file transaction. This is a supported API subset,
not complete ecosystem compatibility or multimodal replay.

[Responses qualification](../speed-bench/experiments/responses-hour11-20261004/README.md)
retains two wrong-marker model answers; full-input controls reproduced one
marker error. [Conversations qualification](../speed-bench/experiments/conversations-hour13-20261004/README.md)
passed its corrected protocol/model checks, including restart and concurrent-edit
409. [Installed SDK/Pi smoke](../speed-bench/experiments/sdk-preset-hour36-20261004/README.md)
passed real tool loops through the wrapper but made no CLM edit or suffix graft;
compatibility is not a speed result.

Keep system/tools stable across stages. In
[Item18](../speed-bench/experiments/stable-instructions-hour18-20261004/README.md),
`bc6896ae`, moving phase directives out of changing top-level instructions cut
two-task time 40.49% through ordinary exact-prefix reuse, despite MORE generated
thinking. This is a client-layout gain separate from approximate suffix reuse.

## Credit and replication

Credit [Context Language Models, Shao et al.](https://arxiv.org/abs/2609.37725)
and [authors' implementation](https://github.com/facebookresearch/context-language-models)
for context as an editable artifact. [ds4-recipe](https://tangled.org/astrra.space/ds4-recipe)
chapters 12/13 describe runtime support for external clients' edits, not an
editing-agent harness. To port, implement one file mirror, ordinary tool access,
bounded instructions and safe import first. Verify an actual model archives,
deletes and later retrieves a changed external fact. Add cache acceleration
after that workflow works; preserve no-graft and failure records.
