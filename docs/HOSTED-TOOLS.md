# Responses hosted tools

The server can run a decision helper during a generative Response. The generator
requests the decision, the server executes it in process, and the generator
continues with the result. The client receives the receipt and final answer in
one Response. HTTP JSON, SSE and WebSocket share this behavior.

This uses the [OpenResponses provider extension rules](https://www.openresponses.org/specification#extending-tools):
`ds4:decision` and its corresponding `ds4:decision_call` receipt are local
extensions. Ordinary `function` tools still return to the client for execution.
[Programmatic tool calling](PROGRAMMATIC-TOOLS.md) can compose native decisions
and reduce their results in server-side JavaScript. MCP and background agent
tasks remain separate work.

## Start and discover

```nu
nu tools/zig-build ds4-server
./zig-out/bin/ds4-server --model gguf/flint-int4 --decision-model gguf/clef-flash-int4
http get http://127.0.0.1:8000/v1/models
```

Any supported generative model can be the main engine. The helper must already
be loaded with `--decision-model`; discovery lists its actual model ID and
`decisions` capability. No model is loaded or switched by a tool call. See
[Clef setup and both decision APIs](CLEF.md).

## Request

```json
{
  "model": "flint",
  "input": "The customer was charged twice and requests a refund.",
  "instructions": "Use the decision helper to choose billing or technical support, then explain its choice.",
  "tools": [{"type": "ds4:decision", "model": "clef-flash"}],
  "max_tool_calls": 4,
  "max_output_tokens": 1536
}
```

The generator sees `ds4_decide(questions, input?)`. Its `questions` use the
existing OpenAI Decisions predicate/choice/score schema. For example, generated
arguments can be:

```json
{
  "questions": [{
    "type": "choice",
    "name": "department",
    "instructions": "Who should handle the request?",
    "choices": [
      {"value": "billing", "description": "Payments and refunds"},
      {"value": "technical", "description": "Software faults"}
    ]
  }]
}
```

Omitting `input` uses the latest resolved user message. A client can instead
bind evidence in the tool descriptor's `input` field, as text or a native
Decisions input array. The generator can override that evidence with its own
`input` argument. It cannot change the pinned helper model.

Binding evidence avoids making the generator copy a long document into tool
arguments. Latest-user PNG/JPEG images retain their placement; an explicit
native Decisions input can use the helper's other supported media. The receipt
includes the resolved evidence, so large media also increase response/storage
size. Existing durable Responses storage still excludes main-conversation images.

One hosted decision binding is supported per request. An unavailable model or
a same-name ordinary function declaration is rejected before generation.
`allowed_callers` defaults to direct invocation; programmatic or both-route
bindings use the [code-mode declaration](PROGRAMMATIC-TOOLS.md#tool-routes-and-execution).
`tool_choice: "auto"` permits calls; `"none"` disables them. Required and forced
tool choices remain unsupported. The positive `max_tool_calls` defaults to four
and counts hosted invocations across all internal rounds; `max_output_tokens`
is likewise shared across the whole Response. Exhausting the call limit yields
a failed receipt and failed Response. Invalid arguments produce a failed receipt
and a tool error the generator can correct within those budgets.

## Receipts, continuation and clients

Each invocation produces a `ds4:decision_call` item with `id`, `call_id`, `model`,
the original `arguments` string, resolved `input`, native `result` and `status`.
Its `call_ids` names the complete original assistant batch, preserving parallel
versus successive call/result groups during replay.
Streams use ordinary `response.output_item.added`/`done` events with stable IDs
and output indices. There is one terminal response event after all internal rounds.
Helper usage lives inside its native result; outer usage counts the generator's
tokens, including the tool-result continuation input.

Completed and failed receipts replay as call/result history without executing
again. Use `previous_response_id`, a conversation reference or full prior output
when continuing. Stored receipts survive restart when
`DS4_SERVER_RESPONSE_STORE_DIR` is configured; WebSocket `store:false` parents
retain the existing connection-local lifetime. In-progress receipts cannot be
submitted as completed history.

Hosted and ordinary client functions can appear in the same batch. The server
executes its hosted calls and yields the ordinary calls. Return only those
client results with the parent reference or full history; the server retains its
own results. It waits for the client results before generating again. A bare
call-ID-only continuation without that history is insufficient for mixed batches.
Flint currently yields after each closed XML invocation: a client-first pair
therefore spans two Responses, while a hosted-first pair can reach its client
call inside one Response. DeepSeek's syntax supports grouped invocations.

The OpenAI Python SDK accepts the extension dictionaries and retains receipt
fields in `model_dump(mode="json")` over HTTP and WebSocket. Its official type
unions do not know these local types, so serialization may warn; use
`warnings=False` when dumping them, or use raw JSON. No official SDK type for
this provider extension is claimed.

## Qualification

```nu
nu tools/zig-build hosted-decisions-test server-test
./zig-out/bin/test-hosted-decisions
./zig-out/bin/ds4_test --server
nu tests/hosted-decisions.nu --url http://127.0.0.1:19332
nu tests/hosted-decisions.nu --url http://127.0.0.1:19332 --edges
# A DeepSeek server with its matching --vision encoder:
nu tests/hosted-decisions.nu --model deepseek-v4.1-flash --media
# Restart with the same response store, then:
nu tests/hosted-decisions.nu --url http://127.0.0.1:19332 --after-restart
```

These wire tests require the already loaded Flint/Clef-Flash pair and the same
WebSocket/SDK environment as `tests/mixed-serving.nu`. Model-free Zig tests cover
declaration ownership, replay and stream invariants. Raw requests, events and
responses stay under `.workspaces/`.
The maintained wire drivers leave reasoning settings unset and inherit native
server/model defaults, including DeepSeek's enabled reasoning.

The [qualification report](../speed-bench/experiments/hosted-decisions-20261025/RESULTS.md)
records the tested revisions/settings, both generator paths, image binding and
known answer-format limitations. Helper result parity and answer-format quality
are separate checks.
