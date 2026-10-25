# Server-side JavaScript and decision tools

Responses can run a small JavaScript program over request-bound data, invoke a
native decision helper, and return a compact result to the generator. The model
can reference a document and question template by name instead of generating
their contents again. Code can also filter or aggregate bound JSON without a
decision-model call.

This implements the server-owned part of [OpenAI Programmatic Tool Calling](https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling):
the `programmatic_tool_calling` declaration and `program`/`program_output` items.
The `ds4` bindings below are provider extensions. The local runtime uses pinned
[QuickJS-NG v0.17.0](../src/vendor/quickjs/README.md), embedded through Zig;
OpenAI's hosted implementation uses V8.

This subset completes a program before returning to the generator. A program
cannot suspend for client-owned functions, MCP or approvals. Ordinary direct
client functions still work outside a program. Requests marking client-owned
tools as programmatic are rejected rather than silently executing them locally.

## Start the server

Use the existing native model bundles; no Python or external JavaScript runtime
is required for serving:

```nu
nu download-model.nu flint
nu download-model.nu clef-flash
nu tools/zig-build ds4-server
./zig-out/bin/ds4-server --model gguf/flint-int4 --decision-model gguf/clef-flash-int4 --ctx 8192 --tokens 2048
```

Any already supported generator can use this shared Responses path. Discover
the actual loaded helper IDs with `GET /v1/models`. See [Clef setup](CLEF.md)
and the [direct hosted decision tool](HOSTED-TOOLS.md). One helper model is
pinned by the request; programs do not load models or choose a different model.
Pure JavaScript programs do not require a helper.

## Bind evidence and a question template

Send this JSON to `POST /v1/responses`, or add `type: "response.create"` for the
Responses WebSocket connection:

```json
{
  "model": "flint",
  "input": "Classify the attached customer request and report its department.",
  "max_output_tokens": 2048,
  "max_tool_calls": 4,
  "tools": [
    {
      "type": "programmatic_tool_calling",
      "ds4": {
        "inputs": {"evidence": "The customer was charged twice and requests a refund."},
        "schemas": {
          "department": {
            "type": "choice",
            "name": "department",
            "instructions": "Who should handle the customer request?",
            "choices": [
              {"value": "billing", "description": "Payments and refunds"},
              {"value": "technical", "description": "Software faults"}
            ]
          }
        },
        "description": "inputs.evidence is the attached customer request. schemas.department is its department question template."
      }
    },
    {"type": "ds4:decision", "model": "clef-flash", "allowed_callers": ["programmatic"]}
  ]
}
```

The generator sees `ds4_exec({code: string})` and a manifest describing the bound
names, types, array lengths and object fields. Question manifests include their
type, choice count and up to sixteen short choice labels. Full input values and
question descriptions stay in the executor. Keep `ds4.description` concise: it
is visible to the generator and should explain what the bindings mean.

`ds4.inputs` and `ds4.schemas` are JSON objects. Their values become the
JavaScript globals `inputs` and `schemas`. A schema value can be a complete
question object or a question array; pass the corresponding array to the helper.
The model could generate:

```js
const result = await tools.ds4_decide({
  input: inputs.evidence,
  questions: [schemas.department]
});
text({department: result.answers[0].choice});
```

Only `text(value)` emissions form the program's logical result. Multiple calls
append JSON values to an `output` array; the ordinary JavaScript return value is
ignored. The example yields `{"output":[{"department":"billing"}],"error":null}`.
The generator receives this compact result and can answer the user or continue
with another bounded program.

Bindings are request-local names, not persistent file handles. Redeclare them
when a later request needs to run new code over the same data. Completed history
can replay from its receipts without granting new execution access to old tools.

## Tool routes and execution

`allowed_callers` on `ds4:decision` follows the PTC routing convention:

| Value | Allowed use |
|---|---|
| Omitted, `null` or `["direct"]` | Ordinary hosted `ds4_decide` call |
| `["programmatic"]` | `tools.ds4_decide(...)` inside JavaScript |
| `["direct", "programmatic"]` | Either route |

A programmatic helper requires the `programmatic_tool_calling` declaration in
the same request. A program-only helper is not exposed as a direct model tool;
a direct attempt produces a visible error. The helper's native Decisions input,
question, context and media rules still apply. Omitting `input` inside code uses
the decision descriptor's bound input, or otherwise the latest resolved user
message, just as for a direct hosted call.

Native helper callbacks run synchronously. `await` works, but `Promise.all`
does not create parallel native inference. Batch related questions into one
helper call when they share evidence. A question's `name` labels its answer;
it does not select a record. To classify records separately, copy the template
and identify each target record in its `instructions`.

JavaScript can use the structured `answers`: choice questions return `choice`,
`confidence` and `probabilities`; predicates return `probability`; score
questions return `score`, `confidence` and `probabilities`. Select or aggregate
the evidence needed in the final emission. Native tool errors throw catchable
JavaScript errors with the native message, and remain visible in the audit.

`max_tool_calls` is shared across the entire Response and defaults to four. Each
outer program and each native helper invocation consumes one call: one program
with two helper invocations consumes three. `max_output_tokens` likewise covers
all generated code, reasoning and final answer across internal rounds. The
program's emitted result is subsequent input to the generator.

## Runtime limits

Each invocation creates a fresh QuickJS runtime/context. JavaScript and native
tools run in the serving process. Only the bound JSON, `tools.ds4_decide` and
`text` are registered; no Node, filesystem, network, subprocess, console or
module-loading interface is installed. Shared-memory/blocking atomic APIs are
removed. There is no persistent JavaScript state between invocations.

| Limit | Current value |
|---|---|
| Generated JavaScript source | 64 KiB |
| QuickJS heap | 64 MiB |
| JavaScript stack | 512 KiB |
| Program deadline, including native work | 10 seconds |
| Final serialized program result | 64 KiB |
| One native helper argument body | 8 MiB |
| Native audit byte accounting | 16 MiB of arguments, resolved inputs and native outputs before serialization |

The deadline and disconnect/shutdown cancellation are checked by the VM,
before/after native callbacks, and through the helper's cancellation points.
Already submitted GPU work retires at those native boundaries; this is
cooperative interruption. The JS heap cap covers its bindings and allocations,
while loaded model weights and native inference workspace have their existing
owners and memory policies.

Exceptions, stalled promises and resource limits produce a structured error in
the result. The generator can handle an ordinary failure within its remaining
budgets. Exhausting the shared call budget terminates the Response as failed.
Tool choice currently supports `auto` and `none`; required/forced choices remain
unsupported. The Responses declaration is type-only plus optional provider
`ds4`; the Agents API's `enabled` field is not accepted here.

## Receipts, replay and clients

Each invocation adds two public output items:

- `program`: `id`, `call_id`, generated `code` and an opaque `fingerprint`.
  Provider `ds4.arguments` preserves the original invocation string,
  `ds4.call_ids` identifies its complete assistant batch, and `ds4.bindings`
  snapshots the request bindings.
- `program_output`: its own `id`, the same `call_id`, a serialized `result`
  string and terminal status `completed` or `incomplete`. Provider `ds4.calls`
  records each native helper's `name`, `model`, actual `arguments`, resolved
  `input`, native `result`, numeric `status` and `elapsed_ms`.

Bindings and native audit remain checkable in the public transcript and durable
response store. Logical model replay reconstructs only the original `ds4_exec`
call and compact program result. This keeps bulk intermediate data out of the
generator's context; HTTP and stored receipt sizes still include that data.

HTTP JSON, SSE and WebSocket use the shared hosted-tool loop. Streams emit
ordinary `response.output_item.added`/`done` events with stable IDs and indices;
there is one terminal Response event after its internal rounds. The installed
OpenAI Python SDK 3.37.0 recognizes both program item types and preserves the
provider fields when dumping/replaying output.

Use `previous_response_id`, a conversation reference or complete prior output
for continuation. Stored receipts survive restart when
`DS4_SERVER_RESPONSE_STORE_DIR` is configured; WebSocket `store:false` ancestry
keeps its existing connection-local lifetime. Replaying a terminated local
program/output pair restores its result and never executes the program again.
A terminal `incomplete` output replays as the recorded tool error. Orphaned
outputs, missing outputs and suspended client-tool programs are rejected.

With DeepSeek's normal reasoning enabled and its public summary omitted, durable
`previous_response_id` restores the parent's private logical reasoning text.
Public-output-only replay retains the recorded program results without execution,
but lacks hidden reasoning: a cold recovery can recompute visible history with a
different reasoning context. Restoring logical reasoning and reusing exact KV
state are separate concerns; KV reuse follows the existing cache policy.

Run `nu tests/programmatic-decisions.nu` against an already loaded server for
protocol checks; its `--suite scenarios` compares ordinary and favorable cases.
On a fixed Flint/Clef-Flash queue screen, code mode reduced the larger task's
median time from 17.85 to 13.78 seconds, while the small task grew from 5.48 to
12.04 seconds. Larger-task answers had correct fields but duplicated JSON in
their final text. Expose this tool for workflows with useful bound data rather
than enabling it on every request. See the [measurement and limitations](../speed-bench/experiments/ptc-decisions-20261025/RESULTS.md);
successful execution alone does not establish task quality or speed.
