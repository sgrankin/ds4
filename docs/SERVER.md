# Serving Models

[README](../README.md) | [Client setup](CLIENTS.md)

## Start locally

```sh
./ds4-server --ctx 32768
```

The default address is `http://127.0.0.1:8000`. Use `--host 0.0.0.0` to listen
on other interfaces. Restrict access to trusted clients; for an Internet-facing
deployment, put authentication and TLS in front of the server.

`--cors` enables browser cross-origin headers. It does not change the listening
address or provide access control.

The selected build chooses the backend. Pass `-m FILE` for an explicit model.
Use `--chdir /path/to/ds4` when starting outside the project directory so
relative runtime files such as Metal kernels can be found.

## APIs

| Endpoint | Use |
| --- | --- |
| `GET /v1/models` | Loaded model information |
| `POST /v1/chat/completions` | OpenAI-style chat |
| `POST /v1/responses` | Responses-style requests and continuations |
| `POST /v1/completions` | Text completions |
| `POST /v1/messages` | Anthropic-style messages |
| `POST /v1/systemone` | Clef SystemOne decisions |
| `POST /v1/decisions` | Clef OpenAI Decisions |

The generative model names are aliases within the loaded family. The GGUF or
Flint directory passed at startup selects that engine. Add independently loaded
Clef helpers with `--decision-model DIR`, or pass a Clef directory as `--model`
for decisions-only serving. See the [Clef guide](CLEF.md). The decision endpoints
are stateless and do not stream.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Explain Redis streams."}],"stream":true}'
```

Chat, Responses, and Anthropic support tools and SSE streaming. Reasoning is
returned separately from visible text in each API's native form. Standard
sampling and output-budget fields are supported; explicit request parameters
take precedence over defaults.

Function tools are executed by the client. Client orchestration can call the
native decision APIs directly; the server owns model inference and conversation
state. Embedded JavaScript, MCP execution and hosted decision tools were retired
after the workflow experiments; concise historical results remain in speed-bench.

The default sampling settings are temperature 1, top-p 1, and min-p 0.05.
For DeepSeek, thinking is on by default. `reasoning_effort=max` selects Think
Max only with sufficient context; otherwise it falls back to normal thinking.
`xhigh` maps to normal thinking, not Think Max. Use `think:false`, a disabled
thinking object, or a non-thinking model alias for direct answers.

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

## Prompt cache boundaries

DS4.1 text requests accept OpenAI-style explicit checkpoint hints on text
content blocks in Chat Completions and Responses:

```json
{
  "model": "deepseek-v4.1-flash",
  "prompt_cache_options": {"mode": "explicit"},
  "input": [{
    "role": "developer",
    "content": [{
      "type": "input_text",
      "text": "Stable project context goes here.",
      "prompt_cache_breakpoint": {"mode": "explicit"}
    }]
  }, {"role": "user", "content": "What should we change next?"}]
}
```

The default `implicit` mode retains automatic caching and honors marked ends.
`explicit` mode permits inference-cache reads only at marked prefix endpoints
and writes at the last four marked ends. Historical marks remain eligible for
lookup; more than four marks are accepted. With no marks in the resolved
conversation, explicit mode disables inference-cache reuse and writes. Logical
Response/Conversation storage remains independent, including `store:true` and
`previous_response_id` recovery. Private replay preserves marked blocks.

A boundary ends a prefix; changed earlier text still prevents exact reuse of
later state. It does not independently cache a suffix. The metadata adds no
model-visible text. Encoding splits at marked ends so stored token IDs are
actual prefixes of the complete encoded request; moving a mark can change a
token merge at that boundary. SSD restoration also verifies the restored IDs
against the current encoded prompt before accepting the hit.

Cache budgets and minimum checkpoint sizes still apply. Explicit controls
currently require DS4.1 text; marked images and other model families return a
clear unsupported error. Ordinary Flint caching is unchanged. A tool-result-only
request using explicit controls needs full replay, `previous_response_id`, or a
Conversation reference so it can recover without unmarked live KV. TTL/prewarm
options are not implemented. These breakpoint fields are documented provider
extensions to core OpenResponses.

## Multiple sessions

```sh
./ds4-server --ctx 4096 --batched-session 4
```

Without `--batched-session`, there is one resident session. With it, the server
preallocates independent KV states and queues requests when all slots are busy.
Choose context and slot count together: a context that fits once may not fit
four times. Idle slots can be cached before reuse; active requests are not evicted.

Where the model supports it, the slots share one prefill workspace instead of
each keeping its own, so an extra slot costs only its caches. That matters most
for Qwen3.8 Flash Next, whose transients are sized by the prefill chunk rather
than by the context: at the default chunk they run to several GiB per session.
The startup line reports both figures.

| Backend/model | Decode execution |
| --- | --- |
| Metal, resident V4.1 Flash | Native decoding for 2-8 sessions |
| Metal RDMA TP, V4.1 Flash | Native decoding for 3-8 sessions; ordered fallback for two |
| Metal SSD streaming, V4.1 Flash | Ordered fallback |
| Metal, GLM 5.3 | Native batching through 2051 visible tokens; ordered fallback afterward |
| Metal, Qwen3.8 Flash Next | Native batching of the shared work; recurrent state, caches and PLE history stay per session |
| Single-GPU CUDA, including Spark | Ordered fallback |

Fallback executes the rows separately. It provides concurrency and scheduling
fairness, not the aggregate speedup of native batching. Native grouping may
change floating-point reduction order slightly. V4.1 sessions containing images
use the ordered fallback.

Long prefills yield to active decoders in bounded intervals, normally 128
tokens. `--mixed-prefill-quantum N` changes that interval for testing.
Session-batched serving uses ordinary target decoding, except Qwen3.8 on
Metal, where `--mtp` also batches speculative decoding. Its
`--mtp-exact-sampling` mode uses ordinary batches for nonzero-temperature
requests. DS41 constrained requests can instead take bounded exclusive
speculative turns between other sessions' model work. Other ordinary native
batches do not use MTP/DSpark while session batching is active.
For retained CUDA serving options, see
[CUDA GPUs](CUDA_MULTI_GPU.md#serving-and-network-tensor-parallelism).

## Images

Start with the matching language GGUF and `--vision FILE`; see
[model-specific instructions](MODELS.md#vision).

OpenAI chat and Responses accept inline PNG/JPEG data URIs. Anthropic accepts
base64 image sources. Remote URLs and server-side file paths are rejected.
Image blocks preserve their order in the request. The limit is 16 images and
a 64 MiB HTTP body.

## Disk KV cache

Disk caching saves useful prefixes across slot reuse and server restarts:

```sh
./ds4-server --ctx 100000 \
  --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Clients can resend complete conversation histories. The server first tries
the live token prefix, then compatible rendered-text prefixes from disk, and
prefills the new suffix. With multiple slots, each slot has its own live state;
disk is the additional persistence layer, not the only way to retain sessions.

TP cache loading rebuilds the saved token prefix on both ranks. It is not an
instantaneous restoration of both GPUs. Pipeline loading redistributes the
saved layer state over its route.

Defaults are intended to avoid saving fragile token boundaries. For unusual
workloads, the controls are `--kv-cache-min-tokens`,
`--kv-cache-cold-max-tokens`, `--kv-cache-continued-interval-tokens`,
`--kv-cache-boundary-trim-tokens`, and `--kv-cache-boundary-align-tokens`.
Check `./ds4-server --help` for their defaults.

Quantization variants may share compatible prefixes. Add
`--kv-cache-reject-different-quant` for same-quant reuse only.
Cache files contain prompt text and model state: treat the directory as
private. It is disposable; stop the server before clearing it.

## Tool history and debugging

For DeepSeek, the server preserves sampled DSML tool blocks and assigns
unguessable tool IDs. Replaying those IDs avoids retokenizing a differently
formatted JSON history. The bounded replay map can be stored in cache files.
When exact replay is unavailable, canonical rendering may require rebuilding
part of the prefix.

`--tool-memory-max-ids` bounds this map.
`--disable-exact-dsml-tool-replay` disables it for diagnostic comparisons.
Use `--trace .workspaces/ds4-trace.txt` to record prompt rendering, cache decisions,
generated text, and tool-parser events. Traces can contain sensitive content.

Cache formats are implementation details. The current header and extension
definitions are in [ds4_kvstore.h](../src/ds4_kvstore.h) and
[ds4_kvstore.c](../src/kvstore.zig); model-specific payload handling is in
[ds4.c](../src/ds4.c).
