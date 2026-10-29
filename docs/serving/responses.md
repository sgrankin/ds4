# Stateful Responses serving

The native Metal server exposes HTTP and WebSocket Responses at `/v1/responses`.
Both transports use the same request materialization, model preparation,
inference queue and semantic output events. New response envelopes follow the
[OpenResponses 2026-04-24 specification](https://www.openresponses.org/specification).
This implementation qualifies its text/function-tool subset, not every optional
OpenResponses capability.

## Run

```sh
./tools/zig-build ds4-server
./ds4-local --bin-dir ./zig-out/bin --server persistent-fast --model MODEL.gguf
```

The preset configures the existing response store and RAM/SSD model cache.
All server presets default to localhost, two active slots and a durable response
store at `.workspaces/response-store`. Later CLI flags override slot/address
defaults; `DS4_SERVER_RESPONSE_STORE_DIR` selects another store directory.
It also selects its existing approximation settings; protocol qualification used
F32 KV with approximation features disabled. For a manually launched server,
set `DS4_SERVER_RESPONSE_STORE_DIR` to enable stored response/conversation
recovery across reconnects and server restarts.

Connect a standard WebSocket client to `ws://127.0.0.1:8000/v1/responses` and send:

```json
{"type":"response.create","model":"deepseek-v4.1-flash","store":false,"reasoning":{"effort":"none"},"input":"Remember code=EMBER-73. Reply OK."}
```

Read Responses events until `response.completed`, then send the next turn using
that event's `response.id`:

```json
{"type":"response.create","model":"deepseek-v4.1-flash","store":false,"previous_response_id":"resp_FROM_COMPLETED_EVENT","reasoning":{"effort":"none"},"input":"What code did I ask you to remember?"}
```

Only new input is needed. Function results use the normal
`function_call_output` item with the returned `call_id`. Resend current
instructions/tools when required; top-level instructions are not inherited
through `previous_response_id`. Keeping instructions and schemas stable also
helps model-prefix reuse.

## State and concurrency

- One response is processed at a time on each connection. One next turn can
  queue while the current job retires. Use separate sockets for concurrent
  conversations; they share the loaded model and existing scheduler/slots.
- `store=false` retains the latest completed text/function response and private
  logical replay in connection-local memory. It creates no response-store file.
  Reconnect requires full input replay unless the parent was durably stored.
  The separately configured model/KV SSD cache retains its existing policy.
- `store=true` supports durable `previous_response_id` recovery. Existing
  Conversations API references also work, including after server restart.
  `conversation` and `previous_response_id` are mutually exclusive.
- Failed continuations evict the referenced connection-local parent. An
  unavailable parent produces `previous_response_not_found`; start a new chain
  with full input rather than silently continuing with missing history.
- Closing the socket cancels its active/queued work and waits for callback
  retirement. Other clients continue. Transport pings, fragmentation and clean
  close are supported. Connections expire after 60 minutes with
  `websocket_connection_limit_reached`.

OpenResponses requires sequential sockets. Newer OpenAI `stream_id`
multiplexing, `generate:false` warmup and client `response.cancel`/steering events
are outside this version and return explicit unsupported-feature errors. Do not
send HTTP-only `stream`, `stream_options` or `background` fields in a WebSocket
create message. Standalone compaction and hosted execution are unsupported.
DS41/Flint structured output and function-tool constraints use the shared
[request implementation](sampling-and-speculation.md#constrained-output-and-client-tools)
on HTTP and WebSocket alike. Stored/connection-local
continuation currently targets text and function tools; image continuation is
not qualified.

## Ownership

`server/websocket_frames.zig` owns RFC6455 framing. `server/websocket.zig` owns
the connection, its latest parent and an optional queued input. It submits the
ordinary server job with a borrowed event sink. The same output owner emits
JSON events to WebSocket or frames them as HTTP SSE; there is no internal HTTP
proxy. A job must finish before its request or sink can be freed. Immutable
response IDs also identify live approximate-state lineage without requiring a
durable response file.

See the [qualification report](../../speed-bench/experiments/responses-websocket-20261023/README.md)
and [journal](../lab/README.md). Raw runs remain ignored under `.workspaces/`.

## Editing client-owned history

For an edited working view, a client can send full `input` with `store: false` and a stable `prompt_cache_key` for that history. Preserve tool call IDs, actual tool results and unchanged instructions. The key scopes cache discovery; it does not make changed prefixes equivalent. Use `previous_response_id` to append to a server-owned chain; submit full input or edit Conversation items to change its history. Conversations support create/get/delete and item append/list/get/delete; stored Responses support retrieval and deletion. See [checkpoint storage](../storage/session-cache.md) for exact-prefix and approximate suffix reuse.

For native DS4.1 text, a caller can set `ds4_reasoning_budget` on an individual Responses request along with `reasoning.effort`. Zero or omission leaves thinking unrestricted. A nonzero value is a soft, once-per-request closure, not automatic task-stage detection: parser recovery or later reopening may exceed it, the forced closure counts against the output cap, and it disables MTP for that request. Keep instructions and tool definitions stable across turns to preserve prefix reuse. The [thinking screen](../../speed-bench/experiments/server-thinking-hour16-20261004/README.md) and [stable-instructions screen](../../speed-bench/experiments/stable-instructions-hour18-20261004/README.md) document their workload-specific limits.
