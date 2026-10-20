# Installed client smoke

The installed official OpenAI Node SDK **6.26.0**, installed Pi provider **0.83.0**, and SDK Conversations calls all passed against the local server. `client-first/checks/result.json` records `pass: true`, `errors: []`; the launcher returned 0 and the server shut down with return code 0. These are installed versions, not a claim about the latest releases.

One server lifetime used Q2, context 8192, chunk 512, batch 1, requested expert cache 64GB, selected tile 64/limit 1024, resident gap 0 and suffix reuse off. Both clients requested temperature 0, reasoning effort none and at most 256 output tokens. Actual owned files supplied the answers; no read request could access another path.

| Client | Verified flow | Actual answer | Two request seconds |
|---|---|---|---:|
| Official SDK | `responses.stream()` → `finalResponse()` → real read → `previous_response_id` + matching `function_call_output` → final response → GET both stored IDs | `LIME-27` | 26.7823 |
| Pi provider | Streamed tool parser → real read → full-input replay with parsed tool result → streamed final text | `COVE-42` | 22.2454 |

The SDK read stream contained one each of `response.created`, `output_item.added`, `function_call_arguments.delta`, `function_call_arguments.done`, `output_item.done` and `response.completed`. The answer stream also contained content-part and text events. Stored IDs and output fields matched the streamed final responses after removing only SDK-added `parsed: null` / `parsed_arguments: null` annotations from the comparison.

Pi produced `toolUse` and then `stop` / raw status `completed`. Its real provider handled the composite `call_id|fc_item_id` and replayed the correct wire `call_id`. Both requests used `store:false`, retained the system/tool definitions and sent `prompt_cache_key: "hour17-pi-read"`; neither used `previous_response_id`. The second request reported 410 cached and 19 newly evaluated prompt tokens. The SDK follow-up intentionally omitted tools and had common prefix 1 with zero cached prompt tokens. The timings therefore do not compare client speed or caching policy fairly.

Without additional generation, the official SDK created a conversation, appended a text message, listed and retrieved it, retrieved conversation metadata, deleted the item, confirmed an empty list and deleted the conversation. The returned message had an ID, completed status and typed `input_text` content; the final object was `conversation.deleted` with `deleted:true`.

This establishes the exercised text/function-tool SSE, stored response retrieval and essential Conversations CRUD subset through independent installed clients. It does not qualify the complete API schema, hosted/multimodal tools, required/named tool choice, restart durability in this run, or a full Pi coding-agent workflow. Reasoning was disabled, so this smoke does not test reasoning accounting. It also does not test suffix reuse or establish broad task quality.

Evidence: `client-first/checks/result.json`, `wire-*-request.json`, `wire-*-response.txt`, SDK/Pi `*-events.jsonl`, payloads and final responses. `client-first/execution.json` records server settings; `client-first/result.json` records the exact Node command and actual return codes. No retry or repair was used.
