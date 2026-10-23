# Stateful Responses WebSocket qualification

Base: `3d64d468` on `zig-rewrite`, implementation change `vtuvrqyqlmxt`.
Main remains `ec5424de5562`. Native DS4.1
Metal, local M5 Max, 1017 IQ2_XXS/Q2_K imatrix weights. No numerical kernels or
checkpoint formats changed.

## Implementation and scope

- Native RFC6455 upgrade/framing, sequential `response.create` turns and
  concurrent sockets over the existing scheduler.
- Direct semantic-event sink shared with HTTP; complete current response
  envelopes, matching SSE event names and monotonic sequence numbers.
- Latest connection-local private replay for `store=false`, failed-parent
  eviction, and durable Responses/conversation reconnect/restart.
- Explicit callback retirement and disconnect cancellation. Ephemeral IDs also
  identify live approximate-state lineage without writing response-store files.

Reference: [OpenResponses 2026-04-24](https://www.openresponses.org/specification),
checkout `openresponses/openresponses` at
`8907bc3b7c4300ff5c21fd647cfde9929243afa1`. That version requires one in-flight
response per socket. Newer OpenAI multiplexing/warmup/steering extensions are
not implemented. See [usage and limits](../../../src/serving/RESPONSES.md).

## Results

| Check | Result |
| --- | --- |
| Framing fixture, ReleaseFast + Debug | 5 cases each pass |
| Native request/model fixture, ReleaseFast + Debug | 5 cases each pass |
| Existing server suite, ReleaseFast + Debug | pass, including ephemeral lineage |
| Real WebSocket integration, ReleaseFast | 26/26 required checks |
| Real WebSocket integration, Debug | 26/26 required checks |
| Official OpenResponses selected acceptance tests | 10/10 pass |
| Ordinary stable tool continuation | correct file facts; 292/317 prompt tokens cached |

Integration uses standard `websockets` 15.0.1 and official `openai` Python 3.37.0.
It exercises fragmentation/ping, event consistency, ephemeral continuation and
eviction, errors/recovery, two concurrent sockets, actual function-call storage
retrieval, durable reconnect/restart, disconnect with a surviving client, the
official SDK and clean server shutdown. All model jobs run serially under the
existing pressure guard; every server exits cleanly without forced kill.

Model-answer quality is recorded separately: 12/15 answer observations pass in
each full run. Three cancellation probes repeat one prompt requesting
`{"survived":true}`; ordinary control, cancellation survivor and fresh retry all
return `{"surived":false}`. The independent retained C server returns exactly
the same error with the same 33 input / 6 output tokens. This is a pre-existing
answer-quality limitation for this prompt, not evidence of cancellation damage.
All archive facts, concurrent-conversation facts and restart answers are correct.

The ordinary tool probe keeps instructions/schemas stable and reads a real
record absent from the prompt. It reuses 292 of 317 input tokens, prefills 25 and
returns the correct record. The broader integration changes
instructions between tool turns and observes zero cached tokens there. These
are favorable and changed-prefix observations, not a whole-agent speedup claim.

The stock official CLI launches all selected tests concurrently with a 30 s
timeout. Its initial run passed 3/10; seven queued requests timed out on the
two-slot server. Running the unchanged official cases serially with 180 s
timeouts passes 10/10 (individual durations 1 ms–20.1 s). The selected cases are
basic response, HTTP streaming, all six base WebSocket tests, function calling
and multi-turn. This is not a claim that every optional OpenResponses feature
passes the full acceptance suite.

## Reproduce

```sh
tools/zig-build --prefix .workspaces/responses-websocket-20261023/native \
  websocket-frames-test serving-boundary-test server-test ds4-server
.workspaces/responses-websocket-20261023/native/bin/test-websocket-frames
.workspaces/responses-websocket-20261023/native/bin/test-serving-boundary
.workspaces/responses-websocket-20261023/native/bin/ds4_test --server

python3 -m venv .workspaces/responses-ws-venv
.workspaces/responses-ws-venv/bin/python -m pip install \
  websockets==15.0.1 'openai[realtime]==3.37.0'
python3 -B speed-bench/ideas/pressure_run.py FRESH_GUARD_DIRECTORY -- \
  .workspaces/responses-ws-venv/bin/python -B tests/test_responses_websocket.py \
  --server .workspaces/responses-websocket-20261023/native/bin/ds4-server \
  --output FRESH_RESULT_DIRECTORY --sdk
```

Repeat the build with `-Doptimize=Debug --prefix .../debug` and use that server
for the Debug campaign. The integration script configures two batched slots,
ctx8192/chunk512/mixed quantum512, 32GB expert cache, draft1, F32 KV, 512MiB
disk-only state cache and all approximation features off.

For official acceptance, run `bin/compliance-test.ts` from the reference checkout
against a running server with `--base-url http://HOST:PORT/v1 --api-key local
--model deepseek-v4.1-flash --filter TEST_ID --json`, one selected test at a time.
The recorded serial wrapper invokes the unmodified exported `runAllTests` with
one template and `timeoutMs:180000`. Raw commands, responses, first failures,
pressure samples and the C control remain under
`.workspaces/responses-websocket-20261023/`; the reference checkout is also
ignored. No generated dumps are added to source control.
