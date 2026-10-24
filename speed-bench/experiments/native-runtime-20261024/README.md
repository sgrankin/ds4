# Shared Metal runtime migrations

Baseline: qualified model port `e37e53d9` (`qwen-glm-zig`). Work proceeds on
`native-runtime`; main remains unchanged. The [ordered plan](../../ideas/next-experiments.md)
uses one-hour ceilings including qualification. Existing model arithmetic,
formats, cache policy and thread behavior stay fixed while ownership moves.
Raw binaries, original sources, commands and logs are ignored under
`.workspaces/runtime-migrations-20261024/`.

## 1. Qwen model preparation — complete

Completed within the first one-hour budget, starting at about 10:33 UTC on
2026-10-24. The pure `src/models/qwen38/prompt.zig` owns full prompts, live tool
tails, schema preambles, ordered/raw tool calls, assistant echo and checkpoint
suffix delimiters. `src/models/tool_schema.zig` uses the standard JSON scanner
while retaining nested JSON spelling, field order and duplicate keys.

`Model.prepare` and the normal server adapter select that owner directly.
Canonical conversion borrows the actual request/tool records; the model imports
only native serving values. The server's duplicate Qwen formatter and old C
formatting bodies are removed. A small fixture-only assistant adapter preserves
the existing independent tests. Common preparation options no longer use the
DeepSeek-specific options type.

Qualification:

- Two focused Qwen CPU tests plus the shared schema test pass with allocator
  leak checking; pure serving-boundary assembly passes six cases.
- The unchanged existing server fixture passes its exact prompt/tool/reasoning,
  sampled raw replay, parallel argument ordering and visible/live checkpoint
  boundaries. The frozen fixture passes too.
- Real Qwen Responses/WebSocket qualification passes all 25 protocol and
  quality checks, including tool execution, concurrent conversations and
  response/conversation recovery after restart. The pressure guard completes
  normally. Existing downloaded Q4 weights are reused.

Reproduce with `nu tools/zig-build ds4-server server-test serving-boundary-test`,
then `zig-out/bin/ds4_test --server` and `zig-out/bin/test-serving-boundary`.
The real-model check uses the existing `tests/test_responses_websocket.py`
with `--family qwen`, under `speed-bench/ideas/pressure_run.py`, using the local
Python environment with its standard WebSocket client installed.
Evidence: `baseline/server-test.log`, `slice1-{build-final,server-final,boundary}.log`,
`slice1-qwen-responses/`, and `slice1-ws-guard/` beneath the workspace above.

## 2. GLM model preparation — complete

Completed in its next one-hour window, with final real-model results available
by 11:09 UTC. `src/models/glm53/prompt.zig` owns the GLM full/live template,
schema rendering, ordered/duplicate arguments, raw sampled calls, grouped
observations and reasoning retention. Both GLM53 and the compatible older GLM
template select this owner; older numerical inference remains a separate path.
Server duplicate GLM schema/call algorithms and C formatting bodies are removed.

Six pure GLM tests, seven native preparation assembly cases and all existing
exact server fixtures pass. Real GLM Responses/WebSocket qualification passes
all 25 protocol/quality checks, including tools, overlapping conversations and
restart recovery. This uses the existing Q2 model, SSD24GB and bounded per-layer
mapping, with the unchanged pressure guard. Evidence: `slice2-{build-final,
server-final,boundary}.log`, `slice2-glm-responses/` and `slice2-ws-guard/`.

## 3. Responses publication/integration — complete

The native publisher owns conversation publication, ephemeral IDs, response/replay
JSON envelopes and durable atomic store writes. Generic object rendering and
legacy prompt-field parsing now call native owners directly; old C algorithms
are removed, with fixture-only adapters retained. The on-disk envelope and
publication order are unchanged.

Existing server and response-store fixtures pass. All 25 real Qwen WebSocket
protocol/quality checks pass, including store=false isolation, concurrent
sessions, disconnects and restart recovery. Evidence: `slice3-{build,server,store}.log`,
`slice3-qwen-responses/` and `slice3-ws-guard/`.

Later slices remain pending integration/qualification. Payload I/O,
snapshot-envelope, selected-worker and common-session preparations are retained
separately until their preceding slices qualify.
