# Aggregate handwritten server qualification

At9d0d96c0, native Renderer/JSON/Reply/OpenAI/Anthropic owners pass a real-model
HTTP screen against the retained C server-controller control. Both selections
pass22 checks and five Responses JSON-answer checks. Four of the22 checks are
additional generated-answer cases through OpenAI and Anthropic, each streaming
and nonstreaming. Both return the exact expected JSON and protocol terminators.
The other18 checks reuse the existing cumulative HTTP qualifier: two actual
overlapping clients/coordinator decode, disconnect during generation with a
surviving request, byte-exact durable GET, distinct-process restart, parent and
Conversation continuation, item CRUD and tombstones. All expected fact answers
are correct. Original qualifier credit/source: ../zig-server-20261022/qualify_http.py.

The retained control is .workspaces/zig-server-renderer-20261022/control/bin/
ds4-server, with native engine/session/support but C server render/JSON/reply/
OpenAI/Anthropic controllers. The candidate is the Item75e native binary. Each
runs separate fresh SSD/Response stores. Exact binary hashes, commands/settings,
requests, raw wire, trace and answers are retained in evidence/. No cache/model
or executable files are checked in. The active server/Metal/build source hashes
match before and after the entire campaign. No source changes or compiler jobs
ran during it; agents prepared ignored source drafts only.

C phases46.536/9.954s; native47.457/10.197s. These are one ordered functional
pair, not a controlled latency estimate. Existing question facts/outputs match;
we make no speed claim. Both guard runs remain at normal pressure (level1),
with unchanged0.007265625GiB swap and clean child exits. Runtime settings are
exact F32, no suffix/replay/resident approximation, two batched sessions,
32GBexpert cache, ctx8192/chunk512 and disk-only512MiB checkpoint cache.

```
python3 -B speed-bench/ideas/pressure_run.py .workspaces/http-repeat/guard -- \
  python3 -B speed-bench/experiments/zig-server-protocols-http-20261022/qualify_http.py \
  --server .workspaces/zig-server-anthropic-20261022/native/bin/ds4-server \
  --output .workspaces/http-repeat/run
```

Model defaults to the1017 IQ2_XXS/Q2_K imatrix GGUF. Use a fresh output path.
Integration/runtime screen17:37–17:40UTC; server materialization/generation/
routing and other inventoried C owners remain subsequent work.
