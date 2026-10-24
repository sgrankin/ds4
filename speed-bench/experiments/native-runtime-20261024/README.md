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

## 4. Generic payload I/O — complete

Native modules own bounded FILE copies, little-endian words, tensor transfers
and F16/F32 wire conversion. FILE-only exports stay separate from GPU exports,
so CPU fixtures cannot extract a second runtime. The focused production-linked
fixture checks all 65,536 half encodings and 1,143,805 float encodings against
the prior ARM conversions, plus partial transfer/accounting and staged copies.

The full host/script suite passes. Both Qwen and GLM pass frozen → native →
frozen canonical payload exchange at positions 1/3/4/5/129: every wire byte and
fresh continuation logit matches, with invalid/truncated state checks retained.
Both guards complete normally. Build/run `payload-io-test` / `test-payload-io`;
the maintained `payload_exchange.nu` uses the frozen and candidate helpers.
Evidence: `slice4-{build,host}.log`, `slice4-{qwen,glm}-payload/` and guards.

## 5. Snapshot/staged-file envelopes — complete

Native stdio ownership covers snapshot resize/save/load/free and staged-file
create/save/flush/measure/copy/unlink. Public APIs borrow the opaque session;
model codecs retain wire ordering and frontier mutation. Internal DS41 suffix
capture needs the same envelope, so `snapshot_buffer.zig` shares the concrete
memory-stream rules with thin checkpoint adapters. Unused DS41 C exports and
its duplicate staging implementation are removed.

Focused snapshot lifetime/publication checks and the full host suite pass.
Qwen and GLM payload/fresh-logit checks pass against frozen state. The frozen
`e37e53d9` source built the DS41 control helper; all eight existing F32/F16,
return-import and cross-storage campaigns pass at 1/2/3/127/128/129. Same-storage
wire and fresh logits remain identical; cross-storage rounding is reported.
Evidence: `slice5-{build,envelope,host}.log`, `slice5-{qwen,glm,ds41}-payload/`
and guards. Build/run `session-snapshot-test` / `test-session-snapshot`; the
existing canonical `qualify.py` reproduces the DS41 matrix.

## 6. Selected-expert worker — complete

The native process worker owns start, one mailbox, event waiting, ID readback,
validation, expert loads and consumed completion. Actual job/pthread layouts
remain single-source; count inputs are captured at submission. Non-Apple legacy
workers retain their separate compatibility branch.

The real-pthread CPU fixture passes busy, completed-but-unconsumed, event/read/ID,
cache-load/override failure and reuse checks. Actual DS41 Metal execution with
`DS4_METAL_V41_ASYNC_EXPERT_LOAD=1` passes canonical/fresh continuation comparisons
against the frozen control at 1/2/3/127/128/129, including failed-restore recovery.
This flag reaches the moved async path; ordinary defaults alone would not.
Evidence: `slice6-{build,worker}.log`, `slice6-ds41-async/` and `slice6-async-guard/`.
Build/run `stream-selected-test` / `test-stream-selected`; the real model uses
the same `payload-roundtrip` helper with that environment setting.

## 7. Page-in and view handoffs — complete

Native code owns bounded pread, page touches, madvise/readahead, range partition,
selected-ID run coalescing, pthread start/join, occupied slots and model-view
array handoff. DS41 reuses the same pread leaf; no I/O sorting policy is added.
A temporary C leaf selects family spans; slice 10 removes it with explicit
shape/resident-policy inputs.

The real-file/thread fixture and full host suite pass. DS41 and GLM canonical
bytes/fresh logits match frozen controls. Profiled 256-token model prefills reach
all 40 DS41 and all 45 GLM layer preads, with `ok=1`; short state tests alone do
not enter the full-layer preparation path. Other hint modes use real OS I/O in
the focused fixture. Evidence: `slice7-{build,pagein,host}.log`,
`slice7-{ds41,glm}-pread/`, `slice7-{ds41,glm}-prefill.*` and guards.
Build/run `pagein-test` / `test-pagein`; enable
`DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE=1` for the model check.

## 8. Common session routing/control — complete

One synchronous borrowed view exposes actual Core, Frontier, Control, Pending
and selected model state. It is not a second Session layout. Native code owns
ordinary sync/eval, image validation/publication, invalidation, callback scopes
and family payload selection. Common frontier/image helpers and control exports
are cold; CPU fixtures cannot extract the model controllers.

Full host checks pass. Qwen public `eval_argmax` now uses its actual graph (the
old branch omitted Qwen); four tokens and every vocabulary logit match separate
`eval`+`argmax`. GLM cancellation at 17/19 rejects partial state and rebuilds
154,880 logits exactly. Qwen/GLM/DS41 frozen payload and fresh-logit comparisons
pass, as do all 28 real Flint Responses protocol/quality checks. Image-conditioned
assembly remains part of combined qualification. Evidence: `slice8-build-final.log`,
`slice8-host.log`, `slice8-qwen-argmax.*`, `slice8-glm-cancel.*`,
`slice8-{qwen,glm,ds41}-payload/` and `slice8-flint-responses/`.

An audit found common batch validation/eligibility/publication still in C for
Qwen/GLM/Flint. Its model arithmetic is already native; a separate bounded
follow-up will move that policy after the actual pool exists (slice 9).

## 9. Session resources — complete

Native resource transactions compose existing model allocators and own common
host buffers, Qwen scratch sharing/pool slots and GLM live reservations. One
actual engine Pool replaces the old scattered members. C keeps only allocation
of its private conditional Session shell, address capture and publication.

Qwen's full n-gram/error recovery, ordinary/MTP batches and shared arena growth
pass. DS41 repeated small/wide allocations charge and release their exact sizes,
including failed-start cleanup. All 25 GLM Responses checks and a Flint short
prefill/decode/teardown pass. Independent review caught a stale transient reserve
captured before first-session cache fitting; the fix uses the fresh post-fit
value. A CPU probe verifies reserve 99→7 reaches admission and reservation16
returns to zero, and the corrected live GLM cancellation/recovery plus host
suite pass. Final combined serving checks will run on immutable final binaries.
Evidence: `slice9-build-fixed.log`, `slice9-host-fixed.log`, `slice9-qwen-state.*`,
`slice9-ds41-resources.*`, `slice9-glm-responses/`, `slice9-flint.*`,
`slice9-order-probe.log` and `slice9-glm-fixed.*`.

A bounded 8b follows now that the shared Pool exists, closing the discovered
common batch-policy gap before engine startup/lifecycle integration.

## 8b. Common batch policy — complete

Native shared batching owns membership/token/context/duplicate validation,
optimized model dispatch, logits/frontier publication, serialized fallbacks,
mixed-prefill order and all-member failure invalidation. It borrows the same
SessionView and actual Pool/QwenPolicy. DS41 retains its existing native batch
coordinator; numerical model controllers and cache policy are unchanged.

Independent review restored the old zero-rotary guard for GLM's dense batch
admission. Host checks, Qwen full ordinary/MTP/rejection/error-recovery/arena
fixture and all 28 Flint Responses checks pass. Resident GLM batched numerical
execution remains unmeasured with this SSD-host setup. Evidence:
`slice8b-{build-final,host}.log`, `slice8b-qwen.*`, `slice8b-flint-responses/`
and guards. An additional bounded continuation/rewind follow-up closes the
small remaining common C policy found in final routing review.

## 10. Engine admission/cache policy and shared spans — complete

Native policy owns GLM admission, first-session cache fitting, hotlist seed,
automatic/configured cache sizing and resident-layer choices. Shared model
spans own full/static/runtime/token/output builders; DS41 and generic page-in
reuse them. The temporary C span-selection callback is gone. Cold size/budget
exports stay separate from hot initialization.

Source differential checks match 50 original-C cache-policy cases, 3,240 span
vectors and 972 seven-predicate cases. Maintained span and real-file page-in
fixtures pass, as does the full host suite. Fully assembled `slice10-final`
GLM/DS41 state bytes and fresh continuation logits match frozen controls. Real
256-token runs complete every GLM45/DS4140 layer pread with matching decoded
control text. The DS41 span-field dependency was corrected during assembly.
Evidence: `slice10-build-qualified.log`, `slice10-host-final.log`,
`slice10-final-{spans,pagein}.log`, `slice10-final-{glm,ds41}-payload/` and
`slice10-{glm,ds41}-prefill.*`. Earlier partial-build attempts are not the
qualification binaries. `model-spans-test` joins the normal host driver.

Final routing review found additional actual common sampling, multimodal,
continuation/rewind and stateless-generation controllers. Those get separate
bounded migrations before combined qualification; existing model capabilities
and numerical algorithms remain fixed.

## 11. Native engine factory — complete

One native factory composes DS41, Flint, Qwen and GLM model/config/weight/vocab/
vision owners, options, cache admission, Metal setup/maps and retirement. The
actual Core embeds its selected immutable configuration once; session/resource
captures and basic metadata reads use it. C only allocates its private Engine
shell, captures actual component addresses and publishes legacy snapshots.

Review found early option failures could trigger global teardown before owning
anything. The factory now publishes the actual ownership flag only after lock
acquisition or Flint handle creation; unowned errors free only the shell. A real
Qwen fixture rejects two early requests while retaining the first engine and
then matches all logits from four eval_argmax calls.

All five programs build; the full host suite passes after isolating a pending
multimodal source draft from cold input linkage. DS41/GLM frozen canonical state
and fresh-logit checks pass, Flint prefill/decode/teardown passes, and all three
GGUF inspect modes return zero. Source review checks all 50 marker spellings and
family tokenizer choices. Evidence: `slice11-build-isolated.log`,
`slice11-host-final.log`, `slice11-qwen-owner.*`, `slice11-{glm,ds41}-payload/`,
`slice11-flint.*` and `slice11-inspect/results.json`. The immutable qualification
prefix is `slice11-final`; the small ownership regression is in the existing
Qwen n-gram fixture.

The remaining bounded ordinary controllers identified by route tracing are
still pending integration/qualification; source preparation is not completion.

Later slices remain pending integration/qualification. Payload I/O,
snapshot-envelope, selected-worker and common-session preparations are retained
separately until their preceding slices qualify.
