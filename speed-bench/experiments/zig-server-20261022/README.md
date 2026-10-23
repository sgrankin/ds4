# Native server ownership

This cumulative report starts with Item 64, the HTTP transport owner. It will
record each server slice only after its bounded qualification. HTTP transport,
request/message/Responses normalization and tool-memory ownership have passed
the existing ReleaseFast/Debug model-free fixtures, as have the coordinator
and retained C-server controls. The corrected model-backed integration check passed all eighteen structural
checks and all five generated-answer checks. The original failed probe is
retained; its incorrect DELETE-response expectation was fixed without changing
the server binary.

## Retained control

The control uses the original C server implementation with the previously
qualified native engine and support modules. `-Dnative=true` in its recorded
build commands does not mean that the server body was native at this point.
The frozen controls are under
`.workspaces/zig-server-input-20261022/control/bin` and `control-debug/bin`.

| Control step | Whole process | Result |
| --- | ---: | --- |
| ReleaseFast install and server fixture build | 0.565 s | Exit 0 |
| Debug server fixture build | 0.620 s | Exit 0 |
| ReleaseFast `ds4_test --server` | 0.136 s | Exit 0 |
| Debug `ds4_test --server` | 0.175 s | Exit 0 |

These are cached development builds and existing CPU fixtures, not clean-build
or server-runtime performance measurements. The exact commands and raw logs
are retained under [evidence/control/](evidence/control/), with a compact
[summary](summary.json). Both build stdout/stderr files are empty; the fixture
output and expected diagnostics remain unmodified in the evidence.

The existing production-linked fixture is built using the selected real server
implementation and support objects, then invoked without a model:

```sh
tools/zig-build -Dnative=true -Doptimize=ReleaseFast \
  --prefix .workspaces/zig-server-input-20261022/control server-test install
.workspaces/zig-server-input-20261022/control/bin/ds4_test --server
tools/zig-build -Dnative=true -Doptimize=Debug \
  --prefix .workspaces/zig-server-input-20261022/control-debug server-test
.workspaces/zig-server-input-20261022/control-debug/bin/ds4_test --server
```

Keep these binaries before activating the candidate server owner; rebuilding
these commands after activation would select the new body. Root owns all
active integration, serial compiler work and guarded model/GPU qualification.
No candidate success, quality or latency claim is made from this control step.


## Item 64: HTTP transport

The native [HTTP owner](../../../src/server/http.zig) owns one-request framing,
its temporary request/body buffers, response/SSE formatting, progress deadlines
and socket configuration/disconnect policy. The shared
[request record and API](../../../ds4_server_http.h) preserve libc body ownership
and the bounded method/path grammar. The C accept loop retains descriptor
close, signal and thread ownership; endpoint JSON, model inference and queueing
remain separate. This stage is not a complete native server claim.

The [first native build](evidence/http/http-build.stderr) failed because Zig's
translated `htons` helper has an `anytype` parameter, so `@truncate` could not
infer its destination width. Explicit `u16` narrowing preserves the original C
argument conversion and resolved the compile error; no protocol policy changed.
That failure and the successful second build remain in the evidence.

| HTTP step | Whole process | Result |
| --- | ---: | --- |
| Initial ReleaseFast build | 6.322 s | Compile failure, retained |
| Corrected ReleaseFast install and server fixture build | 0.730 s | Exit 0 |
| Debug server fixture build | 1.466 s | Exit 0 |
| ReleaseFast `ds4_test --server` | 0.141 s | Exit 0 |
| Debug `ds4_test --server` | 0.177 s | Exit 0 |

The selected [six-file source snapshot](evidence/http/http-source.json) is frozen
for these HTTP results. Later integration may change shared source/build files;
the snapshot describes this measured step rather than asserting that every
later cumulative source file remains byte-identical. Exact commands, build
logs and fixture output are retained under [evidence/http/](evidence/http/).
The successful tests reported `server: OK` and `ds4 tests: ok`; their diagnostic
about ignoring incomplete Qwen tool calls is retained unchanged. These cached
build and single CPU-fixture timings are not server throughput or latency
comparisons. The later cumulative model-backed screen below exercises the cohesive server
owners together.


## Item 65: request normalization and logical values

The native request owner now handles endpoint-field policy and lifetime,
OpenAI/Anthropic messages, Responses item order/reasoning/tool-result groups,
tool-schema values. It shares the actual
server records through [the input header](../../../ds4_server_input.h), uses
libc-owned allocations, and retains the existing primitive C JSON grammar and
image-resource services. C still materializes durable history, live bindings,
rendered prompts and tokenization; the native request code does not import
engine/session layouts or reparse through a second JSON DOM. Exact sampled
tool-text cache ownership is qualified separately in Item 67 below.

The [first build](evidence/input/input-build.stderr) reported fourteen Zig
0.17 API/type errors. Corrections replaced removed `dupeZ` with
`dupeSentinel`, adopted existing buffers through standard `ArrayList.empty`,
made libc double-pointer frees explicit with `@ptrCast`, and renamed a local
that shadowed `properties`. The failed attempt is retained; the corrected
build and both production-linked fixtures passed.

| Request-input step | Whole process | Result |
| --- | ---: | --- |
| Initial ReleaseFast build | 5.735 s | Compile failure, retained |
| Corrected ReleaseFast install and server fixture build | 0.955 s | Exit 0 |
| Debug server fixture build | 1.625 s | Exit 0 |
| ReleaseFast `ds4_test --server` | 0.139 s | Exit 0 |
| Debug `ds4_test --server` | 0.187 s | Exit 0 |

The [fifteen-file selected snapshot](evidence/input/input-source.json), exact
commands and all raw logs are retained under [evidence/input/](evidence/input/).
These are existing CPU-fixture and cached-build observations, not real model
or request-throughput results.

## Coordinator preparation

The first integration preparation stopped when a strict regex expected one
`parse_tools_value` definition but found both the native adapter and C fallback.
The subsequent [records-only build](evidence/coordinator/coordinator-records-only-build.json)
completed in 5.948 s while the coordinator implementation was still C. Its
[preparation note](evidence/coordinator/coordinator-preparation.txt) and warning
log are retained to make that boundary explicit. It is not counted as a native
coordinator build or passing coordinator qualification. Actual native
coordinator results are recorded separately below.


## Item 66: resident inference coordinator

The native [coordinator](../../../src/server/coordinator.zig) owns pending versus
in-flight admission, per-slot generation/evaluation, decode coalescing, actual
session-batch grouping and worker completion/shutdown. It uses the one shared
[Server/Slot/Job record layout](../../../ds4_server_runtime.h), borrows the
existing mutex/condition-variable ownership and calls public engine/session
APIs. The HTTP accept loop and endpoint/request materialization remain outside
this owner.

The first [actual native build](evidence/coordinator/coordinator-build.stderr)
failed on three local/parameter names shadowing `stop` or `quantum`. Renaming
those bindings resolved the errors. The C runtime header also uses a
`struct rax` forward declaration rather than introducing a duplicate C99
typedef. This native build is distinct from the earlier records-only C build.

| Coordinator step | Whole process | Result |
| --- | ---: | --- |
| Initial native ReleaseFast build | 5.747 s | Compile failure, retained |
| Corrected native ReleaseFast install/fixture build | 4.261 s | Exit 0 |
| Native Debug fixture build | 1.476 s | Exit 0 |
| ReleaseFast `ds4_test --server` | 0.134 s | Exit 0 |
| Debug `ds4_test --server` | 0.185 s | Exit 0 |

The [six-file snapshot](evidence/coordinator/coordinator-source.json) and raw
build/fixture logs retain this step's measured source and results. These CPU
fixtures do not replace a real concurrent generation/cancellation check.

## Item 67: sampled tool-text memory

The [native tool-memory owner](../../../src/server/tool_memory.zig) borrows the
actual cache record and server mutex. It keeps the sampled DSML tool text
associated with protocol IDs so a client rebuilding/reordering argument JSON
can replay the original model text. Cache allocation, bounded eviction,
lookups, store/resolve operations and cleanup are owned here; live request and
model policy remain outside it. It uses the actual
[shared records](../../../ds4_server_tool_memory.h), not copied server layouts.

| Tool-memory step | Whole process | Result |
| --- | ---: | --- |
| ReleaseFast install and server fixture build | 5.881 s | Exit 0 |
| Debug server fixture build | 1.455 s | Exit 0 |
| ReleaseFast `ds4_test --server` | 0.139 s | Exit 0 |
| Debug `ds4_test --server` | 0.185 s | Exit 0 |

The [eight-file source snapshot](evidence/tool-memory/tool-memory-source.json)
and all build/test output are retained under
[evidence/tool-memory/](evidence/tool-memory/). Together, the four server slices
and retained control have ten successful ReleaseFast/Debug invocations of the
existing production-linked fixture. This count refers to fixture executions,
not ten independent model workloads or measured test-branch coverage.

## Published model-backed screen

[qualify_http.py](qualify_http.py) is the self-contained serial two-process
workflow prepared for root's existing pressure guard. It uses two concurrent
Responses clients, a real in-generation SSE disconnect with a surviving
client, exact stored-response retrieval, process restart, both
`previous_response_id` and conversation references, then durable item/response
CRUD. Five generated JSON answers are graded against fixed facts. The corrected
guarded workflow completed successfully as recorded below.
The four small save/request/text/stop helpers are copied verbatim from the
prior durable-response screen; there is no import of an ignored workspace
module and no new harness library.


## All-C fallback

The explicit `-Dnative=false` reference also passed ReleaseFast and Debug
`ds4_test --server`. The [four commands and raw logs](evidence/fallback/) retain
its actual C engine/support/server selection: ReleaseFast build 9.380 s, Debug
build 0.583 s, fixture executions 0.137/0.176 s. These two additional CPU runs
bring this cumulative campaign to twelve successful existing-fixture
invocations. Neither build time is a matched performance result.

## First real HTTP attempt: failed probe assumption

The first [executed script](evidence/http-first/qualify_http.py) and
[unaltered result](evidence/http-first/result.json) are preserved as a failed
attempt. Both server processes exited normally; all five generated JSON
answers and seventeen of eighteen structural checks passed. The run observed
real two-row decode batches, an in-generation SSE disconnect with a surviving
request, exact durable-response/conversation retrieval after process restart,
and actual disk-prefix-cache hits. The guard completed with child return code
3 because the probe's `item_deleted` check was false.

The probe expected an item DELETE to return `deleted:true`. The existing
[`ds4_conversations.h`](../../../ds4_conversations.h) instead removes that
item, persists the changed conversation and returns its conversation object.
The [raw DELETE response](evidence/http-first/after/item-delete.response.json)
contains that existing response shape. This is a probe-contract error; no
runtime code was changed. The published probe now requires the returned
conversation ID/object and performs a GET of the same deleted item expecting
404. The fresh corrected rerun passed as recorded below. The original result is
not regraded.

Both process logs, traces, wire requests/responses and commands are retained
under [evidence/http-first/](evidence/http-first/). Cache chunks, store backing
files and weights are deliberately excluded. The compact
[pressure summary](evidence/http-first/guard/pressure-summary.json) and original
guard result are retained instead of the full polling stream. These are
functional observations, not a C/native latency comparison.


## Corrected cumulative HTTP qualification

The [fresh corrected run](evidence/http-pass/result.json) passed all eighteen
structural checks and all five generated-answer checks. It used exactly the
same frozen server executable as the first attempt, SHA-256
`cef691c7137f42aee750f245783bbc40f2bf73a995e5bcf5d8496a6b4ca48ec2`,
in all four processes across both attempts. No runtime changes were made
between those model runs. Later source-only preparation for another unit does
not alter these frozen binary identities or the stage-specific source snapshots.

The first process accepted two overlapping Responses requests and observed
actual two-row decode batches. Both answers preserved their distinct fixed
facts and the requested audit array. A separate SSE request emitted a real
text delta before the client closed; the trace recorded cancellation after
one generated token. The surviving client completed correctly. The process
then exited normally. A second process restored exact stored responses and
conversation items, continued through `previous_response_id` and a conversation
reference, and produced both correct remembered-fact answers. Its log contains
two exact disk-prefix hits of 478 tokens with memory-cache budget zero. The
item DELETE returned the existing Conversation object; a subsequent
[same-item GET](evidence/http-pass/after/item-missing.result.json) returned 404.
Response deletion and conversation/item CRUD also passed.

| Generated response | Output tokens | Input / cached tokens | Grade |
| --- | ---: | ---: | --- |
| Initial A: IVY-46, count 37, audit 0..15 | 47 | 478 / 0 | Pass |
| Initial B: CORAL-82, count 231, audit 0..15 | 47 | 478 / 0 | Pass |
| B survives other client's cancellation | 12 | 541 / 525 | Pass |
| A continues after restart | 12 | 542 / 478 | Pass |
| B conversation continues after restart | 12 | 571 / 478 | Pass |

All five were completed responses with zero reasoning tokens and below their
128-token caps. The two process phases took 42.906 and 10.431 s. This is a
functional integration measurement of the native package, not an uncontended
C/native timing comparison or broad quality benchmark.

The run used the 1017 IQ2_XXS/Q2_K target, 32 GB SSD expert-cache target, context
8,192, prefill chunk/mixed quantum 512, two batched-session slots, MTP width 1,
temperature zero and no reasoning. Checkpoint memory-cache budget was zero,
disk-cache budget 512 MB; suffix reuse, decoder replay, compressed-KV F16 and
resident-route approximations were off. Actual commands, environment and
binary hashes are recorded in each process's `execution.json`.

To reproduce after building this selected server, use a fresh output and the
existing guard:

```sh
python3 -B speed-bench/ideas/pressure_run.py .workspaces/zig-server-http-rerun-guard -- \
  python3 -B speed-bench/experiments/zig-server-20261022/qualify_http.py \
    --server .workspaces/zig-server-input-20261022/native/bin/ds4-server \
    --output .workspaces/zig-server-http-rerun
```

The [successful guard](evidence/http-pass/guard/result.json) completed with
child return code zero. Its [52 samples](evidence/http-pass/guard/pressure-summary.json)
all had normal pressure and zero swap growth, with peak wired/compressed memory
42.125/10.365 GiB. Together with the preserved first attempt, all 105 model-run
pressure samples were normal with zero swap growth. No cache chunks, backing
store directories, weights or executable binaries are checked into this report.

The qualified scope is native HTTP transport, request/message/Responses values,
resident coordinator and sampled tool-text memory on the retained engine.
C still owns the accept loop, endpoint materialization and other unported
server policy. Responses output serialization is a separate follow-up, not
claimed as part of Items 64–67.
