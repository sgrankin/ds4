# Serving project journal

Last updated: 2026-10-23. This is the compact handoff for resumed work and context
compactions. Keep the current state accurate and append short decision/progress
entries at meaningful checkpoints. Raw experiment output stays local.

## C frontend warning cleanup — 2026-10-23

Baseline `ec1dd845` compiled obsolete C-to-Zig migration facades and fixture-only
adapters into the production support translation units: 94 unused-function
warnings in `ds4_server.c` and 74 in `ds4_agent.c`. Removed 118 helpers with no
runtime or fixture callers. The remaining fixture adapters compile only for
their existing test macros, including agent fixtures that define only
`DS4_AGENT_TEST_NO_MAIN`. Removed the old unused attributes; production warning
flags remain enabled. Runtime bridges, Qwen/GLM rendering and shared prompt
bytes are retained.

Both support files compile without diagnostics under the production C flags
plus `-Werror=unused-function`. Before/after object exports and the `__text`,
`__cstring` and `__const` sections match exactly; this is source/build cleanup,
not an inference optimization. All five executables build cleanly with
`nu tools/zig-build ds4 ds4-server ds4-agent ds4-bench ds4-eval`.
`nu tests/verify.nu` passes, as does the model-free suffix transaction fixture;
the real-vocabulary live-context fixture also builds. No model job was needed.
Raw baseline copies, warning logs and checks remain ignored under
`.workspaces/c-warning-cleanup-20261023/`. Main is unchanged.

## Flint reasoning visibility fix — 2026-10-23

The user's dsh `openai-responses` profile received no thinking because our server
required `reasoning.summary` even though OpenResponses does not require that
hidden-by-default policy. Flint Responses now emits its generated reasoning
through the existing summary events/items by default, on HTTP and WebSocket.
An explicit `reasoning.summary: null` suppresses that text. Effort/thinking hints
are accepted and ignored, matching Flint's model-controlled reasoning policy;
the response envelope reports `reasoning.effort: null` instead of a fictitious
effort. DeepSeek visibility defaults are unchanged.

No regression checks were added or run, at the user's request. The obsolete
fixture assertion that Flint must reject effort was removed.
`nu tools/zig-build ds4-server` succeeded, and the root executable was replaced
atomically for `ds4-local` without interrupting the running server.
The already-running server needs restarting to pick up the new binary.

## Intended product

The user wants a focused local inference server for their M5 Max, with possible
M1 Max support. We follow upstream implementations and selectively copy or
reimplement relevant model changes, kernels and fixes, retaining attribution.
We do not need every upstream backend, frontend or deployment mode.

- One agent-facing protocol family: Responses, ideally including its WebSocket
  transport and incremental conversations. HTTP and WebSocket should use the
  same request handling and output events.
- Many independent conversations per engine. Durable history outlives sockets,
  active inference slots and cache eviction. Concurrent conversations need fair
  scheduling, cancellation and bounded prefill work.
- One process per loaded model is the preferred initial arrangement. Two models
  can run as two engines, optionally routed by model name. They share the host's
  physical RAM and SSD bandwidth; configure their budgets accordingly.
- Assemble a serving executable from protocol/conversation handling, scheduling,
  a model implementation, Metal operations and RAM/SSD storage. Start with normal
  linked modules and explicit composition; derive shared interfaces from actual
  implementations rather than speculative plugin requirements.
- Model implementations own tokenizer/prompt rendering, tool/reasoning and
  multimodal translation, execution, and saved-state semantics. Original DS4.1
  and 1017 can share family code with explicit variant configuration. Other
  possible implementations include Flint, Qwen and GLM; these are directions,
  not completed native backends.
- Learn from ds4-agent's context editing, archival, compaction and reuse behavior.
  Expose useful serving controls through explicit API extensions as needed so
  other agent harnesses can use them. Agents can execute their own tools.

## State and storage boundaries

Keep three concepts explicit:

1. Durable conversation items and response ancestry.
2. The selected/edited/compacted working context presented to a model.
3. Model inference state: KV, compression/recurrent state, pre-RoPE values and
   any other state needed for continuation.

Persistent-agent facts can leave working context while remaining in checkable
storage. Qualification must exercise actual archival and tool retrieval.
Approximate activations can preserve influence from an earlier prefix; rebuilding
from text may change that influence. Preserve checkpoint lineage and distinguish
checkpoint restoration from rebuilding a context.

Share mechanisms such as immutable byte storage, I/O, pinning, eviction and
accounting where useful. Each model owns capture/restore/relocation and validity
rules. Current DS4.1 checkpoint formats are not a universal per-token KV format.
Expert weights and conversation checkpoints have different residency policies.

USRBIO refers to 3FS's user-space ring I/O interface, not DeepInfer's IPC or GPU
communication rings. Its reusable lessons are batching, reusable buffers,
completion handling and overlapping I/O with computation. A single engine
process can start with an ordinary bounded queue and I/O workers; cross-process
shared-memory machinery is unnecessary for that arrangement. Measure submission,
copying and SSD wait costs before choosing a more elaborate queue.

## Candidate serving features — discussion 2026-10-23

Record these as a design palette: the user may want some, all, or a remix.
This discussion does not authorize implementation or select a scripting runtime,
memory database, or universal tool framework. Steering remains separately
deferred with its zero-speed-regression acceptance requirement.
Connectome-like context aging and CLM are reference ideas; no particular memory
system or agent harness is selected.

| Piece | Intended role and execution ownership |
| --- | --- |
| Server-side programmatic tool calling / code mode | Model-generated programs compose server-exposed capabilities within a response. OpenAI's current Responses API provides a precedent; our pinned OpenResponses core does not standardize it, but permits hosted extensions. Tool execution location remains explicit: a server-run program may still need results from client-owned tools. |
| Client-side composition, like `functions.exec` | The external agent composes its own tools in its own runtime. This exists in our current client tooling; it is not an implemented DS4 server capability. Keep it compatible with, rather than dependent on, server-side composition. |
| Direct tools and program bindings | Expose capabilities directly to the model and through code bindings when useful. Prefer the same names, schemas and result semantics in both surfaces; avoid divergent duplicate tool definitions. |
| Server-side MCP | Let the serving process connect to MCP servers, discover their tools and execute calls. Those tools could be available directly and from server-side programs, enabling server plugins without moving the agent frontend into the model process. Not implemented yet. |
| Explicit and automatic compaction | Support deliberate `/responses/compact` calls and separately consider request-configured thresholds and automatic/default policy. OpenResponses 2026-04-24 includes the standalone endpoint; OpenAI's current automatic `context_management` configuration goes beyond that pinned core. Our server implements neither compaction mode yet. |
| Specific CLM/context edits | Operate on the server-owned working view using stable turn/item IDs, preserving the full journal and archived evidence. Offer inspect/archive/replace/remove primitives; programs may compose them, but an individual edit can also be structured arguments without code. Apply changes at a safe inference boundary and return a receipt. |
| Cross-session memory | Keep useful facts, decisions and evidence in durable, checkable storage, available to later sessions through retrieval tools. This is distinct from response ancestry, prompt/KV caching and a compacted working context. No storage/index choice is selected. Qualification should retrieve an archived fact absent from the active prompt. |

### Candidate memory and thread tools

The user specifically suggested explicit memory, thread listing, archiving and
search. Web/browser capabilities can come through server-side MCP; concentrate
our own tools on state the serving process owns. Names below are sketches, not
an implemented API or a commitment to separate tool schemas.

| Capability | Useful contract |
| --- | --- |
| `memory.put/get/update/retract` | Curated cross-session notes: facts, decisions, constraints and evidence references, scoped to a project or user. Preserve source thread/item IDs and distinguish superseded notes from current ones. Start with explicit text records; indexing and automatic memory promotion remain undecided. |
| `threads.list/open` | Discover threads by title, project, recency or status; inspect a bounded outline or selected messages instead of dumping full history into the active prompt. |
| `recall.search/read` | Search notes, archived material and thread history with source filters; return small snippets and stable references, then fetch exact selected ranges. Keep search/read behavior consistent across these sources. |
| `threads.archive/unarchive` | Remove completed work from the active thread list while retaining its history and searchability. This differs from deleting data or removing turns from working context. |
| `context.inspect` | Report token budget, a bounded turn index and large context contributors so the model can decide what to retain or archive without rereading its entire context. |
| `context.archive/replace/remove` | Archive selected older turns durably, then reduce the working view to a summary and references. Return an edit receipt and resulting token usage; suffix selection/recomputation stays runtime policy. |
| `threads.fork` (later) | Explore an alternative from a selected turn without overwriting the original. Fork logical history and reuse KV opportunistically; do not imply that an exact approximate-state snapshot exists. |
| `threads.ask` (later) | Run a bounded question against another thread and return an answer with source references, reducing how much unrelated history enters the current model context. This would spend additional inference; it is not just a search alias. |
| `context.restore` (later) | Undo a bad working-context edit using a retained logical revision. Keep logical restoration distinct from exact restoration of approximate inference state. |

Concrete qualification workflow: finish an investigation, archive its detailed
turns, save the measured conclusion with source references, and leave a small
pointer in working context. In a fresh session, find why a kernel variant was
rejected and retrieve its actual measurement and command. Archive completion
precedes removal of the only copy. For concurrent sessions, target edits at a
known thread head/revision so stale changes fail visibly. Direct calls and code
bindings should share these operations rather than grow parallel implementations.

### Candidate server-side agent tasks

The user added server-side agent tasks to the palette: delegate a bounded goal
to a model/tool loop on the server; detached jobs may outlive the client or
one response stream. Examples include investigating an archived thread,
running a bounded experiment, and preparing an evidence-backed memory update.
These are execution jobs, distinct from a todo-list record or a cron schedule.

Clarified user intent: the central capability is **an agent as a server-hosted
tool**, not only a separately submitted background job. A model (directly or
through a server-side program) invokes something like `agent.run`; our server
runs that child agent's model/tool loop and returns its result to the caller.
The parent need not orchestrate the child's individual tool calls, and the
external client need not execute that inner loop. The child has its own working
context, explicit goal/tool scope and budget; return a concise result, artifacts
and evidence references, with its detailed trace retained separately.

The same runner can support two call policies: await a bounded child result and
continue the parent within its response, or detach a longer job with a durable
handle for later retrieval. These are invocation/lifetime choices around one
implementation. An MCP adapter would expose the same runner to external agents;
it is not a second executor. This remains a design clarification, not an
instruction to start implementation.

- Detached invocation sketch: submit a task with a source thread/context, goal,
  allowed tools and budget; return a stable task ID promptly. Inspect progress/results, supply
  requested input or cancel later. Task listing/discovery is also useful as our
  own capability, regardless of whether a wire protocol has a list method.
- A task may own its own conversation, working context and memory references.
  Multiple logical jobs share the existing engine/scheduler; this does not imply
  parallel GPU execution or a speedup. Keep explicit inference/tool budgets.
- Detached task lifetime must be independent of the invoking socket; an awaited
  child may instead follow parent cancellation. Current Responses
  WebSocket disconnect cancellation does not implement this behavior. Persist
  sufficient task/context/progress state for the recovery promise we choose;
  returning an ID alone does not make execution restart-safe. Record completed
  actions so recovery does not blindly repeat external side effects.
- The MCP Tasks extension is a possible interoperability layer, not the agent
  runtime itself. An MCP tool could start an agent job and return a task handle;
  the host polls in protocol code instead of spending model turns on polling.
  An optional MCP endpoint for our own agent tasks is distinct from the already
  proposed outbound MCP client used to plug third-party tools into the server.
- Reuse one job owner behind any hosted Responses tool and MCP adapter; do not
  build two schedulers. Server-side MCP composition must also handle tasks
  returned by downstream tools, with input requests and cancellation propagated.

Protocol research on 2026-10-23: MCP introduced experimental Tasks in
`2025-11-25`. The `2026-07-28` revision moved/redesigned them into the official
`io.modelcontextprotocol/tasks` extension: polling/final results through
`tasks/get`, input through `tasks/update`, cancellation through `tasks/cancel`.
Do not copy the old `tasks/result` or `tasks/list` wire flow into a new adapter.
The current [Claude Code documentation](https://code.claude.com/docs/en/mcp#mcp-tasks-that-run-on-the-server)
describes public-beta support requiring v2.1.304+; that confirms support, not
the exact day its rollout became visible. Protocol sources:
[original experimental Tasks](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks),
[2026-07-28 changelog](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/changelog.mdx),
and the [current Tasks extension](https://modelcontextprotocol.github.io/ext-tasks/specification/2026-07-28/tasks.html).

### Opportunistic background compaction — proposal

The user suggested preparing compaction or Connectome-like memory consolidation
while the server is otherwise idle. Treat computation and publication separately:
an idle job can prepare a candidate without changing a completed response or the
current working context. This is not implemented or scheduled for implementation.

Proposed minimal flow:

1. Response R completes. Freeze a useful older prefix and prepare one private
   compaction candidate for that thread, retaining complete conversational/tool
   boundaries. Summarization must not overwrite the live session or R's history.
2. New input U arrives. Foreground work takes priority. Use a finished candidate
   only when its source prefix and existing model/prompt/configuration checks
   still match and the current request permits compaction. Otherwise use the
   ordinary context; do not wait for speculative compaction when avoidable.
3. At the start of response S, form the actual working context from the compacted
   prefix, exact retained tail and all fresh user/tool input. Publish a replayable
   compaction item covering that complete input frontier before S's assistant
   output, and generate from that same view. Never append new output to completed R.
4. Continue normal Responses chaining. Stateless clients retain/replay the new
   checkpoint; stateful reconstruction recognizes it as replacing covered context,
   not as another copy of the old history. Keep the previous recovery path until
   the new checkpoint is durable or completely delivered as required by the mode.

The checkpoint cannot merely contain the earlier background summary: if it is
emitted after U and the client prunes everything before it, U and retained tail
must still be recoverable. Each branch gets a checkpoint for its actual adopted
context even if it shares an immutable prepared prefix with another branch.

This is a proposed scheduling policy behind compaction, not a standardized idle
maintenance command. OpenAI documents compaction output during an active response;
our pinned OpenResponses core does not define unsolicited idle compaction output.
No extra idle event is needed for the flow above. A future task/inspection surface
could expose preparation status when useful. The Responses `background` field
means asynchronous execution of a requested response, not general idle maintenance.

Foreground priority needs bounded GPU work and yield/cancel checks; host-thread
priority alone is insufficient. Preserve the interactive KV where practical and
include session-switch/restoration cost in observed foreground latency. Run only
when pressure/headroom and configured budgets permit; storage behavior must honor
the session's persistence policy. Memory extraction/linking can be a separate
idle task without rewriting the active context.

Start with one candidate per thread and existing fallback/eligibility machinery.
Measure foreground delay, candidate adoption, saved compaction/prefill time,
wasted background work and practical retrieval/answer quality. Do not assume that
idle compute is free or that a smaller prompt necessarily lowers whole-task time.
Sources: [OpenAI compaction](https://developers.openai.com/api/docs/guides/compaction),
[background responses](https://developers.openai.com/api/docs/guides/background),
and the local [OpenResponses stream/extension contract](../.workspaces/openresponses-reference/src/specifications/2026-04-24.mdx).

### Compaction and recovery contract

Server-side compaction need not make the server the sole owner of recoverable
state. The client need not receive a readable summary: automatic compaction can
return an opaque `compaction` item in ordinary response output, and standalone
compaction returns a replacement input window. Clients can retain and replay
these items without interpreting their contents.

- A client-held, self-contained compaction payload plus the retained window can
  recover without durable per-session server history. The compatible decoder
  and any decryption keys must survive; an opaque reference into server RAM is
  not a self-contained checkpoint, nor does this imply cross-provider portability.
- If the client keeps only response IDs/deltas and the server never returns
  sufficient replay state, reliable restart recovery requires durable server
  storage of the effective context and its referenced data.
- Replaying the full original history is another fallback, but recompacting it
  can produce a different working context. Keep that distinct from restoring
  the previous compacted state.
- Durable logical history/working-context state and optional inference snapshots
  have different jobs. Ordinary KV can be rebuilt from effective tokens. An
  approximate suffix graft may carry influence absent from the edited text;
  restoring that particular state needs its snapshot or an explicit rebuild.
- Account for disconnection before checkpoint delivery: retain enough old history
  or a durable committed checkpoint to recover. Do not silently replace the only
  recoverable state with an undelivered ephemeral compaction.

Existing durable Responses storage is a starting mechanism, not proof that
compaction recovery is already implemented. A possible arrangement is durable
logical context plus replayable client checkpoints, with GPU/SSD inference caches
managed independently. Keep the recovery choice explicit when later selecting
features from this palette.

Sources: [OpenAI programmatic tool calling](https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling),
[OpenAI compaction](https://developers.openai.com/api/docs/guides/compaction),
and the local [OpenResponses hosted-tool and compaction specification](../.workspaces/openresponses-reference/src/specifications/2026-04-24.mdx).

## Current implementation

- `main`, `origin/main` and `flint-metal` identify the qualified Flint stack at
  `635d5d534e77`, published at the user's request on 2026-10-23 and verified
  directly on origin. `zig-rewrite` and `zig-engine` retain the earlier qualified
  Zig runtime at `37498248b4cb`. The stack starts
  at condensed base `60813d03`. The original condensed commit before artifact cleanup was
  `da7940e2f2d8`; use bookmarks for its rewritten successor.
- Detailed implementation history is preserved at
  `zig-rewrite-history-20261022`; the mechanical translation is archived at
  `zig-mechanical-archive`.
- Supported native scope is DeepSeek V4.1 Flash and Flint on Metal plus the
  shared server/CLI stack. DeepSeek also supports the native Agent/benchmark/
  Eval/Web stack. Flint's internal ds4-agent tool mapping is not implemented;
  use its Responses server with an external agent. Qwen/GLM retain C engines.
- Actual Metal objects/ARC operations and pipeline catalogues remain Objective-C;
  MSL kernels and third-party C dependencies remain in their existing languages.
- Build using `tools/zig-build`; selected binaries are in `zig-out/bin`.
  Make delegates to the same build and supplies root executable names. Presets
  can select a separate installation with `./ds4-local --bin-dir ./zig-out/bin ...`.
- The server already has one engine, multiple active session slots, a decode /
  bounded-prefill coordinator, Responses continuation and durable conversations.
  Stateful Responses WebSocket serving is implemented and qualified. See
  [usage and supported scope](serving/RESPONSES.md).

Request preparation now has a C-free canonical input/model boundary. Engine
detection and C conversions stay in the runtime adapter; DS4.1 formatting and
reasoning policy live in native model modules. Execution/scheduling still import
shared C server/session records, and state storage still understands DS4.1
checkpoint kinds. Further component independence is unfinished.

## Qualification and costs

See [the final qualification report](../speed-bench/experiments/zig-qualified-20261022/README.md)
and [the implementation guide](README.md) for exact scope, commands and limits.

- Recorded campaign: 98 host/CPU fixture invocations, 45 GPU invocations and six
  diagnostic invocations; complete ordinary prefill vectors match across arms.
- Practical checks include repeated agent context edits / SSD suffix reuse with
  actual archive retrieval, server restart and concurrent conversations, trained
  DSpark comparisons, and vision tasks.
- Measured ordinary/task runtime is approximately neutral, with a slower ordered
  vision point. This establishes neither a speedup nor universal non-regression.
- Full rebuild: C 9.221 s, native 11.938 s. Measured controller edit: C 3.319 s,
  native 1.092 s. Native no-op build overhead is about one second.
- Passing behavior checks does not establish good source architecture. The
  original mechanical translation failed readability expectations. Handwritten
  owners improve readability, while C interop remains a source of complexity. The adoption cleanup removes
  the duplicate C implementations rather than maintaining two products.

## Current architectural slice and next steps

Completed serving goal (authorized 2026-10-23): stateful Responses WebSocket
serving using OpenResponses, reusing the native request/model path and qualified
for incremental conversations, concurrency and reconnect/restart under the
pressure guard. Main remains unchanged and raw results remain local.

OpenResponses `2026-04-24` requires sequential turns on each WebSocket; concurrent
conversations use separate sockets. Newer OpenAI documentation's `stream_id`
multiplexing is a separate extension and is not the initial target. Reference
checkout: `.workspaces/openresponses-reference`, commit
`8907bc3b7c4300ff5c21fd647cfde9929243afa1` (ignored).

Implemented: RFC6455 framing/upgrade, direct semantic-event sink
shared with HTTP, latest connection-local parent for `store=false`, durable
fallback for stored responses, and complete OpenResponses response envelopes.
ReleaseFast and Debug each pass the framing/native-boundary fixtures and all
26 integration checks, including the official SDK, real tool retrieval and clean
shutdown/restart. Ten selected unchanged official OpenResponses tests pass when
run serially. A stable tool loop reuses 292/317 prompt tokens and correctly reads
stored facts. The observed answer error in three optional cancellation probes
also reproduces on the independent C baseline. See the
[qualification report](../speed-bench/experiments/responses-websocket-20261023/README.md).
The installed `zig-out/bin/ds4-server` is byte-identical to the qualified binary.

On 2026-10-23 the user authorized the canonical request/model architectural
change. The completed slice separates logical request/history handling from
DS4.1 prompt, tool and reasoning materialization.

- New native records live in `serving/types.zig`, with model policy in
  `models/ds4.zig` and the DS4.1 renderer in `models/deepseek41/prompt.zig`.
- Production request materialization invokes those modules through
  `server/canonical.zig` and `server/model.zig`; required C interop remains in
  those compatibility adapters. The old whole DS4.1 renderer was removed.
- Existing native server fixtures and five standalone boundary cases pass in
  ReleaseFast and Debug. The standalone fixture requires no server/engine
  objects or Metal. Guarded HTTP restart/concurrency campaigns pass 22/22 checks
  and 5/5 graded answers each, for C -> refactored and refactored -> refactored.
  See [the focused report](../speed-bench/experiments/zig-serving-boundary-20261023/README.md).

Next proposed work: qualify server-owned context editing/suffix relocation and
idle-state eviction through an ordinary WebSocket agent workflow. Keep archival
and real retrieval in that task; existing HTTP/Agent features are not new
WebSocket qualification of every approximation setting.
Execution/session records and persisted payload semantics remain DS4-specific;
further backend independence is unfinished. Let a second actual model reveal
additional shared interfaces. Defer a universal kernel framework, runtime plugin
ABI and custom ring-I/O implementation.

Keep one heavy model job at a time under the pressure guard. Use bounded
experiments and existing tools; do not expand the harness after a useful
measurement is possible. Large RAM usage is expected; watch pressure and swap.

## Progress log

### 2026-10-23

- Recorded candidate server/client code composition, direct tool visibility,
  server-side MCP, manual/automatic compaction, specific context edits and
  cross-session memory in the design palette above. These are discussion notes,
  not new implemented capabilities or authorization to start them. Recovery
  requires sufficient state somewhere: client-carried replayable compaction,
  durable server context, or an explicit full-history reconstruction fallback.

- Planned, explicitly deferred: add native Metal V4.1/1017 activation steering,
  starting with FFN outputs. **Do not implement yet. Zero speed degradation is
  a hard acceptance requirement with steering disabled/zero and enabled.** Do
  not accept a small slowdown in exchange for behavioral control; compare fixed
  token workloads so shorter answers cannot hide slower inference. Prefer
  projection fused into existing kernels, retaining cache exclusions initially.
  The restriction already existed in upstream's first V4.1 implementation; it
  was not a feature removed by the Zig rewrite. The initial future experiment
  has a one-hour ceiling; see the [deferred plan](../speed-bench/ideas/next-experiments.md#deferred-v411017-activation-steering--planned-2026-10-23).

- Completed the M4 Pro and M5 Pro ports and cross-profile qualification on the
  available M5 Max. Auto detection selects Recursive's 36/32 resident schedule
  on Pro devices; `--flint-profile` permits a compatible smaller profile for
  qualification. M4 uses portable SIMD-group matrix/attention prefill with a
  2048-token chunk cap; M5 retains the existing NAX path and 4096-token cap.
  Profiles have distinct shader entry names; oversized resident grids and M5
  NAX on M4 are rejected before loading weights. The original M5 Max decoder
  source is unchanged.
- All three profiles ran two prompts with greedy/default sampling. Current
  M5 Max and M5 Pro reproduce the frozen native Max's 128-token continuations
  and eight teacher distributions exactly. M4 differs numerically: 8/8 teacher
  top tokens agree, mean KL 0.001505 and delta-NLL -0.014220 nats on the small
  eight-point probe; its free-running continuations eventually differ.
  Twelve Pro scalar/packet continuation comparisons match. Model-free GPU
  fixtures cover portable matrices, expert tails, attention/window/sinks,
  actual KV bank crossing/local-ring wrap and sampler handoff.
- Both Pro profiles pass 29 WebSocket/protocol checks and 17 answer probes,
  including tools, concurrent conversations, SDK, cancellation, durable restart,
  and 4855-token prefill followed by cached continuation. The host/server/
  storage/agent/script regression suite passes. On this M5 Max, M4's 798-token
  prefill is 478 ms versus 217 ms for NAX, with generation around 219 versus
  232 tokens/s. These are cross-profile measurements, not predictions for Pro
  hardware. Physical M4/M5 Pro residency, compiler behavior and performance
  remain unqualified. See the [profile report](../speed-bench/experiments/flint-profiles-20261023/README.md);
  raw evidence stays in `.workspaces/flint-profiles-20261023/`. `make all`
  installed the qualified executables; installed CLI/server bytes match the
  copies used for qualification. Changes remain on the local Flint stack.

- Added `download-model.nu flint` / `flint-int4` for the qualified native
  safetensors directory. Uses Recursive's pinned Hugging Face revision
  `3b93f9053b530488405ae9fca51d4900d6ce21f0`, twelve file identities from the
  original packaging manifest, and the existing resumable HF/SHA-256 flow.
  Downloads into `gguf/flint-int4` without changing the default GGUF symlink.
  All 15 offline download tests pass. A live run downloaded all seven metadata
  files from the pinned revision and verified the five existing weight shards
  through workspace symlinks, avoiding another 22.6 GB transfer. Nu parse and
  format checks pass; no inference/kernels changed.

- Completed Recursive Flint's native Zig/Metal backend and existing Responses
  HTTP/WebSocket integration. User explicitly chose the existing Metal runtime
  only: no MLX library or embedded Python in production. Reused licensed shader
  source and the original M5 Max persistent decode schedule. Source reference
  is `/Volumes/Models/flint` at `e53663b262af`; installed affine INT4 weights
  are under its `.local/share/rec/models/fdc85248...` directory (22.57 GB).
  [Usage and reproduction](../docs/FLINT.md),
  [qualification report](../speed-bench/experiments/flint-native-20261023/README.md).
- Reference baseline completed under the pressure guard after the user stopped
  their DS4 agent. Original Python Engine, M5 Max, context 262144, 128 generated
  tokens, three repetitions; medians of repetitions 1 and 2: short 21-token
  prompt prefill 43.5 ms / greedy 246.1 tok/s / default grammar 224.94 tok/s;
  798-token prompt prefill 227.3 ms / greedy 239.75 tok/s / default grammar
  212.61 tok/s. Raw results and exact driver remain ignored under
  `.workspaces/flint-baseline-20261023/`. This is the reference, not a native
  performance result. Default sampling is T=1, top-p=.95, top-k=20.
- Native tokenizer/prompt, raw safetensors bank packing, batched prefill,
  resident decode and GPU filtered sampling are implemented. Tokenizer matches
  47 HF cases in Release/Debug; seven packaged templates match. Native short
  prefill improved 202 -> 61 -> 37 ms by testing one small-batch QMV change at
  a time. Selected packed QMV retains NAX for larger batches. Default-sampling
  decode is 247.1 / 225.9 tok/s (short / 798-token), reference 224.9 / 212.6;
  greedy longer-task total is neutral (756.8 vs 757.7 ms), with slightly slower
  decode. These are narrow warm measurements, not a universal speed guarantee.
- Eight teacher-forced distributions inside reasoning: mean delta-NLL +0.013817
  nats, mean KL 0.0010015, all eight top tokens equal; perplexity ratio 1.0139.
  This is a small two-prompt proxy. Native RNG differs from MLX; native serving
  has no grammar masks. Do not claim exact output equivalence.
- Final real-model WS qualification passes 29 checks and 17 quality grades:
  independent concurrent conversations, real tool retrieval, disconnect,
  reconnect/restart and official SDK. A 4,855-token prompt plus post-decode
  continuation passes; input 5,011 correctly credits 4,855 cached prefill rows.
  Generated continuation is recomputed through prefill, so it is not counted as
  reused KV. Actual GPU fixtures pass token-65,536 bank crossing and wrapped
  local-ring publication; full 262k-token inference is not yet qualified.
- Per-conversation BF16 KV and bounded local-ring/logit snapshots are native.
  An engine mutex serializes prefill/resident packets: overlapping two resident
  kernels risks GPU barrier deadlock. Flint packets are distinct from MTP.
  Durable Responses JSON survives restart; Flint GPU-state SSD checkpoints,
  suffix relocation, vision and internal ds4-agent tools remain unsupported.
  The server-only `flint` preset selects model defaults and clears incompatible
  DeepSeek cache settings. Main is unchanged; new work stays local.
- Full model-free host/script verification passes, as do Debug server/session
  fixtures and all 11 launcher cases. A guarded existing DeepSeek control and
  final runtime produce identical selected IDs and all reported top-logprobs.
  Raw artifacts remain ignored in `.workspaces/flint-baseline-20261023/` and
  `.workspaces/flint-tests/`; commit source, concise reports and reproduction
  helpers only.

- Followed up on the user's script preference: use plain `#!/usr/bin/env nu`
  and `nu script.nu` invocations, removing blanket `--no-config-file` flags
  from entry points, callers and test children.
- Qualified the Nushell script migration using Nu 0.116.1. `ds4-local` and
  `tools/zig-build` keep their executable names, arguments and process-replacing
  behavior. Downloader/test/benchmark/service drivers use `.nu` names; the
  downloader is now `download-model.nu` with a standard JSON target catalogue.
- Keep Python for model/numerical work and the small split-model binary-I/O
  helper (flock, streaming copy/hash and fsync). All 30 download targets and eight
  pinned identities are preserved. Historical experiment snapshots retain their
  recorded scripts; the C-era build helper is a Nu wrapper for archived objects.
- Validation: the full Nu host verifier passes; `make test-scripts` passes ten
  launcher cases, thirteen genuinely hashed offline download cases and driver
  contracts. One-time controls match 66 GPU CLI invocations, DSpark cases, GLM
  prompts and sweep status/arguments. JPEG fixture bytes reproduce exactly.
  Nu parse/format checks and Linux-runner helper checks pass; Linux service
  actions and model/GPU runs were not part of this script migration.
- `make test` now delegates to `tests/verify.nu`; `make test-scripts` selects its
  fast script-only subcommand. Build flags are forwarded as arguments. Runtime
  binaries remain unchanged while the user's agent is running. Raw evidence is
  local under `.workspaces/nu-launchers/`, `.workspaces/nu-downloader-20261023/`,
  `.workspaces/nu-drivers-20261023/` and `.workspaces/nu-scripts-20261023/`.
  Keep these script changes separate from the pending research journal notes.

- Promoted the qualified runtime through `37498248b4cb` to `main` and pushed to
  `origin` at the user's explicit request. Confirmed remote main had not moved,
  ancestry was linear and no conflicts/raw benchmark dumps were introduced.
  Publication reused existing qualification. Pending local research notes were
  preserved outside the published commits.

- Completed a source review of Strata, local Flint and Luciano Martins's eight
  October 6–8 vLLM optimization PRs against Zig runtime `37498248b4cb`.
  This is research and proposed work; no runtime change or model benchmark ran.
  Flint remains at `e53663b262af`; Strata snapshot is `7ddc17a38fe7`.
- Strata's [cache-conditioned router](https://github.com/Niko1221/Strata/blob/7ddc17a38fe7f31637e895ecfba58c2cd43d65ec/src/kernels/cuda/route_prior.cu)
  biases selection toward resident experts while retaining unbiased mixture
  weights. DS4 already has global decayed hotness, bounded resident substitutions,
  selected-cache overlap and exact decoder dependency suffixes. Proposed follow-up:
  compare a simple residency bonus with the existing policy, retaining DS4's
  transformed-affinity/bias semantics and measuring task quality and SSD bytes.
- The first architectural target is exposed host/GPU handoff cost on cache hits.
  Existing [SSD experiments 62–66](../speed-bench/experiments/ds41-ssd/README.md#62-full-session-route-and-exposed-io-census)
  found 75.3% all-resident layer events, a 30.7% decode-time reduction with free
  future routes plus deferred checks, and a slower real prefetch predictor.
  These are historical workload results and oracle bounds, not current Zig speed
  measurements. Refresh the baseline before changing the selected-load schedule.
- A small independent candidate remains head-only Q4_K. Header-only inspection
  of the local 1017 checkpoint confirms `output.weight` is Q8_0, 5120 × 129280,
  670.703 MiB; Q4_K would occupy 355.078 MiB. Existing native dense dispatch
  accepts Q4_K. Profile head time, then measure speed, continuation delta-NLL/KL
  and practical outputs; the 47.1% head-byte reduction is not an end-to-end gain.
- vLLM sources: [attention tuning](https://github.com/vllm-project/vllm/pull/60317),
  [KV-sharing metadata](https://github.com/vllm-project/vllm/pull/60445),
  [vision/audio](https://github.com/vllm-project/vllm/pull/60478),
  [MoE shapes](https://github.com/vllm-project/vllm/pull/60504),
  [vision batching](https://github.com/vllm-project/vllm/pull/60643),
  [diffusion batching](https://github.com/vllm-project/vllm/pull/60663),
  [text fusion](https://github.com/vllm-project/vllm/pull/60674) and
  [MTP/head work](https://github.com/vllm-project/vllm/pull/60689).
  Only 60643 is merged at this review; seven remain open. Transferable proposals
  are actual-shape M5 tuning, fewer intermediate passes and host dependencies,
  and speculation selected by accepted tokens per wall time and unique expert
  bytes. Preserve the existing pressure guard and one-change/hour workflow.

- Completed source layout after the user clarified that C/H files and backend
  trees belong under `src/` too: 225 additional tracked files preserved. Common
  C/H/include tables are in `src/`; Metal driver/shaders join `src/metal/`;
  retained CUDA/ROCm sources are in `src/cuda/` and `src/rocm/`. No root C/H or
  backend source directories remain. Algorithms and supported-backend scope
  are unchanged; CUDA/ROCm are retained source, not newly hardware-qualified.
- `make all` / `make test`, native suffix transactions, four packet/ownership
  fixtures and the official scorer build pass. Default shader discovery works
  in a guarded DeepSeek run; tokens and all reported top-20 logits/logprobs
  match the saved pre-layout baseline exactly. The preset launcher also works.
- Updated active helpers and documentation links, leaving historical experiment
  source lists intact. JPEG tests now load the existing production image library
  instead of compiling the obsolete C decoder alone; all three pass. Existing
  benchmark helper checks pass 8 + 5 cases, and the calibration helper discovers
  the repository and extracts 184 Eval records without rewriting datasets.
- Independent review confirms file preservation and path-only backend changes.
  Reproduce the host checks with `make all test`; run extra fixtures through
  `tools/zig-build agent-context-suffix-test native-activation-test
  native-norm-test native-hc-projection-test native-f16-publication-test
  quality-score`. The layout smoke uses the same model command as the adoption
  report. Raw logs/manifests remain in
  `.workspaces/source-layout-completion-20261023/`; JPEG/helper logs are under
  `.workspaces/src-layout-20261023/`. Main is unchanged.

- Adoption cleanup completed: all five production mains are Zig. Removed the
  migration build option, native-selection switches and duplicate C owners;
  retain Qwen/GLM C engines behind the shared frontends and actual Objective-C
  Metal driver boundary. Server signal flags have one Zig owner shared with C
  signal/I/O helpers. Make now delegates to Zig, preserving root launch paths;
  GGUF scoring links the production engine. Main is unchanged.
- Qualified frozen-parent vs cleanup inference on DeepSeek, GLM and Qwen:
  short generated continuations and every reported top-20 logit/logprob value
  match exactly. GLM uses SSD streaming; Qwen uses its normal Metal path with
  residency/warmup disabled (n-grams remain disk-only, no Qwen SSD support).
  22 ReleaseFast host fixtures, seven Debug fixtures, Qwen/GLM GPU suites and
  all 26 required live WebSocket checks pass. Both live servers exit cleanly.
  Installed the same executables through Make at the usual root paths; both
  `make all` and `make test` pass end to end.
- The default DeepSeek Metal fixture has a pre-existing FP4 rounding-midpoint
  failure reproduced identically by the frozen parent; no kernel fix is bundled
  with adoption. The earlier three WebSocket answer-quality failures also remain
  unchanged. A resident GLM baseline was pressure-stopped before switching to
  the working SSD path. Full details and commands are in the
  [cleanup report](../speed-bench/experiments/zig-adoption-20261023/README.md);
  raw results stay under `.workspaces/zig-adoption-cleanup-20261023/`.
- Completed the authorized `zig/` to `src/` directory rename as a separate
  change after adoption. Component structure and runtime behavior are unchanged;
  build paths, C includes, test imports and current documentation links follow
  the new location. Historical experiment source lists retain their old paths
  for the revisions they describe. Build metadata stays at root; the resume
  journal is now `src/JOURNAL.md`. `make all` and `make test` pass, as do
  the model-free suffix transaction fixture and four Metal packet/ownership
  fixtures. All 376 files are preserved; implementation content matches after
  path substitution, and all 31 updated Markdown links resolve. No new model
  or latency campaign was needed for this directory-only change. Raw build/test
  logs stay under `.workspaces/src-rename-20261023/`.

- User authorized the request/model architectural change. Added native logical
  records and a C-free DS4.1 preparation implementation, replacing the previous
  whole renderer. Model aliases, reasoning policy and runtime format selection
  now belong behind the model adapter; durable history remains server-owned.
- ReleaseFast and Debug build and CPU checks pass. A build caught a signed C
  enum conversion mismatch; review caught the disabled fallback for unknown C
  reasoning values. Both were corrected before final CPU verification.
- Reusing the existing guarded HTTP two-session/restart/cancellation campaign;
  both cross-version and refactored-only campaigns completed normally. They
  cover fresh requests, concurrent decode, cancellation, surviving requests,
  streaming and durable Responses/conversation restart. No numerical kernel or
  checkpoint format changed. Results are local under
  `.workspaces/zig-serving-boundary-20261023/`; concise results and commands are
  [tracked](../speed-bench/experiments/zig-serving-boundary-20261023/README.md).
- The current slice is complete. Responses WebSocket transport remains the next
  feature in this sequence; the subsequent user-authorized goal completed it.
  A generic execution backend and second native model remain later work.
- Completed stateful WebSocket serving using OpenResponses `2026-04-24`, with
  sequential sockets and cross-socket concurrency. Added direct event sinks,
  ephemeral parent replay, durable recovery, framing and shared response schemas.
- ReleaseFast and Debug integration each pass 26 required checks; the real
  function tool reads a stored record whose facts are absent from its prompt.
  Official Python SDK 3.37.0 works. All test servers exit cleanly.
- The stock OpenResponses runner overfills the two-slot server with ten parallel
  tests and a 30 s timeout (3/10 pass). The unchanged ten official cases pass
  serially with 180 s timeouts. Do not present the first timeout run as a schema
  failure or claim that all optional protocol features are supported.
- Quality records remain separate: each full run passes 12/15 answer probes;
  the other three repeat one wrong JSON answer before/during/after cancellation.
  An independent C-server control produces the same answer and token counts.
- Stable instructions/tools yield 292/317 cached prompt tokens in a real file
  lookup loop; changed instructions in the broader integration yield no prefix
  hit, as expected. This is reuse evidence, not a whole-agent speedup claim.
- Installed the qualified server in `zig-out/bin`. Results, first failures and
  reference sources stay local under `.workspaces/`; only concise reports,
  implementation and necessary tests are tracked. Serving goal is complete.

### 2026-10-22

- Completed and qualified the handwritten rewrite, condensed it into one branch
  commit, and retained detailed history without advancing main.
- Reviewed source quality: handwritten Zig owners are generally preferable for
  engine/agent work; small C routines can remain more direct. Organization is a
  large part of the benefit. Keep the real Objective-C Metal boundary.
- User clarified the serving product and USRBIO reference. Adopt the direction
  above, with simple in-process I/O as the starting point.
- User requested readable commits: keep raw benchmark logs, generated snapshots
  and run dumps local/ignored. Keep concise reports, reproduction scripts and
  necessary fixture inputs tracked. Removed 6,749 new generated files (55.5 MB)
  from tracking, preserving every local file byte-for-byte; 234 reports, scripts
  and fixture inputs remain tracked. Older main history is outside this cleanup.
  Verification inventory is local under
  `.workspaces/benchmark-tracking-cleanup-20261022/`. Runtime source is unchanged.
- Added this journal and its resume/update instructions in `AGENTS.md`.

## Source references

- [DeepInfer storage deployment notes](../../DeepInfer/doc/inference_deployment_qa.md)
- [3FS USRBIO](https://github.com/deepseek-ai/3FS/blob/main/src/lib/api/UsrbIo.md)
- [Responses WebSocket semantics](https://developers.openai.com/api/docs/guides/websocket-mode)
- [Responses conversation state](https://developers.openai.com/api/docs/guides/conversation-state)
