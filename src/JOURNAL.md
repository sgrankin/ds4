# Serving project journal

Last updated: 2026-10-23. This is the compact handoff for resumed work and context
compactions. Keep the current state accurate and append short decision/progress
entries at meaningful checkpoints. Raw experiment output stays local.

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

## Current implementation

- `zig-rewrite` and `zig-engine` identify the rewrite branch: condensed base
  `60813d03`, followed by the focused request/model boundary change;
  `main` remains at `ec5424de5562`. Do not advance main or push without further
  authorization. The original condensed commit before artifact cleanup was
  `da7940e2f2d8`; use bookmarks for its rewritten successor.
- Detailed implementation history is preserved at
  `zig-rewrite-history-20261022`; the mechanical translation is archived at
  `zig-mechanical-archive`.
- Supported native scope is DeepSeek V4.1 Flash on Metal plus the shared
  Agent/server/CLI/benchmark/Eval/Web stack. Other model families and backends
  retain their C implementations.
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
