# Serving project journal

Last updated: 2026-10-22. This is the compact handoff for resumed work and context
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

- `zig-rewrite` and `zig-engine` identify the condensed handwritten rewrite;
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
  Root-level Make executables are independent. Presets select native binaries
  with `./ds4-local --bin-dir ./zig-out/bin ...`.
- The server already has one engine, multiple active session slots, a decode /
  bounded-prefill coordinator, Responses continuation and durable conversations.
  Responses WebSocket serving is not implemented.

The new files expose useful owners, but dependency boundaries still largely
follow the old C program. Serving code imports shared C server/session records,
detects model families and selects prompt syntax. State storage still understands
DS4.1 checkpoint kinds. This is a qualified foundation for the intended component
architecture; component independence is unfinished.

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
  owners improve readability, while C interop and retained dual implementations
  still impose complexity.

## Next proposed implementation slice

The architecture discussion has not authorized another broad rewrite. A useful
next slice, if we proceed, is:

1. Separate canonical conversation/request handling from DS4-specific prompt,
   tool and reasoning materialization. Use native Zig records at this boundary;
   keep needed C interop inside the implementation.
2. Feed Responses HTTP and new WebSocket transport through that same path.
3. Qualify two interleaved conversations, reconnect/restart, idle-state eviction
   and server-owned context editing/suffix reuse using a real agent client.
4. Let a second actual model reveal further shared interfaces. Defer a universal
   kernel framework, runtime plugin ABI and custom ring-I/O implementation.

Keep one heavy model job at a time under the pressure guard. Use bounded
experiments and existing tools; do not expand the harness after a useful
measurement is possible. Large RAM usage is expected; watch pressure and swap.

## Progress log

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
