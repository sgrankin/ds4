# Project direction and current status

Updated 2026-11-03. **The upstream port through `890898cf` is complete.**
The organized manuals and lab journal are available under the [documentation index](README.md).
The initial adaptation omitted upstream functionality; the completion audit
covered all 118 commits and integrated the missing paths. All upstream
functionality is required unless we explicitly excised it from this fork;
hardware qualification limits do not remove that implementation obligation.

The resident-routing experiments are complete. Caps through six are implemented;
cap five is adopted in the `resident-fast` and `persistent-fast` presets after
longer fixed-history quality checks. Autonomous agent quality is not yet
qualified. See the
[timing sweep](lab/journal/2026-11-02-resident-routing-through-six.md) and
[longer quality follow-up](lab/journal/2026-11-02-resident-long-quality.md).

The [32K throughput and dependency profile](lab/journal/2026-11-03-context32-profile.md)
establishes large-context rates after that adoption. The
[routing-overlap follow-up](lab/journal/2026-11-03-route-overlap.md)
implements shared-expert overlap, bounded event polling and precise expert-cache
retirement. It is enabled by the resident fast presets; ordinary runtime
defaults remain opt-in.
The [profile after overlap](lab/journal/2026-11-03-post-overlap-profile.md)
selects the existing attention-head transform and per-token RoPE table as the
next candidate for SSD streaming decode. It is a recommendation, not an enabled
change; the timing refresh also records unresolved variation in warm appends.

## What we are building

A focused, native local inference server for the M5 Max, with possible M1 Max
support. Agents connect through stateful Responses HTTP/WebSocket sessions.
Model implementations, scheduling and RAM/SSD storage should be understandable
components that we can improve independently.

- Follow upstream DS4 and official model references. Prefer DeepInfer semantics
  and dataflow unless a difference benefits our single-Mac environment.
- Keep many independent conversations, with durable history separate from
  the model's working context and disposable inference caches.
- Support useful large models and fast helpers. The desired next architectural
  capability is independently owned, co-resident generative models with a shared
  scheduler and memory budget; two generators are not implemented yet.
- Keep production inference native. Learn from Python implementations and reuse
  appropriate kernels and formats without adopting their Python harnesses.
- Let the client agent own tools, PTC, MCP and workflow state. The server owns
  inference, decision APIs, sessions, caches and steering. Generic hosted
  orchestration was tried and removed.

Persistent-agent facts may leave working context while remaining in checkable
client storage. Suffix reuse helps materialize those edits; it does not itself
summarize, archive facts or run a compaction agent.

## Where we stand

This describes the local stack after `main`, not a new upstream release.
The completed work is installed locally; the stack has not been published.

| Area | Current state | Detail |
| --- | --- | --- |
| Native serving | Zig entry points and ordinary Metal implementations for DeepSeek V4.1 Flash, Flint, Qwen 3.8 Flash Next, GLM 5.3 Flash and Clef. Some legacy models and platform/library boundaries remain C/Objective-C. | [Source map](../src/README.md), [model manuals](models/README.md) |
| Agent protocol | Stateful Responses HTTP/WebSocket, independent conversations, durable response ancestry, client function calls and native decision APIs. Generic server tool execution is removed. | [Responses contract](serving/responses.md), [Clef](models/clef/usage.md) |
| Model interaction | Official-reference prompt/tool handling, client sampling overrides, explicit cache breakpoints and native constrained decoding for DeepSeek and Flint. Typed partial tool calls survive generation limits. | [Interaction qualification](../speed-bench/experiments/interaction-catchup-20261027/README.md), [DSML limits](../speed-bench/experiments/dsml-length-20261028/README.md) |
| Persistent inference state | RAM/SSD checkpoint storage, suffix relocation from immutable pre-RoPE originals, packed index/main KV, and useful drafter-history persistence. Packed KV and DSP1 are selected through DeepSeek fast presets; ordinary engine defaults differ. | [Recipe index](../recipes/README.md), especially chapters 02, 03, 14 and 15 |
| Context edits | Semantic endpoints, bounded hot retention, anchor selection and capture during ordinary prefill are complete. Server `persistent-fast` enables this package; the direct agent and other presets do not gain these new boundary defaults. | [Current checkpoint recipe](../recipes/16-semantic-context-checkpoints.md) |
| Upstream maintenance | Complete through `890898cf`, including ordinary graph fusions, tiny-batch expert reuse, pre-M5 selectors, negotiated replicated TP and inline gates. Host, GPU and installed-model checks pass; physical two-Mac/RDMA and older-device performance remain unmeasured. | [Catch-up journal](lab/journal/2026-11-02-upstream-catchup.md), [upstream mechanisms](models/deepseek/upstream-performance.md) |
| Experimental leftovers | Candidate-only scoring and block-only look-back recovery retain narrower or diagnostic status. The earlier opt-in HC collapse/norm experiment is separate from the integrated upstream graph fusions. Slower integer/packed-activation prefill hooks were removed. | [DeepInfer comparison](models/deepseek/deepinfer/comparison.md), [not-adopted lessons](../recipes/not-adopted/README.md) |

The earlier sequence—DSML truncation, drafter-history persistence,
physical main KV packing, one fusion, and context checkpoints—is complete.
The checkpoint follow-up removed extra streamed-weight sweeps. Its five-pair
zero-edit comparison finds no clear request-time overhead; all 20 answers pass.
[Latest result](../speed-bench/experiments/context-capture-20261028/NO_EDITS.md).
Different reports use different baselines. Their percentages are not additive
and do not establish one cumulative whole-agent speedup.

Cache features also have combination limits: semantic suffix grafts currently
exclude DSpark/MTP, steering, TP and decoder replay. MTP checkpoint persistence
is a separate capability. Unsupported in-sweep capture uses the older capture path.

## What remains a choice

The useful directions below are alternatives, not an execution queue:

1. **Use and qualify the current stack in a real agent workflow.** Establish
   whether the accumulated changes improve representative work. Longer
   append-only sessions have not received the latest five-pair comparison.
2. **Complete multi-generator composition.** DS4 plus Flint in one process
   still needs model routing, aggregate admission and scheduling. Clef helpers
   alongside a generator do not establish this capability.
3. **Continue targeted DeepInfer convergence.** Scheduling quantum, batched
   look-back recovery, raw FP8 and broader decoder replay are candidates. Choose
   one bottleneck and one user-visible objective before implementing another.

Server-hosted compaction, multi-span suffix planning and combined block-only
recovery/suffix reuse remain unimplemented. Voice and additional model families
remain ideas. No investigation of these starts automatically.

## How we work from here

- Select one outcome with the user. Give its next experiment an initial one-hour
  budget, including preparation, build, execution and analysis. At the boundary,
  report the result and decide whether another slice is worthwhile.
- Keep one heavy model job under the pressure guard. Performance claims need
  five samples per side, alternating order, all samples retained and variation
  reported. Quality differences are measured tradeoffs, not automatic rejection.
- Parallelize only preparation or review that serves the selected outcome.
  Record side ideas without starting side experiments.
- Write one concise result report for a measurement. Keep commands and detailed
  samples there; update the recipe's conclusion and add one short journal entry.
  Do not copy the full result into the queue, journal and design documents.
- Stop when the selected question is answered. Reuse existing tools and keep
  raw artifacts ignored. No automatic next experiment, new harness or publication.

## Where to read

- **This page:** current goals, capability boundaries and choices.
- **[Recipes](../recipes/README.md):** how adopted techniques work and their tradeoffs.
- **[Lab journal](lab/README.md):** experiment narratives, negative results and original evidence.
- **[Experiment queue](../speed-bench/ideas/next-experiments.md):** selected work only.
- **`speed-bench/experiments/`:** historical measurement evidence, read through a specific link rather than as a project overview.
