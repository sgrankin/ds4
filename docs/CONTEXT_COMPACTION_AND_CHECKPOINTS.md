# Context compaction, suffix reuse, and checkpoints at message boundaries

Design discussion captured from a side conversation, 2026-10-28. The original
discussion proposed work; the user subsequently authorized implementation and
evaluation in the main conversation.

Current checkpoint, 2026-10-28: the first opt-in implementation is built and
live-quality qualified at `d5b8e8a3`; final installed source `9b826389` adds
reasoning-enabled identity and padding-adjusted control-admission guards.
It captures semantic prefill endpoints,
retains complete bounded RAM-only sources sharing immutable chunks, and guards
ongoing-task freshness with actual instruction/tool controls. Semantic byte ends
floor to an existing completed token; control admission still rounds up. The
parallel smoke imports 1830 unchanged C tokens from a RAM-only checkpoint.
A same-runtime freshness off/on test answers correctly with 410 exact tokens versus
1,688 imported C tokens. All 10 changed/deleted-fact and user-control replies
pass, as do client archive retrieval and READY→MAPLE `previous_response_id`
continuation. Actual restart loses the transient source, falls back with
cached=0/no-aligned-source, recomputes and answers the changed LARCH fact.

The rows=72/eight-edit run started 10:50 UTC and was interrupted as calibration
when its runtime exceeded the checkpoint budget; all raw data are retained.
A fresh 15-process off/sparse1/fuller4 campaign completed with rows=24 and two
edits, five fresh samples per setting. Across both workflows, fuller4 population
costs 81.8% more, two repeated edits cost 38.0% less, and initial+two edits cost 2.14%
more. All 90 replies pass. Sparse1 imports the same spans without early exact
prefix anchors and supplies no material edit-time gain. The [five-round report
and Pareto plot](../speed-bench/experiments/context-boundaries-20261028/RESULTS.md)
retain variation, order and actual geometry. The separate final-binary eight-edit
retention diagnostic passes 18/18 replies; all 16 edited imports retain the
original RAM-only C, with no capture errors. This is functional evidence, not
an eight-edit comparative speed measurement. Calculated break-even is 2.17
edits, assuming retained sources and similar future costs; any eight-edit speed
projection is calculated, not measured five-pair evidence. Server `persistent-fast`
now enables boundaries/ongoing1, cap4, spacing512 and HOT512, with explicit
overrides honored. Engine defaults, the agent and other presets receive no new
boundary defaults. The wrapper retains its existing64GB/resident routing/512
chunk policy; the timed native32GB/configured8192/effective2048 setup is distinct.
Final padded corrected-fact and medium→high checks each pass2/2, with thinking
on throughout; the changed high request has cached0/no aligned source. Unit
rendering separately distinguishes none/medium without disabling generation
reasoning. Final guards stop cleanly with no pinned cache endpoints and hot
bytes within512 MiB. Aggregate evictions include later QA and do not identify
which individual C was evicted. These are functional checks, not new timing
samples; the high-reasoning, unpadded benchmark stays frozen at `d5b8e8a3`.
No speed claim comes from the quality smokes. Full
capture/serialization/hash work remains; there is no incremental GPU collector
or server-hosted compaction agent. Callers materialize edits and own the archive.
See the [qualification report](../speed-bench/experiments/context-boundaries-20261028/README.md),
[self-contained recipe](../recipes/16-semantic-context-checkpoints.md) and
[journal](../src/JOURNAL.md). The original analysis below retains its historical
measurements and proposals; this status block supersedes its now-implemented
RAM-retention/freshness proposals.

Implementation observations were checked against the local tree at
`a9593e124f536ce7d8223a19f61f503ade77b791`. Historical measurements and calculated
sizes below are explicitly distinguished from fresh performance measurements.

The implementation and size discussion concerns native DeepSeek V4.1 text on
Metal, rather than a generic cache shared by all supported model families.

The working proposal is to preserve an append-only logical history, materialize
selective context edits, and capture useful message/block boundaries with
incremental shared state. Revisit the latest-user freshness rule separately.
Bounded replay could make older boundaries cheaper to retain; combining that
recovery with suffix relocation remains unimplemented. A multi-span planner is
not a prerequisite for the first experiment.

## Motivation and external reference

The user asked whether Shelley's automatic compaction works with our suffix KV
reuse, then suggested taking a checkpoint after every compactable block: a
message, assistant tool call, tool result, or equivalent model-input region.

Reference: Philip Zeyliger and Maisem Ali, [Automatic Compaction: How Shelley
Keeps Going and Going Without Losing the Plot](https://blog.exe.dev/compaction),
2026-10-27. The web page was inaccessible through this side conversation's
network tools, so the user supplied a PDF, which was read directly. The local
copy is [available here](</Users/sgrankin/Downloads/Personal — Automatic Compaction: How Shelley Keeps Going and Going Without Losing the Plot - exe.dev blog.pdf>).

The article describes Shelley's third compaction implementation. The older
public documentation described replacing history with a summary and retaining
recent messages; that is not the whole design in this article.

The current article's process is:

1. At a configurable threshold, approximately 200K tokens for their frontier
   models, the harness nudges the model to compact.
2. The model obtains a message index, then chooses particular tool outputs to
   trim and groups of messages to replace with shorter summaries.
3. A linter checks the amount of compaction; unsuccessful attempts can be retried.
4. An append-only SQLite log retains the original messages and compaction events.
   The harness materializes the edited working view when making model requests.
5. Compaction-control exchanges are omitted from that working view. Summaries
   retain references to the originals, which the agent can retrieve later.

This fits the persistent-agent direction discussed in the main conversation:
facts can move from working context into checkable storage. The 200K threshold
is their tuning choice, not a locally measured DS4 target.

## What our matcher actually matches

For approximate suffix grafting, the path is:

```text
API JSON messages
  -> model-specific prompt rendering, including roles and tool-call syntax
  -> token IDs
  -> eligible saved checkpoint candidates
  -> one matching token span
```

The suffix matcher compares token IDs, not JSON object identity, message IDs,
or semantic equivalence. Message/tool identity has separate uses in ordinary
continuation handling. Changing outer JSON formatting does not necessarily
change the rendered prompt, but changing rendered content, roles, or tool syntax
can change the tokens. A stable conversation/cache identity scopes candidate
discovery; it does not establish that their tokens match.

Current suffix matching, in [src/suffix_match.zig](../src/suffix_match.zig):

- Take one immutable source checkpoint with a fixed token endpoint.
- Compare its final 64 tokens against possible ends in the edited input, before
  a caller-supplied cutoff.
- Extend matching hits backward token by token.
- Require at least 64 reused tokens and compatible even-aligned source and
  destination starts. Right shifts require their opt-in and pre-RoPE support.
- Choose the plan with the most reused tokens. The server compares plans across
  eligible sources; the agent also has its existing disk/frontier selection policy.

This is not a general message diff or a multi-span planner. The imported span
must reach its source checkpoint's endpoint. An arbitrary earlier endpoint
cannot be obtained simply by shortening that checkpoint's token list.

### Example: trim a tool result and preserve later work

These are abbreviated message arrays, not complete API requests. Assume suitable
checkpoints, a stable cache identity, and retained content longer than 64 tokens.

Before:

```json
[
  {"role":"user","content":"Fix the parser."},
  {"role":"assistant","tool_calls":[{"id":"read-1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"parser.zig\"}"}}]},
  {"role":"tool","tool_call_id":"read-1","content":"[30K-token file dump]"},
  {"role":"assistant","content":"[4K tokens of subsequent investigation]"}
]
```

After trimming, followed by a new user request:

```json
[
  {"role":"user","content":"Fix the parser."},
  {"role":"assistant","tool_calls":[{"id":"read-1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"parser.zig\"}"}}]},
  {"role":"tool","tool_call_id":"read-1","content":"[short relevant excerpt]"},
  {"role":"assistant","content":"[4K tokens of subsequent investigation]"},
  {"role":"user","content":"Now add regression tests."}
]
```

Conceptually:

```text
old: [unchanged prefix] [large dump] [investigation]
new: [unchanged prefix] [excerpt]    [investigation] [new request]
```

Evaluate the edited prefix, import the matching investigation state at its new
positions, then evaluate the fresh request/tail. For illustration only, a plan
could be:

```json
{
  "source_start": 32000,
  "source_end": 36000,
  "destination_start": 2000,
  "destination_end": 6000,
  "reused_tokens": 4000,
  "fresh_tokens": 500
}
```

Those positions illustrate the algorithm; they were not obtained by tokenizing
the shortened JSON above. A real source must actually end at the selected span.

## The latest-user cutoff is an explicit policy

The server's suffix caller computes:

```text
cutoff = min(prompt_length - minimum_fresh_tail, latest_user_marker_position)
```

The agent makes the equivalent freshness decision. Imported suffix state must
end before that cutoff. This rule does not ask whether the latest user message
is new, changed, or already processed in the retained history.

**Important correction to a broad description from the discussion:** this does
not mean every request recomputes everything after its latest user message.
Normal exact continuation can reuse an unchanged cached prefix, including tool
activity after that message. The restriction applies to approximate suffix
imports after an edit. An available exact prefix before the first edit may
still be retained or restored.

In the JSON example, remove the final "Now add regression tests" message.
The latest user becomes "Fix the parser" near the beginning. The unchanged
investigation is then beyond the cutoff and cannot be imported. This is
particularly restrictive for a long autonomous tool loop under one user task.

Proposed adjustment: distinguish genuinely new or changed user instructions
from unchanged user content already represented in the source. Protect the
fresh request and an appropriate replay tail without automatically excluding
the entire ongoing task. This is not implemented or qualified by this note.
Do not infer that matching message IDs alone makes cached activations exact.

See [server cutoff](../src/server/persist.zig),
[agent selection](../src/agent/context_runtime.zig), and
[ordinary continuation/generation](../src/server/generation.zig).

## What checkpoints contain, and why their endpoints matter

A checkpoint is a frozen inference-session state after N evaluated tokens.
It contains token IDs and position, recent raw attention windows, cumulative
compressed KV and index arrays, unfinished compression-pair state, logits, and
relevant headers/provenance. Suffix-enabled captures additionally retain
immutable pre-RoPE rotary originals.

The model's mutable GPU state advances; the checkpoint remains immutable.
Model weights are not duplicated into it. Logical conversation history and the
agent's factual archive are separate responsibilities.

DeepSeek's raw attention window rolls forward. At token 12K, raw state needed
to resume exactly at 6K may have been overwritten, and compression/frontier
state has advanced. A complete state at 12K is not automatically an exact
savepoint at every earlier token.

Example:

```text
original: [A: 4K] [B: 4K] [C: 4K]
saved:          ^4K               ^12K

edited:   [A: 4K] [summary B: 1K] [C: 4K] [new request]
```

Restore the 4K checkpoint, evaluate the summary, use the 12K checkpoint as C's
relocation source, and evaluate the new request. The suffix step is approximate:
C's saved activations still reflect its original prefix B.

The runtime captures checkpoints automatically when enabled, rather than
requiring the model to call a checkpoint tool:

- Server: successful prefill endpoints; it can deliberately stop at
  `input_end - fresh_tail` and again at `input_end`. Batched scheduling can
  capture intermediate prefill endpoints too.
- Agent: after assistant generation rounds and before a private compaction
  exchange; it retains a bounded collection of historical frontiers.
- Storage: immutable chunks, an optional bounded hot-RAM tier, SSD persistence,
  and manifests identifying how to reassemble each endpoint. Successful current
  cache puts synchronously publish to SSD. Live GPU graphs remain separate.

See [payload](../src/engine/deepseek41/state_payload.zig),
[capture](../src/engine/deepseek41/suffix_transaction.zig), and
[storage recipe](../recipes/03-state-cache.md).

## Proposal: capture after each compactable block

The user's proposed boundaries are meaningful: assistant tool calls, tool
results, responses, and other compactable input regions. Capture at their
evaluated token boundaries, rather than assuming each JSON object is already
a model-state boundary.

```text
original: [A] [B] [C] [D] [E]
captures:    ^   ^   ^   ^   ^
edited:   [A] [summary B] [C] [summary D] [E]
```

An endpoint after C lets the matcher select C even though D changed later.
With only a source ending after E, it cannot arbitrarily extract the state
ending after C. More endpoints improve the available choices; the current
one-span matcher still cannot import C and E together in one edit.

If already evaluated at a boundary, capture needs no additional model forward
pass. Capturing boundaries inside a multi-message prefill can require splitting
the computation and introduces additional synchronization points.

Keep message/block identity outside the normal model-visible text where
possible. A temporary sequential compaction index need not cause wholesale
renumbering of the ordinary prompt. Stable identifiers and reproducible
rendering preserve unchanged token regions.

Do not implement this merely by adding an explicit prefix-cache marker to every
message: the current explicit policy creates at most four marked checkpoints
per request and opts out of automatic suffix matching. Automatic boundary
capture and explicit prefix-breakpoint semantics need separate treatment.

## Cost of the present full-snapshot path

Capture currently synchronizes GPU work, serializes the complete payload and
originals, hashes all chunks, then persists missing chunks. Storage deduplication
does not avoid the full serialization/hash work that precedes it. Capture work
therefore grows with history length.

Calculated logical sizes for the native packed-index format, F32 main-cache
wire values, even endpoints, and no optional drafter appendix:

| Evaluated tokens | Canonical payload | Pre-RoPE originals | Combined |
| ---: | ---: | ---: | ---: |
| 8,192 | 51.85 MiB | 5.66 MiB | 57.52 MiB |
| 32,768 | 175.93 MiB | 20.72 MiB | 196.65 MiB |
| 131,072 | 672.24 MiB | 80.96 MiB | 753.20 MiB |
| 200,000 | 1,020.24 MiB | 123.19 MiB | 1,143.44 MiB |

The 8K canonical payload is also a recorded measurement. Other table entries
are layout calculations, not fresh capture measurements. Private approximate
suffix payloads can use the larger index representation. Runtime F16 main KV
does not imply F16 checkpoint wire values.

For these assumptions and N >= 128:

```text
canonical_bytes = 11,002,932 + 5,294 * N
original_bytes  = 675,852 + 257 * sum(floor(N / r) for r in [2, 2, 2, 1])
```

Current raw-ring chunking is another cost: each of 40 layers has a
128 x 512 x 4-byte window, or 256 KiB. Any changed byte makes that entire
layer-window chunk different. New endpoints can therefore retain roughly
10 MiB of new raw-window chunks each, before changing tail pages, logits and
originals. One hundred endpoints can accumulate roughly a GiB of raw-window
versions alone. This is a layout consequence, not a measured workload total.
Changing raw-ring chunk granularity could improve storage sharing but would
not by itself eliminate full capture and hashing.

Historical timing anchor: the old Item9 microbenchmark published two
already-captured sources totaling 63.58 MiB in approximately 192-199 ms,
including synchronous SSD publication. GPU capture was separate. This is not
a fresh five-sample measurement of current per-message or 200K-token latency.

There is also a current server admission gate: a suffix source plus originals
and 8 MiB staging must fit below 512 MiB. Large full snapshots in the table
will be skipped. The agent has its own configured frontier/staging budget.
Dense long-context checkpoints need attention to materialization as well as
capture and retention.

Sources: [state-cache chunking](../src/state_cache.zig),
[source admission](../src/server/persist.zig),
[measured packed checkpoint size](../recipes/14-packed-index-cache.md), and
[historical cache microbenchmark](../speed-bench/experiments/state-cache-hour9-20261003/README.md).

## Proposed cheaper capture and retention

The preferred direction discussed was frequent useful boundaries with less
work per boundary, rather than calling the existing full-snapshot-and-fsync
path after every short message:

- Share previously captured immutable compressed/index history.
- Capture newly appended history and changed frontier state; consider finer
  chunks or deltas for overwritten raw-ring rows.
- Retain recent boundaries in bounded RAM, with less frequent or batched SSD
  publication. Current puts are always persistent; RAM-only capture is a proposal.
- Keep the logical message/archive log independently durable. Losing a KV
  checkpoint should cost recomputation, not lose the agent's factual record.
- Preserve original pre-RoPE values for repeated relocation. They prevent
  accumulated rotation/requantization noise, not stale-prefix influence.

This does not require a generic snapshot framework, a model-operated save tool,
or a multi-span planner as a prerequisite. Start with the boundaries and
capture/storage costs needed for the actual agent workflow.

## Alternative retention tier: bounded SWA reconstruction

DeepInfer documents a complementary lossy recovery mechanism. Retain cached
compressed KV/index blocks; when the local sliding-window state is missing,
replay the preceding 128 tokens to reconstruct an approximate local window.
During replay, local attention for position i starts at
`max(replay_start, i - 127)`. Global compressed history remains available.

For a cached prefix ending at 32,768, replay 32,640 through 32,767, then process
new input. The early replayed tokens lack their original full local context,
so the reconstructed keys/values and frontier are approximate. This can make
older block boundaries usable without retaining complete raw-state versions.

DeepInfer uses bounded replay both for encoder-side cache recovery and for
decoder-side prefill reduction. These are different uses of the mechanism.
The deployment notes describe production cache policy; the public release
does not provide its entire cache-management controller.

Locally, [swa_recovery.zig](../src/engine/deepseek41/swa_recovery.zig) implements
an explicit diagnostic SWB1 block-only import and 128-row scalar replay. It
preserves compressed/index arrays and rebuilds all 40 raw rings. It is not
ordinary SSD admission/recovery policy, and it does not isolate only the
reference's encoder-half recovery. It marks the result approximate and currently
excludes pre-RoPE archive use and suffix grafting. Combining recovery with
relocation needs deliberate implementation and qualification.

Two possible retention choices follow:

| Boundary state retained | Recovery cost / fidelity |
| --- | --- |
| Complete exact frontier | Restore its saved state; pay frontier storage/capture |
| Compressed blocks and tokens, no complete local window | Pay bounded reconstruction; accept approximate state |

Keep recovery of an unchanged cached prefix separate from reusing old activations
after a context edit. They address different missing information and can compose
only with an explicit numerical/provenance policy.

References: [DeepInfer deployment notes](../../DeepInfer/doc/inference_deployment_qa.md),
[local alignment inventory](DEEPINFER_ALIGNMENT.md),
[block-only experiment](../speed-bench/experiments/deepinfer-alignment-20261027/README.md),
and [separate decoder-replay implementation](../recipes/12-bounded-decoder-replay.md).

## Open decisions and a bounded qualification sequence

These are proposed next steps, not completed work or an instruction to start:

1. Measure current boundary capture at representative contexts before changing
   policy. Separate GPU drain/copy, serialization, hashing, SSD publication,
   retained bytes, peak staging and whole-task latency. Compare a few selected
   endpoints with every semantic block, including many short tool rounds.
2. Make capture/retention cheaper where the profile justifies it; keep suffix
   eligibility unchanged for that comparison. Plot frequency versus overhead
   and useful imported coverage.
3. Separately test a less restrictive freshness rule for an unchanged ongoing
   user task. New/changed instructions remain fresh. Retain exact continuation
   behavior and measure ordinary, favorable, corrected-fact and deleted-fact
   cases, including repeated compactions and retrieval of archived originals.
4. Consider block-only recovery for older boundaries as a separate approximation;
   do not silently treat its state as a canonical prefix or a relocatable source.
5. Add multi-span planning only if one-span coverage is a measured limitation
   after the simpler boundary/freshness changes.

Use the project's short experiment loop and existing tools. Performance
comparisons require at least five measured samples per side, alternating order
and retaining variation. Measure practical answers and whole-task savings as
well as delta-NLL/KL; output differences are acceptable tradeoffs, not automatic
rejection. Record actual source availability, source selection, imported/fresh
tokens, fallback causes, restart/eviction behavior and approximation identity.

The user requested preservation of this discussion. No serving behavior,
checkpoint policy, cache format, latest-user cutoff or main bookmark was changed
to create this document.
