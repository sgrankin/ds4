# Semantic checkpoints for client-side context compaction

**Implemented and live-quality qualified; repeated-edit savings trade against
initial population cost.** This opt-in DeepSeek V4.1 Metal feature lets any client materialize
selective context edits while the server reuses an unchanged later span. It
adds useful checkpoint endpoints and bounded retention, not a server-hosted
compaction agent. Server `persistent-fast` enables the selected package;
ordinary engine defaults, the agent and other presets get no new boundary defaults.

Qualification uses the 1017 IQ2_XXS/Q2_K model on the M5 Max, native expert
routes, packed main/index KV and no DSpark/MTP. Comparative timing freezes
`d5b8e8a3` with high reasoning. Final installed source `9b826389` adds two
control-admission guards and passes retention/padding/thinking-control checks;
those guards change no rendered tokens or inference math in the measured cohort.
The [experiment report](../speed-bench/experiments/context-boundaries-20261028/README.md)
retains the failed smoke, floor-mapping correction, logs and reproduction.

## Why an intermediate endpoint matters

Consider parallel tool results A, C and D:

```text
original: [controls][large A dump][unchanged investigation C][old D fact]
saved ends:                    ^A                         ^C          ^D
edited:   [controls][A excerpt] [unchanged investigation C][new D fact]
```

The useful source must really end at C. A final D checkpoint cannot be made
into a C checkpoint by truncating its token list: raw rings, compression carries
and other frontier state would still describe D. Evaluate the shortened prefix,
relocate the saved C span, then evaluate fresh D and the final tail. The current
matcher imports one span; this is not a general message diff or multi-span planner.

## Implementation sequence

1. **Render once and record boundaries as metadata.** Record byte ends for
   semantic messages, tool invocations and results outside model-visible text.
   Tokenize normally. Map each semantic end to the last existing completed token
   at or before that byte offset, then deduplicate ends. A BPE piece crossing
   `</tool_result>` and `>\n\n` must stay fresh; do not split it or insert a marker.
   Separately round the actual-controls admission end up, so no instruction is
   accidentally admitted before its final token has been evaluated.
2. **Choose a bounded set beyond the evaluated frontier.** Prefer recent ends,
   with a maximum count and minimum token spacing. Do not split ordinary prefill
   merely because a boundary was already evaluated by an exact hit or graft.
   Explicit client cache breakpoints keep their independent policy.
3. **Capture a complete frontier at each selected end.** Keep tokens, logits,
   compressed/index owners, raw windows and pending carries needed to restore
   that actual endpoint. Retain immutable pre-RoPE originals for relocation.
   Deduplicate immutable pages in the existing bounded hot/SSD cache. Automatic
   intermediate endpoints are RAM-only; final ordinary endpoints remain durable.
   This implementation still captures/serializes/hashes a full snapshot, rather
   than collecting only an appended GPU tail.
4. **Match the materialized edited prompt.** Discover sources in the same model,
   runtime policy and client cache scope. Compare token IDs and choose the longest
   eligible unchanged suffix reaching its source endpoint. Require at least 64
   reused tokens and even-compatible relocation starts; right shifts retain the
   existing opt-in and pre-RoPE requirements. Leave the configured final tail fresh.
5. **Import without mutating the source.** Pin its endpoint/chunks, materialize
   it, evaluate the edited left prefix, rotate from immutable BF16 pre-RoPE
   originals, and import the matching span at its destination. Evaluate the
   remaining fresh content. Repeated captures retain private approximate identity;
   an approximate graft must never be published as an exact canonical prefix.
6. **Fall back normally when unavailable.** Release pins after the transaction.
   Hot-budget eviction retires complete unpinned transient endpoints, rather than
   leaving partial sources. Process restart loses RAM-only checkpoints. Recompute
   from the client's current view when no eligible source exists.

The resulting activations still carry influence from their original prefix.
Pre-RoPE originals prevent accumulated rotation/requantization drift; they do
not erase that influence. Grade changed/deleted facts and instructions, rather
than assuming a token match establishes semantic equivalence.

## Ongoing-task freshness is a separate decision

The old approximate-suffix cutoff is the latest actual user marker. That can
exclude C during one long autonomous task. The opt-in ongoing policy scopes
sources by **ordered actual system/developer/user controls plus rendered tool
declarations and response schema**. Synthetic User wrappers around tool results
are not new user instructions. Repeated identical user turns remain distinct
ordered controls; do not hash a set of unique strings.

Only sources captured after every actual control was evaluated receive this
task identity. New, changed or deleted controls/declarations invalidate it.
Late controls keep the admission frontier conservative. A new user message
remains fresh; ordinary prefix reuse before it can still work. This is a
conservative eligibility rule, not proof that old activations are exact.

The automatic path currently applies to full materialized text requests.
Explicit breakpoints, images, live tool-state requests and stored
`previous_response_id`/conversation materialization keep their existing paths.
The underlying suffix transaction also excludes steering, TP, decoder replay
and DSpark/MTP; canonical MTP persistence is qualified separately.

## State and factual memory have different owners

The client keeps an append-only message/archive log and materializes a shorter
working view. Archive records retain the original dump and an identifier that
the agent can retrieve later. A deleted KV checkpoint costs recomputation; it
must not delete the factual record. Our retrieval control actually rereads the
saved original JSON and supplies its dump as a fresh tool result. It does not
claim the model autonomously chose a memory search or that the server hosts it.

RAM retention is bounded, not a guarantee to keep C indefinitely. Selected
sources are pinned during use and all materialized chunks get fresh recency.
Candidate enumeration also touches token-history chunks; aggregate cache reads
are not useful-graft hit counts. Logs expose geometry and aggregate evictions,
not a per-endpoint eviction history.

## Configuration and reproduction

| Setting | Meaning/default |
|---|---|
| `DS4_SERVER_CONTEXT_SUFFIX_REUSE=1` | Enable the underlying suffix feature |
| `DS4_SERVER_CONTEXT_BOUNDARIES=1` | Enable automatic semantic capture; server `persistent-fast` enables it |
| `DS4_SERVER_CONTEXT_SUFFIX_ONGOING_USER=1` | Enable actual-control-scoped ongoing reuse; server `persistent-fast` enables it |
| `DS4_SERVER_CONTEXT_BOUNDARY_LIMIT` | At most 4 recent ends by default; cap 16 |
| `DS4_SERVER_CONTEXT_BOUNDARY_MIN_TOKENS` | Engine default2048; server `persistent-fast` and trial512 |
| `DS4_STATE_CACHE_HOT_MB` | Server `persistent-fast`/trial512; explicit0 disables automatic capture |
| `DS4_SERVER_CONTEXT_SUFFIX_STAGING_MB` | Source plus original tape plus 8 MiB staging must fit; default 512 |

Use a stable per-conversation `prompt_cache_key` and keep full request input
available for fallback. The foreground [server helper](../speed-bench/experiments/context-boundaries-20261028/serve.nu)
fixes the qualified settings, clears inherited experimental variables and uses
frozen adjacent shaders. Caller owns its guarded lifetime:

```nu
python3 speed-bench/ideas/pressure_run.py GUARD -- nu speed-bench/experiments/context-boundaries-20261028/serve.nu FROZEN/bin/ds4-server FRESH_SERVER_DIR --ongoing 1
nu speed-bench/experiments/context-boundaries-20261028/run.nu http://127.0.0.1:18355 deepseek-v4.1-flash CLIENT_OUTPUT --case corrected --rows 24 --edits 1 --archive-control
```

The observed coordinator quantum is 2048 despite configured prefill 8192. Early
semantic stops can add model passes, not just serialization cost. Keep initial
population, every edit, capture costs, generated lengths and retained coverage
in the whole-workflow measurement. A short exact prefix anchor can amortize over
later edits; it is not free to create.

The server preset is available with:

```nu
./ds4-local --server persistent-fast --model gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf --prefill-chunk 8192
```

Explicit boundary/ongoing `0`, limit, spacing and HOT values override the new
preset settings. The wrapper keeps its existing64GB expert target, resident
substitution policies and512 default chunk; the command explicitly selects the
8192 configured chunk used for qualification. The five-round benchmark instead
uses native routes and32GB, so this command is not an exact measured package.
No chunk default changes. The ordinary coordinator still splits at2048.

## Completed quality and measured tradeoff

| Check | Actual evidence |
|---|---|
| Floor mapping/control/cache fixtures | Final 14 renderer/control cases and 4 cache-policy cases pass, including reasoning-enabled identity and padding-adjusted admission; production transient/pin/eviction/promotion/restart/packed-format checks pass |
| Same-runtime freshness off/on | Both answer CEDAR correctly; off 410 exact tokens/no aligned suffix, on imports 1,688 C tokens from a durable source |
| Parallel semantic endpoint | C captured at 2872 as RAM-only; 1,830 tokens imported from 1042:2872→488:2318; all four replies correct |
| Changed/deleted facts and user controls | All 10 initial/edited replies pass: corrected LARCH, deleted NONE, new/changed user MAPLE, deleted user returns remaining CEDAR authority |
| Archive/protocol | Original archived CEDAR remains retrievable while working fact is LARCH; READY→MAPLE `previous_response_id` continuation passes |
| Actual restart | Transient C unavailable; cached=0/no-aligned-source, fresh evaluation correctly returns LARCH |
| Eight-edit retention | Final installed source9b826389 passes 18/18 replies; all 16 edited grafts reuse original RAM-only C with no capture errors; functional check, not comparative timing |
| Final admission guards | Padded CEDAR→LARCH2/2 correct with actual `prefix_pad=1`; medium→high2/2 correct with thinking on, changed high request cached0/no aligned source; CPU separately distinguishes medium/none |

These smokes establish functionality and narrow practical quality, not speed
or universal factual equivalence. They do not require identical reasoning/IDs.
Final shutdown has no pinned endpoints, hot bytes536,007,540 within512 MiB and9
aggregate evictions including later QA; this is not a per-C eviction history.
The complete 15-process campaign compares off/sparse1/fuller4, five fresh samples
per setting in alternating order, ordinary serial and favorable parallel cases,
rows=24 and initial population plus **two edits**. The preceding rows=72/eight-edit
run was stopped as runtime calibration and remains separate. The completed eight-edit
retention diagnostic cannot establish five-pair eight-edit speed; any extension
of the timing curve is labeled a calculated projection. Across both workflows,
fuller4 initial population costs 80.005→145.486 s (+81.8%), the measured two edits
158.755→98.390 s (−38.0%), and whole initial+two edits 238.761→243.876 s (+2.14%).
All 90 answers pass; reasoning lengths differ slightly. Sparse1 imports the same
C spans but lacks the early exact anchors and provides no material edit-time
gain. All five rounds, order effects, sample variation, storage and the Pareto
plot are in the [measured report](../speed-bench/experiments/context-boundaries-20261028/RESULTS.md).

At a constant observed edit cost, calculated break-even is 2.17 edits: the third
edit/fourth total request would first benefit. This assumes retention and similar
future edits. Retention passes for this sequence, so fuller4/spacing512 is enabled
only in server `persistent-fast`. This remains a scoped engineering choice,
not a measured eight-edit speedup or a universal default.

```nu
python3 speed-bench/ideas/pressure_run.py PARETO_GUARD -- nu speed-bench/experiments/context-boundaries-20261028/run-five.nu FROZEN/bin/ds4-server FRESH_PARETO --include-sparse --spacing 512 --rows 24 --edits 2
```

[Current status and drivers](../speed-bench/experiments/context-boundaries-20261028/README.md).

## Credit and related mechanisms

Shelley's [Automatic Compaction](https://blog.exe.dev/compaction), by Philip
Zeyliger and Maisem Ali, motivates model-selected edits backed by an append-only
archive and retrievable originals. This feature supplies cache support for that
client pattern, rather than implementing Shelley's harness or adopting its
approximately 200K trigger as a local default.

[DeepInfer's pinned deployment description](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md#L179-L191)
separates durable compressed/index blocks from recoverable local SWA state.
Our first slice retains complete frontiers; its separate 128-token SWB1 recovery
prototype is not integrated with these suffix sources. This is not a port of
their omitted production cache controller.

The local building blocks are [suffix/pre-RoPE reuse](02-suffix-reuse.md),
[immutable hot/SSD storage](03-state-cache.md) and
[physical main KV](15-packed-main-kv.md). The [design journal](../docs/CONTEXT_COMPACTION_AND_CHECKPOINTS.md)
preserves the original discussion; this recipe states the implemented subset.
