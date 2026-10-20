# Suffix reuse after a context edit

## Algorithm and its approximation

An edit `[A old_B C] → [A new_B C]` destroys exact-prefix reuse beyond A.
Evaluate the changed prefix, import cached states for surviving C at its new
positions, and recompute a fresh tail. The cached states of C still encode
their OLD left context. This is the semantic approximation, even if relocation
mechanics are perfectly reproducible.

```text
source = capture_evaluated_checkpoint_with_originals(old_prompt)
new_prompt = apply_edit(old_prompt)
span = longest_eligible_surviving_span(source.ids, new_prompt.ids)
require span ends before latest User and leaves a fresh tail
evaluate new_prompt before span.destination_start
import source span, repairing position-dependent state
evaluate remaining tail and continue generation
```

Our qualified implementation is native DS4.1 text on Metal, one immutable source
and one reused span per edit. Source/destination starts and displacement must
respect pair compression: even starts and even shifts. Both left deletions and
right insertions/growing replacements work. The layout code handles raw rings,
compressed pools/index keys and pending pairs; a superficially matching byte
range is insufficient. See [layout](../ds4_suffix_layout.h),
[matcher](../ds4_suffix_match.h), [public API](../ds4.h) and
`ds4_session_reuse_suffix_relocated` in [core](../ds4.c).

Keep the complete latest User message and following history fresh, as well as
the requested minimum tail. A long current user/tool loop may therefore offer
no reusable span. Record actual imported/fresh coverage, not just the requested
tail. An unsupported layout, unavailable source/originals or insufficient staging
budget falls back to normal evaluation under the selected routing policy.

## Avoid accumulated relocation noise

Capture the normalized BF16 rotary lanes BEFORE RoPE and cache quantization.
Store them separately from canonical KV. On every move, read these immutable
originals, apply the destination rotation once, and quantize the rotary lanes
once. Never make the previously relocated, quantized rotary lanes the new
original. Preserve nonrotary state and rebuild model-specific frontier state.

“Pre-RoPE original” refers to those activation lanes, not BF16 model weights.
It prevents repeated rotation/requantization damage; it does not erase stale
prefix influence or every numerical difference from full recomputation.
Approximate suffix endpoints may retain their own originals and serve later
grafts, but must remain separate from ordinary exact-prefix lookup/export.
The local approximation tag tracks suffix lineage, not every optional routing
intervention.

In [Item8](../speed-bench/experiments/pre-rope-hour8-20261003/README.md), runtime
`52278b7b`, eight even left moves from one unchanged source saved 58.16% edit
latency; originals/checkpoint hashes stayed unchanged. Mean ΔNLL−.00001536,
KL.00001351 and 0/32 greedy changes covered only four teacher tokens per move.
In [Item39](../speed-bench/experiments/recipe-rightshift-hour39-20261008/README.md),
`4cfff6cb`, right 66/left 128/right 514 imports likewise preserved one source;
removing originals caused the right import to reject before session mutation.
These are mechanical diagnostics, not long-agent quality guarantees.

## Local controls

Agent: `DS4_AGENT_CONTEXT_SUFFIX_REUSE=1`,
`DS4_AGENT_CONTEXT_SUFFIX_PREROPE=1`, `DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL=128`.
`DS4_AGENT_CONTEXT_SUFFIX_RIGHT=1` admits right moves. Agent presets also enable
four context frontiers with 512 MiB staging. Legacy pre-RoPE0 requires a mirror
file and native decoder routing; it is not the combined preset path.

Server: `DS4_SERVER_CONTEXT_SUFFIX_REUSE=1`, `_FRESH_TAIL=64`, `_RIGHT=1`
(the latter names share `DS4_SERVER_CONTEXT_SUFFIX`). `_CANONICAL=1` captures
stable full-input source histories; `_PAD=1` may add one old-prefix newline for
even alignment. Padding changes actual input usage. Right support itself
requires originals, not canonical/padding flags. Server imports require a stable
`prompt_cache_key` identity hint, supplied from Conversation ID by default.
Use `context-fast` for suffix reuse with native routing or `persistent-fast`
for the combined package. Right flags are normally off; `persistent-fast`
enables them and accepts explicit 0 overrides.

## What the measurements support

| Workload | Result | Scope |
|---|---|---|
| [Item38 faithful compaction](../speed-bench/experiments/recipe-compaction-hour38-20261008/README.md), `9c6918f0` | 11.66–12.38% whole-task saving | Two READ/edit/WRITE/FINAL coding tasks, routing off, all 10 first codes 161/161; control drift .0081% |
| Item38 standalone teachers | ΔNLL−.01434 to+.01781, mean KL.00227–.01042 | Six synthetic-partition screens,137/174-token real code teachers; one raw fence failure retained separately |
| Item39 growing context | 19.46% whole-task saving /18.80% versus faster control | Two expansion tasks, routing off; all 6 first server writes 161/161; standalone raw fence failures remain failures |
| [Item31 repeated Conversation edits](../speed-bench/experiments/repeated-conversations-hour31-20261004/README.md), `91c9cf21` | 58.24% launcher /66.33% edited-wall saving | One shrinking history,8 deletions, restart after 4,27 replies/reads correct;7 approximate SSD sources |

Item38's requested tails 64/128/256 all imported the same 441/453 tokens and
performed the same total fresh coverage because latest-User protection dominated.
That sweep does not establish 128 as a universal optimum. Standalone edit-only
savings of 66–77% are not whole-task savings.

The [early deletion stress curve](../speed-bench/experiments/suffix-clm-hour1-20261003/README.md)
removed the only authoritative code fact. Fresh evaluation answered UNKNOWN;
reuse with 64/128 fresh tokens answered stale-looking wrong codes, with
ΔNLL+1.690/+.556. Tail256 returned UNKNOWN but still had+.188 ΔNLL. This is a
real tradeoff: useful fact-preserving compaction does not prove forgetting or
safe deletion of facts. Choose a longer replay or full recomputation when that
behavior matters, and keep this stress case separate from normal summaries.

## Replication and credit

Test unchanged, ordinary summary, irrelevant cleanup, growing insertion,
corrected fact and deleted authoritative fact. Sweep fresh tails, verify the
new user is fresh, score fixed assistant teachers and grade free first outputs.
Then run multiple edits, source eviction and process restart. Retain unmatched
geometry as an observed fallback rather than repeatedly retuning until a graft
appears. The DS4 code does not implement the recipe's multi-source/multi-span
dynamic-programming planner.

Credit [Context Language Models / Suffix Cache Reuse](https://arxiv.org/abs/2609.37725)
and the [authors' implementation](https://github.com/facebookresearch/context-language-models/tree/main/suffix_cache_reuse).
[ds4-recipe](https://tangled.org/astrra.space/ds4-recipe), chapter 13, documents
pre-RoPE masters and broader production planning on AMD. DeepSeek's compressed
KV/SWA architecture comes from its [model report](https://arxiv.org/abs/2609.19969);
the exact reconstruction/repair must be rederived for another model/runtime.
