# Planning A → B → C — 2026-10-26

A/B/C complete within their one-hour caps. The constructor fixes a demonstrated
question-targeting defect, and explicit JSON persistence passes restart/ownership
qualification. Natural planning remains mixed: C never calls stash/load in this
screen, so its faster unfamiliar result is not a cache speedup. All 24 natural
control/A/B/C answers have correct numeric fields and clean JSON.

A begins at 00:38 UTC with a 01:38 checkpoint, including measurement. Baseline
runtime is `f4f5381025370faef25638517e736feb6b107e3a`; its executable was copied to
`.workspaces/planning-abc-20261026/a-baseline/bin/ds4-server` before candidate
builds. A completes all 12 control/candidate samples: ordinary/favorable improve
7.8%/6.4%, while unfamiliar slows 13.5% and its generated helper questions lose
their record selectors. All final answers remain correct. This is a mixed
tradeoff, not a blanket planning improvement.

The [previous fixed screen](../mcp-tools-20261025/RESULTS.md) remains a historical
baseline: ordinary main39.791/direct92.241/code-available69.689 seconds; favorable
main57.931/direct59.211/code105.714 seconds. Ordinary code availability invokes no
tool; favorable code dumps the queue, refetches and emits all helper answers.
These known numbers are not new runs of candidate A.

## Matched setup and limits

The new control/candidate comparison uses the same ordinary/favorable fixtures,
plus an unfamiliar export of the same eight favorable records. The new case's
task, policy, source notes, charge values, question template and gold oracle are
unchanged. Only the source field layout/tool contract differs. The original tool
retains its old schema/result without a new output schema.

M5 Max, 128 GiB; DeepSeek V4.1 Flash 1017 IQ2_XXS/Q2_K imatrix, SSD streaming
with a 32 GB expert cache, and resident Clef-Flash INT4. Context is 16,384;
prefill/mixed quantum 2,048, MTP draft one and two sessions. Every request uses
temperature zero, eight tool calls and the same shared 4,096 output-token budget.
Native DeepSeek HIGH reasoning remains enabled; no override or separate reasoning
cap. Helper prefix caching, decoder replay and suffix reuse are disabled. Loading
the model is outside request wall time; main cached tokens are zero throughout
the control.

Each frozen binary runs as a sequential block: ordinary twice, favorable twice,
then unfamiliar twice. Control precedes candidate, with a model-server restart.
This is not a balanced cross-binary ABBA experiment. OS file-cache warmth and
expert residency can affect wall time, so report token/round/evidence changes
alongside it. Two repeats of fixed inputs do not establish broad generality,
kernel performance or numerical equivalence. Output quality is measured directly
rather than requiring identical generated text.

## Fresh frozen control

Wall values are medians and observed ranges of two runs. Tokens are main input /
all generated output / reasoning included in output. Counts are identical within
each repeated case. MCP counts exclude initialize/list discovery, which still
runs per configured request and is included in wall time.

| Case | Median wall | Range | Input / output / reasoning | Programs / fetches / helpers | Main-visible emitted bytes |
|---|---:|---:|---:|---:|---:|
| Ordinary | 67.933s | 67.395–68.471s | 1,253 / 727 / 642 | 0 / 0 / 0 | 0 |
| Favorable | 105.696s | 104.784–106.607s | 3,300 / 696 / 416 | 2 / 2 / 1 | 10,887 |
| Unfamiliar | 121.344s | 121.052–121.636s | 3,410 / 899 / 530 | 2 / 2 / 1 | 11,308 |

All six control answers complete with correct IDs/count/maximum and clean JSON.
Each handoff has two grounded sentences without invented completed remediation.
The unfamiliar handoff qualifies its status statement as not confirmed locked or
refunded, rather than asserting that no refund happened. No response exhausts its
output/tool budget or encounters a native/program error. The control guard exits
cleanly with normal pressure and no swap growth.

Ordinary makes no tool invocation: its cost remains tool availability and model
deliberation. Favorable reproduces the earlier full-queue emission, second fetch
and probability dump. Its helper classifies all eight records correctly (TP3,
TN5, FP0, FN0), with 2,573 helper input tokens and 1.463/1.529 seconds helper time.

Unfamiliar also adapts successfully, but after emitting the full 9,642-byte export.
Its second program resolves `payloads[index.dataset_key]` dynamically, reads the
nested ticket fields and builds targeted questions. It does not hardcode the
opaque payload key, so this is real layout adaptation. It nevertheless fetches
again and emits 1,666 bytes of complete helper answers/probabilities. It also
copies current notes into question instructions as well as the shared helper
input, increasing helper input to 3,851 tokens. Labels remain TP3/TN5/FP0/FN0;
helper time is 1.867/2.030 seconds. These are repeated records, not 16 independent
classification examples. Source adaptation is preserved separately from the
inefficient bulk-result boundary.

Raw control requests, responses, actual code, quality grades and metrics remain
under `.workspaces/planning-abc-20261026/a-control/`.

## A: truthful contracts and output schemas

Frozen candidate is jj change `zptwwnoprwmv`, snapshot `4abc57b3`; subsequent
edits are reports only. It removes misleading claims that an identifier binding
contains full evidence, replaces the first-value/helper example with coherent
fetch/classify/reduce guidance, and imports MCP output schemas into the visible
contract. Runtime globals, native inference and task inputs are unchanged.

| Case | A median wall | Range | Change from control | Input / output / reasoning | Main-visible emitted bytes |
|---|---:|---:|---:|---:|---:|
| Ordinary | 62.642s | 62.559–62.725s | −7.8% | 1,258 / 610 / 521 | 0 |
| Favorable | 98.881s | 97.172–100.590s | −6.4% | 3,614 / 557 / 243 | 11,458 |
| Unfamiliar | 137.699s | 135.034–140.364s | +13.5% | 4,299 / 1,090 / 751 | 11,918 |

The ordinary arm again invokes no tool. Its prompt is five tokens larger rather
than shorter, but reasoning/generated output falls. Both tool cases still use
two programs, two fetches and one helper. They first emit the entire MCP result,
then refetch and emit the full helper result. A additionally calls
`text(JSON.stringify(...))`, double-encoding those values. The intended compact
dataflow is not adopted. Favorable native labels remain TP3/TN5/FP0/FN0.

Unfamiliar preserves correct dynamic `payloads[index.dataset_key]` navigation,
but generates this question construction in both repeats:

```js
{ name: t.ticket_ref,
  instructions: `Target ticket ID ${t.ticket_ref}. Apply this policy: ...`,
  ...schemas.triage }
```

The final spread overwrites the targeted instructions with the template's generic
instruction. Native audit confirms eight different labels but **one identical
instruction with no record selector**. Every helper answer is active: TP3, FP5,
TN0, FN0 in both repeats. This is a generated question-construction defect,
not evidence that Clef misclassified correctly targeted questions. The main
model nevertheless returns the correct three IDs after having received all
original notes. Final-answer success therefore masks failed delegation; retain
the native confusion separately from the final numeric grade.

All six A answers are clean JSON with correct IDs/count/maximum and grounded
two-sentence handoffs. No runtime error, truncation or budget exhaustion occurs;
main cached tokens remain zero. Both control/candidate guards exit cleanly with
normal pressure and no swap growth. No prompt tuning follows these observations
inside A. Truthful contracts/schema metadata are retained, while the repeated
selector-assembly failure provides a concrete target for B's question constructor.
Raw A records remain under `.workspaces/planning-abc-20261026/a-candidate/`.

## B: targeted-question constructor

B begins 01:11 UTC with a 02:11 checkpoint and completes the same six requests.
Frozen source is jj change `xnnkvzyv`, snapshot `cc87a8bf`, following A checkpoint
`77831bba`. It adds one pure-JS `lib.questionsFor(rows,template,{idKey,target})`
constructor and a short contract. No fixture, task, budget or inference change.
The executed test binaries pass 13 QuickJS and 35 hosted checks.

| Case | B median wall | Range | Change from A / original control | Input / output / reasoning | Main-visible emitted bytes |
|---|---:|---:|---:|---:|---:|
| Ordinary | 53.766s | 53.654–53.878s | −14.2% / −20.9% | 1,278 / 481 / 396 | 0 |
| Favorable | 108.139s | 107.805–108.473s | +9.4% / +2.3% | 3,381 / 733 / 445 | 11,074 |
| Unfamiliar | 128.147s | 127.995–128.299s | −6.9% / +5.6% | 6,914 / 768 / 429 | 24,718 |

Ordinary again invokes no tool; its improvement accompanies less deliberation,
not constructor execution. Both tool cases actually use `lib.questionsFor` and
produce eight distinct targeted instructions. Native labels are TP3/TN5/FP0/FN0
in every tool-case repeat, fixing A's unfamiliar selector-overwrite failure.
Unfamiliar retains dynamic index lookup, normalizes nested rows in ordinary JS,
then calls the constructor with `idKey:'id'`. No final-main rescue of an observed
helper error is needed in B.

Dataflow remains inefficient: both tool cases still execute two programs/two
fetches/one helper. Unfamiliar first emits the full export, then emits
`JSON.stringify({rows,qs,answers})`, including all notes, generated questions and
probabilities. Its 24,718 emitted bytes are 9,820 plus 14,898; main input grows
despite fewer generated/reasoning tokens. The favorable latency regresses versus
A. This is a useful correctness helper, not a demonstrated compact workflow or
general speedup. Native helper times are 1.406/1.521 seconds favorable and
1.478/1.532 unfamiliar; source data and oracle are unchanged.

All six B final answers have correct numeric fields, clean JSON and grounded
two-sentence handoffs, with no runtime error or truncation. Main cached tokens
remain zero. The guard exits cleanly at normal pressure with no swap growth.
Raw evidence remains under `.workspaces/planning-abc-20261026/b-candidate/`.

## C: immutable JSON values — qualified foundation

C begins 01:32 UTC with a 02:32 checkpoint. Frozen source is jj change
`wrxyrlnw`, snapshot `eb80fd0b`, following B checkpoint `839fac52`. Explicit
`values.stash(value)` returns an opaque reference; `values.load(ref)` returns a
fresh JSON copy. Programs retain fresh JS contexts. Inspection is ordinary JS
over loaded values (keys/counts/samples), without a separate inspection DSL.
The serialized ledger is capped at 8 MiB; failed programs discard partial stashes.

The request owns its ledger. New snapshots appear in public
`program_output.ds4.values`, outside the compact logical result. Stored Responses
also retain the complete private ledger, so descendants inherit values even when
they create none. Conversations reconstruct values from retained program receipts;
deleting the owning complete pair removes those values from that Conversation.
An ephemeral WebSocket parent owns its ledger on that connection. References
carry data, not tool authority, native handles, MCP sessions or inference state.

Executed C checks pass: 16 QuickJS/ledger and 35 hosted tests, full host/script
verification, then **37 live Flint seed checks plus 11 after an actual process
restart**. SDK `Response.model_validate`/`model_dump` retains value snapshots.
The wire checks establish immutable loaded copies, null/error-shaped data,
stored child inheritance, explicit public replay without refetch, sibling
isolation, failed-program rollback, HTTP `store:false`, connection-local WS
continuation/reconnect boundaries and Conversation restart/deletion ownership.
Exactly one MCP source fetch occurs across the seed/replay/load checks.

| Forced qualification step | Model-visible program result | Newly snapshotted data | Private ledger | Stored record |
|---|---:|---:|---:|---:|
| Fetch/stash three values | 128 B | 2,838 B | 2,838 B | 16,636 B |
| Fresh-program load/mutate/reload | 96 B | omitted | 2,838 B | 14,206 B |
| Load after process restart | 96 B | omitted | restored from parent | not stored |

The source-only marker never enters either stored logical replay. The child
retains all three values despite emitting no new snapshot; after restart it
returns the original two-record count, an unmodified fresh copy, literal null
and an ordinary error-shaped object's data. Bulk remains in client-visible audit
and snapshots/private storage, so context savings do not imply smaller durable
records. These are explicitly supplied integration programs, not speed or
automatic-adoption evidence.

Two earlier qualification attempts are retained: v1 reused a generic current
user message while changing system code, and Flint treated earlier execution as
already satisfying it; v2 generated nested `{arguments:{code:...}}`, rejected by
the unchanged strict invocation contract. The v3 driver places code in the
current user turn and explicitly states `{code:source}`. No native runtime or
reasoning-policy change was made to obtain the pass. Raw failures remain in
`c-wire/` and `c-wire-v2/`; successful data/restart evidence is in `c-wire-v3/`.
Both Flint guard processes stop cleanly. DeepSeek independently passes all ten smoke checks with
native reasoning, and its guard stops cleanly. Raw smoke evidence is in
`c-ds4-smoke/`. A separate fresh engine runs the unchanged natural screen below.

## C: natural planning and actual adoption

All six C samples complete by 02:02 UTC, within C's hour, using the frozen
runtime and the same fixtures/budgets. The screen restarts the engine after
qualification, preserving the fresh-engine block setup used for A/B.

| Case | C median wall | Range | Change from B / original control | Input / output / reasoning | Programs / fetches / helpers | Emitted bytes |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary | 57.020s | 56.987–57.052s | +6.1% / −16.1% | 1,334 / 542 / 465 | 0 / 0 / 0 | 0 |
| Favorable | 109.047s | 108.709–109.385s | +0.8% / +3.2% | 4,557 / 750 / 473 | 2 / 2 / 1 | 15,264 |
| Unfamiliar | 75.432s | 75.162–75.702s | −41.1% / −37.8% | 3,524 / 520 / 364 | 1 / 1 / 0 | 9,642 |

**No C natural program calls `values.stash` or `values.load`, and no value
snapshot is created.** Ordinary again answers directly. Favorable still fetches
twice: first it emits the full queue, then uses `lib.questionsFor` and emits all
questions and helper answers. Its native labels remain TP3/TN5/FP0/FN0 in both
repeats, while the emission grows to 15,264 bytes.

Unfamiliar changes plan to one fetch and `text(res)`, then the main model answers
directly from the full export. It skips the second program/fetch and the helper;
there are no native classification labels to grade for this case. This accounts
for the observed workflow difference; the 41.1% reduction versus B cannot be
attributed to reference reuse. The 9,642-byte source still enters main context,
so the compact-evidence objective is not achieved there either.

All six final handoffs have two grounded sentences and no invented completed
remediation. Final numeric/JSON grades pass, no program/native error or truncation
occurs, and main cached tokens remain zero. The screen guard stops cleanly with
normal pressure and no swap growth. Raw evidence is under `c-candidate/`.
Among the measured code-mode configurations, no point dominates all three fixed
cases: B is fastest ordinary, A fastest favorable, and C fastest unfamiliar via a
different main-model plan.
Explicit persistent values are implemented/qualified, while efficient natural
use remains unestablished. No extra prompt tuning or model test follows this
bounded sequence.

The unfamiliar result's truthful output schema and returned index describe its
nested original notes/charges/IDs; no gold labels or summary enter either model.
Native helper/JS correctness and practical task quality remain separate. Extra
inspection is allowed when useful, including inside one program. C's explicit
persistence remains useful independently of natural adoption; its qualification
and automatic-planning observations above are distinct results.
