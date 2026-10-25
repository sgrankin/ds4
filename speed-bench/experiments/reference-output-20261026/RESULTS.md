# Reference-output boundary — 2026-10-26

The initial nine requests are measured. Disabling the boundary completes all
three tasks; both enabled limits fail both source-fetching tasks before a final
answer. References preserve data and reduce bulk/main input, but repeated
inspection and re-emission exhaust the shared call budget. A contextual hint also
fails all four follow-up requests at the same eight-call allowance. Failed
completion is not a speedup. All 17 requests, including the bounded twelve-call
diagnostic, finish within the **02:20–03:20 UTC** hour. Normal serving remains C;
the boundary experiment stays separate and disabled by default.

Frozen initial source: jj change `nvrwnmrw`, snapshot `95748ae5`, parent
`aef775c2` (completed C). Default-zero schema is byte-identical; executed focused
tests pass 55 checks and full host/script verification passes.

## Matched setup

Reuse the ordinary/favorable/unfamiliar planning tasks, schemas and oracle.
Change only PTC `ds4.max_inline_bytes`: disabled `0`, then `512` and `2048`.
Each point initially has one request per case. Keep native DeepSeek reasoning,
temperature zero, eight tool calls and the shared 4,096 output-token budget.
Record the frozen server build, engine/cache settings, guard outcome and actual
cached-token counts alongside results. Source/cache settings are inherited from
the [planning screen](../planning-abc-20261026/RESULTS.md), not newly measured by
this document. Actual initial order rotates the limits across cases:
O0, F512, U2048, O512, F2048, U0, O2048, F0, U512. The same guarded engine runs
all nine; main cached-token counts are zero. Limits are **per serialized
`text(value)`**, not a whole-program byte allowance. The existing 64 KiB program
result cap remains. Emitted bytes below sum all program-result wrappers across
the response, so they can exceed an individual value's limit.

## Initial nine-request screen

| Case | Inline limit | Time, s | Status | Main input / generated / reasoning | Programs / fetches / helpers | Emitted bytes | Numeric / JSON / grounding |
|---|---:|---:|---|---|---|---:|---|
| Ordinary | 0 | 57.248 | complete | 1,334 / 542 / 465 | 0 / 0 / 0 | 0 | pass / pass / pass |
| Favorable | 0 | 108.402 | complete | 4,557 / 750 / 473 | 2 / 2 / 1 | 15,264 | pass / pass / pass |
| Unfamiliar | 0 | 74.759 | complete | 3,524 / 520 / 364 | 1 / 1 / 0 | 9,642 | pass / pass / pass |
| Ordinary | 512 | 51.486 | complete | 1,402 / 445 / 363 | 0 / 0 / 0 | 0 | pass / pass / overstatement |
| Favorable | 512 | 104.639 | failed | 1,690 / 936 / 329 | 7 / 1 / 1 | 1,250 | no final answer |
| Unfamiliar | 512 | 127.614 | failed | 2,129 / 1,378 / 527 | 7 / 2 / 0 | 756 | no final answer |
| Ordinary | 2048 | 62.394 | complete | 1,403 / 587 / 506 | 0 / 0 / 0 | 0 | pass / pass / pass |
| Favorable | 2048 | 93.108 | failed | 1,668 / 777 / 258 | 7 / 1 / 1 | 1,122 | no final answer |
| Unfamiliar | 2048 | 124.648 | failed | 2,379 / 1,143 / 360 | 8 / 1 / 0 | 1,781 | no final answer |

Five numeric/JSON grades pass; four enabled source-fetching tasks fail. The
ordinary arms invoke no program, so their time/wording differences accompany the
enabled policy's prompt, not spilling or cache execution. Ordinary512 asserts
that no account was locked and no refund issued, which the notes do not
establish. The other completed handoffs are two grounded sentences. Disabled
control tokens and observed times reproduce C's plans closely.

## Inspection and decision

All four failed points actually load automatically created references; none
manually calls `values.stash`. Large emissions remain exact in value snapshots,
including JSON-encoded strings. The boundary works, but loading does not ensure
the model computes a compact handoff:

- Favorable512 loads a two-record sample, then emits it unchanged and creates
  another reference. A correct helper batch follows (TP3/TN5/FP0/FN0), but the
  program stores `{questions,answers}` where `answers` is the full native result.
  Later code treats `x.answers` as an array rather than using its nested
  `x.answers.answers`, producing a TypeError before the final budget block.
- Favorable2048 repeatedly exports all records, inspects the question template,
  then gets a correct helper batch. Its next outer program is blocked by the
  shared budget before it can reduce the answers.
- Unfamiliar2048 successfully inspects keys, the dynamic dataset index and ticket
  count. It then exports all notes, exports all questions, loads/texts the same
  question array again, and inspects one question. The eventual helper attempt
  cannot run within the remaining budget. This is repeated export/inspection,
  not a failure to load references or navigate the layout.
- Unfamiliar512 first generates a SyntaxError. Its repaired inspection emits a
  JSON string, which the cache preserves as a string. It eventually parses it
  correctly but confuses that projected inspection object with the original
  export, refetches, and repeatedly emits a 603-byte string under a 512-byte
  limit. No helper executes before budget exhaustion.

Program counts include the final blocked outer attempt. The four failures
contain one generated SyntaxError, one generated TypeError and four terminal
call-budget blocks. Native calls themselves succeed. No data loss, wrong-type
coercion or new tool-authority path is observed. Successful reference loads
avoid some refetches, but none of these failed plans is a practical task-quality
tradeoff yet. Raw HTTP output also grows as receipts retain snapshots: for
example favorable2048 returns 90,601 B versus 66,444 B disabled, despite much
less main-visible output.

No reasoning items are exposed in these Responses; interpretation above follows
the executed code/results rather than guessing hidden thoughts.

## Contextual-hint follow-up

Frozen source `5df95bae` changes only descriptor feedback from initial
`95748ae5`. Ref/type fields, parser, tasks, native container schemas and budgets
stay unchanged. Both full and minimal descriptors add:

> Load this ref for computation or tool inputs. text(loadedValue) will spill
> again; emit a smaller projection or reduced result.

Each case repeats twice at 2,048 bytes and eight shared calls. All four still
fail before a final answer; they are not pooled with the initial nine.

| Case/repeat | Time, s | Main input / generated / reasoning | Programs / fetches / helpers | Emitted bytes | Final / native quality |
|---|---:|---|---|---:|---|
| Favorable 0 | 127.035 | 2,289 / 986 / 359 | 7 / 1 / 1 | 3,242 | no final; TP3/TN5 |
| Favorable 1 | 145.119 | 2,301 / 1,214 / 627 | 7 / 1 / 1 | 3,279 | no final; TP3/TN5 |
| Unfamiliar 0 | 133.915 | 2,425 / 1,202 / 467 | 7 / 1 / 1 | 2,003 | no final; TP2/TN4/FP1/FN1 |
| Unfamiliar 1 | 122.151 | 2,376 / 1,113 / 450 | 7 / 1 / 1 | 1,682 | no final; TP3/TN5 |

There is genuine progress alongside remaining re-spills. Favorable reaches a
correct helper result earlier and shows all three qualifying IDs to the main
model. It then repeatedly tries to expose the three selected full notes. The
blocked final attempts finally narrow to 400-character previews (repeat0) or one
record (repeat1), which would fit 2,048 bytes. Unfamiliar1 similarly reaches
correct helper answers and is blocked just before emitting its small native
result. This motivates testing the existing call allowance at twelve, rather
than assuming the reference feature loses data or changing another message.

Unfamiliar0 has a separate, concrete generated handoff defect. One program maps
`narrative:{current:note}` to `narrative:note`; its consumer still reads
`t.narrative.current`. That evaluates to undefined, so JSON serialization omits
every note from the helper input. The helper receives IDs/charges without the
source evidence and produces TP2/TN4/FP1/FN1, missing Q-LF6 and selecting Q-MR2.
The cache retained the projected value exactly. This is producer/consumer shape
confusion, not cache corruption or correctly targeted/evidenced model accuracy.
More calls could complete a wrong answer unless the evidence is repaired.

All four hint trials actually load automatically spilled values; none manually
stashes a value. Each ends at the shared call limit, without a user-facing
handoff or numeric/JSON success. The next diagnostic changes only the existing
`max_tool_calls` from eight to twelve, retaining the hint and 2,048-byte limit.
Its quality and completion will be judged separately; extra allowance is not
assumed to preserve evidence or make these plans practical.

## Existing-resource diagnostic: twelve shared calls

The same `5df95bae` runtime/hint and 2,048-byte limit run with only the existing
`max_tool_calls` raised from eight to twelve. No task, recipe or native schema
changes. Two favorable repetitions complete; both unfamiliar repetitions fail.

| Case/repeat | Time, s | Status | Main input / generated / reasoning | Programs / fetches / helpers | Emitted bytes | Final / native quality |
|---|---:|---|---|---|---:|---|
| Favorable 0 | 159.163 | complete | 2,538 / 1,184 / 575 | 6 / 1 / 1 | 4,556 | correct JSON/grounded; TP3/TN5 |
| Favorable 1 | 141.263 | complete | 2,251 / 1,155 / 491 | 5 / 1 / 1 | 3,039 | correct JSON/grounded; TP3/TN5 |
| Unfamiliar 0 | 245.776 | failed | 3,402 / 2,162 / 1,014 | 11 / 1 / 1 | 5,796 | no final; TP3/TN5 |
| Unfamiliar 1 | 239.540 | failed | 3,490 / 2,020 / 1,006 | 11 / 1 / 1 | 6,694 | no final; TP3/TN5 |

Both favorable samples actually load retained source/helper data and eventually
emit a bounded result. Repeat0 narrows the three selected notes to 500-character
excerpts. Repeat1 performs the final ID filtering/count/maximum computation
inside JS and emits the requested JSON. All current-note facts needed for the
handoffs remain available; both final handoffs have two grounded sentences.

Their medians are 150.213 seconds, 2,394.5 main input tokens and 3,797.5 emitted
bytes: **38.6% slower**, **47.5% less main input** and **75.1% less emitted data**
than the single disabled favorable control. Full HTTP size instead rises from
66,444 B to a median of 77,017.5 B because receipts retain snapshots. This is a
context/latency tradeoff, not a speed win. Importantly, the successful plans use
only eight/seven actual calls, so the higher allowance is not proven necessary
for either trajectory. Opaque references and generated trajectories vary;
completion here cannot be causally attributed to raising the cap alone.

Unfamiliar0 repeatedly inspects and re-exports values. It changes ticket fields
to `id/charge/narrative`, then initially consumes them as the original nested
shape, causing a caught TypeError. It repairs the mapping and obtains a correct
helper batch. More question inspection and full-note emission consume the
twelve-call allowance; the final blocked program finally requests 260-character
snippets that would fit. More allowance still does not provide a practical
completed answer on this trajectory. There is no cache corruption or lost
stored source: the generated producer/consumer contracts and repeated inspection
remain the limitation. No further design iterations are planned in this hour.

Unfamiliar1 likewise gets correct helper labels but confuses its projected
`charge/narrative` fields with the original nested fields, causing a caught
TypeError. It repairs this and inspects the three selected notes individually,
then exhausts twelve calls while trying to inspect the excluded high-charge
record. Neither unfamiliar trajectory produces a final answer. Cache loads
preserve the stored data; generated handoffs and repeated verification remain
the problem.

## Decision

Keep normal C serving installed and do not promote an inline limit into a fast
preset. The separate local `reference-output` experiment retains opt-in context
control with default zero; its `hint-build` binary is available for explicit
testing. Favorable demonstrates a completed context/latency tradeoff, without
proving the higher cap caused success. Unknown-layout natural use is unreliable
even at twelve calls. No general speed or quality gain is established. Both
owned guard runs exit zero with normal pressure and zero swap growth. No further
trials or design iterations follow this one-hour screen.

One request per point is an initial screen, not a stable speed or quality claim.
Record any useful repeat, then choose a measured tradeoff or keep the boundary
disabled. Raw requests, responses, fixture calls and wire output stay ignored in
`.workspaces/reference-output-20261026/`.
