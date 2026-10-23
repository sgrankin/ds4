# Spend reasoning where the task phase needs it

## Adopted interface: a request-local soft budget

An agent often has distinct phases: understand/read, implement after a successful
read, and summarize a completed write. Let the caller set a thinking budget for
that known phase. Preserve stable system instructions so changing the policy
does not discard useful exact-prefix state.

Our Responses extension is `ds4_reasoning_budget` (integer 0..65536), qualified
for native DS4.1 text. Zero means UNLIMITED/default thinking, not no thinking.
Positive values count newly generated non-control thinking tokens, then force
one validated closing-think token at a safe DSML/render boundary with enough
remaining output slots. Count state survives recovery; defer unsafe closure.
It is soft: deferral or later reopening can exceed the requested count. It is
not a WRITE detector, a persisted conversation setting or a bound on total task
tokens. Nonzero disables MTP for the request in this implementation.

```text
request(stable_instructions, effort=high, budget=caller_phase_budget)
for generated step:
    if inside_thinking: count new non-control thinking token
    if budget>0 and count>=budget and not already_forced:
        if parser_boundary_safe and room_for_close_and_continuation:
            evaluate validated closing-think token once
            already_forced = true
        else record deferral
    continue ordinary tool/content generation
```

Do not merely inject the spelling of a closing marker into visible output;
update the model session and parser state consistently. The server implementation
and force/defer counters are in [ds4_server.c](../src/ds4_server.c). Presets leave
thinking policy unchanged. Earlier `PT_WRITE_REASONING_BUDGET` controls belong
to private experiment helpers and do not enable this behavior in ordinary
`ds4-agent`; use the implemented server request field or port the small policy
at the harness's generation boundary.

## Separate measured decisions

| Experiment and baseline | Observed result | Limit |
|---|---|---|
| [Item16](../speed-bench/experiments/server-thinking-hour16-20261004/README.md), `bc6896ae`; left-join/window tasks, high effort, suffix/routing off | WRITE256 saved 34.78% whole-task time;512 saved 24.72%; all 8 first codes 161/161 | Two reused tasks; READ/FINAL unlimited; shorter WRITE history also lowered FINAL prefill |
| [Item20](../speed-bench/experiments/final-thinking-hour20-20261004/README.md), `bc6896ae`; ledger/LRU, WRITE128 fixed | FINAL high/64 saved 20.65%; later high/1 saved 24.55%; all 10 first codes 161/161 | High/1 selected on same tasks, not independent confirmation |
| [Item35](../speed-bench/experiments/fresh-thinking-curve-hour35-20261004/README.md), `91c9cf21`; dedupe/bucket, suffix+resident fixed | WRITE1 saved 8.51% vs 128 bracket /7.13% versus faster 128;32 saved 3.76% /2.31% | All6 new codes 161/161; drift 3.00%; READ remained high/unlimited |

Item20 effort-none FINAL used zero thinking but was 16.44% SLOWER: high effort
reused 3086 input tokens and appended 87, while effort none freshly prefilling 2795
tokens outweighed the decoding reduction. This is an observed prompt/cache
effect, not a claim that thinking is intrinsically faster.

The useful settings are task-scoped candidates. The combined fresh coding
screen used READ0/WRITE128/FINAL1 in EVERY arm; do not add these reasoning gains
to its 23.52% routing+suffix result. Earlier WRITE-only controls sometimes shifted
thinking into FINAL, so count all phases and actual tools, not just shorter
WRITE traces. When a final answer requires new reasoning, a completed-write
summary budget may be inappropriate.

## Tried: hesitation-word logit penalties

We implemented the paper-inspired policy as a separate optional sampler
intervention: subtract `lambda` from logits for exact tokenizer mappings of a
hesitation-marker list before temperature/filtering. Preserve raw forward logits
for diagnostics; expose adjusted sampling logits separately. Local flags are
`DS4_OVERTHINK_PENALTY`, `DS4_OVERTHINK_SCOPE=think|all` and
`DS4_OVERTHINK_AFTER`. Positive policy disables speculative shortcuts; zero is
disabled. Think-only and delayed gating are our adaptations, not the paper's
fixed-every-step experiment. Split word prefixes are skipped to avoid penalizing
unrelated tokens; verify the actual tokenizer mapping.

The [guarded two-case test](../speed-bench/experiments/ds41-ssd/OVERTHINK_LOGIT_POLICY.md)
had baseline 2/2 correct finals (1356 tokens), but immediate and delayed lambda 1
each 1/2 with a cap (1413/1439 tokens). The easier four-case calibration was
byte-identical at lambdas 0/.5/1, hence no reasoning-length benefit.
In [Item4](../speed-bench/experiments/thinking-hour4-20261003/README.md), all three
all-scope penalties failed to produce the intervals solution within its cap;
ordinary no-thinking produced an incorrect empty-interval answer. A phase-specific
WRITE cutoff retained both task successes. These are negative results for the
tested configurations, not proof that all penalties or all no-thinking use fail.

## Replication and credit

Credit [Quantized Reasoning Models Think They Need to Think Longer, but They Do Not](https://arxiv.org/abs/2606.00206)
for the hesitation-token intervention. Its dense-model/quantizer findings do not
establish that our Q2 MoE's mistakes are caused by quantization. The request-local
cutoff is a separate protocol intervention.

Compare native with several budgets on fixed tasks, then lock one candidate for
unused tasks. Keep all failed first writes and capped finals. Record reasoning,
visible tokens, forced closures, deferrals, exact-prefix hits and total wall
through FINAL. Re-test tokenizer mappings and ordinary legitimate uses of
targeted words if porting the logit policy. Use paired ΔNLL only for a fixed
teacher/history; different generated reasoning paths are task-level evidence.
