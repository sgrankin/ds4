# Canonical full-input suffix sources: source audit

This is a source proposal and a small read-only inspection of completed item21
cache token metadata. No model, tokenizer, compiler, helper or new probe ran.
Item21 used runtime `fe5d6e2670e7d154bf97234df01d1ee8db4a51e8`;
its [build/settings record](../hot-cache-hour21-20261004/build.json) and
[four-session driver](../concurrent-sessions-hour19-20261004/run.py) are the
existing authorities.

## Concrete boundary finding

All four item21 warm tool-result requests used the **Responses visible-prefix**
tier (`match=visible-prefix ids=1`), not the later call-ID-only tier. Public A/B
read arguments, actual file contents and READY text match across all capacities
after removing generated IDs. Thinking usage is zero. Thus missing hidden
reasoning is not the explanation for this screen.

The existing `build_live_prompt_suffix` copies actual sampled checkpoint IDs,
then tokenizes only the appended tool result. A later edited full input is
rendered and tokenized as one complete prompt. Equal text can therefore have
different token joins. Exact raw DSML attachment also survives public replay;
disabling its attachment alone does not disable either live splice tier.

I read only manifest metadata and each relevant state's 52-byte header plus
token-ID bytes (at most 6.8 KB), from `.workspaces/server-agent-day-20261004/hot21-cache0`.
I did not read KV, pre-RoPE, logits or model data. The native warm endpoints and
the already captured edited endpoints show:

| Session family | Native source end | Matching last-64 anchor end in edited IDs | Displacement | Current eligibility |
| --- | ---: | ---: | ---: | --- |
| A/C | 1693 | 1342 | 351 | Odd shift: rejected |
| B/D | 1691 | 1341 | 350 | Even shift: accepted |

For B/D, backward extension gives 527 tokens, source `1164:1691` to destination
`814:1341`, exactly matching the log. For A/C the last-64 anchor **exists**; its
odd displacement explains why that full tool-result endpoint cannot be used.
The observed shorter source `660:1073` to `308:721` has even displacement352
and reuses413, leaving652 fresh staging tokens. This is not evidence that the
tool result was missing from disk, and does not justify relaxing pair alignment.

Relevant A/C manifest pairs are native `fd34da82fadf25e1747c8ee63bef41b1ccfce734`
with edited `24cf25b1f0f4c8ca8228be5da9e6a6d3e6ff431b`, and native
`405651b2d3821073c3f6fb4b9f21e9daef8285e4` with edited
`0c68a4524db3eb0c0c60126ffb367fc7fde64883` (names use `ds4sc-…manifest`).
The inspected source tails end with the normal assistant-prefix tokens
`128804,128822`. A future canonical warmup may move that boundary by one token;
that is a hypothesis to measure, not a promised alignment fix.

## Minimal opt-in change

Use `DS4_SERVER_CONTEXT_SUFFIX_CANONICAL=1` together with existing suffix reuse.
For complete text Responses inputs with a nonempty `prompt_cache_key`, skip
**all non-token splice tiers**: Responses-visible, Responses tool-output-ID and
memory-text. Keep exact memory-token reuse and exact state-cache token lookup.
Skip legacy disk-text materialization for the same requests. This leaves the
ordinary sync path consuming the already parsed/rendered `req.prompt`.

The narrow request gate excludes image requests, `previous_response_id`,
`conversation`, and tool-output-only inputs requiring live tool state. Those
incremental requests retain their original continuation semantics. Current
instructions/tools and the full public input remain the rendering authority;
raw DSML replay is kept unchanged. Default flag0 retains the existing tiers.

The candidate ROOT diff uses one shared `request_canonical_suffix` gate in slot
selection and legacy disk loading, so routing and materialization agree. Its
existing focused probe assertions cover visible/text skipping, incremental
retention and exact-token retention. I found no concrete source blocker in
those gates. No test or build was run by this audit.

## Small experiment and interpretation

Reuse the unchanged item19 four-session file workload, suffix64, hot0 and
fresh cache/store paths. Compare old splice behavior with canonical behavior;
keep the same advertised paths, facts, native routing, tools and model settings.
Record cold warmup separately from edited AB and restarted edited CD, actual
read/answer checks, endpoint/anchor geometry, fresh staging and total wall.
Retain odd fallbacks and the READY-only formatting failures. A one-line cache
miss window is sufficient if a new boundary remains unexplained; no new harness
or exhaustive token dump is needed. Do not claim generated-history paired NLL.

Current server/core suffix configuration does **not** reject resident GAP.
State-cache profiles hash all sorted `DS4_METAL_*` settings, and suffix profiles
also bind `prompt_cache_key`. A combined GAP.90 experiment needs newly warmed
GAP.90 sources; existing GAP0 sources belong to a different profile. Profile
separation is not a ban on combining resident routing and suffix reuse.
