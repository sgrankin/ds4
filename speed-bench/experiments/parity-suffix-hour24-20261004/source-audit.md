# One-token prefix newline: bounded proposal

This is a source proposal, not a runtime change. It follows item23's completed
canonical-source screen and keeps the core's even source/destination rules.
No model, tokenizer, helper, compiler or new experiment ran during this audit.

## What item23 actually established

All three arms completed8/8 task steps with8 actual file reads, clean READY
warmups and correct edited fact pairs. All guards completed0 with normal
pressure and zero measured swap growth. Canonical edited group wall was
139.37109s, against143.86377/144.33544s in the existing-behavior brackets;
whole launcher wall was324.49285s against329.07610/328.96593s. Those small
aggregate gains conceal a useful A/C gain and B/D fallback.

I inspected only state-cache manifests and their state header/token-ID ranges,
at most6.8KB per endpoint, in the cache named by
[canonical execution.json](../canonical-suffix-hour23-20261004/canonical/before-execution.json).
KV, pre-RoPE and logit arrays were not read. The same-profile native source and
edited endpoint IDs give:

| Family | Source end | Last-64 match ends at | Shift | Backward common span | Result |
| --- | ---: | ---: | ---: | --- | --- |
| A/C |1694|1342|352 even|1035 tokens, starts659→307|Trim one to660→308:1034 reused|
| B/D |1692|1341|351 odd|1034 tokens, starts658→307|No eligible graft|

B/D's native source really contains the matching tool-result span. Its last64
IDs match exactly; the odd displacement is the concrete reason it cannot be
relocated. Canonical warmup added one input token for all four tool-result
requests: A/C1693→1694 and B/D1691→1692. A/C gained reuse413→1034; B/D lost
reuse527→0. Changing one common rendering rule cannot make both geometries even.

Relevant B/D pairs in that cache are source
`ds4sc-c8fad1cb71c7ec08d458b53c3179f22b93bd5c15.manifest` with edited
`ds4sc-8e9cf85d9c8dbcfba1b211a5d6ade661a4d7ee55.manifest`, and source
`ds4sc-2eaf9ad00b542a4486274d1e849533f02b2aaf3c.manifest` with edited
`ds4sc-55956ca5a427b143e85252ecc9e4d8cd666d9ca2.manifest`.

## Smallest useful experiment

For the one-change item24 screen, keep the existing splice behavior
(`DS4_SERVER_CONTEXT_SUFFIX_CANONICAL=0`), suffix64 and the existing file
workflow. Add one opt-in server flag, defaultoff. In `server_try_context_suffix`, compare
the unchanged target against a target with exactly one newline token inserted
immediately after the **first** user delimiter. Consider padding only when
that insertion is before the latest user and before the selected reused slice;
it must be prefix padding, not a change to the new query. Use the same immutable
candidate sources and existing `ds4_suffix_match` for both arrays. Select the
padded plan only if its reused-token count is strictly greater; keep unchanged
on ties. No odd-shift core path or token fabrication is needed.

The engine must establish that standalone `"\n"` is exactly one ordinary token
whose piece is exactly newline; otherwise decline padding. Use the existing
tokenizer and token-piece API, not a hardcoded numeric guess. Tokenizing only
this tiny literal is enough. The inserted token is fed through normal prefix
evaluation, not hidden from the session checkpoint.

The existing-before cache confirms A/C's raw common tool-result span is527
tokens, source1166:1693 to destination815:1342, currently odd shift351.
An insertion before position815 moves its destination to816:1343, even
shift350, with both starts even. That would improve A/C reuse413→527. B/D's
existing527-token span1164:1691→814:1341 already has even shift350; padding
would make it odd349, so the unchanged plan remains preferred. This algebra
justifies trying the change; it is not a measured task or latency result.

With canonical1, the same operation would instead repair B/D's1034-token
span658:1692→307:1341 to308:1342, while A/C keeps its existing1034-token plan.
That is a possible later combined point, not the predicted result of the
canonical0 single-change screen.

## Essential integration details

* Build an independently owned padded `ds4_tokens`; never alter source IDs or
  the original request while comparing plans. Require room for the extra token
  under the same strict context limit and existing fresh-tail requirement.
* Recompute latest-user position and fresh cutoff on the selected array.
  Every token of the latest user, plus the existing fresh minimum, stays fresh.
  Both selected boundaries must pass the existing even-left-shift matcher.
* Evaluate the selected prefix, then call the unchanged relocation API. It
  verifies the destination's exact checkpoint prefix and every reused ID,
  source engine/context/pre-RoPE coverage, and a nonempty evaluated fresh tail.
* On successful padding, move the owned array into the caller's existing
  `effective_prompt` and keep `prompt_for_sync` pointing there through final
  ordinary sync and generation. Do not return a pointer to a stack local.
  Recompute `prompt_tokens`, progress span and input/cache usage for +1 token;
  the current caller computes those before the graft, so this must be explicit.
* If selection/materialization/graft fails, free padding and continue from the
  untouched original prompt. The existing graft failure invalidation remains.
  The selected source ref stays pinned until copying and staging finish.
* Store successful padded checkpoints as the existing approximate DSSD kind
  under the session-key profile, never in the shared exact-prefix namespace.
  Keep public replay unchanged; document the newline as an intentional model
  input approximation. A later canonical replay may again need a graft.

This is a deliberate semantic newline approximation. Measure all8 strict task
steps and file reads, cold warmup, edited group wall, actual reused/fresh counts,
padding activation/fallback and total wall. Retain worse answers and fallbacks.
Do not promise B/D success from the algebra, reject approximation for unequal
baseline prose, or report paired NLL for different generated histories.

Current suffix support permits resident routing. Cache profiles hash resident
`DS4_METAL_*` settings, so a combined policy needs its own warmed sources; it
does not require weakening core pairing or mixing GAP0/GAP.90 namespaces.
