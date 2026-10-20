# Retaining an approximate live continuation

Source-only proposal after the completed item24 screen. No model, compiler,
helper, runtime edit or new test was run by this audit.

## Observed avoidable work

Primary padding kept canonical0: all8 steps passed, and all four edited READ
requests reused527 tokens. A/C selected prefix_pad1; B/D kept the unpadded plan.
Its edited group sum was129.408389s against144.059122/143.431448s in the
existing-behavior brackets. Whole launcher was315.586557s against
329.373753/326.913435s. All file reads and edited gold pairs were correct.

The separate canonical1+padding1 point completed7/8 strict steps, solely because
Cwarm emitted correct inventory prose after READY. All four edited gold pairs
and all8 actual reads passed. Its six actual grafts were:

| Request stage | Reused | Fresh staging | Padding | Source kind |
| --- | ---: | ---: | ---: | --- |
| A edited READ |1034|31|0|Native SSD|
| B edited READ |1034|31|1|Native SSD|
| C edited READ after restart |1043|61|0|Native SSD; longer Cwarm history|
| D edited READ after restart |1034|31|1|Native SSD|
| B tool result → final answer |1034|150|1|Native SSD again|
| D tool result → final answer |1034|150|1|Native SSD again|

B/D's second requests report cached1034, rather than retaining their entire
live post-READ checkpoint. The extra grafts take7.454/7.992s in the server log.
A/C final answers use an exact canonical token prefix, cached1530/1569, and do
not regraft. See the two [server logs](../parity-suffix-hour24-20261004/canonical-padded/)
and `http/{before,after}/result.json`; group slices must be counted only once.
Combined edited group sum was71.503374s and whole launcher260.671082s, with
the Cwarm failure retained. All four guards stayed normal with zero swap growth.

## Why visible continuation can help

Canonical full-input mode currently bypasses Responses-visible, call-ID and
memory-text splicing. That is useful while constructing **native** warm sources:
the stored IDs represent the whole rendered input rather than sampled IDs
plus a separately tokenized append. After a successful padded graft, however,
the live graph intentionally contains a newline that is absent from the public
replay. Its ordinary token prefix may not match the next canonical request.
The Responses-visible tier can prove that the public transcript is unchanged,
then preserve this approximate checkpoint and append only the fresh tool result.

The existing initial approximate gate requires the same suffix cache profile
and, for an explicit previous response, the matching last-response ID. Then the
visible tier requires a valid matching live frontier, a strict byte extension
of the remembered public visible transcript and compatible image identity.
`build_live_prompt_suffix` copies the actual checkpoint IDs and tokenizes the
new suffix. This preserves the intentional pad and avoids rebuilding the old
READ turn. Core suffix sources remain immutable.

## One concrete hazard in the proposed broad condition

Changing the shared local gate to
`canonical_suffix = request_canonical_suffix(s, req) && !session_is_approximate`
reenables **all** splice tiers on an approximate graph. A full-input request can
change an earlier user message while retaining the currently pending call IDs.
The visible byte-prefix check then fails, but the later call-ID tier still
matches those IDs and ignores the changed earlier input. Consequently, the
claim that prefix changes necessarily fall through to a new graft is false
with that condition alone. The call-ID tier was designed to support incremental
tool-output-only clients, not to prove unchanged full replay.

## Minimal narrower change

Keep `canonical_suffix = request_canonical_suffix(s, req)` unchanged. Relax only
the Responses-visible branch to run when either canonical mode is inactive or
the current graph is approximate. Its existing same-profile and visible-prefix
proofs remain mandatory. Keep call-ID, memory-text and legacy disk-text bypass
unchanged for canonical full-input requests. An edited public prefix then misses
visible continuation and reaches exact token lookup or normal suffix matching.

Native graphs still use canonical warming. Default canonical0 behavior is
unchanged. Incremental/no-history requests already make
`request_canonical_suffix` false and retain their original ID-only behavior.
Previous-response and conversation requests are likewise outside that gate;
their parent/replay semantics need no change. No new output pointer or source
ownership path is needed: use the existing visible materialization and cleanup.

ROOT's corrected diff was inspected after this finding: only the visible branch
now permits `(!canonical_suffix || ds4_session_is_approximate(slot->session))`.
The shared canonical boolean, preceding profile/frontier checks, ID-only and
memory-text gates, and legacy disk-text gate stay unchanged. The continuation
log additionally records `approximate=…`. I found no concrete source blocker.
This is source review, not proof of the native continuation; the existing CPU
fixture has no approximate-graph hook, so actual HTTP qualification is needed.
ROOT retained the previous binary at
`.workspaces/server-agent-day-20261004/ds4-server-before-approx-continuation`
(compiled revision `00da420b`) for the unchanged-behavior bracket.

Measure the unchanged item19 file workload with canonical1/pad1 fixed, old and
new continuation behavior in fresh cache/store namespaces. Retain Cwarm prose,
all fallback/wrong/cap outcomes, native source warm cost and all edited latency.
The concrete target is four initial edited grafts instead of six, with B/D
second requests logging `match=visible-prefix` and retaining their whole live
checkpoint. That is an expected mechanism to qualify, not a claimed result.
