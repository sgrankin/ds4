# Independent item28 qualification audit

All eight generated READ→edited-WRITE→FINAL pipelines completed without caps,
performed exactly one config read and one first write, and preserved the config.
All eight saved first codes pass the existing161 checks after the missing
standard builtin is supplied to the grader. The original strong-LRU grading
failure remains visible. This audit read completed small artifacts/source and
ran no model, build, grader or test.

## Grading artifact, unchanged tasks and native repeats

Strong subs3 LRU writes valid `next(iter(cache))`; its original grader reports
`NameError: name 'iter' is not defined`, `tests_complete:false`, child3 and
`pass:false`. The task permits builtins; the existing AST language check accepts
this code. The correction adds only `iter` to the existing execution scope
(`(*existing.SAFE_NAMES, 'iter')`). Fixture generation, exact gold comparison,
input-mutation checks and candidate bytes are unchanged. Retained
`regrade-iter.stdout` reports161/161 on all four already-generated native-before
and subs3 first codes. Later subs1/native-after original grades report161/161.
There is no generated-code repair, model retry or new gold chosen after output.
Raw subs3 failed grade/child3 must remain alongside the corrected interpretation.

All four arms' saved task/config/system records match. The native-before/after
first code is byte-identical: ledger1159B, SHA256
`0f42a32a29d33b3c1fd25f2230fbe7c8f08b1d8bbfec9ac21d3ae43b5189cbe5`;
LRU806B, `d5c2ea3b4a837814941c99600bccfe5de2c7364572587a0bff3e4ff475768ca8`.
All three role usages/statuses and public output items also repeat after
removing only dynamic response/item/call IDs and timestamps. This is observed
public/code repeatability, not unrecorded private-logit/RNG equivalence.
Subs3 ledger has the same bytes; its LRU is755B. Subs1 writes964B/709B, and its
LRU also uses `iter`, passing under the corrected scope.

## Fixed suffix path and policy-matched sources

Every arm enables suffix64/canonical1/PAD1 with hot0/SSD4096. “Native” here
means native resident routing, not absence of suffix approximation. GAP0 and
native_weights0 controls report profile
`44f656302d6a106059a893439add7bfdf814706e` before and after restart.
Subs3 reports `a5344dab32c9490536d1ca5ecf1859d13982c11b` for both tasks;
subs1 reports `413c59a1ce28b4fbfb4d8f210c8591b1486e1d6a` for both.
Each arm starts its own cold cache/store and warms under its actual policy.
No endpoint is borrowed across routing profiles. Advertised file paths match.

All arms graft the same ledger473 tokens (source1118:1591→destination436:909,
fresh299/staging235, latest User1006) and LRU455 tokens
(1118:1573→436:891, fresh174/staging110, latest User987). Both sources are
native-source SSD endpoints and pad0. Each reused slice ends before the entire
new WRITE User; the real config result is also fresh. Trailing User prevents
the call-ID-only tool-output shortcut from bypassing the authored history edit.
Each FINAL uses valid visible-prefix continuation on the resulting approximate
graph, with no additional graft. Actual cached FINAL frontiers vary with the
generated WRITE length, as expected.

The server restarts between completed ledger and LRU tasks, not while an LRU
source is pending. LRU performs its actual READ after restart. This screen
therefore does not claim restart recovery of an earlier LRU history. Settings
otherwise remain Q2/cache64GB, ctx8192/chunk512/two slots, tile64/limit1024,
temperature0, high effort, READ0/WRITE128/FINAL1 and global2048 output cap.

## Actual resident activation

Subs3 uses GAP.90, mass.25, rank64, maximum3 substitutions, total_mass.50,
replace/native_weights1. Its ledger/LRU route counts are12,384/14,296;
native-weight footer commit counts match exactly. Total26,680 substitutions
and265,553,510,400 logical skipped bytes; drops0. Subs1 changes only maximum3→1,
counts3,897/3,943, total7,840 and78,033,715,200 logical skipped bytes; drops0
and double/triple layers0. Neither native-routing control emits an active route
footer. These are logical skipped loads and activation of the reviewed
ID-only/native-weight path, not independent physical SSD bytes or GPU bit tests.

## Timing and scope

| Arm | READ sum, s | WRITE sum, s | FINAL sum, s | Task sum, s | Whole launcher, s | Output / reasoning tokens |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Native before |98.137823|96.115379|15.706481|209.964891|214.380558|1385 /373|
| Subs3 |95.877535|87.028161|18.934988|201.845898|206.633537|1482 /350|
| Subs1 |103.649087|91.626324|16.575799|211.856439|216.569393|1345 /353|
| Native after |101.835654|97.388485|16.192013|215.422042|219.867092|1385 /373|

Task wall includes actual tools and context edit, excludes startup/external
grading; launcher wall includes both server lifetimes, grading and restart.
Subs3 task savings are5.100% against the native bracket mean and3.867% against
the faster control; whole savings4.831%/3.614%. Subs1 task savings are only
.394% against the mean and it is.901% slower than the faster control;
whole savings.255% against the mean,1.021% slower than the faster control.
Native task/whole drift is2.599%/2.559%. Subs1 is timing-neutral here, not a
demonstrated improvement. Generated role lengths/code differ, so no paired
cross-prefix NLL/KL or routing-only per-token latency claim is made.

This is one native-before→subs3→subs1→native-after sequence on two already-used
coding tasks. Subs1 was a bounded follow-up selected after subs3, recorded at
20:13 UTC; it is not a separately preregistered replication. The broader
coding-accuracy claim remains limited to the actual first-code161 results.

## Guards and metadata

Guard children are0/3/0/0; subs3 child3 is the retained grading-scope error,
not pressure termination or model incompletion. Normal sample counts are
207/200/210/213; every sample is level1, no reason and zero consecutive bad.
Initial swap is.017636719GiB for the first three arms and.018125GiB for the
closing control. Growth is0/0/.000488281/0GiB (subs1 adds about.5MiB).
Peak wired+compressor is86.061417/86.301468/86.346786/86.591980GiB.

Metadata records actual arm order, bounded subs1 decision, raw iter-scope
failure and unchanged-code regrades. See [build.json](build.json),
[method.md](method.md), arm execution/launcher JSON and server logs,
`http/{ledger,lru}/{ledger,lru}/` first-code/grade/regrade/request/response
records, and `guard-{native-before,subs3,subs1,native-after}`. No model failure
was converted to success by repairing generated code or changing expected golds.
