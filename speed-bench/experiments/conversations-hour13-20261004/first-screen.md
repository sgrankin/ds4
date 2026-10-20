# Conversations first screen

The first qualification was stopped at 14:03 UTC after a real HTTP routing bug: the existing request reader stripped query strings before Conversations item listing. Requests for `order=asc&limit=100` therefore received the default descending order and limit 20. ROOT preserved this attempt and corrected query preservation before starting a fresh qualification with separate output, cache and store directories.

The retained [before result](first/http/before.json) contains **14 PASS, 5 FAIL, 0 ERROR and 0 SKIP** over 73.055 seconds. These are mixed protocol and model checks, not an accuracy rate. The five failed checks are the ascending route listing, source append ordering, edited public history, edited append ordering and pending-tool append ordering. The first 24-item listing returned only 20 descending items.

The wrong ordering also caused a real fixture error: the driver's initial item selection deleted the **active-facts item**, while retaining the obsolete log. The subsequent answer still returned `{"site":"MAPLE","total":416}`, but the previous assistant answer already contained those fields. A correct body therefore does not establish the intended fact-preserving deletion or vindicate this history selection. The isolated CEDAR response and pending tool call also completed; restart, later routes and conflict publication were not qualified by this partial run.

The [guard result](first-guard/result.json) records `status=interrupted`, signal 15 and child return code −15; ROOT reported outer exit 143 after its deliberate SIGTERM. This was **not a pressure rejection**. All 73 retained [pressure samples](first-guard/pressure.jsonl) were normal (level 1, no reason, zero consecutive bad samples). Swap stayed at 0.012998 GiB with zero growth; peak wired plus compressor memory was 84.309891 GiB.

Successful item deletion and answer checks are separate from observed suffix activation. This interrupted screen supports no suffix speed claim and is not pooled with the corrected qualification. Original requests, replies, server logs, store and guard evidence remain under `first/` and `first-guard/`.
