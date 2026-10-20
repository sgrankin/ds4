# Independent item27 qualification audit

All six generated READ→edited-WRITE→FINAL pipelines completed, made exactly
one actual config read and one actual first write, preserved the config, and
avoided caps. All six saved first implementations pass the existing 161 checks.
The original native-before driver/guard failure is retained separately: its
grader received a relative code path under the grader's different cwd and
reported FileNotFoundError. This audit read completed small artifacts and
source; it ran no grader, model, build or test.

## First failure and code quality

Both native-before `result.json` records still contain `pass:false`, the
FileNotFoundError and grader return3; their original `grader.stdout` files are
unchanged. The model itself completed all three roles, wrote its first code,
had no protocol errors/caps, and left the config unchanged. Retained
`regrade.stdout` records report `correct:true, tests:161` for these exact saved
files. ROOT changed only `str(code)` to `str(code.resolve())` in the grader argv
before reuse and native-after. There was no model retry or code repair.
The reuse and native-after original graders each report161/161 with return0.

Ledger implements highest revision per exact key, last occurrence on ties,
sorted winning puts and optionally winning deletes. LRU implements configured
capacity, hit/miss accounting, recency moves and ordered evictions, including
nonpositive capacity and no input mutation. The final messages describe these
implementations; they do not claim that the model ran external tests.

Native-before/after first code bytes match exactly: ledger1090B,
SHA256 `b4cd0742a9cc3a71ec49e08d13588069c06fbdd29100396c42a232a4be80d4fc`;
LRU988B, `f20c52c648049e69dab4ac2df8f550e8460b6a39cd3bae6063b912c8bcb39638`.
All three role usages/statuses and public output items also match between
native brackets after removing only response/item/call IDs and timestamps.
Reuse writes different correct code: ledger1159B and LRU806B. This is observed
public/code repeatability, not an unrecorded private-logit/RNG equivalence claim.

## Authored edit, fresh User and actual reuse

The before/after source files and request1 histories show that only the two
marked obsolete note spans are removed. Stable task instructions, tools,
requirements and the actual config result remain. The request ends with a
new User asking for the single write. `responses_prepare_live_continuation`
returns when the message tail is User, so call-ID-only tool-output matching
cannot hide this earlier history edit. Native WRITE logs show full replay
with cached0; FINAL legitimately ends in a tool result and uses continuation.

The reuse arm's actual native-source SSD grafts are:

| Task | Source → destination | Reused | Fresh total / staging | Latest User start | Padding |
| --- | --- | ---: | ---: | ---: | ---: |
| Ledger |1118:1591 →436:909|473|299 /235|1006|0|
| LRU |1118:1573 →436:891|455|174 /110|987|0|

Both destination ends precede the latest User, so the entire new write request
is evaluated fresh, not merely the configured final64 tokens. Sources have
`source_ssd=1, source_approximate=0`; source endpoints are the original
requirement/prompt frontier, not a synthetic state containing a future tool
result. Fresh staging also includes the current actual config result.
FINAL uses `match=visible-prefix approximate=1` at cached1679/1469 in reuse,
versus approximate0 at cached1652/1518 in native controls. There is no extra
FINAL graft. Initial cold READ requests in reuse visibly fall back normally.

The server restarts **between complete tasks**: ledger finishes before LRU's
READ begins. This is not pending-LRU-source reuse after restart, despite each
graft being served through the SSD cache. All arms advertise the same actual
config/solution paths and use a fresh cache/store, hot0/SSD4096, GAP0,
ctx8192/chunk512, expert cache64GB, two slots, tile64/limit1024, temperature0,
high effort and READ0/WRITE128/FINAL1 under one2048-output allowance.

## Divergence and timing attribution

Initial READ request JSON is identical across all three arms for each task.
Nevertheless, reuse READ thinking counts differ **before any graft**:
ledger26→76, LRU62→40 (these are the actual public usage/log counts).
READ output counts are122→172 and157→135. All WRITE thinking counts are128;
ledger FINAL has1 and LRU FINAL0. Whole output counts are1403 in both native
controls and1385 in reuse; reasoning totals345 versus373. Do not describe the
saving as fewer total reasoning tokens or solely the edited graft's cost.
Generated histories/code differ, so paired cross-prefix KL/NLL was not measured.

One concrete scheduling distinction exists before READ: in batched
`server_session_sync`, suffix-enabled requests with a context profile retarget
the final prefill quantum to `N-fresh` and then finish the tail. Suffix-off
requests do not. These different partition boundaries are a possible
explanation for pre-graft numerical/generation divergence, not a proven cause
from the recorded public outputs alone. No runtime change is required for this
package comparison; its scope includes the source-capture schedule.

| Arm | READ sum, s | WRITE sum, s | FINAL sum, s | Task wall sum, s | Whole launcher, s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native before |105.649405|113.192803|16.892364|235.740203|240.171301|
| Suffix reuse64 |98.541489|93.799335|15.446834|207.793248|212.455934|
| Native after |105.051273|113.607978|16.858695|235.523718|240.377766|

Task wall includes actual tools and the authored edit, excluding startup and
external grading; launcher includes both server lifetimes, grading and restart.
Against native bracket means, task wall falls11.814%, launcher11.578%,
WRITE17.285%. Against the faster native control, task wall falls11.774% and
launcher11.540%. Native task/launcher drift is−.092%/+.086%. This is one
bracketed sequence on two already-used coding tasks with first-code correctness
retained; it does not establish broad coding accuracy or a suffix-only causal
explanation for every latency difference.

## Guards and evidence

Guard statuses are complete with child3/0/0. The first child3 is the retained
grading-path failure, not pressure termination or model incompletion. Normal
sample counts are232/206/232; every sample is level1, has no reason and zero
consecutive bad samples. Initial swap is0.012998GiB in all arms; measured
growth is0/0/.004638672GiB (about4.75MiB), so the closing control is not flat
zero. Peak wired+compressor is84.427109/84.189224/84.308029GiB.

See [build.json](build.json), [method.md](method.md), each arm's
`launcher-summary.json`, `*-execution.json`, server logs and
`http/{ledger,lru}/{ledger,lru}/` request/response, first-code, grading,
edit/history/result files. Guards are `guard-{native-before,reuse,native-after}`.
Original grading failures and saved-code regrades remain separately visible.
