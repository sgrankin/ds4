# Local DS4 Flash experiment review — 2026-10-03

The most promising points are **WRITE 256 plus strong resident affinity**,
**suffix reuse for fact-preserving edits**, and simple affinity alone. On two
unused code/tool tasks, the combination completed all 161 first-code checks and
clean stops in `100.218 s`, versus native-bracket mean `131.687 s`
(**23.90% less**). It also saved 22.64% against the faster native-after control.
Both controls passed both tasks. The combination uses neither suffix reuse nor
original staging.

Suffix reuse made the
measured edit 2.5–3.3× faster on ordinary short-answer cases with small likelihood
changes. Strong affinity saved 17.2% of logical expert reads and 16–19.5% of
fixed-input decode time; its four generated tasks took about 8.79% less total
time. These are different workloads and timing scopes, not one combined gain.

The initial five items finished within their one-hour limits; that review was
saved before item 5's 15:17:23 UTC hard stop. Item 6 then integrated idle
live-context editing into the main agent and measured storage-backed workflows.
Approximations remain opt-in. A later small preset wrapper exposes context-fast
and resident-fast; the measured native-weight resident path is now in main.
The successful WRITE-budget hook still needs to be forwarded. MAIN runtime
defaults are unchanged. Wrapper integration passed the existing agent unit suite
but adds no new model performance measurement.
One model job ran at a time under the existing pressure guard. Degraded answers,
caps and slower points remain in the curves; numerical disagreement was not an
automatic rejection rule. Task success matters more than a universal ΔNLL or
KL cutoff, and these small cohorts do not establish generalization.

| Item | Completed cohort and timing scope | Main finding |
|---|---|---|
| [1 — Suffix / CLM](suffix-clm-hour1-20261003/README.md) | Five fact-preserving cases, one factual-removal case; 15 reuse settings; two separate CLM comparisons. Model work 11:57–12:27 UTC. Timing covers the edit, excluding old-context setup and answer generation. | Ordinary edits 2.5–3.3× faster; factual removal needs more fresh recomputation. |
| [2 — Resident experts](resident-functional-hour2-20261003/README.md) | 23 fixed-input configurations / 8,064 scored rows; 20 generated attempts on four tasks. Measured 12:29–13:17 UTC. Decode, reads and whole generated-task time are separate. | Strong affinity contributes; functional matching did not beat the simpler selector. |
| [3 — Original arrival](original-arrival-hour3-20261003/README.md) | 25 fixed-input configurations / 14,080 scored rows; 16 generated attempts on the same four tasks. Model work ended 14:08 within the 13:17–14:17 budget. | Removing unnecessary joins makes age 8 timing-neutral; age 1 offers a completion/cost tradeoff. |
| [4 — Thinking](thinking-hour4-20261003/README.md) | Nine arms × two calibration code/tool tasks, 161 first-code checks per task. Pressure ledger 14:19–14:44 within the 14:10–15:10 budget. Instrumented whole-helper time. | WRITE cutoff 256 saves 13.13% with 2/2 strict passes; penalties fail and no-thinking has a code bug. |
| [5 — Unused confirmation](confirmation-hour5-20261003/README.md) | Sixteen new teacher cases, two unused practical tasks with native brackets, and four disjoint generated-answer tasks × three policies completed. Preparation counts from 14:17:23 toward the 15:17:23 UTC deadline; model work began 14:46:35. | Affinity's decode gain carries over. WRITE 256 plus affinity saves 23.90% with both practical tasks passing. Generated factual success is 1/4 in every policy. |
| [6 — Actual-agent context edits](agent-suffix-hour6-20261003/README.md) | Thirteen workflows with actual tool retrieval, two suffix fresh-tail settings, short/long retained contexts and single/repeated edits. Budget 15:35:50–16:35:50 UTC. | Main agent imports idle edits. Long repeated workflow saves 24.65%, including exact reconstruction on its second edit; all 19 graded answers pass. |

## Suffix: useful speed, with an explicit adverse case

Ordinary cleanup retained a 512-token unchanged suffix while deleting 128 tokens.
At fresh tails of 64/128, measured fresh/reuse edit times were roughly
`12.8–13.0 / 3.9–5.3 s`. Independently expected answers 3141, 42 and 206 survived;
the reported case means ranged from −.00247 to +.000170 nats/token. These are
short synthetic, non-thinking answers and warmed paired measurements.

Factual removal exposed stale influence: fresh execution answered UNKNOWN,
while reuse with tails 64/128 returned wrong codes. At 256/384 fresh tokens it
answered UNKNOWN, with ΔNLL +.188/+.125 and speedups 1.68×/1.26×. The curve,
including wrong answers, is useful evidence for choosing recomputation. Exact
CLM fresh/graft comparisons qualify only the separate private isolation policy;
they do not demonstrate whole-agent speed or semantic forgetting.

## Resident experts: affinity first, arrivals as a tradeoff

On 24 fixed prompts × 32 teacher tokens, strong affinity had mean ΔNLL −.01260,
KL .02056 and worst case mean ΔNLL +.06217. Native brackets reproduced all 768
vectors, while profiles and run variation explain part of the timing spread.
Functional similarity and its router band did not establish an advantage.
Logical demand bytes include prefill and decode; they are not physical SSD
traffic. Generated histories differ, so task times do not imply cross-prefix KL.

Original arrivals initially slowed strong-affinity decode by 10–18% because an
unconditional early selected-load join sacrificed overlap. A six-line branch
joins only admitted tickets. Age 8 then measured −.21%, inside the .65% control
range: timing-neutral, KL .02013 versus .02056, with .295% extra logical reads.
Age 1 cost 7.40% more fixed-input decode and 1.48% more whole generated-task time,
but completed 3/4 tasks versus affinity's 2/4. Those four reused tasks justify
keeping the tradeoff point, not a broad accuracy claim.

## Thinking: the cutoff helps, but does not bound whole-task reasoning

WRITE 256 retained both correct first implementations and clean tool/final-stop
protocols, saving 13.13% against the mean native brackets. It did not meet the
20% time-reduction goal. Intervals saved 854 WRITE decisions but added 430 in
the final response, where thinking is unrestricted. Ledger gained no work
saving at this cutoff. Native token/RNG tuples matched before/after; light audits
did not compare full raw logits.

Hesitation penalties .1/.5/1 all exhausted intervals' 2048 global decisions
without a first solution, while ledger passed. No-thinking stayed outside
thinking but retained `[8,8]` instead of ignoring empty intervals. Its first
case also paid a 50.881 s system-prompt cache miss; .1 paid a similar miss on
switching back. Keep actual wall times and those cache differences visible.

## Unused confirmation

The completed item-5 teacher screen has 512 scored rows: native-before/after
vectors are exact, with decode `30.2498/30.2744 s` (.0812% spread). Strong affinity
decode was `24.587 s` (**−18.75%**), mean ΔNLL −.023243 and KL .018666, with
18 changed argmaxes. Age 8 decoded in `24.613 s`, only .105% above affinity,
with .2604% extra logical reads, four promotions and ten future uses. Age 8's
KL .019266 is higher than affinity's .018666 on these new cases: its quality
tradeoff is mixed. Its lower
whole runtime is confounded by prefill variation and 3.63% native whole-time
drift; it does not establish a staging latency win.

The held-out replacement-count curve is now complete. These are later single
runs against the same native reference, with fixed gap, rank and mass limits:

| Maximum substitutions | Decode saving | Mean delta-NLL | Mean KL | Logical demand GB |
|---|---:|---:|---:|---:|
| 1 | 8.79% | +.003166 | .009795 | 303.187 |
| 2 | 16.55% | −.013260 | .012503 | 272.829 |
| 3 | 18.75% | −.023243 | .018666 | 252.276 |

Count 3 is fastest in decode; count 2 offers lower KL and fewer changed argmaxes.
Its worst case mean delta-NLL is nevertheless higher than count 3's, so fewer
replacements is not a guarantee of a uniformly safer change. Preserve the
per-case tails alongside this curve.

On the two new practical tasks, WRITE 256 alone took `120.054 s`; adding strong
affinity took `100.218 s` (**16.52% less**), and both implementations passed all
161 checks with actual final stops. Routing footers report 10,509 and 10,954
substitutions with native weights. Keep this directly measured combination
point separate from multiplying the earlier individual gains. Native-before
took `133.825 s`, native-after `129.548 s`; both passed. Relative to their mean,
WRITE 256 alone saved 8.83%, and the combination saved 23.90%. All four policy
arms completed both tasks without a numerical-parity rejection gate. It is only
two held-out code prompts and one fixed seed.

All 12 disjoint generated-answer attempts stopped normally with no caps, but
native, strong affinity and age 8 each passed **only 1/4 factual tasks**. Their
parsed factual answers were identical:

| New task | Every policy's first answer | Independent gold |
|---|---|---|
| Bitmask | 22 / hex 16 | 36 / hex 24 |
| Superseded access record | 8736 / 5127 | 8736 / 5127 — correct |
| Recurrence | a6 = 17 | a6 = 20 |
| Stack program | 18, pops 9 and 6 | 10, pops 9 and 6 |

Keep those baseline errors. There was no posthoc repair, relaxed grader or
rejection of an approximation for failing to turn these into perfect answers.
Observed native/affinity/age-8 whole times were `29.128/23.257/23.717 s`, but
startup was `9.189 s` for first native versus `5.352 s` for affinity, with
unsymmetrical order. These times are diagnostic, not a general 23% speed claim.

## Actual-agent storage-backed context edits

Item 6 moved an earlier fact out of the visible transcript into a permanent
JSON file, retaining an unchanged active inventory result. The actual agent
retrieved the stored fact with a tool and answered an independently graded
calculation. Repeated variants changed that permanent fact and edited the
earlier tool result again; the next answer used the updated store.

At fresh minimum 128, whole workflow time fell 16.44% short / 31.81% long for
one edit, and 14.71% short / **24.65% long** for two. The long repeated sequence
took `128.494 s` versus mean native brackets `170.522 s` (23.77% saving against
the faster native control). The first edit reused 1,540 tokens and replayed
177 old tokens; the second reconstructed exactly from a cached 1,986-token
prefix. Its cost is included. Tail 256 forced 1,514 fresh old tokens at the
available checkpoint boundary, so saved only 6.55% on the long single edit.

All 13 workflows and 19 separately graded answers passed, with actual archive
reads and fresh incoming questions. This narrow lookup/arithmetic screen does
not qualify general reasoning, semantic forgetting or canonical persistence.
Different generated histories were not treated as paired likelihood probes.
The new actual-agent result is separate from the private two-code combination.

Only the frontend was changed/rebuilt, and the existing agent suite passed.
The main binary now contains the idle importer; the source change reuses the
existing transaction without adding knobs. All model jobs completed under the
pressure guard with 1,762 samples and no adverse sample. Maximum observed swap
growth was approximately 0.87 MiB; high RAM alone was expected.

## Next use of the time budget

The [inventory](agent-suffix-hour6-20261003/packaging-inventory.md) records the
small shell wrapper and forwarded native-weight path. WRITE 256 still needs
integration. Current suffix reuse also requires resident GAP zero, so retain
separate modes before testing a combination. Known preset values do not require
an hour-long experiment. Disk-backed suffix sources and pre-RoPE relocation
tests are the next separately budgeted experiments.
No defaults are promoted by these small screens.
