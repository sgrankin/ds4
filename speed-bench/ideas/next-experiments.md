# Next experiments: one hour each

## Recipe follow-up authorized 2026-10-08

Compare against `../ds4-recipe` revision `0316b956f8236d810f50c2ae0a6fda9f77ce35d6`.
Run these items sequentially; each budget includes implementation, builds,
measurement, interpretation and a small commit. Preserve degraded points.

| Item | Budget (UTC) | Deliverable |
| --- | --- | --- |
| 37. Warm-prefill resident routing | 15:22–16:32, excluding requested 10-minute pause | One opt-in extension of the existing bounded resident policy, using residency frozen at prefill-call start. Compare native with two strengths on cold and warm ordinary/favorable turns; record actual prefill time, substitutions, task outcomes and paired continuation quality where existing tooling permits. |
| 38. Real compaction quality | 16:26:32–17:26:32 UTC | Ordinary summaries, deduplication and irrelevant-log edits alongside changed-fact stress. Compare the same edited context with fresh execution and suffix reuse; use longer assistant continuations and independently check outcomes. Record ΔNLL/KL and the tail tradeoff without requiring exact output reproduction. |
| 39. Single-source right shifts | 17:20–18:20 UTC | Extend the existing one-span path to insertions/replacements with even right shifts. Keep latest user fresh and pre-RoPE originals immutable; test insertion, replacement and fallback, then measure ordinary task quality and edit latency. Defer multi-source planning until measured coverage justifies it. |
| 40. Combined fresh-task confirmation | 21:43–22:43 UTC | Confirm decoder and warm-prefill resident routing together with left/right suffix reuse on two unused coding families. Bracket the combined and suffix-only arms with native controls; grade first writes, retain failures, and decide whether to expose a named opt-in package. |
| 41. Direct agent integration | 2026-10-09 02:40–03:40 UTC | Done: persistent-fast works directly with ds4-agent, including pre-RoPE left/right SSD-backed reuse and built-in compaction without a mirror file. Five real CLI runs passed all 10 fresh-file answers; actual compaction reused 696 tokens and retained its release code. The favorable aligned workflow saved about 46.5% against native controls; odd-shift fallback is retained. [Report](../experiments/agent-package-hour41-20261009/README.md). |

Use the existing pressure guard, one heavy model job, fixed native brackets,
and the current practical quality targets below. Keep raw caches and useful
scratch under `.workspaces/`; retain small reports/CSVs/plots in `speed-bench/`.
Long-context candidate-only index scoring and shared DSpark scheduling remain
separate follow-ups rather than expanding these three budgets.

Item37 completed within its active-hour budget. Three substitutions saved
10.64% warm-append time and 5.98% coding-task time against native means
(9.32% / 2.86% against faster controls), with all eight first codes passing
161 checks. Mean paired ΔNLL was +0.005043 nats/token. One substitution reduced
drift but did not establish latency savings; all timing outliers are retained.
The 20% speed target was not met. Existing presets remain unchanged; the new
prefill knob is opt-in. [Report and curve](../experiments/recipe-prefill-hour37-20261008/README.md).

Item38 measured faithful discussion compaction through the actual server
READ/edit/WRITE/FINAL workflow. Suffix reuse saved 11.66–12.38% whole-task
time against native brackets that differed by only 0.0081%; all ten first
writes passed 161 checks. Six standalone screens retained normal, favorable,
equivalent-instruction and stale-rule cases, with mean ΔNLL from −0.01434 to
+0.01781 nats/token and all cached code passing. The fresh instruction case
emitted a Markdown fence: its raw format failure and separately passing
unfenced function are both retained. Requested server tails64/128/256 produced
the same total fresh coverage, so the fastest arm is not a general optimum.
Keep the existing context-fast preset; the 20% whole-task speed target was
not met. [Report and curve](../experiments/recipe-compaction-hour38-20261008/README.md).

Item39 adds default-off `DS4_SERVER_CONTEXT_SUFFIX_RIGHT=1` for even right
shifts from immutable pre-RoPE sources. Actual server insertion and growing
replacement tasks saved 19.46% against native means, or 18.80% against the
faster controls, with all six first writes passing 161 checks. Three moves
(right66, left128, right514) preserved one original checkpoint and its masters;
missing originals rejected a right move without changing the session. Four
coding screens measured 64/128/256 fresh-tail tradeoffs: mean ΔNLL ranged from
−0.00393 to +0.00229 nats/token. Five raw outputs used Markdown fences; their
format failures and separately passing extracted functions remain visible.
The whole-task 20% target was narrowly unmet. Presets retain their existing
values; the documented server flag enables the extension.
[Report and curve](../experiments/recipe-rightshift-hour39-20261008/README.md).

Item40 confirmed the combined settings on unused interval-grouping and
dependency-layer families, with faithful compaction and expansion respectively.
Whole READ/edit/WRITE/FINAL time was196.35s, versus native brackets263.22/250.25s:
23.52% saved against their mean and21.54% against the faster control. Suffix
reuse alone took223.76s; routing contributed another12.25% saving. All ten
first writes passed their161-case graders with no retries or caps. The combined
repeat took196.43s and matched code/usage; native drift was−4.93%, retained in
the comparisons. The20% practical speed target is met on this small screen.
The opt-in server `persistent-fast` preset now sets the measured prefill and
right-shift knobs, retaining explicit0 overrides and caller-controlled
thinking. [Report and tradeoff plot](../experiments/combined-fresh-hour40-20261008/README.md).

2026-10-03. Current status: first six items measured and reviewed; the preset wrapper is added. Run experiments in order, with one heavy model job
at a time. Each item gets 60 minutes of wall time, including source changes,
builds, runs and analysis. At the deadline, record the result or concrete
obstacle and move on. An extension needs a measured reason and a new budget.

Use existing helpers, graders and pressure guard. Aim for at most 15 minutes
of preparation, 35 minutes of measurement and 10 minutes of interpretation.
Do not turn preparation into another harness project. Record commands,
settings, revision and a small results CSV; preserve failures as observations.

## Sequence

| Item | Budget | Question and deliverable |
| --- | --- | --- |
| 1. Suffix reuse and CLM edits | 60 min | Measure ordinary and favorable edits alongside changed facts. Start with existing suffix reuse; use the corrected ordinary CLM oracle to distinguish isolation-policy quality from cache reuse. Produce matched timings and an initial fresh-tail tradeoff plot. |
| 2. Functional resident experts | 60 min | Try the completed expert-similarity map in native routing. Compare native, existing router-affinity substitution and functional substitution. Sweep replacement count or one distance/mass threshold, retaining native routing weights. Produce quality, latency and SSD-read points. |
| 3. Originals arriving from SSD | 60 min | Compare a fixed resident policy with and without the prepared asynchronous original-expert reader. Measure how often originals arrive before use and whether the earlier mandatory join costs more than it saves. Sweep one wait/deadline knob if the first pair works. |
| 4. Quantized overthinking | 60 min | Run independently graded reasoning/code tasks with the existing stopping/budget intervention. Sweep unset and two budgets first; try penalty strength only if time remains. Measure correct answers, thinking tokens, caps and whole-task time. |
| 5. Confirm useful points | 60 min | Recheck the most promising points on unused cases and with reversed timing order. Fill a gap near a Pareto knee. Combine features only if the change is small and each already contributes. Report what deserves more time. |
| 6. Persistent-agent suffix edits | 60 min | Import ordinary external edits while idle; move facts into actual storage and retrieve them. Measure whole workflows and repeated edits, including exact reconstruction. Completed below. |
| 7. Package available opt-ins | Small task, completed | A shell wrapper supplies known settings for context-fast / resident-fast. The native-weight route was forwarded and rebuilt. WRITE-budget production integration remains separate. |
| 8. Pre-RoPE and repeated relocation | 60 min | Retain true pre-RoPE BF16 originals, relocate from those originals repeatedly, and test core plus ordinary agent edits. |
| 9. Shared hot RAM / SSD cache | 60 min | Add automatic prefix/suffix chunk caching across sessions, with bounded hot storage, disk spill, restart reuse and interleaved session qualification. |

Initial commitment: five hours, then review. Curves begin within each item;
item 5 is confirmation, not a reason to postpone measurement.

Item1 results: [suffix / CLM first hour](../experiments/suffix-clm-hour1-20261003/README.md).
Fifteen compaction configurations and two ordinary CLM policy probes completed
in about30 minutes of model work. Ordinary edits showed2.5–3.3x diagnostic edit
speedups; factual removal supplied a clear speed/quality curve, including two
wrong-answer points. No approximation was rejected for failing byte parity
with ordinary native execution; that policy drift was measured separately.

Item2 results: [resident experts second hour](../experiments/resident-functional-hour2-20261003/README.md).
The strongest simple affinity point reduced 24-case decode time16–19.5% and
logical expert reads17.2%, with mean delta-NLL-.01260. Functional similarity
did not establish an advantage over affinity. Twenty generated tasks retain
baseline errors and caps; three-replacement affinity improved one first answer
but not the completion count. All approximate points remain visible.

Item3 results: [original arrivals third hour](../experiments/original-arrival-hour3-20261003/README.md).
The original unconditional early join made strong routing10–18% slower. A
six-line change to join only admitted requests recovered most or all of that
penalty. Age8 is timing-neutral versus affinity with slightly lower KL and
0.295% extra logical reads; age1 completed3/4 generated tasks versus2/4 for
affinity, at1.48% extra whole-task time and7.4% extra fixed-input decode time.
These are diagnostic tradeoffs; both remain opt-in private prototypes.

Item4 results: [thinking controls fourth hour](../experiments/thinking-hour4-20261003/README.md).
Nine settings produced18 first-attempt tool/code runs. WRITE256 preserved2/2
strict successes and reduced whole-task time13.1% against stable native brackets.
WRITE512/1024 also passed; some saved WRITE thinking moved into the final turn.
All three marker penalties capped the interval task before writing a solution.
No-thinking passed1/2, with an empty-interval bug and a mode-dependent startup
cache miss retained in its observed time. Approximate failures remain on the curve.

Item5 results: [unused confirmation](../experiments/confirmation-hour5-20261003/README.md).
Strong affinity retained an18.75% decode gain on16 unused teacher cases. On two
unused executable tasks, WRITE256 plus affinity reduced complete task time23.90%
against native brackets and passed all first-code checks and clean stops.
The four separate generated-answer tasks passed1/4 under every policy; baseline
errors remain visible. See the [five-item review](../experiments/20261003-idea-review.md)
for the points worth carrying forward, limits and the next time-budget priorities.

## Suffix qualifications

Use one no-edit control and seven edited cases. Begin with one case from each
family; expand within the hour. Report families separately rather than pooling
their means into a single verdict.

| Family | Cases |
| --- | --- |
| Favorable | Rewrite irrelevant earlier planning while retaining the active facts; clean earlier logs when the retained suffix contains the complete relevant tool result. |
| Ordinary CLM editing | Remove a completed verbose tool log; replace a log with a concise summary retaining required facts; deduplicate notes while retaining decisions and constraints. |
| Stress | Change a fact needed by the answer; delete a previously retrievable fact and ask a wholly fresh question about it. |

The no-edit control detects a broken cache path. Existing invariant tests are
enough unless a new failure appears. Favorable cases also need a fresh execution
that actually succeeds; otherwise an unchanged wrong answer proves little.

For every edited case, compare reuse with a fully fresh run of the **same edited
context**, using the same tokenizer, continuation and sampling settings. Keep
the entire latest user query fresh. Score an independent expected answer or
task outcome, plus short fixed teacher continuations for delta-NLL/KL.

Start with a typical long suffix and the existing fresh-tail setting. Vary
64/128/256 fresh tokens where the query and implementation permit it; include
fully fresh execution as the reference. Keep the complete final context fixed
while moving the cached/fresh boundary. Different context or suffix lengths
are separate workloads, each with its own matched fresh baseline.

The private CLM block oracle currently implements fixed-position rewrites.
Its native-versus-isolation comparison measures a policy tradeoff; its
fresh-isolation-versus-graft comparison checks reuse under that policy. Actual
token-shortening compaction belongs to the existing shifted suffix path.
Do not label padded fixed-position rewrites as deletion compaction.

If time remains, run a short sequence of ordinary edits. Include any exact
rebuild between approximate edits in its time; single-edit gains may not survive
that workload. Count capture, copying, repair and fresh-tail evaluation in
edit-to-next-answer latency, rather than reporting only the cached block.

## Small practical suite for the other items

Reuse existing unused fixtures: two executable code/tool tasks, one arithmetic
task and one context-fact task. Add cases before seeds in the initial screen;
use another seed and unused cases for promising points. Grade final outcomes
against independent answers/tests. Record pass counts, errors and caps, including
baseline failures. A small screen can identify regressions, not certify a
five-percentage-point accuracy bound.

For resident experiments, reuse paired fixed teacher tokens for distribution
scores. For thinking interventions, task success and token/time savings are
primary: differing sampled reasoning histories are not paired KL comparisons,
and raw-model NLL cannot grade an intervention that only changes sampling.

## Working targets and curves

- Aim for at least **20% lower relevant wall time** (1.25x speedup), with 2x a
  stretch target for long-suffix edits. SSD-read savings explain a result but
  do not substitute for practical latency.
- Preferred likelihood target: mean delta-NLL at most **0.0953 nats/token**
  (10% perplexity increase). Initial generous ceiling: **0.1823 nats/token**
  (20% perplexity increase). Use identical fixed teacher tokens; declare
  token-weighted averaging and also report each case/family. This is a working
  engineering target, not a universal accuracy guarantee or automatic rejection.
- Aim to preserve baseline task success; a later larger screen can target at
  most five percentage points of loss. Keep degraded configurations on the
  curve so their extra speed remains visible. Initially report actual counts.
- KL, total variation and changed answers explain drift. Do not invent a
  universal KL pass threshold or let a low mean hide a large relevant-token error.

Perplexity is exp(mean NLL), so exp(mean delta-NLL) gives the paired perplexity
ratio. The 20% ceiling corresponds to 0.8333x geometric mean target-token
probability; a 20% drop in that probability instead gives 0.2231 nats/token.
Definition: [Transformers perplexity documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/perplexity.md).

Start each tunable feature with the baseline and two settings, then refine a
promising gap to four or five settings if the budget permits. Vary one knob at
a time. Save a CSV and a simple standalone plot of latency versus likelihood
degradation, with task pass counts attached. For thinking, plot latency/tokens
versus task success. Show per-family curves and mark single-run timing as
diagnostic. A point is Pareto dominated only when another measured point is
both faster and at least as good on the declared quality axis; retain the
other quality and resource measurements alongside it.

## Existing evidence to carry forward

Suffix reuse already has mixed evidence, not a uniformly poor result. The
earlier same-position edited-fact fixture retained the answer and reported
40.454 s fresh versus 17.551 s reuse (2.30x), mean KL 0.0000133 and summed
delta-NLL +0.0000882 across three rows. This is an archived single comparison,
not a new balanced throughput result:
[small primary summary](evidence/suffix-semantic-code-c512-t1/summary.json).

The later moved-881-token suffix fixture reported mean delta-NLL +0.56957 and
mean KL 0.450493 across four target tokens, with a large error on the relevant
number. It measured neither task success nor speed:
[archived report](../experiments/suffix-fact-likelihood-20261003/README.md).
Treat these as different tested geometries, not competing global verdicts.

## Follow-up authorized 2026-10-03: persistent-agent suffix edits

**Item 6 budget: 15:35:50–16:35:50 UTC (60 minutes, including all prep,
builds, measurements, analysis and commit).** Integrate the existing suffix
transaction into idle live-context edits so an external context organizer can
age history into checkable storage before a new user turn. The incoming query
must be appended after the edit is imported and evaluated fresh.

Use a short ordinary storage-backed workflow: preserve active facts, move an
older record into a permanent JSON file, and retrieve it with an actual tool.
Compare identical edited contexts and commands with suffix reuse off and at
two fresh-tail settings. Record full-sequence and edit-to-answer times,
accepted/rejected edits, actual reused/fresh counts, exact fallbacks and final
task fields. Retain geometry failures. Existing source invariants/tests and
standard JSON are sufficient; no new validation framework. Keep other
approximations off and native controls around the candidate runs.

After this screen, inventory which earlier options are present in production
and which require private code before designing a single opt-in package.
An environment file alone cannot integrate missing runtime code.

**Item 6 complete:** [actual-agent report](../experiments/agent-suffix-hour6-20261003/README.md).
Thirteen workflows passed, with 19 separately graded answers and actual
permanent-file retrieval. Fresh minimum 128 saved 16.44%/31.81% whole time
on short/long single edits and 14.71%/24.65% on short/long repeated edits.
The second edit reconstructs exactly from a cached prefix; its cost is included.
Fresh minimum 256 saved only 6.55% on the long single edit because available
checkpoints forced 1,514 fresh old tokens rather than 177. These are small
lookup/arithmetic fixtures; no generated-history delta-NLL/KL was inferred.

Main source and rebuilt agent now accept idle live-context edits. The
[packaging inventory](../experiments/agent-suffix-hour6-20261003/packaging-inventory.md)
distinguishes options already present from private measured code.

## Presets and cache follow-up, 2026-10-03

User correction: known preset names/values only need a small shell wrapper,
not a one-hour experiment. `ds4-local` supplies context-fast and resident-fast;
later CLI arguments override its context/cache/chunk defaults. The native-weight
route needed by resident-fast was copied from the measured private path into
main and rebuilt; the existing agent suite passes. No new model quality or
latency measurement is implied. WRITE-budget integration is still pending.

Existing disk KV covers the system prefix and explicitly saved sessions in
`~/.ds4/kvcache` (`DS4_AGENT_CACHE_DIR` overrides it). Exact sessions save graph
state; approximate suffix sessions save text and rebuild on reopen. The retained
context frontiers are RAM-only. Item 8 should persist immutable exact sources,
using existing KVC model/layout checks. Disk spill alone is insufficient:
qualify a real restart/resume before editing, and avoid overwriting the old
live-context mirror during startup. Never relabel an approximate result as an
exact source.

The current code refuses chained approximate grafts. Its one relocation repairs
already quantized post-RoPE values, so it still incurs rounding once. Item 9's
pre-RoPE archive should regenerate moved keys from an untouched source rather
than repeatedly rotating/requantizing yesterday's moved copy. DS4 rotates only
the last 64 dimensions; a tail-only archive is a useful small candidate. This
addresses numerical relocation error, not old-prefix conditioning. Even shifts
retain compression pair boundaries; odd shifts require separate repooling work
and stay out of this initial experiment. Set a fresh budget when starting each
of these two experiments.

## Authorized implementation follow-up: 2026-10-03

User requests both true pre-RoPE originals with repeated relocation and an
automatic hot-memory/SSD cache for prefix/KV and suffix state, rather than
manual session saves. Order updated to implement relocation first, then the
shared cache. Item 8 budget: **23:43:58–00:43:58 UTC**. Item 9 receives its own
60-minute budget after item 8. Preparation, builds, measurements and commit
count toward each budget.

Pre-RoPE: keep true normalized BF16 rotary lanes from native scalar and batched
publication. Each move starts from an immutable archive; even shifts retain
compressed pair boundaries. Preserve latest incoming queries as fresh. Measure
3/8 relocations and a storage-backed agent sequence; report any exact prefix
reconstruction. Numerical relocation loss and stale-prefix conditioning are
separate effects.

Cache: study oMLX's dedicated DeepSeek V4.1 delta representation, reuse DS4's
existing disk payload infrastructure, and store immutable chunks plus endpoint
state needed by compressed attention. Qualify RAM hits, SSD reloads, restart
and alternating sessions. Exact prefix keys depend on the preceding context;
approximate suffix originals belong to a separate cache namespace. Concurrent
clients/interleaved sessions are distinct from simultaneous GPU execution;
report the actual scheduler capability rather than implying cache storage
alone provides continuous batching.

Item 8 first screen: [pre-RoPE and repeated relocation](../experiments/pre-rope-hour8-20261003/README.md). Eight immutable-source moves and three actual-agent edits passed.
Item 9 qualification begins 00:26:37 UTC, with a 00:56:37 UTC deadline; about
30 minutes of source preparation already charged from the overlapping item8 window.

Item8 timing correction: the original native system cache missed on its
live-context path. A matching warmed native arm runs within its original
00:43:58 UTC deadline; the edited-turn saving is reported separately.

Item8 complete at 00:37:34 UTC. Matching warmed control confirmed the system
prefix hit and all three fresh archive reads; use the corrected comparison in
the report. Shared-cache qualification continues under its 00:56:37 deadline.

Item9 complete at00:48:16 UTC: [hot RAM/SSD cache report](../experiments/state-cache-hour9-20261003/README.md). Twelve native reopen/isolation checks, two concurrent server clients before/after restart, and three disk-only repeated agent grafts passed. A token-zero prefix selector failure was retained, fixed and regression-tested.

## Approved server follow-up, 2026-10-04

The user approved the [server and persistent-agent day](server-agent-day-20261004.md).
Continue sequentially with its items10–18, with at most one hour per item,
including preparation, build, checks, model work, analysis and commit.
Make a results handoff available at18:00 Eastern and reserve the final hour
before the stated19:00 reset for a selected follow-up and handoff.

Item10 starts **12:53:40 UTC**, with a **13:53:40 UTC** deadline.
Integrate opt-in server suffix reuse for ordinary full-input Responses requests,
using the existing original-backed runtime and RAM/SSD tier. Pi's ordinary
provider already replays complete input with store:false, so durable response
references are the next separate item, not a prerequisite. Preserve the whole
incoming user message as fresh and keep approximate state scoped and explicit.
First qualification is an actual HTTP source plus two fact-preserving deletions;
then move to useful durable response chaining and broader workflow qualification.

Item10 finished early at **13:20:15 UTC**. [HTTP results](../experiments/server-suffix-hour10-20261004/README.md):
the first cold 48-line screen saved20.1% across all requests, with one odd-shift
fallback. Matching warm 47-line edits saved62.2% across the edited turns and
reused an approximate SSD source on the second edit. Both cohorts passed6/6
answers; single-order first screens, not a general quality/latency guarantee.

Item11 implementation starts **13:21 UTC**, with **14:06 UTC** deadline:
15 minutes of read-only preparation are already charged to its one-hour budget.
Add opt-in durable text/function-tool response files, standard append/fork
previous_response_id, GET/DELETE, current instructions and a restart check.

Item11 finished early at **13:34:40 UTC**. [Durable response report](../experiments/responses-hour11-20261004/README.md):
38/40 first checks passed; two outputs copied an earlier marker despite correct
facts/counts. All13 restart checks passed, including pending tool calls, hidden
reasoning and child continuation after parent deletion. Greedy full-input versus
stored-reference controls agreed in both cases, including the count case's
wrong marker. Preserve the base-model mistakes; no harness repair is needed.

Item12 starts **13:35 UTC**, with **14:25 UTC** deadline;10 minutes of preparation
are charged. Qualify full-input HTTP edits with actual file retrieval, separate
session hints, slot turnover and SSD restart. Fixed archived facts before the
restart are ordinary cases; updated archive facts afterward are reported as a
separate changed-fact stress family.

Item12 finished early at **13:52 UTC**. The actual-file HTTP workflow passed
all7 steps in both arms, with all6 requested file reads fresh. Ordinary edits
saved24.3%; after-restart changed-file edits saved15.6%, with an approximate SSD
graft in A and an unsupported-alignment fallback in B. One timing order only;
the [report](../experiments/server-workflow-hour12-20261004/README.md) keeps cold
source preparation and edited-turn savings separate.

Item13 starts **13:54 UTC**, with **14:44 UTC** deadline;10 minutes of read-only
preparation are charged. Add the small durable text/function Conversations
subset and use item deletion to edit history before a fresh incoming query.
Approximate suffix reuse remains globally opt-in; manual item edits discard
private replay and use the edited public history. Check restart and storage
independence, without chasing the full API surface.

Item13 finished early at **14:06:30 UTC**. The durable Conversations subset
passed45/45 HTTP checks after a query-stripping routing bug was caught and fixed.
History item deletion activated suffix reuse; pending tool continuation survived
restart; conversation history survived response deletion; concurrent manual
editing produced a visible409 without losing the appended item. The first
failed screen is preserved. See the [report](../experiments/conversations-hour13-20261004/README.md).

Item14 starts **14:07 UTC**, with **14:57 UTC** deadline;10 minutes of read-only
preparation are charged. Measure original-backed core suffix tails64/128/256
against fixed fresh contexts across ordinary/favorable/changed-fact families.
Keep core likelihood/edit-latency curves separate from ordinary HTTP task
timings and SSD costs; report actual fresh tokens when query/geometry limits a
configured tail.

Item14 finished early at **14:46 UTC**. [Original-backed tail curves](../experiments/server-suffix-curve-hour14-20261004/README.md):
all9 fixed-context fresh/reuse pairs answered correctly, with near-zero
three-token likelihood drift. The actual HTTP sweep passed7/7 in every arm;
tail64 saved27.6% on ordinary edits,22.1% across edited turns and13.7% including
cold sources. Tail128 saved20.6% across edited turns. All three settings retained
the recurring B2 alignment fallback. Core timing and HTTP task timing remain
separate;64 is the next practical candidate, not a new default.

Item15 starts **14:46 UTC**, with **15:36 UTC** deadline;10 minutes of prior
preparation are charged. Compare carried shipping/rounding decisions against
fresh shortened context and an explicit current archive-rule control. Capture
source state before generating an old answer, so latent conditioning cannot be
confused with retaining an assistant's earlier answer. Include superseded rules.

Item15 finished early at **15:09 UTC**. [Retained-history report](../experiments/suffix-history-hour15-20261004/README.md):
one coherent ordinary case benefited from reuse (correct calculation versus
fresh abstention); a superseded-rule case was harmed (reuse abstained versus
fresh correct calculation). No usable recall of an entirely omitted rule was
shown. Actual archive tool reads recovered both current rules after an observed
output cap was raised; strict integer-format failures remain visible. Original
failed calibration, notes-only variant, caps and family NLL/KL are retained.

Item16 starts **15:09 UTC**, with **15:59 UTC** deadline;10 minutes of source-only
preparation are charged. Add a small opt-in Responses request-scoped thinking
budget, then qualify actual post-read write requests at0/256/512. A generic
request cutoff does not inherit the earlier private WRITE256 result until its
actual role and full read/write/final task costs are measured.

Item16 measurement finished at **15:41:57 UTC**, ahead of the15:59 deadline.
[Request-scoped reasoning report](../experiments/server-thinking-hour16-20261004/README.md):
all8 READ/WRITE/FINAL task executions passed their161 code checks with no caps.
Native brackets produced identical code, final text and per-role token usage,
with410.269/409.786 seconds across both tasks. Relative to their410.027-second
mean, post-read WRITE budgets512/256 saved24.7%/34.8% whole task time.
Each positive arm forced exactly two native thinking closes at its requested
count; no unsafe-boundary deferral occurred. The control remains request-scoped,
opt-in and native DS4.1 text only. These are the previously used join/window
qualification tasks, not unused confirmation cases. Shorter private history
also reduced the next turn's prefill cost; this is not a decode-only saving.

Item16 report and results committed at **15:46:30 UTC** (`c5ecdb63`).

Item17 starts **15:46:30 UTC**, with a **16:36:30 UTC** deadline;10 minutes of
earlier SDK/Pi source inspection are charged. Exercise the installed SDK and Pi
streaming tool path against the server, qualify post-read budgets on new
practical coding cases, and confirm tail64 with another actual-storage suffix
workflow. Keep those separate screens attributable and reverse candidate/native
timing order. All model jobs remain sequential beneath the pressure guard.

Item17 finished early at **16:23 UTC**. [Real-client and confirmation report](../experiments/server-confirmation-hour17-20261004/README.md):
installed SDK/Pi streaming and essential Conversations CRUD passed. New LRU
code passed161/161 in all arms; WRITE128 saved22.88% completed-task wall.
New ledger128 completed with correct code; native capped before writing and
256 wrote correct code but capped in FINAL. Retain those attempted times as
capped outcomes, not equal-completed-work speedups. Fresh-fact suffix workflows
passed7/7 and6 reads in both arms; tail64 saved21.95% ordinary edits,18.48%
all edited turns and14.19% whole launcher wall. Three of four edits grafted,
including an approximate SSD source after restart. All pressure samples normal.

Item18 starts **16:23 UTC**, with a **17:13 UTC** deadline;10 minutes of source
inspection and method planning are charged. The measured bottleneck is changing
top instructions on each tool stage, which reduces the live common prefix to
about411 tokens. Compare staged128 before/after with a natural tool loop using
the same stable task instructions, tools, actual read/write contract and128
post-read budget. Preserve first code, independent checks, caps and actual
prefix reuse. This is a caller experiment using the current server, not a new
runtime feature or agent framework. Keep later overthinking visible.

The short server preset task completed alongside source preparation at
**16:26 UTC**. `ds4-local --server context-fast|resident-fast` supplies the
known settings, two slots, localhost and durable response storage. Server
context-fast uses the measured opt-in tail64 candidate with an environment
override; resident-fast explicitly disables suffix reuse. Agent defaults remain
available. Shell syntax and both modes' help checks passed without model startup;
no new performance measurement is implied. Request thinking budgets remain
caller-selected, and README documents the API subset and measured limits.

Item18 finished early at **16:46:33 UTC**. [Stable instruction report](../experiments/stable-instructions-hour18-20261004/README.md): natural instructions saved40.49% completed two-task wall against staged before/after brackets; all six first implementations passed161/161 and completed the actual read/write/final pipeline. Natural used more thinking but reduced fresh input7262→2053 through ordinary exact prefix reuse; suffix approximation was off. Staged code/final/usage repeated exactly, with0.523% bracket wall drift. This is a caller protocol comparison, including shorter prompts and different histories, not an additional runtime speedup to add to prior percentages.

Item19 starts **16:48 UTC**, with a **17:38 UTC** deadline;10 minutes of earlier concurrency outline planning are charged. Qualify four active sessions with two clients in flight and two server slots, forcing turnover and SSD restart. Compare full-input Responses suffix64 with native execution, retaining eight independent facts/actual reads per arm, group makespan, queue-inclusive latency, cache/fallback evidence and state isolation. One model process and the existing pressure guard remain in use.

Item19 finished early at **17:05 UTC**. [Concurrent sessions report](../experiments/concurrent-sessions-hour19-20261004/README.md): four histories/two clients/two slots completed eight real reads per arm. All edited facts passed; native strict replies8/8 versus suffix7/8, solely C's warmup READY plus extra correct inventory prose, before any graft. All four edited suffix grafts used native SSD sources, two after restart. Edited group makespan saved38.64%, whole launcher17.63%; cold source groups were2.13% slower. Both arms logged four evictions. No pressure event or swap growth; candidate exit3 reflects the retained format failure. This is one concurrent workflow pair, not evidence of general parallel GPU throughput.

Item20 starts **17:05 UTC**, with a **17:55 UTC** deadline;10 minutes of earlier source preparation and method planning are charged. Keep the natural stable-instructions caller and READ high/unrestricted, WRITE high/budget128 fixed. Compare FINAL high/unrestricted brackets, high/budget64, and effortnone. Measure whole task, final-request wall, exact first code/usage repeat, cache costs, actual final text and caps against the same frozen161-check ledger/LRU graders. No new runtime feature or automatic role detector.

Item20 finished early at **17:24 UTC**. [FINAL policy report](../experiments/final-thinking-hour20-20261004/README.md): all ten pipelines/first-code implementations passed161 checks with no caps. READ/WRITE emitted content, private reasoning and first code were byte-identical across policies; native FINAL outputs/usage repeated with0.112% wall drift. FINAL high/budget64 saved20.65% whole-task wall; a bounded lower-point follow-up high/budget1 saved24.55%, with accurate completion summaries. Effortnone was16.44% slower: it lost prefix reuse and freshly prefilling2795 tokens outweighed zero thinking. High policies reused3086 and appended87 tokens. Budgets remain per request; no automatic phase detector or broader accuracy claim. All pressure samples normal with zero swap growth.

Item21 starts **17:24 UTC**, with a **18:14 UTC** deadline;10 minutes of earlier cache-source/launcher preparation are charged. Compare host hot-cache limits0/128/512MiB on the same four-session suffix64 workflow. Reuse existing driver and immutable SSD blocks; report actual hot/disk reads and bytes, edited-group and whole wall, restart behavior and outcome differences. Add only a small existing-counter shutdown log if needed for measurement; no cache framework or new API.

Item21 finished at **17:59 UTC**. [Hot-cache report](../experiments/hot-cache-hour21-20261004/README.md): all32 actual reads and16 edited facts passed; strict warm replies8/7/8/7 retain two READY-plus-prose failures before any graft. Hot512 served all173 pre-restart chunk reads from RAM without speeding up edited AB; hot128's apparent gain against opening hot0 was reproduced by closing hot0, whose public answers/token totals match all eight hot128 steps. No hot-capacity latency gain or default promotion follows. RAM ownership stayed within configured caps, all four pressure guards stayed normal with zero swap growth, and persisted sources worked after restart. Counters describe logical chunk reads, not physical SSD traffic.

Item22 starts **17:59 UTC**, with an **18:49 UTC** deadline;10 minutes of earlier source/launcher preparation are charged. Keep stable instructions, READ high/unrestricted, WRITE high/budget128 and FINAL high/budget1 fixed. Compare native routing brackets with resident GAP.90/max substitutions3 and1 on the same ledger/LRU actual tool tasks and frozen161-check graders. Keep suffix reuse off, first implementations, caps and completion visible; measure the combined resident/reasoning package against the same reasoning policy with routing disabled. No new runtime code or harness.

Item22 measurement finished at **18:07:17 UTC**. [Resident package report](../experiments/resident-combined-hour22-20261004/README.md): all8 first implementations passed161 checks and completed without caps. Native code/public outputs/private emissions/role usage repeated exactly, with0.326% wall drift. Maxsubs3/1 saved9.704%/3.575% against native bracket mean with fixed READ0/WRITE128/FINAL1. Actual counters confirm21,856/7,566 native-weight substitutions and no drops. Candidate ledger code differed while remaining correct; generated histories differ, so this is package task evidence rather than paired NLL or a decode-only claim. Suffix reuse remained off.

Item22 report/results committed at **18:11 UTC**.

Item23 starts **18:11 UTC**, with a **19:01 UTC** deadline;10 minutes of earlier source audit/launcher/candidate preparation are charged. A/C full source1693 matches the edited prompt's last64 tokens ending1342, but the351-token displacement is odd; B/D's350-token move is supported. Test whether canonical full-input materialization changes the source boundary enough to improve graft coverage, measuring warm cost as well as edited and whole wall. Compare existing-before/canonical/existing-after with suffix64/hot0/SSD4096 fixed on the unchanged four-session driver. The opt-in flag bypasses non-token live/text splices only for complete text Responses suffix clients; keep incremental/parent/Conversation semantics and even-move restriction. It is a policy experiment, not an assumed bug fix.

Item23 finished early at **18:35 UTC**. [Canonical source report](../experiments/canonical-suffix-hour23-20261004/README.md): all24 steps/reads passed with no caps; controls repeated all8 public texts/token totals with0.033% whole-wall drift. Canonical saved1.38% whole/3.28% edited group wall. A/C gained1034-token grafts, but B/D lost their527-token grafts because canonical rendering changed their displacement to odd351. Coverage fell4→2; source token metadata confirms the parity cause. Warm group costs stayed near controls; all guards normal with zero swap growth. Keep the feature opt-in; it does not solve alignment universally.

Item24 starts **18:35 UTC**, with a **19:25 UTC** deadline;10 minutes of prior padding-source audit/launcher/candidate preparation are charged. Add one opt-in prefix newline only if it gives a strictly longer supported even graft than every ordinary candidate, with ordinary preferred on ties. Keep canonical0 initially, suffix64/hot0/SSD4096/fixed four-session driver, and compare existing-before/padded/existing-after. Preserve immutable sources, latest-user freshness, incremental request semantics and actual +1 input usage. If the bounded primary screen permits, separately measure canonical1+padding1 as the natural configuration follow-up; do not conflate it with padding alone.

Item24 finished early at **19:07 UTC**. [Parity-padding report](../experiments/parity-suffix-hour24-20261004/README.md): PAD alone saved3.83% whole/9.97% edited wall; the separate canonical-plus-PAD point saved20.56% whole/50.26% edited against bracket means. All32 reads/16 edited facts passed with no caps. Combined strict7/8 preserves C's warm READY-plus-prose failure before any graft, its39 extra output tokens and changed source. Six combined grafts include B/D tool-result regrafts; all used immutable native SSD originals. Controls repeated all public texts/token totals with0.747% wall drift; all1,191 guard samples were normal with zero swap growth. Both input policies remain opt-in.

Item25 starts **19:07 UTC**, with a **19:57 UTC** deadline;10 minutes of prior continuation-source audit/launcher/candidate preparation are charged. Hold canonical1/PAD1/suffix64 and the four-session driver fixed; compare saved old binary before/after with a narrow visible-continuation candidate. A same-profile approximate graph may continue only when the entire submitted visible history matches its recorded prefix. Keep canonical bypass for ID-only, memory-text and legacy-disk paths. Measure actual avoided completion regrafts, input/cache usage, reads/facts/format/caps and whole/edited wall. Do not broaden continuation to edited-prefix or wrong-hint requests.

Item25 finished early at **19:29 UTC**. [Approximate continuation report](../experiments/approximate-continuation-hour25-20261004/README.md): all24 steps/reads and12 edited facts passed with no caps. Four verified visible-prefix continuations retained1530-token approximate frontiers and eliminated two B/D completion grafts, reducing edited wall30.40% and whole launcher12.38% against old-binary brackets. Primary1034-token grafts stayed unchanged; all controls repeated public content/usage with1.63% whole-wall drift. All938 pressure samples normal with zero swap growth. This comparison uses its own materially slower graft-cost brackets, not earlier timing points; no additive claim.

Item26 starts **19:29 UTC**, with a **20:19 UTC** deadline;10 minutes of prior launcher/method preparation are charged. Keep the qualified canonical1/PAD1/suffix64/approximate-visible-continuation runtime fixed on Item19's four-session workflow. Compare native routing before/after with GAP.90/maxsubs3/nativeweights1. Each arm warms and restarts its own new cache under its actual routing profile. Preserve fresh reads, all facts/format/caps, source geometry/fallbacks, actual substitutions, edited group makespan and whole launcher costs. No additional runtime or harness feature.

Item26 finished early at **19:47 UTC**. [Combined routing/suffix report](../experiments/suffix-resident-package-hour26-20261004/README.md): all24 steps/reads passed with no caps. The strong routing arm performed13,697 native-weight substitutions; all arms retained four native-source SSD grafts/four approximate visible continuations. Edited wall fell2.00%; whole wall fell1.42% against bracket mean but only0.13% against the faster control, with2.60% control drift. This establishes compatibility without a firm additional whole-time gain. All848 guard samples normal, zero swap growth.

The short named-profile task also finished at **19:47 UTC**. Server context-fast now enables canonical1/PAD1 with native routing; server-only persistent-fast adds the qualified strong resident settings. Server cache defaults hot0/SSD4096 match the recent screens; agent defaults remain unchanged. Shell syntax, all help paths and the server-only gate passed without model startup. Request thinking budgets remain caller decisions; no automatic role detector was added.

Item27 starts **19:47 UTC**, with a **20:37 UTC** deadline;10 minutes of prior caller/launcher/method preparation are charged. Qualify actual first-write coding after removing two explicitly obsolete note spans from full-input history, preserving task requirements, the actual READ call/result and a fresh WRITE query. Compare native-before/reuse64/native-after against the frozen ledger/LRU161 graders, with stable high effort/READ0/WRITE128/FINAL1 and a global2048 cap. Measure sources/fallbacks rather than forcing geometry; keep first failures and complete task/launcher costs. Restart between completed tasks is not a pending-source recovery test.

Item27 finished early at **20:04 UTC**. [Actual CLM coding report](../experiments/clm-coding-hour27-20261004/README.md): all six actual pipelines completed without caps; every saved first implementation passed161 checks. Task-pair wall fell11.81%, launcher11.58%, against native brackets with0.09% drift and identical public/code/usage. Actual473/455-token native SSD grafts left current config output and latest WRITE User fresh; FINAL retained approximate live frontiers. READ thinking diverged before graft, and suffix capture changes prefill boundaries, so this is a package saving rather than pure relocation attribution. Original native-before path-grade errors/child3 remain alongside saved-code regrades; the one-line absolute path fix changed no HTTP inputs and no generation was retried. All670 pressure samples normal; closing swap growth4.75MiB is recorded.

Item28 starts **20:04 UTC**, with a **20:54 UTC** deadline;10 minutes of prior unchanged-caller/launcher/method preparation are charged. Keep suffix64/canonical1/PAD1 and the repaired CLM caller fixed. Compare native routing before/after with strong GAP.90/maxsubs3/nativeweights1 on the same actual ledger/LRU read/edit/write/final tasks and frozen161 first-write graders. Warm all sources under each arm's own policy and retain caps, code changes, route counters and whole/task costs. This tests the combined coding package, not a sum of earlier gains.

Item28 finished at **20:29 UTC**. [Combined CLM coding report](../experiments/clm-resident-package-hour28-20261004/README.md): all eight pipelines completed without caps and all saved first implementations passed161 checks. Strong maxsubs3 saved5.10% task time against native-routing bracket mean,3.87% against the faster control; maxsubs1 was timing-neutral. Ledger benefited while strong LRU was1.29% slower. Each arm retained the same473/455-token grafts and approximate FINAL continuations. Original strong-LRU grader failure from a missing permitted builtin remains alongside unchanged-code regrades; only iter was added to the execution scope. All830 pressure samples normal; subs1 swap growth.5MiB.

Item29 starts **20:29 UTC**, with a **21:19 UTC** deadline;10 minutes of prior quota-source/launcher preparation are charged. Keep the qualified four-session suffix64/canonical1/PAD1/visible-continuation workflow fixed and compare SSD4096 before,512,128,4096 after, with hot0/native routing. Measure actual eviction, source misses, owned bytes, whole/edited wall, all reads/facts/format/caps and restart. Capacity loss is a measured outcome; do not rescue checkpoints or add a cache API.

Item29 finished early at **20:54 UTC**. [SSD quota report](../experiments/ssd-budget-hour29-20261004/README.md):512MiB retained all four grafts/four approximate continuations, with1.23% additional whole wall and24 endpoint evictions.128MiB lost all four sources, used short exact-prefix recomputation, and added20.69% whole/53.88% edited wall. All32 real reads/all16 edited facts passed; the128 Cwarm READY-plus-correct-prose failure remains, so strict31/32. Both4096 controls repeated public content/usage with0.92% whole drift. All1188 guard samples normal, zero swap growth; observed owned bytes remained under quotas and no admission warnings occurred. Keep the preset4096 headroom;512 is a qualified smaller capacity on this workload.

Item30 starts **20:54 UTC**, with a **21:44 UTC** deadline;10 minutes of earlier source preparation/API review are charged. Compare native-before/reuse64/native-after on two durable Conversations with three public item deletions each. Retain active inventory tool results, require eight actual current-file reads, keep A archive code stable and change B on each edit. Restart after edit1 with the same arm-owned cache/store. Credit approximate SSD sources only when actual graft logs prove them; canonical full-input policy excludes Conversation references, while PAD1 remains eligible. Preserve first replies, format/caps, whole costs and fallbacks without retries or geometry tuning.

Item30 finished early at **21:19 UTC**. [Repeated Conversations report](../experiments/repeated-conversations-hour30-20261004/README.md):all24 logical turns/actual reads and48 HTTP replies completed correctly without caps; all18 public deletions retained every other item and advanced version once. Whole launcher saved36.66%, edited time including deletion saved39.56% against repeatable native brackets (whole drift1.89%). Six actual grafts included four from approximate SSD originals after restart, with all newest Users fresh; B read57→58→59 correctly. The whole saving includes Bwarm's separate1316-token exact native prefix hit. All1116 pressure samples normal with zero swap growth. The caller uses Conversations rather than ds4-agent or explicit cache keys.

Item31 starts **21:19 UTC**, with a **22:14 UTC** deadline;5 minutes of earlier source-only fixture/range preparation are charged. Extend the same caller to one Conversation and eight real obsolete-item deletions, reading archive57→64 afresh and restarting halfway. Compare native-before/reuse/native-after if the budget permits, each with its own cache/store; eliminate the second Conversation's shared-prefix warm benefit. Plot wall and quality by deletion count, retaining actual source kinds and fallback counts rather than assuming an eight-link chain. No runtime or harness framework change.

Item31 finished early at **21:58 UTC**. [Eight-edit report](../experiments/repeated-conversations-hour31-20261004/README.md):all27 logical turns/real reads and54 HTTP replies passed without caps; all24 public deletions retained other items/version invariants. Whole launcher saved58.24% and edited time66.33% against native controls with0.16% whole drift and identical public output/usage. All warm stages stayed near97seconds. Eight actual SSD/pre-RoPE grafts included seven approximate sources, with four edits after restart and every current57→64 code correctly read. Latest Users stayed fresh. All1963 pressure samples normal; native-before swap grew3.75MiB, then remained steady. This is one shrinking-history fixture and a bounded eight-deletion result, not unlimited error-free state retention.

Item32 source preparation began after Item31 models finished at **21:54 UTC**, capped at10minutes and charged to its hour. Two unused deduplication/token-bucket task fixtures use standardJSON and the existing safe execution/checker, frozen161 cases each. The copied full-input read/edit/write caller retains READhigh0/WRITE128/FINAL1; passed-check counts and first failures measure acceptable degradation. Native-before/context-fast/persistent-fast/native-after is ready for the final-hour screen, pending the optional user preference until18:05Eastern. No runtime change or new harness framework.

Item32 starts at **22:05 UTC**, with a **22:55 UTC** deadline including10 minutes prior source preparation. User selected fresh coding quality measurements. Freeze dedupe/bucket161-case fixtures and current runtime; run native-before, context-fast, persistent-fast, native-after sequentially with their own SSD/store and fixed READ0/WRITE128/FINAL1. Retain all first implementations, passed/failed/unresolved checks, caps, real config reads and actual graft/resident counters. No retries or code repair.

Item32 finished early at **22:23 UTC**. [Fresh coding report](../experiments/fresh-clm-coding-hour32-20261004/README.md):all eight first codes passed161/161; actual read/edit/write/final pipelines complete, no caps/errors. Suffix-only saved11.57% task time and combined resident/suffix17.31% against native brackets with1.83% drift; latter6.49% faster than suffix-only. Both candidates logged two native-status SSD/pre-RoPE grafts and approximate FINAL continuations; strong22019 substitutions/no drops. READ thinking diverges before grafts, so package conclusions remain qualified. All826 guard samples normal, zero swap growth. Independent raw-first audit passed.

Item33 starts **22:24 UTC**, deadline **22:50 UTC**, with3 minutes source preparation charged. Fill the supported resident MAX_SUBS1/2/3 curve on unchanged fresh32 dedupe/bucket caller and frozen161-case grades. Opening3 and count0 reference remain in32; run1,2,closing3 with own fresh cache/store/same visible paths, suffix64 and READ0/WRITE128/FINAL1 fixed. Keep first-code partial grades/caps/graft/route evidence and check closing3 input/output/usage repeat; no runtime change or widening parser limits.

Item33 finished early **22:36 UTC**. [Fresh resident curve](../experiments/fresh-resident-curve-hour33-20261004/README.md):all six new first codes161/161 and pipelines complete/no caps. Limits1/2 took193.17/194.23seconds; limit3 opening/closing185.88/188.80, with exact first code/public/usage/routing repeats and1.57% drift. All own-policy native SSD grafts/current Users fresh, zero drops. Actual substitutions6978/14627/22019; all570 new guard samples normal/swap steady. Limit3 fastest tested on these contracts,1vs2 difference unresolved within drift. References remain Item32 evidence, not new independent measurements.

Item34 starts **22:38 UTC**, deadline **22:53 UTC**, including4 minutes source preparation. Qualify actual installed Pi CLI full coding-agent read/write on frozen dedupe contract through the actual ds4-local persistent-fast wrapper. Isolate config/project/cache/store, use Responses/thinkoff/max2048 and grade the first generated write161 checks without retry or repair. This is client/preset compatibility, not a latency comparison or CLM-graft measurement.

Item34 finished early **22:42 UTC**. [Actual Pi agent report](../experiments/pi-agent-preset-hour34-20261004/README.md):installed CLI through actual persistent-fast wrapper completed one real read/write/final; first512-byte implementation161/161, no retries/caps/errors/config changes. Preset suffix64/canonical1PAD1 and6267nativeweight substitutions confirmed, no CLM edit or graft claimed. All52 guard samples normal/swap steady; independent raw-event audit passed. Compatibility task wall53.13seconds has no latency comparator.

Item35 starts **22:42 UTC**, deadline **22:58 UTC**, including4 minutes source preparation. On unchanged frozen fresh32 contracts and persistent suffix/resident profile, vary WRITEhigh thinkingbudget1/32/128 while READhigh0/FINALhigh1 stayfixed. Opening128 reference is33closing3; run1,32,closing128 with fresh owned caches/stores/same visible paths. Retain first grade passed/failed/unresolved/caps and current source/roles/route evidence; no retries. Whole timing brackets and practicalquality form the curve; no pairedNLL/general accuracy claim.

Item35 finished early **22:55 UTC**. [Fresh WRITE-budget curve](../experiments/fresh-thinking-curve-hour35-20261004/README.md):all six new first codes161/161 and pipelinescomplete/no caps. WRITE1/32 saved8.51%/3.76% task wall vs128 mean; fastercontrol savings7.13%/2.31%, with3.00% bracket drift. Actual force1/32/128, unchanged READ38/33 and same graftcoverage; closing128 code/public/usage/routes repeat exactly. All549 new guard samples normal/swap steady. Independent raw-first audit passed; fixedtwo-task post-read evidence rather than generalWRITE1 default. FinalserverSHA matchesfrozenruntime, no experimentserver/agentprocesses remain. Updated linked Pi docs with the actual Responses/8192/2048 preset recipe; standardJSON parsing passed.

Item36 starts **22:56 UTC**, deadline **22:59:30 UTC**, including45 seconds source preparation. Run unchanged installedSDK/Pi provider smoke from17 through actual persistent-fast wrapper: streamedResponses/parentGET/ConversationsCRUD/realtools. Outer120-second lifecycle bound, owncache/store/hot0SSD4096. Compatibility only, no latency/graft claim.

Item36 finished **22:58:30 UTC**. [Actual SDK preset report](../experiments/sdk-preset-hour36-20261004/README.md):unchanged SDK/Pi/Conversations checks allpass/errorsnone, realLIME/COVEreads, storedparentGET exact andCRUD200. Actualwrapper/profile/nativeweights2789/drops0 confirmed; no short-context graft/latencyclaim. All40 guard samples normal/zero swap growth, serverexit0; independent audit passed. Finalprocess check finds no ds4-server/agent. All planned measurements/reports committed before19Eastern reset.

Item42 starts **2026-10-09 14:54 UTC**, deadline **15:54 UTC**, including source review and preparation. Fix the observed live-context read/edit loop with explicit no-dump instructions, stable turn IDs, short code recipes, and concise edit receipts. Labels remain mirror metadata and never enter model tokens/KV. Retain IDs after edits/deletions and preserve original token identities for untouched turns. Run focused model-free checks and one guarded actual-agent editing screen if no user model is active; preserve the first outcome rather than tuning the harness or retrying generation. This item does not change prefill-chunk defaults.

Item42 measurements completed **15:19 UTC**, within budget. [Stable-context report](../experiments/agent-context-structure-hour42-20261009/README.md): build/agent tests/focused CPU checks pass. Actual model archived the exact older turn, deleted ID1 once without renumbering retained IDs2/3/4, then retrieved EMBER-7319 and recalled COVE-3917. Two bounded searches; no full mirror dump, cleanup loop or compaction. First real run found own-export rejection of unlabelled session-date text; retained failure, fixed complete-prefix handling and regression, reran unchanged scenario. The 431-token deletion used normal recomputation (no additional suffix-speed claim). All129 guarded samples normal, swap variation0.25MiB. Runtime `d2952200`; no prefill-default change or new archive framework.

Item43 starts **2026-10-09 23:12 UTC**, one-hour cap ending **2026-10-10 00:12 UTC**; expect a short format follow-up. Replace monotonically allocated turn IDs with opaque random hex IDs at user request; file order defines conversation order, IDs stay with edited/moved turns. Reuse focused CPU fixture to verify deletion, append uniqueness and a reordered edit preserve identities. Build the agent, execute the exact prompt recipe, record outcome and stop; no new model performance screen or harness expansion.

Item43 checks complete **23:15 UTC**. [Opaque-ID report](../experiments/agent-context-opaque-ids-20261009/README.md): random hex64 IDs replace the counter; recipes use string IDs, file order defines order. Build/agent tests/strict-C11 suffix mock pass, including reordered edit preserving ID/body associations and unique append. Exact embedded deletion recipe preserves the kept block/prefix and prints zero bytes. No new model or performance screen; previous Item42 evidence stays intact. Stop this format follow-up here.

Item44 — **2026-10-20, one-hour kernel experiment cap.** On the verified 1017
Q2 checkpoint, test one selected-cache IQ2 gate/up epilogue change: retain four
row reductions in registers through SwiGLU. Source preparation and existing
numerical tests precede the **19:31–19:37 UTC** fixed-teacher ABBA. Keep first
failures; do not expand this into a kernel framework. A short dispatch-only
profile after the scheduler bracket completed the coverage check by **20:02 UTC**.

Item44 complete: baseline/candidate means **19.525/19.530 tokens/s**, no useful
gain. The affected production path is observed in 120 gate/up profile records;
profiled timings are excluded. Mixed SSD-address numerical checks pass, but the
SSD-experts retained-RSS check fails on both baseline and candidate. An ineffective
begin-only pool trial was reverted; root cause remains unresolved. Runtime
shaders/defaults unchanged. [Recipe and evidence](../../recipes/10-q2-kernel-epilogue.md).

Item45 starts **2026-10-20 19:37 UTC**, deadline **20:37 UTC**. Vary only the
existing mixed-prefill quantum across **128 → 512 → 2048 → 128**, with one native
1017 model process at a time, 64 GiB expert budget, two sessions, an active JSON
stream and a new 8,493-token prompt. Measure complete first answers, long-request
wall and active-stream content-event gaps; plot the practical tradeoff.

Item45 finished **20:00:48 UTC**. The first run was interrupted during quantum512
when the user started Homebrew security updates; preserve it separately. After
the user reported completion, restart the entire bracket in a fresh directory.
All four clean arms passed both answers. Quantum128 bracket mean long wall/gap:
**212.11 s / 7.32 s**; quantum512 **304.50 s / 19.30 s**; quantum2048
**96.02 s / 20.39 s**. Long-wall bracket drift **−3.45%**. Quantum2048 saves54.73%
long-request wall but increases the worst content gap to2.78×; quantum512 is
dominated by128. Keep interactive128 default, document2048 as the throughput
tradeoff. This is one fixture, not general task throughput or a DeepInfer
asynchronous-scheduler port. [Report and curve](../../recipes/08-official-inference-lessons.md).

Item46 starts **2026-10-20 21:57:33 UTC**, deadline **22:57:33 UTC**.
Following the approved DeepSeek-inspired sequence, add opt-in incremental tool
schemas: newly discovered Responses tools arrive as late V4.1 system messages,
preserving the initial tool catalog and useful exact prefix cache. Reuse current
parsers, tool ordering and request history. Compare the frozen server with the
candidate on ordinary and favorable tool-search/retrieval workflows; retain first
outputs, protocol failures, cached/fresh input tokens and complete request wall.
Keep routing, suffix approximation, KV disk cache and sampling fixed. Run one
heavy model job under the existing guard. Stop parser/harness work when the
cache/prompt invariant and actual first tool use can be measured.

Queued after Item46: one hour for opt-in approximate decoder replay, then one
hour for a narrowly scoped packed runtime KV prototype. Charge preparation,
builds, measurements and analysis to each item's budget. Keep practical quality
and continuation delta-NLL/KL visible; numerical disagreement alone is not a
rejection criterion for the replay approximation.

Item46 finished **22:23 UTC** (ABBA complete22:16:52). Incremental schemas
remain opt-in: favorable durable-parent workflow saved46.30%, ordinary live
workflow regressed8.17%; all8 first workflows/actual reads passed. Four arms
used the same binary with flag0/1/1/0, and650 guard samples stayed normal with
zero swap growth. A narrow post-measure metadata precedence fix passed the
rebuilt server suite without changing measured prompts.
[Recipe and retained first outputs](../../recipes/11-incremental-tool-schemas.md).

Item47 starts **22:24 UTC**, deadline **23:21 UTC**, including3 minutes
prior source/driver review. Sourcebase61bee1df. Add opt-in fixed decoder replay
to the existing wide-prefill branch only; retain full encoder/compressed/index
state and explicit raw replay visibility. Compare native,128,256 and a closing
native bracket on ordinary/favorable fixed histories with shared teacher rows
and actual first answers. Record direct-sync speed and approximate-state/cache
boundaries; no claim of HTTP2048-chunk savings. One heavy job under guard.

Item47 finished **23:04 UTC** (finalHTTPbracket22:59:17). Replay alone
was1.55–2.00% slower; profiling exposed unchanged whole decoder-layer reads.
Using existing selected-expert batches with128-row replay saved38.17% on the
three direct8192-token fixtures (256 saved34.32%), all12 first answers pass;
128 shared-teacher meanΔNLL+0.00003250/KL0.000004469, noargmaxflips.
Actualunbatched8625-token search/read/answer saved7.03% vsnativebrackets
with0.267% drift, all3firstworkflows/realreads pass. HTTPgainincludes9fewer
searchtokens. Preserved the below-thresholdfixture and initialHTTP+105%
regression caused by discarding approximate live state; a narrow profile fix
now permits verified matching live continuations while canonicalexports and
suffixcapture stay blocked. Existingbatched2k presets unchanged.
[Portable recipe, curves and all first outputs](../../recipes/12-bounded-decoder-replay.md).
