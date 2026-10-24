# Next experiments: one hour each

## Next C-to-Zig migrations — authorized 2026-10-24

The user approved the full sequence. Start from the qualified native
model port at `e37e53d9` (`qwen-glm-zig`), leaving main unchanged. The goal is a
Zig-owned single-Metal engine assembled from model preparation/execution,
session lifecycle and shared storage/I/O. Preserve current model arithmetic,
wire formats, cache policies and threading behavior during each migration.

Each numbered slice has a one-hour ceiling including review, implementation,
builds, qualification and a short report. These are twelve ceilings, not a
requirement to consume twelve hours. At a deadline, record the qualified result
or the concrete obstacle and revise the next slice. Do not widen an unfinished
slice into a generic engine framework or a new test harness.

| Order | Deliverable | Existing qualification |
| --- | --- | --- |
| 1 | Qwen prompt/tool mapping in its model module; use the native model preparation boundary | Rendered bytes/token IDs, server fixtures and real tool/image continuation |
| 2 | GLM53 prompt/tool mapping in its model module, including live tool tails | Same checks with the installed GLM model and bounded SSD settings |
| 3 | Remaining Responses publication transaction and server integration policy in Zig | Response-store fixtures, HTTP/WebSocket continuation and restart recovery; retain small OS/varargs adapters where useful |
| 4 | Generic byte/tensor/F16 payload I/O leaves | Frozen/native payload exchange, short reads and exact recovery using the existing helper |
| 5 | Public snapshots/staged files and dispatch to native family codecs | Existing state/snapshot and SSD-cache fixtures, cancellation/invalid-frontier checks |
| 6 | Selected-expert asynchronous worker start/finish and ownership | Existing SSD-expert fixture, failed-read recovery and short streaming frontier comparison |
| 7 | Page-in range preparation/join, model-view handoffs and read/madvise services | Streaming fixtures plus DS4.1/GLM prefill/decode/MTP; verify joins precede retiring maps/tensors |
| 8 | Shared session dispatch, validation, cancellation and frontier control | Existing state, multimodal, cancellation and Responses fixtures |
| 9 | Single-Metal session creation/free, pools, scratch and reservations | Repeated lifecycle, Qwen arena growth and GLM invalid-state recovery |
| 10 | Engine startup/configuration/admission assembly | Existing metadata/inspect checks, installed model startup and partial-failure release |
| 11 | Engine Metal initialization/close and final family dispatch | All four native families' short execution and shutdown; actual model/kernel owners remain unchanged |
| 12 | Remove superseded production C bodies, verify dispatch and qualify the assembled runtime | Full host suite, selected all-family state/Responses/persistence checks, stable live-append/decode comparisons and build timings |

Keep one authoritative ownership record per component. Existing shared C
headers may remain the ABI definition while callers move; do not add mirrored
Engine/Session structs. Keep the CPU-safe/GPU object split so model-free fixtures
cannot extract a duplicate core. Prefer ordinary modules and explicit family
dispatch over a plugin registry. Complete a slice by replacing its C owner,
with a small compatibility shell where needed; a Zig forwarding facade alone
does not complete an algorithm migration.

Independent source preparation may overlap, with explicit file ownership;
integration and guarded model runs stay sequential. Reuse the downloaded
weights and the established stable within-session append/decode workloads.
Qwen first-prefill variability is already documented; do not turn that noisy
startup measurement into a port regression or speedup claim. Investigate any
repeatable runtime/build regression before accepting that slice. Keep raw
artifacts ignored. Retain small working checkpoints, then consolidate qualified
related slices into a short reviewable stack.

Objective-C Metal ownership/ARC, MSL kernels and third-party codecs/linenoise/
radix trees remain explicit dependencies. Legacy/reference/distributed C stays
separate from the normal single-Metal ownership path; migrate it only for a
concrete use. Cache capability extensions and I/O optimizations are subsequent
experiments, after existing behavior has one clear owner.

Slices 1–11 and common batch follow-up8b are qualified. Route review also found
ordinary C continuation/rewind, sampling, multimodal assembly, canonical rewrite/
live steering, stateless generation and GLM SSD map/cache policy. Their bounded
native follow-ups are complete, using actual records and existing fixtures.
Slice12 combined qualification is complete after dead-helper removal:103 live
protocol checks, host/GPU/state/frozen-logit checks, matched runtime/build screens.
The small Flint terminal-selection regression was fixed. No material steady-
state regression appears in these screens; retain recorded timing variation.
Do not count
inactive legacy fallbacks as active owners or extend this work into a generic
framework. [Progress/evidence](../experiments/native-runtime-20261024/README.md).

## Residual steering — implemented and qualified 2026-10-24

The user replaced the original zero-overhead target with an acceptable cost of
roughly a few percent, selected residual-stream projection, and then requested
both projection and addition. No compatibility with old module-output vectors
is required. Native DS41, Flint, Qwen and GLM Flash pass both modes on installed
weights; retained Pro/full-GLM source is adapted but lacks checkpoint validation.
See the [qualification report](../experiments/steering-20261024/README.md).

- Project uses normalized per-layer directions and unscaled per-layer strength.
  Add preserves learned magnitude and divides strength across nonzero layer rows.
  Default is post-FFN/full-block residual; optional attention-residual editing
  uses the same bank. HC streams are edited independently using one shared axis.
- Ordinary/zero kernels stay selected without an active edit. Tests cover exact
  zero logits, independent live strengths, batches, Qwen/GLM speculation and
  Flint full eight-token packets. Live changes affect subsequent computation;
  existing KV retains its earlier influence.
- Short mirrored fixed-token screens and their controls are recorded. Keep raw
  results local and report variation; no universal zero-cost claim. One small
  Flint additive condition-hoist follow-up closes optimization for this slice.
- Reuse the extractor, layer masks and Nu strength sweep for practical banks.
  These tiny contrastive banks establish functionality, not useful refusal/style
  control. Held-out behavioral/quality curves are a later experiment with a
  chosen behavior objective, not a requirement to perfect this implementation.
- Keep automatic KV SSD/suffix reuse excluded when a bank is loaded, including
  zero preload; expert-weight SSD streaming remains available. Cache identity
  and restoration compatibility require a separate concrete experiment.


## Selective Zig migration, authorized 2026-10-22

Keep `main` fixed. Preserve the previous translation on `zig-rewrite`; work
from the C implementation on `zig-modular`. The user extended the window to
about seven hours. Each item has a one-hour ceiling including source review,
builds, qualification and a small commit; finish sooner when the evidence is
sufficient. Keep readable source and meaningful original comments as acceptance
criteria alongside behavior and build/runtime cost. Do not fill the budget by
expanding the harness or migrating disconnected code merely for line counts.

| Order | Ceiling | Deliverable |
| --- | --- | --- |
| 1. Checkpoint cache | 03:27:33–04:27:33 UTC | Build original C/ObjC with Zig, substitute only the handwritten checkpoint cache, run the existing production-linked fixture in Debug/release, and check actual SSD restart plus tool retrieval in both C/Zig directions. |
| 2. KV persistence | One hour | Review the handwritten store against C formats, ownership and eviction policy; migrate only if it remains legible. Qualify the actual object with existing fixtures and a real caller. |
| 3. Engine cursor boundary | One hour | Source review identified a stronger small seam than another Metal wrapper: bounded GGUF cursor reads and metadata skipping, with no graph/OS ownership. Preserve exact bounds, position and error behavior; qualify real production calls. Keep tensor/model tables and ARC host ownership in C. |
| 4. Small host utilities | One hour | Reuse clear SSD/configuration and prompt-prefix modules. Preserve comments, test against existing C fixtures, and measure incremental builds. Avoid migrating CUDA argument policy merely for coverage. |
| 5. Numerical helper | One hour | Migrate a bounded engine helper only with its numerical caveats and production-linked tests intact. Engram is a candidate; select based on the preceding review. |
| 6. CLI boundary | One hour | Extract bounded transcript/session helpers through the existing opaque engine API. Source review found a whole CLI rewrite too large for one qualified hour; keep terminal/decode glue in C rather than dropping REPL, vision or diagnostic behavior. |
| 7. Metadata directory | One hour | Extend the established GGUF cursor boundary to borrowed key/type/value-offset records. Keep model allocation and fatal policy in C; qualify partial progress/alignment and real model loading. |
| 8. Tokenizer maps and BPE | One hour | Replace tokenizer-local maps with opaque borrowed-key maps and represent merged BPE symbols as spans of one encoded allocation. Preserve ranks/ties/fallback; compare exact real-vocabulary IDs and warm tokenization cost. |
| 9. Metal publication | One hour | Selectively salvage the readable F16 publication controller through a minimal typed Objective-C bridge. Check retained command lifetime and actual GPU results before keeping it. |
| 10. HC prefill controller | One hour | If the minimal bridge qualifies, extend only for the ordinary HC prefill controller and its real GPU/shape cases. Preserve the current fallback. |
| 11. Benchmark frontend | One hour | Move the complete benchmark frontend with real snapshot/replay and multiplicative cases; preserve public CLI output. |
| 12. Pretokenizers | One hour | Move all three ordered family splitters using existing Unicode tables; compare actual vocabularies and mixed microtimings. |
| 13. Complete CLI | One hour | Build on the chat/options boundaries to move one-shot, REPL and diagnostics; qualify actual generation, session changes and SIGINT rollback. |
| 14. RMS/QKV dispatch | One hour | Extend the typed resource bridge only for ordinary RMS controllers; qualify actual CPU, GPU and model callers. |
| 15. Image decode | One hour | Move decode ownership, orientation and identity over existing codecs; compare full pixels/fingerprints and warm decode cost. |
| 16. Chat rendering | One hour | Move family rendering/control policy; use standard marker scanning and measure actual prompt/token parity and formatting cost. |
| 17. Vocabulary | One hour | Move construction/release and token-byte decoding over real GGUF cursors; compare every real vocabulary entry. |
| 18. Image preprocessing | One hour | Move resizing/patch/layout policy; measure geometry, full arrays, ordinary/extreme inputs and official DeepInfer layout agreement. |
| 19. Activation controllers | One hour | Move quantization/Engram/pool host dispatch; qualify ownership, actual GPU kernels and alternating model performance. |
| 20. Streaming tool parser | One hour | Move four syntax parsers and call owners over shared real records; run existing production-linked Debug/release fixtures and retain discovered failures. |
| 21. Combined qualification | One hour | Run all retained modules together, measure real build/runtime costs, test cross-restart retrieval and repeated suffix reuse. |
| 22. Numeric sampling | 05:33–06:33 UTC | Move active argmax/RNG/filter/probability/correction algorithms behind a primitive typed API; keep session/penalty/stop policy in C. Reuse actual sampling hooks/fixtures, record production versus finite-aware diagnostic math, then qualify real seeded generation and microtimings. |
| 23. Resident-route policy | 05:43–06:43 UTC | Move the complete active nine-function route policy using existing records and frozen-C differential fixtures. Preserve caps/ordering/arithmetic, then compare actual resident-fast coding/route counters. |
| 24. Final review | Remaining window | Inspect retained tradeoffs and exact remaining C scope, install qualified binaries and consolidate a short reviewable branch while retaining individual experiments. |

Items1–20 are retained with separate reports linked from [the module guide](../../src/README.md).
Item21 passed against frozen code `b8c71f98`; item22 sampling also passed CPU and actual fixed-seed callers. Resident-route source preparation follows those measurements. The item numbers record sequence,
not hours consumed. Source preparation overlapped; compiler/model measurements
remained sequential. The server JSON survey retained C because standard-scanner
compatibility changes would expand this migration. No harness framework or
whole-runtime translation dump was introduced.

The final qualification will retain timings and approximation differences,
record the substantial engine/Objective-C scope that remains, and preserve the
individual experiment commits behind a short reviewable branch. Main stays fixed.

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

Item29 finished early at **20:54 UTC**. [SSD quota report](../experiments/ssd-budget-hour29-20261004/README.md):512MiB retained all four grafts/four approximate continuations, with1.23% additional whole wall and24 endpoint evictions.128MiB lost all four sources, used short exact-prefix recomputation, and added20.69% whole/53.88% edited wall. All32 real reads/all 16 edited facts passed; the128 Cwarm READY-plus-correct-prose failure remains, so strict31/32. Both4096 controls repeated public content/usage with0.92% whole drift. All1188 guard samples normal, zero swap growth; observed owned bytes remained under quotas and no admission warnings occurred. Keep the preset4096 headroom;512 is a qualified smaller capacity on this workload.

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

Item48 starts **23:04 UTC**, deadline **2026-10-21 00:03 UTC**, including
1 minute prior preparation. Sourcebaseaf29fced. Prototype narrowly scoped packed
runtime KV using existing typed loads/conversion where possible; original
quantizer outputs are BF16-representable. Keep defaultF32behavior, avoid a
permanent expanded mirror, measure actual cache allocation plus nativebracket
latency/sharedteacherquality/firstanswers. ReuseItem47fixtures/probe ratherthan
new qualityharness. Keepcanonicalpayload/suffixcompat explicit; retain negatives.

Item48 measurements finished **23:29 UTC**, within the one-hour cap.
Main compressed KV now has opt-in F16 storage for the already-quantized values,
with no expanded full-cache mirror. At context capacity262144, actual main
allocation halves1280.008→640.004MiB; main+index+raw falls1610.010→970.006MiB.
Expert-cache budget stays64GiB. The same-binary0/1/1/0 bracket passed all12 first
answers; all4×22 teacher rows were bit-identical (ΔNLL=KL=0). Mean direct time
69.960→69.282s is unresolved:0.969% saving vsmean,0.474% slower than the faster
control,2.914% native drift. Odd8193 canonical F32→F16→F32 snapshots and fresh
8194 logits passed bytewise; suffix relocation remains excluded.
Server/agent suites and focused GPU checks pass. Default full Metal suite fails
an unchanged quantizer tie at index5428 on both parent and candidate; the
existing safe-math diagnostic passes the full suite. Defaults/timed settings
remain unchanged and the first failure is retained, rather than widening this
storage experiment into a quantizer change.
[Portable recipe and all evidence](../../recipes/13-packed-compressed-kv.md).

Item49 starts **2026-10-21 01:18 UTC**, one-hour cap ending **02:18 UTC**.
Remove duplicate tool definitions from incremental Responses discovery: retain
one chronological system announcement and a short tool-result receipt, while
preserving explicit caller output/result and original private schema metadata.
Qualify live/full-input and durable-parent workflows with the existing Item46
driver, then promote the single-copy rendering to the DeepSeek41 default with
legacy0 available. Preserve first outcomes and native brackets; no new harness
or cache-policy implementation belongs in this follow-up.

Item49 measurements finished **01:33 UTC**, within budget. The default-unset
0/u/u/0 bracket passed all8 first workflows and real file reads. Single-copy
announcements saved57.45% on the durable-parent workflow (188.734→80.300s),
81.62% after discovery. The short live workflow remained4.00% slower
(29.502→30.683s); R2+R3 was4.23% slower, with9 extra format tokens rather than
the old implementation's73. Candidate R1, before any schema announcement,
accounted for0.769s of the1.181s whole-workflow gap; raw timings are retained.
Keep this coherent server default for persistent-agent discovery, with explicit0
as the legacy escape hatch. Build/server tests pass; all633 guard samples
normal, peak wired+compressed98.10GiB, max swap growth3.87MiB. Cache restrictions
were clarified as unfinished prototype integration, without altering runtime.
[Recipe and current evidence](../../recipes/11-incremental-tool-schemas.md).

Item50 starts **2026-10-21 01:40 UTC**, deadline **02:40 UTC**.
Support durable replay checkpoints using a small versioned state envelope around
the existing native body and an explicit resumable replay kind in the SSD
manifest. Preserve native/grafted formats and immutable chunk sharing. Bind
replay policy to cache identity, restore its approximate status, and keep replay
out of suffix-source lookup. Reuse existing model-free cache checks and one
guarded restart/continuation qualification; no general format rewrite.
Item51 follows sequentially, with one hour for F16 suffix relocation and
repeated moves from immutable originals, using the existing probe.

Item50 complete **01:52 UTC**, within budget: private DRP1/REPLAY persistence
retains approximate identity and old formats. Odd8193 SSD reopen into F16
retained all bytes and next-token129280 logits exactly. Two fresh server PIDs
passed actual read workflows; restart restored8616 tokens with no replay sweep
(47.713s cold/18.628s restart, one ordered pair). All90 pressure samples normal.
Existing HTTP usage underreports state-cache hits; log evidence is retained and
a small reporting correction is a separate follow-up.
[Report](../experiments/replay-persistence-hour50-20261021/README.md).

Item51 starts **2026-10-21 01:53 UTC**, deadline **02:53 UTC**.
Support suffix capture, canonical-prefix restore and zero/even left/right grafts
with F16 main compressed KV. Reuse the canonical source plus one bounded F32
rotary-repair tile; preserve immutable pre-RoPE originals, no full F32 mirror.
Qualify repeated moves and both layout transitions with the existing Item39
probe, then existing SSD/HTTP integration if time permits. Keep F32 controls
and measure additional distribution change separately from stale suffix error.

Item51 complete **02:34 UTC**, within budget. F16 suffixcapture/restore/zero and
evenleft/right grafts use one256KiBrepairtile. All6lowlevelarms pass; all5
originalbackedlogitfiles byteidentical (native16+grafted16freshrowsperarm),
zeroadditionalF16DNLL/KL. LegacyF16left4rows passes separately. F16edit
time+6.55%vsF32controls for halvedmainallocation; keepoptional. All7HTTPsteps
and6actualreads pass, including1368token nativeSSDrestore andtwo832token
approximateSSDgraftsafterrestart. [Report](../experiments/f16-suffix-hour51-20261021/README.md).

Item52 starts **2026-10-21 02:35 UTC**, 10-minute cap **02:45 UTC**.
Correct demonstrated HTTPusage underreporting for state-cache prefix hits.
Carry actual newly restored prefix credits out of sync, preserving live/graft
counts and excluding newly computed tokens. Reuse Item50 two-process restart
driver with cached_tokens required to equal the actual hit length; no new
harness or general cache-accounting redesign.

Item52 complete **02:38 UTC**, within10mincap. Bothactualreadworkflows pass;
restartHTTP reports8618cached/0writes matching8618restoredtokens. R2/R3
countsretainlive8675/8827andfresh88/31; all6read+write totalsmatchinput.
Server/agentsuites pass; all67guard samplesnormal, zeronewswap.
[Report](../experiments/state-cache-usage-hour52-20261021/README.md).

Item53 starts **2026-10-21 02:46 UTC**, one-hour cap **03:46 UTC**.
Profile F16 vs F32 small-edit costs, separating prefix/fresh work from graft
repair. Inspect publication/gather/attention kernel launches and typed-copy
overheads; try one targeted simplification after a concrete baseline. Phase
CSV inspection assigns94% of Item51's total gap to prefix reconstruction,
so do not attribute it to rotary repair. Selected F16 rows currently expand
to F32 then repack for scalar attention; remove that round trip and reuse
existing gathered staging. Reuse focused GPU checks and ordinary ABBA plus
favorable paired model probes, preserving capacity/routing. Retain quality,
phase timing and the first measured result.

Item53 complete **03:14 UTC**, within the one-hour cap. Packed scalar gather
removes the F16→F32→F16 round trip and enables existing gathered staging.
Two isolated primitive repeats improve gather+attention by1–16%, but ordinary
ABBA apparent1.91% saving is inside7.04% control drift; favorable singlepair
0.47% is weak evidence. Same-frontier256-step paired decode is neutral:
20.7582→20.7405tokens/s, all273frontiers exact. All6taskanswers pass and
15sharedteacherpositions match. Three original-backed moves/recaptures pass;
24freshnative/graft rows matchItem51counterparts exactly. All669pressure
samplesnormal. Keepreader experimental/defaultoff; no model-wide gain claim.
[Report](../experiments/f16-gather-hour53-20261021/README.md).

Item53 default follow-up: the packed reader now follows opt-in F16 storage
automatically, with `DS4_METAL_V41_PACKED_F16_GATHER=0` as the comparison
override. Exact measured outputs and neutral overall speed do not justify
a separate enable switch. The paired decode tool now accepts an explicit
control environment value so default-on comparisons still exercise both paths.

Item54 — **2026-10-22 03:27 UTC**, selective Zig restart on `zig-modular`.
Main stays at `ec5424de`; the complete mechanical translation is preserved as
`zig-rewrite` (`108f90cf`). Each replacement has a one-hour ceiling, with source
preparation in parallel and builds/model measurements serialized. Report folder
`hourN` names are item numbers, not hours consumed. Retained so far: build graph,
checkpoint/KV persistence, GGUF cursor/index, SSD/prefix utilities, Engram,
CLI transcript helper, token maps/span BPE, and typed F16/HC Metal controllers.
Each has a linked qualification under `src/README.md`; exact measured behavior
and real cache/model workflows are distinguished from source-only work.

Next in order, each at most one hour: complete benchmark frontend (snapshot and
replay at successive frontiers); complete pre-tokenization (ordinary/multilingual
corpora and all available model families); complete interactive CLI (one-shot,
REPL, diagnostics and cancellation); then the normal-path RMS Metal controller
family if its small bridge remains readable. Close with combined real serving,
build/incremental cost, comments/source review and a short rebaseable stack.
Overall work window ends about **10:27 UTC** or earlier if tokens expire. Stop
an individual migration if its boundary becomes a generic translation layer;
keep the existing C reference build available. No runtime-default changes are
part of the language migration.

Selective Zig restart update **2026-10-22 05:59 UTC**: the bounded components
through numeric sampling and resident policy are qualified. The sampler saves
57.76%/42.85% CPU time in its existing alternating fixture; this is not a model
throughput claim. The route policy preserves all six coding results and actual
substitution counts, with1.60% more whole-task time against its C bracket mean.
Reports and exact scope remain linked from `src/README.md`.

Next, sequential qualification with at most one hour per item (source prep may
overlap): agent atomic file/edit operations; GGUF typed getters and tensor
directory; F32 Metal matmul dispatch. Keep each existing C reference and actual
production fixtures. Finish with the default installation, source/comment
review and a short stack preserving the detailed steps. The work window still
ends about10:27UTC or when tokens run out; leave main unchanged.

Selective Zig restart final qualification **2026-10-22 06:32 UTC**: 27 handwritten
implementation files / 9,532 physical lines, plus the small binding adapter.
All 78 CPU fixture invocations and four builds pass; the final Debug router
passes all 90 GPU cases. Final C/native persistent-agent arms complete twelve
turns, six fresh archive reads/correct answers and six SSD/pre-RoPE grafts,
including four after prior approximate states. All 437 pressure samples remain
normal with no swap growth. All five default installed binaries match the
qualified native binaries. [Final report](../experiments/zig-final-20261022/README.md).

The focused build comparison records Make/native full points 9.184/9.557 s,
no-op means 0.0127/0.3997 s and migrated Metal-controller edit means
3.2460/0.5005 s. Retain the costs and both stopped measurement attempts;
no whole-model speedup is claimed. The implementation and journal are being
consolidated into a short stack, preserving detailed steps separately. Main
and the earlier mechanical rewrite stay unchanged. Substantial engine,
Metal resource/attention, server and agent orchestration still remain C;
see `src/README.md` for exact boundaries. This closes the accepted selective
restart stage; do not open another boundary just to consume the time budget.

Item55 starts **2026-10-22 12:18 UTC**, one-hour first-stage cap **13:18 UTC**.
The user approved continuing the handwritten rewrite through the Metal engine
and remaining applications. Work on `zig-engine`, based on `zig-modular`;
main remains frozen. First stage: actual native model mapping/lifetime,
explicit DS41 model configuration and typed weight binding. Preserve private
versus shared mappings and exclusion of Engram/Qwen disk-only tails. Reuse the
actual model fixtures, add only open/config/bind/close identity/lifetime coverage,
and run the existing guard serially across all three real model families.

Subsequent stages, broken into measured one-hour work blocks: a working native
DS41 graph/prefill/decode path; session/checkpoint/replay/suffix state; remaining
Metal family controllers/speculation/batching/vision; server and agent
orchestration. Source preparation may overlap; builds and model jobs stay
serialized. Keep Engine/Session handles opaque and move resource ownership in
cohesive units. Use one actual shared record where transitional C callers
need its fields; never mirror giant engine/session layouts or add a field
accessor framework. Keep the ARC/resource bridge, shaders and third-party
libraries in their natural languages. Record partial scope at each checkpoint.

Item55 input ownership passes ReleaseFast/Debug real model lifecycle identity
checks across DeepSeek, GLM and Qwen, the existing negative mapping fixtures,
the model-free session fixture, and a short Metal continuation. Native DS41
configuration and weight binding use values and borrowed tensor descriptors;
other families still retain C validation/binding. No startup/throughput gain is
claimed. [Report](../experiments/zig-engine-input-20261022/README.md).

Item56 graph ownership started 12:41 UTC (one-hour cap 13:41 UTC), finishing
qualification by 12:51. Layout/allocation/workspace/reset/free now use explicit
configuration and the actual shared graph record. Existing C/native planning
and real allocation/accounting fixtures pass, including Debug native objects.
The 512-prefill/128-teacher ordinary ABBA bracket retains exact frontier logits
with measured inference time +0.28%; F16 pair also matches. This is a neutral
ownership change, with control drift retained. C attention/execution still runs.
[Report](../experiments/zig-engine-graph-20261022/README.md).

Item57 scalar and batch attention starts 12:52 UTC, one-hour cap 13:52 UTC.
Move the actual controllers over explicit configuration and public Metal APIs;
preserve rounding/publication, pre-RoPE originals, F16 caches, TP partitions and
causal score tiling. Qualify real scalar decode as well as prefill; saved prefill
frontier vectors alone cannot detect a decode-only regression. Keep the existing
C reference and one pressure-guarded model job at a time.

Item57 qualification completed by 13:09 UTC. Six full prefill frontier vectors
match within storage mode. Three 32-step greedy decode pairs (F32, packed F16,
legacy F16) match every selected ID and top-20 record, as does native Debug.
The six-turn native agent check passes all three fresh archive reads and three
SSD/pre-RoPE grafts, including two from prior approximate states. Throughput is
neutral in the noisy short bracket; no inference speed gain is claimed.
[Report](../experiments/zig-engine-attention-20261022/README.md).

Item58 complete DS41 execution starts 13:10 UTC, one-hour cap 14:10 UTC.
Activate the reviewed native layer/token executor, MoE, streaming-map/preparation
owners, prefill sweep and session-batch controllers as one cohesive path. Keep
private diagnostic/cache services and DSpark ownership explicit at the boundary;
preserve Metal flags, cancellation, replay, image/TP routing, retained views and
thread joins. Qualify ordinary scalar and prefill, warm/selected appends, actual
archive/suffix use and a longer carry case with existing tools. Keep a frozen
attention-stage control, and record failures before changing scope.

Item58 completed by 13:30 UTC. All eight ordinary/F16/8,192-token frontier
vectors match the frozen attention control. Three 32-step scalar pairs and
native Debug match full top-20 records. Image encoding plus a follow-up query,
batch admission and six-turn archive/repeated-suffix use pass. Native execution
owns layer/token sequencing, MoE, mapped preparation, prefill and small batches;
outer session policy and encoder residency remain C. No shader change.
[Report](../experiments/zig-engine-execution-20261022/README.md).

Item59 canonical DS41 payload activation starts 13:32 UTC (one-hour cap14:32).
Its source/fixture preparation overlapped the preceding qualification. Own the
unchanged DSV4v2 numerical wire format over actual Graph/Config; keep Session
policy and canonical/replay/SSD envelopes in C. Compare actual cross-build disk
files and fresh full-vocabulary continuations at odd/even/raw-ring frontiers,
with F32 and F16 runtime storage. Test the demonstrated mutation boundary with
three bounded failure cases, and record cross-storage rounding separately.

Item59 finished by13:36 UTC. Unchanged DSV4v2 payload files and freshly computed
full-vocabulary logits match C↔native in F32/F16 at six frontiers. Both storage
conversion imports and native Debug also pass. Invalid header/token preserves
the old frontier; post-mutation short reads invalidate it and good reload
recovers.63 normal pressure samples, no swap growth. No latency claim.
[Report](../experiments/zig-engine-payload-20261022/README.md).

Item60 suffix/pre-RoPE graph activation starts13:37 UTC (cap14:37), following
source preparation/review during prior jobs. Keep actual session provenance,
canonical/exact prefix eligibility, checkpoint ownership and fresh-tail sync
in C. Native owns original tape, graph-copy/relocation, F16 repair tiles and
coverage/publication over actual records. Reuse the existing right/left/zero
repeat probe and practical six-turn archive workflow; compare payload-stage
control with one changed graph owner at a time.

Item60 completed by14:25 UTC. All three C/native pairs (F32 originals, F16
originals and legacy F16 left) retain identical complete teacher vectors,
tokens, geometry/source hashes and quality records. Native Debug legacy F16
also matches. The six-turn agent passes three accepted edits, three fresh
archive reads and three SSD/pre-RoPE grafts, including two after prior
approximation and one using an approximate source. No new speed claim.
[Report](../experiments/zig-engine-suffix-20261022/README.md).

Item61 DSpark ownership activation starts14:26 UTC (cap15:26), after
source-only preparation/review during prior qualifications. Move the actual
draft records once; native owns allocation/capture, three-stage draft math
and bounded target verify/commit. Preserve a pre-activation target fixture,
then compare real target capture/rollback and fresh continuations in F32/F16
and Debug. Trained draft inference is unqualified without matching weights;
record that limit instead of treating target-only coverage as a draft result.

Item61 completed by14:44 UTC. Five target arms retain identical full vectors
and state records. Matching1017 DSpark weights were downloaded and hash-checked;
all49 trained oracle tensors match C/native ReleaseFast/Debug byte-for-byte.
Both runtime fixtures pass keep0..6, rewinds/eight fresh steps, exact sampling
0/1, payload restore, cancellation and context clamping. Two common loader
admission/binding failures were isolated and fixed (redundant family metadata,
upstream Markov names); strict layouts remain. No throughput claim; outer
Session cycle remains C. [Report](../experiments/zig-engine-dspark-20261022/README.md).

Item62 V4.1 speculative-cycle activation starts14:46 UTC (cap15:46).
Compose the qualified native drafter/target executor with existing numeric
sampling and explicit borrowed action arguments; move the actual stats record
once. Keep C TP dispatch/public admission/rewind and both fallbacks. Preserve
a C-cycle/native-drafter control first. Compare all 16 cycle accepted-token/RNG
records plus existing real causal/rewind/restore/cancel/context invariants;
qualify native Debug too. No new runtime approximation or tuning choice.

Item62 completed by14:53 UTC. C-cycle/native-drafter control and native
ReleaseFast/Debug pass all runtime assertions and16/16 exact token/RNG records.
Both sampling modes actually accept multi-token drafts (3/8 ordinary cycles,
8/8 exact cycles). C retains public admission, transport dispatch and rewind;
native owns the synchronous cycle. No throughput claim.
[Report](../experiments/zig-engine-cycle-20261022/README.md).

Item63 native vision controller activation starts14:55 UTC (cap15:55).
Keep the existing BF16 sidecar binder and ARC/PSO/shader leaves in C/Objective-C;
native owns per-image workspace,32block/BF16/3x3 aligner flow and readback.
Preserve a C controller before selecting the standalone native GPU entry.
Compare complete finite/repeated embeddings in C/native/Debug on the real1017
sidecar, then actual image+follow-up CLI. Legacy4096-sidecar remains unqualified.

Item63 completed by15:03 UTC. C/native ReleaseFast/native Debug produce identical
complete 921,600-value embeddings, with repeated encodes internally exact. Actual
image question and follow-up both answer RED. Native owns the workspace/layer
controller; sidecar binding and shader leaves stay C/Objective-C. Legacy4096
sidecar remains unqualified. No latency claim.
[Report](../experiments/zig-engine-vision-20261022/README.md).

Item64 HTTP activation starts15:04 UTC (cap16:04). Freeze existing C-transport
control, activate the prepared native framing/socket owner, and run existing
server CPU fixtures in ReleaseFast and Debug. Follow with the cohesive input
codecs and coordinator, then one cumulative real concurrent/durable HTTP check.

Item64 transport CPU qualification passes by15:05 UTC in ReleaseFast and Debug.
One explicit uint16 annotation at htons fixed the first compile failure.

Item65 cohesive endpoint/tools/messages/Responses input activation starts15:05 UTC
(cap16:05). Compile one export facade over pure owners and the actual shared
records, with the existing C primitive JSON parser and single materialization
pass. Run the broad existing server fixture in ReleaseFast/Debug, preserving
first failures. Keep runtime/render/model policy in C.

Item65 request qualification passes by15:07 UTC in ReleaseFast and Debug.
The original broad endpoint/tool/schema/image/Responses fixture passes unchanged;
first compiler API/type errors are retained. No request grammar change.

Item66 coordinator activation starts15:07 UTC (cap16:07). Native owns admission,
round-robin/quantum scheduling, pending/in-flight cancellation, eval batching
and shutdown over the one actual shared server/slot/job definition. Keep C
routing/session generation/cache policy, exercise existing CPU schedules first.

Item66 CPU qualification passes by15:09 UTC in ReleaseFast and Debug. Native
coordinator preserves existing fairness/quantum/pending/in-flight cancellation
assertions. First shadowing compile errors and a harmless records-only preparation
build are retained separately.

Item67 exact DSML tool-memory ownership starts15:09 UTC (cap16:09). Native owns
shared immutable blocks, ID/refcount/LRU and unchanged KTM1 trailer read/write
over actual records; C keeps enable-policy and directory scan. Run existing
replay/prune/trailer/server tests, then the cumulative actual HTTP workflow.

Item67 CPU qualification passes by15:11 UTC in ReleaseFast and Debug. Existing
exact replay/pruning/trailer tests pass, with no new I/O callbacks or cache policy.
The cumulative native production server is frozen for one real two-slot, actual
disconnect, durable Conversation/Responses restart campaign under the pressure
guard. C fallback build/tests are checked before model execution.

Items64–67 cumulative real HTTP qualification passes by15:16 UTC: five graded
answers, overlapping clients, actual two-row decode batches, cancellation after
a real SSE text delta, exact stored objects/history across restart, independent
continuations and CRUD including item GET404. First probe falsely expected a
delete receipt where this existing API returns the Conversation; fixed only the
probe and retained both campaigns. C/native ReleaseFast/Debug CPU fixtures pass.
[Report](../experiments/zig-server-20261022/README.md). No latency claim.

Item68 Context transaction activation begins15:17 UTC (cap16:17). Freeze a
C-context/native-engine control first; exercise existing model-free stable-ID /
suffix rollback tests and guarded real-vocabulary import invariants. Preserve
C worker graph/graft/import ordering. One demonstrated fixture omission for
decoder-replay detection is repaired before selecting the native owner.

Item68 completed by15:21 UTC. Native ReleaseFast/Debug pass the existing
model-free stable-ID/control/edit/transaction/rollback cases. C-context control,
native and native Debug also pass real-vocabulary imports without evaluating
model tokens. The C-control crash was a stale mock missing decoder-replay
detection; its fake Session now explicitly reports no decoder replay. Keep
all first failures. Native owns text/turn transactions; C owns graph decisions.
[Report](../experiments/zig-agent-runtime-20261022/README.md).

Item69 Bash process owner activation starts15:21 UTC (cap16:21). Native owns
spawn/process group/monitor, descriptors, timeout/reaping and output snapshots.
C retains job list/IDs/terminal UI. Reuse the actual process tests including
closed stdio, descendants, isolation, timeout and large output, in ReleaseFast
and Debug. Run the combined practical agent task after the final agent slice.

Item69 CPU qualification passes by15:25 UTC. Actual native spawn/closed-stdio,
large-output capture, timeout, descendant kill and process-group isolation
assertions pass in ReleaseFast and Debug. Cumulative suffix/Context mocks also
pass both modes. First ABI/length inference compile errors are retained.
[Report](../experiments/zig-agent-runtime-20261022/README.md). The actual model
agent integration follows after the final compaction planner slice.

Item70 Responses output activation starts15:25 UTC (cap16:25). Native owns
stream/nonstream item identity, reasoning/text/tool lifecycle, usage/status and
sequence emission over actual records. Keep C durable publication and tolerant
argument codec as whole library actions; preserve the original fallback. Add
only demonstrated lifecycle gaps to the existing server socket fixture, then
run the already-qualified actual concurrent/durable HTTP workflow again.

Item70 completed by15:30 UTC. Native first compile was clean; C-control/native
ReleaseFast/Debug plus full-C fallback fixtures pass. Targeted socket cases
cover UTF-8 reasoning/text transitions, incomplete reasoning and tool-event
identity/order. The actual concurrent HTTP/cancel/durable restart campaign
passes five answers and eighteen structural checks. No latency claim.
[Report](../experiments/zig-server-output-20261022/README.md).

Item71 compaction planner activation starts15:30 UTC (cap16:30). Freeze the
C two-phase plan before selecting native summary-prefix/budget/tail/same-message
resume construction. Keep the single C model/UI/image/graft/rollback controller.
Check existing boundaries/mocked full compaction and one real-vocabulary open
thinking-assistant case, then real interactive /compact and final six-turn
archive/edit/retrieval workflows. Final ordinary runtime/build measurements
follow at the same frozen source.

Item71 completed by15:47 UTC. Ten CPU fixtures and all three real-vocabulary
C/native/Debug resume cases pass. Actual interactive C/native `/compact` both
retain summary-only facts and answer correctly:5459→3365 /5459→3314,819-token
tails. Interactive start dates differ (fixed-time override is noninteractive
only), so these are independent functional proofs, not matched-output/timing
comparisons. No new compaction policy.
[Report](../experiments/zig-agent-compaction-20261022/README.md).

Item72 final Metal tensor resources activation starts15:49 UTC (cap16:49).
Native owns allocation/view/range policy, standard live registry and accounting;
keep retained Objective-C handles and true ARC/buffer/blit leaves. Preserve a
C-resource control, test nested views/zero ranges/parent retirement/double free
with the actual API, then existing8192-cycle debug-layer memory modes and full
Metal numeric fixture. Final actual agent workflow/runtime/build comparisons
follow at this frozen source. No further owner expansion before measurements.

Item72 completed by15:59 UTC. Native ReleaseFast/Debug build cleanly. All twelve
actual command-memory cases pass with Metal validation, including nested views,
parent retirement and duplicate-free protection; all three complete numeric
suites pass with the existing math-safe setting. No shader or command ownership
change. Final same-source C/native agent/runtime/build qualification follows.
[Report](../experiments/zig-metal-resources-20261022/README.md).

Selected-engine checkpoint qualification completes16:12 UTC. Matching C/native
six-turn agents pass all twelve turns, six correct fresh archive reads and six
SSD/pre-RoPE grafts (four after prior approximation). Ordinary C/native/native/C
frontiers match; steady decode17.77/17.93t/s with8.4%prefill control drift, so no
runtime speed claim. Warm five-program full builds9.731/8.929s; F32 edit means
3.481/.618s; native no-op overhead.524s remains. Main stays fixed. This is a
qualified partial rewrite checkpoint, not the end of the whole rewrite goal.
[Report](../experiments/zig-engine-final-20261022/README.md).

Next sequence toward completing the supported Metal serving path:

73. Session frontier/sync/eval/invalidate and scoped encoder residency. Move one
    actual shared frontier record; preserve other-family dispatch in C. Budget
    one hour including integration and targeted state/continuation checks.
74. Agent Markdown presentation, then tool-stream projection as separate owners.
    Preserve actual output and comments; no Worker mirrors. One hour per owner.
75. Server DS4.1 prompt/replay renderer and shared tool codec, then the remaining
    generated-reply/protocol output owners. One hour per owner.
76. Metal model-view range/registry policy, retaining four true ARC/driver
    leaves. Check actual repeated target-map replacements and auxiliary reads.
77. Engine V4.1 memory admission and workspace/cache rebalance over one actual
    accounting record. Exercise fit/reject/rollback and actual workspace growth.
78. Exact Metal model-view cache keys, range admission and lifetime policy.
    Preserve command-boundary eviction; check real retained-handle ownership.

Prepared follow-on owners remain separate experiments: Agent Stream after74,
primitive server JSON and generated-reply parsing after75. Each gets its own
one-hour integration/qualification cap; source preparation can overlap the
single guarded model job, but activation and measurements stay sequential.

Follow those with Session lifecycle/budgets/persistence/public batch policy,
server materialization/generation/routing, agent tool/history/worker/UI owners,
and remaining Metal cache/dispatch policy. Explicitly inventory remaining active
project-owned C before claiming completion; true ARC/driver leaves, Metal shader
language and third-party libraries remain legitimate foreign boundaries.

Item73 actual record/controller integration begins16:20 UTC (cap17:20).
First move the seventeen fields once and compile/run the C-controller control;
then select native Session/Frontier/Residency. Keep CPU/distributed routing on
its existing path. Reuse session-state assertions in both model-free and GPU
selections, existing graph/frontier/cancellation and actual model continuations.

Item73 completes17:03 UTC. C/native/Debug scalar Session fixtures and all four
GPU state selections pass. Eight real-model arms have identical full prefill
frontier vectors across F32/F16, demand growth2048→4096 and selected replay256.
C/native default prefill fixtures pass all five cache/snapshot/continued-decode
frontiers and cancellation recovery; both encoder fixtures pass partial-lock
cancel, acquisition, exact cache/logits/decode and budget restoration. Normal
pressure throughout. These single pairs do not establish a speed improvement.
[Report](../experiments/zig-session-frontier-20261022/README.md).

Item74 Presentation activation begins17:04 UTC (cap18:04). First preserve the
shared-record C control; select native rendering only after its byte/screen
fixtures pass. Use the existing terminal/PTY/hint helpers and real code repaint
case in C/native ReleaseFast/Debug. Stream remains a separate next owner.

Item74 Presentation completes17:09 UTC. C/native ReleaseFast/Debug and full-C
fallback all pass actual Agent, terminal/hint/PTY/code-screen and suffix/Context
fixtures. Fourteen stable artifacts match bytewise across all five modes; PTY
uses its existing concurrent screen assertions. No model/performance claim.
[Report](../experiments/zig-agent-presentation-20261022/README.md).

Item74b Stream activation begins17:10 UTC (cap18:10). Move actual projection
records once, preserve an otherwise-native C Stream control, then compose pure
Presentation directly. Reuse byte-split parser/visualization, terminal/PTY and
suffix assertions; actual whole-agent archive/edit workflow follows tool owners.

Item74b completes17:15 UTC. All five actual C/native/Debug/fallback Agent,
terminal/PTY and suffix selections pass; fourteen stable artifacts match exactly.
Native Stream composes pure Renderer directly. No new test body/model/performance
claim. [Report](../experiments/zig-agent-stream-20261022/README.md).

Item75 server prompt/replay rendering activation starts17:16 UTC (cap18:16).
Freeze the otherwise-native C control after moving actual argument records and
adding the shared primitive minify action. Then select native DS41 full/live
rendering and tool codecs; run the unchanged production server fixture in
C/native ReleaseFast/Debug and fallback. Primitive JSON and Reply follow as
separate bounded owners before aggregate actual HTTP generation.

Item75 completes17:26 UTC. All five final C/native/Debug/fallback production
server fixtures pass. Existing Qwen checkpoint rendering exposed allocated-empty
buffer adoption losing its pointer; the same input-vector bug was reproduced
with a valid repeated stop field. Fixed actual helpers and added one same-suite
regression; C renderer controls rebuilt afterward. No assertions weakened.
[Report](../experiments/zig-server-renderer-20261022/README.md).

Item75b primitive JSON activation starts17:27 UTC (cap18:27). Keep the existing
grammar/cursor/output contracts and reuse the unchanged broad server suite;
the nine primitive actions include the single shared minifier. Reply and each
protocol controller remain subsequent bounded items.

Item75b completes17:29 UTC. All five C/native/Debug/fallback server fixture
selections pass the existing accepted-grammar/cursor/ownership/nesting suite;
no compiler/runtime failures or new test cases. The shared minifier is native.
[Report](../experiments/zig-server-json-20261022/README.md).

Item75c generated-reply parsing/recovery activation starts17:30 UTC (cap18:30).
Retain otherwise-native C parser controls, then select native syntax/reasoning/
tool-wrapper recovery with actual shared protocol constants and records. Reuse
the unchanged full server fixture; OpenAI and Anthropic output remain separate.

Item75c completes17:34UTC. All five final C/native/Debug/fallback fixtures pass.
Review fixed the allocated-empty ownership rule in reply calls/repair; the
existing repair case now reuses both allocations. Initial fixture-only compile
failure and stale run are retained and excluded; final controls pass.
[Report](../experiments/zig-server-reply-20261022/README.md).

Item75d OpenAI output activation starts17:34UTC (cap18:34). Move actual stream
records once, preserve C controls, then select complete native response/SSE/live
projection. Reuse unchanged full socket/parser suite before Anthropic.

Item75d completes17:36UTC. All five C/native/Debug/fallback server fixtures pass
cleanly, with unchanged socket/live-ID/UTF8/reasoning assertions. No new test or
model/performance claim. [Report](../experiments/zig-server-openai-20261022/README.md).

Item75e Anthropic output starts17:36UTC (cap18:36). Compose its distinct block/
signature/tool lifecycle over the same records and byte helpers. Add one actual
V4.1 split-UTF8 socket case, qualify C-first then native and fallback.

Item75e completes17:37UTC. All five C/native/Debug/fallback server fixtures pass,
including the new actual V4.1/split-UTF8 socket case. Native protocol composition
retains one shared actual-record binding. Aggregate HTTP/restart check follows
before Metal model views. [Report](../experiments/zig-server-anthropic-20261022/README.md).

Aggregate server screen completes17:40UTC at9d0d96c0. Retained C and native each
pass22 checks, five Responses fact answers and four OpenAI/Anthropic generated
wire answers. Durable restart, actual cancellation and survivor all pass; sources
unchanged, normal pressure/fixed swap. Ordered functional pair, no speed claim.
[Report](../experiments/zig-server-protocols-http-20261022/README.md).

Item76 ModelViews activation starts17:40UTC (cap18:40). Move actual ARC registry
once, qualify original C policy, then native range/coverage/compaction with the
existing4100-replacement SSD fixture, actual auxiliary GPU reads and numerics.

Item76 completes17:43UTC. All twelve C/native/Debug/fallback GPU fixtures pass,
including4100 target replacements+actual retained auxiliary reads. Real-model
C/native short generations return identical MAPLE bytes, source unchanged.
Normal pressure/fixed swap; ordered functional pair, no timing claim.
[Report](../experiments/zig-metal-model-views-20261022/README.md).

Item77 Memory admission/growth starts17:43UTC (cap18:43). Move eleven existing
Engine counters once, preserve C record-only control, then compose native Memory
into Session directly. Exercise simulated budgets/fit/reject/accounting plus
actual demand workspace growth and scalar Session persistence/continuation.

Item77 completes17:49UTC. Four selections pass model-free state and all twelve
admission/accounting/scalar Session cases. Real8192-prefix C/native pair grows
2048→4096 and has identical129280-logit prefill frontiers. Source unchanged,
normal pressure/fixed swap; functional pair only. Native Session now composes
Memory directly. [Report](../experiments/zig-engine-memory-20261022/README.md).

Item78 ExactCache starts17:49UTC (cap18:49). Replace formatted cache keys with
native values; preserve retained ARC ownership and command-boundary eviction.
Use real cache identity/owned/transient GPU offset-read cases with small-bound
and unlimited budgets, then existing broad GPU/numeric and actual model paths.

Item78 completes17:54UTC. All24 final C/native/Debug/fallback GPU cases pass,
covering true ordinary/unretained commands, forced/unlimited eviction and real
owned/transient/cache offset reads. First misconfigured C trial retained/excluded.
Real-model MAPLE bytes match, source frozen, normal pressure/fixed swap. No speed
claim. [Report](../experiments/zig-metal-exact-cache-20261022/README.md).

79. Session canonical/snapshot/private-replay/staged checkpoint transactions.
    Budget one hour. Reuse actual scalar Session/frontiers and the existing
    F32→SSD-close/reopen→fresh-F16 DRP1 probe, including fresh next-token logits.
80. Session create/free resource lifecycle over actual Device/Memory/Frontier.
    Budget one hour. Existing accounting/create-recreate and real Session/DSpark.
81. Session sampling/overthink/public logits over actual policy record. One hour.
82. Engine startup cache sizing/reserve (three ordered phases). One hour.
83. Target residency/static map transaction and owned MapSelection. One hour.
Then integrate queued Agent tools/edit/saved-session/history/checkpoint/executor/
context owners, and Server Conversation/materialization/tracker/reuse/persistence/
generation owners. One hour per coherent owner; final aggregate workflows/build
and ordinary timings after a source freeze. Remaining Metal index selection and
attention/dispatch owners have their own source/qualification boundaries.

Item79 starts17:54UTC (cap18:54). Preserve current native-engine C checkpoint
control, append pure transactions to the one graph facade, then qualify real
payload/frontier/private persistence with no mirrored records or format change.

Item79 completes18:05UTC. All four selections pass model-free state/server and
sixteen GPU/real-model cases: canonical F32/F16 plus8193-token DRP1 SSD→F16.
Fresh logits and private bytes roundtrip exactly; normal pressure/fixed swap,
source frozen. C-first exposed/fixed only the old fixture's F32-write assumption
for F16 backing; original assertions retained, failure artifacts recorded.
[Report](../experiments/zig-session-checkpoint-20261022/README.md).

Item80 SessionResources starts18:05UTC (cap19:05). Move actual Device/TP records
once, qualify C control, then native scoped construction/publication/free and
actual accounting/recreation/DSpark, reusing current scalar/typed fixtures.

Item80 completes18:15UTC. Four builds/model-free Session suites and thirteen
GPU/model cases pass. Actual trained DSpark C/native decision traces are byte-
identical; native allocation accounting/free/recreate and scalar continuation
match. Source unchanged, normal pressure; no timing or two-host TP claim.
[Report](../experiments/zig-session-resources-20261022/README.md).

Item81 SessionSampling starts18:15UTC (cap19:15). Move the actual overthink
policy once, run its meaningful public Session fixture C-first, then select
native selection/raw diagnostics and compare ordinary/active-penalty runtime.

Item81 completes18:18UTC. Four builds pass actual Session/numeric policy suites;
six real-model arms preserve seeded outputs and raw logprob JSON for ordinary
and active0.75-penalty settings. C include rebase failure retained/corrected;
normal pressure, unchanged source, no speed/full-KL claim.
[Report](../experiments/zig-session-sampling-20261022/README.md).

Item82 StartupMemory starts18:18UTC (cap19:18). Select the three ordered cache
sizing phases over actual Memory; compare auto, byte and expert-count startup.

Item82 completes18:21UTC. C/native auto/32GB/512-slot budgets and all129280
frontier logits match in six real-model arms; Debug auto matches too. Existing
SSD and extended memory-plan cases pass. Compile-width correction retained;
source unchanged and pressure normal.
[Report](../experiments/zig-engine-startup-memory-20261022/README.md).

Item83 TargetResidency starts18:21UTC (one-hour cap). Move mapping selection
once, select complete cache/static-lock/model-view transaction, compare F32/F16
and trained DSpark support-map ordering before Core integration.

Item83 completes18:25UTC. Ordinary F32/automatic and F16/32GB map/frontier
pairs match; Debug F16 and trained DSpark causal/accepted/RNG/cleanup pass.
Native compiles first attempt; corrected reused driver failure retained.
Source unchanged, normal pressure; no speed/two-host TP claim.
[Report](../experiments/zig-engine-target-residency-20261022/README.md).

84. Loaded V4.1 Metal Engine Core, actual fields once, complete post-family
    startup and retirement. Starts18:25UTC; one-hour cap. C record-only control
    first, then actual ordinary/F16/DSpark/vision, inspect/failure and cleanup.
85. Strict support tensor/metadata binding. One hour.
86. Strict DeepSeek vision-sidecar binding. One hour.
Then queued Agent/Server owners and remaining Metal dense/index/attention/MoE
controllers, with the same bounded integrate/measure/decide loop.

Item84 completes18:37UTC. Four selections pass broad CPU suites; eight actual
Core frontier arms match across ordinary/F16/penalty/paired-sidecar startup.
Inspect and trained DSpark decisions match; Debug/full fallback F16 match.
Actual failed startup→same-process reopen/accounting passes all four selections.
Only target-name/parameter-shadow compile failures, retained; no runtime fix.
Source unchanged, normal pressure; no speed or image-encoding quality claim.
[Report](../experiments/zig-engine-core-20261022/README.md).

Item85 strict SupportWeights starts18:37UTC (one-hour cap). Reuse qualified Core
C-binder binaries as fixed control, select native metadata/layout binding, run
existing metadata and actual trained DSpark/inspect paths before VisionWeights.

Item85 completes18:50UTC. Trained support inspect and complete verifier decision
traces match fixed C baseline; native ReleaseFast/Debug and C fallback CPU suites
pass. Shared-predicate guard compile miss corrected/retained. Isolated inspect
delay did not repeat in C/N/N/C. Source unchanged, normal pressure, no speed claim.
[Report](../experiments/zig-engine-support-weights-20261022/README.md).

Item86 strict VisionWeights starts18:50UTC (one-hour cap). Compare actual paired
sidecar full embeddings C/native/Debug and ordinary image/read follow-up, then
select the shared cache-runtime before queued Agent/Server integrations.

Item86 completes18:56UTC. C/native/Debug complete921600-value image embeddings
are byte-identical and internally repeatable; actual CLI image+follow-up both
answer RED. Native CPU suites pass, source unchanged, pressure normal. Header
context correction retained; no numerical or latency claim.
[Report](../experiments/zig-engine-vision-weights-20261022/README.md).

Item87 shared cache-runtime starts18:56UTC (one-hour cap). Select the common
Agent/server profile/open transaction and actual generated Blob/Stats records;
qualify current StateCache and frontend fixtures, then real cross-restart reuse.

Item87 completes19:00UTC. Native ReleaseFast/Debug cache/server/Agent suites pass;
C→native real SSD restart passes all 22HTTP/five Responses answer checks and
actual disk hits. Public Blob/Stats now derive from their real header. Baseline
fixture argument mistakes retained/corrected; normal pressure, source unchanged.
[Report](../experiments/zig-cache-runtime-20261022/README.md).

Item88 Agent ToolActions starts19:00UTC (one-hour cap). Move actual Pager and
Observation records once; qualify selected streamed read/search/owned image
observations, first with the existing CPU/terminal fixture and then real tools.

Item88 completes19:06UTC. Four selections pass production Agent/terminal/PTY
with14 stable artifacts equal; N/D suffix fixtures pass. Actual C/N/D image
Observation cloning/retirement/multimodal continuation passes with identical
geometry/next tokens. New fixture natural-versus-expanded assumption corrected
C-first; no production numeric fix. Normal pressure/source unchanged.
[Report](../experiments/zig-agent-tool-actions-20261022/README.md).

Item89 Agent Edit starts19:06UTC (one-hour cap). Qualify actual write/preflight/
atomic edit/line effects and upto forcing C-first, then select complete native
actions over the actual shared Forcer and ToolActions Buffer.

Item89 completes19:08UTC. C/native/Debug/fallback actual write/edit/preflight/
upto+terminal suites pass;14 stable artifacts match. Native compiles first
attempt, no runtime correction; whole-task model gate remains aggregate.
[Report](../experiments/zig-agent-edit-20261022/README.md).

Item90 Agent SavedSessions starts19:08UTC (one-hour cap). Move actual identity/
metadata once and select complete catalog/codec/atomic strip after C-first
real-file fixtures; save/reload/strip model workflow follows live Checkpoint.

Item90 completes19:10UTC. Four selections pass real-file catalog/codec and
production terminal suites with14 stable artifacts equal. Native compiles first
attempt; fixture invocation rebase retained. Actual KV/strip model gate pending
live Checkpoint integration.
[Report](../experiments/zig-agent-sessions-20261022/README.md).

Item91 Agent History starts19:10UTC (one-hour cap). Select complete saved-text
replay using existing pure Stream/Presentation; compare actual output queue,
summary/tool-only semantics and bounded UTF8 tails C-first.

Item91 completes19:11UTC. Four selections pass actual saved-history/terminal
suites;17 stable artifacts match including three real output-queue cases.
Native compiles first attempt; no runtime correction or task-quality claim.
[Report](../experiments/zig-agent-history-20261022/README.md).

Item92 Agent Checkpoint starts19:11UTC (one-hour cap). Select live saved-file
transactions over actual Session; C-first full-context rebuild plus a real
nonzero-KV save/reload/strip/reload/history cycle under the pressure guard.

Item92 completes19:17UTC. Four CPU/terminal selections pass; C/native/Debug
full-context recovery plus actual1.4MB KV→switch→strip→switch/history cycle
passes with identical tokens/next token/stdout. Existing TP predicate test
reconnected to native policy; no runtime fix. Normal pressure/source unchanged.
[Report](../experiments/zig-agent-checkpoint-20261022/README.md).

Item93 Agent Executor starts19:17UTC (one-hour cap). Move actual ToolRuntime
once; qualify complete dispatch/job lifetime/terminal projection C-first then
native, retaining real Bash process/monitor ownership in its existing library.

Item93 completes19:21UTC. Four selections pass real dispatch/process/job/
terminal suites;17 stable captures match. Missing actual linenoise include
corrected/recorded; no runtime fix. Model/live-context gate remains next.
[Report](../experiments/zig-agent-executor-20261022/README.md).

Item94 Agent ContextRuntime starts19:21UTC (one-hour cap). Move actual Worker
record once, select complete live-context/frontier/SSD/suffix transactions,
then reuse actual vocabulary/image and six-turn persistent-agent workflows.

Item94 completes19:42UTC. Four CPU/terminal selections match17 stable captures; real vocabulary/image checks pass. Fixed C/native six-turn tasks pass identical answers/stdout and three identical SSD pre-RoPE graft geometries, including reuse after approximation. One ordered timing pair is not a latency claim. Source unchanged and normal pressure.
[Report](../experiments/zig-agent-context-runtime-20261022/README.md).

Item95 Agent Result starts19:42UTC (one-hour cap). Select complete tool-result import/fit/commit transactions and qualify C-first; then integrate communications and the turn owner before the next real-model aggregate.

Item95 completes19:43UTC. Selected native ReleaseFast/Debug pass production terminal suites and17 stable captures match fixed C baseline. Build-selection and actual native commit fixture bridge corrections retained; model transaction gate pending Turn aggregate.
[Report](../experiments/zig-agent-result-20261022/README.md).

Item96 Worker Communications starts19:43UTC (one-hour cap). Exercise real condition/pipe/output-ownership handoffs C-first, then select the native transaction.

Item96 completes19:45UTC. C-first/native ReleaseFast/Debug real thread/pipe/output handoffs and production terminal suites pass;17 stable captures match. Varargs pointer typing compile corrections retained; model gate remains Turn aggregate.
[Report](../experiments/zig-agent-communications-20261022/README.md).

Item97 Agent Turn starts19:45UTC (one-hour cap). Select complete raw/chat/tool generation and direct progress-aware sync; compare CPU/terminal artifacts before real model aggregate.

Item97 completes19:46UTC. Native ReleaseFast/Debug production fixtures match17 captures. Existing pointer-identity/byte tests caught incorrect address-of imported extern literals; fixed direct symbol pointers and retained failures. Actual generation gate is the final Agent aggregate.
[Report](../experiments/zig-agent-turn-20261022/README.md).

Item98 Agent System/Summary starts19:46UTC (one-hour cap). Select complete system construction/private summary inference and exact transcript/image rollback; retire the final temporary summary services.

Item98 completes19:47UTC. Native ReleaseFast/Debug production fixtures match17 captures; first semantic compile passes after applying the known literal pointer correction. Private model summary is an explicit final Agent aggregate gate.
[Report](../experiments/zig-agent-summary-20261022/README.md).

Item99 Worker Sessions/Lifecycle starts19:47UTC (one-hour cap). Select real saved-session orchestration, shared cache startup and complete worker command-thread lifetime; reuse existing fixtures then actual saved-KV/compaction/task gates.

Item99 completes19:48UTC. Native ReleaseFast/Debug production suites and17 stable captures match C on first compile. Saved-KV/thread model qualification remains final Agent aggregate.
[Report](../experiments/zig-agent-lifecycle-20261022/README.md).

Item100 Agent Terminal starts19:48UTC (one-hour cap). Move actual editor/queue records once, select live terminal/queue controllers and compare real PTY behavior.

Item100 completes19:50UTC. Native ReleaseFast/Debug real terminal/PTY and17 stable captures match C. Explicit SDK/type/discard compile corrections retained; no runtime fix. Interactive model frontend gate remains next.
[Report](../experiments/zig-agent-terminal-20261022/README.md).

Item101 Agent Frontend starts19:50UTC (one-hour cap). Select complete options/bootstrap/headless/interactive/commands, retaining sole signal/backend-fact leaves; then run actual six-turn task, image/save/reload and interactive compact aggregate.

Item101 completes20:07UTC. Release/Debug/fallback fixtures and17 stable captures match C; actual six-turn SSD reuse, images, thinking, raw generation and nonzero saved-KV cycle pass. Interactive Debug exposed default-thread-stack exhaustion; explicit8MiB fixes the same complete compaction/follow-up task. Failed process report retained; normal pressure/source accounting.
[Report](../experiments/zig-agent-frontend-20261022/README.md).

Item102 Server Conversations starts20:07UTC (one-hour cap). Apply reviewed declaration/fallback guards, record C-controller baseline with added actual storage/image fixtures, then enable native owners one at a time.

Item102 completes20:11UTC. C-controller/fallback baseline plus native Conversation release/Debug pass actual server/storage fixtures. First native compile passes, no runtime fix. Later controllers remain C behind installed inverse guards until their individual selections.
[Report](../experiments/zig-server-conversations-20261022/README.md).

Item103 materialize: sequential native release/Debug server suite passes. One-hour cap, actual model gate remains combined serving HTTP. [Report](../experiments/zig-server-materialize-20261022/README.md).

Item104 decode: sequential native release/Debug server suite passes. One-hour cap, actual model gate remains combined serving HTTP. [Report](../experiments/zig-server-decode-20261022/README.md).

Item105 reuse: sequential native release/Debug server suite passes. One-hour cap, actual model gate remains combined serving HTTP. [Report](../experiments/zig-server-reuse-20261022/README.md).

Item106 persist: sequential native release/Debug server suite passes. One-hour cap, actual model gate remains combined serving HTTP. [Report](../experiments/zig-server-persist-20261022/README.md).

Item107 generation: sequential native release/Debug server suite passes. One-hour cap, actual model gate remains combined serving HTTP. [Report](../experiments/zig-server-generation-20261022/README.md).

Item108 serving: sequential native release/Debug server suite passes. One-hour cap, actual model gate remains combined serving HTTP. [Report](../experiments/zig-server-serving-20261022/README.md).

Item109 boot: sequential native release/Debug server suite passes. One-hour cap, actual model gate remains combined serving HTTP. [Report](../experiments/zig-server-boot-20261022/README.md).

Item110 images: sequential native release/Debug server suite passes. One-hour cap, actual model gate remains combined serving HTTP. [Report](../experiments/zig-server-images-20261022/README.md).

Items102–110 combined actual serving HTTP complete: release→Debug restart, concurrent decode, cancellation and SSD/durable response recovery pass. [Report](../experiments/zig-server-full-20261022/README.md).

Item111 Session controllers starts20:22UTC; completes20:33UTC within one hour. Release/Debug compile and CPU suites pass; trained C/native/Debug16cycle records exact; public image and concurrent/durable HTTP pass. [Report](../experiments/zig-session-controllers-20261022/README.md).

Item112 Metal controller integration starts20:34UTC. C-after-record-move baseline first, then select Index/Dense/Attention/Routed/Cache/Load/Output/RoPE/Compressor/Scheduling/Embedding/Quant sequentially; one-hour cap per owner, existing real GPU fixtures and final whole-model aggregate.

Item112 completes20:49UTC. All15 native families compile and pass focused GPU fixtures; one packed-gather launch-width runtime bug fixed against the unchanged byte guard. Final combined Debug/model/performance pending. [Report](../experiments/zig-metal-controllers-20261022/README.md).

Item113 Eval starts20:54UTC; completes21:15UTC within one hour. Complete native grading/trace/options/runtime/UI selected; C/native/Debug20CPU cases exact, actual model byte blocks and15 trace regrades exact, seven real PTY finish/pause/navigation/interrupt cases pass. Truncated Law samples remain INCOMPLETE, so no benchmark-grade or speed claim. [Report](../experiments/zig-eval-frontend-20261022/README.md).

Item114 remaining Metal controllers starts21:15UTC (one-hour cap per owner). Original-C controller baseline passes after actual record moves, including new existing-fixture parallel-FFN/deferred-KV invariants. Select HC, shared/router, primitives, supported carry/BF16 and TP/prefetch sequentially, then aggregate.

Item114 completes21:17UTC. Six remaining controller groups compile and focused GPU fixtures pass; TP needs one opaque-pointer type correction. C/native actual parallel-FFN and deferred-KV invariants pass. Full combined qualification follows. [Report](../experiments/zig-metal-final-20261022/README.md).

Item115 Web frontend starts21:17UTC; completes21:20UTC within one hour. Complete native Web/HTTP/WebSocket controller selected; same existing local recovery fixture passes C/native/Debug. Initial compiler diagnostics retained, actual browser/website qualification not implied. [Report](../experiments/zig-web-20261022/README.md).

Items116–118 close ModelLifecycle, RouteDiagnostics and Bootstrap. Source drafts prepared before the pause; integration resumes22:21UTC. Four configuration builds and98CPU/host fixtures pass,45 GPU fixture calls pass, and six diagnostic record/replay/merge calls retain identical bytes. No runtime math fix. Complete model/performance campaign is next. [Report](../experiments/zig-metal-lifecycle-20261022/README.md).

Item119 final matched qualification completes:98 host/CPU,45 GPU and six diagnostic invocations pass; full ordinary frontier vectors exact, both six-turn Agent tasks pass, durable C→native HTTP restart and all 16 trained cycles match, both six-case vision suites pass. Runtime is near neutral on these points; full rebuild 29.5%slower but F32 edit 67.1%faster. Include-path-only Make fix retained with first failure. [Report](../experiments/zig-qualified-20261022/README.md).
