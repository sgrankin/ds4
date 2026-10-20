# Fast approximation experiments (2026-10-02)

The baseline is local DeepSeek V4.1 Flash Q2 on M5 Max/128 GiB with SSD
streaming. All experimental runtime changes remain opt-in. Start with a fixed
baseline, change one thing, and measure practical speed and quality tradeoffs.
Output degradation is acceptable; delta-NLL and KL describe it rather than
automatically rejecting a technique. Expand the checks only when a result or
failure warrants it. The baseline is the local Q2 model. See the
[project guidance](../../AGENTS.md) for the current engineering approach.
The [next experiments](next-experiments.md) put the work in order, with one-hour
budgets, practical qualifications and initial tradeoff targets.

## Paired quality probe

`quality_probe.c` teacher-forces a frozen official continuation. It stores the
baseline's full raw logits once, then subsequent builds read those records.
Candidate inference is performed once per position; calculating distribution
metrics does not require another model forward pass. Token IDs and a hash of
the complete prompt/truncated continuation must match the reference exactly.

- NLL is `-log p(reference_token)` in nats, using unfiltered temperature-one
  softmax. Delta-NLL is candidate minus baseline: positive is worse reference
  likelihood, negative is better. `exp(delta-NLL)` is the perplexity ratio.
- KL is `D_KL(p_baseline || p_candidate)`, over the complete vocabulary under
  the identical teacher-forced prefix. It detects distribution drift, not
  whether that drift improves task quality. API top-20 alternatives are not
  sufficient to compute this full KL.
- Total variation, argmax changes, per-prompt deltas and tail KL accompany the
  means. A raw byte-equality counter separately verifies disabled-mode parity;
  KL alone cannot distinguish a constant shift in all logits.
- Optional sampling-policy logits are measured separately from raw model
  logits. Compile with `-DDS4_IDEAS_POLICY_HOOK` against the penalty runtime.
  These are pre-temperature/pre-filter policy metrics, not the exact sampled
  top-p/min-p distribution.
- Each row scores before feeding its target token; `decode_ms` measures that
  token's evaluation, which produces the following row's distribution. The
  first scored row comes from exact prefill for decode-only modifications.

`screen.tsv` and `heldout.tsv` are separate systematic prompt subsets of the
tracked official V4.1 router fixtures. They contain non-thinking continuations;
they can screen expert substitutions but cannot establish the quality of a
thinking-only logit penalty. Reasoning interventions require free generation
with independent final-answer checks. Reference likelihood and low KL alone
do not establish coding/tool-task success or reasoning improvement.

Build against a completed runtime (no compilation during final timing):

```sh
sh speed-bench/ideas/build_probe.sh . speed-bench/ideas/artifacts/quality-probe-base
sh speed-bench/ideas/build_probe.sh .workspaces/idea-resident speed-bench/ideas/artifacts/quality-probe-resident
speed-bench/ideas/artifacts/quality-probe-base --self-test
python3 speed-bench/ideas/run_screen.py speed-bench/ideas/artifacts/resident-screen \
  --baseline speed-bench/ideas/artifacts/quality-probe-base \
  --candidate speed-bench/ideas/artifacts/quality-probe-resident \
  --manifest speed-bench/ideas/screen.tsv --cases 16 --tokens 32
```

The runner freezes binaries and external Metal sources, records hashes/settings,
checks native disabled-mode and repeat baseline byte parity, and keeps every
per-position score. Large reference binary files remain outside version control.
Its timings exclude metric arithmetic/reference I/O but still come from an
instrumented quality run; use them to select candidates, not to claim production
speedup. Validate latency using fixed replay without metric copies and actual
generated tasks. Prompt-bootstrap intervals in the screening summary are
descriptive and do not replace independent task coverage.

## Experiment tracks

1. **Resident expert rerouting:** preserve the five largest original mixture
   contributions; consider a cached replacement for only the weakest miss.
   Closeness in router score is an engineering heuristic, not proof of expert
   function similarity. Later guards bound unbiased affinity on both sides and
   optionally cap replaced mixture mass. Prefill remains exact. The route is
   frozen before its kernel runs. A separate, bounded background-read prototype
   is being tested; the first version stages originals for later native requests.
2. **Overthinking penalty:** subtract a fixed positive raw-logit bias from exact
   tokenizer-mapped leading-space hesitation markers. Think-only gating is our
   adaptation; the paper applied its list at every output position. No extra
   forward pass. Compare final correctness, cap failures, tokens, and elapsed
   time, including legitimate correction tasks.
3. **Exact context frontiers:** restore a saved matching prefix before an edit
   instead of resetting the entire graph. Snapshot copy/storage and matching
   costs count. Compare full logits/state as well as latency, and check cold
   versus staged numerical scheduling separately.
4. **Context Language Models:** give the agent an editable live-context artifact;
   synchronize at completed tool boundaries. This harness feature does not
   require approximate suffix reuse or retraining.

Approximate SCR additionally preserves stale suffix state after edits.
V4.1's raw windows, compressed/index caches, unfinished pair pooling and Engram
history require coordinated handling. The current work does not assume that
ordinary KV relocation implements that correctly.

## Recorded outcomes

The aligned middle-edit benchmark rebuilt a 2,816-token context after changing
256 tokens behind a saved 2,048-token prefix. Exact restoration reduced mean
edit latency from 59.026 s to 24.362 s (**58.73%, 2.423x**). Including the one-time
12.396 ms snapshot save still gives 58.71%. All 129,280 logits, complete serialized
graph state and four continuation rows matched both the staged reference and
the cold replay in this aligned case. This is one same-process ABBA workload,
not a general agent throughput claim. See
[`context-frontier-20261002`](../experiments/context-frontier-20261002/README.md).

The actual CLM agent test passed: one tool transaction removed 1,018 tokens
(3,478 to 2,460), retained the required fact, and answered `CODE=2718` correctly.
The interface is `--live-context-file PATH`; use
`DS4_AGENT_COMPACT_PREFIX_CACHE=1` to enable validated bootstrap restoration on
edits/compaction. See the retained
[`CLM test`](../experiments/clm-live-context-20261002/README.md).

`DS4_AGENT_CONTEXT_FRONTIERS=2` additionally retains recent complete exact
prefix checkpoints (up to four, with a 128 MiB default total budget). The native
regression passed: longest-match restore at 2,304 tokens, complete graph/logit
byte parity after a changed 512-token suffix, count/byte eviction and safe
corrupt-payload fallback. Literal chat/DSML quotations and Unicode survive
unchanged spans; ambiguous separated edits are rejected transactionally. See
[`recent frontiers`](../experiments/context-frontiers-20261002/README.md).
This reuses the matching prefix and recomputes the suffix. An isolated native
prototype now also grafts an unchanged suffix from an exact old context,
including raw windows, compressed/index owner rows and unfinished pair pooling.
It recomputes the edited middle and at least one fresh final token. Its first
edited-fact test changed `CODE=2718` to `CODE=3141`: both fresh and reuse greedy
answers were `3141`. Rebuilding the edited 256-token middle and 512-token suffix
took 40.454 s; reuse took 17.551 s (56.6% less, 2.30x). Mean full-vocabulary KL
over the three gold-answer rows was 0.0000133; summed gold-answer delta-NLL was
+0.0000882 nats. Identity grafts matched complete graph state and raw logits
byte-for-byte. This is one same-position fixture and one timing comparison,
not agent throughput or arbitrary-compaction validation. Capture accepts exact
sources only, and approximate sessions cannot export canonical checkpoints;
there is no accumulated approximation across edits. Compact evidence is in
[`suffix-semantic-code-c512-t1`](evidence/suffix-semantic-code-c512-t1/summary.json).
The longer 1,024-token unchanged suffix also returned `3141` in both modes:
75.507 s fresh versus 16.841 s reuse (4.48x), mean KL .00000219. A separate
128-token deletion moved a 63-token reused suffix left, repaired raw,
compressed and index RoPE positions, and preserved all 64 surviving fresh
prefix window rows in each of 40 layers byte-for-byte. Both answers were
`3141`; edited-context time was 4.626 s versus 3.105 s (1.49x), KL .0000113.
Identity, invalid-edit, pre-copy cancellation and post-copy invalidation checks
passed. This frozen implementation admits even starts and shifts divisible by
128; the later even66 physical-wrap case below is also validated. Other even
displacements remain prepared, with CPU layout tests only.

The deleted-secret fixture returned `REDACTED` in both modes, with KL .0000829
and 2.34x diagnostic edit speedup. However, the complete old-secret answer's
probability increased from 2.57e-13 to 5.08e-5, including EOS. Correct greedy
output and low forward KL therefore do not establish that removed information
was forgotten. Fresh-tail washout and the calibrated fully fresh Record X query below further
measure this limitation. Every completed probe starts from an exact
source; the later agent integration below rebuilds exact state before another
approximate edit. See the source/binary-separated
[`four-case suffix evidence`](evidence/suffix-semantic-and-relocation-20261002.json).

The original redaction fixture's full prompts, target tokens and fresh raw
logits were byte-identical when varying fresh tail1/64/128. Tail64 reduced the
old-answer probability from5.08e-5 to4.97e-12; tail128 gave9.07e-12. Both
returned `REDACTED`, with mean KL6.28e-7/2.07e-6 and diagnostic speedups2.13x/
1.96x. Recovery was strong versus tail1, but was not monotonic64-to128 and
cannot guarantee removal across prompts; the different calibrated lookup below
still exposes stale influence with128fresh tokens. Full evidence is retained
beside that lookup, including the rejected prior-query calibration.

A calibrated Record X lookup then answered `6829` in the original exact context
and `UNKNOWN` in both fresh and reused edited contexts. Its 36-token query was
wholly fresh, with 128 fresh tail tokens. Recompute time was 40.606 versus
21.199 s (1.916x); mean KL .00339346 and gold sequence delta-NLL +.0118461.
The removed code probability rose from 1.54e-15 to 1.89e-6, despite the correct
greedy answer. This is stale conditioning, not forgetting. The preceding
BEFORE REDACTION fixture had an invalid original retrieval control and is
excluded from semantic conclusions. An even66-token left relocation also
crossed both physical ring wraps: 27 fresh-prefix raw rows per layer and
1,210 compressed/index rows survived exactly; both answers were `3141`, with
mean KL4.259e-6 and diagnostic recompute10.188 versus6.979 s (1.460x). See
[`calibrated lookup and wrap evidence`](evidence/suffix-record-x-wrap-20261002/compact-summary.json).

Actual agent integration now passes the three-turn semantic/reuse checks.
The production agent answered `CODE=2718`, edited one earlier record, then
answered `CODE=3141`. Its trace reused 556 suffix tokens and evaluated 476
tokens fresh; the entire latest user query was fresh. All context bytes outside
the edit were preserved. The 24.878-second edit/graft interval is an unpaired
diagnostic, with no complete-turn speed claim. That runner's original return
code 3 and `pass: false` remain retained: it expected a named save that closing
headless stdin never requested. See the precise
[`production-agent result`](evidence/agent-suffix-native-three-turn-20261002/NATIVE_THREE_TURN_RESULTS.md).

A separate native fixture called the actual idle-worker save function at the
completed-third-turn boundary. It saved 3,142 transcript tokens with zero KV
payload in .609 ms. Checkpoint tokens, full raw logits and the private graph
stayed byte-identical; the session stayed approximate, canonical export remained
blocked, and saving did not replay the context. This validates that function
through an inspectable private frontend; actual reopen remains unrun. See
[`native save result`](evidence/agent-suffix-native-persistence-20261002/NATIVE_PRIMARY_RESULTS.md).

The agent opt-in is `DS4_AGENT_CONTEXT_SUFFIX_REUSE=1`, with a default 128-token
fresh tail (`DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL`, minimum 64). It admits an
unchanged suffix only before the latest recognized user role marker; that
marker and all following tokens are freshly evaluated. The configured tail
is a lower bound. Missing or unsupported role markers take exact fallback
before staging or graph mutation. This policy passed model-free transaction
and helper ASan/UBSan checks, plus the focused MAIN agent build/tests. The
old actual-agent fixture already placed its reused endpoint before the latest
user marker; it is not a new-source execution test. See the
[`fresh-query CPU evidence`](evidence/suffix-protected-user-cpu-20261003-v1/README.md).

The cache otherwise admits an
unchanged suffix from an exact frontier after an eligible same-position edit
or even left relocation. Its trace reports whether the latest user query is
wholly fresh; the three-turn fixture verifies that condition. Unsupported edits
take the exact path. Another edit after approximation rebuilds exact state;
approximate state is never retained as an exact frontier or serialized as
reusable KV. Source ownership, rollback, eviction, cancellation and text-only
save/reopen have additional model-free ASan/UBSan coverage.

A separate opt-in pair-tape prototype now repairs odd left relocations by
rebuilding the ratio-two compressed pairs from bounded pre-pool inputs. The
first native pilot deleted 65 tokens, reused 512, and evaluated a fresh tail of
128. All five native controls matched complete source/fresh states, 18 reference
vectors and teacher tokens exactly. The repaired candidate had mean KL
.0166751, summed teacher delta-NLL -.6728714 and no argmax changes across 17
positions. App-owned extra staging peaked at 113,379,963 bytes under its
128 MiB budget. Fresh replay took 14.5666s and repair 2.8785s in this diagnostic
run; those timings are not a balanced or end-to-end speed claim. The retained
inputs carry old conditioning, so this does not demonstrate forgetting or
general task accuracy. The experimental runtime remains outside MAIN while
short-tail, opposite-parity and wraparound cases are checked. See the
[`odd65 native pilot`](evidence/suffix-pair-tape-odd65-native-20261003-v3/README.md)
for independently recomputed quality, native controls and preserved preparation
failures.

The subsequent zero-edit odd-boundary control reused 65 tokens before a
one-token fresh tail with live raw-ring wraparound. All five native states,
reference vectors and teacher IDs matched exactly. All 17 candidate vectors
were byte-identical, and the diagnostic state body matched canonical fresh
state after validating and normalizing only its four-byte format tag.
Independent FP64 KL, TV and delta-NLL are zero; the helper's retained
2.927e-26 KL is numerical roundoff. Actual extra owned staging peaked at
117,722,879 bytes. This proves identity in that geometry, not equivalence after
an edit. See the
[`zero-edit identity control`](evidence/suffix-pair-tape-zero-identity-native-20261003-v1/README.md).

The opposite-parity short-tail check deleted 65 tokens, reused 65 from an odd
source start into an even destination, and consumed the reconstructed pending
pair with the next fresh token. All five native controls were exact; repair
used 195 inputs and rebuilt 96 rows. Independent mean KL was .1165639, TV
.0682032, summed teacher delta-NLL -1.372691, with one argmax change. The
one-token tail exercises mechanics and does not recommend changing the agent's
minimum 64-token fresh tail. Staging peaked at 106,814,075 bytes and all pressure
samples were normal with zero swap growth. See the
[`inverse-parity pending-pair check`](evidence/suffix-pair-tape-inverse-native-20261003-v1/README.md).

The odd-deletion wrap check used 65-token prefill chunks and crossed both the
2,048-token tape position and raw-ring boundaries. All five native controls
matched reference logits, source/fresh states and teacher tokens exactly.
Repair reused 387 inputs and rebuilt 195 pair rows; mean KL was .05739446,
TV .07719907 and summed teacher delta-NLL +.4222063, with one argmax change.
Its one-token tail tests boundary mechanics only. Unique staging peaked at
118,107,119 bytes, below 128 MiB; all 1,398 pressure samples were normal with
zero swap growth. Timings are diagnostic and retained conditioning remains
approximate. See the
[`odd-deletion wrap check`](evidence/suffix-pair-tape-wrap-native-20261003-v1/README.md).

The integrated native controls passed at cache64: default router-off and
positive-gap/native-mode each matched the original seven phase-end full-vocabulary
vectors and final canonical graph byte-for-byte. All210 pressure samples were
normal, with no swap growth. This is native parity, not an integrated performance
measurement. See
[`integrated native control`](evidence/integrated-native-parity-20261002/RESULTS.md).


The conservative resident candidate was chosen on the screening split and
locked before the disjoint held-out split: gap .05, maximum original mixture
mass .12, one weakest native miss, rank at most 12. Both use SSD streaming,
64 GiB requested cache, 512-row prefill and no-thinking official continuations.

| Split | Positions | Delta-NLL (nats/token) | Mean KL | Top-1 changes |
|---|---:|---:|---:|---:|
| Screen | 492 | -0.000202 | 0.004716 | 6 |
| Held out | 512 | -0.005589 | 0.005058 | 10 |

Held-out per-prompt bootstrap interval for mean delta-NLL is
[-0.015890, +0.003864]; worst prompt delta is +0.031736, KL p99 .066368 and
maximum .166436. This small screen gives no clear average reference-likelihood
regression or improvement. It does not establish generated task accuracy.
All held-out raw rows in disabled mode and repeated native controls were
byte-identical to the original-source baseline. The baseline probe uses the
original `67696070` core object and the integrated Metal backend; the added
cache-inspection APIs do not alter the native path. Evidence is retained in
`evidence/` and persistent `artifacts/`.

The separate seven-phase, 343-decode-token ABBA replay did **not** establish
a speedup: candidate turn time was 45.079 s versus native 41.930 s (+7.51%),
decode +9.78% and append prefill +5.68%. The exact startup prefill also differed
by +9.15%, and concurrent background PDF indexing was observed, so the causal
size of the routing overhead is uncertain. Both native repetitions had identical
full logits and graph state; both candidate repetitions also matched each other.
Each candidate skipped 381 requested expert loads (3.53 GiB), which is not proof
of physical SSD traffic saved. Kernel pressure stayed normal. See
`evidence/resident-abba/` and `evidence/resident-abba-pressure/`. This rejects a
production speedup claim for the present evidence; the feature remains opt-in.
The optional two-slot routing mode now also has an evolving-session quality
screen; its neutral controls and larger replacement settings are described below.
A separate drop-weak-miss prototype is retained in its experiment branch and
is not integrated into the main checkout.

The second investigation separates lookup, planning, commits and expert I/O.
Native, lookup-only and full no-op plans reproduced all seven phase logits and
the complete serialized graph byte-for-byte. Lookup/guards consumed 5.824 ms
across 343 decode tokens; complete no-op planning consumed 12.859 ms (0.072% of
native decode). The original conservative policy demanded 4,798 decode expert
loads versus native's 5,143, with 6.71% fewer explicit requested bytes. These
are phase-attribution measurements, not balanced latency evidence; requested
reads may be served by the OS page cache. Process disk-read counters include
mapped prefill and other model I/O. See
[`resident-overhead-profile-20261002`](evidence/resident-overhead-profile-20261002/compact-summary.json).

Broader replacements are locked before testing: balanced gap .15 / mass .18 /
total mass .30 / rank24 / two slots, and broader .30 / .20 / .35 / rank48 / two
slots. Their first 343-token raw-distribution screen gave mean KL .000413 and
.000254, respectively, with one argmax change each. Broader demanded 2,836
decode expert loads versus native's 5,143 (44.9% fewer); this accounts for the
different approximate routing/cache history.
The new 16-prompt split is indices1 mod6, separate from screen0 mod6 and the
first held-out3 mod6. `run_fixture_matrix.py` freezes inputs/binary/shaders and
records one reusable full-distribution reference with native controls around
all candidate policies. Both quality matrices are run through the shared guard.

That independent split is now complete. Both native controls matched all 512
raw rows exactly. Balanced, broader and drop gave delta-NLL +.014105, +.013860
and +.008617 nats/token, respectively (perplexity ratios 1.01421, 1.01396 and
1.00865). Mean KL was .010601, .012651 and .009836; argmax changes were 12, 20
and 9. Prompt-bootstrap intervals cross zero, while the worst prompt deltas
were +.102875, +.113244 and +.120490. Broader skipped 6,464 of 12,718 requested
original misses along its own approximate trajectory; this is not a comparison
against native demand I/O or physical SSD traffic. The one agent replay's small
distribution drift therefore does not generalize to all of these prompts.
All 308 guard samples had normal pressure and zero swap growth. See
[`resident-broad-independent16-20261002`](evidence/resident-broad-independent16-20261002/compact-summary.json).

The bounded original-expert backfill comparison is also complete on that same
16-case calibration set. Native before/after and positive-gap native-mode
controls matched all 512 raw rows and the reference file hash. BF0 repeated the
prior broader rows exactly. BF1 and its repeat matched each other byte-for-byte:
mean delta-NLL +.011937 versus BF0 +.013860, but KL increased from .012651 to
.013095 and worst prompt delta from +.113244 to +.138348. Of 251 staged reads,
24 were consumed, 226 expired and one was discarded at cleanup; 5,506 of 5,833
attempts were blocked by the occupied READY slot. Staging requested 2,498,273,280
bytes. Exact cache-loader demand bytes were not exported by this helper, so net
I/O savings remain unmeasured. Instrumented quality timings are diagnostic;
this result does not establish a speedup or justify promotion. All guard
samples were normal with zero swap growth. See
[`resident-backfill-calibration16-20261002`](evidence/resident-backfill-calibration16-20261002/compact-summary.json).

Exact I/O export then resolved that comparison. Native before/after both
requested/returned 225,302,446,080 bytes; BF0 175,585,812,480 (-22.07% across
prefill plus decode). BF1 and its repeat both used 175,157,821,440 demand bytes
plus 2,498,273,280 staged bytes: total 177,656,094,720, which is 2,070,282,240
bytes (+1.18%) more than BF0. All native raw/IO controls and BF0 prior raw bytes
matched; BF1 repeats matched. This is explicit logical read accounting, not
physical SSD traffic. The occupied global slot prevents useful admissions;
retention and admission need their own measured experiment. Quality timing
remains diagnostic. See
[`resident-backfill-exact-io-20261002`](evidence/resident-backfill-exact-io-20261002/compact-result.json).

A one-substitution experiment then allowed an original expert that finished
loading to replace its temporary resident substitute before routed binding.
That mechanism worked, but only 2 and 3 of about3,500 tentative substitutions
rolled back in its repetitions. The global READY slot blocked over3,200
admissions per arm. Including background reads, staging increased logical
bytes by about1.05–1.15% versus the matching unstaged arm. Approximate NLL
increased on all arms; this calibration establishes neither latency nor
generated-task improvement. Both native controls retained exact raw and demand
I/O parity. See
[`late original restoration`](evidence/resident-backfill-late-one-slot-20261002/BACKFILL_LATE_V1_RESULT.md).
The bounded per-layer pool has now completed its seven-arm quality/I/O study.
Native and unstaged controls reproduced the previous full raw distributions and
exact demand counters. The pool admitted about10.8 times as many reads as the
global slot, but consumed only9–10%. Its26.7–27.0GB of background reads made
total logical reads12.69–12.81% higher than the matching unstaged arm, despite
lower demand reads. Only4 and1 same-token restores occurred. Quality was mixed;
all ownership and I/O balances passed. Capacity and handoff contention were
not limiting, so useful admission/retention needs investigation. No default
or speed promotion follows from this result. See the
[`completed bounded-pool result`](evidence/resident-backfill-pool-completed-20261002/RESULTS.md).

The next retention-only study kept queue TTL80 and the same400MiB owned-memory
cap while extending completed READY retention to1600 layer boundaries. Native
and BF0 full distributions and exact demand I/O controls repeated. Useful
completed reads rose from248/2705 (9.17%) to418/2626 and425/2599 (15.92–16.35%).
Harvest expiry was zero; stored expiry fell2043 to14/18, while unused replacement
rose407 to2166/2130. Total logical reads fell221.859GB to219.778/219.211GB,
still11.75%/11.46% above BF0's196.667GB. Quality was mixed and the async repeats
differed. All ownership/I/O balances passed; all302 pressure samples were
normal with no swap growth. READY replacement now dominates the loss; this
remains an isolated opt-in experiment, with no speed or default promotion.
See the [`completed READY-retention result`](evidence/resident-backfill-ready-age-completed-20261002/RESULTS.md).

An owner-thread native-resident recheck now has a completed negative result.
It removed 31 and 34 redundant queued reads before worker I/O, but the two
rechecked pool attempts still requested 219.719 and 219.878 GB against BF0's
196.667 GB, about 11.7–11.8% more. Native and BF0 controls reproduced their full
512-row raw distributions and exact demand counters. About 82% of completed
pool reads were replaced before use; all ownership and I/O balances passed.
The isolated runtime remains experimental. See the
[`native-resident admission result`](../experiments/ds41-ssd/BACKFILL_POOL_NATIVE_RECHECK_RESULT.md)
for retained attempts, independent quality checks and preserved unrun sources.

A separate one/two/three-substitution calibration is complete, with all three
native controls matching512 full-vocabulary rows and the same reference hash.
Three substitutions gave delta-NLL +.005929 versus +.013860 for two, but mean
KL increased from .012651 to .016694 and maximum row KL from .61608 to1.41472.
Argmax changes were18 versus20; the three-substitution NLL bootstrap interval
crosses zero. It skipped8,299 of13,433 original misses on its own trajectory,
versus6,464 of12,718 for two. Those are different cache histories, and the frozen
helper does not export exact demand bytes. This mixed calibration alone does
not justify default promotion; subsequent checks are recorded below.
See the source-separated
[`substitution breadth calibration`](evidence/resident-three-calibration16-20261002/RESULTS.md).
The reviewed three-substitution option is merged locally, retaining the default
one-substitution and GAP0 paths. The focused production build and resident/agent
CPU tests passed, including100000 old one/two coefficient-bit comparisons and
50000 three-plan invariants. See
[`integration checks`](evidence/main-three-integration-cpu-20261002/proof.json).

The subsequent joint screen holds cache64 and selected TILE64/LIMIT1024 fixed.
Both native controls match all343 full-vocabulary vectors of the native-fast
reference and all seven exact phase I/O records. Balanced2, broader2 and
broader3 reduce total logical demand reads6.84%,8.87% and10.97%; decode-only
reductions are37.35%,45.95% and59.81%. Broader3 gives delta-NLL+.00013095,
mean KL.00029223, maximum KL.01577993 and two argmax changes. This is one
correlated teacher-forced trajectory. The different calibration fixture has
larger tails; the separate generated-code and clean-latency checks follow.
Timings here are diagnostic and no default is promoted. See the
[`joint quality and exact I/O result`](evidence/fixed64-resident-quality-three-completed-20261002/RESULTS.md).

The same broader3 policy then passed a clean profile-off ABBA against native
fast64, with all other CPU work paused. Mean turn append plus decode fell
from31.4840s to29.8724s (-5.119%); decode fell9.503%, while append prefill rose
.471%. Startup-inclusive model time fell1.655%. Both native end-vector/state
hashes match the prior fast64 control; candidate repeats match each other.
These are seven phase-end vectors and the final state, not every decode row.
All332 pressure samples were normal, with no swap growth. This is one fixed
teacher-forced workload, excluding sampling and tools; no default is promoted.
The first macOS interpreter-identity preflight failure is preserved separately.
See the [`clean three-expert timing`](evidence/resident-three-fast64-abba-completed-20261002-v2/README.md).

The separate generated-code comparison completed all four tasks and all34
unchanged gold checks in native-before, broader3 and native-after. Both native
arms reproduce historical fast64 vectors, raw code and sampling decisions.
The candidate exercised12519 substitutions, including135 triple layers, with
zero drops. Its warm and initial full-vocabulary vectors match native exactly,
but later generated decisions and code differ in three tasks; all still pass.
Those initial zero-KL measurements do not describe later generation rows.
All289 pressure samples were normal, with no swap growth. Timings are
diagnostic; broader agent workloads remain necessary before default promotion. See the
[`three-expert generated-code result`](evidence/resident-three-coding-medium-20261002-v1-native/README.md).

The previously untouched confirmation split (16 prompts, 32 teacher-forced
tokens each) now completes with the broader3 policy frozen before outcomes.
Both native controls match all512 full-vocabulary rows and demand I/O counts.
Whole-engine prefill plus decode logical expert reads fall from207.277 GB to
147.995 GB (-28.6002%). Mean delta-NLL is +.00722528, with paired per-prompt
bootstrap95% interval[-.00747410,+.02092182]; mean KL is .01259525, p99 .16984025
and maximum .60362084. There are20 argmax changes and worst-prompt delta-NLL
is+.04389141. This confirms read reduction with measurable distribution changes,
without proving equal quality, physical SSD savings or latency. All140 pressure
samples were normal with no swap growth. The compact archive preserves the
original proof-label ambiguity and independently repeated37 CPU trials; all
four large raw arrays are retained and fully hash-verified at their original
paths. See [`held-out confirmation`](evidence/resident-three-confirmation-completed-20261002-v2/README.md).

A separate mechanism comparison kept the six original GPU-produced router
coefficients while changing only selected expert IDs. All native/no-op controls
matched 512 rows and their full raw hashes; the default affinity path exactly
repeated prior BF0 rows and raw bytes. The new policy gave delta-NLL +.013732
versus +.013860, but mean KL .013791 versus .012651 and maximum KL 1.13317
versus .61608; worst prompt delta was smaller (+.095384 versus +.113244).
This mixed calibration result does not support promotion. See
[`resident-native-weights-calibration16-20261002`](evidence/resident-native-weights-calibration16-20261002/compact-summary.json).

The profile-off broader ABBA at an 80 GiB requested cache did not establish an
end-to-end speedup: mean native turn time 30.860 s versus candidate 31.342 s
(+1.56%). Decode was 2.72% faster and append prefill 7.05% slower. Native turn
time itself varied by 9.5% between repetitions, so the modest mean difference
is uncertain. Both native and both candidate repetitions separately had exact
full-logit and graph-state parity. This budget configures 7,862 expert slots,
above the native 7,680-slot prefill threshold; it changes selected-prefill
batching and needs its own quality reference. Peak wired memory was 90.51 GiB,
with normal pressure and no swap growth in all 342 samples. See
[`resident-broader-abba80-20261002`](evidence/resident-broader-abba80-20261002/compact-summary.json).
Generated completions and separate native/no-op/apply attribution remain next.

The native cache-budget effect is now isolated in a profile-off same-binary
64-versus-80 GiB ABBA. Mean turn time fell from 40.900 s to 31.071 s (24.03%
less): append prefill 22.390 to 13.598 s (39.26% less), decode 18.510 to
17.473 s (5.60% less), and startup prefill 63.055 to 52.962 s (16.01% less).
Each budget's repetitions separately matched full raw logits and graph state
exactly; turn variation within each budget was .13% and .48%. Configured
capacities stayed at 6,136 and 7,862 slots, respectively, through all phases.
All 373 pressure samples were normal, peak wired memory 90.33 GiB, with no
swap growth. This is one replay workload and excludes tools and sampling;
the separate native80-versus-native64 fixture check is described below. See
[`native-cache64-v80-abba-20261002`](evidence/native-cache64-v80-abba-20261002/compact-summary.json).

The frozen native64/80/80 quality matrix then matched all 512 raw rows and
full reference file hashes exactly on 16 official-continuation cases: measured
delta-NLL, full-vocabulary KL, total variation and argmax changes were all zero.
The 80 GiB repeat matched too. All 150 pressure samples were normal, peak wired
memory 86.77 GiB, with zero swap growth. This checks those shorter fixture
prefixes; it does not establish byte parity between the differently batched
long agent-replay states. See
[`native-cache80-quality-completed-20261002-v1`](evidence/native-cache80-quality-completed-20261002-v1/checks.json).

The full evolving replay comparison is also complete: native64 reproduced all
343 original raw decode rows, and native80 repeated all343 byte-exactly against
its own recorded reference. Cross-budget arithmetic is different: token-weighted
delta-NLL +.000222631 nats/token, mean KL .000303374, TV .00187026 and2 argmax
changes. Maximum KL was .0171783. This is one correlated trajectory; phase
bootstrap output from the generic summarizer is not an independent confidence
interval and is excluded from the compact analysis. The 24% profile-off turn
time gain thus has small measured average distribution drift, rather than
long-replay byte parity. Generated task correctness remains a separate check.
All353 guard samples were normal, with512KiB maximum swap growth. See
[`native-cache64-v80-long-replay-quality-20261002`](evidence/native-cache64-v80-long-replay-quality-20261002/compact-summary.json).

At the fixed 64 GiB budget, selected prefill tiles of 64 with a selected limit
of 1024 matched the native80 comparison in all 343 rows' printed non-timing
metrics. Against the retained native64 reference, delta-NLL was +.000222631,
mean KL .000303374, TV .00187026 and 2 argmax changes. The native before/after
controls each reproduced all 343 full-vocabulary raw rows. The compare helper
does not export candidate logits, so matching serialized metrics does not
establish raw byte identity across those policies. All292 guard samples were
normal and swap stayed flat; peak wired memory was 74.30 GiB. These timings
include quality instrumentation. The frozen profile-off ABBA owns the latency
test; the later medium-append coding check below supplies separate generated
task evidence. See the
[`fixed-budget prefill quality archive`](evidence/native-prefill64-tile64-limit1024-quality-completed-20261002-v1/RESULTS.md).

The subsequent profile-off ABBA confirmed the fixed-budget benefit on this
replay: append plus decode averaged 40.7754 seconds native versus 32.5250 seconds
with tile64/limit1024, a 20.2338% reduction. Append prefill fell 36.1114%, decode
0.8046%, startup prefill 16.8178%, and all model phases 18.1416%. Repeat ranges
divided by their means were 0.1438% native and 0.2053% candidate. Both policies
retained 6136 configured expert slots throughout. Recorded seven phase-end
logit and final graph-state hashes repeated within each policy, matching the
earlier native64 and native80 hashes respectively. The driver removed these
binary dumps after hashing; the retained commands, timings and hash records
are archived. All 380 guard samples were normal, with zero swap growth and
peak wired memory 73.77 GiB. Tools and sampling are excluded. This attributes
most of the earlier 80 GiB gain to batching on this workload; it does not
establish general task throughput. The existing opt-in flags are
`DS4_METAL_V41_SELECTED_TILE=64` and `DS4_METAL_V41_SELECTED_LIMIT=1024`, tested
with resident routing disabled at a 64 GiB cache budget. See the
[`fixed-budget prefill timing archive`](evidence/native-prefill64-tile64-limit1024-abba-completed-20261002-v1/RESULTS.md).

A quiet six-arm A/B/C/C/B/A comparison grew the same recorded context to
26,659 tokens at cache64, with default batching, fast batching and fast
batching plus resident-three. Fast batching reduced initial prefill 20.035%,
but append plus decode only 1.044%; its 8,192-token append regressed 3.656%.
Adding resident-three reduced decode 13.524% and append plus decode 2.091%
against native fast. Each policy reproduced all nine phase vectors and its
final canonical state exactly across its two fresh processes, with 6,136
expert slots in every phase. All 869 pressure samples were normal with no
swap growth. This is one teacher-forced workload and does not establish
correctness or equal quality; defaults remain unchanged. The complete archive
retains all phase regressions and exact large raw/state pointers. See the
[`quiet long-context comparison`](../experiments/native-resident-growth-six-20261003/README.md).

The subsequent generated coding check also passed. Native64, tile64/limit1024
at64 GiB, and native80 each produced four first completions that passed all34
unchanged independent checks, with no repair or cap hit. Each case appended
393–414 tokens after47 native warm tokens, exercising nonzero-position medium
prefill. Native64 and selected64 warm vectors were byte-identical; native token
inputs matched across all arms. Selected64 and native80 matched in every initial
post-append full-vocabulary vector, raw code and token decision excluding timing.
Two task outputs differed from native64 and still passed their checks. Emitted
tokens totaled574 native64 versus569 selected64/native80. All319 pressure samples
were normal with no swap growth, peak wired memory89.29 GiB. This is four
bounded tasks and diagnostic generation timing; the profile-off replay ABBA
above supplies the speed result. See
[`medium-append coding evidence`](evidence/native-prefill-coding-medium-completed-20261002-v1/RESULTS.md).

On the first actual ledger-reconciliation coding task, both native and resident
generated fixes passed all five original tests plus 10,000 independent cents
checks and retry/refund/void/generator cases. Original tests and CSV data were
unchanged; only the two requested source files changed. Both transcripts hit
the same 275-second budget before their final waiting marker, so complete-turn
and complete-harness success are false. This validates those generated fixes,
not all three original tasks or task throughput. The shorter initial candidate
attempt and wrong-model-alias startup rejections are excluded. The full-budget
candidate restarted from the unchanged original fixture with the same frozen
binary/settings. See `evidence/resident-coding/`; source snapshots, strict
checks, command/model/binary provenance, raw traces and pressure logs are kept.

A later complete three-turn ledger study exercised one context compaction in
each native-before, broader-three-slot and native-after arm. All nine recorded
task stages passed, and each saved project independently passed all three gold
checkers and its eight-test suite. The original harness still failed because
its flat trace validator rejects compaction; the separate compaction-aware
analysis preserves those failures and the guard's return code 3. The resident
agent used 19 rounds and 22 tool calls versus native's 17/20 and 18/21, so this
supports task completion on the fixture without a task speedup claim. Net SSD
reads were not measured in these agent runs. See the
[`complete agent task evidence`](../experiments/ds41-ssd/AGENT_LIVE_COMPACTION_V3_RESULT.md).

The penalty's four easy calibration problems were all correct with identical
315-token aggregate traces at lambda 0/.5/1. Its apparent timing differences
were cache/order effects. A harder run was interrupted by a system panic;
partial live observations included a correct shorter counting answer and a
baseline-correct/candidate-capped lattice answer. Its raw partial traces were
lost in the reboot and are not treated as a completed accuracy benchmark.
The fully retained easy calibration is under
[`overthink-calibration-20261002-1540`](../experiments/ds41-ssd/overthink-calibration-20261002-1540).

After recovery, two complete guarded hard-task pairs reproduced the failure at
lambda 1, using fixed seed, a 1,024-token budget and independent counting golds.
The circular/lattice native answers were both correct (415 + 941 = 1,356 tokens).
Immediate penalty gave one correct answer and one unclosed capped trace
(389 + 1,024 = 1,413 tokens); delaying until 256 generated thinking tokens gave
the same score with 1,439 tokens. Both native traces were byte-identical across
the two gate settings. The late circular trace was also unchanged. Neither
setting supports adoption. These are two repeated calibration problems, not
broad reasoning accuracy evidence. Complete summaries/provenance are in
[`guarded-hard-two-summary.json`](../experiments/ds41-ssd/overthink-provenance/guarded-hard-two-summary.json),
with persistent raw traces in the logit workspace's artifact directory.

Weaker immediate full-list penalties at lambda .5 and .25 also scored one of
two within 1,024 tokens, using 1,413 and 1,423 total tokens versus native's
1,356. Both lattice traces reached the gold number in thought but did not
produce a completed answer before the cap. Their first divergence suppressed
the ordinary logical token ` or`, which is included in the paper's word list.
Larger-budget prefix-verified extensions and a fixed narrower hesitation list
are prepared separately; the current results do not support enabling the penalty.

The completed narrower hesitation-list test at native effort75 used four
counting problems, lambda0 versus.5 and a2048-token cap. Both policies completed
one correct answer and left three INVALID capped answers. Native emitted6681
tokens versus6550 with the penalty (-1.9608%); the entire131-token reduction
was circular10 (537 to406). No capped-CORRECT answer occurred. The checked
analysis preserves producer and guard rc3 while its own trace-integrity checks
pass. All656 pressure samples were normal, with no swap growth. This result
supports neither an accuracy improvement nor a claim that quantization caused
the long reasoning; native lower-effort comparisons are frozen separately.
See the [`narrow penalty outcome`](../experiments/narrow-half-counting-analysis-20261002/README.md).

The native lower-effort checks used the same four counting problems, seed,
temperature and 2,048-token cap, with no word penalty. Effort25 completed two
correct answers and left two unclosed capped answers, totaling 5,985 tokens
versus effort75's 6,681 (-10.42%). Effort50 completed one correct answer and
left three unclosed capped answers, totaling 6,336 tokens (-5.16%). These are
single-seed observations with different effort-conditioned prompt prefixes;
the effort25 result supports trying native effort control before adding a
penalty, but does not establish a general accuracy advantage or quantization
as the cause. Producer and guard rc3 are preserved for both runs; passing
analysis checks validate the traces, not capped answers. All pressure samples
were normal with no swap growth. The archives retain all four original token
traces, strict gold checks, source/binary provenance and release proofs:
[`effort25`](evidence/native-effort25-completed-20261002-v1/README.md) and
[`effort50`](evidence/native-effort50-completed-20261002-v1/README.md).

A separate six-problem holdout completed the frozen effort order
0/25/75/75/25/0. Effort 25 completed five correct answers and retained one cap,
using 5,002 generated tokens per block; effort 75 completed four correct
answers and retained two caps, using 5,974 tokens. That is 16.27% fewer tokens
on these six problems. Each level's before/after blocks reproduced every
prompt, sampled token, RNG state, piece, raw byte and original score exactly.
The repetitions do not add independent problems. Effort 0 had no strict
format matches and three caps; a separate manual review found correct terminal
numbers in its three stopped explanatory responses, without changing their
original INVALID scores. All 36 first attempts and original rc3 are retained.
All 2,017 pressure samples were normal with no swap growth. This supports a
repeatable native-effort comparison on a small suite, with no cross-prefix
KL/NLL, quantization-causality or timing claim. See the
[`six-problem native effort holdout`](../experiments/native-effort-holdout-20261003/README.md).

The separate closing-token pilot at effort25 added 2 to the FP32 `</think>`
logit after 1,024 generated thinking tokens. Native-before and native-after
exactly reproduced all four original effort25 attempts. The boost completed
the same two correct answers, retained both INVALID caps, and saved only one
token overall (5,985 to 5,984). Its tiny sampled active pre-filter KL values
describe the immediate logit change at the actual sampling temperature,
not corpus loss or quantization causality. All first attempts, producer/guard
return code 3 and the corrected analysis gate's initial refusal are retained.
See the [`completed closing-token pilot`](../experiments/ds41-ssd/CLOSING_FIRST_ATTEMPT_RESULT.md).

## Persistent and monitored execution

Useful workspaces live in the ignored repo `.workspaces/` directory. Large
references, frozen binaries, run manifests and pressure logs live in ignored
`speed-bench/ideas/artifacts/`; compact review evidence is copied to `evidence/`.
Do not place useful experiment workspaces or unique run evidence in `/tmp`.

Every model-opening command, including tokenizer diagnostics, must use the same
guard. It serializes jobs with a file lock, monitors the host and cleans up its
owned process groups on interruption, failed probes or sustained pressure.
Large RAM use alone is permitted. The guard is a benchmark control, not proof
against kernel/driver faults. Sampling, child-process cleanup, probe timeout and
PID-reuse CPU checks are in `test_pressure_run.py`.

```sh
python3 speed-bench/ideas/pressure_run.py \
  speed-bench/ideas/artifacts/RUN-pressure -- \
  python3 speed-bench/ideas/run_screen.py speed-bench/ideas/artifacts/RUN \
  --baseline BASE_PROBE --candidate CANDIDATE_PROBE \
  --manifest speed-bench/ideas/heldout.tsv --gaps 0.05 --mass 0.12
```

The 2026-10-02 16:02:59 UTC panic occurred with one reasoning pilot and two
concurrent tokenizer-only DS4 processes. The pilot used 80.26 GiB resident RAM;
the panic snapshot had 93.41 GiB wired, 33.28 GiB compressed and 13.92 MiB free,
plus 100 swapfiles/low swap space. Those facts establish severe combined memory
pressure and an accidental scheduling violation, not the exact cause of the
watchdog stall. `evidence/panic-memory.json` records the relevant local snapshot.
After recovery the held-out screen completed with normal kernel pressure and
zero swap throughout. Pilots now expose an explicit cache size and produce
persistent, inspectable manifests instead of relying on the automatic cache.

A later timing replay was stopped by the guard before its first timing row,
when kernel pressure became elevated. A separate `CGPDFService` processing
`ESLII_print4.pdf` in a Time Machine backup grew to 97.86 GiB RSS. Its replacement
also grew rapidly and ignored SIGTERM; SIGKILL stopped that verified disposable
worker, without modifying the PDF or backup. This is evidence about the later
pressure incident, not proof of the original panic's cause. The interrupted
replay is retained in `artifacts/resident-abba*` and supplies no timing result.

## Sources

- Context Language Models, sections 4.1/4.3, appendix B:
  https://arxiv.org/html/2609.37725v1
- Quantized Reasoning Models Think They Need to Think Longer, but They Do Not,
  section 5.1 and table 8: https://arxiv.org/html/2606.00206v1
- Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference:
  https://arxiv.org/html/2412.00099v2

The CLM paper's 35% result is server compute reduction on its tested Qwen
workload, not a DS4 latency estimate. Its suffix reuse explicitly approximates
edited histories. The penalty paper did not test DS4.1 Q2/MoE. The cache-aware
routing paper includes DeepSeek-V2-Lite, not this model. Local measurements
determine whether any of these transfer usefully.

The next native-weight resident study completed all 19 guarded arms with exact
old/new default states, full teacher logits, logical-I/O controls, positive-flag
no-action controls and native brackets. Preserving the original six router
coefficients improved mean Delta NLL versus affinity-renormalized replacements
on calibration (0.004714 vs 0.007601) and the earlier confirmation set
(0.000962 vs 0.007225), but the paired confidence intervals include zero and KL
tails are mixed. Confirmation requested bytes were identical for the two
approximations (147,995,320,320), 28.60% below native. These are pre-close
prefill plus decode logical requested/returned bytes, not physical SSD traffic
or latency. The existing confirmation cases are reused evidence, not an
untouched holdout. The new coefficient option remains private and opt-in.
See `../experiments/resident-native-weights-20261003/compact-summary.json`;
36 large arrays/runtime inputs remain hash-bound in their durable workspaces.
Pressure remained normal for all 1,048 samples with zero swap growth.

The private agent pair-tape caller completed its four guarded native controls
and real worker reuse transaction. Pair recording and final native repetition
matched every tested canonical state and all 19 full-vocabulary vectors. The
approximate branch moved the source buffers, repaired 768 pairs across three
boundaries, reused 512 tokens after an odd 65-token deletion, and kept the
whole latest 128-token User turn fresh. Both cancellation paths restored exact
canonical state; canonical export was blocked and actual named-session save
wrote zero KV payload. Transaction admission peaked at 115,687,927 bytes under
the 128 MiB budget; this is not an observed global memory peak. Independent
FP64 raw-logit metrics matched the caller (KL 0.020010, TV 0.038085, zero argmax
flips across 17 rows). This is one structural transaction, not evidence of
semantic forgetting or a latency promotion. See
`../experiments/agent-suffix-pair-native-20261003/checked-summary.json`; large
raw/state/runtime files remain hash-bound in the durable private workspace.

The fixed 1,024-token thinking budget pilot completed native-before, budget,
and native-after on the original four calibration cases. Native repeats were
exact, and each forced case matched the native token/RNG trace until the
single closing-think token was inserted without advancing RNG. Generated
work fell from 5,985 to 4,310 tokens (27.99%); completed correct answers stayed
at two, while caps fell from two to one. The newly completed lattice answer
was incorrect (42 rather than 2,002), and the subset case still capped after
1,023 final-phase tokens. All first attempts and strict INVALID scores remain
in the archive. This is one calibration set and seed, with no equal-quality,
holdout, speed, or quantization-causality claim. All 830 pressure samples were
normal with zero swap growth. See
`../experiments/reasoning-budget-pilot-20261003/compact-summary.json`; generated
text/token traces are not full-vocabulary logit arrays. Production defaults
are unchanged.

The released eight-arm resident calibration is preserved at `../experiments/resident-env-calibration-20261003`. All native and prior-anchor full-vocabulary, teacher, cache and requested-byte controls matched. Five new policies exposed tradeoffs; none passed the fixed full screen, so the frozen selection remains anchor-affinity3 with no advance. ROOT independently recomputed all4096 vocabulary distributions in FP64 and matched every CSV metric within6e-12. This is a calibration result, with no new default, physical SSD or speed claim.

The first actual three-turn suffix task is preserved at `../experiments/agent-suffix-pair-actual-first-failure-20261003`. Both earlier semantic controls passed. The exact byte guard rejected the third prompt because the production stdin protocol appends one LF to the unchanged frozen prompt, before any edit selection, suffix transaction, third evaluation or grader. All first-failure, build, source, trace, pressure and ROOT diagnosis records are retained. Its incidental fresh native system-prompt cache is a durable external reference; named persistence was not tested. A corrective source requires its own CPU qualification, build, preparation and sole-model grant.

The fresh mixed24 resident confirmation is preserved at `../experiments/resident-mixed24-confirmation-20261003`. Native full-vocabulary/ordered teacher/cache/IO brackets matched exactly. The frozen anchor saved23.10% of requested expert bytes, but failed the preset meanDeltaNLL, meanKL, p99KL, argmax-flip and worst-prompt screens: meanDeltaNLL+0.012299, KL0.023915,44/768flips. The24-prompt bootstrap95%DeltaNLL interval[-0.017604,0.043152] crosses zero. A source-frozen independent FP64 audit of2304 full vectors matched every CSV/summary metric within1e-8. All six small category results and first attempts remain retained; no new policy selection, physical SSD/timing or default claim.

The exact-LF corrective suffix task is preserved at `../experiments/agent-suffix-pair-actual-admission-failure-20261003`. Both control turns passed and the third protocol guard passed. The fixed128MiB selector stopped before candidate selection, suffix reuse, third evaluation or grading:137150794 forecast bytes exceed134217728 by2933066. Its30140128-byte undo reservation was not allocated by the selector, and this is not a global peak measurement. The original failure remains unchanged. A separately frozen160MiB memory-admission experiment was chosen before any third-turn semantics, with the same prompts, edits, sampling and grading.

The fixed160MiB first attempt is preserved at
`../experiments/agent-suffix-pair-memory160-first-failure-20261003`. Both control
turns passed and the pre-third selector admitted an odd11-token shift with881
eligible suffix tokens and387fresh tokens. The model searched for the updater
instead of making the required first invocation. By the eventual edit, later
tool-round captures had evicted the selected source; the trace records cached0
and a full4245-token fresh prefill, with no approximate suffix transaction.
Its one external grading attempt passed, but the overall task remains a
retained first failure and provides no semantic-reuse or speed result. All243
pressure samples were normal, with zero swap growth and86.61GiB peak wired plus
compressed RAM. The matching native-off caller compiled cleanly but remains
unrun: its different absolute live-context path would change the system prompt
and violate full opaque source identity. The original structural choice is
preserved without reselection. Named persistence was not exercised; production
runtime files and defaults are unchanged.

### Ragged long-append native-tail qualification (2026-10-03)

Six first-attempt native/control arms completed. Both actual long-tail dispatch changes were exercised:128 rows changed64+64 to sixteen8-row calls;511 rows changed seven64+63 to one511-row call. Native controls matched full logits, canonical state and IO. The first32 teacher vectors were bit-identical; the second32 shifted probabilities. Across64 predictions, independent full-vocabulary FP64 audit confirms mean delta-NLL +0.05274755, mean KL0.00518421, max KL0.18942086 and zero argmax flips. These are two artificial extensions, not generated semantic tasks. Profiled single-order times remain diagnostic; no quiet timing grant, quality-preserving or default claim.

Evidence: `../experiments/ragged-native-tail-qualification-20261003/compact-summary.json`; all six raw/state outcomes retained with literal SHA/size/mode refs in its archive index.
