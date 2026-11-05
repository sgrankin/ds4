# Decode and prefill hill climbing

The user authorized sequential performance experiments from November 4 at
08:35 Eastern through November 6 at 18:31 Eastern. The published starting
runtime is `364ceeab`, which includes the five adopted overnight decode changes.
The outcome is lower decode and prefill time on this M5 Max, with measured
quality tradeoffs and no unrelated architecture work.

Each experiment gets an initial one-hour budget: profile, choose one target,
implement, compare at least five alternating samples per side, and decide.
Subagents prepare or review that same target; only one guarded heavy model
job runs at a time. Periodic comparisons against the starting runtime will
check the accumulated package. The final slice is reserved for combined
qualification before the deadline. Raw artifacts stay ignored under
`.workspaces/hillclimb-20261104/`.

## First question: shared work during selected prefill

The earlier 32K profile showed long waits around selected routed-expert work
in a 512-token append. The batch overlap experiment found only a small shared
branch available to hide, while large delays outside routed GPU execution
remained. Five-pair comparisons were slightly favorable at 8K and inconclusive
at 32K, where both arms varied substantially. One control was interrupted by
the pressure guard; its unfinished pair was retried after pressure recovered.

We retained a narrow cleanup in the existing opt-in path: resident prefill has
already made the selected IDs ready, so it need not submit and wait for another
empty command buffer. A diagnostic confirms 320 fewer empty submissions per
512-token append, with unchanged GPU operation counts. Ordinary, native-routing
and tight-cache quality checks matched exactly. The fast presets remain
unchanged; there is no claimed speedup for this cleanup alone. The
[result report](../../../speed-bench/experiments/2026-11-04-prefill-shared-overlap/README.md)
owns the measurements, failure record and qualifications. Main remains at the
starting revision.

## Second question: stage more attention rows per barrier

We tried the existing sixteen-row attention staging kernel during prefill,
preserving each head's arithmetic and key order. The fixture and 8K model
quality checks passed, but all five timing pairs at both 8K and 32K favored the
original kernel. Fresh prefill rose from 49.34 to 53.25 seconds at 32K and from
20.16 to 20.71 seconds at 8K. The candidate profile places the added work in
attention; it does not isolate shared-memory occupancy from repeated KV loads.

The selector was removed, while the existing decode kernel remains. Reducing
barriers alone did not earn the extra staging memory and smaller head groups.
The [result report](../../../speed-bench/experiments/2026-11-04-prefill-row-blocking/README.md)
preserves all samples, the tested revision and the numerical checks. No preset
or installed binary changed.

## Third question: share packed-cache scale decoding

The packed KV reader repeated scale decoding across four neighboring lanes.
We tried exact half-bit construction and a SIMD broadcast of one decoded scale,
preserving storage, head groups and attention order. Focused GPU checks and
512 full-model positions matched exactly. Five-pair whole-model timing was
mixed and close: slightly favorable at 8K, slightly unfavorable at 32K, with
neither establishing a useful gain. A warmed five-pair attention-operation
comparison was also neutral, about 32.02 versus 32.05 ms per call.

We restored the original reader because the additional implementation did not
earn its complexity, rather than declaring a regression from a small timing
difference. Fewer logical per-lane operations need not mean fewer SIMD
instructions or physical reads: the lanes still execute together and request
the same unique scale bytes. That remains a source-level explanation, not a
hardware-counter finding. The new packed-versus-expanded numerical fixture
is retained. The [report](../../../speed-bench/experiments/2026-11-04-packed-attention-reader/README.md)
owns all samples and the temporary operation benchmark's reproduction details.

## Fourth question: modest staging without losing head reuse

Keeping sixteen heads per group while staging four KV rows reduced the focused
attention operation by about 1.5–1.6% at 512, 2048 and 8192 rows; all fifteen
alternating pairs favored it. Five-pair full-model comparisons showed no clear
change in total time at either 8K or 32K. Append differences followed run order, and the diagnostic confirmed
that this workload's append did not use the new kernel. Numerical checks matched
exactly across 512 full-vocabulary positions and their post-step logit hashes.

We retained the kernel and enabled it in the resident fast source presets.
This is a small operation-level improvement, not an established whole-task
speedup. The original kernel remains available through an explicit override,
and the added staging is temporary threadgroup memory. The
[report](../../../speed-bench/experiments/2026-11-04-attention-four-row/README.md)
owns the samples, activation limits and profile. The remaining per-row softmax
recurrence still performs work when its rescaling factor is one, making that a
candidate for the next bounded question.

## Fifth question: skip a softmax scale of one

The online softmax rescales accumulated output even when its maximum does not
change. Skipping that identity operation reduced isolated attention time by
about 9–10%. Five-pair full-model tests also improved: total inference fell
from 64.33 to 63.37 seconds at 32K and from 35.99 to 35.59 seconds at 8K.
The compiled shortcut changes rounding, however, and small attention errors
propagated into nonzero model-distribution drift. Mean teacher likelihood moved
slightly against the candidate at 8K and in its favor at 32K; neither direction
establishes practical answer quality.

An initial short coding prompt did not exercise this kernel. A longer repeat
confirmed activation and both generated functions passed the independent
207-case oracle. The candidate corrected an erroneous self-test expectation
and compacted once before completing. Those observations remain visible rather
than being reduced to a pass count. The separate profile confirmed dispatch,
but its inflated timings do not provide another speed comparison.

We retained the implementation as an opt-in approximation, without preset
promotion. This is a useful measured tradeoff, not a rejection for differing
bits. The next bounded question is whether explicitly preserving the rounded
KV contribution retains the speed benefit while reducing drift. The
[result report](../../../speed-bench/experiments/2026-11-04-attention-unit-rescale/README.md)
owns samples, numerical tails, task details and the original failed exactness
check. The four-row staging preset remains adopted independently.

## Sixth question: preserve contribution rounding explicitly

We tested a local contraction control and explicit FMA to preserve the rounded
KV contribution while retaining the shortcut. It did not restore the original
fixture values; more elements differed and RMS error increased. Five-pair
operation measurements found no useful gain at 2048 rows and higher mean times
at 512 and 8192 rows. The original shortcut still improved its separate paired
2048-row comparison by about 9% with the same binary.

The new variant was removed without proceeding to full-model measurements.
It failed the intended numerical refinement and did not offer an operation-level
advantage; this is not a judgment of practical task quality from fixture error,
or a rejection based solely on a small noisy timing difference. The
[report](../../../speed-bench/experiments/2026-11-04-attention-contribution-rounding/README.md)
preserves all samples and the prototype revision. The first shortcut remains
opt-in. Combined qualification is the next useful step before choosing whether
to promote it, rather than continuing to chase exact rounding.

## Seventh question: does the package justify preset adoption?

The 12:26–13:26 Eastern slice compared the four-row staging and unit-rescale
shortcut together against published `364ceeab`. Five alternating pairs improved
32K fresh prefill and inference total; the five-pair recorded coding replay was
inconclusive, with a slow final candidate sample retained. Full-logit checks
showed modest average drift with larger local tails. Both reasoning-enabled
coding runs passed the existing 207-case oracle without compaction, but an added
empty-generator probe failed on the candidate despite its iterable docstring.

Adopt the shortcut in the resident fast presets, preserving explicit disablement
and ordinary engine defaults. This accepts a measured approximation with limited
practical evidence, rather than claiming identical task quality. The
[combined report](../../../speed-bench/experiments/2026-11-04-combined-prefill/README.md)
owns all samples, quality tails and the coding limitation. Further hill climbing
uses this package as its local baseline; published main remains unchanged.

## Eighth question: widen the exposed output projection

Attention output-B contributes about 3.7 ms of almost entirely exposed GPU
work per steady token in the diagnostic profile. We doubled its output rows
per threadgroup, preserving each row's arithmetic while halving the group
count. All 512 paired full-vocabulary positions at 8K and 32K matched exactly.
Five-pair timings did not establish a whole-model gain; the final 8K candidate
also coincided with a swap excursion and remains in the reported samples.

Before rejecting the neutral result, we compared the dependency profiles.
They showed neither less output-B work nor a useful newly exposed bottleneck.
The variant and its temporary fixture were removed, with measured revisions
preserved in the [report](../../../speed-bench/experiments/2026-11-04-output-b-rows/README.md).
This is a decision about absent demonstrated benefit, not a claim that a small
noisy difference proves a regression. The qualified preset package is unchanged.

## Ninth question: remove the resident-routing host handoff

We reused the cache's GPU address tables and added resident-only ID repair,
with ordinary routing every eight positions to refresh the cache. The prototype
pins every possible resident choice and preserves native coefficients, but
relaxes the preset's substitution limits. Five-pair tests found neutral fresh
decode and consistently slower warm decode at 32K; 8K total timing was neutral.
Distribution drift increased at 32K, while both reasoning-enabled coding outputs
passed the independent interval tests. A diagnostic trace allocation failure
remains recorded separately from successful task completion.

The profiles establish useful groundwork: the routing handoff shrank, and CPU
sampling exposed redundant address-table writes. Keep the explicit opt-in for
a bounded follow-up removing that preparation work, without promoting it to
presets or replacing the installed binaries. The
[report](../../../speed-bench/experiments/2026-11-04-resident-handoff/README.md)
owns the samples, numerical tails and profile limitations. Measure the follow-up
and the combined path against their respective baselines before judging benefit.

## Tenth question: reuse the resident cache's published addresses

Cache installation and eviction already maintain GPU address tables under the
selected streaming configuration. We skipped repeated writes while preserving
identity checks and all in-flight pins. All 512 fixed-history logit rows and
512 post-token hashes at 8K and 32K matched the previous GPU-routing prototype.
Four five-pair timing comparisons separated this preparation change from the
combined route. Address reuse alone did not establish a decode gain. The
combined path improved fresh 8K steady decode slightly, but remained neutral
overall and consistently slower in warm 32K decode.

Retain the small preparation cleanup behind its explicit opt-in; do not promote
the GPU route to presets. Source inspection also found that the detailed encoder
timeline changes batching and serializes logging with submission, so its reduced
gap cannot be treated as saved native latency. The
[report](../../../speed-bench/experiments/2026-11-04-resident-address-reuse/README.md)
owns the samples, CPU diagnostic and clock limitations. The next bounded question
is whether resident-only execution enables useful overlap between shared and
routed experts, using the existing concurrent FFN machinery with narrow cache
ownership checks.

## Eleventh question: overlap shared and resident routed experts

The resident-only route removes the host load/readback interruption, allowing
the existing parallel FFN machinery to overlap independent shared and routed
branches. Narrow admission preserves cache addresses, pins, the producer barrier
and cleanup. Activation checks exercised every eligible layer; all 512 paired
logit rows and hashes matched serial GPU routing. This adds no measured drift
to that route, but does not erase its approximation relative to CPU routing.

Four five-pair comparisons found useful decode gains. Against ordinary CPU
resident substitution, fresh decode improved at 8K and 32K, warm decode improved
at 8K, and warm 32K timing was neutral. First-token outliers and prefill drift
limit whole-workload conclusions. Retain concurrency as an explicit opt-in,
without preset or installed-binary promotion. The
[report](../../../speed-bench/experiments/2026-11-04-resident-parallel/README.md)
owns all samples and qualification. This is a useful follow-on from an initially
neutral dependency change; the next bounded question is the refresh interval's
speed–quality curve before choosing a preset configuration.

## Twelfth question: choose a resident refresh interval

With runtime and cache size fixed, we compared ordinary CPU routing with GPU
resident selection that returns to the ordinary route every four, eight or
sixteen positions. Five-pair timings and longer fixed-history quality checks
make the tradeoff explicit. The sixteen-position option improves fresh decode
at 32K and across 512-token continuations at 8K. Warm gains are smaller and
less certain. The four-position option has less measured quality drift, but a
retained slow timing sample leaves its fresh-speed estimate unresolved.

Carry sixteen into recorded-agent and fresh coding qualification; do not change
presets or installed binaries yet. The measured average quality cost is modest,
but larger local tails and changed top choices remain visible. Timing and
quality use different window lengths in the 32K sweep; the 8K follow-up matches
them. The [report and tradeoff figure](../../../speed-bench/experiments/2026-11-04-resident-refresh/README.md)
own all samples, variation and interpretation limits. Normal pressure and
read-only system checks did not establish a cause for the timing outliers.

## Thirteenth question: qualify the resident GPU package on agent work

The selected sixteen-position refresh did not earn preset promotion. Five
alternating recorded-session pairs against published `364ceeab` establish no
complete inference gain. Mean fixed-history drift remains modest, but the
candidate failed to produce one coding implementation within either tested
budget; the longer retry repeated a reasoning sentence twenty-one times.
The baseline passed, while both arms passed the other coding task. Different
working-directory text prevents attributing this failure solely to routing.

Native CPU samples nevertheless confirm the intended dependency change: most
waiting moved from expert-selection handoffs to completion of submitted token
work. Keep that groundwork and the GPU route as explicit opt-ins. The
[qualification report](../../../speed-bench/experiments/2026-11-04-resident-agent-qualification/README.md)
owns timing, quality, retry and diagnostic evidence. The next bounded question
holds the full prompt and working directory fixed while comparing routing
policies; GPU execution versus queue gaps remains a subsequent profiling target.

## Fourteenth question: isolate the coding completion failure

Holding the task directory, injected time and visible files fixed, with fresh
system-prefix caches, changed the picture. All five routing configurations
passed the longer greedy prompt. With the background removed, both CPU-routing
controls produced no file while all three GPU configurations passed. At normal
sampling over five matched seeds, current CPU routing passed five times and GPU
refresh sixteen passed four; its remaining run exhausted its call budget planning.
Every completed implementation passed the independent oracle.

These are limited trajectories on one problem, not a general accuracy estimate
or a diagnosis of the earlier failure. No weight, address or lifetime defect was
found. Keep the GPU package experimental, retain the failed traces and return
to performance work without changing routing policy. The
[controlled report](../../../speed-bench/experiments/2026-11-04-resident-behavior/README.md)
owns the settings and results; the benchmarking guide now explains the prompt,
cache and sampling controls. The next target is the resident kernel's single-lane
search, preserving its existing selections while measuring a SIMD alternative.

## Fifteenth question: parallelize resident expert selection

The resident repair kernel searched all experts on one GPU lane. A 32-lane
search preserves its replacement order, tie rules and native mixture weights,
including the scalar implementation's numerical edge cases. Hardware fixtures
and all 512 fixed-history logit rows and hashes matched. Repeated isolated tests
found much cheaper searches when experts were missing, with a small all-hit
cost. This changes execution of the existing approximate GPU policy, not its
quality tradeoff against CPU routing.

Retain the small SIMD implementation as an explicit opt-in. Five-pair 32K
attribution improved fresh decode in every pair; 8K fresh complete decode was
slower in four of five, while warm results were more favorable. First-token and
append variation prevent a general complete-workload improvement claim. The
[report](../../../speed-bench/experiments/2026-11-04-resident-simd/README.md)
owns the isolated and combined comparisons, every sample and the retained slow
cases. Presets and installed binaries remain unchanged. The next selected slice
will inspect the updated dependency profile before choosing another change;
removing this search cost does not itself identify the next bottleneck.

## Sixteenth question: fuse native selection and resident repair

The updated profile found substantial submitted GPU work and completion waiting.
It also exposed a narrow serial boundary between native expert selection and
SIMD resident repair. Combining them removes one dispatch while preserving the
tested probabilities, mixture weights and selected IDs; all 512 fixed-history
logit rows and hashes match. Repeated isolated tests reduce GPU work for missing
experts and CPU encoding cost, while first-pair all-hit variation remains visible.

Retain this small exact fusion as an opt-in. Five alternating model pairs at
both 8K and 32K establish no whole-workload gain or regression, and no further
unlocked improvement has yet been identified. Presets and installed binaries
remain unchanged. The [report](../../../speed-bench/experiments/2026-11-04-post-simd-profile/README.md)
owns all samples, profile limitations and the two microbenchmark series. The
next slice returns to prefill: inspect its remaining work and dependencies, then
select one bounded change rather than extrapolating isolated routing savings.

## Seventeenth question: combine prefill rounding and inverse RoPE

Batch attention separately rounded each output head to BF16 and rotated its
last 64 columns. Reusing the existing fused helper removes the additional tail
pass and one dispatch, without new kernels or workspace. Hardware checks and
all 512 fixed-history logit rows matched. Five isolated large-chunk pairs
consistently shortened this transform, while the smaller append shape was noisy.

Retain the small opt-in implementation. Five-pair model comparisons at both
8K and 32K did not establish a complete-prefill gain or regression. The diagnostic
trace confirms that all 103 intended conversions disappear, with lower local
transform spans and no added aggregate delay to the following projection; it
does not predict a whole-model saving. The [report](../../../speed-bench/experiments/2026-11-04-prefill-followup/README.md)
owns every sample, the corrected test-counter failure and scope limitations.
Presets and installed binaries remain unchanged. The profile also identifies
large batched F16 projections still using a matrix-vector kernel, a larger
remaining prefill target for the next bounded slice.

## Eighteenth question: use matrix multiplication for indexer queries

Large F16 projections still used independent row reductions. Source inspection
showed why publication needs that consistency across sweep sizes, so this slice
changed only the temporary indexer query. The existing matrix backend removes
most of the targeted projection work in the diagnostic trace without allocating
new workspace. It changes some rounded queries and subsequent model predictions;
this is a measured approximation, not a lossless dispatch change.

Keep the selector opt-in for combined qualification. Fresh prefill was faster in
all five pairs at both 8K and 32K; measured phase totals and decode had narrower,
noisier results. Both reasoning-enabled coding solutions passed 174 evaluations
by an independent oracle, but this is only one generated task per arm. The
[report](../../../speed-bench/experiments/2026-11-04-prefill-f16-projections/README.md)
owns all samples, distribution drift and the retained tool-recovery trace.
Presets, installed binaries and published main remain unchanged. The next slice
compares the accumulated package against published `364ceeab` on recorded agent
work, rather than adding isolated percentage gains.

## Nineteenth question: requalify the accumulated package

Five alternating recorded-session pairs against published `364ceeab` gave the
candidate a lower mean time and less variation, with four pairs favoring it.
Most of the mean saving came from decode. An unusually fast first baseline
limits the conclusion; all samples remain. The replay's small, exact prefill
path does not exercise the new indexer-query matrix projection, so its earlier
large-prefill result remains separate evidence.

The 3,371 full-vocabulary predictions match the earlier GPU candidate byte for
byte: recent exact kernel changes add no drift on this replay. The package's
existing difference from published routing remains. Both fresh coding agents
finished, but the candidate treated adjacency as requiring touching endpoints,
missing consecutive integer intervals and failing 30 oracle evaluations across
10 cases. This is one
generated solution per arm, not 30 independent failures or a diagnosis of the
responsible runtime change.

Keep the accumulated GPU package opt-in and retain the useful local reductions;
do not promote the policy from this evidence. The
[report](../../../speed-bench/experiments/2026-11-04-package-requalification/README.md)
owns timings, quality tails, the failed implementation and unchanged publication
status. The next slice separately qualifies the new prefill selectors with the
current CPU-routing fast preset, rather than making their adoption depend on
the experimental GPU policy.

## Twentieth question: qualify the prefill options for fast presets

The exact rounding/RoPE fusion and approximate temporary-query matrix path
were measured together with the ordinary CPU resident policy. Across five pairs
each at 8K, 32K and 8K without prefill substitutions, fresh prefill improved in
every pair and the inference-phase sum improved in fourteen of fifteen. Both
reasoning-enabled coding solutions passed the independent oracle. Fixed-history
quality checks retain the changed prediction distributions and their tails.

Adopt both selectors in `resident-fast` and `persistent-fast`, preserving explicit
zero overrides. This accepts a measured tradeoff: warm decode after ordinary
prefill was slower in all five pairs. A bounded diagnostic found different
expert selections but did not explain the full slowdown; the report preserves
that uncertainty. The [qualification report](../../../speed-bench/experiments/2026-11-04-prefill-preset-qualification/README.md)
owns the evidence, activation limits and installed binary hashes. Launcher tests
and the server build passed; the qualified binaries are installed locally.
GPU resident routing remains experimental, and published main is unchanged.

## Twenty-first question: revisit submissions after removing CPU handoffs

GPU-resident routing changed the dependency graph behind the earlier every-layer
submission choice. Comparing intervals one, two and four favored four in nine of
ten measured fresh complete-decode pairs across 8K and 32K, with small differences.
Warm results remained mixed, including occasional large
append and first-token stalls. All samples remain in the
[report](../../../speed-bench/experiments/2026-11-05-gpu-submit-cadence/README.md).

Select interval four for the experimental GPU package. All 512 checked
predictions and post-token hashes matched. The diagnostic confirms 40 to 11
command buffers on ordinary tokens and 80 to 51 on refreshes; normal encoder
grouping did not show lower host latency. This is a small measured scheduling
gain, not proof that every removed submission reduces elapsed time. CPU-routing
presets, installed binaries and published main remain unchanged. The large
first-decode gap remains a separate candidate for investigation.

## Twenty-second question: use address dispatch from the first decode step

The slow first token used direct expert slots, while later steps used cached
address tables. An existing flag bypasses the four-token dispatch threshold
without changing the separate split-execution gate, selected experts or cache
ownership. Five-pair comparisons at 8K and 32K found small, mixed benefits;
all 512 checked predictions and hashes matched. Keep the existing flag in the
experimental GPU settings, with ordinary presets and binaries unchanged.

The [report](../../../speed-bench/experiments/2026-11-05-first-decode-admission/README.md)
also records the more important negative result: the profile confirms all forty
first-token expert kernels switched to addresses, yet large gaps before expert
work remained. The roughly 0.8-second initial evaluation was not fixed. A bounded
slab-residency experiment is the next candidate; any such change must preserve
cache pressure relief and account for cost moved into prefill or startup.

## Twenty-third question: keep streaming expert slabs resident

An engine-owned Metal residency set was attached to the command queue, covering
only the cache's actual slab allocations. Five-pair comparisons at 8K and 32K
consistently shortened the initial scalar evaluation after fresh prefill but
slowed complete fresh and warm decode. Do not select this queue-attached form. All 512 checked
predictions and hashes matched; cache clear/trim tests and a child-only forced
`mlock` failure exercised the new residency teardown.

The [report](../../../speed-bench/experiments/2026-11-05-streaming-slab-residency/README.md)
preserves the useful dependency finding: large gaps between command buffers
shrank, while repeated waits appeared within refresh buffers near the selected-ID
shared event. This does not prove a driver cause, but supports a bounded follow-up
that requests residency without attaching the whole slab set to every command
buffer. Keep the implementation off by default for that follow-up; ordinary
presets, installed binaries and published main remain unchanged.

## Twenty-fourth question: request slab residency without queue attachment

A standing request, with no queue association, showed no added refresh waits in
its separate diagnostics, but the original fresh first-step gaps remained.
All 512 checked predictions and hashes matched, and the cache-lifetime
and forced-lock-failure checks passed. Five-pair timing at 8K and 32K was mixed
and close to baseline. Do not select this policy: the targeted stall remains,
and no overall gain is established. Small uncertain timing differences alone
are not the reason for that decision.

The [report](../../../speed-bench/experiments/2026-11-05-standing-slab-residency/README.md)
records the samples and the changed profiles. Keep the residency infrastructure
off while a bounded follow-up tests queue association with the shared-event
handoff disabled. That uses an existing control to distinguish the interaction
before adding another mechanism. Ordinary presets, installed binaries and
published main remain unchanged.

## Twenty-fifth question: combine queue residency with a different refresh schedule

The failed queue-residency attempt removed the original first-step gaps but
introduced waits near shared events. Disabling shared-expert overlap also
changes the placement and fusion of shared work; the combined path avoids
those diagnostic waits. Against the original pre-residency package, complete
fresh 128-token decode and total inference improved in all five pairs at both
8K and 32K. All 512 fixed-history logit rows and post-token hashes matched.

Retain the combination as an experimental latency candidate. It has a real
recurring cost: throughput after the first token fell in every pair, and warm
decode did not share the clear fresh-session benefit. A small benchmark
diagnostic using existing full-evaluation timestamps attributes most of the
fresh phase's extra time to periodic CPU refreshes in one pair; the warm phase
also has ordinary-step variation. This supplies a next target, not proof of a
driver cause. The [report](../../../speed-bench/experiments/2026-11-05-slab-event-interaction/README.md)
owns the samples and scope limits. Queue residency replaces the unsuccessful
request-only implementation behind the existing opt-in flag. CPU presets,
installed frontends and published main remain unchanged.

## Twenty-sixth question: omit the shared-work submission on refreshes

Skipping one flush keeps shared gate/up and routed work in the same command
buffer, reducing refresh submissions with matching measured kernels, cache
counters and selected-load sequences in the diagnostics. It did not establish
an overall gain: complete fresh decode was
slower in nine of ten pairs across 8K and 32K. Remove the standalone selector;
all 768 paired logit rows across the primary and smaller-cache configurations
matched, including a case with confirmed scalar reads and cache evictions.
The [report](../../../speed-bench/experiments/2026-11-05-refresh-shared-flush/README.md)
retains every timing sample and the scope of those checks.

The dependency inspection remains useful. The old flush can overlap shared GPU
work with CPU cache/address preparation. The measured bind subtotal covers only
part of that preparation and cannot establish the cause of the timing
difference. Keeping both computations in one
batch removes a boundary needed for parallel FFN, yet the current parallel path
also requires a resident-only selection proof. A bounded follow-up must admit
only an exact cached selection, preserve its pins and use the serial path on
misses. Neither fewer submissions nor neutral timing establishes that future
gain. The prior queue-residency latency candidate remains available; presets,
installed frontends and main are unchanged.

## Twenty-seventh question: parallel FFN for an exact cached selection

The follow-up implements a separate proof for the six CPU-selected experts:
validate and pin their cached identities before entering the existing shared
and routed concurrent schedule. Misses retain serial submission and loading.
All 768 paired logit rows and post-evaluation hashes match across 8K, 32K and a
smaller-cache case with scalar reads and evictions. The smaller-cache case
repeats the 8K histories; these are not 768 independent positions or tasks.

Do not select the new flag in the latency package. Five pairs per configuration
show small or mixed whole-workload differences, with order effects; complete
fresh decode is slower in eight of ten primary pairs. Retain the opt-in
prototype as useful groundwork. The timeline confirms concurrent FFN groups
and slightly shorter local sub-envelopes, alongside larger GPU gaps before
those buffers and longer sampled refresh windows. The source suggests a
tradeoff between early shared submission and later combined submission after
CPU preparation, but the diagnostics do not establish a complete cause.
The [report](../../../speed-bench/experiments/2026-11-05-cached-refresh-parallel/README.md)
owns every sample and scope limit. The next slice qualifies the accumulated
selected package against published `364ceeab` before another runtime change.

## Twenty-eighth question: qualify the accumulated package on a recorded session

Five alternating pairs against published `364ceeab` give a modest gain:
recorded-session model time averages 178.68 to 175.63 seconds, with the candidate
faster in four pairs. Retain the unusually fast first baseline and the
phase-specific regressions. This fixed replay excludes sampling and tool work;
it also lacks the short benchmark's intermediate snapshot saves. Only about
64 ms of the mean 1.53-second decode saving comes from phase-first evaluations.

Both full-vocabulary captures match experiment 19 at all 3,371 positions.
Later changes therefore add no measured drift on this history; the earlier
difference from published CPU routing remains. The interval-coding trial again
fails adjacency in the candidate, while both arms pass the long-background
dependency-layer task. Those are two generated solutions per arm, not hundreds
of independent tasks. Keep the selected GPU package experimental and cached
refresh parallelism off. The [report](../../../speed-bench/experiments/2026-11-05-long-session-qualification/README.md)
owns samples, quality tails, oracle checks and reproduction details. No preset,
installed-frontend or main change follows from this qualification.

## Twenty-ninth question: associate slab residency only with expert computation

Command-buffer association preserves the checked outputs and cache behavior,
but restores the large fresh-start delay. Five alternating pairs at each of
8K and 32K make fresh 128-token decode about 0.56 seconds longer; steady decode
is nearly unchanged. All 768 paired quality rows match, including a smaller
cache with actual evictions. That smaller-cache case repeats the 8K history.

Keep the compute-only policy unselected and retain its opt-in groundwork.
The trace shows the extra GPU gap before its first associated expert buffer;
fresh prefill's separate cache-fill blits have no association. Mixed expert
and router buffers still receive the set, so the change does not isolate every
router handoff. These observations motivate a seed-write association test,
without identifying a driver cause or claiming a future gain. The
[report](../../../speed-bench/experiments/2026-11-05-scoped-residency/README.md)
owns samples, profile boundaries, resource checks and reproduction. The selected
package, presets, installed frontends and main remain unchanged.

## Thirtieth question: associate residency with cache-seed writes too

One existing helper call extends command-buffer association to the blits that
populate the cache. The trace confirms all 40 seed buffers are covered and the
previous large first-compute gap is absent. The old compute-only mode is a
historical reference, not a concurrent third arm; this does not isolate a
driver cause. All 768 paired output rows match, including the repeated 8K
history with a smaller cache and actual evictions.

Keep the extended mode as unselected groundwork. Five pairs at each context
show mixed phase costs without establishing an overall gain over queue
residency. In a separate full-token diagnostic pair, the increase is not
confined to refresh tokens; that pair does not explain the comparative variation. The
[report](../../../speed-bench/experiments/2026-11-05-seed-residency/README.md)
owns the timing, scope and causal limits. The narrower association is now ready
for a scheduling follow-up; the selected package, presets, installed frontends
and main remain unchanged.

## Thirty-first question: restore overlap with scoped residency

Restoring shared-expert overlap makes complete decode and total inference
slower in every primary timing pair at both contexts. With the requested 64 GB cache, 32K fresh decode averages 3.80 to
5.00 seconds and warm decode 3.85 to 5.02 seconds. With a requested 16 GB
cache, warm decode instead improves in all five pairs, while total time remains
mixed. All 768 paired quality rows match; the smaller-cache quality case repeats
the 8K history.

The large-cache trace concentrates positive event-adjacent gaps in 29 mixed
expert/router buffers per refresh. The same associations remain in the matched
smaller-cache trace without comparable large gaps; cache contents and misses
also differ, so this does not isolate resource bytes as the cause. Preserve
that favorable case and keep the selected large-cache package unchanged.
The [report](../../../speed-bench/experiments/2026-11-05-scoped-overlap/README.md)
owns the evidence and timing scopes. A narrow outgoing boundary after actual
CPU-routed expert work is available for a follow-up; no new runtime code was
added in this experiment.

## Thirty-second question: separate expert work from the next router event

A checked flush after actual CPU-routed expert work removed the large positive
event-adjacent gaps in the sampled later refreshes. The trace confirms separate
event and expert buffers on those refreshes; ordinary GPU cadence is unchanged.
This turns the earlier neutral residency work into a useful combined schedule.

Select scoped residency, shared overlap and the new boundary for the
experimental GPU package. Direct five-pair comparisons against the prior package
show about 2% faster fresh complete decode at 8K and 32K; warm decode improves
in nine of ten pairs. A 32K append costs about 0.12 seconds more on average,
and total inference remains mixed. The
[report](../../../speed-bench/experiments/2026-11-05-cpu-expert-boundary/README.md)
owns samples, variation, output checks and resource limits. This is a decode
tradeoff rather than a whole-task speed claim. Fast presets, installed frontends
and main remain unchanged.

## Thirty-third question: avoid repeated residency association

Deduplicating active-batch association removes most calls without changing the
covered buffers, cache behavior or checked outputs. It does not establish a
consistent retained-buffer latency gain: primary timings are mixed, and the
comparison with the frozen pre-change binary is strongly influenced by its
first pair. The changed profile preserves scheduling and reveals no new stall.

Keep the small prototype opt-in and unselected for one bounded follow-up with
unretained command buffers. That existing mode also incurs the repeated
completion-handler captures removed by this change, whereas the current timing
runs use retained buffers. This is a concrete source distinction to test, not
proof of an unlocked performance gain. The
[report](../../../speed-bench/experiments/2026-11-05-residency-once/README.md)
owns the samples, variation, API evidence and limits. The selected package,
normal presets, installed frontends and main remain unchanged.
