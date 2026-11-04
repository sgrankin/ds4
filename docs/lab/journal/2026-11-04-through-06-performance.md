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
