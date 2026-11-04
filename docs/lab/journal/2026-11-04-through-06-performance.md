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
