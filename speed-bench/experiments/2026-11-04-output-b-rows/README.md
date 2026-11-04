# Four output rows for attention output-B

**Status: in progress.** The experiment tests whether computing four output rows per Metal threadgroup improves exposed decode latency. Fixture validation, performance, quality and the adoption decision are pending.

## Context and target

The fixed baseline is the qualified prefill package at `02b7ae26`, with the four-row staging and unit-rescale candidate settings recorded in the [combined prefill report](../2026-11-04-combined-prefill/README.md). The selected one-hour slice is November 4, 2026, 13:24–14:24 Eastern. The [experiment queue](../../ideas/next-experiments.md) asks for the next exposed decode dependency.

The existing intrusive 32K decode anatomy profile attributes 3.691 ms cold and 3.739 ms warm per steady token, summed across attention output-B encoder intervals. An interval overlap analysis found 0.37% and 0.38% overlap with other encoder intervals, respectively, leaving 3.677 ms and 3.725 ms exclusive. Query intervals sum to 3.404 ms and 3.475 ms but overlap other encoder intervals by about 72%; router intervals overlap by about 99.99%. This ranking makes output-B a plausible exposed target. These instrumented interval totals are diagnostic observations, not predictions of end-to-end savings. The source profile is local at `.workspaces/hillclimb-20261104/05-attention-unit-rescale/timeline32k/1-candidate/decode-anatomy.json`; overlap details are to be retained under `.workspaces/hillclimb-20261104/08-decode-dependency/`.

## Change and procedure

The candidate introduces the numeric opt-in `DS4_METAL_V41_OUTPUT_B_ROWS4` for the literal 8192-to-5120 Q8 attention output-B operation with hyper-connection expansion, one input row and no add input. It changes `NR0` from 2 to 4 while retaining the same `NSG`, per-lane K walk and BF16/hyper-connection reduction helper. The dispatch decreases from 2560 to 1280 groups; scratch per group increases from 256 to 512 bytes and each thread holds additional accumulators. Candidate revision and exact settings are pending.

First validate the bounded fixture, including the intended activation and fallback boundaries. Then compare the fixed baseline and candidate with five alternating pairs at both 8K and 32K contexts, using the existing pressure guard and profile-off timing. Retain every sample, report absolute latency and variation, and inspect paired changes and order effects. Record the exact commands, timed scope, cache conditions, excluded decode tokens, resource observations and failures with the results. Run practical quality checks appropriate to the arithmetic change. A separate activation profile will establish whether the candidate path ran; its instrumented timing will not substitute for the profile-off comparison.

## Findings and decision

Pending fixture, timing, quality and activation evidence. Adoption or rejection remains unresolved.
