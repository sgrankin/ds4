# Four output rows for attention output-B

**Status: in progress.** The experiment tests whether computing four output rows per Metal threadgroup improves exposed decode latency. Fixture validation, performance, quality and the adoption decision are pending.

## Context and target

The fixed baseline is the qualified prefill package at `02b7ae26`, with the four-row staging and unit-rescale candidate settings recorded in the [combined prefill report](../2026-11-04-combined-prefill/README.md). The selected one-hour slice is November 4, 2026, 13:24–14:24 Eastern. The [experiment queue](../../ideas/next-experiments.md) asks for the next exposed decode dependency.

The existing intrusive 32K decode anatomy profile attributes 3.691 ms cold and 3.739 ms warm per steady token, summed across attention output-B encoder intervals. An interval overlap analysis found 0.37% and 0.38% overlap with other encoder intervals, respectively, leaving 3.677 ms and 3.725 ms exclusive. Query intervals sum to 3.404 ms and 3.475 ms but overlap other encoder intervals by about 72%; router intervals overlap by about 99.99%. This ranking makes output-B a plausible exposed target. These instrumented interval totals are diagnostic observations, not predictions of end-to-end savings. The source profile is local at `.workspaces/hillclimb-20261104/05-attention-unit-rescale/timeline32k/1-candidate/decode-anatomy.json`; overlap details are in `.workspaces/hillclimb-20261104/08-decode-dependency/dependency-findings.json`.

## Change and procedure

Attention’s grouped low-rank output stage produces 8192 values. Output-B projects those values into the 5120-wide model representation, and the fused hyper-connection expansion writes four residual streams.

The candidate introduces the numeric opt-in `DS4_METAL_V41_OUTPUT_B_ROWS4` for the literal 8192-to-5120 Q8 attention output-B operation with hyper-connection expansion, one input row and no add input. It changes `NR0` from 2 to 4 while retaining the same `NSG`, per-lane K walk and BF16/hyper-connection reduction helper. The dispatch decreases from 2560 to 1280 groups; scratch per group increases from 256 to 512 bytes and each thread holds additional accumulators. The prototype revision is `6f83b139`; full-model candidate settings are pending.

First validate the bounded fixture, including the intended activation and fallback boundaries. Then compare the fixed baseline and candidate with five alternating pairs at both 8K and 32K contexts, using the existing pressure guard and profile-off timing. Retain every sample, report absolute latency and variation, and inspect paired changes and order effects. Record the exact commands, timed scope, cache conditions, excluded decode tokens, resource observations and failures with the results. Run practical quality checks appropriate to the arithmetic change. A separate activation profile will establish whether the candidate path ran; its instrumented timing will not substitute for the profile-off comparison.

## Findings and decision

The prototype built successfully for `ds4-bench`, `ds4-agent`, `ds4-server` and `metal-test`. The guarded fixture ran `./zig-out/bin/test-deepseek41-metal --output-b-rows4`. For both NSG 4 and NSG 2, none of 20,480 values changed, with maximum and RMS differences of zero. Generic controls and the add-input fallback were also exact, and kernel counters verified selection. Exactness here is the observed zero changed values; the fixture exit status alone permits finite drift. The NSG 2 run exercises that kernel specialization on this machine, not a physical two-Mac configuration. These are bounded fixture results, not a claim of equality for all inputs. Source review found no concrete correctness defect.

The fixture also timed five alternating pairs in one process, with 64 projections per sample; timings include encoding, submission and completion. The table retains every per-operation sample in milliseconds. Odd pairs ran baseline first and even pairs ran candidate first.

| Pair | Baseline (ms) | Candidate (ms) |
| --- | ---: | ---: |
| 1 | 0.159015625 | 0.062953125 |
| 2 | 0.052765625 | 0.068593750 |
| 3 | 0.053046875 | 0.052250000 |
| 4 | 0.051765625 | 0.052671875 |
| 5 | 0.051890625 | 0.055062500 |

The baseline mean was 0.073696875 ms (standard deviation 0.047697802 ms), and the candidate mean was 0.058306250 ms (standard deviation 0.007182833 ms). The first baseline sample dominates its mean; the remaining baseline samples range from 0.05177 to 0.05305 ms. The candidate was faster in only two of five pairs, so the nominal mean reduction does not establish a stable operation gain. Raw fixture output and operation summary are retained in `.workspaces/hillclimb-20261104/08-decode-dependency/`.

### 32K profile-off comparison

Five alternating pairs began at 13:31:51 Eastern. Each run prefills a fresh configured frontier, decodes 128 teacher tokens, restores the KV snapshot, appends 512 tokens and decodes another 128 teacher tokens. The expert cache survives the KV restore; the OS cache was not flushed. Phase timings exclude model loading and snapshot save/restore. Steady decode excludes the first token; full decode includes it. Seconds are derived from throughput rounded to two decimals. The following values are means ± sample standard deviations:

| Measure | Baseline | Candidate |
| --- | ---: | ---: |
| Fresh steady decode | 33.874 ± 0.874 tokens/s | 33.862 ± 0.705 tokens/s |
| Warm steady decode | 33.960 ± 0.595 tokens/s | 33.940 ± 0.527 tokens/s |
| Fresh full decode | 4.64869 ± 0.10667 s | 4.60861 ± 0.07501 s |
| Fresh first token | 896.916 ± 59.914 ms | 856.498 ± 37.146 ms |
| Warm full decode | 3.77405 ± 0.06518 s | 3.77651 ± 0.05702 s |
| Total inference | 62.52790 ± 1.77803 s | 62.81285 ± 1.14466 s |

Steady decode is essentially neutral in this comparison. The difference in mean fresh full decode is accounted for by the first token; neither that observation nor the small total-inference difference establishes a stable improvement or regression. Each cell below is baseline / candidate; odd pairs ran baseline first, even pairs candidate first.

| Pair | Fresh steady tokens/s | Warm steady tokens/s | Fresh decode s | First token ms | Warm decode s | Total inference s |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 34.81 / 34.58 | 34.91 / 34.85 | 4.59935 / 4.57143 | 949.760 / 898.561 | 3.67078 / 3.67816 | 59.43319 / 60.79569 |
| 2 | 33.38 / 33.28 | 33.79 / 33.49 | 4.62762 / 4.70935 | 822.909 / 892.445 | 3.79372 / 3.82546 | 63.51912 / 63.61947 |
| 3 | 33.52 / 33.45 | 33.73 / 33.80 | 4.64610 / 4.61428 | 857.032 / 817.356 | 3.79822 / 3.79372 | 63.19677 / 63.14319 |
| 4 | 32.88 / 33.32 | 33.31 / 33.71 | 4.82655 / 4.63936 | 963.737 / 827.836 | 3.84615 / 3.80048 | 63.79719 / 63.36445 |
| 5 | 34.78 / 34.68 | 34.06 / 33.85 | 4.54384 / 4.50863 | 891.140 / 846.290 | 3.76139 / 3.78474 | 62.69323 / 63.14147 |

The complete phase samples and paired analysis are retained in `.workspaces/hillclimb-20261104/08-decode-dependency/timing32k/timing-samples.csv` and `timing-summary.json`.

The 8K comparison is running. Practical quality and the separate activation profile remain pending; adoption or rejection is unresolved.
