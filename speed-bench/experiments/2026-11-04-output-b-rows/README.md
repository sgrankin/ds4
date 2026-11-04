# Four output rows for attention output-B

**Decision: remove the opt-in four-row variant.** The measured end-to-end comparison did not establish a benefit, and separate dependency profiles did not show the intended output-B reduction or a useful newly exposed bottleneck. This decision does not rest on a small, noisy slowdown.

## Context and target

The fixed baseline is the qualified prefill package at `02b7ae26`, with the four-row staging and unit-rescale candidate settings recorded in the [combined prefill report](../2026-11-04-combined-prefill/README.md). The selected one-hour slice is November 4, 2026, 13:24–14:24 Eastern. The [experiment queue](../../ideas/next-experiments.md) asks for the next exposed decode dependency.

The existing intrusive 32K decode anatomy profile attributes 3.691 ms cold and 3.739 ms warm per steady token, summed across attention output-B encoder intervals. An interval overlap analysis found 0.37% and 0.38% overlap with other encoder intervals, respectively, leaving 3.677 ms and 3.725 ms exclusive. Query intervals sum to 3.404 ms and 3.475 ms but overlap other encoder intervals by about 72%; router intervals overlap by about 99.99%. This ranking makes output-B a plausible exposed target. These instrumented interval totals are diagnostic observations, not predictions of end-to-end savings. The source profile is local at `.workspaces/hillclimb-20261104/05-attention-unit-rescale/timeline32k/1-candidate/decode-anatomy.json`; overlap details are in `.workspaces/hillclimb-20261104/08-decode-dependency/dependency-findings.json`.

## Change and procedure

Attention’s grouped low-rank output stage produces 8192 values. Output-B projects those values into the 5120-wide model representation, and the fused hyper-connection expansion writes four residual streams.

The candidate introduces the numeric opt-in `DS4_METAL_V41_OUTPUT_B_ROWS4` for the literal 8192-to-5120 Q8 attention output-B operation with hyper-connection expansion, one input row and no add input. It changes `NR0` from 2 to 4 while retaining the same `NSG`, per-lane K walk and BF16/hyper-connection reduction helper. The dispatch decreases from 2560 to 1280 groups; scratch per group increases from 256 to 512 bytes and each thread holds additional accumulators. The prototype revision is `6f83b139`; `d4578e3e` adds the matched-warmup fixture without changing runtime behavior. The comparisons ran on an M5 Max with 128 GiB memory, using the same candidate executable and matching Metal source from `6f83b139` for both arms: `DS4_METAL_V41_OUTPUT_B_ROWS4=0` for the control and `1` for the candidate. `02b7ae26` identifies the qualified package from which the prototype was built. Exact run environments and arguments are retained in each local timing arm. The workload used `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, Metal SSD streaming with a 64 GB expert cache, `speed-bench/promessi_sposi.txt`, 8192-token prefill chunks and teacher-forced decode.

The bounded fixture preceded five alternating profile-off pairs at each frontier, under the existing pressure guard. For reproduction after removal, check out the measured revision and run from its repository root so the executable and Metal source match. The driver sets the flag to `0` and `1` for the two arms, respectively; use `8192` in place of `32768` for the other frontier:

```sh
nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/output-b-reproduction timing --flag DS4_METAL_V41_OUTPUT_B_ROWS4 --frontier 32768
```

Run this driver under the existing pressure guard. The runs retained all samples and used separate instrumented profiles to check activation and dependencies.

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

### 8K profile-off comparison

The 8K comparison used the same phase sequence, token counts, cache conditions and alternating order as 32K. Mean ± sample standard deviation for fresh steady decode was 34.026 ± 0.136 versus 32.750 ± 4.748 tokens/s; warm steady decode was 32.904 ± 1.049 versus 32.950 ± 2.054 tokens/s. Fresh full decode was 4.661 ± 0.098 versus 5.306 ± 1.818 s; warm full decode was 3.901 ± 0.129 versus 3.911 ± 0.259 s. Total inference was 36.043 ± 0.957 versus 36.892 ± 4.506 s. Baseline precedes candidate in each comparison. Each cell is baseline / candidate; odd pairs ran baseline first, even pairs candidate first. All samples are retained, including the last candidate run.

| Pair | Fresh prefill s | Fresh steady tokens/s | Warm steady tokens/s | Fresh decode s | First token ms | Warm decode s | Total inference s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 20.05926 / 19.06446 | 34.02 / 34.27 | 34.08 / 33.70 | 4.60763 / 4.54545 | 875.284 / 839.584 | 3.76028 / 3.80160 | 35.79302 / 34.71537 |
| 2 | 19.24360 / 19.17064 | 34.06 / 34.20 | 33.50 / 33.43 | 4.55840 / 4.56003 | 828.406 / 845.637 | 3.82432 / 3.83234 | 34.98899 / 34.90667 |
| 3 | 19.50941 / 19.35774 | 33.80 / 35.39 | 33.28 / 34.03 | 4.61095 / 4.44753 | 852.916 / 859.634 | 3.84847 / 3.81066 | 35.34847 / 34.27133 |
| 4 | 20.80402 / 20.47642 | 34.16 / 35.56 | 31.50 / 34.27 | 4.79042 / 4.42142 | 1071.867 / 850.511 | 4.06866 / 3.74050 | 37.09416 / 35.66262 |
| 5 | 20.53699 / 24.19445 | 34.09 / 24.33 | 32.16 / 29.32 | 4.73899 / 8.55615 | 1013.045 / 3333.653 | 4.00125 / 4.37009 | 36.99272 / 44.90185 |

The final candidate run slowed across phases, making arm means uncertain. The guard remained at pressure level 1 with no bad samples. Swap was about 26.012 GiB through pairs 1–3 and about 26.015 GiB before the last candidate, then peaked at 30.9846 GiB and ended at 29.6699 GiB during that run. This excursion correlates with the slowdown; it does not establish its cause. Swap during 32K ranged from 26.0002 to 26.0119 GiB. A later process check showed normal user applications including active WebKit and WindowServer, with no other heavy model; `pmset` reported no thermal or performance warning. Neither observation rules out transient interference. Complete phase samples and paired analysis are retained in `timing8k/timing-samples.csv` and `timing8k/timing-summary.json` under the local raw root.

### Matched-warmup operation fixture

Revision `d4578e3e` changes the fixture warmup to 64 calls, matching the 64 timed calls; the runtime is unchanged. Both NSG specializations and the fallback again produced exact observed values. Five alternating pairs gave baseline 0.062975 ± 0.007039 ms and candidate 0.060550 ± 0.002701 ms per operation, with the candidate faster in three pairs. The first control remained higher than later controls. This rerun and the original fixture do not establish a strong operation speedup.

| Pair | Baseline (ms) | Candidate (ms) |
| --- | ---: | ---: |
| 1 | 0.075140625 | 0.057734375 |
| 2 | 0.059437500 | 0.059203125 |
| 3 | 0.057906250 | 0.063546875 |
| 4 | 0.062875000 | 0.058921875 |
| 5 | 0.059515625 | 0.063343750 |

### Dependency profiles and quality

One intrusive 32K profile per arm, baseline followed by candidate, confirmed 5120 output-B calls per phase and the expected reduction from 2560 to 1280 groups, with the same 32 × 4 threads. Encoder and command-buffer counts and recorded routing and load counts matched. Output-B summed time per steady token was 3.651 → 3.683 ms fresh and 3.687 → 3.704 ms warm; exclusive coverage was 3.639 → 3.671 ms and 3.675 → 3.689 ms, respectively. Output-B remained almost entirely exposed, with roughly 0.3–0.4% overlap. Its surrounding intervals did not identify a useful newly exposed bottleneck. These single, heavily instrumented profiles establish neither a causal regression nor a precise critical path. Full interval details and limitations are in the local `profile-dependency-comparison.txt` and `.json` files.

At 32K, all 256 full-vocabulary teacher-logit rows on identical histories were byte identical, and all 256 post-evaluation logit hashes matched. Delta-NLL and KL were zero on these rows. This bounded comparison does not establish identical behavior on every workload or substitute for a practical task check. At 8K, all 256 full-vocabulary rows were also byte identical and all 256 post-evaluation hashes matched. No practical sampled task was run: the geometry change was exact across these 512 paired positions and the candidate was removed. Details are in `quality32k/quality-summary.json` and `quality8k/quality-summary.json` under the local raw root.
