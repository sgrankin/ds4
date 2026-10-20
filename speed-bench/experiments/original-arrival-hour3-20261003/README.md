# Original-expert arrival, hour 3

The completed first runs made decode slower than affinity substitution alone.
The admitted-ticket join follow-up removed most of that diagnostic overhead:
age 8 is timing-neutral, while age 1 remains slower. The worker sometimes
restored an original expert and incurred extra reads. Generated answers supplied
one completion improvement for age 1, at a small whole-task time cost.

The budget window was **2026-10-03 13:17:32–14:17:32 UTC**; model work finished
at 14:08 and results were recorded by 14:11, within 54 minutes. This is item 3 in
[next-experiments.md](../../ideas/next-experiments.md), following the
[resident comparison](../resident-functional-hour2-20261003/).
Initial adapter source: `0d7960d8a9acf144525a457eddfeabce092ccdb4`.
Commands/settings are retained in [build-plan.json](build-plan.json),
[strong-build-commands.json](strong-build-commands.json), and each arm's
`execution.json`; [run.py](run.py) runs the fixed teacher comparisons.

## Change and controls

Keep the original bounded one-reader/one-slab worker. Affinity chooses resident
substitutes; the worker independently reads one displaced original from the
model file. At one nonwaiting deadline before routed gate/up binding, a READY
original can replace that substitute. Otherwise the substitute stays selected;
late completed originals may serve future tokens until their retention age
expires. Ages **1, 8 and 80** use the same worker and differ by the compile-time
`DS4_OD_READY_AGE` setting. Original GPU coefficients remain unchanged throughout.

Weak policy: gap `.30`, mass `.12`, rank `24`, one substitution.
Strong policy: gap `.90`, mass `.25`, rank `64`, up to three substitutions,
total mass `.50`. Both use native weights and `replace`.
For strong routes, the reader targets **the first changed slot by slot index**;
it can restore only that one original. Other substitutes stay selected.
This does not assume router IDs are sorted by strength.

All runs use normal quantized mode, context `8192`, cache `64 GB`, chunk `512`, selected
tile/limit `64/1024`. The 24 fixed prompts have 32 teacher steps each, totaling
768 scored tokens. Native references and affinity brackets precede/follow the
deadline variants; the strong run reuses the unchanged native reference as
recorded in [reference-source.json](strong24/reference-source.json).
Every non-timing CSV field in the 768 native rows matches item 2; the strong
affinity control likewise matches item 2's `affinity-wide3` quality rows exactly.

## Completed first runs

Decode comparisons use the arithmetic mean of affinity-before/after, not a
confidence interval. NLL/ΔNLL/KL are token means in nats; ΔNLL is against native.
Native mean NLL is `1.60031932735` in both 24-case runs. Approximate outputs may
differ: no quality threshold or native-parity requirement rejected these arms.
These are teacher likelihood measurements, not generated task-success scores.

Weak, from [suite24](suite24/) and [results.csv](results.csv):

| Arm | Mean NLL | ΔNLL | KL | Decode s | vs affinity | Demand + staging GB |
|---|---:|---:|---:|---:|---:|---:|
| Affinity bracket mean | 1.596563 | −0.003756 | 0.008212 | 41.394 | — | 663.764 + 0 |
| Deadline age 80 | 1.596578 | −0.003742 | 0.008228 | 42.441 | +2.53% | 663.685 + 0.348 |
| Deadline age 8 | 1.599770 | −0.000550 | 0.008124 | 43.004 | +3.89% | 663.953 + 1.224 |
| Deadline age 1 | 1.598353 | −0.001966 | 0.008831 | 44.613 | +7.77% | 664.371 + 7.943 |

Strong, from the six completed arms in [strong24](strong24/):

| Arm | Mean NLL | ΔNLL | KL | Decode s | vs affinity | Demand + staging GB |
|---|---:|---:|---:|---:|---:|---:|
| Affinity bracket mean | 1.587720 | −0.012599 | 0.020562 | 35.467 | — | 580.256 + 0 |
| Deadline age 80 | 1.582674 | −0.017645 | 0.019975 | 39.080 | +10.19% | 580.426 + 0.279 |
| Deadline age 8 | 1.587378 | −0.012941 | 0.019554 | 39.646 | +11.78% | 580.545 + 1.184 |
| Deadline age 1 | 1.587912 | −0.012407 | 0.018433 | 42.010 | +18.45% | 580.943 + 8.092 |

Weak ages 80/8/1 recorded `1/10/61` deadline promotions and `10/19/0`
future uses; strong recorded `2/4/46` promotions and `2/20/0` future uses.
Shorter retention starts more reads; that is visible in the staging totals.

## Read accounting and join follow-up

Demand is the whole-job `PROBE_IO` requested-byte total. Staging is the sum of
**every** `ORIGINAL_DEADLINE` pread footer, including all 24 reader generations;
total logical reads are demand plus staging. Peak payload/host counters are
maxima, not sums. These are logical `pread` bytes, **not physical SSD traffic**.
Route counters are recorded before deadline promotion; planned skipped bytes
therefore do not establish net savings. [analyze.py](analyze.py) retains these
separate counters and writes the [decode tradeoff](pareto_decode.png) and
[read breakdown](io.png).

The first implementation checked/joined the ordinary selected-load pool before
the first gate/up whenever a replacement window existed, even when the original
reader could not admit a ticket. That can sacrifice ordinary resident/miss
overlap. The six-line `DS4_OD_JOIN_ONLY_ADMITTED` follow-up preserves the ordinary
path for declined tickets while retaining the mandatory checked join for
admitted ones; [build_admitted.py](build_admitted.py) records that isolated change.

Completed follow-up, from [admitted24](admitted24/) and [results.csv](results.csv):

| Arm | ΔNLL | KL | Decode s | vs affinity | Demand + staging GB |
|---|---:|---:|---:|---:|---:|
| Affinity bracket mean | −0.012599 | 0.020562 | 35.888364 | — | 580.256 + 0 |
| Admitted join, age 8 | −0.011822 | 0.020132 | 35.814620 | −0.2055% | 580.784 + 1.184 |
| Admitted join, age 1 | −0.013764 | 0.019086 | 38.544788 | +7.4019% | 580.694 + 8.132 |

Affinity controls span `35.771568–36.005159 s`, a `0.651%` range relative to
their mean. Age 8's `−0.2055%` is therefore **timing-neutral**. It recorded four
promotions and 20 future uses, with total logical reads `+0.2950%` versus affinity;
age 1 recorded 50 promotions, no future uses, and `+1.4769%` total reads.
The earlier age-8 `+11.78%` slowdown is absent after the six-line branch. These
are separate serial diagnostic runs, not a balanced causal proof of that
difference. Both variants remain approximate; their differing likelihood/KL
measurements are allowed and were not subjected to a parity gate.

[run_generation.py](run_generation.py) uses the same four item-2 prompt/gold
cases with greedy, no-thinking generation, seed `20261003`, and a 256-token cap.
[grade_generation.py](grade_generation.py) reuses the item-2 factual grader;
the CLI footer distinguishes stop-token completion, cap and interruption.
The completed generated results are in [generation/results.csv](generation/results.csv).

| Policy | Correct first answers | Correct completed tasks | Caps | Whole task seconds | Decode seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 2/4 | 2/4 | 2 | 86.60 | 48.60 |
| Strong affinity | 3/4 | 2/4 | 2 | 76.63 | 40.81 |
| Arrival age 1, admitted-only join | 3/4 | 3/4 | 1 | 77.76 | 42.65 |
| Arrival age 8, admitted-only join | 3/4 | 2/4 | 2 | 75.12 | 40.24 |

All resident variants gave the correct weighted median in their first object;
age1 also completed its answer at 247 tokens rather than hitting the 256 cap.
The endian task remained wrong and capped in all policies. Age1 took 1.48% more
whole-task time than affinity while completing one additional task; age8 took
1.97% less time with the same completion count. These four tasks and differing
generated histories are a small diagnostic screen, not a general accuracy bound
or paired KL measurement. No output was suppressed to manufacture a completion.

## Decision and retained evidence

Keep simple strong affinity as the latency candidate. Keep admitted-only age8
as an optional arrival point: it retains that decode speed within observed
variation, with slightly lower KL and 0.295% more logical reads. Keep age1 as a
quality/completion tradeoff point, with 7.4% extra fixed-input decode time. The
six-line branch removed the large unnecessary join penalty; more arrival tuning
does not justify extending this hour before broader task confirmation.

Completed 25 fixed-input configurations / 14,080 scored rows and 16 generated
tasks. The native controls reproduce all 768 reference vectors. No approximate
arm was rejected for numerical disagreement. All 2,627 pressure samples were
normal, swap did not grow, and peak wired plus compressed memory was 80.09 GiB.
All guarded jobs returned successfully. The first compile found an invalid
variadic use of the single-argument ds4_die; it was corrected directly and later
builds had no warnings. Earlier portable ownership checks were reused; no new
test framework was built for the macro branch.

Private binaries and raw vectors remain under the project and are untracked.
[source/runtime.patch](source/runtime.patch), the portable source copies,
[strong-policy.patch](strong-policy.patch) and
[join-only-admitted.patch](join-only-admitted.patch) describe the source changes.
The MAIN runtime and its defaults remain unchanged. Next is the separate
thinking-budget experiment, followed by confirmation on unused tasks.
