# Item 5: new confirmation inputs

The one-hour budget began **2026-10-03 14:17:23 UTC** and ends at
**15:17:23 UTC**. Source/fixture preparation overlapped item4 and counts in
this window. Model measurement began at **14:46:35 UTC**; that start does
not reset the hour. Teacher, practical and generation jobs ran sequentially.
All measurements and this review were complete by **15:14 UTC**, inside the
budget: six teacher configurations (3,072 scored rows), eight practical
attempts and twelve generated-answer attempts.

Sixteen newly authored teacher cases span Python/data interpretation, exact
arithmetic, and supplied fictional records. Four separate generation cases
cover a bitmask, a superseded access record, a recurrence, and a stack program.
These are new literal tasks, not a reseeding or copying of the earlier24 cases.
They are fixed before any model result. Gold explanations are in each gold JSON;
`fixtures/source.py` contains the complete authored content.

Input format matches item2/item3: each TSV row is `case_id<TAB>absolute prompt
path<TAB>absolute continuation path`, without a header. Gold JSON contains a
`gold` object, as expected by the existing required-field grader. Teacher
continuations start with that JSON and then a derivation; the helper scores the
first32 native tokens. No tokenization has been performed during preparation.

The runner uses the `original-deadline-admitted-direct-v1` helper build
(fast strong-policy join only), item3 model settings and fixed strong resident
policy (.90 gap/.25 mass/rank64/three substitutions/.50 total mass), while
reversing candidate order: native-before → deadline-age8 → affinity-strong →
native-after. ROOT must run it within the existing MAIN pressure guard.

Example guarded child argv:

```
python3 speed-bench/experiments/confirmation-hour5-20261003/run.py --manifest /Volumes/Models/ds4/speed-bench/experiments/confirmation-hour5-20261003/teacher.tsv --cases 16
```

For four generated answers, use `--generate --manifest .../generation.tsv
--cases 4 --tokens 128 --group generation --cli MATCHING_CLI`. The CLI path is
an explicitly compiled TTL8 variant, used for all arms. Native and affinity
controls set `DS4_METAL_V41_ORIGINAL_DEADLINE=0`; only deadline arms enable it.
This preparation does not build or run a native CLI.
The copied item2 grader changes only its case-name prefix to `confirm_g`; it
retains factual-field, process status, stop-token and cap grading semantics.

Sources reused: item2 `run.py` and `grade_generation.py` for TSV/gold/generation
format; item3 `run.py` for the deadline helper and policy controls. No model,
tokenizer, compiler or native helper was executed while authoring this packet.

`analyze.py` reuses the item3 CSV/log readers, keeps demand/staging/total IO
separate, and plots decode time against teacher-forced delta-NLL. It does not
read reference or candidate binary arrays. ROOT can render with
`.workspaces/bench-plots-venv/bin/python .../analyze.py`; the matplotlib cache
is placed under the selected output directory. Generated answers are graded
separately by `grade_generation.py`.

After every planned generation arm has completed, ROOT can run:

```
.workspaces/bench-plots-venv/bin/python speed-bench/experiments/confirmation-hour5-20261003/grade_generation.py
```

This writes generation `results.csv`/`results.json` and `task_quality.png`/SVG.
Caps, wrong factual fields and missing completion evidence remain failures;
the chart counts completed passes and uses summed process wall time.

## Completed teacher comparison

The four arms completed all16 cases ×32 fixed teacher tokens (512 rows each).
The helper reports all512 native-after rows bit-equal to native-before, with
zero delta-NLL, KL and argmax flips. CSV/log analysis completed with no skipped
arms; it did not independently read binary arrays.

| Arm | Prefill s | Decode s | Model s | Process wall s | Mean ΔNLL nats/token | Mean KL nats/token | Flips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native before | 34.176 | 30.250 | 64.425 | 69.352 | 0 | 0 | 0 |
| Age8 | 33.168 | 24.613 | 57.780 | 60.365 | −.024766 | .019266 | 14 |
| Strong affinity | 35.118 | 24.587 | 59.705 | 62.314 | −.023243 | .018666 | 18 |
| Native after | 34.002 | 30.274 | 64.276 | 66.835 | 0 | 0 | 0 |

Strong affinity reduced decode time18.753% versus the native bracket mean;
age8 reduced it18.668%. Age8 was **.105% slower in decode than affinity**.
Its3.127% lower full process time coincided with5.553% lower prefill time.
Native before/after process time itself moved3.630%, while native decode moved
only.081%, prefill−.509% and model total−.232%. This run therefore supports the
affinity decode benefit, but does not establish an extra staging latency win.

| IO / arrival metric | Strong affinity | Age8 |
| --- | ---: | ---: |
| Demand requested decimal GB | 252.275835 | 252.166349 |
| Staging requested decimal GB | 0 | .766403 |
| Total requested decimal GB | 252.275835 | 252.932751 |
| Staging starts / completions | — | 77 / 77 |
| Deadline promotions / misses | — | 4 / 73 |
| Future-use events / expired | — | 10 / 47 |

Native demand was341.785682GB in both controls. Affinity saved26.189% of those
logical requested bytes. Age8 demand alone fell a further.0434% versus
affinity, but staging made its total **.2604% higher**. The16 reader footers
sum to231 staging preads and766,402,560 requested/returned bytes, with zero
failed jobs or overflow. Promotions and future uses are distinct reported
event counters, not an inferred count of unique useful jobs.

Worst case mean delta-NLL was+.050834 for affinity's two-color problem and
+.072829 for age8's boolean filter. Largest individual token delta-NLL was
1.946397 and2.235688 respectively; maximum row KL was1.023678 and.894978.
Lower average NLL and these tails describe fixed-history likelihood changes;
they do not grade sampled answers or establish broad task accuracy.

The guard completed0 with251 normal pressure samples, zero bad samples and
zero swap growth; peak wired plus compressor memory was79.1495GiB. Timings
are diagnostic single-run comparisons on one authored16-case packet.

### Later substitution-count refinements

All six teacher configurations completed the same16 cases ×32 tokens:3072
scored rows. The count2 and count1 points were measured later, in that order,
without new native brackets. Gap.90/mass.25/rank64/total mass.50 were retained.

| Maximum substitutions | Decode s | Prefill s | Model s | Process wall s | Demand GB | Mean ΔNLL | Mean KL | Flips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 24.587012 | 35.117835 | 59.704847 | 62.313938 | 252.275834880 | -0.023243094 | 0.018666100 | 18 |
| 2 | 25.254832 | 32.101444 | 57.356276 | 59.814927 | 272.829358080 | -0.013259638 | 0.012503417 | 11 |
| 1 | 27.601768 | 33.603268 | 61.205036 | 63.720901 | 303.186862080 | +0.003165644 | 0.009795169 | 12 |

Count1/2/3 decode time was8.791%/16.546%/18.753% below the earlier native
bracket mean; demand reads fell11.293%/20.175%/26.189%. Count2 decoded2.716%
slower than count3; count1 was9.293% slower than count2 and12.262% slower than
count3. Count2's lower model/wall totals coincided with8.589% lower prefill
than count3, so they do not establish a count2 whole-process latency advantage.
These late timing points are diagnostic rather than repeated balanced controls.

| Maximum substitutions | Worst case mean ΔNLL | Case | Maximum token ΔNLL | Maximum row KL |
| --- | ---: | --- | ---: | ---: |
| 1 | +.076231313 | boolean filter | 1.223090210 | .398070787 |
| 2 | +.087482053 | gcd/lcm | 1.143025908 | .766189261 |
| 3 | +.050834399 | two-color problem | 1.946397480 | 1.023677641 |

Lower average KL is not uniformly better case likelihood, nor are flips
monotonic: count1/2/3 had12/11/18. All512 rows and16 case summaries per arm
remain represented in `results.csv`, `per_case.csv` and the decode/IO plots.

## Completed four-case generation comparison

All12 generation processes returned0 and emitted a non-capped stop-token
footer. Each arm completed **one correct answer out of four**, with zero caps.
Parsed answer fields were the same across all three arms:

| Case | Actual answer | Independently checked gold | Result in every arm |
| --- | --- | --- | --- |
| Bitmask | decimal22, hex `16` | decimal36, hex `24` | Wrong |
| Current access record | current `8736`, obsolete `5127` | Same | Pass |
| Recurrence | a6=17, first difference3 | a6=20, first difference3 | Wrong |
| Stack | final stack `[18]`, popped `[9,6]` | stack `[10]`, popped `[9,6]` | Wrong |

| Arm | Completed passes | Caps | Tokens | Total prefill s | Total decode s | Total process wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 1/4 | 0 | 55 | 16.711 | 5.157 | 29.127 |
| Age8 | 1/4 | 0 | 58 | 15.391 | 3.732 | 23.717 |
| Strong affinity | 1/4 | 0 | 58 | 15.015 | 3.589 | 23.257 |

The first native bitmask process took9.189s, versus5.352s for affinity and
5.534s for age8. There is no repeated native generation bracket, so total
wall differences are sensitive to first-process startup and run order. These
short jobs provide diagnostic timing and an unchanged1/4 score, not a stable
end-to-end speed or accuracy estimate. Age8 was slower than affinity in both
decode and total wall time. The three baseline failures remain failures;
neither approximation gains credit for reproducing an incorrect answer.

`generation/results.csv` retains every factual result, completion state,
stop reason, cap and timing; `generation/task_quality.png`/SVG show the measured
task count against total wall time. Teacher refinements are recorded separately
from the original four-arm comparison and the generated-task scores.

## Final pressure and decision

All five guards completed with child returncode0: teacher, practical,
generation, count2 and count1. `pressure-summary.json` retains each guard's
sample count, peak wired plus compressor memory, initial/max swap and growth.

Across914 samples, all914 were normal, with0 bad samples and zero swap growth.
The maximum wired plus compressor peak was79.149490GiB. Existing swap was.024599609375GiB throughout.

| Guard | Normal samples | Peak wired + compressor GiB | Swap growth GiB |
| --- | ---: | ---: | ---: |
| teacher | 251 | 79.149490 | 0.000000 |
| practical | 469 | 78.799881 | 0.000000 |
| generation | 74 | 73.520874 | 0.000000 |
| affinity-two | 58 | 78.723877 | 0.000000 |
| affinity-one | 62 | 78.709152 | 0.000000 |

Strong affinity remains a useful **opt-in decode/IO tradeoff** on this packet.
Count3 is the fastest decode point; count2 exchanges2.716% slower decode and
8.147% more IO for lower mean KL and fewer flips. Count1 further reduces mean
KL but gives up much of the decode/IO benefit. No setting improves every tail.
Age8 adds staging bytes without an extra decode win over count3 affinity.
The four plain generated tasks remain1/4 correct in every arm, so this result
does not establish a task-accuracy gain or justify default promotion. Separate
practical-task results retain their own first-attempt grading and cap evidence.

## Completed practical combination

Two unused code/tool tasks (stable left join and sliding-window argmax), one
fixed seed, and 161 independent checks per first implementation: every arm
passed both tasks and completed the tool/final-stop protocol.

| Policy | Total helper wall time, two tasks | Strict passes |
| --- | ---: | ---: |
| Native before | 133.825 s | 2/2 |
| Native after | 129.548 s | 2/2 |
| WRITE 256 | 120.054 s | 2/2 |
| WRITE 256 + strong resident affinity | 100.218 s | 2/2 |

The combination saves **23.90%** against the native-bracket mean, **22.64%**
against the faster native control, and **16.52%** against WRITE 256 alone.
It uses neither suffix reuse nor original staging. These are two generated
solutions per policy, not 322 independent model attempts.
[Practical evidence and grading](practical-results/README.md) and the
[task tradeoff plot](practical-results/tradeoffs.png) retain all four arms.
