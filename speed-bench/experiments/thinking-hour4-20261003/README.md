# Thinking controls, hour 4

One-hour experiment: **2026-10-03 14:10:23–15:10:23 UTC**. Compare practical
first-attempt task success and work with one change at a time. All nine arms
completed within the hour; the pressure ledger ends at **14:44:15 UTC**.

## Tasks and method

Two existing independently graded tasks: half-open interval coalescing and
first-event-wins account ledger replay. Each asks the agent to read the entire
`config.json`, write `solution.py` once, then finish with a plain response.
The public sample contains no expected answer. The existing external grader
checks the first code on 161 cases, including empty/boundary/duplicate inputs,
with type-aware results and input/config nonmutation. It runs once with a
four-second owned timeout. Correct code can still fail the overall task if the
agent exhausts its budget or violates/completes neither the tool protocol nor
the final stop. These two calibration tasks do not establish generalization.

All nine arms use the same new frontend and native resident routing:
`native-before → budget256 → budget512 → budget1024 → penalty05 → penalty1 → native-after`.
The appended `nothink` arm uses the real `--nothink` CLI flag with WRITE budget
zero. `penalty01` uses lambda `.1` on all sampling steps. The weak penalty was
chosen after the initial seven results and is exploratory.
Each task has the same visible project/prompt paths across arms; the prior
solution is removed before each attempt. Private audits and first-code snapshots
are separate. No correction, retry or outcome-based task selection occurs.

Settings: effort 25 except the explicit no-thinking override, temperature `.6`,
top-p `.95`, min-p `0`, seed `20261002`,
context `8192`, cache `64 GB`, prefill chunk `512`, selected tile/limit
`64/1024`, scalar sampling, at most three rounds and **2048 global token
decisions**, including forced closes and sampled stop tokens. Source tasks,
settings, shader paths and per-arm changes are in [fixed-spec.json](fixed-spec.json).
[run.py](run.py) is launched only beneath the shared MAIN
[pressure guard](../../ideas/pressure_run.py). Each completed attempt retains
its argv, cwd, explicit environment, stdout/stderr, events and first grade.

## Two different interventions

**WRITE cutoff:** native sampling stays unchanged during READ and final response.
After actual successful READ and WRITE-prompt synchronization, count newly
generated thinking tokens for that assistant turn. At 256/512/1024, when still
inside thinking and at a safe parser/render boundary with two slots remaining,
force the validated `</think>` token `128822` once. This consumes one slot and
one ordinary evaluation, bypasses the sampler/RNG draw, and checks that native
session/parser/render state is outside thinking before continuing. An unsafe
eligible boundary refuses the attempt; it does not silently fall back.

**Hesitation penalty:** subtract lambda `.1`, `.5` or `1` from the mapped hesitation
word logits before normal sampling. Here `DS4_OVERTHINK_SCOPE=all` and
`DS4_OVERTHINK_AFTER=0`, matching the paper's fixed penalty at decoding steps
more closely than a WRITE-only adaptation. This can also alter legitimate words
or tool syntax; those failures stay in the results. Mapping count is retained
in helper stderr. This is the mechanism in
[Quantized Reasoning Models Think They Need to Think Longer, but They Do Not](https://arxiv.org/abs/2606.00206),
adapted by the existing [local policy](../ds41-ssd/OVERTHINK_LOGIT_POLICY.md).
The paper's dense AWQ/GPTQ/FlatQuant results do not establish a result for this
DS4.1 IQ2/Q2 MoE runtime. The WRITE cutoff is a separate protocol intervention.

## Recording and interpretation

`PT_LIGHT_AUDIT=1` is identical in baseline and candidates. It retains token
pieces, RNG/phase/parser events, prompt IDs and first code, while omitting
per-decision full-vocabulary raw copies. Empty `raw.fp32` files are not evidence
of vector identity and are excluded from tracking. No cross-prefix ΔNLL/KL is
computed from diverging generated histories.

[summarize.py](summarize.py) skips unfinished executions and produces
[results.csv](results.csv), [aggregate.csv](aggregate.csv) and
[tradeoffs.png](tradeoffs.png). Strict success requires completed compliant
protocol, an actual matching first-code snapshot and all first code checks.
Code correctness alone is reported separately, including capped attempts.
Sample/inside/round/forced counts and refusal reasons remain visible. Plot
labels include the observed task count so partial arms cannot look complete.
Helper wall time is instrumented and diagnostic, not a quiet speed measurement.

The small [build/source record](build-source/) retains ROOT's actual compile/link
result and binary digest, the matched frontend/public headers, and the
[three-header patch](build-source/headers.patch). Existing core/Metal objects
are linked as recorded; no runtime default changes are proposed here.

## Completed results

The native brackets took `160.163` and `159.599 s`, a `0.35%` spread. Their mean
`159.881 s` is the reference below. ROOT checked exact equality of every native
sample's `(round, token, RNG-before, RNG-after)` tuple: 1573 for intervals and
565 for ledger. [control-equality.json](control-equality.json) records that scope;
raw logits and full hidden state were not compared.

| Arm | Strict / code passes (of 2) | Total helper seconds | Change vs native mean | Decisions | Forced closes |
|---|---:|---:|---:|---:|---:|
| Native before | 2 / 2 | 160.163 | control | 2138 | 0 |
| WRITE 256 | 2 / 2 | 138.890 | −13.13% | 1730 | 2 |
| WRITE 512 | 2 / 2 | 140.981 | −11.82% | 1771 | 1 |
| WRITE 1024 | 2 / 2 | 152.602 | −4.55% | 1992 | 1 |
| Penalty .5 | 1 / 1 | 184.772 | +15.57% | 2613 | 0 |
| Penalty 1 | 1 / 1 | 180.768 | +13.06% | 2613 | 0 |
| Native after | 2 / 2 | 159.599 | control | 2138 | 0 |
| No thinking | 1 / 1 | 133.692 | −16.38% | 883 | 0 |
| Penalty .1 | 1 / 1 | 225.224 | +40.87% | 2613 | 0 |

All three penalties failed intervals: READ consumed 98 decisions, then
WRITE consumed the remaining 1950 and reached the global 2048 cap in stage 1
without a first solution. WRITE inside-thinking samples were 1849 at lambda
`.5` and all 1950 at lambda `.1` and `1`. All completed ledger with all 161
checks and a final stop. The outer runner's zero exit means it retained the
campaign, including these capped attempts.

The practical target is **at least 20% less time while both tasks still work**:
at most `127.905 s` relative to these native brackets. No completed arm meets it.
WRITE 256 is the best completed working point. These are two calibration tasks,
one seed and instrumented serial timing. Carry WRITE 256 into the unused item-5
tasks, including the previously tested strong affinity policy as a combination.
The combination is a new experiment; no default change follows from this curve.

## Why a WRITE cutoff gives a modest gain

Both tasks completed their tool protocol and passed all 161 first-code checks
in both arms. The cutoff fired **once per task** at WRITE thinking count 256,
with token `128822` and unchanged pre/post RNG. The READ rounds were identical
in sampled/inside counts. Later arms, including failures, remain in the same
summary and final decision.

| Task | Native READ/WRITE/final decisions | Cutoff 256 READ/WRITE/final | Native / cutoff code bytes |
|---|---:|---:|---:|
| Intervals | 98 / 1424 / 51 | 98 / 570 / 481 | 966 / 1082 |
| Ledger | 93 / 429 / 43 | 93 / 428 / 60 | 515 / 530 |

Intervals saved 854 WRITE decisions but added 430 in the unrestricted final
round, which contained 444 inside-thinking samples instead of one. Its new
code also had more explicit type/config checks. Ledger was already near the
cutoff: WRITE inside samples changed only `259→257`, while its final explanation
grew; total decisions increased `565→581`. Thus a WRITE cutoff does not bound
whole-task thinking or final response length. Inside-sample counts include a
sampled closing marker and are not the policy's count of reasoning text tokens.

Combined decisions changed `2138→1730` (−19.08%); time fell 13.13% relative to
the mean native brackets. This includes the slower ledger case.

## No-thinking failure and startup cache difference

No thinking really stayed outside thinking in every sample: intervals
READ/WRITE/final decisions were `80/299/161`, and ledger `80/181/82`.
Intervals took `92.621 s` despite only 540 decisions. Both native brackets
started with a **system-prompt KV hit at 1913 tokens**. No-thinking changed that
prompt to 1888 tokens: intervals missed the cache and paid `50.881 s` initial
prefill, then stored it; ledger subsequently hit the 1888-token cache.
Intervals' later round-prefills sum `16.810/17.438 s` in the native brackets,
versus `9.646 s` without thinking. Thus no-thinking's actual whole-task time
includes a different startup/cache condition. The table and plot retain that
observed time; they do not estimate a warmed no-thinking latency or attribute
its modest wall gain entirely to unavoidable prefill work.

Lambda `.1` switched back to the 1913-token effort-25 prompt and paid a
`50.763 s` initial prefill/cache miss on intervals; ledger then hit that cache.
Its wall penalty includes this startup difference. Its quality failure remains
clear: the WRITE round still exhausted all 1950 remaining decisions without code.

It completed the tool protocol but failed unit 16. That input contains only
reversed intervals and `[8,8]`; the required answer is `[]`. Its first code
filters `start > end`, retaining the empty interval and returning `[[8,8]]`.
Ledger passed all 161 checks. The no-thinking point is therefore faster but
does not meet the working-task target; no corrected attempt replaces it.

## Pressure and next step

Both guards completed with child return code zero. Their actual logs contain
**1431 normal samples, zero bad samples**, peak wired-plus-compressor
**83.731 GiB**, and swap `0.024600→0.024600 GiB` with **zero growth**.
[pressure-summary.json](pressure-summary.json) retains each guard's counts,
peaks, time range and result; no new pressure measurement was taken.

The same runner can consume the two unused item-5 tasks with `--tasks-file`,
their existing `--grader`, and `--output` pointing at the confirmation folder.
Visible projects, private audits, summaries and plots then stay in that folder.
Named arm `budget256-affinity` combines WRITE 256 with fixed affinity settings
gap `.90`, mass `.25`, total mass `.50`, rank `64`, three substitutions,
native coefficients and replace policy. It does not use `--nothink`; all
confirmation attempts and failures will remain visible.
