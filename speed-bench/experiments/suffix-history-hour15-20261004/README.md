# Retained-history benefit and harm

## Result and decision

Item15 ran14:46–15:09 UTC,2026-10-04;10 minutes of earlier preparation are
charged against its15:36 deadline. **Retained prefix influence helped one
ordinary calculation and harmed one superseded-rule calculation. It did not
recover an entirely omitted rule. Actual archive retrieval recovered both
current rules.** These are small deterministic reasoning-disabled probes,
not a reliable-memory claim or an accuracy estimate.

The first repeated-filing-prose cohort answered UNKNOWN in every fresh and
archive-control condition; ordinary reuse answered the unrounded459. None of
the9 primary conditions was correct. We retained that failed calibration.
A separate cohort changed only the repeated notes to coherent shipping-review
prose. Ordinary reuse and obsolete fresh began calculating but hit16-token
caps. All other conditions abstained. We repeated only those two shortened
pairs with the cap raised to64; no prompt or cache change accompanied that
follow-up.

| Case, coherent notes /64 cap | Fresh shortened | Original-backed reuse | Practical observation |
| --- | --- | --- | --- |
| Current rule retained |UNKNOWN|Correct arithmetic ending456|Reuse helped this ordinary case.|
| New rule supersedes old |Correct arithmetic ending450|UNKNOWN|Reuse harmed this override case.|
| Current rule omitted,16 cap |UNKNOWN/EOS|UNKNOWN/EOS|No demonstrated latent recall; these completed answers did not need a larger cap.|

The two correct arithmetic outputs included explanations, contrary to the
integer-only instruction. Strict grades remain0/4 in the64-cap cohort; manual
arithmetic review is reported separately. None of these runs produced the
obsolete456 answer under the new10 rule. Harm here means abstaining where the
fresh context produced the correct answer, not copying a stale answer.

Original same-context mean delta-NLL/KL were+.4251/.2031 ordinary,
−1.5789/.3888 omitted rule, and+1.0886/.1453 superseded rule. Coherent notes
gave−.4730/.1833,−1.8448/1.2412, and+.1471/.1476 respectively. Each gold
continuation is only one numeric token plus EOS. Large likelihood improvements
did not make the omitted-rule answer correct; averaging the families would hide
the practical differences. These measured points stay visible rather than
being rejected for differing from fresh execution.

The original/coherent reports and raw scoring tables are
[original-results.md](original-results.md), [coherent-results.md](coherent-results.md),
[first-analysis](first-analysis/core-analysis.json) and
[coherent-analysis](coherent-analysis/core-analysis.json). The bounded-cap
follow-up is [cap64-results.md](cap64-results.md). Core edit times were roughly
13.4–13.7 seconds fresh versus5.3–5.5 seconds reuse, excluding36–40 seconds of
source setup, capture and answer generation. This is diagnostic edit timing,
not whole-agent latency.

## Actual tool retrieval

Both cases made an actual read(path,whole=true,raw=true) of their permanent
policy.json. With the original32 final-token cap both answers were incomplete.
Changing only that cap to128 produced completed, correct arithmetic456/450 in
25.72/25.82 seconds, including tool generation, file read and final response.
Both included explanations, so the original strict format grades remain
false. Final responses used55/49 tokens. These fresh controls have suffix
reuseoff and a shorter context; no cross-context KL or reuse speedup is claimed.
See [HTTP-results.md](HTTP-results.md) and the preserved32/128 wire records.

Source helper is the unchanged item14 binary84aa5c3b…, linked against current
MAIN; server runtime3fb450f7. Exact commands/settings/hash are in execution
records and guard command.json. All five guards stayed at normal pressure with
zero swap growth, including the intentionally failing HTTP strict-grade runs.
The next item is a request-scoped thinking cutoff tested in an actual post-read
write role; do not spend more of this hour rewriting this toy fixture.

## Experiment design and reproduction

Ten minutes of earlier source-only design preparation are charged to item15's hour.

Three preregistered cases use the same new shipment: `18 * 26 - 9 = 459`. Rounding down to a multiple of8 gives456; the superseding multiple10 gives450. The question and derived answer never appear in a previous assistant response: the unchanged item14 oracle captures after source prefill and generates zero source answers.

| Case | Edited visible record | Applicable rule |
| --- | --- | --- |
| ordinary-cleanup | Earlier draft prose removed; current rule retained |8|
| helpful-history | Still-valid rule omitted and moved into real policy.json |8|
| obsolete-history | Old rule replaced by a current visible/archive rule |10|

Each case compares fresh shortened context with original-backed suffix reuse on exactly the same edited token history. A second invocation reads the actual generic `policy.json` and inserts its current rule into a separate visible context. Only that invocation's fresh arm is the explicit archive control; its extra reuse arm is retained as ancillary output because the existing helper always produces both modes. There is no cross-context KL comparison.

The filler refers to the previously approved shipping review and next dispatch, giving the unchanged suffix a semantic link to the earlier decision without stating a numeric rule or answer. The caller uses it to pad A, B and C, so placing a rule there would contaminate the fresh control. The common geometry is prefix1024, source edit256, destination edit128, unchanged512 and fresh128, with the whole final query required fresh. Actual token IDs and geometry are recorded by the existing helper.

The helpful-history fresh arm is intentionally missing a visible rounding rule. UNKNOWN is a meaningful abstention; an incorrect numeric answer there is not by itself evidence of a general model accuracy regression. Useful implicit carryover requires the reused answer to follow the archived valid rule. The obsolete case tests whether that carryover incorrectly resists a new authoritative rule.

The archive file contains the rule, not the computed gold answer. The driver performs the JSON-file read and records its time; this is an explicit summary control, not proof that the model called a retrieval tool. Practical tool retrieval and end-to-end HTTP latency remain separate measurements. Core timings exclude previous-source setup/capture; caps, answers and failures remain in original helper outputs.

ROOT command inside the existing MAIN pressure guard:

```sh
python3 -B speed-bench/experiments/suffix-history-hour15-20261004/run.py --output speed-bench/experiments/suffix-history-hour15-20261004/core
```

The driver clears inherited DS4/PT variables and reapplies the same item14 shader and selected64 configuration. No runtime changes, new helper build or model execution were performed during preparation.

`http_launch.py` separately starts one guarded server with fresh cache/store namespaces and suffix reuse disabled. `http_run.py` asks for an actual `read(path, whole=true, raw=true)` of helpful-history or obsolete-history's generic policy.json, then calculates the new shipment total. It independently grades the final integer456/450 and requires one successful file read, completed public response status and no tool errors. Every request cap, response status, incomplete detail, actual usage, tool argument/result and read time is retained. HTTP does not expose an observed EOS token, so no stop-token claim is inferred from completion status. There is no source warmup or earlier fact answer.

ROOT's separate guarded inner command after the core job releases:

```sh
python3 -B speed-bench/experiments/suffix-history-hour15-20261004/http_launch.py
```
