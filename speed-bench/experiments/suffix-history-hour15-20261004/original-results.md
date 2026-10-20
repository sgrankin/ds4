# Original retained-history cohort

The nine scored conditions produced **0/9 correct arithmetic answers**. All completed through EOS within the 16-token cap. This cohort demonstrates neither a useful retained-memory answer nor an obsolete-answer error. These first outputs remain unchanged; the separate notes-only cohort is a new experiment.

| Case (gold) | Fresh shortened | Reuse shortened | Fresh explicit archive rule | Reuse ΔNLL | KL(fresh ∥ reuse) |
|---|---|---|---|---:|---:|
| Ordinary cleanup (456) | UNKNOWN | 459 | UNKNOWN | +0.425071 | 0.203059 |
| Helpful history (456) | UNKNOWN | UNKNOWN | UNKNOWN | −1.578906 | 0.388839 |
| Obsolete history (450) | UNKNOWN | UNKNOWN | UNKNOWN | +1.088621 | 0.145316 |

The helpful-history shortened inputs lack the still-valid rounding rule. Their UNKNOWN outputs are missing-information abstentions, although they fail the requested arithmetic task. The explicit archive control supplies the rule and still returns UNKNOWN. Its failure prevents a claim that retained state substitutes reliably for a working explicit retrieval control. The ordinary reuse answer, 459, is the unrounded net (`18 × 26 − 9`), rather than the correct multiple-of-eight result, 456. No condition emitted the obsolete multiple-of-eight answer, 456, when the current rule required 450.

The paired raw-logit recomputation uses the identical shortened final context and two teacher rows per case, including EOS. The row-weighted mean ΔNLL is −0.021738 and KL is 0.245738. These averages cancel large case-specific changes and are not evidence of task benefit. Cross-context KL is intentionally absent for the explicit archive control.

Core edit timing is diagnostic: shortened fresh took 13.54–13.63 s; reuse took 5.40–5.46 s. Fresh evaluates 640 tokens; reuse evaluates 256 and retains 384. These times exclude approximately 39–40 s of source setup, source capture, and answer generation. The source generated no earlier answers; each 49-token current query was fully fresh. The archive rule was read by the driver from the fixture file, rather than retrieved by a model tool. Actual HTTP retrieval and whole-task timing are separate measurements.

Settings: unchanged hour14 pre-RoPE oracle (`84aa5c3b1847b745122b4cf26ff0343a750c40d82b87261bd9280e509ecc3886`), fresh tail 128, source prefix 1024, source edit 256, shortened edit 128, unchanged suffix 512, Q2/cache64/ctx4096/chunk512, greedy cap16, selected TILE64/LIMIT1024/GAP0. The six original helper invocations and their raw outputs remain in [core](core); only the fresh arm of each archive-rule invocation is scored as the archive control.

Analysis command:

```sh
.workspaces/bench-plots-venv/bin/python speed-bench/experiments/suffix-history-hour15-20261004/analyze.py --input speed-bench/experiments/suffix-history-hour15-20261004/core --output speed-bench/experiments/suffix-history-hour15-20261004/first-analysis
```

Recomputed metrics agreed with the helper summaries within 1e-9. No run was missing. [Condition results](first-analysis/core-results.csv), [teacher rows](first-analysis/core-teacher-rows.csv), and [analysis JSON](first-analysis/core-analysis.json) retain the full scores and first texts.
