# Notes-only coherent cohort

The separately frozen notes-only cohort scored **0/9 correct integer-only answers** at cap16. Seven conditions returned UNKNOWN through EOS; ordinary reuse and obsolete fresh started arithmetic explanations and hit the cap without a final answer. These two incomplete outputs are budget failures, not evidence of incorrect completed arithmetic. The original cohort remains in [original-results.md](original-results.md).

| Case (gold) | Fresh shortened | Reuse shortened | Fresh explicit archive rule | Reuse ΔNLL | KL(fresh ∥ reuse) |
|---|---|---|---|---:|---:|
| Ordinary cleanup (456) | UNKNOWN | `18*26=468; 468-9=459; floor to multiple` (cap16) | UNKNOWN | −0.472956 | 0.183331 |
| Helpful history (456) | UNKNOWN | UNKNOWN | UNKNOWN | −1.844831 | 1.241158 |
| Obsolete history (450) | `18*26=468; 468-9=459; round down to` (cap16) | UNKNOWN | UNKNOWN | +0.147148 | 0.147648 |

The helpful shortened context omits the rounding rule. Its fresh and reuse UNKNOWN outputs are missing-information abstentions; the explicit archive control contains the rule and still fails. There is no demonstrated retained-history task benefit. No condition emitted the obsolete answer, 456, for the current multiple-of-ten case. The paired distributions changed substantially, but a favorable teacher likelihood alone does not establish a usable recalled answer.

The raw-logit recomputation compares the same shortened final context, with two teacher rows per case including EOS. Its row-weighted mean ΔNLL is −0.723546 and KL is 0.524046; all paired metrics agreed with the helper summaries within 1e-9. No cross-context KL is assigned to the explicit archive control, and no run was missing.

This cohort changes the unchanged notes to a neutral statement that the recorded approved policy governs the next dispatch and newly supplied current policy takes precedence. The source generates no answer. The unchanged oracle binary is `84aa5c3b1847b745122b4cf26ff0343a750c40d82b87261bd9280e509ecc3886`; settings remain Q2/cache64/ctx4096/chunk512, prefix1024/source-edit256/edited128/unchanged512/fresh-tail128, greedy16, TILE64/LIMIT1024/GAP0. Each 49-token query is fully fresh.

Diagnostic shortened edit time was 13.39–13.47 s for fresh (640 tokens evaluated) versus 5.33–5.46 s for reuse (256 evaluated, 384 retained). Source setup of approximately 39 s, capture, and answer generation are excluded. The archive control uses driver file I/O; the actual HTTP tool-read controls are reported separately. The later cap64 continuation is excluded from this report.

Analysis command:

```sh
.workspaces/bench-plots-venv/bin/python speed-bench/experiments/suffix-history-hour15-20261004/analyze.py --input speed-bench/experiments/suffix-history-hour15-20261004/core-coherent --output speed-bench/experiments/suffix-history-hour15-20261004/coherent-analysis
```

[Condition scores and first texts](coherent-analysis/core-results.csv), [paired teacher rows](coherent-analysis/core-teacher-rows.csv), and [analysis JSON](coherent-analysis/core-analysis.json) retain the failures and cap flags.
