# Core fresh-tail curve

All nine measured points returned `3141` and stopped at EOS in both fresh and reuse arms. These are three authored fixtures repeated at three boundaries. The 64-token boundary is fastest here; additional fresh tokens do not improve every likelihood metric monotonically.

| Family | Fresh tail | Fresh / reuse edit seconds | Speedup | Mean delta-NLL | Mean KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordinary-deletion | 64 | 13.165 / 4.005 | 3.29x | -2.25377e-05 | 1.21418e-05 |
| ordinary-deletion | 128 | 13.455 / 5.290 | 2.54x | -3.08984e-05 | 2.112e-05 |
| ordinary-deletion | 256 | 13.151 / 7.707 | 1.71x | -1.90887e-05 | 1.1479e-05 |
| best-self-contained | 64 | 12.903 / 3.877 | 3.33x | +1.55113e-05 | 9.96637e-06 |
| best-self-contained | 128 | 12.922 / 5.122 | 2.52x | -1.19992e-05 | 8.58448e-06 |
| best-self-contained | 256 | 12.952 / 7.577 | 1.71x | -1.87509e-05 | 9.57371e-06 |
| stress-changed-fact | 64 | 13.165 / 3.995 | 3.30x | +9.52186e-06 | 1.94598e-05 |
| stress-changed-fact | 128 | 13.162 / 5.245 | 2.51x | +3.86927e-06 | 5.88684e-06 |
| stress-changed-fact | 256 | 13.225 / 7.763 | 1.70x | -1.34493e-06 | 6.76165e-06 |

Raw full-vocabulary logits were recomputed independently with float64 log-softmax. Delta-NLL is reuse minus fresh NLL; KL is KL(fresh || reuse), without sampling temperature. The recomputed sums/means agree with caller summaries within 1e-9. All old, edited and gold token IDs match across tails within each family. The complete fresh teacher logit vectors are also byte-identical across each family’s three tails. Byte identity was observed for these baselines; it was not an approximation acceptance gate.

Each point scores only three teacher tokens including EOS: 27 vectors total, all for the same short answer. Token-weighted aggregate delta-NLL is −8.41305e-6 nats/token; mean KL is 1.16637e-5. There are no greedy caps. These saturated short-answer fixtures provide little evidence about difficult generation or broader accuracy.

The changed-fact fixture has a small retained influence despite correct output: at tail 64, the stale 2718 sequence probability rises from 3.12e-9 fresh to 8.62e-6 reuse. Its absolute probability remains small. The mean delta-NLL for this point is +9.52e-6; correctness alone does not demonstrate absence of historical influence.

Core `edit_ms` starts after restoring the unchanged prefix; it includes changed-middle evaluation and fresh/reused suffix work. It excludes prior-source setup, capture, model startup and answer generation. Prior-source setup took 36.38–40.47 seconds per point; capture itself took 2.54–3.25 milliseconds. This caller uses RAM-held immutable pre-RoPE sources, so these core figures do not include shared SSD cache reopening. Nine process runs took 641.14 seconds including setup and diagnostics. Whole HTTP workflow latency and practical tool outcomes are separate measurements in the main report.

All actual current queries contain 24 tokens including the assistant prefix, so every 64/128/256 boundary keeps the full query fresh. Reused lengths are 448/384/256 tokens respectively, at a fixed 128-token left shift and identical final 1664-token context; source context length is 1792 tokens.

Core guard: 621 samples, all pressure level 1; swap held at 0.012998 GiB with zero growth. Peak wired RAM 73.55 GiB; peak compressed RAM 9.94 GiB. Guard and loop completed with exit 0.

Commands/revision: [build.json](build.json), per-point `execution.json`, [guard command](core-guard/command.json). Full row data: [results.csv](results.csv), [teacher-rows.csv](teacher-rows.csv), [analysis.json](analysis.json). Plots: [delta-NLL curve](pareto.png), [KL curve](kl.png), with matching SVG exports.
