# Coherent cap64 follow-up

The two shortened pairs were rerun with generation cap64, retaining the cap16 records. **All four answers reached EOS without hitting the cap; strict integer-only correctness remains 0/4.** Separate mathematical review finds two correct calculations with explanations.

| Case | Arm | Completed first text | Tokens including EOS | Strict pass | Mathematical review |
|---|---|---|---:|---|---|
| Ordinary cleanup, gold456 | Fresh | UNKNOWN | 3 | false | Abstains despite the visible rule |
| Ordinary cleanup, gold456 | Reuse | `18*26=468; 468-9=459; floor to multiple of 8: 456.` | 24 | false | Correct calculation; explanatory format |
| Obsolete history, gold450 | Fresh | `18*26=468; 468-9=459; round down to multiple of 10 = 450.` followed by `450` | 27 | false | Correct current-rule calculation; explanatory format |
| Obsolete history, gold450 | Reuse | UNKNOWN | 3 | false | Fails to produce the current answer; does not emit obsolete456 |

The two cap16 truncations were budget failures and finish correctly under the larger budget. Ordinary reuse improves mathematical completion over its fresh control, but the valid rule is already visible in this case: it does not establish recall of an omitted rule. In the obsolete case, reuse loses the correct calculation rather than producing an old answer. Helpful-history was not rerun in this follow-up.

Direct JSON comparisons confirmed identical `old`, `edited`, `gold`, and `stale` token arrays between cap16 and cap64 for both cases; both old/new prompt files are also byte-identical. Each cap64 arm begins with exactly its complete cap16 token stream: the formerly capped arms match all first16 IDs, and the UNKNOWN arms match all three IDs including EOS. All four cap64 streams end with EOS token1. The helper binary hash remains `84aa5c3b1847b745122b4cf26ff0343a750c40d82b87261bd9280e509ecc3886`; only the generation bound changes from16 to64.

Teacher scores are unchanged from [the coherent cap16 cohort](coherent-results.md), rather than new independent likelihood points: ordinary mean ΔNLL −0.472956 / KL0.183331; obsolete +0.147148 / KL0.147648. Summary row counts, KL/TV, ΔNLL, argmax flips, byte-equal counts, and per-arm gold/stale log probabilities match exactly.

The [guard](cap64-guard/result.json) completed with child return code0. Across144 pressure samples over147.80 s, every sample was level1 with no pressure reason; swap stayed at0.012998 GiB, peak wired72.622 GiB, peak compressor10.066 GiB, and minimum free0.228 GiB. Individual helper processes completed in75.54/72.31 s. These diagnostic repeats do not constitute a new final-latency comparison.

Raw summaries: [ordinary](core-coherent-cap64/ordinary-cleanup-shortened/summary.json), [obsolete](core-coherent-cap64/obsolete-history-shortened/summary.json). Their adjacent token and execution records retain the completed first outputs and exact commands.
