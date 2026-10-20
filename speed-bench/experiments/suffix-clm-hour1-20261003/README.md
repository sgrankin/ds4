# Suffix / CLM first hour

Started 2026-10-03 11:57:42 UTC; model measurements finished 12:27:41 UTC.
Local DeepSeek V4.1 Flash Q2, Metal SSD streaming, 64 GiB expert-cache target.
All model jobs used the existing MAIN pressure guard, serially. No runtime
defaults or MAIN model sources changed.

## What worked

Actual 128-token left compaction, retaining a 512-token unchanged suffix:

| Case | Fresh tail | Fresh / reuse edit time | Mean delta-NLL | Greedy result |
| --- | ---: | ---: | ---: | --- |
| Ordinary record/planning cleanup | 64 | 12.895 / 3.955 s (3.26x) | -0.0000334 | Both3141 |
| Later self-contained tool result | 64 | 12.755 / 3.855 s (3.31x) | -0.0000150 | Both3141 |
| Changed relevant fact | 64 | 12.915 / 3.955 s (3.27x) | +0.00000470 | Both3141 |
| Retained arithmetic inputs | 128 | 12.962 / 5.181 s (2.50x) | +0.000170 | Both42 |
| Retained Python function | 128 | 13.028 / 5.259 s (2.48x) | -0.00247 | Both206 |

The first three cases also ran at128/256 fresh tokens. Full old and edited
token sequences were checked to stay identical within each tail sweep.
All eleven non-removal configurations returned the independently expected
greedy answer. Delta-NLL/KL were independently recomputed from the complete
raw logits using NumPy float64. Scores include the answer tokens and EOS.

## A real adverse tradeoff

Remove an earlier record whose code6829 was demonstrably retrievable, then ask
a wholly fresh query. Fully fresh execution consistently answers UNKNOWN.

| Fresh tail | Reuse answer | Mean delta-NLL | Speedup | UNKNOWN sequence probability |
| ---: | --- | ---: | ---: | ---: |
| 64 | 6821 | +1.690 | 3.31x | 0.63% |
| 128 | 6712 | +0.556 | 2.50x | 18.9% |
| 256 | UNKNOWN | +0.188 | 1.68x | 56.9% |
| 384 | UNKNOWN | +0.125 | 1.26x | about69% |

The384 point meets the initial engineering targets: at least20% less edit time
and mean delta-NLL below ln(1.2)=0.1823. The256 point is a legitimate faster
choice with slightly more degradation. Correct greedy output still leaves a
substantial raw likelihood change; these probabilities are untempered
teacher-forced sequence likelihoods, not observed sampled success rates.
Removed-code likelihood remains a separate column in the CSV. No forgetting
claim follows from a correct answer.

## CLM block isolation

Both fresh-versus-grafted state/raw comparisons were exact under the private
isolation policy, in both ordinary planning cleanup and a self-contained later
tool result. Compared with ordinary native execution on the same edited
context, four-token mean delta-NLL was +0.0000395 and +0.0000646 respectively.
These callers teacher-force CODE=3141; they do not run free generation.
Their C-only timing excludes other edit work and is not a practical speedup.
The ordinary-mode correction passed three focused checks; all caller builds
completed without warnings, using the already-built private core and auxiliaries.

## Files and limits

- [Tradeoff curves](pareto.png), [vector plot](pareto.svg), [results CSV](results.csv).
- `fixtures/` contains the authored text. Arithmetic42 and function result206
  have independent golds. Each run directory retains its command/settings,
  small summary, row metrics, actual greedy results and raw logits.
- `planning_cleanup/` and `self_contained_tool_result/` retain copied caller
  sources, build commands, five-arm execution records and policy scores.
- `*pressure/` retain host samples and child exit results. Pressure stayed
  normal; swap was stable. Large RAM usage was allowed by the existing guard.

These are synthetic, non-thinking, short-answer screens. Timings are paired
fresh-then-reuse measurements in a warmed process, not balanced whole-agent
throughput. They include changed-middle evaluation, cache copying/repair and
fresh-tail replay, but exclude previous-context setup and subsequent answer
generation. Whole-run I/O footers do not isolate edit-branch saved bytes.
Repeated-edit costs and actual agent/tool workloads remain confirmation work.

Practical initial choice: try64/128 fresh tokens for fact-preserving compaction;
use the factual-removal curve to choose extra recomputation or fully fresh
execution when that behavior matters. Keep all failed points, and do not pool
ordinary and removal cases into one average quality verdict.
