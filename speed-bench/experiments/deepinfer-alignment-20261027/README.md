# DeepInfer convergence experiments, 2026-10-27

The user authorized four sequential, initially one-hour experiments, with
DeepInfer as the preferred design reference unless Apple/SSD measurements or
constraints justify a difference. Inventory: [49-entry comparison](../../../docs/DEEPINFER_ALIGNMENT.md).
Reference upstream `553741557a57df44a46fb3ab72b6aafeaf945337`; fixed local runtime
`e2a38f6e2bf4`, audit parent `aeac7ed091cf`. Main remains unchanged.
Raw outputs live in ignored `.workspaces/deepinfer-alignment-20261027/`.

## 1. Candidate-only scalar index scoring

Completed12:50–13:33UTC within the initial one-hour budget. Implemented behind
`DS4_V41_CANDIDATE_INDEX=1`; default remains dense during qualification.
Layer20 chooses its unchanged 2,048 eight-key blocks. Source layers24/28/32/36
score only those keys, scattering scores to their original global IDs. Existing
filter/top-k and tie ordering stay unchanged. The initial implementation cleared
scores on GPU (a CPU clear would race queued work). Inspection then removed that
redundant pass: the existing filter already writes `-inf` for every excluded row.
The scorer writes only selected rows, then its caller applies that filter.
At <=16,384 keys the runtime keeps the original dense path.

Production-linked Metal fixture passes exact score/ID comparisons for finite
values and exact ties at37,8192,32768,32771,65536,262144 rows, including reverse
candidate order, partial eight-key tails and output canaries. Unchanged layer20
selection is covered by its existing fixtures; these checks target its readers.
Initial warmed alternating score + filter + top-k timings, eight repeats per point:

| Keys | Scored candidate keys | Dense ms | Candidate ms | Time reduction |
|---:|---:|---:|---:|---:|
| 8,192 | 8,192 | 0.788 | 0.823 | -4.5% |
| 32,768 | 16,384 | 1.011 | 0.780 | 22.8% |
| 65,536 | 16,384 | 1.151 | 0.590 | 48.7% |
| 262,144 | 16,384 | 2.683 | 0.750 | 72.1% |

After removing the redundant clear, the same fixture still passes. New matched
kernel pairs are0.625/0.918ms at8K,1.025/0.710ms at32K,1.391/0.679ms at64K,
and2.969/0.781ms at256K (dense/candidate). Independent sessions/runs vary, so
compare each matched pair rather than attributing all between-run change to
the simplification.

These are isolated scoring/selection timings, not complete token latency. The
8K point intentionally forces the all-candidate kernel to demonstrate why the
runtime bypasses it at short contexts. Prefill and batched/MTP verification are
unchanged by this scalar experiment.

Full-model dense/candidate/dense-repeat comparison completed at32K with the
1017 IQ2_XXS/Q2_K-imatrix model, normal reasoning, chunk8192 and48GiB expert cache.
The reused teacher fixture measures common token distributions and separately
generates greedy answers for ordinary, recent-record and changed-fact cases.
All three tasks pass in every arm. All22 teacher distributions are byte-identical
and every generated token, including reasoning, matches: measured ΔNLL=KL=0.
Candidate decode time is2.8%,2.8%,4.3% higher than the bracket mean for ordinary,
favorable and changed-fact cases, respectively. Closing dense controls are
9.3%,11.3%,12.5% slower than opening controls. This noisy bracket establishes
correctness, not a whole-model speed win. A64K bracket with the simpler scorer
completed; preserve this first result rather than replacing it.
Teacher targets are fixed answer tokens at the same reasoning frontier; they
measure distribution changes, not a standalone task-quality score.
The initial64GiB-cache attempt was stopped by the unchanged pressure guard on
macOS pressurelevel2, before producing a result. Its process exited;48GiB retry
starts from pressurelevel1. Preserve the failed attempt, not a timing sample.

At64K with the simpler scorer, all8 teacher rows and all27 generated tokens
are byte-identical. Whole-answer decode is3.224/3.257/3.289s for the
dense/candidate/dense-repeat bracket: candidate is0.03% slower than the control
mean, effectively neutral here. Most whole-answer cost remains elsewhere.

A short loop then restores **one identical in-memory64K checkpoint**, replays
the same27 tokens in alternating dense/candidate order and excludes the first
pair as warmup. Four measured pairs average1.208s dense versus1.188s candidate,
**1.68% less token-evaluation time**, with each pair improving1.0–2.2%.
Restore and snapshot time are outside that window. Repetition warms the expert
cache too: do not compare its1.2s directly to the first-answer3.2s or call this
a cold-request benefit. The useful win is small and applies to cache-resident
decode; retain the optimization opt-in, with no preset/default promotion yet.
The ordinary8K runtime gate is also checked through this loop:87 tokens average
3.663s dense versus3.670s with the flag on (0.17% variation on the same dense
path). No default/preset or installed executable is changed in this item.

Reproduce under the pressure guard (one GPU/model job at a time):

```sh
nu tools/zig-build metal-test decoder-replay-bench --prefix .workspaces/alignment-build -j6
python3 speed-bench/ideas/pressure_run.py .workspaces/alignment-gpu-guard -- .workspaces/alignment-build/bin/test-deepseek41-metal --index-candidates
python3 speed-bench/ideas/pressure_run.py .workspaces/alignment-model-guard -- nu speed-bench/experiments/deepinfer-alignment-20261027/candidate-index.nu .workspaces/alignment-build/bin/decoder-replay-bench .workspaces/alignment-model --tokens 32768
python3 speed-bench/experiments/deepinfer-alignment-20261027/compare.py .workspaces/alignment-model
# The same loop used to reduce first-answer cache/order effects:
python3 speed-bench/ideas/pressure_run.py .workspaces/alignment-loop-guard -- nu speed-bench/experiments/deepinfer-alignment-20261027/candidate-index.nu .workspaces/alignment-build/bin/decoder-replay-bench .workspaces/alignment-loop --tokens 65536 --case favorable --hot-loop
```

## Remaining authorized experiments

2. Block-only cache hit plus128-row bounded attention recovery. Start with a
   direct aligned-frontier prototype, keeping durable compressed/index blocks
   immutable. Measure quality, bytes and restore cost; distinguish an all-layer
   prototype from upstream's encoder-only controller.
3. One real packed producer/reader, initially128-wide index FP4 with UE8M0 scales
   (68B versus512B per row). Verify numerical ABI and measure the whole boundary,
   rather than claiming savings from dtype names or CPU-only repacking.
4. Preserve useful128-position DSpark target features across canonical/SSD
   restore so short cached requests need not warm128 future scalar tokens.
   Measure first-cycle activation and persistence/rebuild cost separately.
