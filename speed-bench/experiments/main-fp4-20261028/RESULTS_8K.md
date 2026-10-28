# Main FP4: five-pair 8K result

Completed 2026-10-28 05:59 UTC on the M5 Max, revision `aee06ac9`.
The unchanged pressure guard exits 0. All ten fresh processes, in order
AB/BA/AB/BA/AB, are retained: A=expanded F32 main, B=physical FP4 main.
Only `DS4_V41_COMPRESSED_KV_FP4` changes. Packed index/SIMD, normal expert
routing and 48 GiB configured expert cache are fixed. Context 16384,
prompt 8192 and prefill chunk 8192; reasoning is high, maximum generation 384.
HC fusion, candidate scoring and compressed activation experiments are off.

**Decision guidance: latency-neutral, useful memory reduction.** Aggregate
answer prefill+generation averages **75.601→74.768 s (−1.10%)**, but the median
paired change is **+0.005%** and three of five pairs are slightly slower.
The first control has a longer changed-case prefill; its sample remains
included. These data do not establish a repeatable speedup or a material
whole-task regression. All-phase time is −0.57%; answer prefill −1.41% and
normal generation +0.63%. Main owners save 68.757 MiB at ctx=16384, and
canonical prompt payloads save 34.375 MiB. All 30 answers pass, generated IDs
match, and all 110 paired fixed-teacher rows are byte-identical.

The completed [32K follow-up](RESULTS_32K.md) measures a 1.98% task-time cost and
profiles the attention reader. The subsequent decision enables packing through
the opt-in fast preset package, keeping ordinary engine defaults unchanged.
Expert capacity is fixed in this experiment; it does not measure spending the
saved RAM on more resident experts.

Model:
`gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`.
The frozen executable, source revision, adjacent shaders and settings are in
`full-8k/manifest.json`; raw outputs and local `summary.json` remain under
`.workspaces/main-fp4-20261028/full-8k/`.
Initialization is excluded. Both prefills have fresh KV; the answer prefill
uses experts warmed by the preceding teacher branch. OS file caches are not
flushed. This is direct prefill/generation, not an HTTP/cache-restore timing.

## All five primary task samples

Seconds, including answer prefill and normal-reasoning generation:

| Case | Arm | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 |
|---|---|---:|---:|---:|---:|---:|
| ordinary | control | 26.523330 | 26.657143 | 26.419883 | 25.977979 | 25.964831 |
| ordinary | packed | 27.475469 | 26.061263 | 26.266954 | 26.160661 | 25.628876 |
| favorable | control | 26.237349 | 23.634077 | 24.484616 | 24.188922 | 24.339089 |
| favorable | packed | 25.073889 | 24.349397 | 24.514536 | 24.051325 | 24.350230 |
| changed | control | 28.487661 | 24.766952 | 23.818213 | 23.108522 | 23.397475 |
| changed | packed | 24.281043 | 23.686723 | 24.571563 | 23.642864 | 23.726332 |
| all_cases | control | 81.248340 | 75.058172 | 74.722712 | 73.275423 | 73.701395 |
| all_cases | packed | 76.830401 | 74.097383 | 75.353053 | 73.854850 | 73.705438 |

| Case | Arm | Mean (s) | Median (s) | Range (s) | Sample SD (s) |
|---|---|---:|---:|---:|---:|
| ordinary | control | 26.309 | 26.420 | 25.965–26.657 | 0.319 |
| ordinary | packed | 26.319 | 26.161 | 25.629–27.475 | 0.691 |
| favorable | control | 24.577 | 24.339 | 23.634–26.237 | 0.983 |
| favorable | packed | 24.468 | 24.350 | 24.051–25.074 | 0.378 |
| changed | control | 24.716 | 23.818 | 23.109–28.488 | 2.200 |
| changed | packed | 23.982 | 23.726 | 23.643–24.572 | 0.420 |
| all_cases | control | 75.601 | 74.723 | 73.275–81.248 | 3.239 |
| all_cases | packed | 74.768 | 74.097 | 73.705–76.830 | 1.324 |

All paired percentage changes are retained; positive means packed slower.
AB has three pairs, BA has two, so the order groups are descriptive.

| Case | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean-time change | AB mean paired | BA mean paired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ordinary | +3.590% | -2.235% | -0.579% | +0.703% | -1.294% | +0.038% | +0.572% | -0.766% |
| favorable | -4.434% | +3.027% | +0.122% | -0.569% | +0.046% | -0.443% | -1.422% | +1.229% |
| changed | -14.766% | -4.362% | +3.163% | +2.312% | +1.406% | -2.970% | -3.399% | -1.025% |
| all_cases | -5.438% | -1.280% | +0.844% | +0.791% | +0.005% | -1.102% | -1.530% | -0.245% |

## Phase costs

Per-case arithmetic means, seconds, shown as control→packed. The primary
combined metric is the last column. The full JSON retains per-case phase
samples, medians, ranges, SD and order groups.

| Case | Teacher prefill | Teacher continuation | Answer prefill | Generation | Answer prefill + generation |
|---|---:|---:|---:|---:|---:|
| ordinary | 19.958→20.305 | 1.368→1.329 | 19.532→19.607 | 6.777→6.712 | 26.309→26.319 |
| favorable | 20.716→20.931 | 0.678→0.686 | 22.013→21.862 | 2.564→2.606 | 24.577→24.468 |
| changed | 21.647→21.181 | 0.627→0.596 | 22.485→21.655 | 2.231→2.327 | 24.716→23.982 |

Aggregate across all three cases in each process:

| Phase | Arm | Mean (s) | Median (s) | Range (s) | Sample SD (s) |
|---|---|---:|---:|---:|---:|
| Teacher prefill | control | 62.321 | 61.679 | 59.971–66.684 | 2.640 |
| Teacher prefill | packed | 62.417 | 61.616 | 60.380–65.122 | 1.973 |
| Forced teacher continuation | control | 2.673 | 2.620 | 2.557–2.946 | 0.157 |
| Forced teacher continuation | packed | 2.611 | 2.612 | 2.527–2.691 | 0.065 |
| Answer prefill | control | 64.029 | 63.024 | 61.776–69.473 | 3.129 |
| Answer prefill | packed | 63.124 | 62.459 | 62.169–65.184 | 1.272 |
| Normal-reasoning generation | control | 11.572 | 11.500 | 11.400–11.775 | 0.158 |
| Normal-reasoning generation | packed | 11.645 | 11.647 | 11.396–11.928 | 0.236 |
| Answer prefill + generation | control | 75.601 | 74.723 | 73.275–81.248 | 3.239 |
| Answer prefill + generation | packed | 74.768 | 74.097 | 73.705–76.830 | 1.324 |
| Teacher prefill + continuation | control | 64.994 | 64.339 | 62.528–69.630 | 2.793 |
| Teacher prefill + continuation | packed | 65.028 | 64.143 | 63.071–67.774 | 1.982 |
| All measured phases | control | 140.595 | 139.062 | 136.229–150.878 | 6.000 |
| All measured phases | packed | 139.796 | 139.496 | 136.777–143.240 | 2.749 |

| Aggregate phase | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 | Mean-time change | AB mean paired | BA mean paired |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher prefill | -4.328% | +3.979% | -0.101% | +0.870% | +0.682% | +0.154% | -1.249% | +2.424% |
| Forced teacher continuation | -11.343% | +1.254% | -5.013% | -0.431% | +5.257% | -2.330% | -3.700% | +0.411% |
| Answer prefill | -6.174% | -2.339% | +0.797% | +1.105% | +0.104% | -1.414% | -1.758% | -0.617% |
| Normal-reasoning generation | -1.094% | +4.631% | +1.095% | -0.899% | -0.531% | +0.627% | -0.176% | +1.866% |
| Answer prefill + generation | -5.438% | -1.280% | +0.844% | +0.791% | +0.005% | -1.102% | -1.530% | -0.245% |
| Teacher prefill + continuation | -4.625% | +3.870% | -0.304% | +0.816% | +0.869% | +0.052% | -1.353% | +2.343% |
| All measured phases | -5.062% | +1.115% | +0.312% | +0.803% | +0.402% | -0.568% | -1.449% | +0.959% |

For the aggregate primary metric, paired SD is 2.617 percentage points;
range −5.438% to +0.844%, median +0.005%. The changed-case mean improvement
is driven by the early slower control: three of its five packed task pairs
are actually slower. Retaining it shows why the mean alone is insufficient
for a speedup claim. No sample is filtered and no timing from correctness
fixtures is included.

## Numerical and practical quality

The teacher rows force expected-answer tokens plus EOS at the reasoning
frontier. They are narrow answer-position probes, not broad reasoning scores.
Mean NLL comes directly from each run's `rows.csv`. Matched teacher logits
are byte-identical in all five pairs, as are repeated controls versus the
opening control. This gives ΔNLL=0, KL=0 and no argmax flips at every measured
position, without requiring approximate error thresholds.

| Case | Teacher positions per process | Mean NLL, both arms | ΔNLL | KL | Flips, five pairs | Generated tokens, both arms | Correct answers |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary | 7 | 1.416736628 | 0 | 0 | 0/35 | 87 | 10/10 |
| favorable | 8 | 1.153435699 | 0 | 0 | 0/40 | 31 | 10/10 |
| changed | 7 | 1.298985372 | 0 | 0 | 0/35 | 26 | 10/10 |

All generated token IDs and final answer text match across paired arms and
repeated processes: `CODE=EMBER-7319`, `CODE=CEDAR-4826`,
`CODE=LARCH-9062`. Normal reasoning remains enabled; the full generated lengths
are 87/31/26 in both arms. This storage experiment does not shorten reasoning.

## Allocation and canonical byte counts

All five samples per arm have identical allocation/payload counters (SD 0).
They are actual tensor-owner allocations and serializer-reported canonical
sizes respectively, rather than inferred row savings alone.

| Quantity | Control bytes | Packed bytes | Saved bytes |
|---|---:|---:|---:|
| Actual main owners | 83,894,272 | 11,797,632 | 72,096,640 |
| Actual packed index owners | 2,785,552 | 2,785,552 | 0 |
| Actual raw rings | 10,485,760 | 10,485,760 | 0 |
| Actual total KV | 97,165,584 | 25,068,944 | 72,096,640 |
| Planned main + index | 86,679,824 | 14,583,184 | 72,096,640 |
| Engine-bound KV + buffer plan | 7,140,200,888 | 7,068,104,248 | 72,096,640 |
| Canonical payload, each 8192-token case | 54,371,380 | 18,326,580 | 36,044,800 |

Actual total KV is 92.664→23.908 MiB; main owners alone shrink 85.9375%.
The per-case canonical payload is 51.853→17.478 MiB (−66.29%); the private
suffix-capture size reports the same values in this direct fresh-state probe.
These counts predict the canonical bytes stored by this layout, but this
campaign does not measure on-disk block allocation, cache-hit latency or SSD
write traffic. Persistence correctness and actual SSD reopen are qualified
separately in the [main report](README.md#persistence-and-suffix-representation).

Rounded cleanup counters also repeat in all five processes: static context
buffers 6809.43→6740.67 MiB, runtime tensor peak 6275.40→6206.64 MiB,
scratch 267.59 MiB in both arms. Rounded whole-startup plan 64.02→63.95 GiB
includes weights/expert cache beyond the engine-bound KV/buffer plan above.
The configured expert capacity is unchanged: live 40.88 GiB, 50,690 cache hits,
14,198 misses and 56.97 GiB expert pread in every process. These I/O counters
cover all cases and teacher branches, not just the primary timed window.
There is no expert-I/O reduction claim from this experiment.

Reproduce with [run.nu](run.nu) and [analyze.py](analyze.py), or rerun only the
analysis of retained samples:

```nu
python3 speed-bench/experiments/main-fp4-20261028/analyze.py .workspaces/main-fp4-20261028/full-8k --output .workspaces/main-fp4-20261028/full-8k/summary.json
```
