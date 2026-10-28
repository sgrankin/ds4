# Checkpoint overhead without context edits

The new semantic checkpoints show **no clear request-time penalty** in this
workload. Five alternating pairs compare checkpoints disabled with anchors
and in-sweep capture enabled. Each case sends one fresh request and makes no
context edits. Request time includes prefill and generation.

| Case | Checkpoints off | New checkpoints on | Change in mean |
| --- | ---: | ---: | ---: |
| Ordinary | 39.717 ± 1.358 s | 38.908 ± 0.371 s | −2.04% |
| Favorable | 41.586 ± 0.846 s | 41.871 ± 0.712 s | +0.69% |
| Both cases combined | 81.303 ± 1.539 s | 80.779 ± 1.041 s | −0.64% |

Values are means ± sample standard deviations. The ordinary control has one
slower first sample, and paired differences vary in direction. These results
do not establish either a speedup or a regression. All 20 answers are correct
and complete.

This comparison isolates the added semantic checkpoint machinery. Ordinary
prefix/suffix caching and the RAM/SSD tiers remain enabled on both sides. The
timed requests have zero cached input tokens, so none benefit from a previous
context. The candidate records four transient captures per process, compared
with zero in the control; neither arm records a capture failure.

The [earlier collector comparison](COLLECTION.md) answers a different question:
new in-sweep checkpoints versus old segmented checkpoints. Its Initial column
already excludes subsequent edits, but both arms pay for checkpoints. This new
comparison measures the cost against disabling semantic checkpoints entirely.
It does not measure a long conversation that only appends turns.

## Samples and variation

Each row is one pair. Process order alternates off/on, on/off, off/on, on/off,
off/on. Times are seconds; no samples are discarded.

| Pair | Ordinary off | Ordinary on | Favorable off | Favorable on |
| --- | ---: | ---: | ---: | ---: |
| 1 | 42.047680 | 39.238963 | 41.097454 | 42.004897 |
| 2 | 39.751837 | 39.243882 | 42.924956 | 42.328795 |
| 3 | 39.163112 | 38.489686 | 41.364444 | 41.167620 |
| 4 | 38.796016 | 38.535544 | 40.731837 | 41.121148 |
| 5 | 38.827562 | 39.031849 | 41.812229 | 42.733260 |

Ordinary paired changes range from −6.68% to +0.53%, with a sample standard
deviation of 2.77 percentage points. Favorable paired changes range from
−1.39% to +2.21%, with a sample standard deviation of 1.61 percentage points.
Combined changes range from −2.29% to +1.40%.

Server startup is recorded separately and excluded from every request time.
All startup samples are retained: off `[3.539826, 0.504351, 0.505911, 0.506183,
0.506087]` seconds; on `[0.502726, 0.506357, 0.506899, 0.506233, 0.511095]`
seconds. The first process takes longer to start. No OS file-cache flush is
performed. The two cases use separate contexts within each process; the
favorable case runs second and shares warm model weights.

## Settings and reproduction

Both arms use frozen runtime `f74180e070516d54c6095f29ad3b9a64846e959e`
on the M5 Max with 128 GiB RAM and the installed DeepSeek V4.1 Flash 1017
IQ2_XXS/Q2_K checkpoint. Settings are high reasoning, temperature 0, min_p
0.05, no MTP, native expert routes, a 32 GB expert budget, packed main/index
KV, context 32768, configured prefill chunk 8192 and ordinary coordinator
quantum 2048. The hot cache is 512 MiB and the SSD cache is 4096 MiB. Inputs
contain 3672 ordinary or 4238 favorable tokens. Outputs contain 29 or 30
tokens, including reasoning.

The [control settings](no-edits-control.json) disable semantic checkpoints,
ongoing-user suffix reuse and in-sweep capture. The [candidate settings](no-edits-candidate.json)
enable those features with anchor selection. Both retain limit 4 and spacing
512. There are five fresh processes per arm, 24 fixture rows and zero edits.

```nu
python3 speed-bench/ideas/pressure_run.py .workspaces/checkpoint-capture-20261028/no-edits-guard -- caffeinate -i nu speed-bench/experiments/context-capture-20261028/run-five.nu .workspaces/checkpoint-capture-20261028/rewrite/bin/ds4-server .workspaces/checkpoint-capture-20261028/no-edits-five --control-env speed-bench/experiments/context-capture-20261028/no-edits-control.json --candidate-env speed-bench/experiments/context-capture-20261028/no-edits-candidate.json --rows 24 --edits 0 --pairs 5 --spacing 512 --limit 4 --chunk 8192
```

Use a new output directory when repeating the run. The driver generates the
numerical summary with [no-edits-report.py](no-edits-report.py). Raw requests,
answers, timings, manifests and server logs remain local and ignored under
`.workspaces/checkpoint-capture-20261028/no-edits-five/`. The pressure guard
finishes with child return code 0; pressure remains normal and swap stays
unchanged at about 22.99 GiB.
