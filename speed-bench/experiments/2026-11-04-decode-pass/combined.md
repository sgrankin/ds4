# Combined decode qualification

The adopted combination improved full decode in every measured 8K and 32K pair and reduced recorded coding replay model time. Fixed-history logits matched, and both builds passed a fresh coding smoke. This report compares the starting runtime with the adopted combination from the overnight decode session. Individual experiments used successive baselines, so their measured changes cannot be added to estimate the combined effect. The final comparison measures the package directly; it does not establish performance or quality for other models, machines or agent tasks.

## What changed

The starting revision is `8971dfbd`. Its binary and runtime Metal shader sources were built in `.workspaces/overnight-decode-20261104/starting-8971dfbd`. The frozen candidate revision is `328117d5`; earlier revision `b4f823fa` is runtime-equivalent and contains the five adopted changes. Both roots were built with the same Zig toolchain in ReleaseFast, including `ds4-bench`, `ds4-agent`, `ds4-server` and `agent-session-replay`. The baseline uses its original runtime shader sources.

The candidate combines five changes enabled by the resident fast presets:

- [Direct F16 gather](f16-gather.md) expands selected compressed KV directly into half-precision scratch, avoiding float32 intermediate traffic and permitting fused staging when eligible.
- [Every-layer submission](flush-cadence.md) submits eligible queued decode work after each layer, while retaining the router event and final drain for synchronization and resource retirement.
- [Shared-down overlap](shared-down.md) schedules the shared expert's down projection before the router wait completes, then joins it with the routed result using existing scratch.
- [Query RoPE table](query-rope-table.md) reuses per-token rotation coefficients in the eligible short Q8 query instead of recalculating them for each head.
- [Selected KV reuse](selected-kv-reuse.md) retains gathered compressed rows for eligible consumer layers in the same group; the publisher refreshes them for each token.

The [router SIMD](router-simd.md) and [clock-stride](clock-stride.md) candidates were not adopted. The [raw-clock](raw-clock.md) candidate was also not adopted. The linked reports own their individual evidence, activation conditions and limits.

## Setup

The qualification runs on an M5 Max with 128 GiB memory using `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, SSD streaming, a requested 64 GB expert cache and 8,192-token prefill chunks. The [driver](combined.nu) clears inherited `DS4` variables and records each arm's explicit settings, arguments, revision and executable hash. The baseline settings are in [starting-settings.json](starting-settings.json); the candidate settings are in [final-settings.json](final-settings.json). Each executable runs from its own source root because Metal shader sources load relative to the working directory.

Preflight found no active inference, thermal warning or performance warning. An existing idle Flint rec service (PID 62183, approximately 0% CPU and 4 MiB RSS) was left running. All three completed final timing batches exited successfully under the pressure guard, with pressure level 1 and swap ranging from 24.9723 to 24.9785 GiB. Raw qualification logs remain locally under `.workspaces/overnight-decode-20261104/combined/`.

The profile-free timing uses five alternating pairs at a 32,768-token frontier with 512 teacher-forced decode steps, followed by KV restore, a 512-token append and another 512 decode steps. A second five-pair comparison uses an 8,192-token frontier and 128 decode steps on each side of the same append. The fresh decode follows prefill. The benchmark saves the prefill frontier before that decode, then restores it, appends 512 tokens and runs the warm decode. The expert cache survives restore; the OS cache is not flushed. Inference phase times exclude model loading and snapshot save and restore. Steady decode excludes the first token, which does not prove initialization is complete; first-token latency and the complete decode window are reported separately.

Five alternating recorded coding replays cover 3,371 teacher-forced decode tokens over contexts growing from 1,859 to 8,710 tokens. Replay wall time includes model loading and the final snapshot; phase CSV timings are the inference comparison. Replay fixes the recorded history and therefore cannot establish model choices, tool choices or autonomous task quality.

A separate full-vocabulary replay comparison measures fixed-history distribution changes. A reasoning-enabled fresh coding smoke tests one generated function per build against 207 oracle cases. These tests answer different questions; the smoke test is bounded and does not establish general agent quality.

## Results

### 32K, 512 decode steps

The candidate improved steady throughput and complete decode duration in all five pairs, both fresh and after append. Fresh steady decode averaged 30.730 ± 0.603 → 32.826 ± 0.120 tokens/s; warm steady decode averaged 29.982 ± 0.386 → 32.304 ± 0.170 tokens/s (mean ± sample SD). Complete fresh decode averaged 17.571 ± 0.303 → 16.474 ± 0.038 s, and warm decode 17.084 ± 0.214 → 15.864 ± 0.081 s. Full durations are derived from throughput rounded to two decimal places.

| Pair (first arm) | Fresh steady tokens/s | Warm steady tokens/s | Full fresh s | Full warm s | First fresh ms | First warm ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (baseline) | 31.800 → 33.000 | 30.500 → 32.220 | 17.044 → 16.452 | 16.792 → 15.896 | 973.883 → 964.689 | 40.199 → 36.641 |
| 2 (candidate) | 30.550 → 32.890 | 29.610 → 32.510 | 17.601 → 16.421 | 17.291 → 15.764 | 873.700 → 880.253 | 35.595 → 48.016 |
| 3 (baseline) | 30.490 → 32.750 | 29.780 → 32.410 | 17.722 → 16.521 | 17.193 → 15.802 | 961.932 → 920.295 | 37.138 → 34.943 |
| 4 (candidate) | 30.470 → 32.790 | 29.740 → 32.070 | 17.692 → 16.484 | 17.216 → 15.965 | 920.010 → 903.158 | 36.513 → 32.702 |
| 5 (baseline) | 30.340 → 32.700 | 30.280 → 32.310 | 17.796 → 16.490 | 16.926 → 15.896 | 955.426 → 863.265 | 46.566 → 81.762 |

Each cell shows baseline → candidate. First fresh steps averaged 936.99 ± 40.67 → 906.33 ± 39.20 ms; first warm steps averaged 39.20 ± 4.46 → 46.81 ± 20.41 ms. The candidate's fifth warm first step took 81.762 ms, although its complete warm window still improved. Fresh prefill averaged 48.004 ± 1.416 → 48.308 ± 0.775 s; the 512-token append averaged 7.678 ± 0.256 → 7.770 ± 0.161 s. Total inference phases averaged 90.337 ± 2.129 → 88.416 ± 0.825 s; wall time averaged 93.298 ± 0.721 → 90.793 ± 0.718 s and has a broader scope. Baseline pair 1 had a 45.483 s fresh prefill and 31.80 tokens/s fresh steady decode, versus 48.52–48.86 s and 30.34–30.55 tokens/s in the other baseline pairs; these observations do not establish a cause. The longer 512-token history changes context and expert mix relative to the earlier 128-step experiments, so their rates are not directly comparable.

### 8K, 128 decode steps

Fresh steady decode averaged 31.494 ± 0.065 → 34.358 ± 0.178 tokens/s and warm steady decode 31.720 ± 0.159 → 34.620 ± 0.158 tokens/s. Full fresh decode averaged 4.917 ± 0.048 → 4.571 ± 0.025 s and full warm decode 4.051 ± 0.032 → 3.705 ± 0.010 s. Every pair favored the candidate for both steady throughput and full windows. Durations are derived from throughput rounded to two decimal places.

| Pair (first arm) | Fresh steady tokens/s | Warm steady tokens/s | Full fresh s | Full warm s | First fresh ms | First warm ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (baseline) | 31.400 → 34.640 | 31.880 → 34.870 | 4.982 → 4.533 | 4.039 → 3.694 | 938.114 → 866.235 | 55.583 → 51.493 |
| 2 (candidate) | 31.540 → 34.430 | 31.840 → 34.630 | 4.854 → 4.601 | 4.025 → 3.700 | 828.363 → 912.950 | 35.902 → 32.354 |
| 3 (baseline) | 31.540 → 34.250 | 31.770 → 34.630 | 4.893 → 4.567 | 4.033 → 3.700 | 866.257 → 858.086 | 35.349 → 32.933 |
| 4 (candidate) | 31.450 → 34.240 | 31.510 → 34.500 | 4.934 → 4.573 | 4.105 → 3.716 | 895.314 → 863.326 | 74.494 → 33.474 |
| 5 (baseline) | 31.540 → 34.230 | 31.600 → 34.470 | 4.921 → 4.581 | 4.054 → 3.717 | 894.721 → 870.375 | 35.347 → 32.374 |

Each cell shows baseline → candidate. First fresh steps averaged 884.554 ± 40.559 → 874.194 ± 22.122 ms and first warm steps 47.335 ± 17.491 → 36.526 ± 8.380 ms. Fresh prefill averaged 19.609 ± 0.511 → 19.343 ± 0.083 s, the warm append 7.359 ± 0.024 → 7.314 ± 0.070 s, and total inference phases 35.936 ± 0.542 → 34.933 ± 0.168 s.

### Recorded coding replay

The replay contains three user messages, 14 assistant messages and 20 tool outputs. Tool outputs are replayed: it does not execute tools or make fresh tool choices. Decode phases averaged 112.720 ± 0.992 → 104.507 ± 0.250 s, a 7.286% reduction; all model phases averaged 183.795 ± 2.153 → 176.164 ± 0.930 s, a 4.152% reduction. All five pairs favored the candidate for decode, model phases and process wall time.

| Pair (first arm) | Decode s | Append prefill s | All model phases s | Process wall s |
| --- | ---: | ---: | ---: | ---: |
| 1 (baseline) | 110.948 → 104.425 | 53.620 → 55.775 | 179.948 → 175.350 | 182.169 → 177.578 |
| 2 (candidate) | 113.129 → 104.283 | 56.460 → 56.573 | 184.620 → 175.886 | 186.753 → 177.947 |
| 3 (baseline) | 113.208 → 104.925 | 56.613 → 57.001 | 184.803 → 177.750 | 186.857 → 179.788 |
| 4 (candidate) | 113.115 → 104.526 | 56.565 → 56.464 | 184.701 → 176.125 | 186.749 → 178.156 |
| 5 (baseline) | 113.201 → 104.377 | 56.725 → 56.442 | 184.904 → 175.711 | 186.924 → 177.750 |

Each cell shows baseline → candidate. Startup prefill averaged 15.078 ± 0.170 → 15.206 ± 0.360 s; append prefill averaged 55.997 ± 1.332 → 56.451 ± 0.440 s. The first baseline append total was 53.620 s, versus 56.46–56.72 s for the other baseline samples. The sum of warm turns, excluding startup prefill, averaged 168.717 ± 2.323 → 160.958 ± 0.621 s, a 4.599% reduction. Process wall time averaged 185.890 ± 2.081 → 178.244 ± 0.890 s, a 4.114% reduction. Model phase times exclude loading, phase-logit writes, snapshot, sampling and tools; process wall includes loading, phase-logit writes and the final snapshot.

### Fixed-history quality

At 32K, all 1,024 paired full-vocabulary logit rows were byte-identical, with zero top-1 mismatches, delta-NLL, KL and maximum logit delta. All 1,024 post-evaluation full-logit hashes also matched. Each hash is taken after evaluating its teacher token, one position after its captured logit row; the sequences largely overlap and the hashes are not an independent second sample. At 8K, all 256 paired rows and post-evaluation hashes also matched, with zero top-1 mismatches, delta-NLL, KL and maximum logit delta. Together these cover 1,280 synthetic positions. The coding replay also matched all 3,371 full-vocabulary rows across its 14 decode phases (129,280 logits per row). Top-1 matched throughout, and delta-NLL, KL, maximum absolute logit difference and total variation were zero, including their reported tails. Across synthetic and replay histories, 4,651 full-vocabulary positions matched. The replay guard completed successfully.

### Fresh coding smoke

Both builds ran one fresh `merge_ranges` coding task with reasoning enabled, temperature 0, a 1,024-token generation limit, 8,192-token context and 2,048-token prefill chunks. Each wrote `intervals.py` and ran its own checks in an isolated working directory. An independent checker passed each produced function on 207 cases: seven named cases and 200 valid inclusive interval sets generated with `Random(0)`. The checker uses a point-union/grouping oracle and verifies the list-of-tuples result and input immutability. The generated sources differed, but both passed. The user task was the same; complete prompts were not identical. This is one task per build, not 207 tasks or a performance comparison. The smoke guard completed at pressure level 1 with swap stable at 24.978457 GiB.

The task asked for inclusive integer interval merging: sort the result, merge overlapping or adjacent intervals, return tuples, handle empty input and preserve the caller's input. The independent oracle enumerated each bounded interval's integer points and grouped consecutive points; it therefore did not duplicate either generated merge algorithm.

## Reproduction

Both roots must be built in ReleaseFast with the same Zig toolchain, including `ds4-bench` and `agent-session-replay`; the baseline root must contain revision `8971dfbd` and its matching runtime Metal sources. All four final binaries (`ds4-bench`, `ds4-agent`, `ds4-server` and `agent-session-replay`) built successfully; launcher preset tests passed after selected-KV adoption. There was no runtime change after the candidate freeze. The driver defaults to the baseline checkout and model paths described above. The measured binaries were invoked by explicit paths.

After qualification, the tested `ds4-agent`, `ds4-server` and `ds4-bench` were installed from `zig-out/bin` to the root compatibility paths used by `ds4-local`; hashes matched, prior binaries were saved under `.workspaces/overnight-decode-20261104/preinstall/`, and `./ds4-local persistent-fast --help` and `./ds4-local --server persistent-fast --help` passed. Use a new output path for each invocation and run only one guarded model job at a time. For example:

```sh
python3 speed-bench/ideas/pressure_run.py .workspaces/overnight-decode-20261104/reproduce-32k-guard -- nu speed-bench/experiments/2026-11-04-decode-pass/combined.nu .workspaces/overnight-decode-20261104/reproduce-32k timing --final-settings speed-bench/experiments/2026-11-04-decode-pass/final-settings.json --candidate-revision 328117d5 --frontier 32768 --tokens 512
```

For the other runs, use distinct guard and output paths and change the arguments after the output path as follows; keep the same `--final-settings` and `--candidate-revision` options:

| Run | Stage and workload arguments |
| --- | --- |
| 8K timing | `timing --frontier 8192 --tokens 128` |
| Coding replay timing | `replay-timing` |
| 32K quality | `quality --frontier 32768 --tokens 512` |
| 8K quality | `quality --frontier 8192 --tokens 128` |
| Coding replay quality | `replay` |

The timing stages default to five alternating pairs; quality stages capture one pair. Raw outputs include manifests, phase data and guard records; comparing full-vocabulary captures requires separate numerical analysis.

## Executable hashes

The driver records SHA-256 hashes for each run:

| Executable | Baseline | Candidate |
| --- | --- | --- |
| `ds4-bench` | `7ca2c32384beab616ce40cf77ecf5be96de51361dd85ba1df31e8dbe4140c7b8` | `6c679855b2e007278ecc92cbb6f672fe0441b3bfe559c301a2a0b6f65b8c6183` |
| `agent-session-replay` | `d101652f6ef41ccef2e5305ee2cdc98c51f6fa21462c3bd28e8170e2445d5bc6` | `bd36bcb5a8b528bef2e1b733f26a9f143c16f17fffb5b981456b7dae47bb37ab` |

## Additional timing samples

Each cell is baseline → candidate, in seconds. Inference totals exclude loading and snapshot save/restore; process wall has the broader scope described above. Values derived from the benchmark CSV throughput are approximate.

| Context and pair | Fresh prefill | Warm append | Inference phases | Wall |
| --- | ---: | ---: | ---: | ---: |
| 32K 1 | 45.483 → 46.935 | 7.223 → 7.769 | 86.542 → 87.052 | 92.075 → 89.625 |
| 32K 2 | 48.563 → 48.476 | 7.768 → 7.573 | 91.223 → 88.233 | 93.504 → 90.630 |
| 32K 3 | 48.862 → 48.774 | 7.808 → 7.755 | 91.585 → 88.853 | 93.957 → 91.055 |
| 32K 4 | 48.596 → 48.681 | 7.763 → 7.733 | 91.267 → 88.864 | 93.621 → 91.209 |
| 32K 5 | 48.516 → 48.672 | 7.828 → 8.021 | 91.065 → 89.079 | 93.331 → 91.448 |
| 8K 1 | 20.500 → 19.239 | 7.334 → 7.221 | 36.856 → 34.687 | 38.923 → 36.826 |
| 8K 2 | 19.261 → 19.296 | 7.347 → 7.258 | 35.487 → 34.856 | 37.610 → 36.944 |
| 8K 3 | 19.321 → 19.333 | 7.349 → 7.372 | 35.595 → 34.972 | 37.651 → 37.060 |
| 8K 4 | 19.395 → 19.405 | 7.398 → 7.336 | 35.833 → 35.030 | 37.952 → 37.149 |
| 8K 5 | 19.567 → 19.444 | 7.365 → 7.380 | 35.908 → 35.122 | 38.006 → 37.239 |

Replay startup prefill samples, baseline → candidate in seconds: 1: 15.380 → 15.150; 2: 15.031 → 15.030; 3: 14.983 → 15.823; 4: 15.021 → 15.134; 5: 14.978 → 14.892.

## Remaining targets

The latest [selected-KV diagnostic](selected-kv-reuse.md) ranks warm kernel work by summed duration: IQ2 gate/up 4.336 ms, Q2 sum6 down 2.669 ms, attention output and hyper-connection 3.621 ms, query 3.305 ms, and low projection 2.692 ms per token. These kernels overlap, so their sums are not additive recoverable savings. The 5.902 ms uncovered router-to-routed interval is a possible scheduling target, but is not proven CPU execution; the CPU trace was mostly wait, and the clock changes did not improve decode. A focused profile of that interval or of the largest exposed kernel work would be a tentative next choice, subject to a new decision and measurement.

## Decision

Retain all five changes in the resident fast presets. The direct combined comparison supports a decode improvement on these workloads, while prefill and append results vary and the coding smoke is limited to one task per build. Main was not moved or pushed.
