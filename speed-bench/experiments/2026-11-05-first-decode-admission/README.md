# First decode expert admission

This experiment asks whether selecting the existing address-based expert dispatch on the first scalar decode step removes the roughly 800 ms delay observed after fresh prefill. This step is one part of inference, not request time to first token. We retained the existing flag in the selected experimental GPU package: the timing effects were small and mixed, and all 512 measured quality rows matched exactly. It is not a fix for the delay or an established general speedup. No ordinary preset, binary or runtime implementation changed.

## Why test this path?

The preceding [submission-cadence experiment](../2026-11-05-gpu-submit-cadence/README.md) traced the first scalar step with native grouping and separate timeline instrumentation. In its cadence-four diagnostic, the first step's command scope took 841.1 ms, while the 30 ordinary fresh steps averaged 28.0 ms and the warm first step took 37.5 ms. All 40 first-step routed expert encoders used the six-slot kernel; subsequent ordinary and refresh steps used address-based kernels. Large gaps appeared before expert command buffers across many layers. This localizes an opportunity to test, but does not establish a driver or resource-admission cause.

The diagnostic's command scope starts after Engram reader startup and is not the benchmark token-start interval. GPU span and gap sums are not disjoint work categories or utilization, and host and GPU timestamps were not aligned. Timeline instrumentation inflated warm-step timing substantially, so it cannot establish a native warm stall. Pipeline creation accounted for 5.588 ms in the cadence-four trace; selected-resource binding accounted for 1.966 ms across four refresh tokens. Whole-run cache counters cannot attribute reads to the first step or distinguish logical from physical SSD traffic. Raw traces and their detailed mapping remain in `.workspaces/hillclimb-20261104/22-first-decode-admission/existing-profile-note.md`.

The [earlier slab-size experiment](../2026-11-03-expert-slabs/README.md) already showed that a faster first step can trade against later decode. We keep its 4 GiB slabs fixed and measure the complete generation and request scopes as well as the first step.

## Intervention and setup

The November 5, 00:49–01:49 Eastern slice ran on an M5 Max with 128 GiB memory. Both sides use runtime `909faab5`, checkout `0dff6c76`, and the same `ds4-bench` binary (SHA-256 `a8fad44806932fad5e041693b7001b913e147751e08e1790a58b3b067f7fb5e9`). The selected experimental GPU package, including four-layer submissions, is fixed by [candidate-settings.json](../2026-11-05-gpu-submit-cadence/candidate-settings.json). Both sides use `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf`, the `promessi_sposi.txt` prompt, a 64 GB expert cache and 8K prefill chunks. The harness evaluates 128 teacher-forced tokens at the fresh frontier and again after a 512-token append. The sole variable is `DS4_METAL_ENABLE_STREAMING_EXPERT_ADDR_TABLE`: it is **unset** for baseline and set to `1` for candidate. This is a presence flag: setting it to `0` also enables it.

Normally, [`splitReady`](../../../src/metal/expert_cache.zig) requires at least four decode tokens and sufficient cached entries before the ordinary selected-expert path requests address dispatch. Before that threshold the first CPU-refresh step uses six-slot dispatch. The candidate flag makes [`selected.prepare`](../../../src/metal/routed_single_selected.zig) request the unmasked address path immediately when the necessary kernels and addresses are available. It does not enable the masked path by itself. Selected IDs, missing-expert loading and cache pins remain in the same preparation and lifetime machinery. Prefill already publishes cache addresses; this test introduces no new publication phase. The effect on the measured path, rather than an inferred mechanism for the delay, is the question.

A separate instrumented 32K pair confirms the intended path change: the first step changed from 40 six-slot kernels to 40 unmasked address kernels, with 51 command buffers on each side. Positive inter-buffer GPU gaps summed to 683.177 → 666.487 ms and the GPU envelope was 815.183 → 797.968 ms. Gaps still occurred immediately before address-based expert kernels. These are intrusive diagnostic observations, not comparative performance claims; they show the dispatch change did not eliminate the exposed delay. The local `first-path-comparison.json` retains the mapping.

## Profile-off timing at 32K

Five alternating pairs retained every run. The fresh step remained near 800 ms: the candidate was lower in four pairs, but reduced the mean by only 8.0 ms. Fresh full decode was also lower in four pairs, by 17.5 ms on average. This does not remove the dominant first-step delay. Steady decode excludes only the first token, so it is not a guarantee that all initialization has finished.

| Pair (first arm) | Fresh first step, baseline → candidate (ms) | Fresh 128-token decode, baseline → candidate (s) | Total inference, baseline → candidate (s) | Process wall, baseline → candidate (s) |
| --- | ---: | ---: | ---: | ---: |
| 1 (baseline) | 802.848 → 821.011 | 4.317 → 4.340 | 57.746 → 59.445 | 59.935 → 61.608 |
| 2 (candidate) | 805.010 → 797.284 | 4.316 → 4.297 | 63.850 → 63.941 | 65.982 → 66.136 |
| 3 (baseline) | 815.817 → 810.791 | 4.333 → 4.310 | 63.794 → 63.263 | 66.027 → 65.396 |
| 4 (candidate) | 824.975 → 794.709 | 4.358 → 4.307 | 63.719 → 63.486 | 65.892 → 65.680 |
| 5 (baseline) | 823.269 → 808.078 | 4.333 → 4.316 | 63.697 → 63.587 | 65.877 → 65.740 |
| Mean ± sample SD | 814.384 ± 10.175 → 806.375 ± 10.670 | 4.331 ± 0.017 → 4.314 ± 0.016 | 62.561 ± 2.692 → 62.744 ± 1.861 | 64.743 ± 2.688 → 64.912 ± 1.866 |

Fresh steady throughput was 36.110 ± 0.095 → 36.210 ± 0.085 tokens/s. Warm 128-token decode averaged 3.870 ± 0.189 → 3.936 ± 0.033 s; the first baseline run was unusually fast, and paired changes were mixed. Total inference and wall changes are also mixed; these data do not demonstrate a whole-workload gain or establish where any cost moved. Fresh and warm inference totals include prefill and decode but exclude model loading and snapshot save/restore. Wall includes the process. Seconds for inference phases are derived from two-decimal throughput values. The expert cache survives KV restore; the OS file cache was not flushed. The local raw `timing32k/timing-samples.csv` retains all phase samples.

## Profile-off timing at 8K

At the shorter frontier, the candidate first step was slower in four of five pairs and averaged 12.2 ms longer. Complete fresh decode was faster in three pairs, while inference and wall were faster in four. The variation and the opposite first-step result limit attribution to this dispatch change.

| Pair (first arm) | Fresh first step, baseline → candidate (ms) | Fresh 128-token decode, baseline → candidate (s) | Total inference, baseline → candidate (s) | Process wall, baseline → candidate (s) |
| --- | ---: | ---: | ---: | ---: |
| 1 (baseline) | 803.038 → 842.927 | 4.236 → 4.272 | 33.523 → 32.656 | 35.573 → 34.775 |
| 2 (candidate) | 809.433 → 812.381 | 4.411 → 4.355 | 33.631 → 33.125 | 35.705 → 35.161 |
| 3 (baseline) | 802.136 → 810.576 | 4.440 → 4.230 | 34.103 → 33.179 | 36.166 → 35.280 |
| 4 (candidate) | 801.052 → 831.382 | 4.227 → 4.260 | 34.081 → 33.726 | 36.173 → 35.806 |
| 5 (baseline) | 812.527 → 791.918 | 4.457 → 4.208 | 34.639 → 34.829 | 36.738 → 36.895 |
| Mean ± sample SD | 805.637 ± 5.048 → 817.837 ± 19.794 | 4.354 ± 0.113 → 4.265 ± 0.056 | 33.996 ± 0.444 → 33.503 ± 0.833 | 36.071 ± 0.460 → 35.583 ± 0.820 |

Fresh steady throughput was 35.820 ± 1.121 → 36.850 ± 0.568 tokens/s. Warm decode averaged 3.657 ± 0.130 → 3.620 ± 0.139 s. The same timing scopes and precision limits apply as at 32K. All phase samples are in the local raw `timing8k/timing-samples.csv`.

## Quality

At each 8K and 32K fresh frontier and its corresponding plus-512 warm frontier, 128 fixed-history full-vocabulary teacher-logit rows were byte-identical across the two sides. All 512 post-evaluation hashes also matched. Delta NLL and KL were zero for these rows, with no top-one changes. This is a bounded numerical comparison, not autonomous task quality.

## Reproduction

All five guarded jobs completed successfully at pressure level 1, with no adverse samples and swap unchanged at 28.983 GiB. The retained raw manifest, guard records, logs and generated samples live under `.workspaces/hillclimb-20261104/22-first-decode-admission/`. Quality used the existing pressure guard around the following command:

```sh
nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/hillclimb-20261104/22-first-decode-admission/quality8k quality --flag DS4_METAL_ENABLE_STREAMING_EXPERT_ADDR_TABLE --control-unset --candidate 1 --frontier 8192 --tokens 128 --extra-settings speed-bench/experiments/2026-11-05-gpu-submit-cadence/candidate-settings.json
```

The 32K comparison used the same guard and fixed settings:

```sh
nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/hillclimb-20261104/22-first-decode-admission/timing32k timing --flag DS4_METAL_ENABLE_STREAMING_EXPERT_ADDR_TABLE --control-unset --candidate 1 --frontier 32768 --tokens 128 --samples 5 --extra-settings speed-bench/experiments/2026-11-05-gpu-submit-cadence/candidate-settings.json
```

The 8K timing command differs only in output directory and frontier:

```sh
nu speed-bench/experiments/2026-11-04-decode-pass/run.nu .workspaces/hillclimb-20261104/22-first-decode-admission/timing8k timing --flag DS4_METAL_ENABLE_STREAMING_EXPERT_ADDR_TABLE --control-unset --candidate 1 --frontier 8192 --tokens 128 --samples 5 --extra-settings speed-bench/experiments/2026-11-05-gpu-submit-cadence/candidate-settings.json
```

The 32K quality command uses `quality32k` as its output directory and `--frontier 32768`; otherwise it is identical to the 8K quality command. The resulting experimental package adds the one flag to the fixed settings above; subsequent package settings are recorded in [candidate-settings.json](candidate-settings.json).
