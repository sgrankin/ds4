All five frozen arms completed with guard return code 0. Both native controls reproduced all 343 full-vocabulary vectors byte for byte against REFcf5, including complete raw-file SHA, C row comparisons, and zero ΔNLL/KL/TV/argmax changes. Their exact seven-phase I/O counters are identical. Every phase in every arm retained 6136 configured slots and resident entries.

All arms held explicit cache64GiB, TILE64/LIMIT1024, context8192 and chunk512 constant. Candidate policies changed only resident routing. The frozen source, original preparation README/manifest and prepared plan remain unchanged at `1ab2106bac7fe7ce6dbd16aa154b304442c2a7cc`, based on reviewed three-slot runtime `b9408c3b8e3ec5b22ad7efff3e365fd81818c317`. No backfill, pool or suffix reuse was enabled.

| Arm | Mean ΔNLL | Mean KL | Mean TV | Flips / 343 | Max / p99 KL | Demand returned bytes | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native-before | +0.000000000 | 0.000000000 | 0.000000000 | 0 | 0.000000000 / 0.000000000 | 261,741,404,160 | +0.0000% |
| balanced2 | -0.000481409 | 0.000885105 | 0.003531218 | 3 | 0.065622877 / 0.028815325 | 243,825,500,160 | -6.8449% |
| broader2 | -0.002051647 | 0.000728112 | 0.002870688 | 3 | 0.059762281 / 0.021899205 | 238,520,401,920 | -8.8717% |
| broader3 | +0.000130948 | 0.000292227 | 0.001800548 | 2 | 0.015779930 / 0.010273734 | 233,026,191,360 | -10.9708% |
| native-after | +0.000000000 | 0.000000000 | 0.000000000 | 0 | 0.000000000 / 0.000000000 | 261,741,404,160 | +0.0000% |

Requested bytes, returned bytes and cache-loader returned bytes agree for every phase. These are direct logical demand-loader syscall counters, not physical SSD traffic or estimates from route skip counts. Candidate-prefill I/O can change after earlier approximate decode states; the first two prefill phases are identical across all five arms.

| Arm | Prefill demand bytes | Change | Decode demand bytes | Change |
| --- | ---: | ---: | ---: | ---: |
| native-before | 211,636,592,640 | +0.0000% | 50,104,811,520 | +0.0000% |
| balanced2 | 212,432,855,040 | +0.3762% | 31,392,645,120 | -37.3460% |
| broader2 | 211,437,527,040 | -0.0941% | 27,082,874,880 | -45.9476% |
| broader3 | 212,890,705,920 | +0.5926% | 20,135,485,440 | -59.8133% |
| native-after | 211,636,592,640 | +0.0000% | 50,104,811,520 | +0.0000% |

Exact per-phase I/O, decode-phase quality means, full metric CSVs, route footers, tail rows and worst positive ΔNLL are retained in `native-evidence/`. `derived-results.json` contains the complete structured comparison. Routing footers record 2610 / 3166 / 4516 substitutions for balanced2 / broader2 / broader3, with 0 / 0 / 21 triple layers and no drops. Footer skip estimates are preserved separately and are not used to compute the measured I/O savings.

| Arm | Instrumented prefill seconds | Instrumented decode seconds |
| --- | ---: | ---: |
| native-before | 64.622 | 18.238 |
| balanced2 | 65.608 | 16.981 |
| broader2 | 67.405 | 16.568 |
| broader3 | 67.348 | 16.089 |
| native-after | 67.180 | 18.211 |

These elapsed values include quality scoring, streamed raw export and I/O instrumentation. This ordered single replay is diagnostic, with no profile-off latency claim, policy promotion or independent-prompt confidence interval. Lower aggregate teacher-forced NLL is not generated-answer correctness. Broader3 has lower KL and fewer argmax changes here, but slightly positive aggregate ΔNLL; this result alone does not establish superiority on other workloads.

The guard recorded 418 pressure samples, all kernel level1, with no adverse reason or swap growth (swap remained 0.017763672GiB). Minimum free memory was 0.1134GiB; maximum wired/compressor values were 77.9401/8.6342GiB and maximum simultaneous wired+compressor was 86.4006GiB. File-backed reclaimable memory is recorded in every raw sample.

Session32356 and guard27395 completed0. ROOT independently acquired/released the MAIN flock at 22:33:54UTC with no joint probes/drivers remaining, before its next granted narrow-half job. A later child process check at22:34:34UTC confirms guard27395 and all joint processes absent; the occupied lock then belongs to ROOT narrow-half guard31276. Both time-scoped proofs are archived. No further model jobs were launched by this experiment.

Five original raw streams remain in the persistent ignored artifact output directory, each177380400bytes (886902000bytes total). Each SHA was independently rehashed after completion and is listed in `native-evidence/manifest.json`; native before/after equal REFcf5. Compact evidence is versioned in a new child of the frozen preparation. No GGUF metadata or contents were accessed after guard completion.
