All three frozen generated-code arms completed with guard return0. Each produced four completed correct first answers, passed all 34 unchanged gold checks and had no cap hits. No generated code was repaired or selected from multiple attempts; the existing score records are preserved without rerunning them.

| Arm | Correct tasks | Gold checks | Generated tokens | Cap hits |
| --- | ---: | ---: | ---: | ---: |
| native-before | 4 / 4 | 34 / 34 | 569 | 0 |
| broader3 | 4 / 4 | 34 / 34 | 574 | 0 |
| native-after | 4 / 4 | 34 / 34 | 569 | 0 |

Both native controls reproduce the historical selected64 control on all retained warm and initial full-vocabulary vector bytes, prompt token identities, generated code and sampling decisions including EOS. Candidate broader3 also has byte-exact warm and initial full vectors on all four tasks, with KL/TV zero only at these measured distributions. Its subsequent generated decisions and code differ on three of four tasks; split_escaped is exact. Do not infer all-step vector parity or KL from the warm/initial equality.

| Task | Candidate first decision difference | Generated code exact | Warm / initial vectors exact |
| --- | ---: | --- | --- |
| round_cents | 50 | False | True / True |
| merge_intervals | 63 | False | True / True |
| topological_order | 5 | False | True / True |
| split_escaped | None | True | True / True |

The candidate exercised 12519 substitutions and 135 triple layers, with 0 drops. Its route footer is retained in summary/stderr. Skipped-load/byte counters are policy diagnostics, not measured net I/O or physical SSD traffic. All arms use explicit cache 64 GiB, selected TILE64/LIMIT1024, context 4096, chunk 512, greedy/no-thinking, cap 512 and the same frozen fixture and seed. No backfill, pool or suffix reuse is enabled.

Preparation/source parent is `3248c84b108ce80428f8598d42216f16a5c95939`; frontend source revision `911c99381e90ed8027991d09d2c488476578e111`; runtime parent `b9408c3b8e3ec5b22ad7efff3e365fd81818c317`. Frozen plan SHA is 851ffa9623b568d94459372e705d2e09703a78e94e86810b3f34619919f432d2. All prepared asset hashes were rechecked without accessing GGUF metadata. The copied build/source/gold/prompt proof includes original successful compile command and 17 CPU tests.

ROOT model session 65972 completed and reacquired/released the MAIN lock with no matching live model process, as archived in root-release-proof.json. All 289 pressure samples were kernel level 1, with 0 adverse samples and 0 swap growth; peak simultaneous wired+compressor was79.5634 GiB.

Compact logs, generated code, prompt IDs, all token traces, score/generation records, source/build CPU proof, summary, ROOT preflight/launch/release and full pressure telemetry are versioned here. The 24 full-vector binary files remain at their original persistent paths; sizes and freshly checked SHA identities are recorded in raw-vector-identities.json. Timings are instrumented diagnostic values with no speed promotion. These four previously used tasks are calibration, without a broad accuracy or independent-prompt generalization claim.
