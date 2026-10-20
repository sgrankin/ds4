# Actual-agent three-turn V3: separate offline task evidence

The native-before, broader-three-slot and native-after agents each exited0 and
passed all three recorded held-out/unit stages. The nine recorded stage checks
passed, and each saved final project independently passed checkers0/1/2 and its
eight-test suite. Immutable CSV and existing tests stayed byte-exact; the ten
required files existed and both reports matched exact integer-cent gold.

The original V3 harness outcome remains **FAILED** for all three arms. Each
private frontend returned1 at trace validation because the original flat
replay extractor rejects compaction; the original aggregate guard returned3.
The separate frozen `02e5d3b1` offline analysis is not a rewrite of that result.

| Arm | Traced rounds | Tool calls | Compactions | Trace epochs | Traced live history tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| native-before | 17 | 20 | 1 | 2 | 3861 |
| broader3 | 19 | 22 | 1 | 2 | 4055 |
| native-after | 18 | 21 | 1 | 2 | 4078 |

The final token count is reconstructed from traced P/D/C operations and excludes
untraced later EOS/tool/transcript additions. Internal summary generation is
not token-traced; its live KV/cache effects prevent a cold-reset replay proof.
The parser accepts UTF-8 and JSON-compatible escaped text. Other producer
forms such as `\\xNN` or invalid UTF-8 fail closed.

This study establishes actual task completion for these three trajectories.
It does not establish flat replay correctness, native bit identity, a speed
benefit, physical SSD I/O savings or a default-promotion decision. Candidate
routing was exercised, but requested-load avoidance counters alone do not
measure net engine reads.

The completed guard recorded1264 normal pressure samples, no adverse samples,
zero swap growth and83.0539856GiB peak wired-plus-compressor. ROOT's retained
release proof shows the completed actual-agent job released its MAIN lock and
had no matching live model processes before the next scheduled job.

`agent-live-compaction-v3-evidence/index.json` pins compact analyses, complete
compressed typed ledgers, all eleven saved project files per arm, original
outcomes/checks, analyzer source/CPU proof, exact preparation and production
build proof, runtime source compressed from immutable `b58344d4`, and guard
pressure/release evidence. `retained-originals.json` names and hashes every
uncopied full trace/stdout/binary. All original raw traces, projects and checks
remain in the persistent V3 run; none was edited during analysis or archiving.
