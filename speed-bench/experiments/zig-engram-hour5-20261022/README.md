# Native Engram hashing and sparse reads

Completed 2026-10-22 04:13 UTC, within the one-hour ceiling. Base is the
SSD/prefix commit `ce868ee4` on `zig-modular`.

`zig/engram.zig` replaces the C helper using the real public records and SDK
bindings. It preserves hash/history masking, sparse absolute file reads, the
four persistent token workers, bounded batch dispatch, error propagation and
free/join ownership. The pthread once values come from the SDK initializer,
not a guessed Darwin structure. Original numerical and memory-policy comments
are retained. Four explicit strict-float regions preserve E4M3 decoding,
BF16 rounding, overflow and signed-zero behavior under release optimization.

All five production programs build in C and native configurations. The existing
public-object `tests/test_engram.c` passes in C/native × Debug/ReleaseFast ×
lookup unset/enabled: **eight successful fixture runs**. Its independent oracle
covers all 65,536 code/scale pairs, hash/history masks, sparse offsets above
32 bits, batch boundaries, short reads and persistent-worker error/free paths.
The older fixture that embeds the C implementation is not used to qualify Zig.
`DS4_ENGRAM_DECODE_LUT` is presence-based: unset disables it; `0` enables it too.

A serial C/Zig/Zig/C model comparison then uses the real 1017 IQ2 model, a fixed
2,048-token prompt, chunk 512, 128 teacher-forced decode tokens, and a 48 GiB
streaming-expert cache. All four full frontier-logit arrays are **identical**.
The forced tokens control the decode trajectory; this is not a generated-task
quality measurement. All runs complete under the existing pressure guard.

| Implementation | Prefill tokens/s | Decode tokens/s | Steady decode tokens/s | Whole process seconds |
| --- | ---: | ---: | ---: | ---: |
| C, first | 31.92 | 21.47 | 22.15 | 74.085 |
| Zig, first | 30.66 | 21.35 | 22.40 | 77.592 |
| Zig, second | 29.94 | 21.14 | 22.11 | 75.969 |
| C, last | 30.32 | 21.01 | 21.91 | 77.819 |

Mean decode throughput is effectively unchanged (21.240 versus 21.245 tokens/s).
Mean prefill is 2.6% lower with Zig, while the bracketing C prefill runs differ
by 5.0%; this short comparison cannot isolate a small port cost from drift.
Whole-process means differ by 1.1%. No runtime speedup is claimed.

Decision: **retain** the bounded native helper. The graph, model tensors, Metal
kernels and Objective-C host remain their original implementations. Build the
fixture with `tools/zig-build engram-test`; use `-Dnative=false` for the C
reference and `-Doptimize=Debug` for checked host code. Run the installed
`test-engram` directly. The recorded [loop](qualify.py) and [results](results.json)
include commands and binary hashes. Source/build/fixture/pressure evidence is
under `.workspaces/zig-engram-20261022/`.
