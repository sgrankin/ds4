# Native HC projection controller

Completed 2026-10-22 04:41 UTC, inside the one-hour ceiling. Base is
`fba24cc1` on `zig-modular`. The preceding native build is the model baseline,
so only the HC controller changes in the real-model comparison.

Handwritten Zig owns eligibility, bounds, fallback choice, RMS/matrix shader
arguments and launch geometry. Objective-C retains cached PSOs, mapped weight
buffers and command encoders through a typed two-stage bridge. Both languages
share the real 144-byte RMS and 88-byte matrix packets. The original C
implementation remains the reference; no shader or numerical-policy change.

A command token spans both stages; a mapped weight retain is released after
successful owned completion or on failure. Borrowed batches stay active. Narrow
autorelease pools bound Apple temporaries, and inner buffer offsets are added
once. Environment-disable presence and the original fallback/failure distinction
are preserved. Ordinary 1017 prefill uses this path at width 20480 and at least
512 rows on this M5; other shapes retain their existing eligibility rules.

All five C/native production programs and the actual Metal fixture build.
The standalone C oracle calls the actual native controller and passes **1,152
cases / 50 fused launches** in both Debug and ReleaseFast, including ownership,
shape boundaries, disabled paths and partial failures. It links libc only.
Actual C/native `test-deepseek41-metal --hc-scaled` GPU checks also pass.

Four serial real-1017 runs use C/Zig/Zig/C controllers, 512 prompt tokens,
chunk 512, 64 forced decode tokens and a 48 GiB expert cache, under the pressure
guard. **All full frontier logits are identical.**

| Controller | Prefill tokens/s | Steady decode tokens/s | Process seconds |
| --- | ---: | ---: | ---: |
| C before | 33.55 | 22.14 | 22.396 |
| Zig | 34.13 | 21.97 | 22.290 |
| Zig | 34.00 | 21.94 | 19.984 |
| C after | 33.88 | 21.20 | 23.040 |

These short brackets establish no material runtime regression and no reliable
speedup. Process timing includes startup. The broad default Metal quantization
oracle issue and strict-IEEE qualification are documented in the preceding F16
report; this item uses the dedicated HC selector as well as actual inference.

Decision: **retain**. Build with `tools/zig-build metal-test
native-hc-projection-test`, then run the CPU binary directly and GPU/model jobs
serially under `speed-bench/ideas/pressure_run.py`. [Commands/results](results.json)
and the [real-model loop](qualify.py) are retained here; complete source hashes,
stdout/stderr and guard samples remain in `.workspaces/zig-metal-hc-20261022/`.
