# Native Metal activation, Engram and pair pooling controllers

Completed 2026-10-22, inside the one-hour ceiling. Base `af9e7863`.
Three host controllers move to a98-line handwritten Zig module: V4.1
activation quantization, Engram gating and pair pooling/carry updates. Typed
bridge additions only bind retained Metal resources and encode actual shader
packets. Existing RMS/common ModelBuffer bridge sections remain unchanged.
Shaders remain unchanged. The original C controllers remain selectable.

Comments retain alignment/environment selection, the current eight-byte linear
count packet, masked Engram fallback and the last-even input carry policy. In
particular, the old translated implementation's four-byte count was not copied.

All five C/native ReleaseFast programs build. A libc-only production-object CPU
fixture passes Debug/ReleaseFast:524 packet/alignment/environment/mask/pool-carry
and command-ownership cases per mode, plus failure checks. The actual GPU test
now has a small `--activation-dispatch` selector for its existing BF16-linear,
Engram and pool checks. This passes both C/native in default math. The complete
existing GPU suite also passes both arms with `DS4_METAL_MATH_SAFE=1`.
The previously recorded default-fast-math FP4 oracle failure is unchanged;
no shader, tolerance or math default was modified to hide it.

Four alternating C/native/native/C real1017 runs use512 prompt tokens,
128 teacher-forced decode tokens,48GiB expert cache and512-row chunks under
the pressure guard. Full-frontier logit arrays match in all four runs.

| Arm | Prefill tokens/s | Steady decode tokens/s |
| --- | ---: | ---: |
| C opening | 33.93 | 22.22 |
| Native | 34.58 | 22.25 |
| Native | 34.13 | 21.32 |
| C closing | 34.25 | 21.35 |

Mean steady decode is21.785tokens/s in both arms, with about4% bracket drift.
The result supports retaining the controller port without a noticeable runtime
regression on this case; it does not establish a model-speed gain. The full C
comparison also includes the preceding selected native modules.

Decision: **retain**. `tools/zig-build native-activation-test metal-test` builds
the actual fixtures. [Commands, source/binary hashes and results](results.json),
[guarded GPU/model loop](qualify.py). Full logs/pressure records remain under
`.workspaces/zig-metal-activation-20261022/`.
