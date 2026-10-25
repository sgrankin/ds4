# Native Clef vision checkpoint

Working-tree experiment on 2026-10-25, based on Clef plan revision
`6262ae0ee7ea`; published main remains `ce3afb7999b7`. M5 Max, 128GB.
One pressure-guarded model process at a time; all final family runs exited zero.
This qualifies the native tower components, not complete image/video decisions.

The tower is native Zig plus Metal. It uses the published BF16 vision weights,
FP32 activations, the existing Metal device/queue runtime and credited Cleffa
72-wide attention. The patch convolution is transposed byte-exactly at load from
MLX `[out,time,y,x,channel]` weights into HF patch-row order. Dense Clef and Flash
share 27 layers of width 1152; Omni also exports three DeepStack merger outputs.
No Python or MLX is linked into production.

The source is [vision.zig](../../../src/models/clef/vision.zig) and
[vision.metal](../../../src/metal/clef/vision.metal). The attention adaptation
comes from [Cleffa](https://github.com/zknpr/cleffa), pinned locally at
`24fc91e28d154ff8dadfa58462a30b437413c3f4`; the copied MIT notice is retained
beside the kernels. Reference execution uses each pinned checkpoint's MLX-VLM
tower through [vision_oracle.py](vision_oracle.py).

## Matched feature comparison

A synthetic gradient image is decoded/prepared into 256×256 pixels, then 256 HF
patch rows, grid `[1,16,16]`, yielding 64 merged feature rows. Native CPU preparation
matches the frozen reference on all 393,216 FP32 patch values. Compare all merged
features against an FP32 reference using the same BF16 weights widened to FP32.
This isolates graph/kernel differences from the default BF16 execution's rounding.

| Model/output | Dimensions | Cosine | Mean absolute difference | Maximum absolute difference |
| --- | ---: | ---: | ---: | ---: |
| Flash merged | 64×4096 | 0.99999577 | 0.0002550 | 0.06941 |
| Full merged | 64×5120 | 0.99999746 | 0.0007859 | 0.35222 |
| Omni merged | 64×2048 | 0.99995833 | 0.0023764 | 0.04066 |
| Omni DeepStack0 | 64×2048 | 0.99997382 | 0.0005634 | 0.00648 |
| Omni DeepStack1 | 64×2048 | 0.99996953 | 0.0011700 | 0.01069 |
| Omni DeepStack2 | 64×2048 | 0.99994895 | 0.0011896 | 0.01318 |

Native merged-output cosines versus the reference's ordinary BF16 path are
0.9988790/0.9997409/0.9996491 for Flash/full/Omni. These are feature comparisons,
not delta-NLL or decision accuracy. Actual decisions and ordinary images require
an end-to-end screen before claiming practical multimodal parity.

## Warm component speed

Each run performs one untimed tower invocation, then four timed invocations on
the same input. Loading, CPU preprocessing and the language backbone/head are
excluded. Medians are intentionally approximate; these are small component
screens rather than a throughput benchmark.

| Model | Native FP32 | MLX FP32 | MLX default BF16 |
| --- | ---: | ---: | ---: |
| Flash | 19.58 ms | 18.42 ms | 16.25 ms |
| Full | 19.59 ms | 19.72 ms | 16.21 ms |
| Omni incl. DeepStack | 20.67 ms | 18.25 ms | 16.14 ms |

One measured tuning attempt increased GEMM tile rows 16→64. Flash's merged output
remained bit-identical, but median time increased 19.58→30.04 ms. The source retains
the 16-row tile. No claim that larger tiles or FP32 precision improve performance.

## Record placement and reproduction

The exact media template fixtures also pass: dense 3 cases/484 token IDs and
Omni 5 cases/798 token IDs, including every 3-axis position, media row mapping,
question/option span and lexical slice. Cases cover images, multiple images,
video timestamps, audio 17 and interleaved video+audio 17. These fixtures were
produced from the pinned official record encoder and reference processor/rope
functions; they do not run learned towers. Float timestamps use libc's
round-to-even formatting to match Python's `.1f` ties (`0.25→0.2`).

```nu
nu tools/zig-build clef-vision-smoke --prefix .workspaces/clef-runtime-tests
python3 speed-bench/ideas/pressure_run.py .workspaces/vision-check/guard -- .workspaces/clef-runtime-tests/bin/clef-vision-smoke gguf/clef-flash-int4 .workspaces/clef-media-20261024/image-patches.f32 1 16 16 .workspaces/vision-check/features.f32
```

Use a new output/guard directory each time. Replace the model directory with
`gguf/clef-int4` or `gguf/clef-omni-int4` for the other towers. The oracle takes
`MODEL PATCHES OUTPUT --grid 1 16 16 --fp32`; omitting `--fp32` selects ordinary
checkpoint precision. It requires the reference-only Python environment.

Raw outputs stay ignored under `.workspaces/clef-media-20261024/`: `vision-native`,
`vision-native-warm`, `vision-native-tile64`, `full-vision-native`,
`omni-vision-native` and corresponding `*-vision-{fp32,bf16}` reference folders.
The small learned-position corner test passes with the other native head helpers.
