# Native Flint Metal kernels

The decoder comes from `recursive-oss/rec` revision
`e53663b262af4cb1ca498e596ca8b1a2d79e0800` (Apache-2.0; `LICENSE.rec`).
`tools/export-flint-kernels.py ../flint` runs its source transformations without
importing MLX or loading a model, and writes ordinary plus `decode*-steer.metal`
libraries for all three profiles. Serving compiles the
checked-in MSL through the existing Metal driver; it does not load Python or MLX.
The BF16 math overloads follow Apple's MLX source (MIT; see
`prefill/vendor/LICENSE`). The low-level packet interface is deliberately fixed
to this checkpoint's affine group-64 INT4 projections/experts and INT8 embedding,
49 layers, 256 experts/top-six routing, and BF16 KV. It is not a general GGUF or
BF16-checkpoint loader.

The resident schedules mirror the source runtime:

| Profile | Resident groups × threads | MLP/head groups | Prefill chunk |
| --- | --- | --- | --- |
| M4 Pro | 36 ×256 | 32 | 2048 |
| M5 Pro | 36 ×256 | 32 | 4096 |
| M5 Max | 76 ×256 | 64 | 4096 |

Device-wide barriers require every resident group to make progress. The profile
selector rejects a larger residency envelope and the M5 tensor path on M4.
A smaller profile can be exercised on M5 Max; that does not establish its speed
or residency on the target chip. M1 Max still needs its own implementation and
qualification. Compile with safe math; explicit fast operations are in the shader.
The ordinary M5 Max entrypoints remain unchanged. M4/M5 Pro use separate generated files and
entry-point names so a cached Metal pipeline cannot silently retain another
profile's launch envelope. The exporter also retains upstream's mixed-expert
specialization flag (off on M4, on on M5); the installed INT4 checkpoint does not
take the optional INT3 specialization branch.

Both scalar and greedy entry points use these buffer bindings. M5 Max retains
`flint_decode_scalar`/`flint_decode_greedy8`; Pro variants append `_m4_pro` or
`_m5_pro`:

| Binding | Payload |
| --- | --- |
| 0–12 | Thirteen aligned affine weight banks |
| 13 | Metadata: `uint[50][64]` |
| 14–17 | Four BF16 KV banks |
| 18 | `uint[4]`: input token, position, initial position, layer count |
| 19 | BF16 SiLU table, 65,536 entries |
| 20 | Private atomic control, `uint[1024]`, zeroed for each launch |
| 21 | Scalar constants, or greedy `float[8]`: count, EOS count, four EOS IDs, attention/FFN scales |
| 22 | BF16 vocabulary logits: 150,144 entries per step |
| 23 | BF16 layer trace: `50 × 2048` entries per step |
| 24 | Diagnostics: 76-word stride; smaller profiles use the first36 words |
| 25 | Scalar profile scratch, or greedy `uint[17]` token-chain record |
| 26 | BF16 workspace, 78,848 entries |
| 27 | FP32 attention partials, 2,097,152 entries |
| 28 | FP32 statistics, 65,536 entries |
| 29 | Routing/handoff scratch, `uint[1280]` |
| 30 | Active steering only: FP32 residual directions, `[49,2048]` |

The greedy kernel writes up to eight steps and stops on EOS. Its record contains
eight input/output token pairs followed by the produced count. Scalar layer count
zero means inactive and leaves KV untouched, for the filtered packet chain.

Residual steering has two explicit modes. Both apply after the normal BF16
residual addition, `R = BF16(h + block)`:

- **Project (0, default):** `R = BF16(R - strength * d * dot(d, R))`.
  The loader normalizes each nonzero direction row to unit length.
- **Add (1):** `R = BF16(R + (strength / L) * v)`, where `L` counts nonzero
  selected rows. Original vector magnitudes are preserved. The live getter and
  setter use total user strength; the GPU scale is divided by `L` once.

Each attention or FFN hook edits its current residual. With attention strength
zero and FFN strength one, the hook edits the complete transformer-block result.
The direction file contains finite little-endian FP32 `[49,2048]` values. Zero
rows leave unselected layers unchanged. There is no stacking of direction banks.

Project uses separate `_steer` resident entrypoints and a local reduction; add
uses `_additive` entrypoints without a dot product or barrier. Their libraries
compile only when a direction file is loaded, and PSOs prepare at startup.
Prefill replaces its existing residual add/combine dispatch with the selected
fused operation. No extra device-wide phase, launch or host readback is added.
Enabled overhead still requires measurement. No vectors or both scales zero
select the original library and original 30-binding entrypoints, whose source
remains unchanged. All three hardware profiles support both modes.

Immutable vectors are shared; live FFN strengths belong to each session. Changes
apply to future evaluations while retaining existing KV influence. The no-model
`flint-smoke --compile-only` fixture checks both fused residual operations,
resident helpers, and captures against exact BF16 expectations at zero, positive
and negative strengths.

The direction builder captures the residual stream with
`DS4_METAL_GRAPH_DUMP_PREFIX`, `DS4_METAL_GRAPH_DUMP_NAME=residual` (or
`attn_residual`) and `DS4_METAL_GRAPH_DUMP_POS=0`. Each prefill layer captures the
final token's residual **after addition and before its steering edit** using a
diagnostic variant of the existing residual dispatch. After the complete
prompt, files `PREFIX_residual-L_pos0.bin` contain 2048 FP32 values for each of
49 layers. Multi-chunk prefill retains the last chunk's final row; decode never
overwrites it. Without the prefix there is no capture allocation, extra dispatch
or readback. Former module-output vector files are not the residual convention.

The failed precomputed-down-row experiment is retained only in the local
`.workspaces/steering-20261024/failed-derived-source` snapshot. It is not a
runtime option: its additional work slowed decoding, and affine BF16 tuple-sum
rounding made its projection approximation larger than expected.

`sampling.metal` implements the other packet path. A single command buffer encodes
up to eight resident scalar launches and GPU sampling/handoff launches; each
scalar has independent atomic control and diagnostics. The sampler applies
full-vocabulary nucleus filtering, min-p, top-k, then temperature. The fast path
specializes top-k ≤20; another supports ≤128. Unfiltered top-k zero is supported
with top-p zero/one and min-p zero. Other configurations fail explicitly.
Its counter-based Gumbel random stream is native and differs from MLX's keys.
The shared serving facade can fall back to scalar decoding for unsupported
packet configurations. This GPU sampler does not implement grammar masking,
presence penalties, or MLX seed-for-seed reproduction.

KV metadata slots 61/62 are BF16-element offset/capacity. Global layers use
65,536-token blocks; local layers use 513-token rings in bank zero. Prefill
compacts retained local KV chronologically and updates the initial position;
decode then starts overwriting the oldest ring slot. `src/models/flint/decode.zig`
owns these buffers and exposes their handles to the native prefill graph.
