# Native Flint Metal kernels

The decoder comes from `recursive-oss/rec` revision
`e53663b262af4cb1ca498e596ca8b1a2d79e0800` (Apache-2.0; `LICENSE.rec`).
`tools/export-flint-kernels.py ../flint` runs its source transformations without
importing MLX or loading a model, and writes `decode.metal`. Serving compiles the
checked-in MSL through the existing Metal driver; it does not load Python or MLX.
The BF16 math overloads follow Apple's MLX source (MIT; see
`prefill/vendor/LICENSE`). The low-level packet interface is deliberately fixed
to this checkpoint's affine group-64 INT4 projections/experts and INT8 embedding,
49 layers, 256 experts/top-six routing, and BF16 KV. It is not a general GGUF or
BF16-checkpoint loader.

The resident schedule is specifically **M5 Max: 76 groups of 256 threads, 64
MLP/head groups**. Its device-wide barriers require all groups to make progress.
Other chips need a qualified schedule rather than a larger arbitrary grid.
Compile with safe math; explicit fast operations are already in the shader.

Both `flint_decode_scalar` and `flint_decode_greedy8` use these buffer bindings:

| Binding | Payload |
| --- | --- |
| 0–12 | Thirteen aligned affine weight banks |
| 13 | Metadata: `uint[50][64]` |
| 14–17 | Four BF16 KV banks |
| 18 | `uint[4]`: input token, position, initial position, layer count |
| 19 | BF16 SiLU table, 65,536 entries |
| 20 | Private atomic control, `uint[1024]`, zeroed for each launch |
| 21 | Scalar constants, or greedy `float[6]`: count, EOS count, four EOS IDs |
| 22 | BF16 vocabulary logits: 150,144 entries per step |
| 23 | BF16 layer trace: `50 × 2048` entries per step |
| 24 | One synchronization diagnostic per resident group |
| 25 | Scalar profile scratch, or greedy `uint[17]` token-chain record |
| 26 | BF16 workspace, 78,848 entries |
| 27 | FP32 attention partials, 2,097,152 entries |
| 28 | FP32 statistics, 65,536 entries |
| 29 | Routing/handoff scratch, `uint[1280]` |

The greedy kernel writes up to eight steps and stops on EOS. Its record contains
eight input/output token pairs followed by the produced count. Scalar layer count
zero means inactive and leaves KV untouched, for the filtered packet chain.

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
