# Native Flint prefill shaders

This is a Metal implementation of Flint's ordinary, chunked BF16 prefill. It
does not load MLX or embed Python. `src/models/flint/prefill.zig` encodes the
graph, and `prefill_shaders.zig` assembles these source files for the existing
Metal compiler/queue.

- `ops.metal`: embedding, norms, split-half RoPE, activation/residual operations,
  routing/counting-sort, expert combination and FullModel KV publication.
- `matmul.metal`: affine group-64 INT4 dense/expert matmul, INT8 entry, and BF16
  router matmul using Apple's M5 NAX tile implementation.
- `attention.metal`: BF16, 128-dimensional heads, GQA factor 8, and SDPA with
  learned sinks. Causal and window 513 bounds are computed in the kernel; no dense score/mask tensor is
  allocated. Global attention bypasses RoPE.
- `matmul_simd.metal` and `attention_simd.metal`: portable 8×8 SIMD-group
  matrix operations for the M4 Pro profile. These use 32-row matrix and
  query/key tiles, share the same affine/BF16 boundaries, and contain no NAX
  or MetalPerformancePrimitives dependency. Their entry names end in `_simd`
  so a driver's pipeline cache can hold both backends during qualification.
- `vendor/`: the required Metal header closure from MLX 0.32.0, copyright Apple,
  under the included [MIT license](vendor/LICENSE). No MLX host/runtime source is
  linked. These headers retain their original text; shader assembly fixes the
  three quantized alignment constants for Flint's widths.

Dense batches with at most 8 rows and routed batches with at most 768 rows use
SIMD matrix-vector products. The routed row count is six times the token count.
Larger batches use 64-row NAX tiles. The small path uses MLX's packed integer
dot with affine scale/bias accumulated in FP32; the NAX path dequantizes weight
tiles to BF16. Projection outputs are BF16 in both cases. This avoids computing
an entire 64-row matrix tile for each sparsely populated expert in a short prompt.

The matrix/attention wrappers derive from MLX's `quantized_nax.h`, `quantized.h`,
and `steel/attn/kernels/steel_attention{,_nax}.h`. The original repository is
<https://github.com/ml-explore/mlx>. Changes are the argument blocks, fixed entry
points, and analytic local-window bounds. The model graph and rounding rules
follow <https://github.com/recursive-oss/rec>, revision
`e53663b262af4cb1ca498e596ca8b1a2d79e0800`, under
[Apache-2.0](../LICENSE.rec).

KV uses the resident decoder's four banks and `[K/V,4,capacity,128]` layout.
Local prefill first assembles the prior chronological ring plus the chunk,
attends, then compacts the final 513 entries back into the ring. Global attention
reads its bank directly below 65,536 tokens and assembles a contiguous temporary
view when spanning banks. Each successful chunk becomes a new prefill base;
the native engine retains a bounded local-ring/logits snapshot so generated
continuation can be recomputed through prefill before the next prompt extension.

BF16 boundaries are deliberate: quantized projection output, Q/K norm before
RoPE, sigmoid/gate/up products, routing mixture weights, each weighted expert
product and each of the six expert additions, shared output, and residuals.
Routing currently uses an explicit descending top-six order; MLX's argpartition
does not promise that ordering. Numerical/output comparisons must measure its
effect rather than assume exact model reproduction.

`prefill.Model.backend` selects `.nax` or `.simd`.
`prefill_shaders.sourceFor(allocator, backend)` assembles the selected source;
`source(allocator)` retains the original NAX default. Source compilation and
host/GPU fixtures are separate checks from model parity and measured latency.
Testing a forced M4 Pro/M5 Pro profile on M5 Max does not qualify that physical
chip's residency or performance. Runtime results belong in the experiment report.
