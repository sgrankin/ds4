// Native Clef weight/activation glue; matrices retain the MLX affine layout.
#include <metal_stdlib>
using namespace metal;
struct clef_rows {
    uint rows, width;
};
struct clef_embedding {
    uint rows, width, bits, pad;
};
kernel void clef_embed(constant clef_embedding &a [[buffer(0)]],
                       device const uint *ids [[buffer(1)]], device const uint *w [[buffer(2)]],
                       device const bfloat *scales [[buffer(3)]],
                       device const bfloat *bias [[buffer(4)]], device float *out [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.width || gid.y >= a.rows)
        return;
    const uint k = gid.x, row = ids[gid.y], pack = 32 / a.bits, mask = (1u << a.bits) - 1;
    const uint code = (w[(ulong)row * (a.width / pack) + k / pack] >> ((k % pack) * a.bits)) & mask;
    const ulong group = (ulong)row * (a.width / 64) + k / 64;
    out[(ulong)gid.y * a.width + k] =
        float(bfloat(float(code) * float(scales[group]) + float(bias[group])));
}
kernel void clef_widen(constant clef_rows &a [[buffer(0)]], device const bfloat *x [[buffer(1)]],
                       device float *y [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < a.rows * a.width)
        y[i] = float(x[i]);
}
kernel void clef_residual(constant clef_rows &a [[buffer(0)]], device float *x [[buffer(1)]],
                          device const bfloat *y [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < a.rows * a.width)
        x[i] = float(bfloat(x[i] + float(y[i])));
}
kernel void clef_round(constant clef_rows &a [[buffer(0)]], device float *x [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < a.rows * a.width)
        x[i] = float(bfloat(x[i]));
}

// BF16-input form of Cleffa's SwiGLU in backbone.metal (see LICENSE.cleffa).
// The projection already rounded its output to BF16. Widen each loaded value
// here instead of materializing a full FP32 gate/up matrix between kernels.
struct clef_swiglu_args { uint width; };
struct clef_activation_args { int f16; float limit; };
kernel void clef_swiglu_bf16(constant clef_swiglu_args &a [[buffer(0)]],
                            device const bfloat *gate_up [[buffer(1)]],
                            device ushort *out [[buffer(2)]],
                            constant clef_activation_args &act [[buffer(3)]],
                            device const int *starts [[buffer(4)]],
                            device atomic_int *overflow [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.width)
        return;
    // The current Clef act16 convention always writes BF16 and does not clamp
    // or update overflow. Keep these bindings for the identical call contract.
    (void)act;
    (void)starts;
    (void)overflow;
    const ulong row = ulong(gid.y) * 2 * a.width;
    const float gate = float(gate_up[row + gid.x]);
    const float up = float(gate_up[row + a.width + gid.x]);
    const float value = (gate / (1.0f + exp(-gate))) * up;
    out[ulong(gid.y) * a.width + gid.x] = as_type<ushort>(bfloat(value));
}

// Opt-in numerical-oracle capture; no dispatch exists in ordinary serving.
kernel void clef_trace_rows(constant clef_rows &a [[buffer(0)]],
                            device const float *x [[buffer(1)]], device float *y [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    if (i < a.rows * a.width)
        y[i] = x[i];
}
struct clef_media_scatter_args {
    uint rows, width, additive, pad;
};
kernel void clef_media_scatter(constant clef_media_scatter_args &a [[buffer(0)]],
                               device const uint *rows [[buffer(1)]],
                               device const float *features [[buffer(2)]],
                               device float *hidden [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= a.rows * a.width)
        return;
    const ulong destination = ulong(rows[i / a.width]) * a.width + i % a.width;
    const float feature = float(bfloat(features[i]));
    hidden[destination] = a.additive ? float(bfloat(hidden[destination] + feature)) : feature;
}
