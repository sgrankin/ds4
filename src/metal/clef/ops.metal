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
