// Clef-Omni glue for the existing native affine NAX/Gather-MM kernels.
#include <metal_stdlib>
using namespace metal;
struct clef_omni_prep {
    uint rows, heads, kv_heads, dim;
    float eps;
};
// Standard RMSNorm weights (not Qwen3.5's 1+w), ordinary text RoPE initially.
// Media position IDs select temporal/row/column with Omni's [24,20,20] schedule.
kernel void
clef_omni_prepare(constant clef_omni_prep &a [[buffer(0)]], device const bfloat *qkv [[buffer(1)]],
                  device const float *qw [[buffer(2)]], device const float *kw [[buffer(3)]],
                  device const int *pos [[buffer(4)]], device const float *freq [[buffer(5)]],
                  device bfloat *Q [[buffer(6)]], device bfloat *K [[buffer(7)]],
                  device bfloat *V [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                  uint lane [[thread_index_in_simdgroup]]) {
    const uint t = tg.x, h = tg.y, row = (a.heads + 2 * a.kv_heads) * 128;
    const bool query = h < a.heads;
    const uint kh = h - a.heads;
    const device bfloat *src = qkv + (ulong)t * row + h * 128;
    const device float *w = query ? qw : kw;
    float x[4], sum = 0;
    for (uint j = 0; j < 4; ++j) {
        x[j] = float(src[lane + 32 * j]);
        sum += x[j] * x[j];
    }
    const float inv = rsqrt(simd_sum(sum) / 128.0f + a.eps);
    for (uint j = 0; j < 4; ++j)
        x[j] *= inv * w[lane + 32 * j];
    for (uint j = 0; j < 2; ++j) {
        const uint pair = lane + 32 * j, axis = pair < 60 ? pair % 3 : 0;
        const float angle = float(pos[3 * t + axis]) * freq[pair], c = precise::cos(angle),
                    s = precise::sin(angle);
        const float lo = x[j], hi = x[j + 2];
        x[j] = lo * c - hi * s;
        x[j + 2] = hi * c + lo * s;
    }
    for (uint j = 0; j < 4; ++j) {
        const ulong offset = ((ulong)(query ? h : kh) * a.rows + t) * 128 + lane + 32 * j;
        if (query)
            Q[offset] = bfloat(x[j]);
        else {
            K[offset] = bfloat(x[j]);
            V[offset] = qkv[(ulong)t * row + (a.heads + a.kv_heads + kh) * 128 + lane + 32 * j];
        }
    }
}
struct clef_omni_route {
    uint rows, experts, top_k, width;
};
kernel void clef_omni_route_top8(constant clef_omni_route &p [[buffer(0)]],
                                 device const bfloat *logits [[buffer(1)]],
                                 device uint *ids [[buffer(2)]],
                                 device float *weights [[buffer(3)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    float candidates[4], values[8];
    uint chosen[8];
    for (uint i = 0; i < 4; ++i)
        candidates[i] = float(logits[row * 128 + lane + 32 * i]);
    for (uint rank = 0; rank < 8; ++rank) {
        float best = -INFINITY;
        uint index = ~0u;
        for (uint i = 0; i < 4; ++i) {
            const uint id = lane + 32 * i;
            if (candidates[i] > best || (candidates[i] == best && id < index)) {
                best = candidates[i];
                index = id;
            }
        }
        const float maximum = simd_max(best);
        const uint winner = simd_min(best == maximum ? index : ~0u);
        chosen[rank] = winner;
        values[rank] = maximum;
        for (uint i = 0; i < 4; ++i)
            if (lane + 32 * i == winner)
                candidates[i] = -INFINITY;
    }
    if (lane == 0) {
        const float peak = values[0];
        float sum = 0;
        for (uint r = 0; r < 8; ++r) {
            values[r] = exp(values[r] - peak);
            sum += values[r];
        }
        for (uint r = 0; r < 8; ++r) {
            ids[row * 8 + r] = chosen[r];
            weights[row * 8 + r] = values[r] / sum;
        }
    }
}
kernel void clef_omni_histogram(constant clef_omni_route &p [[buffer(0)]],
                                device const uint *ids [[buffer(1)]],
                                device atomic_uint *counts [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
    if (i < p.rows * 8)
        atomic_fetch_add_explicit(counts + ids[i], 1, memory_order_relaxed);
}
kernel void
clef_omni_scatter(constant clef_omni_route &p [[buffer(0)]], device const bfloat *x [[buffer(1)]],
                  device const uint *ids [[buffer(2)]], device const uint *offsets [[buffer(3)]],
                  device atomic_uint *cursors [[buffer(4)]], device uint *permutation [[buffer(5)]],
                  device uint *sorted_ids [[buffer(6)]], device bfloat *sorted_x [[buffer(7)]],
                  uint row [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint destination;
    if (tid == 0) {
        const uint expert = ids[row];
        destination =
            offsets[expert] + atomic_fetch_add_explicit(cursors + expert, 1, memory_order_relaxed);
        permutation[row] = destination;
        sorted_ids[destination] = expert;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint col = tid; col < p.width; col += 128)
        sorted_x[(ulong)destination * p.width + col] = x[(ulong)(row / 8) * p.width + col];
}
struct clef_omni_rows {
    uint rows, width;
};
kernel void clef_omni_swiglu(constant clef_omni_rows &p [[buffer(0)]],
                             device const bfloat *gu [[buffer(1)]],
                             device bfloat *out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.width)
        return;
    const uint row = i / p.width, col = i % p.width;
    const ulong base = (ulong)row * 2 * p.width;
    const float gate = float(gu[base + col]), up = float(gu[base + p.width + col]);
    out[i] = bfloat((gate / (1 + exp(-gate))) * up);
}
kernel void clef_omni_combine(constant clef_omni_rows &p [[buffer(0)]],
                              device const bfloat *experts [[buffer(1)]],
                              device const uint *permutation [[buffer(2)]],
                              device const float *weights [[buffer(3)]],
                              device float *hidden [[buffer(4)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.width)
        return;
    const uint row = i / p.width, col = i % p.width;
    float sum = 0;
    for (uint rank = 0; rank < 8; ++rank)
        sum += float(experts[(ulong)permutation[row * 8 + rank] * p.width + col]) *
               weights[row * 8 + rank];
    hidden[i] = float(bfloat(hidden[i] + sum));
}
