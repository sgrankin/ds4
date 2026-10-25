// Text-prefix attention adapted from zknpr/cleffa 24fc91e28d (MIT).
// See LICENSE.cleffa. Keep the ordinary Clef BF16 output boundary and the
// original separate 32-key online-softmax updates within each 64-key load.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

namespace clef_prefix {

struct act_args {
    int f16;
    float limit;
};

// Native Clef uses BF16 projection operands in both ordinary and cached paths.
static inline ushort act16(float value, constant act_args &, device atomic_int *) {
    return as_type<ushort>((bfloat)value);
}

static inline float sigmoid(float value) {
    return 1.0f / (1.0f + exp(-value));
}

static inline simdgroup_float8x8 rescale_fragment(simdgroup_float8x8 value,
                                                simdgroup_float8x8 scale) {
    thread float2 &v = reinterpret_cast<thread float2 &>(value.thread_elements());
    thread float2 &s = reinterpret_cast<thread float2 &>(scale.thread_elements());
    v = fma(v, s, float2(0.0f));
    return value;
}

struct prep_args {
    int nh, nkv, hd, n_rot, row;
    float eps;
    int kvT, koff;
};

// Q and G are compact suffix rows. K/V retain each full-attention layer's
// prefix, with a per-head stride of kvT rows and suffix insertion at koff.
// Positions already contain the suffix's absolute token positions.
[[host_name("clef_attn_prep_prefix")]]
kernel void prep(constant prep_args &a [[buffer(0)]],
                 device const float *qkv [[buffer(1)]],
                 device const float *qnorm [[buffer(2)]],
                 device const float *knorm [[buffer(3)]],
                 device const int *pos [[buffer(4)]],
                 device const float *inv_freq [[buffer(5)]],
                 device float *Q [[buffer(6)]],
                 device float *K [[buffer(7)]],
                 device float *V [[buffer(8)]],
                 device float *G [[buffer(9)]],
                 constant int &T [[buffer(10)]],
                 uint2 group [[threadgroup_position_in_grid]],
                 uint lane [[thread_index_in_simdgroup]]) {
    const int t = group.x, head = group.y;
    const int hd = a.hd, per = hd / 32;
    device const float *row = qkv + (long)t * a.row;
    const bool is_q = head < a.nh;
    const int kh = head - a.nh;
    device const float *src = is_q ? row + (long)head * 2 * hd
                                  : row + (long)a.nh * 2 * hd + (long)kh * hd;
    float value[8];
    float sum = 0.0f;
    for (int j = 0; j < per; ++j) {
        value[j] = src[lane + 32 * j];
        sum += value[j] * value[j];
    }
    sum = simd_sum(sum);
    const float inverse = rsqrt(sum / (float)hd + a.eps);
    device const float *norm = is_q ? qnorm : knorm;
    for (int j = 0; j < per; ++j)
        value[j] = value[j] * inverse * norm[lane + 32 * j];

    // Identical partial NeoX RoPE to ordinary attn_prep. Text sets all three
    // M-RoPE coordinates to the same absolute position.
    if (lane < uint(a.n_rot / 2)) {
        const float angle = float(pos[3 * t + int(lane % 3)]) * inv_freq[lane];
        const float cosine = precise::cos(angle), sine = precise::sin(angle);
        const float x0 = value[0], x1 = value[1];
        value[0] = x0 * cosine - x1 * sine;
        value[1] = x1 * cosine + x0 * sine;
    }

    if (is_q) {
        for (int j = 0; j < per; ++j) {
            Q[((long)head * T + t) * hd + lane + 32 * j] = value[j];
            G[(long)t * a.nh * hd + (long)head * hd + lane + 32 * j] =
                src[hd + lane + 32 * j];
        }
    } else {
        device const float *vsrc = row + (long)a.nh * 2 * hd +
                                   (long)a.nkv * hd + (long)kh * hd;
        for (int j = 0; j < per; ++j) {
            const long output = ((long)kh * a.kvT + t + a.koff) * hd + lane + 32 * j;
            K[output] = value[j];
            V[output] = vsrc[lane + 32 * j];
        }
    }
}

struct attention_args {
    int nh, nkv, hd, T;
    float scale;
    int kvT, qoff;
};

// One record, 256-dimensional heads, 32 suffix queries per threadgroup.
// qoff is a multiple of32, preserving every original query/key tile boundary.
// Host clears Q's8 final slack rows and K/V's unused tail rows after qoff+T
// inside each head (plus64 slack rows after the final head). Every speculative
// tile read stays inside those allocations and sees finite masked values.
[[host_name("clef_attention_prefix_64")]]
kernel void attention(constant attention_args &a [[buffer(0)]],
                      device const float *Q [[buffer(1)]],
                      device const float *K [[buffer(2)]],
                      device const float *V [[buffer(3)]],
                      device const float *G [[buffer(4)]],
                      device const int *seq_start [[buffer(5)]],
                      device ushort *O [[buffer(6)]],
                      constant act_args &act [[buffer(7)]],
                      device atomic_int *overflow [[buffer(8)]],
                      uint2 group [[threadgroup_position_in_grid]],
                      uint simd_group [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    constexpr int QB = 4, BK = 64, HD = 256, DT = HD / 8;
    constexpr int scratch_stride = 8 * BK + 64;
    threadgroup float scratch[QB * scratch_stride];
    const int first = group.x * 32, head = group.y;
    const int kv_head = head / (a.nh / a.nkv), T = a.T;
    device const float *queries = Q + ((long)head * T + first) * HD;
    device const float *keys = K + (long)kv_head * a.kvT * HD;
    device const float *values = V + (long)kv_head * a.kvT * HD;
    simdgroup_float8x8 accumulator[QB][DT / QB];
    float maximum = -INFINITY, denominator = 0.0f;
    #pragma clang loop unroll(full)
    for (int b = 0; b < QB; ++b) {
        #pragma clang loop unroll(full)
        for (int d = 0; d < DT / QB; ++d)
            accumulator[b][d] = simdgroup_float8x8(0.0f);
    }
    const int row = lane / 4, column = (lane % 4) * 8;
    const int key_end = min(first + 32, T) + a.qoff;
    for (int key_block = 0; key_block < key_end; key_block += BK) {
        simdgroup_float8x8 scores[BK / 8];
        for (int j = 0; j < BK / 8; ++j)
            scores[j] = simdgroup_float8x8(0.0f);
        for (int d = 0; d < DT; ++d) {
            simdgroup_float8x8 query = simdgroup_float8x8(0.0f);
            if (first + simd_group * 8 < T)
                simdgroup_load(query, queries + simd_group * 8 * HD + d * 8, HD);
            for (int j = 0; j < BK / 8; ++j) {
                simdgroup_float8x8 key;
                simdgroup_load(key, keys + (long)(key_block + 8 * j) * HD + d * 8,
                               HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(scores[j], query, key, scores[j]);
            }
        }
        threadgroup float *own = scratch + simd_group * scratch_stride;
        for (int j = 0; j < BK / 8; ++j)
            simdgroup_store(scores[j], own + 8 * j, BK);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Do not merge these updates: the ordinary kernel rescales once per
        // 32 keys, then applies its four P.V fragments in their original order.
        for (int sub = 0; sub < 2 && key_block + sub * 32 < key_end; ++sub) {
            threadgroup float *diagonal = own + 8 * BK;
            const int query_row = first + simd_group * 8 + row;
            const bool valid = query_row < T;
            const int start = valid ? seq_start[query_row] : 0;
            float probability[8], block_maximum = -INFINITY;
            for (int c = 0; c < 8; ++c) {
                const int key = key_block + sub * 32 + column + c;
                const bool visible = valid && key >= start && key <= query_row + a.qoff;
                probability[c] = visible ? own[row * BK + sub * 32 + column + c] * a.scale
                                         : -INFINITY;
                block_maximum = max(block_maximum, probability[c]);
            }
            block_maximum = max(block_maximum, simd_shuffle_xor(block_maximum, 1));
            block_maximum = max(block_maximum, simd_shuffle_xor(block_maximum, 2));
            const float next_maximum = max(maximum, block_maximum);
            float correction = 1.0f, sum = 0.0f;
            if (next_maximum != -INFINITY) {
                correction = maximum == -INFINITY ? 0.0f : exp(maximum - next_maximum);
                for (int c = 0; c < 8; ++c) {
                    probability[c] = probability[c] == -INFINITY ? 0.0f
                                                                : exp(probability[c] - next_maximum);
                    sum += probability[c];
                }
            } else {
                for (int c = 0; c < 8; ++c)
                    probability[c] = 0.0f;
            }
            sum += simd_shuffle_xor(sum, 1);
            sum += simd_shuffle_xor(sum, 2);
            denominator = denominator * correction + sum;
            maximum = next_maximum;
            for (int c = 0; c < 8; ++c)
                own[row * BK + sub * 32 + column + c] = probability[c];
            diagonal[row * 8 + (lane % 4) * 2] = correction;
            diagonal[row * 8 + (lane % 4) * 2 + 1] = correction;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma clang loop unroll(full)
            for (int b = 0; b < QB; ++b) {
                simdgroup_float8x8 scale;
                simdgroup_load(scale, scratch + b * scratch_stride + 8 * BK, 8);
                #pragma clang loop unroll(full)
                for (int d = 0; d < DT / QB; ++d)
                    accumulator[b][d] = rescale_fragment(accumulator[b][d], scale);
            }
            #pragma clang loop unroll(disable)
            for (int j = 0; j < 4; ++j) {
                simdgroup_float8x8 p[QB];
                #pragma clang loop unroll(full)
                for (int b = 0; b < QB; ++b)
                    simdgroup_load(p[b], scratch + b * scratch_stride + sub * 32 + 8 * j, BK);
                #pragma clang loop unroll(full)
                for (int d = 0; d < DT / QB; ++d) {
                    simdgroup_float8x8 value;
                    simdgroup_load(value,
                                   values + (long)(key_block + sub * 32 + 8 * j) * HD +
                                       simd_group * (HD / QB) + d * 8,
                                   HD);
                    #pragma clang loop unroll(full)
                    for (int b = 0; b < QB; ++b)
                        simdgroup_multiply_accumulate(accumulator[b][d], p[b], value,
                                                      accumulator[b][d]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    threadgroup float *own = scratch + simd_group * scratch_stride;
    threadgroup float *diagonal = own + 8 * BK;
    diagonal[lane] = 0.0f;
    diagonal[lane + 32] = 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane % 4 == 0)
        diagonal[row * 8 + row] = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int output_column = (lane % 4) * 2;
    #pragma clang loop unroll(full)
    for (int b = 0; b < QB; ++b) {
        const float inverse = scratch[b * scratch_stride + 8 * BK + row * 8 + row];
        #pragma clang loop unroll(full)
        for (int d = 0; d < DT / QB; ++d) {
            simdgroup_store(accumulator[b][d], own, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const int query_row = first + b * 8 + row;
            if (query_row < T) {
                for (int c = 0; c < 2; ++c) {
                    const long output = (long)query_row * a.nh * HD + (long)head * HD +
                                        simd_group * (HD / QB) + d * 8 + output_column + c;
                    O[output] = act16(fma(own[row * 8 + output_column + c], inverse, 0.0f) *
                                          sigmoid(G[output]),
                                      act, overflow + seq_start[query_row]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

} // namespace clef_prefix
