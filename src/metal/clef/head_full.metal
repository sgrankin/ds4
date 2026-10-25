// Full learned joint head. All activations stay FP32 and learned matrices BF16.
#include <metal_stdlib>
using namespace metal;
struct hf_rows {
    uint rows, width, mode, pad;
};
struct hf_mm {
    uint rows, columns, inner, pad;
};
static inline float hf_sum(float v, threadgroup float *s, uint tid, uint nt) {
    v = simd_sum(v);
    uint lane = tid % 32, sg = tid / 32;
    if (lane == 0)
        s[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = simd_sum(lane < (nt + 31) / 32 ? s[lane] : 0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return v;
}
// A&S 7.1.26 erf approximation; max absolute error is about 1.5e-7.
static inline float hf_gelu(float x) {
    const float z = x * 0.70710678118654752f, t = 1.0f / (1.0f + 0.3275911f * fabs(z));
    const float poly =
        (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t +
         0.254829592f) *
        t;
    const float e = copysign(1.0f - poly * exp(-z * z), z);
    return 0.5f * x * (1.0f + e);
}
kernel void hf_gemv(constant hf_mm &a [[buffer(0)]], device const float *x [[buffer(1)]],
                    device const bfloat *w [[buffer(2)]], device float *y [[buffer(3)]],
                    uint2 group [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group.y, channel = group.x * 16 + sg * 4;
    float sums[4] = {0, 0, 0, 0};
    for (uint k = lane; k < a.inner; k += 32) {
        const float v = x[(ulong)row * a.inner + k];
        for (uint j = 0; j < 4; ++j)
            if (channel + j < a.columns)
                sums[j] += v * float(w[(ulong)(channel + j) * a.inner + k]);
    }
    for (uint j = 0; j < 4; ++j) {
        const float sum = simd_sum(sums[j]);
        if (lane == 0 && channel + j < a.columns)
            y[(ulong)row * a.columns + channel + j] = sum;
    }
}
kernel void hf_pool(constant hf_rows &a [[buffer(0)]], device const float *x [[buffer(1)]],
                    device const uint2 *spans [[buffer(2)]], device float *out [[buffer(3)]],
                    uint2 p [[thread_position_in_grid]]) {
    if (p.x >= a.width || p.y >= a.rows)
        return;
    const uint2 span = spans[p.y];
    float sum = 0;
    for (uint t = span.x; t < span.y; ++t)
        sum += x[(ulong)t * a.width + p.x];
    out[(ulong)p.y * a.width + p.x] = sum / float(span.y - span.x);
}
kernel void hf_post(constant hf_rows &a [[buffer(0)]], device float *x [[buffer(1)]],
                    device const float *bias [[buffer(2)]], device float *residual [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= a.rows * a.width)
        return;
    const float v = x[i] + bias[i % a.width];
    if (a.mode == 1)
        x[i] = hf_gelu(v);
    else if (a.mode == 2)
        residual[i] += v;
    else
        x[i] = v;
}
kernel void hf_options(constant hf_rows &a [[buffer(0)]], device float *ctx [[buffer(1)]],
                       device const float *lex [[buffer(2)]],
                       device const float *question [[buffer(3)]],
                       device const uint *map [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i < a.rows * a.width)
        ctx[i] = (ctx[i] + lex[i]) + question[(ulong)map[i / a.width] * a.width + i % a.width];
}
struct hf_attention {
    uint queries, keys, width, heads, q_stride, kv_stride, value_bias, pad;
};
kernel void hf_attend(constant hf_attention &a [[buffer(0)]], device const float *q [[buffer(1)]],
                      device const float *k [[buffer(2)]], device const float *v [[buffer(3)]],
                      device const float *bias [[buffer(4)]], device float *out [[buffer(5)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg.x, head = tg.y;
    float query[2], sum[2] = {0, 0};
    for (uint j = 0; j < 2; ++j)
        query[j] = q[(ulong)row * a.q_stride + head * 64 + lane + j * 32] * 0.125f;
    float maximum = -INFINITY, total = 0;
    for (uint t = 0; t < a.keys; ++t) {
        float score = 0;
        for (uint j = 0; j < 2; ++j)
            score += query[j] * k[(ulong)t * a.kv_stride + head * 64 + lane + j * 32];
        score = simd_sum(score);
        const float next = max(maximum, score), correction = exp(maximum - next),
                    p = exp(score - next);
        total = total * correction + p;
        for (uint j = 0; j < 2; ++j)
            sum[j] =
                sum[j] * correction + p * v[(ulong)t * a.kv_stride + head * 64 + lane + j * 32];
        maximum = next;
    }
    for (uint j = 0; j < 2; ++j) {
        const uint col = head * 64 + lane + j * 32;
        out[(ulong)row * a.width + col] = sum[j] / total + (a.value_bias ? bias[col] : 0.0f);
    }
}
struct hf_question {
    uint offset, count, kind, pad;
};
kernel void hf_summary(constant hf_rows &a [[buffer(0)]], device const float *base [[buffer(1)]],
                       device const float *options [[buffer(2)]],
                       device const hf_question *questions [[buffer(3)]],
                       device float *out [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[32];
    float sums[4] = {0, 0, 0, 0}, maximum = -INFINITY, total = 0;
    const hf_question question = questions[row];
    for (uint option = question.offset; option < question.offset + question.count; ++option) {
        float score = 0;
        for (uint j = 0; j < 4; ++j) {
            const uint col = tid + j * 256;
            score += base[(ulong)row * 1024 + col] * options[(ulong)option * 1024 + col];
        }
        score = hf_sum(score, scratch, tid, 256) * 0.03125f;
        const float next = max(maximum, score), corr = exp(maximum - next), p = exp(score - next);
        total = total * corr + p;
        for (uint j = 0; j < 4; ++j)
            sums[j] = sums[j] * corr + p * options[(ulong)option * 1024 + tid + j * 256];
        maximum = next;
    }
    for (uint j = 0; j < 4; ++j)
        out[(ulong)row * 1024 + tid + j * 256] = sums[j] / total;
}
kernel void hf_fields(constant hf_rows &a [[buffer(0)]], device const float *base [[buffer(1)]],
                      device const float *summary [[buffer(2)]],
                      device const float *global [[buffer(3)]],
                      device const bfloat *types [[buffer(4)]],
                      device const hf_question *questions [[buffer(5)]],
                      device float *out [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    if (i < a.rows * a.width) {
        uint row = i / a.width, col = i % a.width;
        out[i] = ((base[i] + summary[i]) + global[col]) +
                 float(types[questions[row].kind * a.width + col]);
    }
}
kernel void hf_features(constant hf_rows &a [[buffer(0)]], device const float *fields [[buffer(1)]],
                        device const float *options [[buffer(2)]],
                        device const uint *map [[buffer(3)]], device float *out [[buffer(4)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= a.rows * a.width)
        return;
    const uint row = i / a.width, col = i % a.width;
    const ulong base = (ulong)row * 4 * a.width;
    const float f = fields[(ulong)map[row] * a.width + col], o = options[i];
    out[base + col] = f;
    out[base + a.width + col] = o;
    out[base + 2 * a.width + col] = f * o;
    out[base + 3 * a.width + col] = fabs(f - o);
}
struct hf_score {
    uint rows, width, hidden, pad;
    float prior, joint, gate, bias;
};
kernel void
hf_logits(constant hf_score &a [[buffer(0)]], device const float *question [[buffer(1)]],
          device const float *global [[buffer(2)]], device const float *lexical [[buffer(3)]],
          device const float *fields [[buffer(4)]], device const float *options [[buffer(5)]],
          device const uint *map [[buffer(6)]], device const float *residual [[buffer(7)]],
          device float *out [[buffer(8)]], uint row [[threadgroup_position_in_grid]],
          uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scratch[32];
    float dot = 0, an = 0, ln = 0;
    for (uint i = tid; i < a.hidden; i += 256) {
        const float x = question[(ulong)map[row] * a.hidden + i] + global[i],
                    y = lexical[(ulong)row * a.hidden + i];
        dot += x * y;
        an += x * x;
        ln += y * y;
    }
    dot = hf_sum(dot, scratch, tid, 256);
    an = hf_sum(an, scratch, tid, 256);
    ln = hf_sum(ln, scratch, tid, 256);
    const float prior = dot / (max(sqrt(an), 1e-12f) * max(sqrt(ln), 1e-12f));
    float joint = 0, fn = 0, on = 0;
    for (uint i = tid; i < a.width; i += 256) {
        const float f = fields[(ulong)map[row] * a.width + i],
                    o = options[(ulong)row * a.width + i];
        joint += f * o;
        fn += f * f;
        on += o * o;
    }
    joint = hf_sum(joint, scratch, tid, 256);
    fn = hf_sum(fn, scratch, tid, 256);
    on = hf_sum(on, scratch, tid, 256);
    if (tid == 0)
        out[row] = a.prior * prior + a.gate * (a.joint * joint / max(sqrt(fn) * sqrt(on), 1e-8f) +
                                               residual[row] + a.bias);
}
