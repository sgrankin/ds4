// DeepSeek V4.1 keeps the RoPE tail in the quantized KV vector. Unlike V4,
// indexer Q/K have no Hadamard transform, and compressed KV has E4M3 scales.
struct ds4_metal_args_dsv41_quantize {
    uint width;
    uint rows;
    uint mode;
};

struct ds4_metal_args_dsv41_index_candidates {
    uint n_rows, n_head, head_dim, cache_f16;
    float scale;
};

// Same72-byte packet as ds4_metal_index_batch_args. Keep its declaration local:
// this shader source is concatenated before the generic indexer definitions.
struct ds4_metal_args_dsv41_index_batch {
    uint n_rows, n_tokens, n_head, head_dim;
    uint pos0, cache_f16, row_group_size, pad0;
    ulong q_token_stride, q_head_stride, weights_token_stride, score_token_stride;
    float scale;
};

// Same32-head/128-column dot and head accumulation as the dense scalar
// indexer. Only the row address changes; selected scores keep their global ID.
kernel void kernel_dsv41_indexer_score_candidates_local(
        constant ds4_metal_args_dsv41_index_candidates &args,
        device const float *q,
        device const float *weights,
        device const float *keys,
        device float *scores,
        device const int *blocks,
        threadgroup float *shared [[threadgroup(0)]],
        uint candidate [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int block = blocks[candidate / 8u];
    if (block < 0 || uint(block) >= (args.n_rows + 7ul) / 8ul) return;
    const uint row = uint(block) * 8u + candidate % 8u;
    if (row >= args.n_rows) return;
    threadgroup float *ktg = shared;
    threadgroup float *psum = ktg + 128u;
    if (tid < 128u) ktg[tid] = keys[(ulong)row * 128u + tid];
    float acc = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint head0 = 0; head0 < 32u; head0 += 4u) {
        const uint head = head0 + uint(sg);
        device const float4 *q4 = (device const float4 *)(q + (ulong)head * 128u);
        threadgroup const float4 *k4 = (threadgroup const float4 *)ktg;
        float s = dot(q4[lane], k4[lane]);
        s = simd_sum(s);
        if (lane == 0) psum[sg] = max(s * args.scale, 0.0f) * weights[head];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            acc += psum[0];
            acc += psum[1];
            acc += psum[2];
            acc += psum[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) scores[row] = acc;
}

static inline float dsv41_bf16(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}
static inline float4 dsv41_bf16(float4 v) {
    return float4(dsv41_bf16(v.x), dsv41_bf16(v.y), dsv41_bf16(v.z), dsv41_bf16(v.w));
}

struct ds4_metal_args_dsv41_hc_collapse_norm {
    uint width, rows, pre_stride, channels;
    float eps;
};

// Supplied PRE belongs to the preceding half-layer. Do not generate or select
// a new mix here. Keep the BF16 collapse boundary before the usual RMS tree;
// each thread retains at most two float4 values instead of reloading x.
kernel void kernel_dsv41_hc_collapse_norm(
        constant ds4_metal_args_dsv41_hc_collapse_norm &args,
        device const float4 *residual, device const char *previous_pre,
        device const float4 *weight, device float4 *collapsed,
        device float4 *normalized, threadgroup float *sums [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort threads [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;
    if (!sg) sums[lane] = 0;
    const uint width4 = args.width / 4u;
    device const float *pre = (device const float *)(previous_pre + (ulong)row * args.pre_stride);
    device const float4 *r = residual + (ulong)row * 4u * width4;
    float4 values[2];
    float sum = 0;
    #pragma clang loop unroll(full)
    for (uint slot = 0; slot < 2u; slot++) {
        const uint col = tid + slot * threads;
        values[slot] = 0;
        if (col < width4) {
            // Retain the old runtime HC loop and its positive-zero seed.
            // A fully unrolled expression lets fast math erase that first
            // addition and changes all-negative-zero inputs to negative zero.
            float4 v = 0;
            #pragma clang loop unroll(disable)
            for (uint h = 0; h < args.channels; ++h)
                v += r[h * width4 + col] * pre[h];
            uint4 bits = as_type<uint4>(v);
            const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
            bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
            values[slot] = as_type<float4>(bits & 0xffff0000u);
            collapsed[(ulong)row * width4 + col] = values[slot];
            sum += dot(values[slot], values[slot]);
        }
    }
    sum = simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!lane) sums[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = simd_sum(sums[lane]);
    const float scale = 1.0f / sqrt(sum / float(args.width) + args.eps);
    #pragma clang loop unroll(full)
    for (uint slot = 0; slot < 2u; slot++) {
        const uint col = tid + slot * threads;
        if (col < width4) {
            const float4 value = (values[slot] * scale) * weight[col];
            uint4 bits = as_type<uint4>(value);
            const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
            bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
            normalized[(ulong)row * width4 + col] = as_type<float4>(bits & 0xffff0000u);
        }
    }
}

static inline float dsv41_pow2_ceil(float x) {
    const uint bits = as_type<uint>(x);
    return as_type<float>((bits & 0x7f800000u) +
                         ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

kernel void kernel_dsv41_bf16_linear(
        constant ulong &count,
        device uint *x,
        uint gid [[thread_position_in_grid]]) {
    const ulong first = (ulong)gid * 4u;
    if (first + 4u <= count) {
        uint4 bits = *((device uint4 *)(x + first));
        const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
        bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
        *((device uint4 *)(x + first)) = bits & 0xffff0000u;
    } else {
        for (ulong i = first; i < count; i++) {
            uint bits = x[i];
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16u) & 1u);
            x[i] = bits & 0xffff0000u;
        }
    }
}

struct ds4_metal_args_dsv41_router {
    uint n_expert, top_k, has_bias;
    float scale;
};

// One-token router for up to 1024 experts in one dispatch: the softplus/sqrt
// probability transform, the biased top-k through the same bitonic network as
// kernel_argsort_f32_i32_desc (padding indices sort last), and the weights
// normalised as the generic chain does: gather, SIMD sum over the top_k lanes,
// clamp, divide, scale. Threadgroup memory holds nth ints and 2 * nth floats.
kernel void kernel_dsv41_router_one(
        constant ds4_metal_args_dsv41_router &args,
        device const float *logits,
        device const float *bias,
        device float *probs,
        device int32_t *selected,
        device float *weights,
        threadgroup int32_t *shmem_i32 [[threadgroup(0)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint2 tpos [[thread_position_in_threadgroup]],
        uint2 tsize [[threads_per_threadgroup]]) {
    const uint col = tpos.x, ntg = tsize.x;
    const int width = (int)args.n_expert;
    logits += tg.y * args.n_expert;   /* one threadgroup per row */
    probs += tg.y * args.n_expert;
    selected += tg.y * args.top_k;
    weights += tg.y * args.top_k;
    threadgroup float *score = (threadgroup float *)(shmem_i32 + ntg);
    threadgroup float *prob = score + ntg;
    shmem_i32[col] = (int)col;
    if ((int)col < width) {
        const float x = logits[col];
        const float ex = exp(x);
        const float em = min(ex, 0.03125f);
        const float poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
        const float sp = select(select(log(1.0f + ex), poly, ex < 0.03125f), x, x > 20.0f);
        const float p = sqrt(sp);
        probs[col] = p;
        prob[col] = p;
        score[col] = args.has_bias ? p + bias[col] : p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 2; k <= (int)ntg; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = (int)col ^ j;
            if (ixj > (int)col) {
                const int a = shmem_i32[col], b = shmem_i32[ixj];
                const bool swap = ((int)col & k) == 0 ?
                    (a >= width || (b < width && score[a] < score[b])) :
                    (b >= width || (a < width && score[a] > score[b]));
                if (swap) { shmem_i32[col] = b; shmem_i32[ixj] = a; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (col < args.top_k) {
        const int32_t idx = shmem_i32[col];
        selected[col] = idx;
        const float w = prob[idx];
        const float total = clamp(simd_sum(w), 6.103515625e-5f, INFINITY);
        // two stores as in the generic chain: the quotient is rounded before the scale
        threadgroup volatile float *quotient = (threadgroup volatile float *)(prob + ntg) + col;
        *quotient = w / total;
        weights[col] = *quotient * args.scale;
    }
}

struct ds4_metal_args_dsv41_rope {
    uint width, heads, rows, start, inverse, stride;
    float frequencies[32];
    uint round_all;
};

kernel void kernel_dsv41_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(args.start + group.y * args.stride) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong head = ((ulong)group.y * args.heads + group.x) * args.width;
    if (args.round_all) {
        for (uint k = lane; k + 64u < args.width; k += 32u) x[head + k] = dsv41_bf16(x[head + k]);
    }
    const ulong i = head + args.width - 64u + 2u * lane;
    float re = x[i], im = x[i + 1u];
    if (args.round_all) { re = dsv41_bf16(re); im = dsv41_bf16(im); }
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

kernel void kernel_dsv41_quantize(
        constant ds4_metal_args_dsv41_quantize &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    const ulong index = (ulong)group.y * args.width + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        // Exact divisions: a fast-math reciprocal can land one ulp past a
        // rounding midpoint and flip the E4M3 scale or E2M1 value.
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(precise::divide(max(amax, 0.01171875f), 6.0f))
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(precise::divide(abs(value), scale)), value) * scale;
    }
    if (valid) x[index] = dsv41_bf16(result);
}

struct ds4_metal_args_dsv41_rope_quantize {
    uint width, start, inverse, mode;
    float frequencies[32];
};

// One row's rope, quantization and store into a cache row: the rotation of
// the last 64 columns as kernel_dsv41_rope applies it, then the block
// quantization of kernel_dsv41_quantize, written to dst.
static inline void dsv41_rope_quantize_body(
        constant ds4_metal_args_dsv41_rope_quantize &args,
        device const float *x,
        device float *dst,
        uint group,
        uint lane) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group * block + lane;
    const bool valid = lane < block && column < args.width;
    float value = valid ? x[column] : 0.0f;
    const float other = simd_shuffle_xor(value, 1u);
    const uint rotary = args.width - 64u;
    if (valid && column >= rotary) {
        const float theta = float(args.start) * args.frequencies[(column - rotary) >> 1u];
        const float c = precise::cos(theta);
        const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
        const float re = (column & 1u) ? other : value;
        const float im = (column & 1u) ? value : other;
        // the contraction kernel_dsv41_rope compiles to
        value = (column & 1u) ? dsv41_bf16(fma(im, c, re * s)) : dsv41_bf16(fma(re, c, -(im * s)));
    }
    value = dsv41_bf16(value);
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(precise::divide(max(amax, 0.01171875f), 6.0f))
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(precise::divide(abs(value), scale)), value) * scale;
    }
    if (valid) dst[column] = dsv41_bf16(result);
}

kernel void kernel_dsv41_rope_quantize(
        constant ds4_metal_args_dsv41_rope_quantize &args,
        device const float *x,
        device float *dst,
        uint group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    dsv41_rope_quantize_body(args, x, dst, group, lane);
}

/* The decode indexer query's rope and quantization in one dispatch over every head, in place: dsv41_rope_quantize_body
 * with (cos, sin) read from the step's RoPE table (the same theta and contraction, so the same bits). */
kernel void kernel_dsv41_rope_quantize_heads_tab(
        constant ds4_metal_args_dsv41_rope_quantize &args,
        device float *x,
        device const float2 *tab,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    device float *row = x + (ulong)group.y * args.width;
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    float value = valid ? row[column] : 0.0f;
    const float other = simd_shuffle_xor(value, 1u);
    const uint rotary = args.width - 64u;
    if (valid && column >= rotary) {
        const float2 cs = tab[(column - rotary) >> 1u];
        const float c = cs.x;
        const float s = args.inverse ? -cs.y : cs.y;
        const float re = (column & 1u) ? other : value;
        const float im = (column & 1u) ? value : other;
        /* the contraction kernel_dsv41_rope compiles to */
        value = (column & 1u) ? dsv41_bf16(fma(im, c, re * s)) : dsv41_bf16(fma(re, c, -(im * s)));
    }
    value = dsv41_bf16(value);
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(precise::divide(max(amax, 0.01171875f), 6.0f))
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(precise::divide(abs(value), scale)), value) * scale;
    }
    if (valid) row[column] = dsv41_bf16(result);
}

struct ds4_metal_args_dsv41_norm_pair {
    uint n0, n1, ntg0, ntg1;
    float eps;
};

// Two weighted RMS norms in one dispatch, each row on its own thread count
// (ntg0 threads then ntg1), so every reduction matches the single-row kernel.
kernel void kernel_dsv41_norm_pair(
        constant ds4_metal_args_dsv41_norm_pair &args,
        device const float4 *x0,
        device const float4 *w0,
        device float4 *y0,
        device const float4 *x1,
        device const float4 *w1,
        device float4 *y1,
        threadgroup float *shmem [[threadgroup(0)]],
        uint2 tg [[threadgroup_position_in_grid]],
        ushort2 tpos [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const ushort tid = tpos.x;
    const bool second = tid >= args.ntg0;
    const uint ntg = second ? args.ntg1 : args.ntg0;
    const uint t = second ? tid - args.ntg0 : tid;
    const uint sg = second ? sgitg - args.ntg0 / 32u : sgitg;
    const int n = int(second ? args.n1 : args.n0);
    const int ne00_t = n / 4;
    const uint row = tg.y;   /* grid row = activation row */
    device const float4 *x = (second ? x1 : x0) + (ulong)row * ne00_t;
    device const float4 *w = second ? w1 : w0;
    device float4 *y = (second ? y1 : y0) + (ulong)row * ne00_t;
    threadgroup float *sh = shmem + (second ? 32u : 0u);
    if (sg == 0) {
        sh[tiisg] = 0.0f;
    }
    float sumf = 0.0f;
    for (int i = int(t); i < ne00_t; i += int(ntg)) {
        sumf += dot(x[i], x[i]);
    }
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sh[sg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = sh[tiisg];
    sumf = simd_sum(sumf);
    const float mean  = sumf/n;
    const float scale = 1.0f/sqrt(mean + args.eps);
    for (int i = int(t); i < ne00_t; i += int(ntg)) {
        float4 v = x[i]*scale;
        v = v*w[i];
        y[i] = ds4_bf16_round(v);
    }
}

kernel void kernel_dsv41_prerope_quantize(
        constant ds4_metal_args_dsv41_quantize &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < 64u;
    const ulong index = (ulong)group.y * args.width + args.width - 64u + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(precise::divide(max(amax, 0.01171875f), 6.0f))
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(precise::divide(abs(value), scale)), value) * scale;
    }
    if (valid) x[index] = dsv41_bf16(result);
}

struct ds4_metal_args_dsv41_engram {
    uint width;
    uint rows;
    float eps;
    uint masked;
};

kernel void kernel_dsv41_engram_add(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (args.masked && !mask[group.x]) return;
    const ulong offset = ((ulong)group.x * 4u + group.y) * args.width;
    const ulong key_offset = ((ulong)group.x * 5u + group.y) * args.width;
    const ulong value_offset = ((ulong)group.x * 5u + 4u) * args.width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint i = lane; i < args.width; i += 32u) {
        const float h = residual[offset + i];
        const float k = dsv41_bf16(kv[key_offset + i]);
        const uint wi = group.y * args.width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (q_weight[wi] * k_weight[wi]) * k;
    }
    h2 = simd_sum(h2);
    k2 = simd_sum(k2);
    dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
          rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
    const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
    for (uint i = lane; i < args.width; i += 32u)
        residual[offset + i] = dsv41_bf16(residual[offset + i] +
            gate * dsv41_bf16(kv[value_offset + i]));
}

/* A single product, rounded as written: fast math may not reassociate it into a neighbouring product. */
static inline float dsv41_engram_mul(float a, float b) {
#pragma METAL fp math_mode(safe)
    return a * b;
}

/* kernel_dsv41_engram_add over eight SIMD groups per stream (width 5120): each lane's chain runs in order, segment by
 * segment, handing its three accumulators on, with the fma contractions the 32-lane kernel compiles to written out. */
kernel void kernel_dsv41_engram_add_tg256(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        threadgroup float *sh [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    if (args.masked && !mask[group.x]) return;
    constexpr uint W = 5120u, J = W / 256u;
    const uint e0 = sg * 32u * J + lane;
    device float *hp = residual + ((ulong)group.x * 4u + group.y) * W + e0;
    device const float *kp = kv + ((ulong)group.x * 5u + group.y) * W + e0;
    device const float *vp = kv + ((ulong)group.x * 5u + 4u) * W + e0;
    device const float *qp = q_weight + (ulong)group.y * W + e0;
    device const float *wp = k_weight + (ulong)group.y * W + e0;
    float h[J], k[J], p[J], v[J];
    for (uint j = 0; j < J; j++) {
        h[j] = hp[32u * j];
        k[j] = dsv41_bf16(kp[32u * j]);
        p[j] = dsv41_engram_mul(dsv41_engram_mul(h[j], qp[32u * j]), wp[32u * j]);
        v[j] = vp[32u * j];
    }
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint s = 0; s < 8u; s++) {
        if (sg == s) {
            if (s) { h2 = sh[lane]; k2 = sh[32u + lane]; dot = sh[64u + lane]; }
            for (uint j = 0; j < J; j++) {
                h2 = fma(h[j], h[j], h2);
                k2 = fma(k[j], k[j], k2);
                dot = fma(p[j], k[j], dot);
            }
            if (s < 7u) { sh[lane] = h2; sh[32u + lane] = k2; sh[64u + lane] = dot; }
        }
        if (s < 7u) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 7u) {
        h2 = simd_sum(h2);
        k2 = simd_sum(k2);
        dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
              rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
        const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
        if (lane == 0u) sh[96] = gate;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gate = sh[96];
    for (uint j = 0; j < J; j++)
        hp[32u * j] = dsv41_bf16(fma(gate, dsv41_bf16(v[j]), h[j]));
}

struct ds4_metal_args_dsv41_pool {
    uint width;
    uint pairs;
    uint tail;
};

kernel void kernel_dsv41_pool2(
        constant ds4_metal_args_dsv41_pool &args,
        device float *out,
        device const float *kv,
        device const float *scores,
        device const float *previous_kv,
        device const float *previous_scores,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.pairs) return;
    const long a = (long)index.y * 2 - args.tail;
    const ulong b = (ulong)(a + 1) * args.width + index.x;
    const float ka = a < 0 ? previous_kv[index.x] : kv[(ulong)a * args.width + index.x];
    const float sa = a < 0 ? previous_scores[index.x] : scores[(ulong)a * args.width + index.x];
    const float sb = scores[b], peak = max(sa, sb);
    const float ea = exp(sa - peak), eb = exp(sb - peak);
    out[(ulong)index.y * args.width + index.x] = dsv41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
}

struct ds4_metal_args_dsv41_indexer_all {
    uint rows;
    uint start;
    uint ratio;
    uint top_k;
};

// Rows with at most top_k visible keys attend to all of them.
kernel void kernel_dsv41_indexer_all(
        constant ds4_metal_args_dsv41_indexer_all &args,
        device int *ids,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.top_k || index.y >= args.rows) return;
    const uint visible = (args.start + index.y + 1u) / args.ratio;
    if (index.x < min(visible, args.top_k))
        ids[(ulong)index.y * args.top_k + index.x] = (int)index.x;
}

struct ds4_metal_args_dsv41_candidates {
    uint width;
    uint rows;
    uint start;
    uint ratio;
    uint mask_stride;
};

kernel void kernel_dsv41_candidate_blocks(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *blocks,
        device const float *unused,
        uint2 index [[thread_position_in_grid]]) {
    (void)unused;
    const uint count = (args.width + 7u) / 8u;
    if (index.x >= count || index.y >= args.rows) return;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    float best = -INFINITY;
    for (uint i = index.x * 8u; i < min(visible, (index.x + 1u) * 8u); i++)
        best = max(best, scores[(ulong)index.y * args.width + i]);
    if (visible && index.x == (visible - 1u) / 8u) best = INFINITY;
    blocks[(ulong)index.y * count + index.x] = best;
}

kernel void kernel_dsv41_candidate_filter(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *out,
        device const float *block_mask,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.rows) return;
    const ulong offset = (ulong)index.y * args.width + index.x;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    out[offset] = index.x < visible &&
        block_mask[(ulong)index.y * args.mask_stride + index.x / 8u] == 0.0f
        ? scores[offset] : -INFINITY;
}

/* The 2048 candidate blocks of a 0/-inf mask in ascending order; one
 * threadgroup of 1024 threads, four mask entries per thread and step. */
kernel void kernel_dsv41_candidate_list(
        constant uint &blocks,
        device const float *mask,
        device uint *list,
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint part[33];
    uint base = 0u;
    for (uint b0 = 0u; b0 < blocks; b0 += 4096u) {
        const uint b = b0 + tid * 4u;
        bool keep[4];
        uint count = 0u;
        for (uint j = 0u; j < 4u; j++) {
            keep[j] = b + j < blocks && mask[b + j] == 0.0f;
            count += keep[j] ? 1u : 0u;
        }
        const uint before = simd_prefix_exclusive_sum(count);
        if (lane == 31u) part[simd] = before + count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {
            const uint c = part[lane], c_before = simd_prefix_exclusive_sum(c);
            part[lane] = c_before;
            if (lane == 31u) part[32] = c_before + c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint at = base + part[simd] + before;
        for (uint j = 0u; j < 4u; j++) {
            if (keep[j] && at < 2048u) list[at] = b + j;
            at += keep[j] ? 1u : 0u;
        }
        base += part[32];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/* Positions in the candidate list back to key ids, in place and in order. */
kernel void kernel_dsv41_candidate_ids(
        constant uint &count,
        device const uint *list,
        device int *selected,
        uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const uint p = (uint)selected[i];
    selected[i] = (int)(list[p >> 3] * 8u + (p & 7u));
}

struct ds4_metal_args_dsv41_carry {
    uint width, rows, words, format, pack;
};

kernel void kernel_dsv41_carry_copy(
        constant ds4_metal_args_dsv41_carry &args,
        device uint *packed, device float *plain,
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint col = group.x * 128u + tid;
    const ulong row = group.y;
    if (args.format == 0u) {
        if (col >= args.width) return;
        device ushort *p = (device ushort *)(packed + row * args.words);
        if (args.pack) p[col] = ushort(as_type<uint>(plain[row * args.width + col]) >> 16);
        else plain[row * args.width + col] = as_type<float>(uint(p[col]) << 16);
    } else {
        const uint word = col / 32u;
        if (args.pack) {
            const bool allowed = col < args.width && plain[row * args.width + col] == 0.0f;
            const uint bits = simd_sum(allowed ? 1u << lane : 0u);
            if (!lane && word < args.words) packed[row * args.words + word] = bits;
        } else if (col < args.width) {
            const uint bits = packed[row * args.words + word];
            plain[row * args.width + col] = bits & (1u << lane) ? 0.0f : -INFINITY;
        }
    }
}

struct ds4_metal_args_dsv41_prerope { uint width, rows, pack; };

// Preserve mode3's original per16 E4M3 scale at publication. Both input BF16
// rounding and E2M1 ties match kernel_dsv41_quantize; only storage is different.
// Keep precise divisions here and in every fused/pre-RoPE FP4 publisher. A
// reciprocal multiply can cross a midpoint (e.g. 37.5 / 7.5 == 5) and select
// code 7 instead of the even code 6, changing a stored value from 30 to 45.
kernel void kernel_dsv41_main_pack_fp4(
        constant uint2 &args, device const float *source,
        device const int *unused_ids, device uchar *out,
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    (void)unused_ids;
    if (row >= args.x) return;
    for (uint col = tid; col < 512u; col += 128u) {
        const float value = dsv41_bf16(source[(ulong)row * 512u + col]);
        float amax = abs(value);
        for (uint delta = 8u; delta; delta >>= 1u)
            amax = max(amax, simd_shuffle_xor(amax, delta));
        const float scale = dsv4_e4m3fn_dequant(precise::divide(max(amax, 0.01171875f), 6.0f));
        const float x = min(precise::divide(abs(value), scale), 6.0f);
        uint code = 0;
        float best = abs(x - dsv4_e2m1fn_values[0]);
        for (uint i = 1; i < 8u; ++i) {
            const float error = abs(x - dsv4_e2m1fn_values[i]);
            if (error < best || (error == best && !(i & 1u) && (code & 1u))) {
                code = i;
                best = error;
            }
        }
        code |= (as_type<uint>(value) >> 28u) & 8u;
        const uint neighbor = simd_shuffle_xor(code, 1u);
        if (!(col & 1u)) out[(ulong)row * 288u + col / 2u] = uchar(code | (neighbor << 4u));
        if (!(col & 15u)) {
            const uint bits = as_type<uint>(scale);
            const uint encoded = scale < 0.015625f ? uint(scale * 512.0f) :
                (((bits >> 23u) - 120u) << 3u) | ((bits >> 20u) & 7u);
            out[(ulong)row * 288u + 256u + col / 16u] = uchar(encoded);
        }
    }
}

// Recover an exact physical representation of already-decoded mode3 rows.
// Small subnormal E4M3 scales can round the original amax/6 up by 50%, so
// decoded amax can be 4*scale. Applying the fresh quantizer again is lossy.
// Try the usual scale, validate all16 values, then recover the 4*scale case.
static inline uint dsv41_main_repack_code(float magnitude, float scale) {
    const float x = min(precise::divide(magnitude, scale), 6.0f);
    uint code = 0;
    float best = abs(x - dsv4_e2m1fn_values[0]);
    for (uint i = 1; i < 8u; ++i) {
        const float error = abs(x - dsv4_e2m1fn_values[i]);
        if (error < best || (error == best && !(i & 1u) && (code & 1u))) {
            code = i;
            best = error;
        }
    }
    return code;
}

kernel void kernel_dsv41_main_repack_fp4(
        constant uint2 &args, device const float *source,
        device const int *unused_ids, device uchar *out,
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    (void)unused_ids;
    if (row >= args.x) return;
    for (uint col = tid; col < 512u; col += 128u) {
        const float value = source[(ulong)row * 512u + col];
        float amax = abs(value);
        for (uint delta = 8u; delta; delta >>= 1u)
            amax = max(amax, simd_shuffle_xor(amax, delta));
        float scale = dsv4_e4m3fn_dequant(precise::divide(max(amax, 0.01171875f), 6.0f));
        uint code = dsv41_main_repack_code(abs(value), scale);
        uint invalid = dsv4_e2m1fn_values[code] * scale != abs(value);
        for (uint delta = 8u; delta; delta >>= 1u)
            invalid |= simd_shuffle_xor(invalid, delta);
        if (invalid) {
            scale = max(amax * 0.25f, 0.001953125f);
            code = dsv41_main_repack_code(abs(value), scale);
        }
        code |= (as_type<uint>(value) >> 28u) & 8u;
        const uint neighbor = simd_shuffle_xor(code, 1u);
        if (!(col & 1u)) out[(ulong)row * 288u + col / 2u] = uchar(code | (neighbor << 4u));
        if (!(col & 15u)) {
            const uint bits = as_type<uint>(scale);
            const uint encoded = scale < 0.015625f ? uint(scale * 512.0f) :
                (((bits >> 23u) - 120u) << 3u) | ((bits >> 20u) & 7u);
            out[(ulong)row * 288u + 256u + col / 16u] = uchar(encoded);
        }
    }
}

template<typename T>
kernel void kernel_dsv41_main_unpack_fp4(
        constant uint2 &args, device const char *source,
        device const int *unused_ids, device T *out,
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    (void)unused_ids;
    const uint col = group.x * 128u + tid;
    if (col < 512u && group.y < args.x)
        out[(ulong)group.y * 512u + col] = T(dsv41_main_fp4_value(source, group.y, col));
}
template [[host_name("kernel_dsv41_main_unpack_fp4_f32")]] kernel decltype(kernel_dsv41_main_unpack_fp4<float>) kernel_dsv41_main_unpack_fp4<float>;
template [[host_name("kernel_dsv41_main_unpack_fp4_f16")]] kernel decltype(kernel_dsv41_main_unpack_fp4<half>) kernel_dsv41_main_unpack_fp4<half>;

template<typename T>
kernel void kernel_dsv41_main_gather_fp4(
        constant uint2 &args, device const char *source,
        device const int *ids, device T *out,
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const uint col = group.x * 128u + tid;
    const uint row = group.y;
    if (col >= 512u || row >= args.y) return;
    const int id = ids[row];
    out[(ulong)row * 512u + col] = id >= 0 && uint(id) < args.x
        ? T(dsv41_main_fp4_value(source, uint(id), col)) : T(0);
}
template [[host_name("kernel_dsv41_main_gather_fp4_f32")]] kernel decltype(kernel_dsv41_main_gather_fp4<float>) kernel_dsv41_main_gather_fp4<float>;
template [[host_name("kernel_dsv41_main_gather_fp4_f16")]] kernel decltype(kernel_dsv41_main_gather_fp4<half>) kernel_dsv41_main_gather_fp4<half>;

/* Same row order as get_rows_f16, without its half-to-float conversion. */
kernel void kernel_dsv41_gather_kv_f16_packed(
        constant uint2 &args, device const ushort *source,
        device const int *ids, device ushort *out,
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort threads [[threads_per_threadgroup]]) {
    const uint row = group % args.y;
    const uint column = (group / args.y) * threads + tid;
    if (column >= 512u) return;
    const int id = ids[row];
    if (id < 0 || uint(id) >= args.x) return;
    out[(ulong)row * 512u + column] = source[(ulong)id * 512u + column];
}

kernel void kernel_dsv41_prerope_copy(
        constant ds4_metal_args_dsv41_prerope &args, device ushort *packed, device float *plain,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= 64u || index.y >= args.rows) return;
    const ulong p = (ulong)index.y * 64u + index.x;
    const ulong x = (ulong)index.y * args.width + args.width - 64u + index.x;
    if (args.pack) packed[p] = ushort(as_type<uint>(plain[x]) >> 16);
    else plain[x] = as_type<float>(uint(packed[p]) << 16);
}

#ifdef DS4_METAL_HAS_TENSOR
kernel void kernel_dsv41_indexer_pack(
        constant uint4 &args,
        device const float *q, device const float *keys,
        device uint *flags, device bfloat *packed_q, device bfloat *packed_keys,
        threadgroup uint *valid [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const bool query = group < args.y;
    const uint count = query ? 32u * 128u : 64u * 128u;
    const ulong offset = query ? (ulong)group * count : (ulong)(group - args.y) * count;
    bool exact = true;
    for (uint i = tid; i < count; i += 128u) {
        const float value = query ? q[offset + i] :
            offset + i < (ulong)args.x * 128u ? keys[offset + i] : 0.0f;
        const bfloat converted = bfloat(value);
        if (query) packed_q[offset + i] = converted;
        else packed_keys[offset + i] = converted;
        exact = exact && float(converted) == value;
    }
    const bool same = simd_all(exact);
    if (!(tid % 32u)) valid[tid / 32u] = same;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) flags[group] = valid[0] && valid[1] && valid[2] && valid[3];
}

kernel void kernel_dsv41_indexer_scores_packed(
        constant uint4 &args, constant uint2 &range,
        device const float *q, device const float *weights,
        device const float *keys, device float *scores,
        device const uint *flags, device const bfloat *packed_q,
        device const bfloat *packed_keys,
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int HEADS = 32, KEYS = 64, DIM = 128;
    const uint width = args.x, token = group.y, row0 = group.x * KEYS;
    const uint visible = (args.z + token + 1u) / args.w;
    if (row0 >= visible) {
        if (tid < KEYS && row0 + tid < width)
            scores[(ulong)token * width + row0 + tid] = -INFINITY;
        return;
    }
    matmul2d<matmul2d_descriptor(HEADS, KEYS, DIM, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto result = tensor(scratch, dextents<int32_t, 2>(KEYS, HEADS));
    if (flags[range.y + token] && flags[range.x + group.x]) {
        auto query = tensor((device bfloat *)packed_q + (ulong)(range.y + token) * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto key = tensor((device bfloat *)packed_keys + (ulong)row0 * DIM,
                           dextents<int32_t, 2>(DIM, KEYS));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    } else {
        auto query = tensor((device float *)q + (ulong)token * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto all_keys = tensor((device float *)keys, dextents<int32_t, 2>(DIM, int(width)));
        auto key = all_keys.slice(0, int(row0));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEYS && row0 + tid < width) {
        float sum = 0;
        for (uint h = 0; h < HEADS; h++)
            sum += max(scratch[h * KEYS + tid] * (1.0f / 64.0f), 0.0f) * weights[token * HEADS + h];
        scores[(ulong)token * width + row0 + tid] = row0 + tid < visible ? sum : -INFINITY;
    }
}
#endif


// Physical index-FP4 pilot. The local68B row has64 code bytes then4 exponent
// bytes; upstream index cache data/scales are block-wide, a different layout.
kernel void kernel_dsv41_indexer_pack_fp4(
        constant ds4_metal_args_dsv41_index_candidates &args,
        device const float *original,
        device const float *unused_weights,
        device const float *unused_keys,
        device uchar *packed,
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    (void)unused_weights; (void)unused_keys;
    if (row >= args.n_rows) return;
    const float value = dsv41_bf16(original[(ulong)row * 128u + tid]);
    const float amax = simd_max(abs(value));
    // Keep the canonical mode2 scale and exact value division, including
    // BF16 publication and signed zero; record the original scale here.
    const float scale = dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
    const float x = min(precise::divide(abs(value), scale), 6.0f);
    uint code = 0;
    float best = abs(x - dsv4_e2m1fn_values[0]);
    for (uint i = 1; i < 8; i++) {
        const float error = abs(x - dsv4_e2m1fn_values[i]);
        if (error < best || (error == best && !(i & 1u) && (code & 1u))) {
            code = i;
            best = error;
        }
    }
    code |= (as_type<uint>(value) >> 28u) & 8u;
    const uint neighbor = simd_shuffle_xor(code, 1u);
    if (!(lane & 1u)) packed[(ulong)row * 68u + tid / 2u] = uchar(code | (neighbor << 4u));
    if (!lane) packed[(ulong)row * 68u + 64u + sg] = uchar(as_type<uint>(scale) >> 23u);
}

kernel void kernel_dsv41_indexer_score_fp4(
        constant ds4_metal_args_dsv41_index_candidates &args,
        device const float *q,
        device const float *weights,
        device const uchar *packed,
        device float *scores,
        device const int *blocks,
        threadgroup float *shared [[threadgroup(0)]],
        uint work [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    uint row = work;
    if (args.cache_f16) { // Fixed pilot packet: this field selects candidates.
        const int block = blocks[work / 8u];
        if (block < 0 || uint(block) >= (args.n_rows + 7ul) / 8ul) return;
        row = uint(block) * 8u + work % 8u;
    }
    if (row >= args.n_rows) return;
    threadgroup float *ktg = shared;
    threadgroup float *psum = ktg + 128u;
    if (tid < 128u) {
        const ulong base = (ulong)row * 68u;
        const uint pair = packed[base + tid / 2u];
        const uint code = (pair >> ((tid & 1u) * 4u)) & 15u;
        const float scale = as_type<float>(uint(packed[base + 64u + tid / 32u]) << 23u);
        const float sign = as_type<float>((code & 8u) ? 0x80000000u : 0u);
        ktg[tid] = dsv41_bf16(copysign(dsv4_e2m1fn_values[code & 7u], sign) * scale);
    }
    float acc = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint head0 = 0; head0 < 32u; head0 += 4u) {
        const uint head = head0 + uint(sg);
        device const float4 *q4 = (device const float4 *)(q + (ulong)head * 128u);
        threadgroup const float4 *k4 = (threadgroup const float4 *)ktg;
        float s = dot(q4[lane], k4[lane]);
        s = simd_sum(s);
        if (lane == 0) psum[sg] = max(s * args.scale, 0.0f) * weights[head];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            acc += psum[0]; acc += psum[1]; acc += psum[2]; acc += psum[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) scores[row] = acc;
}

// One SIMDgroup owns each key. Its 32 lanes retain four consecutive values
// apiece, so scoring needs neither a shared key tile nor threadgroup barriers.
// Keep dot, SIMD reduction and head accumulation order matched to the controls.
static inline void dsv41_indexer_simd_score(
        constant ds4_metal_args_dsv41_index_candidates &args,
        device const float *q,
        device const float *weights,
        device float *scores,
        float4 key,
        uint row,
        ushort lane) {
    float acc = 0.0f;
    for (uint head = 0; head < 32u; head++) {
        device const float4 *q4 = (device const float4 *)(q + (ulong)head * 128u);
        const float s = simd_sum(dot(q4[lane], key));
        if (lane == 0) {
#pragma clang fp contract(off) reassociate(off)
            // The old shared psum store rounds this product before adding it.
            // Do not contract the register-only equivalent into an FMA.
            const float contribution = max(s * args.scale, 0.0f) * weights[head];
            acc += contribution;
        }
    }
    if (lane == 0) scores[row] = acc;
}

static inline uint dsv41_indexer_simd_row(
        constant ds4_metal_args_dsv41_index_candidates &args,
        device const int *blocks,
        uint work) {
    if (!args.cache_f16) return work;
    const int block = blocks[work / 8u];
    if (block < 0 || uint(block) >= (args.n_rows + 7ul) / 8ul) return 0xffffffffu;
    return uint(block) * 8u + work % 8u;
}

kernel void kernel_dsv41_indexer_score_simd_f32(
        constant ds4_metal_args_dsv41_index_candidates &args,
        device const float *q,
        device const float *weights,
        device const float *keys,
        device float *scores,
        device const int *blocks,
        uint group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = dsv41_indexer_simd_row(args, blocks, group * 4u + sg);
    if (row >= args.n_rows) return; // Uniform within each independent SIMDgroup.
    const float4 key = ((device const float4 *)(keys + (ulong)row * 128u))[lane];
    dsv41_indexer_simd_score(args, q, weights, scores, key, row, lane);
}

static inline float dsv41_indexer_fp4_value(uint code, float scale) {
    const float sign = as_type<float>((code & 8u) ? 0x80000000u : 0u);
    return dsv41_bf16(copysign(dsv4_e2m1fn_values[code & 7u], sign) * scale);
}

static inline float4 dsv41_indexer_fp4_key(device const uchar *packed, uint row, ushort lane) {
    const ulong base = (ulong)row * 68u;
    const uint codes = *((device const ushort *)(packed + base + uint(lane) * 2u));
    const float scale = as_type<float>(uint(packed[base + 64u + lane / 8u]) << 23u);
    return float4(
        dsv41_indexer_fp4_value(codes & 15u, scale),
        dsv41_indexer_fp4_value((codes >> 4u) & 15u, scale),
        dsv41_indexer_fp4_value((codes >> 8u) & 15u, scale),
        dsv41_indexer_fp4_value((codes >> 12u) & 15u, scale));
}

kernel void kernel_dsv41_indexer_score_simd_fp4(
        constant ds4_metal_args_dsv41_index_candidates &args,
        device const float *q,
        device const float *weights,
        device const uchar *packed,
        device float *scores,
        device const int *blocks,
        uint group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = dsv41_indexer_simd_row(args, blocks, group * 4u + sg);
    if (row >= args.n_rows) return;
    // Every local row is 2-byte aligned. Each lane loads four codes once; the
    // eight consecutive lanes in each scale group cover 32 original values.
    const float4 key = dsv41_indexer_fp4_key(packed, row, lane);
    dsv41_indexer_simd_score(args, q, weights, scores, key, row, lane);
}

kernel void kernel_dsv41_indexer_unpack_fp4(
        constant ds4_metal_args_dsv41_index_candidates &args,
        device const uchar *packed,
        device const float *unused_weights,
        device const float *unused_keys,
        device float *plain,
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    (void)unused_weights; (void)unused_keys;
    if (row >= args.n_rows) return;
    const ulong base = (ulong)row * 68u;
    const uint pair = packed[base + tid / 2u];
    const uint code = (pair >> ((tid & 1u) * 4u)) & 15u;
    const float scale = as_type<float>(uint(packed[base + 64u + tid / 32u]) << 23u);
    plain[(ulong)row * 128u + tid] = dsv41_indexer_fp4_value(code, scale);
}

// Small-batch/non-tensor fallback: one grid dimension per query, with explicit
// causal publication bounds and the same SIMD score arithmetic as decode.
kernel void kernel_dsv41_indexer_scores_simd_fp4(
        constant ds4_metal_args_dsv41_index_batch &args,
        device const float *q,
        device const float *weights,
        device const uchar *packed,
        device float *scores,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = group.x * 4u + sg, token = group.y;
    if (row >= args.n_rows || token >= args.n_tokens) return;
    const ulong output = (ulong)token * args.n_rows + row;
    const uint visible = (args.pos0 + token + 1u) / args.row_group_size;
    if (row >= visible) {
        if (lane == 0) scores[output] = -INFINITY;
        return;
    }
    const float4 key = dsv41_indexer_fp4_key(packed, row, lane);
    float acc = 0.0f;
    for (uint head = 0; head < 32u; head++) {
        device const float4 *q4 = (device const float4 *)(q + ((ulong)token * 32u + head) * 128u);
        const float s = simd_sum(dot(q4[lane], key));
        if (lane == 0) {
#pragma clang fp contract(off) reassociate(off)
            const float contribution = max(s * args.scale, 0.0f) * weights[(ulong)token * 32u + head];
            acc += contribution;
        }
    }
    if (lane == 0) scores[output] = acc;
}

#ifdef DS4_METAL_HAS_TENSOR
// Decode the physical cache straight into the existing transient BF16 tiles;
// no persistent or intermediate F32 key array is needed.
kernel void kernel_dsv41_indexer_prepare_fp4(
        constant uint4 &args,
        device const float *q, device const uchar *keys,
        device uint *flags, device bfloat *packed_q, device bfloat *packed_keys,
        threadgroup uint *valid [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const bool query = group < args.y;
    const uint count = query ? 32u * 128u : 64u * 128u;
    const ulong offset = query ? (ulong)group * count : (ulong)(group - args.y) * count;
    bool exact = true;
    for (uint i = tid; i < count; i += 128u) {
        float value = 0.0f;
        if (query) value = q[offset + i];
        else if (offset + i < (ulong)args.x * 128u) {
            const uint row = uint((offset + i) / 128u), col = uint((offset + i) % 128u);
            const ulong base = (ulong)row * 68u;
            const uint pair = keys[base + col / 2u];
            const uint code = (pair >> ((col & 1u) * 4u)) & 15u;
            const float scale = as_type<float>(uint(keys[base + 64u + col / 32u]) << 23u);
            value = dsv41_indexer_fp4_value(code, scale);
        }
        const bfloat converted = bfloat(value);
        if (query) packed_q[offset + i] = converted;
        else packed_keys[offset + i] = converted;
        exact = exact && float(converted) == value;
    }
    const bool same = simd_all(exact);
    if (!(tid % 32u)) valid[tid / 32u] = same;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) flags[group] = valid[0] && valid[1] && valid[2] && valid[3];
}

kernel void kernel_dsv41_indexer_scores_batch_fp4(
        constant uint4 &args, constant uint2 &range,
        device const float *unused_q, device const float *weights,
        device const uchar *unused_keys, device float *scores,
        device const uint *unused_flags, device const bfloat *packed_q,
        device const bfloat *packed_keys,
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    (void)unused_q; (void)unused_keys; (void)unused_flags;
    constexpr int HEADS = 32, KEYS = 64, DIM = 128;
    const uint width = args.x, token = group.y, row0 = group.x * KEYS;
    const uint visible = (args.z + token + 1u) / args.w;
    if (row0 >= visible) {
        if (tid < KEYS && row0 + tid < width)
            scores[(ulong)token * width + row0 + tid] = -INFINITY;
        return;
    }
    matmul2d<matmul2d_descriptor(HEADS, KEYS, DIM, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto result = tensor(scratch, dextents<int32_t, 2>(KEYS, HEADS));
    auto query = tensor((device bfloat *)packed_q + (ulong)(range.y + token) * HEADS * DIM,
                        dextents<int32_t, 2>(DIM, HEADS));
    auto key = tensor((device bfloat *)packed_keys + (ulong)row0 * DIM,
                      dextents<int32_t, 2>(DIM, KEYS));
    auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
    for (uint16_t i = 0; i < dots.get_capacity(); i++)
        if (dots.is_valid_element(i)) dots[i] = 0;
    mm.run(query, key, dots);
    dots.store(result);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEYS && row0 + tid < width) {
        float sum = 0;
        for (uint h = 0; h < HEADS; h++)
            sum += max(scratch[h * KEYS + tid] * (1.0f / 64.0f), 0.0f) * weights[token * HEADS + h];
        scores[(ulong)token * width + row0 + tid] = row0 + tid < visible ? sum : -INFINITY;
    }
}
#endif

// Single-Mac decode: the block input's collapse (the weighted sum of the four streams with the previous block's
// weights) and its norm run first, then the block's own hyper-connection weights run inside the query projection's
// grid, each with its original reduction order. The query groups read the first collapse's norm; the weights' groups
// run dsv41_hc_block_input_body whole, including its own copy of the collapse and norm, without which the compiled
// Sinkhorn split's bits change.
struct ds4_deferred_hc_args {
    int32_t n_embd, n_hc, sinkhorn_iters;
    float hc_eps, norm_eps;
};
kernel void kernel_dsv41_deferred_collapse(
        constant ds4_deferred_hc_args &args,
        device const float *x, device const float *pre,
        device float *collapsed, device const float *norm_weight,
        device float *norm_dst, device atomic_uint *counter,
        threadgroup float *shared [[threadgroup(0)]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg [[threads_per_threadgroup]]) {

    const uint n_embd = (uint)args.n_embd;
    const uint n4 = n_embd >> 2;

    threadgroup float4 *row = (threadgroup float4 *)shared;
    threadgroup float  *sum_shmem = shared + n_embd;

    /* kernel_rms_norm_fuse_impl zeroes its 32 accumulator slots from
     * simdgroup 0 before the parallel sum. */
    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
    }

    /* --- Collapse with the preceding sublayer's pre weights, then its BF16
     * boundary, and the weighted RMS partial sum over the rounded values. --- */
    const float w0 = pre[0], w1 = pre[1], w2 = pre[2], w3 = pre[3];
    device const float4 *x0 = (device const float4 *)(x + 0u * n_embd);
    device const float4 *x1 = (device const float4 *)(x + 1u * n_embd);
    device const float4 *x2 = (device const float4 *)(x + 2u * n_embd);
    device const float4 *x3 = (device const float4 *)(x + 3u * n_embd);
    device float4 *collapsed4 = (device float4 *)collapsed;

    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        float4 acc = 0.0f;
        acc += x0[i] * w0;
        acc += x1[i] * w1;
        acc += x2[i] * w2;
        acc += x3[i] * w3;
        const float4 v = float4(dsv41_bf16(acc.x), dsv41_bf16(acc.y),
                                dsv41_bf16(acc.z), dsv41_bf16(acc.w));
        collapsed4[i] = v;
        row[i] = v;
        sumf += dot(v, v);
    }

    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = sum_shmem[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf / (float)args.n_embd;
    const float rscale = 1.0f / sqrt(mean + args.norm_eps);

    device const float4 *w = (device const float4 *)norm_weight;
    device float4 *y = (device float4 *)norm_dst;
    for (uint i = tid; i < n4; i += ntg) {
        const float4 v = (row[i] * rscale) * w[i];
        y[i] = float4(dsv41_bf16(v.x), dsv41_bf16(v.y),
                      dsv41_bf16(v.z), dsv41_bf16(v.w));
    }

    if (tid == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);

}

/* kernel_dsv41_deferred_collapse as two dispatches over 32 + 10 groups: group s does the original's simdgroup s (the
 * same fma chain, written out) and the norm repeats its second simd_sum on the same 32 partials, so the same bits. */
kernel void kernel_dsv41_deferred_collapse_part(
        constant ds4_deferred_hc_args &args,
        device const float *x, device const float *pre,
        device float *collapsed, device float *slots,
        uint group [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint n_embd = (uint)args.n_embd, n4 = n_embd >> 2;
    const uint i0 = group * 32u + tiisg, i1 = i0 + 1024u;
    const bool h0 = i0 < n4, h1 = i1 < n4;
    device const float4 *x0 = (device const float4 *)(x + 0u * n_embd);
    device const float4 *x1 = (device const float4 *)(x + 1u * n_embd);
    device const float4 *x2 = (device const float4 *)(x + 2u * n_embd);
    device const float4 *x3 = (device const float4 *)(x + 3u * n_embd);
    const float w0 = pre[0], w1 = pre[1], w2 = pre[2], w3 = pre[3];
    float4 a0 = 0.0f, b0 = 0.0f, c0 = 0.0f, d0 = 0.0f, a1 = 0.0f, b1 = 0.0f, c1 = 0.0f, d1 = 0.0f;
    if (h0) { a0 = x0[i0]; b0 = x1[i0]; c0 = x2[i0]; d0 = x3[i0]; }
    if (h1) { a1 = x0[i1]; b1 = x1[i1]; c1 = x2[i1]; d1 = x3[i1]; }
    device float4 *collapsed4 = (device float4 *)collapsed;
    float sumf = 0.0f;
    if (h0) {
        const float4 v = dsv41_bf16(fma(d0, w3, fma(c0, w2, fma(b0, w1, a0 * w0))));
        collapsed4[i0] = v;
        sumf += dot(v, v);
    }
    if (h1) {
        const float4 v = dsv41_bf16(fma(d1, w3, fma(c1, w2, fma(b1, w1, a1 * w0))));
        collapsed4[i1] = v;
        sumf += dot(v, v);
    }
    sumf = simd_sum(sumf);
    if (tiisg == 0) slots[group] = sumf;
}

kernel void kernel_dsv41_deferred_collapse_norm(
        constant ds4_deferred_hc_args &args,
        device const float *collapsed, device const float *slots,
        device const float *norm_weight, device float *norm_dst, device atomic_uint *counter,
        uint gid [[thread_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint n4 = (uint)args.n_embd >> 2;
    float sumf = slots[tiisg];
    sumf = simd_sum(sumf);
    const float mean  = sumf / (float)args.n_embd;
    const float rscale = 1.0f / sqrt(mean + args.norm_eps);
    if (gid < n4) {
        const float4 r = ((device const float4 *)collapsed)[gid];
        const float4 v = (r * rscale) * ((device const float4 *)norm_weight)[gid];
        ((device float4 *)norm_dst)[gid] = float4(dsv41_bf16(v.x), dsv41_bf16(v.y), dsv41_bf16(v.z), dsv41_bf16(v.w));
    }
    if (gid == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);
}

// Fixed 1280 -> 32768 query projection: each physical SIMD group owns an
// adjacent output pair and reuses its input loads for both rows. Eight SIMD
// groups therefore cover 16 output rows, with no cross-group exchange.
// Each row still has four logical NSG=4 accumulators: K tile 4 adds to slot 0,
// then the original two reduction stages produce the BF16 projection value.
kernel void kernel_dsv41_query_short_q8(
        constant ds4_metal_args_mul_mv &mv [[buffer(0)]],
        device const char *weight [[buffer(1)]], device const char *x [[buffer(2)]],
        device float *q [[buffer(3)]], constant ds4_metal_args_mv_rope &rope [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]], ushort simd [[simdgroup_index_in_threadgroup]]) {
    const uint row = group.x * 16u + simd * 2u;
    device const block_q8_0 *wr = (device const block_q8_0 *)(weight + row * mv.nb01);
    device const block_q8_0 *wi = (device const block_q8_0 *)(weight + (row + 1u) * mv.nb01);
    device const float *input = (device const float *)x;
    // Four lanes split each 32-value Q8 block into eight values per lane;
    // the other eight lane groups select the eight blocks of this K tile.
    const short ix = lane / 4, il = lane % 4;
    float partial_re[4] = {0.f, 0.f, 0.f, 0.f};
    float partial_im[4] = {0.f, 0.f, 0.f, 0.f};
    FOR_UNROLL (short tile = 0; tile < 5; ++tile) {
        const short ib = tile * 8 + ix;
        float yl[8];
        FOR_UNROLL (short i = 0; i < 8; ++i) yl[i] = input[ib * 32 + il * 8 + i];
        device const int8_t *qr = wr[ib].qs + il * 8;
        device const int8_t *qi = wi[ib].qs + il * 8;
        float sum_re = 0.f, sum_im = 0.f;
        FOR_UNROLL (short i = 0; i < 8; ++i) {
            sum_re += qr[i] * yl[i];
            sum_im += qi[i] * yl[i];
        }
        partial_re[tile % 4] += sum_re * wr[ib].d;
        partial_im[tile % 4] += sum_im * wi[ib].d;
    }
    FOR_UNROLL (short i = 0; i < 4; ++i) {
        partial_re[i] = simd_sum(partial_re[i]);
        partial_im[i] = simd_sum(partial_im[i]);
    }
    const float slot_re = lane < 4 ? partial_re[lane] : 0.f;
    const float slot_im = lane < 4 ? partial_im[lane] : 0.f;
    const float re = ds4_bf16_round(simd_sum(slot_re));
    const float im = ds4_bf16_round(simd_sum(slot_im));
    // Both BF16 values are now in the same SIMD group. Its lane 0 preserves
    // the forward RoPE multiply/FMA order and second rounding at each tail.
    if (lane == 0) {
        const uint col = row % rope.width;
        if (col >= rope.width - 64u) {
            const float theta = float(rope.start) * rope.frequencies[(col - (rope.width - 64u)) >> 1u];
            const float c = precise::cos(theta);
            const float sn = precise::sin(theta);
            q[row] = ds4_bf16_round(fma(re, c, -(im * sn)));
            q[row + 1u] = ds4_bf16_round(fma(im, c, re * sn));
        } else { q[row] = re; q[row + 1u] = im; }
    }
}

kernel void kernel_dsv41_query_deferred_hc(
        constant ds4_metal_args_hc_norm_mix &hc [[buffer(0)]],
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm &split_args [[buffer(1)]],
        device const char *residual [[buffer(2)]], device const char *hc_weight [[buffer(3)]],
        device char *mix [[buffer(4)]], device const float *scale [[buffer(5)]],
        device const float *base [[buffer(6)]], device char *split [[buffer(7)]],
        device atomic_uint *completion [[buffer(8)]],
        constant ds4_metal_args_mul_mv &mv [[buffer(9)]],
        device const char *weight [[buffer(10)]], device const char *x [[buffer(11)]],
        device float *q [[buffer(12)]], constant ds4_metal_args_mv_rope &rope [[buffer(13)]],
        device char *collapsed [[buffer(14)]], device const char *norm_weight [[buffer(15)]],
        device char *norm [[buffer(16)]], device const float *pre [[buffer(17)]],
        constant ds4_metal_args_dsv41_rope_quantize &kv_args [[buffer(18)]],
        device const float *kv [[buffer(19)]], device float *cache [[buffer(20)]],
        threadgroup char *shmem [[threadgroup(0)]], uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]], ushort simd [[simdgroup_index_in_threadgroup]]) {
    if (group.x < 16u) {
        dsv41_hc_block_input_body<2>(hc, split_args, residual, hc_weight, mix,
            scale, base, split, collapsed, norm_weight, norm, pre, completion, shmem,
            group, lane, simd);
        return;
    }
    if (group.x < 18u) {
        dsv41_rope_quantize_body(kv_args, kv, cache, (group.x - 16u) * 8u + simd, lane);
        return;
    }
    const uint tile = group.x - 18u;
    const float value = ds4_q8_short_value<40, false>(mv, weight, x, uint3(tile*2u,0,0), lane, simd);
    threadgroup float *pairs = (threadgroup float *)shmem;
    if (lane == 0) pairs[simd] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0 && (simd & 1u) == 0) {
        const uint row = tile * 8u + simd, col = row % rope.width;
        const float re = pairs[simd], im = pairs[simd + 1u];
        if (col >= rope.width - 64u) {
            const float theta = float(rope.start) * rope.frequencies[(col - (rope.width - 64u)) >> 1u];
            const float c = precise::cos(theta);
            const float sn = precise::sin(theta);
            q[row] = ds4_bf16_round(fma(re, c, -(im * sn)));
            q[row + 1u] = ds4_bf16_round(fma(im, c, re * sn));
        } else { q[row] = re; q[row + 1u] = im; }
    }
}

/* The decode reduce of flash_attn.metal with V4.1's transforming writer. */
kernel void kernel_dsv41_flash_attn_vec_reduce_heads(
        constant ds4_metal_args_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        device const float2 * rope_tab,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    ds4_flash_attn_vec_reduce_row<true>(args, htmp, dst, rope_tab, tgpig, tiisg, sgitg,
                                        (short)FC_flash_attn_ext_vec_reduce_NWG,
                                        (short)FC_flash_attn_ext_vec_reduce_DV);
}
