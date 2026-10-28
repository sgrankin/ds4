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
kernel void kernel_dsv41_indexer_score_candidates(
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

struct ds4_metal_args_dsv41_rope {
    uint width, heads, rows, start, inverse, stride;
    float frequencies[32];
};

kernel void kernel_dsv41_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(args.start + group.y * args.stride) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong i = ((ulong)group.y * args.heads + group.x) * args.width +
                    args.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
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
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    if (valid) x[index] = dsv41_bf16(result);
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
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
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

struct ds4_metal_args_dsv41_candidates {
    uint width;
    uint rows;
    uint start;
    uint ratio;
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
    const uint blocks = (args.width + 7u) / 8u;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    out[offset] = index.x < visible &&
        block_mask[(ulong)index.y * blocks + index.x / 8u] == 0.0f
        ? scores[offset] : -INFINITY;
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
        const float scale = dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f);
        const float x = min(abs(value) / scale, 6.0f);
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
    const float x = min(magnitude / scale, 6.0f);
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
        float scale = dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f);
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
    // Keep the existing mode2 multiply/scale/codebook operations, including
    // BF16 publication and signed zero; record the original scale here.
    const float scale = dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
    const float x = min(abs(value) / scale, 6.0f);
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
