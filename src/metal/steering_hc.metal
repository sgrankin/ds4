// Residual steering: mode0 removes the selected component; mode1 adds a
// fixed offset. Every HC stream uses the same hidden-width direction.
// This source follows dsv4_hc.metal, dsv41.metal and qwen4.metal.
struct ds4_metal_hc_steering_args {
    uint width, rows, hc, layer;
    float scale;
    uint bf16, slots, part_stride, shared_src;
    float norm_eps;
    uint has_add, mode;
};

static inline float hc_steering_sum(float value, threadgroup float *scratch,
                                    uint tid, uint threads) {
    scratch[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = threads >> 1; step; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float total = scratch[0];
    // A caller may immediately reuse this workgroup scratch for another HC
    // branch. Finish every read before any thread can publish the next sum.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// Addition needs no cross-coordinate reduction and can remain in the HC
// publication. Projection publishes first, then transforms each complete stream.
kernel void kernel_hc_expand_steering(
        constant ds4_metal_args_dsv4_hc_expand & ex,
        constant ds4_metal_hc_steering_args & args,
        device const char *block, device const char *residual,
        device const char *post, device const char *comb,
        device const float *directions, device char *out,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.width * args.rows) return;
    const uint d = gid % args.width, row = gid / args.width;
    const float value = *(device const float *)(block + d * ex.nb_block0 + row * ex.nb_block1);
    float previous[4];
    for (uint src = 0; src < 4; ++src)
        previous[src] = *(device const float *)(residual + d * ex.nb_res0 + src * ex.nb_res1 + row * ex.nb_res2);
    const float shift = args.scale * directions[(ulong)args.layer * args.width + d];
    for (uint dst = 0; dst < args.hc; ++dst) {
        float acc = value * *(device const float *)(post + dst * ex.nb_post0 + row * ex.nb_post1);
        for (uint src = 0; src < args.hc; ++src)
            acc += *(device const float *)(comb + dst * ex.nb_comb0 + src * ex.nb_comb1 + row * ex.nb_comb2) * previous[src];
        if (args.bf16) acc = dsv41_bf16(acc);
        if (args.scale != 0) acc += shift;
        *(device float *)(out + d * ex.nb0 + dst * ex.nb1 + row * ex.nb2) = args.bf16 ? dsv41_bf16(acc) : acc;
    }
}

kernel void kernel_qwen4_hc_combine_steering(
        constant ds4_metal_hc_steering_args & args,
        device float *residual, device const float *block,
        device const float *injection, device const float *directions,
        threadgroup float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint threads [[threads_per_threadgroup]]) {
    if (row >= args.rows || args.width != 2560 || threads != 256) return;
    threadgroup float weights[4];
    if (tid < args.hc)
        weights[tid] = qwen4_hc_inject_weight(injection + (ulong)row * args.hc * QWEN4_HC_CHUNKS * args.hc, args.hc, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float *b = block + (ulong)row * args.width;
    device const float *dir = directions + (ulong)args.layer * args.width;
    device float *r = residual + (ulong)row * args.width * args.hc;
    float updated[4][10], dots[4] = {0, 0, 0, 0};
    for (uint j = 0; j < 10; ++j) {
        const uint d = tid + j * 256;
        for (uint stream = 0; stream < args.hc; ++stream) {
            const float value = r[stream * args.width + d] + weights[stream] * b[d];
            updated[stream][j] = value;
            dots[stream] += value * dir[d];
        }
    }
    for (uint stream = 0; stream < args.hc; ++stream) {
        const float coefficient = args.mode == 1 ? args.scale : args.scale * hc_steering_sum(dots[stream], scratch, tid, 256);
        for (uint j = 0; j < 10; ++j) {
            const uint d = tid + j * 256;
            float value = updated[stream][j];
            if (args.scale != 0) value += (args.mode == 1 ? coefficient : -coefficient) * dir[d];
            r[stream * args.width + d] = value;
        }
    }
}

// Qwen's FFN steering acts on each UPDATED residual branch, rather than on
// the FFN block. Retain those values in registers through the four reductions.
// The native Qwen family fixes E=2560 and HC=4 (ten dimensions per thread).
kernel void kernel_qwen4_moe_reduce_steering(
        constant ds4_metal_hc_steering_args & args,
        device const float *part,
        device const float *weights,
        device const float *shared_gate,
        device float *block,
        device float *residual,
        device const float *injection,
        device const float *shared,
        device const float *directions,
        threadgroup float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint threads [[threads_per_threadgroup]]) {
    if (row >= args.rows || args.width != 2560 || args.hc != 4 || threads != 256) return;
    threadgroup float inject[4];
    if (tid < 4)
        inject[tid] = qwen4_hc_inject_weight(injection + (ulong)row * 4 * QWEN4_HC_CHUNKS * 4, 4, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float *r = residual + (ulong)row * args.width * 4;
    device const float *dir = directions + (ulong)args.layer * args.width;
    float updated[4][10];
    float dots[4] = {0, 0, 0, 0};
    for (uint j = 0; j < 10; ++j) {
        const uint d = tid + j * 256;
        float acc = 0;
        for (uint slot = 0; slot < args.slots; ++slot)
            acc += weights[(ulong)row * args.slots + slot] * part[((ulong)row * args.part_stride + slot) * args.width + d];
        if (args.shared_src == 1)
            acc += qwen4_sigmoid(shared_gate[row]) * part[((ulong)row * args.part_stride + args.slots) * args.width + d];
        else if (args.shared_src == 2)
            acc += qwen4_sigmoid(shared_gate[row]) * shared[(ulong)row * args.width + d];
        block[(ulong)row * args.width + d] = acc;
        for (uint stream = 0; stream < 4; ++stream) {
            const float value = r[stream * args.width + d] + inject[stream] * acc;
            updated[stream][j] = value;
            dots[stream] += value * dir[d];
        }
    }
    for (uint stream = 0; stream < 4; ++stream) {
        const float coefficient = args.mode == 1 ? args.scale : args.scale * hc_steering_sum(dots[stream], scratch, tid, 256);
        for (uint j = 0; j < 10; ++j) {
            const uint d = tid + j * 256;
            float value = updated[stream][j];
            if (args.scale != 0) value += (args.mode == 1 ? coefficient : -coefficient) * dir[d];
            r[stream * args.width + d] = value;
        }
    }
}

// Full GLM has a plain residual stream. Sparse experts contribute two blocks;
// first combine them with the prior residual, then steer that complete sum.
kernel void kernel_residual_steering(
        constant ds4_metal_hc_steering_args & args,
        device float *block,
        device const float *block_add,
        device const float *residual,
        device const float *directions,
        device float *out,
        threadgroup float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint threads [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;
    const ulong base = (ulong)row * args.width;
    device const float *dir = directions + (ulong)args.layer * args.width;
    float dot = 0;
    for (uint d = tid; d < args.width; d += threads) {
        float value = block[base + d];
        if (args.has_add) value += block_add[base + d];
        dot += (residual[base + d] + value) * dir[d];
    }
    const float coefficient = args.mode == 1 ? args.scale : args.scale * hc_steering_sum(dot, scratch, tid, threads);
    for (uint d = tid; d < args.width; d += threads) {
        // Capture all inputs before writes: out may alias block or residual.
        float value = block[base + d];
        if (args.has_add) value += block_add[base + d];
        const float previous = residual[base + d];
        value = previous + value;
        if (args.scale != 0) value += (args.mode == 1 ? coefficient : -coefficient) * dir[d];
        out[base + d] = value;
    }
}

// Preserve the existing scalar add+weighted-RMSNorm formula and thread count.
// Projection uses the old 256-thread tree, while norm uses float4 dots and
// two SIMD reductions, as kernel_add_rms_norm_mul_f32_4 does.
kernel void kernel_residual_norm_steering(
        constant ds4_metal_hc_steering_args & args,
        device float *block,
        device const float *residual,
        device const float *directions,
        device float *sum_out,
        device const float4 *weight,
        device float4 *norm_out,
        threadgroup float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint threads [[threads_per_threadgroup]],
        uint simd_index [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (row >= args.rows) return;
    const ulong base = (ulong)row * args.width;
    device const float *dir = directions + (ulong)args.layer * args.width;
    float dot = 0;
    if (tid < 256)
        for (uint d = tid; d < args.width; d += 256) dot += (residual[base + d] + block[base + d]) * dir[d];
    const float coefficient = args.mode == 1 ? args.scale : args.scale * hc_steering_sum(dot, scratch, tid, 256);
    if (simd_index == 0) scratch[lane] = 0;
    device float4 *b = (device float4 *)(block + base);
    device const float4 *r = (device const float4 *)(residual + base);
    device const float4 *direction = (device const float4 *)dir;
    device float4 *sum = (device float4 *)(sum_out + base);
    float square_sum = 0;
    for (uint d = tid; d < args.width / 4; d += threads) {
        const float4 previous = r[d];
        float4 value = previous + b[d];
        if (args.scale != 0) value += (args.mode == 1 ? coefficient : -coefficient) * direction[d];
        sum[d] = value;
        square_sum += metal::dot(value, value);
    }
    square_sum = simd_sum(square_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) scratch[simd_index] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    square_sum = simd_sum(scratch[lane]);
    const float norm_scale = 1.0f / sqrt(square_sum / args.width + args.norm_eps);
    device float4 *norm = norm_out + (ulong)row * (args.width / 4);
    for (uint d = tid; d < args.width / 4; d += threads)
        norm[d] = (sum[d] * norm_scale) * weight[d];
}

// A complete published residual row is the projection unit. HC callers pass
// rows*HC; each group mutates its own stream, never another group's inputs.
kernel void kernel_residual_transform_steering(
        constant ds4_metal_hc_steering_args & args,
        device float *values, device const float *directions,
        threadgroup float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint threads [[threads_per_threadgroup]]) {
    if (row >= args.rows) return;
    device float *x = values + (ulong)row * args.width;
    device const float *dir = directions + (ulong)args.layer * args.width;
    float dot = 0;
    for (uint d = tid; d < args.width; d += threads) {
        const float published = args.bf16 ? dsv41_bf16(x[d]) : x[d];
        dot += published * dir[d];
    }
    const float coefficient = args.mode == 1 ? args.scale : args.scale * hc_steering_sum(dot, scratch, tid, threads);
    for (uint d = tid; d < args.width; d += threads) {
        float value = args.bf16 ? dsv41_bf16(x[d]) : x[d];
        if (args.scale != 0) value += (args.mode == 1 ? coefficient : -coefficient) * dir[d];
        x[d] = args.bf16 ? dsv41_bf16(value) : value;
    }
}
