// GPU half of Clef's schema head. The small query/option operations remain on
// CPU. Adapted from Cleffa 24fc91e (MIT): FP32 activations, unchanged BF16 weights.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct clef_head_gemm_args {
    int rows, columns, inner;
};
kernel void clef_head_gemm(constant clef_head_gemm_args &a [[buffer(0)]],
                           device float *x [[buffer(1)]], device bfloat *w [[buffer(2)]],
                           device float *y [[buffer(3)]],
                           uint2 tg [[threadgroup_position_in_grid]]) {
    auto X =
        tensor<device float, dextents<int, 2>, tensor_inline>(x, dextents<int, 2>(a.inner, a.rows));
    auto W = tensor<device bfloat, dextents<int, 2>, tensor_inline>(
        w, dextents<int, 2>(a.inner, a.columns));
    auto Y = tensor<device float, dextents<int, 2>, tensor_inline>(
        y, dextents<int, 2>(a.columns, a.rows));
    matmul2d<matmul2d_descriptor(16, 128, dynamic_length_v<int>, false, true, false),
             execution_simdgroups<4>>
        mm;
    auto input = X.slice(0, int(tg.y) * 16);
    auto weight = W.slice(0, int(tg.x) * 128);
    auto output = Y.slice(int(tg.x) * 128, int(tg.y) * 16);
    mm.run(input, weight, output);
}
struct clef_head_norm_args {
    int width;
    float epsilon;
};
static inline float clef_head_sum(float value, threadgroup float *scratch, uint tid, uint threads) {
    value = simd_sum(value);
    const uint group = tid / 32, lane = tid % 32, count = (threads + 31) / 32;
    if (lane == 0)
        scratch[group] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = lane < count ? scratch[lane] : 0;
    sum = simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sum;
}
kernel void clef_head_norm(constant clef_head_norm_args &a [[buffer(0)]],
                           device const float *x [[buffer(1)]], device const float *w [[buffer(2)]],
                           device const float *bias [[buffer(3)]], device float *out [[buffer(4)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint threads [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    const device float *input = x + (ulong)row * a.width;
    float sum = 0;
    for (int i = tid; i < a.width; i += threads)
        sum += input[i];
    const float mean = clef_head_sum(sum, scratch, tid, threads) / float(a.width);
    float variance = 0;
    for (int i = tid; i < a.width; i += threads) {
        const float d = input[i] - mean;
        variance += d * d;
    }
    const float inv =
        rsqrt(clef_head_sum(variance, scratch, tid, threads) / float(a.width) + a.epsilon);
    for (int i = tid; i < a.width; i += threads)
        out[(ulong)row * a.width + i] = (input[i] - mean) * inv * w[i] + bias[i];
}
