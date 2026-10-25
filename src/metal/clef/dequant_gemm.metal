// Experimental Clef dense path: dequantize one matrix once, then use MPP GEMM.
// The affine BF16 dequantizer is MLX's existing implementation (Apple Inc., MIT;
// see ../flint/prefill/vendor/LICENSE). The MPP tensor setup follows Cleffa
// 24fc91e28d (MIT; see LICENSE.cleffa). Compile after Flint's quantized source.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

namespace clef_dequant {
using namespace metal;
using namespace mpp::tensor_ops;
using namespace flint_qmm;

// One 256-thread group handles 2048 weights. Calling the same vendor helper,
// rather than spelling affine arithmetic differently, preserves its BF16
// scale/high-nibble/output rounding. Each thread owns eight scratch values;
// no inter-thread communication or barrier is needed in this kernel.
[[host_name("clef_dequant_q4")]]
kernel void dequantize_matrix(constant FlintMatmul &p [[buffer(0)]],
                              device const uint *packed [[buffer(1)]],
                              device const bfloat16_t *scales [[buffer(2)]],
                              device const bfloat16_t *biases [[buffer(3)]],
                              device bfloat16_t *dense [[buffer(4)]],
                              uint index [[thread_position_in_grid]],
                              uint local_index [[thread_index_in_threadgroup]]) {
    const uint words = uint(p.n) * uint(p.k) / 8;
    if (index >= words)
        return;
    threadgroup bfloat16_t rounded[256 * 8];
    threadgroup bfloat16_t *values = rounded + local_index * 8;
    const uint group = index / 8; // Eight packed uints per 64-weight affine group.
    flint_qmm::dequantize<bfloat16_t, 8, 4>(
        reinterpret_cast<device const uint8_t *>(packed + index),
        scales[group], biases[group], values);
    const ulong first = ulong(index) * 8;
    for (uint i = 0; i < 8; ++i)
        dense[first + i] = values[i];
}

// Y[M,N] = X[M,K] . W[N,K]^T. MPP masks ragged M/N tile edges. BF16 operands
// remain unchanged; explicitly float cooperative accumulators keep this from
// becoming a reduced-precision accumulation path. Storing rounds once to BF16,
// matching the ordinary quantized projection's output boundary. The reduction
// order can differ from NAX, so qualify probabilities as well as elapsed time.
[[host_name("clef_dequant_gemm_64x128")]]
kernel void dense_gemm(constant FlintMatmul &p [[buffer(0)]],
                       device bfloat *input [[buffer(1)]],
                       device bfloat *weight [[buffer(2)]],
                       device bfloat *output [[buffer(3)]],
                       uint2 group [[threadgroup_position_in_grid]]) {
    auto X = tensor<device bfloat, dextents<int, 2>, tensor_inline>(
        input, dextents<int, 2>(p.k, p.m));
    auto W = tensor<device bfloat, dextents<int, 2>, tensor_inline>(
        weight, dextents<int, 2>(p.k, p.n));
    auto x = X.slice(0, int(group.y) * 64);
    auto w = W.slice(0, int(group.x) * 128);
    matmul2d<matmul2d_descriptor(64, 128, dynamic_length_v<int>, false, true, false),
             execution_simdgroups<4>>
        operation;
    auto accumulator = operation.template get_destination_cooperative_tensor<
        decltype(x), decltype(w), float>();
    operation.run(x, w, accumulator);
    // Metal cooperative_tensor::store requires identical element types. Use
    // its supported logical-coordinate API for a single rounded BF16 store,
    // without allocating a full FP32 projection or adding another dispatch.
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < accumulator.get_capacity(); ++i) {
        if (!accumulator.is_valid_element(i))
            continue;
        const auto coordinate = accumulator.get_multidimensional_index(i);
        const int column = int(group.x) * 128 + coordinate[0];
        const int row = int(group.y) * 64 + coordinate[1];
        if (column < p.n && row < p.m)
            output[(ulong)row * p.n + column] = bfloat(accumulator[i]);
    }
}
} // namespace clef_dequant
