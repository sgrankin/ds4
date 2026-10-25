// Clef-only affine INT4 tile candidates, using the existing MLX NAX leaves.
// Derived from MLX v0.32.0 quantized_nax.h (Apple Inc., MIT); see the vendored
// notice in src/metal/flint/prefill/vendor/LICENSE. Compile after Flint's
// assembled quantized source. The ordinary Flint entry points are unchanged.
namespace clef_dense {
using namespace flint_qmm;

// Keep BK=64: affine groups are 64 values, and the existing quantized loader
// requires a K tile no wider than a quantization group. Larger M/N tiles reuse
// inputs or dequantized weights more, at the cost of more register/shared state.
template <int BM, int BN>
inline void dense(constant FlintMatmul &p, device const uint *w,
                  device const bfloat16_t *scales, device const bfloat16_t *biases,
                  device const bfloat16_t *x, device bfloat16_t *y,
                  uint3 group, uint thread_index, uint simd_group, uint lane,
                  threadgroup bfloat16_t *weights) {
    // Stored matrices are padded to 64 columns, not necessarily 128. The wider
    // N candidate must retain the loader/store checks on its last output tile.
    constexpr bool aligned_n = BN == 64;
    qmm_t_nax_tgp_impl<bfloat16_t, 64, 4, aligned_n, BM, 64, BN, 2, 2>(
        w, scales, biases, x, y, weights, p.k, p.n, p.m,
        group, thread_index, simd_group, lane);
}

#define CLEF_DENSE_TILE(NAME, BM, BN, SWAP)                                             \
    [[host_name(#NAME)]] kernel void NAME(                                              \
        constant FlintMatmul &p [[buffer(0)]], device const uint *w [[buffer(1)]],       \
        device const bfloat16_t *scales [[buffer(2)]],                                  \
        device const bfloat16_t *biases [[buffer(3)]],                                  \
        device const bfloat16_t *x [[buffer(4)]], device bfloat16_t *y [[buffer(5)]],    \
        uint3 group [[threadgroup_position_in_grid]],                                  \
        uint thread_index [[thread_index_in_threadgroup]],                            \
        uint simd_group [[simdgroup_index_in_threadgroup]],                            \
        uint lane [[thread_index_in_simdgroup]]) {                                    \
        threadgroup bfloat16_t weights[BN * 72];                                      \
        dense<BM, BN>(p, w, scales, biases, x, y,                                     \
                      SWAP ? uint3(group.y, group.x, group.z) : group,                \
                      thread_index, simd_group, lane,                                \
                      weights);                                                      \
    }

CLEF_DENSE_TILE(clef_qmm4_64x128, 64, 128, false)
CLEF_DENSE_TILE(clef_qmm4_128x64, 128, 64, false)
CLEF_DENSE_TILE(clef_qmm4_rows_first, 64, 64, true)
#undef CLEF_DENSE_TILE

// Group eight token tiles at a time: reuse each weight tile across those rows
// while keeping a bounded (~512-row) activation working set. Arithmetic and
// the physical dispatch grid remain the ordinary 64x64 kernel's.
[[host_name("clef_qmm4_grouped_rows")]]
kernel void grouped_rows(constant FlintMatmul &p [[buffer(0)]],
                         device const uint *w [[buffer(1)]],
                         device const bfloat16_t *scales [[buffer(2)]],
                         device const bfloat16_t *biases [[buffer(3)]],
                         device const bfloat16_t *x [[buffer(4)]],
                         device bfloat16_t *y [[buffer(5)]],
                         uint3 physical [[threadgroup_position_in_grid]],
                         uint thread_index [[thread_index_in_threadgroup]],
                         uint simd_group [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const uint columns = (uint(p.n) + 63) / 64;
    const uint rows = (uint(p.m) + 63) / 64;
    const uint linear = physical.y * columns + physical.x;
    const uint group = linear / (8 * columns);
    const uint first_row = group * 8;
    const uint group_rows = min(8u, rows - first_row);
    const uint within = linear % (8 * columns);
    const uint3 logical(within / group_rows,
                        first_row + within % group_rows, physical.z);
    threadgroup bfloat16_t weights[64 * 72];
    dense<64, 64>(p, w, scales, biases, x, y, logical,
                  thread_index, simd_group, lane, weights);
}
} // namespace clef_dense
