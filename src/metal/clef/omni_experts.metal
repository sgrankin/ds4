// One expert per row tile, adapted from MLX v0.32.3 (Apple Inc., MIT).
// https://github.com/ml-explore/mlx/blob/v0.32.3/mlx/backend/metal/kernels/quantized_nax.h
// The existing MLX/Flint dequantizer and NAX leaves are compiled before this file.
// Unlike the older gather loop, a short expert run never computes every other
// expert's rows in its 64-row neighbourhood.
namespace clef_sparse {
using namespace flint_qmm;

struct TileArgs {
    uint rows, tile_rows, tiles, experts;
};
// Counts and offsets already exist for the sorted router scatter. Publish a
// bounded [first row, valid rows, expert, unused] map for both FFN projections.
[[host_name("clef_omni_expert_tiles")]]
kernel void clef_omni_expert_tiles(constant TileArgs &a [[buffer(0)]],
                                   device const uint *counts [[buffer(1)]],
                                   device const uint *offsets [[buffer(2)]],
                                   device uint4 *tiles [[buffer(3)]],
                                   uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint lengths[128];
    for (uint i = tid; i < a.tiles; i += 128)
        tiles[i] = uint4(0);
    lengths[tid] = (counts[tid] + a.tile_rows - 1) / a.tile_rows;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    uint first = 0;
    for (uint i = 0; i < tid; ++i)
        first += lengths[i];
    for (uint i = 0; i < lengths[tid]; ++i) {
        const uint start = i * a.tile_rows;
        tiles[first + i] =
            uint4(offsets[tid] + start, min(a.tile_rows, counts[tid] - start), tid, 0);
    }
}

template <int BM>
kernel void clef_omni_expert_qmm(
    constant FlintMatmul &p [[buffer(0)]], device const bfloat16_t *x [[buffer(1)]],
    device const uint *w [[buffer(2)]], device const bfloat16_t *scales [[buffer(3)]],
    device const bfloat16_t *biases [[buffer(4)]], device const uint4 *tiles [[buffer(5)]],
    device bfloat16_t *y [[buffer(6)]], uint3 tid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    using T = bfloat16_t;
    constexpr int BN = 64, BK = 64, WM = 2, WN = 2;
    constexpr int SM = BM / WM, SN = BN / WN, SK = 32, BP = BK + 8;
    const uint4 tile = tiles[tid.y];
    if (tile.y == 0)
        return;
    const int K = p.k, N = p.n, M = p.m;
    const int row = tile.x, count = tile.y, expert = tile.z, col = tid.x * BN;
    const int tm = SM * (sg / WN), tn = SN * (sg % WN);
    const short valid = short(clamp(count - tm, 0, SM));
    const bool active = valid != 0, rows_in_bounds = row + tm + SM <= M;
    threadgroup T weights[BN * BP];
    using Loader = QuantizedBlockLoader<T, BN, BK, BP, true, 128, 64, 4>;
    Loader loader((device const uint8_t *)w + ((ulong)expert * N + col) * (K / 2),
                  scales + ((ulong)expert * N + col) * (K / 64),
                  biases + ((ulong)expert * N + col) * (K / 64), K, weights, sg, lane);
    const device T *input = x + (ulong)(row + tm) * K;
    NAXTile<float, SM / 16, SN / 16> result;
    result.clear();
    for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            STEEL_PRAGMA_NO_UNROLL
            for (int kk = 0; kk < BK; kk += SK) {
                NAXTile<T, SM / 16, SK / 16> a;
                NAXTile<T, SN / 16, SK / 16> b;
                volatile int compiler_barrier;
                if (rows_in_bounds)
                    a.load(input + kk, K);
                else
                    a.load_safe(input + kk, K, short2(SK, valid));
                b.template load<T, BP, 1>(weights + tn * BP + kk);
                tile_matmad_nax(result, a, metal::false_type{}, b, metal::true_type{});
                (void)compiler_barrier;
            }
        }
        input += BK;
        loader.next();
    }
    if (active) {
        device T *output = y + (ulong)(row + tm) * N + col + tn;
        if (valid == SM)
            result.store(output, N);
        else
            result.store_slice(output, N, short2(0), short2(SN, valid));
    }
}
#define CLEF_EXPERT_TILE(BM)                                                                       \
    template [[host_name("clef_omni_expert_qmm" #BM)]] kernel void clef_omni_expert_qmm<BM>(       \
        constant FlintMatmul &, device const bfloat16_t *, device const uint *,                    \
        device const bfloat16_t *, device const bfloat16_t *, device const uint4 *,                \
        device bfloat16_t *, uint3, uint, uint);
CLEF_EXPERT_TILE(32)
CLEF_EXPERT_TILE(64)
#undef CLEF_EXPERT_TILE
} // namespace clef_sparse
