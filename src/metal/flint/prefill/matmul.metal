// Derived from MLX 0.32.0 quantized_nax.h (Apple Inc., MIT).
// Changes: fixed Flint shapes, consolidated argument block, explicit entry names.
// The NAX tile/quantized loader implementation in vendor is unmodified.
namespace flint_qmm {
#include "mlx/backend/metal/kernels/quantized_nax.h"
struct FlintMatmul { int m, n, k, pad; };

template<int Bits>
inline void flint_dense(constant FlintMatmul& p, device const uint* w,
    device const bfloat16_t* s, device const bfloat16_t* b,
    device const bfloat16_t* x, device bfloat16_t* y,
    uint3 tid, uint lid, uint sg, uint lane, threadgroup bfloat16_t* scratch) {
    qmm_t_nax_tgp_impl<bfloat16_t,64,Bits,true,64,64,64,2,2>(
        w,s,b,x,y,scratch,p.k,p.n,p.m,tid,lid,sg,lane);
}
#define FLINT_DENSE(NAME,BITS) \
[[host_name(#NAME)]] kernel void NAME(constant FlintMatmul& p [[buffer(0)]], \
    device const uint* w [[buffer(1)]], device const bfloat16_t* s [[buffer(2)]], \
    device const bfloat16_t* b [[buffer(3)]], device const bfloat16_t* x [[buffer(4)]], \
    device bfloat16_t* y [[buffer(5)]], uint3 tid [[threadgroup_position_in_grid]], \
    uint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup bfloat16_t scratch[64*72]; \
    flint_dense<BITS>(p,w,s,b,x,y,tid,lid,sg,lane,scratch); \
}
FLINT_DENSE(flint_prefill_qmm4,4)
FLINT_DENSE(flint_prefill_qmm8,8)
#undef FLINT_DENSE

template<int Bits>
inline void flint_gather_impl(constant FlintMatmul& p, device const bfloat16_t* x,
    device const uint* w, device const bfloat16_t* scales,
    device const bfloat16_t* biases, device const uint* indices, device bfloat16_t* y,
    uint3 tid, uint simd_group_id, uint simd_lane_id, threadgroup bfloat16_t* Ws)
{
  using T=bfloat16_t;
  constexpr int group_size=64, bits=Bits, BM=64, BN=64, BK=64, WM=2, WN=2;
  constexpr bool transpose=true;
  const int M=p.m, N=p.n, K=p.k;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;


  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? N * K_w : K * N_w;
  const size_t stride_s = transpose ? N * K_g : K * N_g;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block
  auto wl = (const device uint8_t*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  scales += transpose ? y_col_long * K_g : y_col / group_size;
  biases += transpose ? y_col_long * K_g : y_col / group_size;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const short sgp_sm =
      align_M ? SM : min(SM, short(max(0, (M - (y_row + tm)))));
  const short sgp_sn =
      align_N ? SN : min(SN, short(max(0, (N - (y_col + tn)))));

  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);
  const bool is_unaligned_bn = align_N ? false : (tgp_bn != BN);

  constexpr short BR = transpose ? TN : TK;
  constexpr short BC = transpose ? TK : TN;

  using AccumType = float;

  // Do as many matmuls as necessary
  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    NAXTile<AccumType, TM, TN> Dtile;
    Dtile.clear();

    const device T* xn = x + tm * K;

    // Prepare threadgroup loading operations
    thread loader_w_t loader_w(
        wl + index * stride_w,
        scales + index * stride_s,
        biases + index * stride_s,
        transpose ? K : N,
        Ws,
        simd_group_id,
        simd_lane_id);

    dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_N || !is_unaligned_bn, [&](auto kAlignedN) {
        for (int k = 0; k < K_it; k++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if constexpr (kAlignedN.value) {
            loader_w.load_unsafe();
          } else {
            loader_w.load_safe(
                transpose ? short2(BK, tgp_bn) : short2(tgp_bn, BK));
          }

          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, BR, BC> Btile;

            volatile int compiler_barrier;

            if constexpr (kAlignedM.value) {
              Atile.load(xn + kk1, K);
            } else {
              Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
            }

            if constexpr (transpose) {
              Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
            } else {
              Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * BN_padded);
            }

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});

            (void)compiler_barrier;
          }

          xn += BK;
          loader_w.next();
        }

        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_w.load_safe(tile_w);
          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, BR, BC> Btile;

            volatile int compiler_barrier;

            const short psk = min(int(SK), max(0, (BK - kk1)));
            Atile.load_safe(xn + kk1, K, short2(psk, sgp_sm));

            if constexpr (transpose) {
              Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
            } else {
              Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * BN_padded);
            }

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});

            (void)compiler_barrier;
          }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const short m_lo_lim = min(int(sgp_sm), max(0, offset - tm));
        const short m_hi_lim = min(int(sgp_sm), max(0, offset_next - tm));

        // Store results to device memory
        if constexpr (kAlignedN.value) {
          if (m_lo_lim == 0 && m_hi_lim == SM) {
            Dtile.store(y + tm * N + tn, N);
          } else {
            Dtile.store_slice(
                y + tm * N + tn, N, short2(0, m_lo_lim), short2(SN, m_hi_lim));
          }
        } else {
          Dtile.store_slice(
              y + tm * N + tn,
              N,
              short2(0, m_lo_lim),
              short2(sgp_sn, m_hi_lim));
        }
      });
    });
  }
}
#define FLINT_GATHER(NAME,BITS) \
[[host_name(#NAME)]] kernel void NAME(constant FlintMatmul& p [[buffer(0)]], \
    device const bfloat16_t* x [[buffer(1)]], device const uint* w [[buffer(2)]], \
    device const bfloat16_t* scales [[buffer(3)]], device const bfloat16_t* biases [[buffer(4)]], \
    device const uint* indices [[buffer(5)]], device bfloat16_t* y [[buffer(6)]], \
    uint3 tid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup bfloat16_t scratch[64*72]; \
    flint_gather_impl<BITS>(p,x,w,scales,biases,indices,y,tid,sg,lane,scratch); }
FLINT_GATHER(flint_prefill_gather_qmm4,4)
FLINT_GATHER(flint_prefill_gather_qmm3,3)
#undef FLINT_GATHER

// MLX's small-vector path keeps affine scale/bias outside the integer dot.
// Four output channels per SIMD reuse each sixteen-value input block.
template<bool Gather>
inline void flint_qmv4_packed(constant FlintMatmul& p, device const uint* packed,
    device const bfloat16_t* scales, device const bfloat16_t* biases,
    device const bfloat16_t* x, device bfloat16_t* y, device const uint* indices,
    uint3 group, uint sg, uint lane) {
    const uint row=group.y,channel=(group.x*4+sg)*4;
    const uint expert=Gather?indices[row]:0;
    device const uchar* w=(device const uchar*)packed+ulong(expert)*p.n*(p.k/2);
    device const bfloat16_t* s=scales+ulong(expert)*p.n*(p.k/64);
    device const bfloat16_t* b=biases+ulong(expert)*p.n*(p.k/64);
    float sums[4]={0,0,0,0};
    for(uint block=0;block<uint(p.k);block+=512) {
        const uint col=block+lane*16;
        float values[16];
        const float sum=load_vector<bfloat16_t,float,16,4>(x+ulong(row)*p.k+col,values);
        #pragma unroll
        for(uint j=0;j<4;++j) {
            const uint out=channel+j;
            if(out>=uint(p.n)) continue;
            const ulong quant_group=ulong(out)*(p.k/64)+col/64;
            sums[j]+=qdot<float,16,4>(w+ulong(out)*(p.k/2)+col/2,values,
                float(s[quant_group]),float(b[quant_group]),sum);
        }
    }
    #pragma unroll
    for(uint j=0;j<4;++j) {
        const float value=simd_sum(sums[j]);
        if(lane==0 && channel+j<uint(p.n)) y[ulong(row)*p.n+channel+j]=bfloat16_t(value);
    }
}

[[host_name("flint_prefill_qmv4_packed")]]
kernel void flint_prefill_qmv4_packed(constant FlintMatmul& p [[buffer(0)]],
    device const uint* w [[buffer(1)]], device const bfloat16_t* s [[buffer(2)]],
    device const bfloat16_t* b [[buffer(3)]], device const bfloat16_t* x [[buffer(4)]],
    device bfloat16_t* y [[buffer(5)]], uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    flint_qmv4_packed<false>(p,w,s,b,x,y,nullptr,group,sg,lane);
}

[[host_name("flint_prefill_gather_qmv4_packed")]]
kernel void flint_prefill_gather_qmv4_packed(constant FlintMatmul& p [[buffer(0)]],
    device const bfloat16_t* x [[buffer(1)]], device const uint* w [[buffer(2)]],
    device const bfloat16_t* s [[buffer(3)]], device const bfloat16_t* b [[buffer(4)]],
    device const uint* indices [[buffer(5)]], device bfloat16_t* y [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    flint_qmv4_packed<true>(p,w,s,b,x,y,indices,group,sg,lane);
}

// The unquantized BF16 router uses the same M5 tile primitive, without staging.
[[host_name("flint_prefill_router_mm")]]
kernel void flint_prefill_router_mm(constant FlintMatmul& p [[buffer(0)]],
    device const bfloat16_t* x [[buffer(1)]], device const bfloat16_t* w [[buffer(2)]],
    device bfloat16_t* y [[buffer(3)]], uint3 tid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr short SM=32,SN=32,SK=32;
    int row=int(tid.y)*64+(sg/2)*SM,col=int(tid.x)*64+(sg%2)*SN;
    short rows=short(max(0,min(int(SM),p.m-row)));
    NAXTile<float,2,2> result; result.clear();
    for(int k=0;k<p.k;k+=SK) {
        NAXTile<bfloat16_t,2,2> a,b;
        a.load_safe(x+row*p.k+k,p.k,short2(SK,rows));
        b.load(w+col*p.k+k,p.k);
        tile_matmad_nax(result,a,metal::false_type{},b,metal::true_type{});
    }
    result.store_safe(y+row*p.n+col,p.n,short2(SN,rows));
}
} // namespace flint_qmm
