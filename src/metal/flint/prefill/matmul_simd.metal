// Portable SIMD-group matrix path for M4, using MLX 0.32.0's MIT-licensed
// affine loaders and 8x8 matrix fragments. No M5 tensor/NAX operation is used.
namespace flint_qmm_simd {
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"
using namespace mlx::steel;
struct FlintMatmul { int m, n, k, pad; };

// A 32x32 output tile has four SIMD groups and four 8x8 fragments per group.
// Expert batches store only the selected contiguous run within that tile.
template<typename Matrix>
inline void flint_store_run(thread Matrix& matrix, device bfloat16_t* y,
    int stride, short lo, short hi) {
    #pragma unroll
    for(short i=0;i<2;++i) {
        const short row=matrix.sm+i*16;
        #pragma unroll
        for(short j=0;j<2;++j) {
            const short col=matrix.sn+j*16;
            thread auto& tile_values=matrix.Ctile.frag_at(i,j);
            if(row>=lo && row<hi) {
                y[row*stride+col]=bfloat16_t(tile_values[0]);
                y[row*stride+col+1]=bfloat16_t(tile_values[1]);
            }
        }
    }
}

template<int Bits,bool Gather>
inline void flint_simd_quantized(constant FlintMatmul& p, device const uint* packed,
    device const bfloat16_t* scales, device const bfloat16_t* biases,
    device const bfloat16_t* x, device bfloat16_t* y, device const uint* indices,
    uint3 group, uint sg, uint lane, threadgroup bfloat16_t* Xs,
    threadgroup bfloat16_t* Ws) {
    constexpr short BM=32,BN=32,BK=32,LD=40;
    constexpr int pack=get_pack_factor<Bits,8>(),bytes=get_bytes_per_pack<Bits>();
    using Matrix=BlockMMA<bfloat16_t,bfloat16_t,BM,BN,BK,2,2,false,true,LD,LD>;
    using InputLoader=BlockLoader<bfloat16_t,BM,BK,LD,1,128>;
    using WeightLoader=QuantizedBlockLoader<bfloat16_t,BN,BK,LD,1,128,64,Bits>;
    const uint first=group.y*BM,column=group.x*BN;
    const short rows=short(min(uint(BM),uint(p.m)-first));
    const ulong weight_stride=ulong(p.n)*p.k*bytes/pack;
    const ulong scale_stride=ulong(p.n)*(p.k/64);
    device const uchar* w=(device const uchar*)packed;
    short lo=0;
    while(lo<rows) {
        const uint expert=Gather?indices[first+lo]:0;
        short hi=rows;
        if(Gather) {
            hi=lo+1;
            while(hi<rows && indices[first+hi]==expert) ++hi;
        }
        Matrix matrix(sg,lane);
        matrix.Ctile.clear();
        InputLoader input(x+ulong(first)*p.k,p.k,Xs,sg,lane);
        WeightLoader weight(w+expert*weight_stride+ulong(column)*p.k*bytes/pack,
            scales+expert*scale_stride+ulong(column)*(p.k/64),
            biases+expert*scale_stride+ulong(column)*(p.k/64),p.k,Ws,sg,lane);
        for(int k=0;k<p.k;k+=BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(rows==BM) input.load_unsafe(); else input.load_safe(short2(BK,rows));
            weight.load_unsafe();
            threadgroup_barrier(mem_flags::mem_threadgroup);
            matrix.mma(Xs,Ws);
            input.next(); weight.next();
        }
        flint_store_run(matrix,y+ulong(first)*p.n+column,p.n,lo,hi);
        lo=hi;
    }
}

#define FLINT_SIMD_DENSE(NAME,BITS) \
[[host_name(#NAME)]] kernel void NAME(constant FlintMatmul& p [[buffer(0)]], \
    device const uint* w [[buffer(1)]], device const bfloat16_t* s [[buffer(2)]], \
    device const bfloat16_t* b [[buffer(3)]], device const bfloat16_t* x [[buffer(4)]], \
    device bfloat16_t* y [[buffer(5)]], uint3 group [[threadgroup_position_in_grid]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup bfloat16_t Xs[32*40],Ws[32*40]; \
    flint_simd_quantized<BITS,false>(p,w,s,b,x,y,nullptr,group,sg,lane,Xs,Ws); \
}
FLINT_SIMD_DENSE(flint_prefill_qmm4_simd,4)
FLINT_SIMD_DENSE(flint_prefill_qmm8_simd,8)
#undef FLINT_SIMD_DENSE

#define FLINT_SIMD_GATHER(NAME,BITS) \
[[host_name(#NAME)]] kernel void NAME(constant FlintMatmul& p [[buffer(0)]], \
    device const bfloat16_t* x [[buffer(1)]], device const uint* w [[buffer(2)]], \
    device const bfloat16_t* s [[buffer(3)]], device const bfloat16_t* b [[buffer(4)]], \
    device const uint* ids [[buffer(5)]], device bfloat16_t* y [[buffer(6)]], \
    uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup bfloat16_t Xs[32*40],Ws[32*40]; \
    flint_simd_quantized<BITS,true>(p,w,s,b,x,y,ids,group,sg,lane,Xs,Ws); \
}
FLINT_SIMD_GATHER(flint_prefill_gather_qmm4_simd,4)
FLINT_SIMD_GATHER(flint_prefill_gather_qmm3_simd,3)
#undef FLINT_SIMD_GATHER

// Same packed-affine vector arithmetic as the selected M5 small-batch path.
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

[[host_name("flint_prefill_qmv4_packed_simd")]]
kernel void flint_prefill_qmv4_packed_simd(constant FlintMatmul& p [[buffer(0)]],
    device const uint* w [[buffer(1)]], device const bfloat16_t* s [[buffer(2)]],
    device const bfloat16_t* b [[buffer(3)]], device const bfloat16_t* x [[buffer(4)]],
    device bfloat16_t* y [[buffer(5)]], uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    flint_qmv4_packed<false>(p,w,s,b,x,y,nullptr,group,sg,lane);
}

[[host_name("flint_prefill_gather_qmv4_packed_simd")]]
kernel void flint_prefill_gather_qmv4_packed_simd(constant FlintMatmul& p [[buffer(0)]],
    device const bfloat16_t* x [[buffer(1)]], device const uint* w [[buffer(2)]],
    device const bfloat16_t* s [[buffer(3)]], device const bfloat16_t* b [[buffer(4)]],
    device const uint* indices [[buffer(5)]], device bfloat16_t* y [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    flint_qmv4_packed<true>(p,w,s,b,x,y,indices,group,sg,lane);
}

[[host_name("flint_prefill_router_mm_simd")]]
kernel void flint_prefill_router_mm_simd(constant FlintMatmul& p [[buffer(0)]],
    device const bfloat16_t* x [[buffer(1)]], device const bfloat16_t* w [[buffer(2)]],
    device bfloat16_t* y [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup bfloat16_t Xs[32*40],Ws[32*40];
    using Matrix=BlockMMA<bfloat16_t,bfloat16_t,32,32,32,2,2,false,true,40,40>;
    using Loader=BlockLoader<bfloat16_t,32,32,40,1,128>;
    const uint first=group.y*32,column=group.x*32;
    const short rows=short(min(32u,uint(p.m)-first));
    Matrix matrix(sg,lane); matrix.Ctile.clear();
    Loader input(x+ulong(first)*p.k,p.k,Xs,sg,lane);
    Loader weight(w+ulong(column)*p.k,p.k,Ws,sg,lane);
    for(int k=0;k<p.k;k+=32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(rows==32) input.load_unsafe(); else input.load_safe(short2(32,rows));
        weight.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);
        matrix.mma(Xs,Ws);
        input.next(); weight.next();
    }
    flint_store_run(matrix,y+ulong(first)*p.n+column,p.n,0,rows);
}
} // namespace flint_qmm_simd
