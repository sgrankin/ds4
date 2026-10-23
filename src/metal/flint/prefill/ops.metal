// Native Flint prefill elementwise, routing and KV operations.
// Model arithmetic follows recursive-oss/rec e53663b262af (Apache-2.0).
// Matrix products and attention live in the MIT-licensed MLX shader wrappers.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
typedef bfloat flint_bf16;

struct FlintRows { uint rows, width; float epsilon; uint pad; };
struct FlintEmbedding { uint rows, width, vocab, pad; };
struct FlintQk { uint rows, heads, position, rope; float epsilon, theta; };
struct FlintRoute { uint rows, experts, top_k, pad; };
struct FlintScatter { uint rows, width, top_k, pad; };
struct FlintKv { ulong offset; uint capacity, position, rows, global, prior, total, initial_position, pad; };

kernel void flint_prefill_embedding(
    constant FlintEmbedding& p [[buffer(0)]],
    device const uint* tokens [[buffer(1)]], device const uchar* weight [[buffer(2)]],
    device const flint_bf16* scale [[buffer(3)]], device const flint_bf16* bias [[buffer(4)]],
    device flint_bf16* out [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.rows*p.width) return;
    uint row = tokens[i/p.width], col = i%p.width;
    if (row >= p.vocab) return;
    uint group = row*(p.width/64)+col/64;
    out[i] = flint_bf16(float(scale[group])*float(weight[row*p.width+col])+float(bias[group]));
}

// One 256-thread group per token. The reduction is FP32; the result is BF16.
kernel void flint_prefill_rms(
    constant FlintRows& p [[buffer(0)]], device const flint_bf16* x [[buffer(1)]],
    device const flint_bf16* weight [[buffer(2)]], device flint_bf16* out [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    float sum=0;
    for(uint col=tid; col<p.width; col+=256) { float v=float(x[row*p.width+col]); sum+=v*v; }
    sum=simd_sum(sum);
    if(lane==0) partial[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=simd_sum(lane<8?partial[lane]:0.0f);
    float factor=rsqrt(total/float(p.width)+p.epsilon);
    for(uint col=tid; col<p.width; col+=256)
        out[row*p.width+col]=flint_bf16(float(x[row*p.width+col])*factor*float(weight[col]));
}

// Norm is rounded before split-half, nontraditional RoPE, as in the MLX graph.
kernel void flint_prefill_qk_norm_rope(
    constant FlintQk& p [[buffer(0)]], device const flint_bf16* x [[buffer(1)]],
    device const flint_bf16* weight [[buffer(2)]], device flint_bf16* out [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[2];
    uint row=group/p.heads, head=group%p.heads, offset=(row*p.heads+head)*128;
    float a=float(x[offset+tid]), b=float(x[offset+tid+64]);
    float sum=simd_sum(a*a+b*b);
    if(lane==0) partial[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=partial[0]+partial[1];
    float factor=rsqrt(total/128.0f+p.epsilon);
    a=float(flint_bf16(a*factor*float(weight[tid])));
    b=float(flint_bf16(b*factor*float(weight[tid+64])));
    if(p.rope) {
        float angle=float(p.position+row)*exp(-log(p.theta)*float(tid)/64.0f);
        float cs=cos(angle), sn=sin(angle);
        out[offset+tid]=flint_bf16(a*cs-b*sn);
        out[offset+tid+64]=flint_bf16(a*sn+b*cs);
    } else {
        out[offset+tid]=flint_bf16(a);
        out[offset+tid+64]=flint_bf16(b);
    }
}

kernel void flint_prefill_swiglu(
    constant FlintRows& p [[buffer(0)]], device const flint_bf16* gate [[buffer(1)]],
    device const flint_bf16* up [[buffer(2)]], device flint_bf16* out [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=p.rows*p.width) return;
    float g=float(gate[i]);
    float tail=1.0f/(1.0f+exp(abs(g)));
    flint_bf16 sig=flint_bf16(g<0?tail:1.0f-tail);
    out[i]=flint_bf16(flint_bf16(gate[i]*sig)*up[i]);
}

kernel void flint_prefill_add(
    constant FlintRows& p [[buffer(0)]], device const flint_bf16* x [[buffer(1)]],
    device const flint_bf16* y [[buffer(2)]], device flint_bf16* out [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if(i<p.rows*p.width) out[i]=flint_bf16(float(x[i])+float(y[i]));
}

// Sigmoid scores are unbiased mixture weights; the bias changes selection only.
// Rank order is explicit so the six BF16 additions have a stable order.
kernel void flint_prefill_route(
    constant FlintRoute& p [[buffer(0)]], device const flint_bf16* logits [[buffer(1)]],
    device const float* bias [[buffer(2)]], device uint* ids [[buffer(3)]],
    device flint_bf16* weights [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    float scores[8], choices[8], selected_scores[6];
    uint selected_ids[6];
    for(uint j=0;j<8;++j) {
        uint e=lane+32*j;
        float value=1.0f/(1.0f+exp(-float(logits[row*256+e])));
        scores[j]=value; choices[j]=value+bias[e];
    }
    for(uint rank=0;rank<6;++rank) {
        float best=-INFINITY; uint index=0xffffffffu;
        for(uint j=0;j<8;++j) {
            uint e=lane+32*j;
            if(choices[j]>best || (choices[j]==best && e<index)) {best=choices[j];index=e;}
        }
        float peak=simd_max(best);
        uint winner=simd_min(best==peak?index:0xffffffffu);
        selected_ids[rank]=winner;
        selected_scores[rank]=simd_shuffle(scores[winner/32],winner%32);
        for(uint j=0;j<8;++j) if(lane+32*j==winner) choices[j]=-INFINITY;
    }
    if(lane==0) {
        float sum=0;
        for(uint rank=0;rank<6;++rank) sum+=selected_scores[rank];
        for(uint rank=0;rank<6;++rank) {
            ids[row*6+rank]=selected_ids[rank];
            weights[row*6+rank]=flint_bf16(selected_scores[rank]/(sum+1e-20f));
        }
    }
}

kernel void flint_prefill_zero_counts(device atomic_uint* counts [[buffer(1)]],
                                     uint i [[thread_position_in_grid]]) {
    if(i<256) atomic_store_explicit(counts+i,0,memory_order_relaxed);
}

kernel void flint_prefill_histogram(
    constant FlintRoute& p [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device atomic_uint* counts [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i<p.rows*6) atomic_fetch_add_explicit(counts+ids[i],1,memory_order_relaxed);
}

kernel void flint_prefill_offsets(
    device const uint* counts [[buffer(1)]], device uint* offsets [[buffer(2)]],
    device atomic_uint* cursors [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if(i>=256) return;
    uint sum=0; for(uint e=0;e<i;++e) sum+=counts[e];
    offsets[i]=sum; atomic_store_explicit(cursors+i,0,memory_order_relaxed);
}

// Counting-sort the routed rows; matrix kernels reuse each expert's weight tile.
kernel void flint_prefill_scatter(
    constant FlintScatter& p [[buffer(0)]], device const flint_bf16* x [[buffer(1)]],
    device const uint* ids [[buffer(2)]], device const uint* offsets [[buffer(3)]],
    device atomic_uint* cursors [[buffer(4)]], device uint* permutation [[buffer(5)]],
    device uint* sorted_ids [[buffer(6)]], device flint_bf16* sorted_x [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint destination;
    if(tid==0) {
        uint expert=ids[row];
        destination=offsets[expert]+atomic_fetch_add_explicit(cursors+expert,1,memory_order_relaxed);
        permutation[row]=destination; sorted_ids[destination]=expert;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint col=tid;col<p.width;col+=128) sorted_x[destination*p.width+col]=x[(row/6)*p.width+col];
}

kernel void flint_prefill_combine(
    constant FlintRows& p [[buffer(0)]], device const flint_bf16* experts [[buffer(1)]],
    device const uint* permutation [[buffer(2)]], device const flint_bf16* weights [[buffer(3)]],
    device const flint_bf16* shared [[buffer(4)]], device flint_bf16* hidden [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    if(i>=p.rows*p.width) return;
    uint row=i/p.width,col=i%p.width;
    flint_bf16 sum=flint_bf16(experts[permutation[row*6]*p.width+col]*weights[row*6]);
    for(uint rank=1;rank<6;++rank) {
        flint_bf16 product=flint_bf16(experts[permutation[row*6+rank]*p.width+col]*weights[row*6+rank]);
        sum=flint_bf16(float(sum)+float(product));
    }
    flint_bf16 mlp=flint_bf16(float(sum)+float(shared[i]));
    hidden[i]=flint_bf16(float(hidden[i])+float(mlp));
}

// FullModel layout: each layer stores [K/V,4,capacity,128] BF16 elements.
// Prefill leaves local rings chronological; global layers split every 65536.
inline device flint_bf16* flint_kv_bank(uint block, device flint_bf16* b0,
    device flint_bf16* b1, device flint_bf16* b2, device flint_bf16* b3) {
    return block==0?b0:(block==1?b1:(block==2?b2:b3));
}

kernel void flint_prefill_kv_publish(
    constant FlintKv& p [[buffer(0)]], device flint_bf16* b0 [[buffer(1)]],
    device flint_bf16* b1 [[buffer(2)]], device flint_bf16* b2 [[buffer(3)]],
    device flint_bf16* b3 [[buffer(4)]], device const flint_bf16* keys [[buffer(5)]],
    device const flint_bf16* values [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    uint retained=min(p.total,p.capacity);
    if(i>=(p.global?p.rows:retained)*512) return;
    uint row=i/512,head=(i%512)/128,col=i%128;
    if(!p.global) {
        uint source=(head*p.total+p.total-retained+row)*128+col;
        ulong target=p.offset+(ulong(head)*p.capacity+row)*128+col;
        b0[target]=keys[source]; b0[target+ulong(p.capacity)*512]=values[source];
        return;
    }
    uint absolute=p.position+row,block=p.global?absolute/p.capacity:0;
    uint slot=absolute%p.capacity;
    device flint_bf16* bank=flint_kv_bank(block,b0,b1,b2,b3);
    ulong offset=p.offset+(ulong(head)*p.capacity+slot)*128+col;
    bank[offset]=keys[i]; bank[offset+ulong(p.capacity)*512]=values[i];
}

kernel void flint_prefill_kv_assemble(
    constant FlintKv& p [[buffer(0)]], device flint_bf16* b0 [[buffer(1)]],
    device flint_bf16* b1 [[buffer(2)]], device flint_bf16* b2 [[buffer(3)]],
    device flint_bf16* b3 [[buffer(4)]], device const flint_bf16* keys [[buffer(5)]],
    device const flint_bf16* values [[buffer(6)]], device flint_bf16* out_k [[buffer(7)]],
    device flint_bf16* out_v [[buffer(8)]], uint i [[thread_position_in_grid]]) {
    if(i>=p.total*512) return;
    uint head=i/(p.total*128),row=(i/128)%p.total,col=i%128;
    if(!p.global && row>=p.prior) {
        uint source=((row-p.prior)*4+head)*128+col;
        out_k[i]=keys[source]; out_v[i]=values[source];
    } else {
        uint cursor=p.position<p.capacity?0:(p.initial_position>=p.capacity?(p.position-p.initial_position)%p.capacity:p.position%p.capacity);
        uint absolute=p.global?row:cursor+row;
        uint block=p.global?absolute/p.capacity:0,slot=absolute%p.capacity;
        device flint_bf16* bank=flint_kv_bank(block,b0,b1,b2,b3);
        ulong offset=p.offset+(ulong(head)*p.capacity+slot)*128+col;
        out_k[i]=bank[offset]; out_v[i]=bank[offset+ulong(p.capacity)*512];
    }
}

kernel void flint_prefill_logits_f32(
    constant FlintRows& p [[buffer(0)]], device const flint_bf16* x [[buffer(1)]],
    device float* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if(i<p.width) out[i]=float(x[i]);
}
