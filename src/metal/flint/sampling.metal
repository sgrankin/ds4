// Native GPU sampler for the Flint resident decoder. The chain follows
// mlx-lm sample_utils.py: full-vocabulary top-p, then top-k, then temperature.
// Each packet stays on the GPU; no token or vocabulary readback between steps.
#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

struct FlintSampleConfig {
    uint step, count;
    float temperature, top_p;
    uint top_k, seed_lo, seed_hi, eos_count;
    uint eos[4];
    float min_p;
    uint padding[3];
};
struct FlintCandidate { float value; uint token; };

inline uint flint_random_bits(uint token, uint position, uint lo, uint hi) {
    // A counter rather than mutable state makes packet boundaries irrelevant.
    // This is a native PRNG stream, not MLX's Threefry key sequence.
    uint x=token*0x9e3779b9u+position*0x85ebca6bu+lo;
    x^=hi+0x9e3779b9u+(x<<6)+(x>>2);
    x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;x^=x>>16;
    return x;
}
inline float flint_gumbel(uint token,uint position,constant FlintSampleConfig& p) {
    float u=(float(flint_random_bits(token,position,p.seed_lo,p.seed_hi)>>8)+0.5f)*0x1p-24f;
    u=clamp(u,0x1p-24f,1.0f-0x1p-24f);
    return -metal::precise::log(-metal::precise::log(u));
}

// Five values per thread cover the vocabulary across 128 x 256 threads.
// Keep each thread's short queue sorted; group-wide selection then only needs
// a max reduction and a queue pop for each of the requested k candidates.
template<uint K,typename T>
inline void flint_sample_partial(constant FlintSampleConfig& p,
    const device T* logits,device FlintCandidate* candidates,
    device float* normalizers,const device uint* parameters,
    uint tid,uint lane,uint simd,uint group,
    threadgroup float* shared_values,threadgroup uint* shared_ids) {
    if(parameters[3]==0u)return;
    float values[5];uint ids[5];
    float maximum=-INFINITY;
    for(uint i=0;i<5u;++i) {
        uint word=(group*256u+tid)+i*32768u;
        float value=word<150144u?float(logits[word]):-INFINITY;
        maximum=max(maximum,value);
        if(p.top_k==0u && word<150144u)
            value=value/p.temperature+flint_gumbel(word,parameters[1]+1u,p);
        values[i]=value;ids[i]=word<150144u?word:0xffffffffu;
    }
    // Full softmax mass is computed before the top-k truncation and before
    // temperature, as in the source harness's apply_top_p().
    float sum=0.0f;
    for(uint i=0;i<5u;++i) {
        uint word=(group*256u+tid)+i*32768u;
        if(word<150144u)sum+=metal::precise::exp(float(logits[word])-maximum);
    }
    float simd_maximum=simd_max(maximum);
    float simd_total=simd_sum(sum*metal::precise::exp(maximum-simd_maximum));
    if(lane==0u){shared_values[simd]=simd_maximum;shared_values[8u+simd]=simd_total;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(simd==0u) {
        float m=simd_max(lane<8u?shared_values[lane]:-INFINITY);
        float z=simd_sum(lane<8u?shared_values[8u+lane]*metal::precise::exp(shared_values[lane]-m):0.0f);
        if(lane==0u){normalizers[2u*group]=m;normalizers[2u*group+1u]=z;}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=1u;i<5u;++i) {
        float value=values[i];uint id=ids[i],j=i;
        while(j>0u && (value>values[j-1u] || (value==values[j-1u] && id<ids[j-1u]))) {
            values[j]=values[j-1u];ids[j]=ids[j-1u];--j;
        }
        values[j]=value;ids[j]=id;
    }
    uint front=0u,count=p.top_k?p.top_k:1u;
    for(uint rank=0u;rank<count;++rank) {
        float best=front<5u?values[front]:-INFINITY;
        uint word=front<5u?ids[front]:0xffffffffu;
        float peak=simd_max(best);
        uint choice=simd_min(best==peak?word:0xffffffffu);
        if(lane==0u){shared_values[simd]=peak;shared_ids[simd]=choice;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(simd==0u) {
            float value=lane<8u?shared_values[lane]:-INFINITY;
            uint id=lane<8u?shared_ids[lane]:0xffffffffu;
            float top=simd_max(value);
            uint chosen=simd_min(value==top?id:0xffffffffu);
            if(lane==0u) {
                candidates[group*K+rank]={top,chosen};
                shared_ids[8]=chosen;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(front<5u && ids[front]==shared_ids[8])++front;
    }
}

template<uint K>
inline void flint_sample_finish(constant FlintSampleConfig& p,
    const device FlintCandidate* candidates,const device float* normalizers,
    const device uint* current,device uint* next,device uint* record,
    const device uint* diagnostic,uint tid,uint lane,uint simd,
    threadgroup float* shared_values,threadgroup uint* shared_ids,
    threadgroup float* top_values,threadgroup uint* top_ids) {
    if(tid==0u) {
        uint active=current[3];
        for(uint group=0;group<76u;++group)if(diagnostic[group])active=0u;
        shared_ids[9]=active;
        if(!active){next[0]=current[0];next[1]=current[1];next[2]=current[2];next[3]=0u;}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!shared_ids[9])return;
    constexpr uint LOCAL=K/2u;
    float values[LOCAL];uint ids[LOCAL];
    uint count=p.top_k?p.top_k:1u;
    for(uint i=0;i<LOCAL;++i) {
        uint index=tid+i*256u,group=index/K,rank=index%K;
        FlintCandidate item=rank<count?candidates[index]:FlintCandidate{-INFINITY,0xffffffffu};
        values[i]=item.value;ids[i]=item.token;
    }
    for(uint i=1u;i<LOCAL;++i) {
        float value=values[i];uint id=ids[i],j=i;
        while(j>0u && (value>values[j-1u] || (value==values[j-1u] && id<ids[j-1u]))) {
            values[j]=values[j-1u];ids[j]=ids[j-1u];--j;
        }
        values[j]=value;ids[j]=id;
    }
    uint front=0u;
    for(uint rank=0;rank<count;++rank) {
        float best=front<LOCAL?values[front]:-INFINITY;
        uint word=front<LOCAL?ids[front]:0xffffffffu;
        float peak=simd_max(best);
        uint choice=simd_min(best==peak?word:0xffffffffu);
        if(lane==0u){shared_values[simd]=peak;shared_ids[simd]=choice;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(simd==0u) {
            float value=lane<8u?shared_values[lane]:-INFINITY;
            uint id=lane<8u?shared_ids[lane]:0xffffffffu;
            float top=simd_max(value);
            uint selected=simd_min(value==top?id:0xffffffffu);
            if(lane==0u){top_values[rank]=top;top_ids[rank]=selected;shared_ids[8]=selected;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(front<LOCAL && ids[front]==shared_ids[8])++front;
    }
    // One group merges the 128 stable partial normalizers. Its maximum is the
    // same unscaled top logit, including vocabulary values outside top-k.
    float best=-INFINITY;
    for(uint i=tid;i<128u;i+=256u)best=max(best,normalizers[2u*i]);
    float m=simd_max(best);
    if(lane==0u)shared_values[simd]=m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m=-INFINITY;
    for(uint i=0;i<8u;++i)m=max(m,shared_values[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float z=0.0f;
    for(uint i=tid;i<128u;i+=256u)z+=normalizers[2u*i+1u]*metal::precise::exp(normalizers[2u*i]-m);
    z=simd_sum(z);
    if(lane==0u)shared_values[simd]=z;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0u) {
        float total=0.0f;for(uint i=0;i<8u;++i)total+=shared_values[i];
        float mass=0.0f,score=-INFINITY;uint selected=top_ids[0];
        for(uint rank=0;rank<count;++rank) {
            if(rank && p.top_k && p.top_p>0.0f && p.top_p<1.0f && mass>=p.top_p*total)break;
            float value=top_values[rank];uint word=top_ids[rank];
            if(rank && p.min_p>0.0f && value<top_values[0]+metal::precise::log(p.min_p))break;
            float candidate=p.temperature==0.0f?value:
                p.top_k?value/p.temperature+flint_gumbel(word,current[1]+1u,p):value;
            if(candidate>score || (candidate==score && word<selected)){score=candidate;selected=word;}
            mass+=metal::precise::exp(value-m);
        }
        bool stop=false;
        for(uint i=0;i<p.eos_count;++i)stop=stop||selected==p.eos[i];
        record[2u*p.step]=current[0];record[2u*p.step+1u]=selected;record[16]=p.step+1u;
        next[0]=selected;next[1]=current[1]+1u;next[2]=current[2];next[3]=stop?0u:49u;
    }
}

#define FLINT_PARTIAL(NAME,K,TYPE) \
kernel void NAME(constant FlintSampleConfig& p [[buffer(0)]], \
    const device TYPE* logits [[buffer(1)]],device FlintCandidate* candidates [[buffer(2)]], \
    device float* normalizers [[buffer(3)]],const device uint* parameters [[buffer(4)]], \
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]) { \
    threadgroup float shared_values[16];threadgroup uint shared_ids[10]; \
    flint_sample_partial<K>(p,logits,candidates,normalizers,parameters,tid,lane,simd,group,shared_values,shared_ids); \
}
#define FLINT_FINISH(NAME,K) \
kernel void NAME(constant FlintSampleConfig& p [[buffer(0)]], \
    const device FlintCandidate* candidates [[buffer(1)]],const device float* normalizers [[buffer(2)]], \
    const device uint* current [[buffer(3)]],device uint* next [[buffer(4)]], \
    device uint* record [[buffer(5)]],const device uint* diagnostic [[buffer(6)]], \
    uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]], \
    uint simd [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float shared_values[16],top_values[K];threadgroup uint shared_ids[10],top_ids[K]; \
    flint_sample_finish<K>(p,candidates,normalizers,current,next,record,diagnostic,tid,lane,simd,shared_values,shared_ids,top_values,top_ids); \
}
FLINT_PARTIAL(flint_sample_partial20_bf16,20,bfloat16_t)
FLINT_PARTIAL(flint_sample_partial128_bf16,128,bfloat16_t)
FLINT_PARTIAL(flint_sample_partial20_f32,20,float)
FLINT_PARTIAL(flint_sample_partial128_f32,128,float)
FLINT_FINISH(flint_sample_finish20,20)
FLINT_FINISH(flint_sample_finish128,128)
