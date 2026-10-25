// Native Omni audio leaves adapted from Cleffa24fc91e2 (MIT).
// License: src/metal/clef/LICENSE.cleffa. Host scheduling/ownership is Zig.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct clef_audio_gemm_args {int rows,columns,inner;};
// The host dispatch divides rows by this same 16-row tile.
kernel void clef_audio_gemm(constant clef_audio_gemm_args &a [[buffer(0)]],
 device float *x [[buffer(1)]],device bfloat *w [[buffer(2)]],device float *y [[buffer(3)]],uint2 tg [[threadgroup_position_in_grid]]) {
 auto X=tensor<device float,dextents<int,2>,tensor_inline>(x,dextents<int,2>(a.inner,a.rows));
 auto W=tensor<device bfloat,dextents<int,2>,tensor_inline>(w,dextents<int,2>(a.inner,a.columns));
 auto Y=tensor<device float,dextents<int,2>,tensor_inline>(y,dextents<int,2>(a.columns,a.rows));
 matmul2d<matmul2d_descriptor(16,128,dynamic_length_v<int>,false,true,false),execution_simdgroups<4>> mm;
 auto input=X.slice(0,int(tg.y)*16);
 auto weight=W.slice(0,int(tg.x)*128);
 auto output=Y.slice(int(tg.x)*128,int(tg.y)*16);
 mm.run(input,weight,output);
}

static inline float erf_f32(float x) {
    const float erx = 8.4506291151e-01f, efx8 = 1.0270333290e+00f;
    const float pp0 = 1.2837916613e-01f, pp1 = -3.2504209876e-01f, pp2 = -2.8481749818e-02f, pp3 = -5.7702702470e-03f, pp4 = -2.3763017452e-05f;
    const float qq1 = 3.9791721106e-01f, qq2 = 6.5022252500e-02f, qq3 = 5.0813062117e-03f, qq4 = 1.3249473704e-04f, qq5 = -3.9602282413e-06f;
    const float pa0 = -2.3621185683e-03f, pa1 = 4.1485610604e-01f, pa2 = -3.7220788002e-01f, pa3 = 3.1834661961e-01f, pa4 = -1.1089469492e-01f, pa5 = 3.5478305072e-02f, pa6 = -2.1663755178e-03f;
    const float qa1 = 1.0642088205e-01f, qa2 = 5.4039794207e-01f, qa3 = 7.1828655899e-02f, qa4 = 1.2617121637e-01f, qa5 = 1.3637083583e-02f, qa6 = 1.1984500103e-02f;
    const float ra0 = -9.8649440333e-03f, ra1 = -6.9385856390e-01f, ra2 = -1.0558626175e+01f, ra3 = -6.2375331879e+01f, ra4 = -1.6239666748e+02f, ra5 = -1.8460508728e+02f, ra6 = -8.1287437439e+01f, ra7 = -9.8143291473e+00f;
    const float sa1 = 1.9651271820e+01f, sa2 = 1.3765776062e+02f, sa3 = 4.3456588745e+02f, sa4 = 6.4538726807e+02f, sa5 = 4.2900814819e+02f, sa6 = 1.0863500214e+02f, sa7 = 6.5702495575e+00f, sa8 = -6.0424413532e-02f;
    const float rb0 = -9.8649431020e-03f, rb1 = -7.9928326607e-01f, rb2 = -1.7757955551e+01f, rb3 = -1.6063638306e+02f, rb4 = -6.3756646729e+02f, rb5 = -1.0250950928e+03f, rb6 = -4.8351919556e+02f;
    const float sb1 = 3.0338060379e+01f, sb2 = 3.2579251099e+02f, sb3 = 1.5367296143e+03f, sb4 = 3.1998581543e+03f, sb5 = 2.5530502930e+03f, sb6 = 4.7452853394e+02f, sb7 = -2.2440952301e+01f;
    uint ix = as_type<uint>(x);
    const bool sign = ix >> 31;
    ix &= 0x7fffffffu;
    if (ix >= 0x7f800000u) return 1.0f - 2.0f * (float)sign + 1.0f / x;   // nan, +-inf
    if (ix < 0x3f580000u) {                                             // |x| < 0.84375
        if (ix < 0x31800000u) return 0.125f * (8.0f * x + efx8 * x);    // |x| < 2^-28
        const float z = x * x;
        const float r = pp0 + z * (pp1 + z * (pp2 + z * (pp3 + z * pp4)));
        const float s = 1.0f + z * (qq1 + z * (qq2 + z * (qq3 + z * (qq4 + z * qq5))));
        return x + x * (r / s);
    }
    float y;
    if (ix < 0x40c00000u) {                                             // |x| < 6
        float c;
        if (ix < 0x3fa00000u) {                                         // |x| < 1.25
            const float s = fabs(x) - 1.0f;
            const float P = pa0 + s * (pa1 + s * (pa2 + s * (pa3 + s * (pa4 + s * (pa5 + s * pa6)))));
            const float Q = 1.0f + s * (qa1 + s * (qa2 + s * (qa3 + s * (qa4 + s * (qa5 + s * qa6)))));
            c = 1.0f - erx - P / Q;
        } else {
            const float ax = fabs(x), s = 1.0f / (ax * ax);
            float R, S;
            if (ix < 0x4036db6du) {                                     // |x| < 1/0.35
                R = ra0 + s * (ra1 + s * (ra2 + s * (ra3 + s * (ra4 + s * (ra5 + s * (ra6 + s * ra7))))));
                S = 1.0f + s * (sa1 + s * (sa2 + s * (sa3 + s * (sa4 + s * (sa5 + s * (sa6 + s * (sa7 + s * sa8)))))));
            } else {
                R = rb0 + s * (rb1 + s * (rb2 + s * (rb3 + s * (rb4 + s * (rb5 + s * rb6)))));
                S = 1.0f + s * (sb1 + s * (sb2 + s * (sb3 + s * (sb4 + s * (sb5 + s * (sb6 + s * sb7))))));
            }
            const float z = as_type<float>(as_type<uint>(ax) & 0xffffe000u);
            c = exp(-z * z - 0.5625f) * exp((z - ax) * (z + ax) + R / S) / ax;
        }
        y = 1.0f - c;
    } else y = 1.0f - 0x1p-120f;
    return sign ? -y : y;
}

static inline float2 omni_twofold_add(float2 a, float2 b) {
    float sum=a.x+b.x;
    float v=sum-a.x;
    float error=(a.x-(sum-v))+(b.x-v)+a.y+b.y;
    float result=sum+error;
    return float2(result,error-(result-sum));
}

struct audio_attention_args { int T,H,heads; float scale; };
kernel void clef_audio_attention(constant audio_attention_args &a [[buffer(0)]],
                            device const float *q [[buffer(1)]], device const float *k [[buffer(2)]],
                            device const float *v [[buffer(3)]], device const int *starts [[buffer(4)]],
                            device const int *ends [[buffer(5)]], device float *out [[buffer(6)]],
                            device atomic_uint *status [[buffer(7)]],
                            uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    int t=group.x,h=group.y,begin=starts[t],end=ends[t];
    size_t row=size_t(t)*a.H+h*64;
    if(begin<0 || begin>t || end<=t || end>a.T || end-begin>104) {
        atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
        out[row+lane]=out[row+lane+32]=NAN;return;
    }
    threadgroup float scores[104];
    float q0=q[row+lane],q1=q[row+lane+32],maximum=-INFINITY;
    bool valid=isfinite(q0) && isfinite(q1);
    for(int s=begin;s<end;s++) {
        size_t index=size_t(s)*a.H+h*64+lane;
        float k0=k[index],k1=k[index+32];
        valid=valid && isfinite(k0) && isfinite(k1);
        float score=simd_sum(q0*k0+q1*k1)*a.scale;
        if(lane==0)scores[s-begin]=score;
        maximum=max(maximum,score);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 denominator=0,acc0=0,acc1=0;
    for(int s=begin;s<end;s++) {
        float probability=exp(scores[s-begin]-maximum);
        size_t index=size_t(s)*a.H+h*64+lane;
        float v0=v[index],v1=v[index+32];
        valid=valid && isfinite(v0) && isfinite(v1);
        denominator=omni_twofold_add(denominator,float2(probability,0));
        float p0=probability*v0,p1=probability*v1;
        acc0=omni_twofold_add(acc0,float2(p0,fma(probability,v0,-p0)));
        acc1=omni_twofold_add(acc1,float2(p1,fma(probability,v1,-p1)));
    }
    float d=denominator.x+denominator.y;
    float y0=(acc0.x+acc0.y)/d,y1=(acc1.x+acc1.y)/d;
    if(!valid || !isfinite(y0) || !isfinite(y1))atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
    out[row+lane]=y0;out[row+lane+32]=y1;
}

struct audio_norm_args { int H; float eps; };
kernel void clef_audio_norm(constant audio_norm_args &a [[buffer(0)]], device const float *x [[buffer(1)]],
                       device const float *weight [[buffer(2)]], device const float *bias [[buffer(3)]],
                       device float *out [[buffer(4)]], device atomic_uint *status [[buffer(5)]],
                       uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    size_t base=size_t(row)*a.H;
    float sum=0;
    for(int d=lane;d<a.H;d+=32)sum+=x[base+d];
    float mean=simd_sum(sum)/float(a.H),variance=0;
    for(int d=lane;d<a.H;d+=32){float delta=x[base+d]-mean;variance+=delta*delta;}
    float inv=rsqrt(simd_sum(variance)/float(a.H)+a.eps);
    for(int d=lane;d<a.H;d+=32) {
        float value=(x[base+d]-mean)*inv*weight[d]+bias[d];
        if(!isfinite(value))atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
        out[base+d]=value;
    }
}

// mode 0: bias; 1: bias+exact GELU; 2: bias then residual. Separate the
// residual addition from the matrix product to match the pinned audio layer.
struct audio_bias_args { int H,rows,mode; };
kernel void clef_audio_bias(constant audio_bias_args &a [[buffer(0)]], device float *x [[buffer(1)]],
                       device const float *bias [[buffer(2)]], device float *residual [[buffer(3)]],
                       device atomic_uint *status [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    if(index>=uint(a.H)*uint(a.rows))return;
    float value=x[index]+bias[index%a.H];
    if(a.mode==1)value=0.5f*value*(1.0f+erf_f32(value*0.7071067811865476f));
    if(a.mode==2)value+=residual[index];
    if(!isfinite(value))atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
    if(a.mode==2)residual[index]=value;else x[index]=value;
}

// NHWC input -> rows [chunk, output frequency, output time], K=[channel,ky,kx].
// Every element, including edge padding, is written; poison cannot leak into GEMM.
struct audio_conv_args { int chunks,height,width,channels; };
kernel void clef_audio_im2col(constant audio_conv_args &a [[buffer(0)]], device const float *x [[buffer(1)]],
                         device float *out [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    int oh=(a.height+1)/2,ow=(a.width+1)/2,K=a.channels*9;
    if(index>=uint(a.chunks)*uint(oh*ow*K))return;
    int k=index%K,row=index/K,channel=k/9,ky=k%9/3,kx=k%3;
    int chunk=row/(oh*ow),y=row/ow%oh*2-1+ky,t=row%ow*2-1+kx;
    out[index]=(y<0 || y>=a.height || t<0 || t>=a.width)?0:
        x[((size_t(chunk)*a.height+y)*a.width+t)*a.channels+channel];
}

// Conv3 NHWC -> [chunk,time,channel,frequency], the conv_out projection order.
kernel void clef_audio_conv_pack(constant audio_conv_args &a [[buffer(0)]], device const float *x [[buffer(1)]],
                            device float *out [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if(index>=uint(a.chunks)*uint(a.height*a.width*a.channels))return;
    int frequency=index%a.height,channel=index/a.height%a.channels;
    int t=index/(a.height*a.channels)%a.width,chunk=index/(a.height*a.channels*a.width);
    out[index]=x[((size_t(chunk)*a.height+frequency)*a.width+t)*a.channels+channel];
}

struct audio_gather_args { int rows,H,padded_rows; };
kernel void clef_audio_gather(constant audio_gather_args &a [[buffer(0)]], device const float *x [[buffer(1)]],
                         device const int *indices [[buffer(2)]], device const float *positions [[buffer(3)]],
                         device float *out [[buffer(4)]], device atomic_uint *status [[buffer(5)]],
                         uint index [[thread_position_in_grid]]) {
    if(index>=uint(a.rows)*uint(a.H))return;
    int src=indices[index/a.H],d=index%a.H;
    float value=x[size_t(src)*a.H+d]+positions[(src%a.padded_rows)*a.H+d];
    if(!isfinite(value))atomic_fetch_or_explicit(status,1u,memory_order_relaxed);
    out[index]=value;
}
