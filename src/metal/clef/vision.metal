// Native Clef vision leaves. GEMM and the 72-wide online attention are adapted
// from Cleffa 24fc91e (MIT; see LICENSE.cleffa); tower policy lives in vision.zig.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct clef_vision_gemm_args {int rows,columns,inner;};
kernel void clef_vision_gemm(constant clef_vision_gemm_args &a [[buffer(0)]],
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
struct clef_vision_norm_args {int width;float epsilon;};
static inline float clef_vision_sum(float value,threadgroup float *scratch,uint tid,uint threads) {
 value=simd_sum(value);const uint group=tid/32,lane=tid%32,count=(threads+31)/32;
 if(lane==0)scratch[group]=value;threadgroup_barrier(mem_flags::mem_threadgroup);
 float sum=lane<count?scratch[lane]:0;sum=simd_sum(sum);threadgroup_barrier(mem_flags::mem_threadgroup);return sum;
}
kernel void clef_vision_norm(constant clef_vision_norm_args &a [[buffer(0)]],
 device const float *x [[buffer(1)]],device const float *w [[buffer(2)]],device const float *bias [[buffer(3)]],
 device float *out [[buffer(4)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint threads [[threads_per_threadgroup]]) {
 threadgroup float scratch[32];const device float *input=x+(ulong)row*a.width;
 float sum=0;for(int i=tid;i<a.width;i+=threads)sum+=input[i];
 const float mean=clef_vision_sum(sum,scratch,tid,threads)/float(a.width);
 float variance=0;for(int i=tid;i<a.width;i+=threads){const float d=input[i]-mean;variance+=d*d;}
 const float inv=rsqrt(clef_vision_sum(variance,scratch,tid,threads)/float(a.width)+a.epsilon);
 for(int i=tid;i<a.width;i+=threads)out[(ulong)row*a.width+i]=(input[i]-mean)*inv*w[i]+bias[i];
}

struct clef_vision_args { int P, E, heads, hd, grid_w, merge; float eps, scale; };
struct clef_vision_rows { uint width, rows, mode; };

kernel void clef_vision_embed(constant clef_vision_rows &a [[buffer(0)]],
 device float *x [[buffer(1)]], device const float *bias [[buffer(2)]],
 device const bfloat *table [[buffer(3)]], device const uint *indices [[buffer(4)]],
 device const float *weights [[buffer(5)]], uint i [[thread_position_in_grid]]) {
 if(i>=a.width*a.rows)return;
 const uint p=i/a.width,d=i%a.width;
 float position=0;
 for(uint k=0;k<4;k++)position+=weights[p*4+k]*float(table[ulong(indices[p*4+k])*a.width+d]);
 x[i]=(x[i]+bias[d])+position;
}
kernel void clef_vision_residual(constant clef_vision_rows &a [[buffer(0)]],
 device const float *branch [[buffer(1)]],device const float *bias [[buffer(2)]],
 device float *x [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i<a.width*a.rows)x[i]=x[i]+(branch[i]+bias[i%a.width]);
}
kernel void clef_vision_rope(constant clef_vision_args &a [[buffer(0)]],
                         device const float *qkv [[buffer(1)]],
                         device const float *bias [[buffer(2)]],
                         device const float *inv_freq [[buffer(3)]],
                         device float *Q [[buffer(4)]],
                         device float *K [[buffer(5)]],
                         device float *V [[buffer(6)]],
                         uint3 gid [[thread_position_in_grid]]) {
    const int i = gid.x, h = gid.y, p = gid.z;   // rotary pair, head, patch
    const int hd = a.hd, rot = hd / 2, quarter = hd / 4, E = a.E;   // `half` is a Metal type
    if (i >= rot || p >= a.P) return;
    // These writes start after the final head, never inside another image's data. The
    // host chooses 32 or 128 rows for the attention kernel's largest masked read. Even
    // the smallest four-patch image supplies 64 head/patch rows, enough for its 32-row tail.
    const long tail = ((long)p * a.heads + h) * rot + i;
    if (tail < (long)32 * rot) {
        const long at = (long)a.heads * a.P * hd + 2 * tail;
        Q[at] = Q[at + 1] = K[at] = K[at + 1] = V[at] = V[at + 1] = 0.0f;
    }
    const int window = a.merge * a.merge, blk = p / window, within = p % window, w2 = a.grid_w / a.merge;
    const float hpos = (float)((blk / w2) * a.merge + within / a.merge);
    const float wpos = (float)((blk % w2) * a.merge + within % a.merge);
    const float ang = (i < quarter ? hpos : wpos) * inv_freq[i < quarter ? i : i - quarter];
    const float c = precise::cos(ang), s = precise::sin(ang);
    device const float *row = qkv + (long)p * 3 * E;
    const long o = ((long)h * a.P + p) * hd;
    for (int part = 0; part < 2; part++) {
        const int base = part * E + h * hd;
        const float x0 = row[base + i] + bias[base + i], x1 = row[base + i + rot] + bias[base + i + rot];
        device float *dst = part == 0 ? Q : K;
        dst[o + i] = x0 * c - x1 * s;
        dst[o + i + rot] = x1 * c + x0 * s;
    }
    const int vb = 2 * E + h * hd;
    V[o + i] = row[vb + i] + bias[vb + i];
    V[o + i + rot] = row[vb + i + rot] + bias[vb + i + rot];
}


constant constexpr int VIS_HD=72,VIS_KB=32;
constant constexpr int VIS_MQ = 4, VIS_DT = VIS_HD / 8;
kernel void clef_vision_attention(constant clef_vision_args &a [[buffer(0)]],
                              device const float *Q [[buffer(1)]],
                              device const float *K [[buffer(2)]],
                              device const float *V [[buffer(3)]],
                              device float *out [[buffer(4)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float smem[VIS_MQ * (8 * VIS_KB + 64)];
    threadgroup float *S = smem + sg * (8 * VIS_KB + 64);   // 8x32 scores/probs
    threadgroup float *D = S + 8 * VIS_KB;                   // 8x8 diagonal
    const int P = a.P, h = tg.y, i0 = tg.x * (VIS_MQ * 8) + sg * 8;
    device const float *Qh = Q + ((long)h * P + i0) * VIS_HD;
    device const float *Kh = K + (long)h * P * VIS_HD;
    device const float *Vh = V + (long)h * P * VIS_HD;
    simdgroup_float8x8 acc[VIS_DT];
    #pragma clang loop unroll(full)
    for (int d = 0; d < VIS_DT; d++) acc[d] = simdgroup_float8x8(0.0f);
    const int r = lane / 4, c0 = (lane % 4) * 8;   // softmax: row r, 8 columns from c0
    const bool qvalid = i0 + r < P;
    float m = -INFINITY, l = 0.0f;
    for (int kb = 0; kb < P; kb += VIS_KB) {
        simdgroup_float8x8 s[4];
        for (int j = 0; j < 4; j++) s[j] = simdgroup_float8x8(0.0f);
        for (int d = 0; d < VIS_DT; d++) {
            simdgroup_float8x8 qt, kt;
            simdgroup_load(qt, Qh + d * 8, VIS_HD);
            for (int j = 0; j < 4; j++) {
                simdgroup_load(kt, Kh + (long)(kb + 8 * j) * VIS_HD + d * 8, VIS_HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s[j], qt, kt, s[j]);
            }
        }
        for (int j = 0; j < 4; j++) simdgroup_store(s[j], S + 8 * j, VIS_KB);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float v[8], mx = -INFINITY;
        for (int c = 0; c < 8; c++) {
            const bool ok = qvalid && kb + c0 + c < P;
            v[c] = ok ? S[r * VIS_KB + c0 + c] * a.scale : -INFINITY;
            mx = max(mx, v[c]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        const float mn = max(m, mx);
        float alpha = 1.0f, sum = 0.0f;
        if (mn != -INFINITY) {
            alpha = m == -INFINITY ? 0.0f : exp(m - mn);
            for (int c = 0; c < 8; c++) { v[c] = v[c] == -INFINITY ? 0.0f : exp(v[c] - mn); sum += v[c]; }
        } else {
            for (int c = 0; c < 8; c++) v[c] = 0.0f;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        l = l * alpha + sum;
        m = mn;
        for (int c = 0; c < 8; c++) S[r * VIS_KB + c0 + c] = v[c];
        D[lane] = 0.0f;
        D[lane + 32] = 0.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane % 4 == 0) D[r * 8 + r] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dm, pf[4];
        simdgroup_load(dm, D, 8);
        for (int j = 0; j < 4; j++) simdgroup_load(pf[j], S + 8 * j, VIS_KB);
        #pragma clang loop unroll(full)
        for (int d = 0; d < VIS_DT; d++) {
            simdgroup_float8x8 t;
            simdgroup_multiply(t, dm, acc[d]);
            for (int j = 0; j < 4; j++) {
                simdgroup_float8x8 vt;
                simdgroup_load(vt, Vh + (long)(kb + 8 * j) * VIS_HD + d * 8, VIS_HD);
                simdgroup_multiply_accumulate(t, pf[j], vt, t);
            }
            acc[d] = t;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    // epilogue: O = diag(1/l) acc
    D[lane] = 0.0f;
    D[lane + 32] = 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane % 4 == 0) D[r * 8 + r] = l > 0.0f ? 1.0f / l : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dm;
    simdgroup_load(dm, D, 8);
    const int er = lane / 4, ec = (lane % 4) * 2;   // 2 elements per lane in the epilogue
    #pragma clang loop unroll(full)
    for (int d = 0; d < VIS_DT; d++) {
        simdgroup_float8x8 t;
        simdgroup_multiply(t, dm, acc[d]);
        simdgroup_store(t, S, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const int qq = i0 + er;
        if (qq < P) {
            for (int c = 0; c < 2; c++) {
                const long o = (long)qq * a.E + (long)h * VIS_HD + d * 8 + ec + c;
                out[o] = S[er * 8 + ec + c];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
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


kernel void clef_vision_activate(constant clef_vision_rows &a [[buffer(0)]],
 device float *x [[buffer(1)]],device const float *bias [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 if(i>=a.width*a.rows)return;
 const float v=x[i]+bias[i%a.width];
 if(a.mode==2){x[i]=v;return;}
 if(a.mode==0){const float z=0.7978845608028654f*(v+0.044715f*v*v*v);x[i]=0.5f*v*(1.0f+precise::tanh(z));}
 else x[i]=0.5f*v*(1.0f+erf_f32(v*0.7071067811865476f));
}
