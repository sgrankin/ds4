// Adapted from zknpr/cleffa 24fc91e28d (MIT). See LICENSE.cleffa.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct act_args { int f16; float lim; };
static inline ushort act16(float v, constant act_args &ac, device atomic_int *ovf) { (void)ac;(void)ovf;return as_type<ushort>((bfloat)v); }
// Sum over a threadgroup of up to 1024 threads (32 simdgroups).
static inline float tg_sum(float v, threadgroup float *scratch, uint tid, uint nthreads) {
    v = simd_sum(v);
    const uint sg = tid / 32, lane = tid % 32, nsg = (nthreads + 31) / 32;
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = lane < nsg ? scratch[lane] : 0.0f;
    r = simd_sum(r);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

static inline float silu(float x) { return x / (1.0f + exp(-x)); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
// torch.nn.functional.softplus(x), beta=1, threshold=20
// log1p is not in the Metal stdlib; Goldberg's formulation keeps full precision for tiny y.
static inline float log1p_acc(float y) { const float u = 1.0f + y; return u == 1.0f ? y : log(u) * (y / (u - 1.0f)); }
static inline float softplus(float x) { return x > 20.0f ? x : log1p_acc(exp(x)); }

// ---------------------------------------------------------------- RMSNorm
// HF Qwen3_5RMSNorm: f32 x * rsqrt(mean(x^2) + eps) * (1 + w); (1 + w) is folded in w.
struct norm_args { int H; float eps; };
kernel void rmsnorm_act(constant norm_args &a [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device const float *w [[buffer(2)]],
                         device ushort *y [[buffer(3)]],
                         constant act_args &ac [[buffer(4)]],
                         device const int *seq_start [[buffer(5)]],
                         device atomic_int *ovf [[buffer(6)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint nt [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    device const float *xr = x + (long)row * a.H;
    float ss = 0.0f;
    for (int i = tid; i < a.H; i += nt) ss += xr[i] * xr[i];
    ss = tg_sum(ss, scratch, tid, nt);
    const float inv = rsqrt(ss / (float)a.H + a.eps);
    device atomic_int *of = ovf + seq_start[row];
    for (int i = tid; i < a.H; i += nt) y[(long)row * a.H + i] = act16(xr[i] * inv * w[i], ac, of);
}

// Same, f32 output (final norm). The head LayerNorm takes it unrounded; HF's BF16 path hands
// the head BF16 here, which CLEF_HEAD_BF16=1 reproduces (layernorm round_input).
kernel void rmsnorm_f32(constant norm_args &a [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device const float *w [[buffer(2)]],
                        device float *y [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint nt [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    device const float *xr = x + (long)row * a.H;
    float ss = 0.0f;
    for (int i = tid; i < a.H; i += nt) ss += xr[i] * xr[i];
    ss = tg_sum(ss, scratch, tid, nt);
    const float inv = rsqrt(ss / (float)a.H + a.eps);
    for (int i = tid; i < a.H; i += nt) y[(long)row * a.H + i] = xr[i] * inv * w[i];
}

// ---------------------------------------------------------------- full attention
// QKV row layout (the fused projection): [nh x (query hd | gate hd)] [nkv x hd] [nkv x hd].
// One simdgroup per (token, head); lane l owns dims l, l+32, ..., l+224 (hd = 256), so
// each RoPE pair (i, i+32), i < 32, lives in one lane.
struct attn_prep_args { int nh, nkv, hd, n_rot, row; float eps; };
kernel void attn_prep(constant attn_prep_args &a [[buffer(0)]],
                      device const float *qkv [[buffer(1)]],
                      device const float *qnorm [[buffer(2)]],
                      device const float *knorm [[buffer(3)]],
                      device const int *pos [[buffer(4)]],
                      device const float *inv_freq [[buffer(5)]],
                      device float *Q [[buffer(6)]],    // [nh][T][hd]
                      device float *K [[buffer(7)]],    // [nkv][T][hd]
                      device float *V [[buffer(8)]],    // [nkv][T][hd]
                      device float *G [[buffer(9)]],    // [T][nh*hd]
                      constant int &T [[buffer(10)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const int t = tg.x, head = tg.y;        // head in [0, nh + nkv)
    const int hd = a.hd, per = hd / 32;
    device const float *row = qkv + (long)t * a.row;
    const bool is_q = head < a.nh;
    const int kh = head - a.nh;
    device const float *src = is_q ? row + (long)head * 2 * hd : row + (long)a.nh * 2 * hd + (long)kh * hd;
    float v[8];
    float ss = 0.0f;
    for (int j = 0; j < per; j++) { v[j] = src[lane + 32 * j]; ss += v[j] * v[j]; }
    ss = simd_sum(ss);
    const float inv = rsqrt(ss / (float)hd + a.eps);
    device const float *nw = is_q ? qnorm : knorm;
    for (int j = 0; j < per; j++) v[j] = v[j] * inv * nw[lane + 32 * j];
    // NeoX partial RoPE on the first n_rot dims: rotate_half pairs (i, i + n_rot/2). Interleaved
    // M-RoPE (apply_interleaved_mrope, mrope_section [11, 11, 10]): pair i turns by the token's
    // temporal, row or column position for i mod 3 = 0, 1, 2; pos is [T][3]. A text token has the
    // same value on all three, so this is the plain RoPE it always was.
    const int rot_half = a.n_rot / 2;   // 32 for Clef: dims lane and lane+32 are slots 0 and 1
    if (lane < (uint)rot_half) {
        const float ang = (float)pos[3 * t + (int)(lane % 3)] * inv_freq[lane];
        const float c = precise::cos(ang), s = precise::sin(ang);
        const float x0 = v[0], x1 = v[1];
        v[0] = x0 * c - x1 * s;
        v[1] = x1 * c + x0 * s;
    }
    if (is_q) {
        for (int j = 0; j < per; j++) {
            Q[((long)head * T + t) * hd + lane + 32 * j] = v[j];
            G[(long)t * a.nh * hd + (long)head * hd + lane + 32 * j] = src[hd + lane + 32 * j];
        }
    } else {
        device const float *vsrc = row + (long)a.nh * 2 * hd + (long)a.nkv * hd + (long)kh * hd;
        for (int j = 0; j < per; j++) {
            K[((long)kh * T + t) * hd + lane + 32 * j] = v[j];
            V[((long)kh * T + t) * hd + lane + 32 * j] = vsrc[lane + 32 * j];
        }
    }
}

// Keep-warm (clef_gpu_keepalive): one thread that reads one element of a buffer, so an idle
// server keeps the weights and activation buffers resident for the next request.
kernel void keepalive(device const ushort *w [[buffer(0)]],
                      device uint *sink [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid == 0) sink[0] = w[0];
}

kernel void fill_zero(device float *p [[buffer(0)]],
                      constant long &n [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    if ((long)gid < n) p[gid] = 0.0f;
}

// Causal GQA attention, online softmax in f32. One simdgroup per (query, head).
// Output is multiplied by sigmoid(gate) and written as the 16-bit o_proj operand (act16).
struct attn_args { int nh, nkv, hd, T; float scale; };
kernel void attention(constant attn_args &a [[buffer(0)]],
                      device const float *Q [[buffer(1)]],
                      device const float *K [[buffer(2)]],
                      device const float *V [[buffer(3)]],
                      device const float *G [[buffer(4)]],
                      device const int *seq_start [[buffer(5)]],
                      device ushort *O [[buffer(6)]],   // [T][nh*hd]
                      constant act_args &ac [[buffer(7)]],
                      device atomic_int *ovf [[buffer(8)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const int t = tg.x, h = tg.y, hd = a.hd, per = hd / 32;
    const int kh = h / (a.nh / a.nkv);
    float q[8], acc[8];
    for (int j = 0; j < per; j++) { q[j] = Q[((long)h * a.T + t) * hd + lane + 32 * j] * a.scale; acc[j] = 0.0f; }
    float m = -INFINITY, l = 0.0f;
    device const float *Kh = K + (long)kh * a.T * hd;
    device const float *Vh = V + (long)kh * a.T * hd;
    for (int s = seq_start[t]; s <= t; s++) {
        float d = 0.0f;
        for (int j = 0; j < per; j++) d += q[j] * Kh[(long)s * hd + lane + 32 * j];
        d = simd_sum(d);
        const float mn = max(m, d);
        const float corr = exp(m - mn), p = exp(d - mn);
        l = l * corr + p;
        for (int j = 0; j < per; j++) acc[j] = acc[j] * corr + p * Vh[(long)s * hd + lane + 32 * j];
        m = mn;
    }
    const float invl = 1.0f / l;
    for (int j = 0; j < per; j++) {
        const long o = (long)t * a.nh * hd + (long)h * hd + lane + 32 * j;
        O[o] = act16(acc[j] * invl * sigmoid(G[o]), ac, ovf + seq_start[t]);
    }
}

// Tiled causal GQA attention (FlashAttention-2 style) on f32 simdgroup matrices.
// Threadgroup = one block of 8 queries x one KV head; simdgroup g handles q-head
// kh*G + g (G = nh/nkv), so the G simdgroups share every K/V tile through the cache.
// Keys are processed 32 at a time: S = Q K^T (8x32), online softmax in threadgroup
// scratch, O = diag(alpha) O + P V. Row scaling uses an MMA with a diagonal matrix, which
// does not depend on how simdgroup_matrix elements map to lanes.
// Tile loads may run up to 31 K/V rows (7 Q rows) past T. For every head but the last those
// rows belong to the next head (finite, masked); the last head's tail rows are zeroed on the
// host after attn_prep on every forward, because masked probabilities are exact zeros and
// 0 * NaN would still poison P.V (review #2 M5; tests/test_poison.sh).
constant constexpr int FA_BQ = 8, FA_BK = 32, FA_HD = 256, FA_DT = FA_HD / 8;
kernel void attention_fa(constant attn_args &a [[buffer(0)]],
                         device const float *Q [[buffer(1)]],
                         device const float *K [[buffer(2)]],
                         device const float *V [[buffer(3)]],
                         device const float *G [[buffer(4)]],
                         device const int *seq_start [[buffer(5)]],
                         device ushort *O [[buffer(6)]],
                         constant act_args &ac [[buffer(7)]],
                         device atomic_int *ovf [[buffer(8)]],

                         uint2 tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float smem[8 * (8 * 32 + 64)];
    const int i0 = tg.x * FA_BQ, kh = tg.y, grp = a.nh / a.nkv, h = kh * grp + sg;
    const int T = a.T;
    threadgroup float *S = smem + sg * (FA_BQ * FA_BK + 64);   // 8x32 scores/probs
    threadgroup float *D = S + FA_BQ * FA_BK;                   // 8x8 diagonal
    device const float *Qh = Q + ((long)h * T + i0) * FA_HD;
    device const float *Kh = K + (long)kh * T * FA_HD;
    device const float *Vh = V + (long)kh * T * FA_HD;

    // Unroll all three loops that index acc (init, update, epilogue) together: no
    // dynamically indexed fragment access should remain. Unrolling only the update
    // did not help in bench/attention_bench.m. Keep the Q.K loop compact; fully
    // unrolling that independent loop was slower. The arithmetic order is unchanged.
    simdgroup_float8x8 acc[FA_DT];
    #pragma clang loop unroll(full)
    for (int d = 0; d < FA_DT; d++) acc[d] = simdgroup_float8x8(0.0f);
    // softmax state for row r = lane / 4 (replicated over the 4 lanes of the row)
    const int r = lane / 4, c0 = (lane % 4) * 8;
    const int q = i0 + r;
    const bool qvalid = q < T;
    const int qs = qvalid ? seq_start[q] : 0;
    float m = -INFINITY, l = 0.0f;

    // A query block may straddle records. Keys are processed once per record present in
    // the block, in 32-key blocks aligned to that record's start, so every query sees the
    // same blocking (and float summation order) as when its record runs alone: results do
    // not depend on batch composition. Rows of other records are masked in a pass, which
    // is an exact no-op for them (max unchanged -> alpha = 1, probabilities = 0).
    const int k_end = min(i0 + FA_BQ, T);                       // exclusive
    for (int seg = seq_start[i0]; seg >= 0;) {
    for (int kb = seg; kb < k_end; kb += FA_BK) {
        simdgroup_float8x8 s[4];
        for (int j = 0; j < 4; j++) s[j] = simdgroup_float8x8(0.0f);
        for (int d = 0; d < FA_DT; d++) {
            simdgroup_float8x8 qt, kt;
            simdgroup_load(qt, Qh + d * 8, FA_HD);
            for (int j = 0; j < 4; j++) {
                simdgroup_load(kt, Kh + (long)(kb + 8 * j) * FA_HD + d * 8, FA_HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s[j], qt, kt, s[j]);
            }
        }
        for (int j = 0; j < 4; j++) simdgroup_store(s[j], S + 8 * j, FA_BK);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float v[8], mx = -INFINITY;
        for (int c = 0; c < 8; c++) {
            const int key = kb + c0 + c;
            const bool ok = qvalid && qs == seg && key >= qs && key <= q;
            v[c] = ok ? S[r * FA_BK + c0 + c] * a.scale : -INFINITY;
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
        for (int c = 0; c < 8; c++) S[r * FA_BK + c0 + c] = v[c];
        D[lane] = 0.0f;
        D[lane + 32] = 0.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane % 4 == 0) D[r * 8 + r] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 dm, p[4];
        simdgroup_load(dm, D, 8);
        for (int j = 0; j < 4; j++) simdgroup_load(p[j], S + 8 * j, FA_BK);
        #pragma clang loop unroll(full)
        for (int d = 0; d < FA_DT; d++) {
            simdgroup_float8x8 t;
            simdgroup_multiply(t, dm, acc[d]);
            for (int j = 0; j < 4; j++) {
                simdgroup_float8x8 vt;
                simdgroup_load(vt, Vh + (long)(kb + 8 * j) * FA_HD + d * 8, FA_HD);
                simdgroup_multiply_accumulate(t, p[j], vt, t);
            }
            acc[d] = t;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
        int next = -1;   // start of the next record in this query block, if any
        for (int rr = 0; rr < FA_BQ && i0 + rr < T; rr++) {
            if (seq_start[i0 + rr] > seg) { next = seq_start[i0 + rr]; break; }
        }
        seg = next;
    }

    // epilogue: O = diag(1/l) acc, times sigmoid(gate), as the 16-bit o_proj operand
    D[lane] = 0.0f;
    D[lane + 32] = 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane % 4 == 0) D[r * 8 + r] = l > 0.0f ? 1.0f / l : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dm;
    simdgroup_load(dm, D, 8);
    const int er = lane / 4, ec = (lane % 4) * 2;   // 2 elements per lane in the epilogue
    #pragma clang loop unroll(full)
    for (int d = 0; d < FA_DT; d++) {
        simdgroup_float8x8 t;
        simdgroup_multiply(t, dm, acc[d]);
        simdgroup_store(t, S, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const int qq = i0 + er;
        if (qq < T) {
            for (int c = 0; c < 2; c++) {
                const long o = (long)qq * a.nh * FA_HD + (long)h * FA_HD + d * 8 + ec + c;
                O[o] = act16(S[er * 8 + ec + c] * sigmoid(G[o]), ac, ovf + seq_start[qq]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// An 8x8 fragment has two local elements on the required 32-lane SIMDgroup.
// Load row scales as another fragment so the lane-to-row layout stays opaque.
static inline simdgroup_float8x8 rescale_fragment(simdgroup_float8x8 value,
                                                simdgroup_float8x8 scale) {
    thread float2 &v = reinterpret_cast<thread float2 &>(value.thread_elements());
    thread float2 &s = reinterpret_cast<thread float2 &>(scale.thread_elements());
    v = fma(v, s, float2(0.0f));
    return value;
}

// Four SIMDgroups share 32 query rows. Each computes scores/softmax for eight
// rows, then owns 64 output columns for all 32 rows, reusing each V fragment.
// BK=64 computes two score blocks per Q load, but applies the original 32-key
// softmax and four P.V updates separately and in order. Merging those updates
// changes rounding and worsens full-model error against FP32.
// Key blocks align to each record's start, preserving batch invariance. Every
// group must reach every barrier; fully padded query groups load zero Q. The
// host zeroes 8 Q and BK K/V tail rows before dispatch to keep masked P.V finite.
template <int QB, int BK>
kernel void attention_reuse(constant attn_args &a [[buffer(0)]],
                         device const float *Q [[buffer(1)]],
                         device const float *K [[buffer(2)]],
                         device const float *V [[buffer(3)]],
                         device const float *G [[buffer(4)]],
                         device const int *seq_start [[buffer(5)]],
                         device ushort *O [[buffer(6)]],
                         constant act_args &ac [[buffer(7)]],
                         device atomic_int *ovf [[buffer(8)]],

                         uint2 tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float smem[4 * (8 * 64 + 64)];
    const int i0 = tg.x * (8 * QB), h = tg.y, kh = h / (a.nh / a.nkv);
    const int T = a.T;
    threadgroup float *base = smem;
    device const float *Qh = Q + ((long)h * T + i0) * FA_HD;
    device const float *Kh = K + (long)kh * T * FA_HD;
    device const float *Vh = V + (long)kh * T * FA_HD;
    simdgroup_float8x8 acc[QB][FA_DT / QB];
    float m = -INFINITY, l = 0.0f;
    #pragma clang loop unroll(full)
    for (int b = 0; b < QB; b++) {
        #pragma clang loop unroll(full)
        for (int d = 0; d < FA_DT / QB; d++) acc[b][d] = simdgroup_float8x8(0.0f);
    }
    const int r = lane / 4, c0 = (lane % 4) * 8;
    const int k_end = min(i0 + FA_BQ * QB, T);
    for (int seg = seq_start[i0]; seg >= 0;) {
        for (int kb = seg; kb < k_end; kb += BK) {
            simdgroup_float8x8 s[BK / 8];
            for (int j = 0; j < BK / 8; j++) s[j] = simdgroup_float8x8(0.0f);
            for (int d = 0; d < FA_DT; d++) {
                simdgroup_float8x8 qt = simdgroup_float8x8(0.0f);
                if (i0 + sg * 8 < T) simdgroup_load(qt, Qh + sg * 8 * FA_HD + d * 8, FA_HD);
                for (int j = 0; j < BK / 8; j++) {
                    simdgroup_float8x8 kt;
                    simdgroup_load(kt, Kh + (long)(kb + 8 * j) * FA_HD + d * 8, FA_HD, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(s[j], qt, kt, s[j]);
                }
            }
            threadgroup float *scores = base + sg * (FA_BQ * BK + 64);
            for (int j = 0; j < BK / 8; j++) simdgroup_store(s[j], scores + 8 * j, BK);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int sub = 0; sub < BK / 32 && kb + sub * 32 < k_end; sub++) {
                {
                    const int b = sg;
                    threadgroup float *S = base + b * (FA_BQ * BK + 64);
                    threadgroup float *D = S + FA_BQ * BK;
                    const int q = i0 + b * 8 + r;
                    const bool qvalid = q < T;
                    const int qs = qvalid ? seq_start[q] : 0;
                    float v[8], mx = -INFINITY;
                    for (int c = 0; c < 8; c++) {
                        const int key = kb + sub * 32 + c0 + c;
                        const bool ok = qvalid && qs == seg && key >= qs && key <= q;
                        v[c] = ok ? S[r * BK + sub * 32 + c0 + c] * a.scale : -INFINITY;
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
                    for (int c = 0; c < 8; c++) S[r * BK + sub * 32 + c0 + c] = v[c];
                    D[r * 8 + (lane % 4) * 2] = alpha;
                    D[r * 8 + (lane % 4) * 2 + 1] = alpha;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                #pragma clang loop unroll(full)
                for (int b = 0; b < QB; b++) {
                    simdgroup_float8x8 dm;
                    simdgroup_load(dm, base + b * (FA_BQ * BK + 64) + FA_BQ * BK, 8);
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < FA_DT / QB; d++) {
                        acc[b][d] = rescale_fragment(acc[b][d], dm);
                    }
                }
                // Only one key fragment's probabilities are live at a time. Each output
                // still receives the same four MMA updates, in the original order.
                #pragma clang loop unroll(disable)
                for (int j = 0; j < 4; j++) {
                    simdgroup_float8x8 p[QB];
                    #pragma clang loop unroll(full)
                    for (int b = 0; b < QB; b++) simdgroup_load(p[b], base + b * (FA_BQ * BK + 64) + sub * 32 + 8 * j, BK);
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < FA_DT / QB; d++) {
                        simdgroup_float8x8 vt;
                        simdgroup_load(vt, Vh + (long)(kb + sub * 32 + 8 * j) * FA_HD + sg * (FA_HD / QB) + d * 8, FA_HD);
                        #pragma clang loop unroll(full)
                        for (int b = 0; b < QB; b++) simdgroup_multiply_accumulate(acc[b][d], p[b], vt, acc[b][d]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        int next = -1;
        for (int rr = 0; rr < FA_BQ * QB && i0 + rr < T; rr++) {
            if (seq_start[i0 + rr] > seg) { next = seq_start[i0 + rr]; break; }
        }
        seg = next;
    }
    // Epilogue stores use only the first 64 floats of each group's score scratch.
    // Keep every group's reciprocal diagonal (offset 8 * BK) intact for its peers.
    threadgroup float *own = base + sg * (FA_BQ * BK + 64);
    threadgroup float *D = own + FA_BQ * BK;
    D[lane] = 0; D[lane + 32] = 0;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane % 4 == 0) D[r * 8 + r] = l > 0 ? 1.0f / l : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int er = lane / 4, ec = (lane % 4) * 2;
    #pragma clang loop unroll(full)
    for (int b = 0; b < QB; b++) {
        const float inv_l = base[b * (FA_BQ * BK + 64) + FA_BQ * BK + er * 8 + er];
        #pragma clang loop unroll(full)
        for (int d = 0; d < FA_DT / QB; d++) {
            simdgroup_store(acc[b][d], own, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const int qq = i0 + b * 8 + er;
            if (qq < T) for (int c = 0; c < 2; c++) {
                const long o = (long)qq * a.nh * FA_HD + (long)h * FA_HD + sg * (FA_HD / QB) + d * 8 + ec + c;
                O[o] = act16(fma(own[er * 8 + ec + c], inv_l, 0.0f) * sigmoid(G[o]), ac, ovf + seq_start[qq]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
template [[host_name("attention_reuse_4")]] kernel void attention_reuse<4, 32>(constant attn_args &, device const float *, device const float *, device const float *, device const float *, device const int *, device ushort *, constant act_args &, device atomic_int *, uint2, uint, uint);
template [[host_name("attention_prefetch_64")]] kernel void attention_reuse<4, 64>(constant attn_args &, device const float *, device const float *, device const float *, device const float *, device const int *, device ushort *, constant act_args &, device atomic_int *, uint2, uint, uint);

// ---------------------------------------------------------------- Gated DeltaNet
// Projection row layout (fused): [q (Hk*dk) | k (Hk*dk) | v (Hv*dv) | z (Hv*dv) | b (Hv) | a (Hv)].
// Causal depthwise conv (width 4, no bias) + SiLU over the first C = 2*Hk*dk + Hv*dv
// columns, zero left-padding at each record start (nn.Conv1d padding=K-1, first T outputs).
struct conv_args { int C, row, ksize; };
kernel void ssm_conv(constant conv_args &a [[buffer(0)]],
                     device const float *P [[buffer(1)]],
                     device const float *w [[buffer(2)]],      // [C][kernel]
                     device const int *seq_start [[buffer(3)]],
                     device float *out [[buffer(4)]],          // [T][C]
                     uint2 gid [[thread_position_in_grid]]) {
    const int c = gid.x, t = gid.y;
    if (c >= a.C) return;
    const int s0 = seq_start[t];
    float acc = 0.0f;
    for (int j = 0; j < a.ksize; j++) {
        const int tt = t - (a.ksize - 1) + j;
        if (tt >= s0) acc += w[c * a.ksize + j] * P[(long)tt * a.row + c];
    }
    out[(long)t * a.C + c] = silu(acc);
}

// The same conv for a record whose first tokens come from a prefix-cache entry: `tail` holds the
// projection rows of the three tokens before row 0, [ksize - 1][C], so the first outputs see what
// an uncached pass over the whole record would. One record per pass (row 0 is its first row).
kernel void ssm_conv_tail(constant conv_args &a [[buffer(0)]],
                          device const float *P [[buffer(1)]],
                          device const float *w [[buffer(2)]],
                          device const float *tail [[buffer(3)]],
                          device float *out [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    const int c = gid.x, t = gid.y;
    if (c >= a.C) return;
    float acc = 0.0f;
    for (int j = 0; j < a.ksize; j++) {
        const int tt = t - (a.ksize - 1) + j;
        acc += w[c * a.ksize + j] * (tt >= 0 ? P[(long)tt * a.row + c] : tail[(long)(tt + a.ksize - 1) * a.C + c]);
    }
    out[(long)t * a.C + c] = silu(acc);
}

// Saves the tail for the next pass: the first C columns of the ksize - 1 projection rows from `first`.
struct tail_args { int C, row, first; };
kernel void ssm_tail_save(constant tail_args &a [[buffer(0)]],
                          device const float *P [[buffer(1)]],
                          device float *tail [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if ((int)gid.x < a.C) tail[(long)gid.y * a.C + gid.x] = P[(long)(a.first + gid.y) * a.row + gid.x];
}

// Per (token, head): L2-normalize q and k over dk (eps 1e-6, FLA convention), scale q by
// dk^-1/2; beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias) (ssm_a = -exp(A_log)).
struct gdn_prep_args { int Hk, Hv, dk, dv, row, C; };
kernel void gdn_prep(constant gdn_prep_args &a [[buffer(0)]],
                     device float *X [[buffer(1)]],            // conv output [T][C], q/k normalized in place
                     device const float *P [[buffer(2)]],      // projection rows (for b, a)
                     device const float *ssm_a [[buffer(3)]],
                     device const float *dt_bias [[buffer(4)]],
                     device float *beta [[buffer(5)]],         // [T][Hv]
                     device float *gate [[buffer(6)]],         // [T][Hv] log-decay g
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
    const int t = tg.x, head = tg.y;   // head < 2*Hk: q/k head; else v-head bookkeeping
    if (head < 2 * a.Hk) {
        device float *x = X + (long)t * a.C + (long)head * a.dk;   // q heads then k heads
        float ss = 0.0f;
        for (int i = lane; i < a.dk; i += 32) ss += x[i] * x[i];
        ss = simd_sum(ss);
        float inv = rsqrt(ss + 1e-6f);
        if (head < a.Hk) inv *= rsqrt((float)a.dk);
        for (int i = lane; i < a.dk; i += 32) x[i] *= inv;
    } else {
        const int h = head - 2 * a.Hk;
        if (lane == 0) {
            const long base = (long)t * a.row + a.C + (long)a.Hv * a.dv;  // skip z (Hv*dv) to b, a
            const float bv = P[base + h], av = P[base + a.Hv + h];
            beta[(long)t * a.Hv + h] = sigmoid(bv);
            gate[(long)t * a.Hv + h] = ssm_a[h] * softplus(av + dt_bias[h]);
        }
    }
}

// Sequential gated delta rule (the recurrent form of torch_recurrent_gated_delta_rule):
//   S *= exp(g_t); kv = S^T k_t; delta = (v_t - kv) * beta_t; S += k_t delta^T; o_t = S^T q_t
// One simdgroup per (v-head, 32/LPC value columns, record); LPC lanes share a column, each
// owning dk/LPC rows of its state in registers. The recurrence is latency-bound, so more
// lanes per column (shorter per-token dependency chain, more simdgroups in flight) is
// faster. The k-head of v-head h is h / (Hv / Hk) (repeat_interleave in the reference).
// ST (prefix cache, one record): the state starts from `state` when st.x is set and is written
// to `state_out` when st.y is set, both [Hv][dk][dv]. They are two arguments so that a pass can
// resume from one checkpoint of an entry and store another. Each lane owns its rows of one
// column and reads them before it writes, so one buffer bound to both is race-free too. The
// ST = false kernels never touch these arguments.
struct gdn_args { int Hk, Hv, dk, dv, C; };
template <int LPC, bool ST>
kernel void gdn_scan(constant gdn_args &a [[buffer(0)]],
                     device const float *X [[buffer(1)]],      // [T][C]: q | k | v
                     device const float *beta [[buffer(2)]],
                     device const float *gate [[buffer(3)]],
                     device const int *seq_bounds [[buffer(4)]],  // [n_seq + 1]
                     device float *O [[buffer(5)]],            // [T][Hv*dv]
                     device const float *state [[buffer(6)]],
                     constant int2 &st [[buffer(7)]],
                     device float *state_out [[buffer(8)]],
                     uint3 tg [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
    constexpr int R = 128 / LPC, CPS = 32 / LPC;   // rows per lane, columns per simdgroup
    const int h = tg.x, col = tg.y * CPS + lane / LPC, part = lane % LPC;
    const int kh = h / (a.Hv / a.Hk);
    const int t0 = seq_bounds[tg.z], t1 = seq_bounds[tg.z + 1];
    const int r0 = part * R;
    float S[R];
    for (int i = 0; i < R; i++) S[i] = 0.0f;
    const long soff = ((long)h * a.dk + r0) * a.dv + col;   // state element (h, r0 + i, col) is soff + i * dv
    if (ST && st.x) for (int i = 0; i < R; i++) S[i] = state[soff + (long)i * a.dv];
    const long qoff = (long)kh * a.dk + r0;
    const long koff = (long)a.Hk * a.dk + (long)kh * a.dk + r0;
    const long voff = 2L * a.Hk * a.dk + (long)h * a.dv + col;
    for (int t = t0; t < t1; t++) {
        device const float *x = X + (long)t * a.C;
        const float decay = exp(gate[(long)t * a.Hv + h]);
        device const float4 *k4 = (device const float4 *)(x + koff);
        float4 kvp = 0.0f;
        for (int i = 0; i < R / 4; i++) {
            const float4 k = k4[i];
            S[4 * i + 0] *= decay; S[4 * i + 1] *= decay; S[4 * i + 2] *= decay; S[4 * i + 3] *= decay;
            kvp += float4(S[4 * i + 0], S[4 * i + 1], S[4 * i + 2], S[4 * i + 3]) * k;
        }
        float kv = (kvp.x + kvp.y) + (kvp.z + kvp.w);
        for (int off = 1; off < LPC; off <<= 1) kv += simd_shuffle_xor(kv, off);
        const float delta = (x[voff] - kv) * beta[(long)t * a.Hv + h];
        device const float4 *q4 = (device const float4 *)(x + qoff);
        float4 op = 0.0f;
        for (int i = 0; i < R / 4; i++) {
            const float4 k = k4[i], q = q4[i];
            S[4 * i + 0] += k.x * delta; S[4 * i + 1] += k.y * delta;
            S[4 * i + 2] += k.z * delta; S[4 * i + 3] += k.w * delta;
            op += float4(S[4 * i + 0], S[4 * i + 1], S[4 * i + 2], S[4 * i + 3]) * q;
        }
        float o = (op.x + op.y) + (op.z + op.w);
        for (int off = 1; off < LPC; off <<= 1) o += simd_shuffle_xor(o, off);
        if (part == 0) O[(long)t * a.Hv * a.dv + (long)h * a.dv + col] = o;
    }
    if (ST && st.y) for (int i = 0; i < R; i++) state_out[soff + (long)i * a.dv] = S[i];
}
#define GDN_SCAN(NAME, LPC, ST) template [[host_name(NAME)]] kernel void gdn_scan<LPC, ST>(constant gdn_args &, device const float *, device const float *, device const float *, device const int *, device float *, device const float *, constant int2 &, device float *, uint3, uint);
GDN_SCAN("gdn_scan_2", 2, false)
GDN_SCAN("gdn_scan_4", 4, false)
GDN_SCAN("gdn_scan_8", 8, false)
GDN_SCAN("gdn_scan_st_2", 2, true)
GDN_SCAN("gdn_scan_st_4", 4, true)
GDN_SCAN("gdn_scan_st_8", 8, true)

// Chunked delta rule, entirely FP32. With G_ij = exp(sum(g[j+1:i+1])) and
// L_ij = beta_i * dot(k_i,k_j) for j<i, form R=(I+L)^-1 by forward substitution.
// With E_i=exp(sum(g[0:i+1])), the prepass writes W=(R_ij*beta_j*E_i)K, U=(R_ij*beta_j*G_ij)V,
// KE_i=k_i*G_end,i, A_ij=dot(q_i,k_j)*G_ij for j<=i, and prefix decays E.
// The scan then computes delta=U-W*S, O=E*(Q*S)+A*delta, S=E_end*S+KE^T*delta.
// Buffers are [value head, record-aligned block, row, column]. Record alignment
// and interval summation are part of the numerical contract; see bench/gdn_bench.m.
// Sum only the requested interval: subtracting large prefix sums can erase
// later decays, and subtracting two overflowed negative sums produces NaN.
static inline float gdn_interval_decay(threadgroup const float *g, int start, int end) {
    float sum = 0.0f;
    for (int i = start; i < end; i++) sum += g[i];
    return exp(sum);
}

template<int B>
kernel void gdn_chunk_prep(constant gdn_args &a [[buffer(0)]],
                          device float *X [[buffer(1)]], device const float *beta [[buffer(2)]],
                          device const float *gate [[buffer(3)]], device float *W [[buffer(6)]],
                          device float *U [[buffer(7)]], device float *KE [[buffer(8)]],
                          device float *A [[buffer(9)]], device float *E [[buffer(10)]],
                          constant int &T [[buffer(11)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
    const int chunk = tg.x, h = tg.y, kh = h / (a.Hv / a.Hk), nc = (T + B - 1) / B;
    const int t0 = chunk * B, valid = min(B, T - t0);
    const long block = (long)h * nc + chunk;
    threadgroup float inverse[B * B], gamma[B], betas[B], row[B], log_gate[B], tail[8 * 128];
    for (int i = tid; i < B * B; i += 128) inverse[i] = 0.0f;
    if (tid < B) {
        float sum = 0.0f;
        for (int i = 0; i <= (int)tid && i < valid; i++) sum += gate[(long)(t0 + i) * a.Hv + h];
        gamma[tid] = sum;
        log_gate[tid] = tid < (uint)valid ? gate[(long)(t0 + tid) * a.Hv + h] : 0.0f;
        betas[tid] = tid < (uint)valid ? beta[(long)(t0 + tid) * a.Hv + h] : 0.0f;
        E[block * B + tid] = exp(sum);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // For the sole B=32 instantiation, full blocks never use the ragged operand
    // scratch. The post-Gram barrier publishes this inclusive decay triangle.
    static_assert(B * B <= 8 * 128, "decay triangle must fit the ragged operand scratch");
    if (valid == B) {
        for (int i = tid; i < B * B; i += 128) {
            const int r = i / B, c = i % B;
            if (c <= r) tail[i] = gdn_interval_decay(log_gate, c + 1, r + 1);
        }
    }
    auto xt = tensor<device float, dextents<int, 2>, tensor_inline>(X + (long)t0 * a.C, dextents<int, 2>(a.C, valid));
    auto kv = xt.slice<128, dynamic_extent>(a.Hk * 128 + kh * 128, 0);
    auto qv = xt.slice<128, dynamic_extent>(kh * 128, 0);
    auto vv = xt.slice<128, dynamic_extent>(2 * a.Hk * 128 + h * 128, 0);
    auto iv = tensor<threadgroup float, extents<int, B, B>, tensor_inline>(inverse, extents<int, B, B>());
    matmul2d<matmul2d_descriptor(B, B, 128, false, true, false), execution_simdgroups<4>> gram;
    gram.run(kv, kv, iv);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < B * B; i += 128) {
        int r = i / B, c = i % B;
        inverse[i] = r < valid && c < r ? -betas[r] * inverse[i] : (c == r ? 1.0f : 0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int r = 1; r < B; r++) {
        if (tid < B) row[tid] = inverse[r * B + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < (uint)r) {
            float sum = row[tid];
            for (int k = tid + 1; k < r; k++) sum += row[k] * inverse[k * B + tid];
            inverse[r * B + tid] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Save the inverse in the eventual attention buffer while using TG scratch
    // for the W coefficient. No division by an underflowed decay is needed.
    for (int i = tid; i < B * B; i += 128) {
        int r = i / B, c = i % B;
        A[block * B * B + i] = inverse[i];
        inverse[i] *= betas[c] * exp(gamma[r]);
    }
    for (int i = tid; i < B * 128; i += 128) {
        int r = i / 128, c = i % 128;
        float k = r < valid ? X[(long)(t0 + r) * a.C + a.Hk * 128 + kh * 128 + c] : 0.0f;
        KE[block * B * 128 + i] = k * (valid == B ? tail[(B - 1) * B + r] : gdn_interval_decay(log_gate, r + 1, B));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    auto wv = tensor<device float, extents<int, 128, B>, tensor_inline>(W + block * B * 128, extents<int, 128, B>());
    auto uv = tensor<device float, extents<int, 128, B>, tensor_inline>(U + block * B * 128, extents<int, 128, B>());
    // Both contraction extents must describe the valid keys, including ragged tails.
    auto coefficients = tensor<threadgroup float, dextents<int, 2>, tensor_inline>(inverse, dextents<int, 2>(max(valid, 8), B), array<int, 2>{1, B});
    matmul2d<matmul2d_descriptor(B, 128, dynamic_length_v<int>, false, false, false), execution_simdgroups<4>> product;
    if (valid < 8) {
        for (int i = tid; i < 8 * 128; i += 128) tail[i] = i / 128 < valid ? X[(long)(t0 + i / 128) * a.C + a.Hk * 128 + kh * 128 + i % 128] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto tv = tensor<threadgroup float, extents<int, 128, 8>, tensor_inline>(tail, extents<int, 128, 8>());
        product.run(coefficients, tv, wv);
    } else product.run(coefficients, kv, wv);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < B * B; i += 128) {
        int r = i / B, c = i % B;
        inverse[i] = c <= r ? A[block * B * B + i] * betas[c] * (valid == B ? tail[r * B + c] : gdn_interval_decay(log_gate, c + 1, r + 1)) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (valid < 8) {
        for (int i = tid; i < 8 * 128; i += 128) tail[i] = i / 128 < valid ? X[(long)(t0 + i / 128) * a.C + 2 * a.Hk * 128 + h * 128 + i % 128] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto tv = tensor<threadgroup float, extents<int, 128, 8>, tensor_inline>(tail, extents<int, 128, 8>());
        product.run(coefficients, tv, uv);
    } else product.run(coefficients, vv, uv);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < B * B; i += 128) inverse[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gram.run(qv, kv, iv);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < B * B; i += 128) {
        int r = i / B, c = i % B;
        A[block * B * B + i] = r < valid && c <= r ? inverse[i] * (valid == B ? tail[r * B + c] : gdn_interval_decay(log_gate, c + 1, r + 1)) : 0.0f;
    }
}
#define DELTA_PREP(B) template [[host_name("gdn_chunk_prep_" #B)]] kernel void gdn_chunk_prep<B>(constant gdn_args &, device float *, device const float *, device const float *, device float *, device float *, device float *, device float *, device float *, constant int &, uint2, uint);
DELTA_PREP(32)

// ST (prefix cache): as in gdn_scan, the state starts from `saved` and goes to `saved_out`,
// [Hv][dk][dv] each.
// A pass that resumes at a multiple of B forms the blocks an uncached pass forms.
template<int B, int VC, bool ST>
kernel void gdn_chunk_scan(constant gdn_args &a [[buffer(0)]],
                            device float *X [[buffer(1)]], device float *O [[buffer(5)]],
                            device float *W [[buffer(6)]], device float *U [[buffer(7)]],
                            device float *KE [[buffer(8)]], device float *A [[buffer(9)]],
                            device float *E [[buffer(10)]], constant int &T [[buffer(11)]],
                            device const float *saved [[buffer(12)]], constant int2 &st [[buffer(13)]],
                            device float *saved_out [[buffer(14)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int h = tg.x, c0 = tg.y * VC, kh = h / (a.Hv / a.Hk), nc = (T + (B - 1)) / B;
    threadgroup float state[128 * VC], delta[B * VC], output[B * VC];
    for (int i = tid; i < 128 * VC; i += 128)
        state[i] = ST && st.x ? saved[((long)h * 128 + i / VC) * 128 + c0 + i % VC] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto sv = tensor<threadgroup float, extents<int, VC, 128>, tensor_inline>(state, extents<int, VC, 128>());
    auto dv = tensor<threadgroup float, extents<int, VC, B>, tensor_inline>(delta, extents<int, VC, B>());
    auto ov = tensor<threadgroup float, extents<int, VC, B>, tensor_inline>(output, extents<int, VC, B>());
    matmul2d<matmul2d_descriptor(B, VC, 128, false, false, false), execution_simdgroups<4>> project;
    matmul2d<matmul2d_descriptor(B, VC, B, false, false, false, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> attend;
    matmul2d<matmul2d_descriptor(128, VC, B, true, false, false, matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> update;
    for (int ch = 0; ch < nc; ch++) {
        const long block = (long)h * nc + ch;
        const int t0 = ch * B, valid = min(B, T - t0);
        auto wv = tensor<device float, extents<int, 128, B>, tensor_inline>(W + block * (B * 128), extents<int, 128, B>());
        auto kev = tensor<device float, extents<int, 128, B>, tensor_inline>(KE + block * (B * 128), extents<int, 128, B>());
        auto av = tensor<device float, extents<int, B, B>, tensor_inline>(A + block * (B * B), extents<int, B, B>());
        auto xt = tensor<device float, dextents<int, 2>, tensor_inline>(X + (long)t0 * a.C, dextents<int, 2>(a.C, valid));
        auto qt = xt.slice<128, dynamic_extent>(kh * 128, 0);
        project.run(wv, sv, dv);
        project.run(qt, sv, ov);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < B * VC; i += 128) {
            const int r = i / VC, c = i % VC;
            delta[i] = U[block * (B * 128) + r * 128 + c0 + c] - delta[i];
            output[i] *= E[block * B + r];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        attend.run(av, dv, ov);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < B * VC; i += 128) if (i / VC < valid)
            O[((long)(t0 + i / VC) * a.Hv + h) * 128 + c0 + i % VC] = output[i];
        for (int i = tid; i < 128 * VC; i += 128) state[i] *= E[block * B + (B - 1)];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        update.run(kev, dv, sv);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (ST && st.y) for (int i = tid; i < 128 * VC; i += 128)
        saved_out[((long)h * 128 + i / VC) * 128 + c0 + i % VC] = state[i];
}

#define DELTA_SCAN(NAME, B, VC, ST) template [[host_name(NAME)]] kernel void gdn_chunk_scan<B, VC, ST>(constant gdn_args &, device float *, device float *, device float *, device float *, device float *, device float *, device float *, constant int &, device const float *, constant int2 &, device float *, uint2, uint);
DELTA_SCAN("gdn_chunk_scan_32_16", 32, 16, false)
DELTA_SCAN("gdn_chunk_scan_32_32", 32, 32, false)
DELTA_SCAN("gdn_chunk_scan_st_32_16", 32, 16, true)
DELTA_SCAN("gdn_chunk_scan_st_32_32", 32, 32, true)

// Qwen3_5RMSNormGated: per (token, v-head) RMSNorm over dv, * w, * silu(z) -> 16-bit out_proj operand.
struct gdn_out_args { int Hv, dv, row, zoff; float eps; };
kernel void gdn_out(constant gdn_out_args &a [[buffer(0)]],
                    device const float *O [[buffer(1)]],
                    device const float *P [[buffer(2)]],
                    device const float *w [[buffer(3)]],
                    device ushort *Y [[buffer(4)]],
                    constant act_args &ac [[buffer(5)]],
                    device const int *seq_start [[buffer(6)]],
                    device atomic_int *ovf [[buffer(7)]],
                    uint2 tg [[threadgroup_position_in_grid]],
                    uint lane [[thread_index_in_simdgroup]]) {
    const int t = tg.x, h = tg.y;
    device const float *o = O + (long)t * a.Hv * a.dv + (long)h * a.dv;
    device const float *z = P + (long)t * a.row + a.zoff + (long)h * a.dv;
    float ss = 0.0f;
    for (int i = lane; i < a.dv; i += 32) ss += o[i] * o[i];
    ss = simd_sum(ss);
    const float inv = rsqrt(ss / (float)a.dv + a.eps);
    for (int i = lane; i < a.dv; i += 32) {
        Y[(long)t * a.Hv * a.dv + (long)h * a.dv + i] = act16(o[i] * inv * w[i] * silu(z[i]), ac, ovf + seq_start[t]);
    }
}

// ---------------------------------------------------------------- MLP
// [gate | up] projection row -> silu(gate) * up, the 16-bit down_proj operand.
struct swiglu_args { int I; };
kernel void swiglu(constant swiglu_args &a [[buffer(0)]],
                   device const float *GU [[buffer(1)]],
                   device ushort *Y [[buffer(2)]],
                   constant act_args &ac [[buffer(3)]],
                   device const int *seq_start [[buffer(4)]],
                   device atomic_int *ovf [[buffer(5)]],
                   uint2 gid [[thread_position_in_grid]]) {
    const int i = gid.x, t = gid.y;
    if (i >= a.I) return;
    device const float *r = GU + (long)t * 2 * a.I;
    Y[(long)t * a.I + i] = act16(silu(r[i]) * r[a.I + i], ac, ovf + seq_start[t]);
}
