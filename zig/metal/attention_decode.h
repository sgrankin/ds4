#ifndef DS4_METAL_ATTENTION_DECODE_H
#define DS4_METAL_ATTENTION_DECODE_H
#include "bridge_types.h"
#include <stdlib.h>
#include <stdio.h>
/* Actual shader records move once; unchanged C batch/fallback aliases these
 * same definitions. No host Session or Engine layout enters this boundary. */
typedef struct {
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} ds4_metal_attention_pad_args;

typedef struct {
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2;
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
} ds4_metal_attention_vec_args;

typedef struct {
    int32_t nrows;
} ds4_metal_attention_reduce_args;

typedef struct {
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_raw;
    uint32_t n_comp;
    uint32_t pad_rows;
    uint32_t shared_pad;
} ds4_metal_attention_stage_args;

typedef struct {
    uint64_t row_bytes;
    uint64_t token_bytes;
    int32_t head_dim;
    int32_t n_dims;
    int32_t n_ctx_orig;
    int32_t inverse;
    uint32_t pos0;
    uint32_t pos_step;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
} ds4_metal_attention_rope_args;

typedef struct {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    slope;
    float    scale;
    float    bias;
    float    val;
    float    min;
    float    max;
} ds4_metal_attention_unary_args;
_Static_assert(sizeof(ds4_metal_attention_rope_args) == 64, "Metal affine RoPE packet");
_Static_assert(sizeof(ds4_metal_attention_vec_args) == 192, "Metal vector-attention packet");
_Static_assert(sizeof(ds4_metal_attention_pad_args) == 104, "Metal attention pad packet");
_Static_assert(sizeof(ds4_metal_attention_unary_args) == 120, "Metal unary packet");
_Static_assert(sizeof(ds4_metal_attention_stage_args) == 24, "Metal KV stage packet");
_Static_assert(sizeof(ds4_metal_attention_reduce_args) == 4, "Metal reduction packet");

/* Each view holds one ARC retain on its underlying MTLBuffer. Tensor-relative
 * offsets become absolute exactly once at capture; native owns later slices. */
typedef struct {
    ds4_metal_model_buffer *buffer;
    uint64_t offset, bytes;
} ds4_metal_attention_view;
ds4_metal_attention_view ds4_metal_attention_tensor_view(const ds4_gpu_tensor *);
void *ds4_metal_attention_contents(ds4_metal_attention_view);
enum { DS4_ATTN_TRANSIENT_MASK, DS4_ATTN_MASK, DS4_ATTN_ZERO_MASK,
    DS4_ATTN_KV, DS4_ATTN_PAD, DS4_ATTN_TMP };
ds4_metal_attention_view ds4_metal_attention_scratch(uint32_t kind, uint64_t bytes,
    const char *label);
enum {
    DS4_ATTN_PRE_M5 = 1, DS4_ATTN_M5 = 2, DS4_ATTN_M3 = 4,
    DS4_ATTN_QUALITY = 8, DS4_ATTN_SSD = 16, DS4_ATTN_TP_SINGLE = 32,
    DS4_ATTN_STAGE_AVAILABLE = 64,
};
enum {
    DS4_ATTN_COPY_F32_CONTIG, DS4_ATTN_COPY_F32_GENERIC,
    DS4_ATTN_COPY_F16_CONTIG, DS4_ATTN_COPY_F16_BLIT,
};
typedef struct {
    uint32_t flags;
    uint64_t max_shared_bytes, stage_threads, fill_threads;
    uint64_t copy_threads[3];
} ds4_metal_attention_facts;
ds4_metal_attention_facts ds4_metal_attention_get_facts(void);
enum { DS4_ATTN_PAD_PSO, DS4_ATTN_VEC_PSO, DS4_ATTN_REDUCE_PSO,
    DS4_ATTN_ROPE_PSO, DS4_ATTN_PACKED_PSO };
typedef struct {
    const ds4_metal_hc_pipeline *handle;
    uint64_t threads, simd_width;
} ds4_metal_attention_pipeline;
ds4_metal_attention_pipeline ds4_metal_attention_get_pipeline(uint32_t kind,
    uint32_t width, uint32_t nsg, uint32_t nwg, bool pad, bool shared_pad);

/* Fixed packet/binding operations only. The caller chooses every stage,
 * geometry, buffer region and deferred-task boundary. */
int ds4_metal_attention_break_encoder(void *command);
int ds4_metal_attention_copy(void *command, uint32_t kind,
    const ds4_metal_copy_args *, uint32_t count,
    ds4_metal_attention_view src, ds4_metal_attention_view dst,
    uint64_t groups, uint64_t threads);
int ds4_metal_attention_fill(void *command,
    const ds4_metal_attention_unary_args *, ds4_metal_attention_view,
    uint64_t groups, uint64_t threads);
int ds4_metal_attention_stage(void *command,
    const ds4_metal_attention_stage_args *, ds4_metal_attention_view raw,
    ds4_metal_attention_view comp, ds4_metal_attention_view dst,
    ds4_metal_attention_view mask, ds4_metal_attention_view pad,
    uint64_t groups, uint64_t threads);
int ds4_metal_attention_pad(void *command, ds4_metal_attention_pipeline,
    const ds4_metal_attention_pad_args *, ds4_metal_attention_view kv,
    ds4_metal_attention_view mask, ds4_metal_attention_view pad);
int ds4_metal_attention_vec(void *command, ds4_metal_attention_pipeline,
    const ds4_metal_attention_vec_args *, ds4_metal_attention_view q,
    ds4_metal_attention_view kv, ds4_metal_attention_view mask,
    ds4_metal_attention_view sinks, ds4_metal_attention_view pad,
    ds4_metal_attention_view out, const ds4_metal_attention_rope_args *rope,
    uint64_t groups_x, uint64_t groups_y, uint64_t groups_z,
    uint64_t threads_x, uint64_t threads_y, uint64_t shared_bytes);
int ds4_metal_attention_reduce(void *command, ds4_metal_attention_pipeline,
    const ds4_metal_attention_reduce_args *, ds4_metal_attention_view tmp,
    ds4_metal_attention_view out, const ds4_metal_attention_rope_args *rope,
    uint64_t rows, uint64_t threads);
/* Existing Metal-local public actions, previously declared by ds4.c. Keep
 * actual declarations here for the production-linked deferred-RoPE fixture. */
int ds4_gpu_decode_attn_rope_fuse_available(void);
void ds4_gpu_set_decode_attn_rope_fuse(uint32_t head_dim, uint32_t n_rot,
    uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base,
    float freq_scale, float ext_factor, float attn_factor,
    float beta_fast, float beta_slow);
int ds4_gpu_decode_attn_rope_fuse_used(void);
#endif
