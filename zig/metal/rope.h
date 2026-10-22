#ifndef DS4_METAL_ROPE_H
#define DS4_METAL_ROPE_H
#include "attention_decode.h"
#include <math.h>
#include <pthread.h>
/* Actual packets, shared once with unchanged C fallback and deferred users. */
typedef struct {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t n_dims;
    int32_t mode;
    int32_t n_ctx_orig;
    int32_t inverse;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    bool src2;
} ds4_metal_rope_tail_args;
typedef struct {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t rot_dim;
    uint32_t rot_offset;
    uint32_t pos0;
    uint32_t n_ctx_orig;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
} ds4_metal_rope_glm_args;
typedef struct {
    int32_t n_head;
    int32_t head_dim;
    int32_t head_dim4;
    int32_t n_dims;
    int32_t n_ctx_orig;
    int32_t pos0;
    int32_t inverse;
    float eps;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
} ds4_metal_rope_head_args;
typedef struct {
    uint32_t width, heads, rows, start, inverse, stride;
    float frequencies[32];
} ds4_metal_rope_v41_args;

_Static_assert(sizeof(ds4_metal_rope_tail_args) == 144, "RoPE tail packet");
_Static_assert(sizeof(ds4_metal_rope_head_args) == 56, "Head norm/RoPE packet");
_Static_assert(sizeof(ds4_metal_rope_glm_args) == 52, "GLM RoPE packet");
_Static_assert(sizeof(ds4_metal_rope_v41_args) == 152, "V4.1 RoPE packet");
/* The affine64 packet is already actual ds4_metal_attention_rope_args. */
enum {
    DS4_ROPE_M3 = 1,
    DS4_ROPE_M5 = 2,
    DS4_ROPE_PAIR_AVAILABLE = 4,
    DS4_ROPE_SHARED_AVAILABLE = 8,
    DS4_ROPE_AFFINE_AVAILABLE = 16
};
enum {
    DS4_ROPE_REFERENCE,
    DS4_ROPE_PAIR,
    DS4_ROPE_SHARED,
    DS4_ROPE_AFFINE,
    DS4_ROPE_FUSED_HEAD,
    DS4_ROPE_PLAIN_HEAD,
    DS4_ROPE_GLM,
    DS4_ROPE_V41
};
typedef struct {
    const ds4_metal_hc_pipeline *pipeline;
    uint64_t max_threads;
} ds4_metal_rope_pipeline;
uint32_t ds4_metal_rope_facts(void);
ds4_metal_rope_pipeline ds4_metal_rope_pipeline_get(uint32_t kind);
typedef union {
    ds4_metal_rope_tail_args tail;
    ds4_metal_attention_rope_args affine;
    ds4_metal_rope_head_args head;
    ds4_metal_rms_args norm;
    ds4_metal_rope_glm_args glm;
    ds4_metal_rope_v41_args v41;
} ds4_metal_rope_packet;
typedef struct {
    uint32_t kind;
    const ds4_metal_hc_pipeline *pipeline;
    ds4_metal_rope_packet args;
    ds4_metal_attention_view input;
    const int32_t *positions;
    uint64_t position_bytes;
    bool position_buffer;
    uint64_t groups_x, groups_y, threads, shared_bytes;
} ds4_metal_rope_launch;
int ds4_metal_rope_encode(void *command, const ds4_metal_rope_launch *);
/* Whole native borrowed-command action for remaining C compressor callers. */
int ds4_native_rope_apply(void *command, ds4_metal_attention_view input,
                          const ds4_metal_rope_tail_args *args, uint32_t tokens, uint32_t heads,
                          uint32_t width, uint32_t position, uint32_t step);
/* SDK-owned pthread layout/initializer, not a handwritten Darwin record. */
static const pthread_once_t ds4_metal_rope_once_init __attribute__((unused)) = PTHREAD_ONCE_INIT;
#endif
