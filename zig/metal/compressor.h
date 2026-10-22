#ifndef DS4_METAL_COMPRESSOR_H
#define DS4_METAL_COMPRESSOR_H
#include "attention_output.h"
#include "rope.h"
/* Original shader records are defined once, shared by native and C fallback. */
typedef struct {
    int32_t nk0;
    int32_t ne01;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne11;
    int32_t ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_set_rows_args;
typedef struct {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    int32_t ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne10;
    int32_t ne11;
    int32_t ne12;
    int32_t ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t ne0;
    int32_t ne1;
    int32_t ne2;
    int32_t ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t dim;
} ds4_gpu_concat_args;
typedef struct {
    uint32_t n;
} ds4_gpu_add_flat_args;
typedef struct {
    uint32_t width;
} ds4_gpu_dsv4_ratio4_shift_args;
typedef struct {
    uint32_t head_dim;
    uint32_t n_comp;
    uint32_t replay;
    uint32_t n_threads;
} ds4_gpu_dsv4_compressor_pack_ratio4_args;
typedef struct {
    int64_t n_rows;
    uint32_t head_dim;
    uint32_t n_comp;
    uint32_t replay;
    uint32_t pad;
} ds4_gpu_dsv4_softmax_pool_ratio4_direct_args;
typedef struct {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos;
    uint32_t ape_type;
} ds4_gpu_dsv4_compressor_store_one_args;
typedef struct {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int64_t ne0;
    int64_t ne1;
    uint64_t nb0;
    uint64_t nb1;
} ds4_gpu_dsv4_softmax_pool_args;
typedef struct {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos0;
    uint32_t n_tokens;
} ds4_gpu_dsv4_compressor_score_ape_args;
typedef struct {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t ne0;
    int64_t ne1;
    int64_t ne2;
    int64_t ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ds4_gpu_kargs_sum_rows;
typedef struct {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t ne11;
    int32_t ne12;
    int32_t ne13;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float scale;
    float max_bias;
    float m0;
    float m1;
    int32_t n_head_log2;
} ds4_gpu_softmax_args;
typedef struct {
    uint32_t n_total;
    uint32_t width;
    uint32_t ratio;
    uint32_t pos0;
} ds4_gpu_cpy_tile_rows_args;
typedef struct {
    ds4_metal_attention_rope_args rope;
    float rms_eps;
    uint32_t pad0;
} ds4_gpu_comp_finalize_args;
_Static_assert(sizeof(ds4_gpu_comp_finalize_args) == 72, "Metal compressor finalize packet");
/* Scratch remains one actual Metal resource owner. Returned views are retained
 * through every encoded use, including unretained command-buffer mode. */
enum {
    DS4_COMP_KV,
    DS4_COMP_SCORE,
    DS4_COMP_SCORE_CONT,
    DS4_COMP_SCRATCH_SOFTMAX,
    DS4_COMP_PRODUCT,
    DS4_COMP_APE,
    DS4_COMP_ADDED_SCORE
};
ds4_metal_attention_view ds4_metal_compressor_scratch(uint32_t kind, uint64_t bytes);
enum {
    DS4_COMP_COPY,
    DS4_COMP_ADD,
    DS4_COMP_FILL,
    DS4_COMP_SET_ROWS,
    DS4_COMP_SUM,
    DS4_COMP_MUL,
    DS4_COMP_CONCAT,
    DS4_COMP_SOFTMAX,
    DS4_COMP_TILE_APE,
    DS4_COMP_SCORE_APE,
    DS4_COMP_STORE,
    DS4_COMP_POOL,
    DS4_COMP_DIRECT_POOL,
    DS4_COMP_PACK,
    DS4_COMP_DECODE_PACK,
    DS4_COMP_EXACT_REDUCE,
    DS4_COMP_EXACT_POOL,
    DS4_COMP_SHIFT
};
ds4_metal_attention_pipeline ds4_metal_compressor_pipeline(uint32_t kind, bool half);
/* One fixed kernel dispatch: native supplies the real packet, absolute regions,
 * row indices and complete geometry. There is no policy in the encoder. */
typedef union {
    ds4_metal_copy_args copy;
    ds4_gpu_add_flat_args add;
    ds4_metal_attention_unary_args fill;
    ds4_gpu_set_rows_args set_rows;
    ds4_gpu_kargs_sum_rows sum;
    ds4_gpu_bin_args multiply;
    ds4_gpu_concat_args concat;
    ds4_gpu_softmax_args softmax;
    ds4_gpu_cpy_tile_rows_args tile;
    ds4_gpu_dsv4_compressor_score_ape_args score_ape;
    ds4_gpu_dsv4_compressor_store_one_args store;
    ds4_gpu_dsv4_softmax_pool_args pool;
    ds4_gpu_dsv4_softmax_pool_ratio4_direct_args direct_pool;
    ds4_gpu_dsv4_compressor_pack_ratio4_args pack;
    ds4_gpu_dsv4_ratio4_shift_args shift;
} ds4_metal_compressor_args;
typedef struct {
    uint32_t kind;
    ds4_metal_attention_pipeline pipeline;
    ds4_metal_compressor_args args;
    ds4_metal_attention_view kv, score, state_kv, state_score, out;
    ds4_metal_attention_view product, score_cont, softmax, ape, indices;
    const int32_t *rows;
    uint32_t row_count;
    uint64_t gx, gy, gz, tx, ty, shared;
} ds4_metal_compressor_launch;
int ds4_metal_compressor_encode(void *, const ds4_metal_compressor_launch *);
uint32_t ds4_metal_compressor_cached_pipelines(void);
bool ds4_metal_compressor_pair_nr4(void);
/* Fixed fused-projection binding sets, not a clone of any runtime owner. */
typedef struct {
    ds4_metal_attention_view weight_kv, weight_score, ape;
    ds4_metal_attention_view out_kv, out_score, state_kv, state_score;
} ds4_metal_compressor_pair_views;
typedef struct {
    uint32_t kind;
    const ds4_metal_hc_pipeline *pipeline;
    ds4_metal_mv_args mv[3];
    ds4_gpu_dsv4_compressor_store_one_args store[2];
    ds4_gpu_comp_finalize_args finalize;
    ds4_metal_compressor_pair_views pairs[2];
    ds4_metal_attention_view input, q_weights, kv_weights, qr, raw;
    ds4_metal_attention_view attn_stage, attn_norm, attn_cache, index_cache, index_norm;
    uint32_t pair_groups;
    uint64_t groups, nsg, shared;
} ds4_metal_compressor_fused;
enum { DS4_COMP_FUSED_PAIR, DS4_COMP_FUSED_QUAD, DS4_COMP_FUSED_QKV, DS4_COMP_FUSED_FINALIZE };
int ds4_metal_compressor_fused_encode(void *, const ds4_metal_compressor_fused *);

#endif
