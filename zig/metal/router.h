#ifndef DS4_METAL_ROUTER_H
#define DS4_METAL_ROUTER_H
#include "bridge_types.h"
#include "router_shared.h"
#include "embedding.h"
#include "compressor.h"
#include "resources.h"
typedef struct {
    uint32_t hash_rows, vocab_size, n_tokens, has_bias, hash_mode;
} ds4_metal_router_visual_args;
enum {
    DS4_ROUTER_SOFTPLUS, DS4_ROUTER_SQRT, DS4_ROUTER_TRANSFORM,
    DS4_ROUTER_FINALIZE, DS4_ROUTER_SIMD, DS4_ROUTER_SIMD_WEIGHTS,
    DS4_ROUTER_TRANSFORM_WEIGHTS, DS4_ROUTER_WEIGHTS_ONE,
    DS4_ROUTER_WEIGHTS_BATCH, DS4_ROUTER_VISUAL_PSO,
    DS4_ROUTER_ADD, DS4_ROUTER_DIVIDE, DS4_ROUTER_SCALE_PSO,
    DS4_ROUTER_CLAMP, DS4_ROUTER_SUM_PSO,
    DS4_ROUTER_GATHER_I32, DS4_ROUTER_GATHER_F32
};
enum { DS4_ROUTER_QUALITY=1, DS4_ROUTER_PRE_M5=2, DS4_ROUTER_M5=4,
    DS4_ROUTER_M3=8, DS4_ROUTER_SSD=16 };
typedef ds4_metal_attention_pipeline ds4_metal_router_pipeline;
enum { DS4_ROUTER_UNARY, DS4_ROUTER_BINARY, DS4_ROUTER_SCALE,
    DS4_ROUTER_GATHER, DS4_ROUTER_SUM, DS4_ROUTER_SELECT_ONE,
    DS4_ROUTER_SELECT_TRANSFORM, DS4_ROUTER_WEIGHT_ONE,
    DS4_ROUTER_WEIGHT_BATCH, DS4_ROUTER_SELECT_VISUAL };
typedef union {
    ds4_metal_attention_unary_args unary;
    ds4_gpu_bin_args binary;
    ds4_metal_embedding_rows_args gather;
    ds4_gpu_kargs_sum_rows sum;
    ds4_metal_router_select_one_args one;
    ds4_metal_router_visual_args visual;
} ds4_metal_router_args;
typedef struct {
    uint32_t kind;
    ds4_metal_router_pipeline pipeline;
    ds4_metal_router_args args;
    ds4_metal_attention_view source, other, target;
    ds4_metal_attention_view probs, bias, hash, tokens, selected, weights, visual;
    uint64_t groups_x, groups_y, threads, shared;
    bool fused_weights, inline_token, direct_threads;
    int32_t token;
    float scale;
} ds4_metal_router_launch;
uint32_t ds4_metal_router_facts(void);
ds4_metal_router_pipeline ds4_metal_router_pipeline_get(uint32_t, bool hot);
int ds4_metal_router_encode(void *, const ds4_metal_router_launch *);
ds4_gpu_tensor *ds4_metal_router_selection_scratch(uint64_t bytes);
ds4_metal_attention_view ds4_metal_router_sum_scratch(uint64_t bytes);
_Static_assert(sizeof(ds4_metal_router_visual_args)==20,"visual router packet");
#endif
