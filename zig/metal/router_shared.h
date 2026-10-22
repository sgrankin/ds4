#ifndef DS4_METAL_ROUTER_SHARED_H
#define DS4_METAL_ROUTER_SHARED_H
#include "attention_decode.h"
#include "dense_matmul.h"
#include "routed_batch.h"
/* Actual router-finalize packet, shared once with ordinary C router selection. */
typedef struct {
    uint32_t has_bias, hash_mode, use_token_buffer, token, hash_rows;
} ds4_metal_router_select_one_args;
enum { DS4_SHARED_VECTOR, DS4_SHARED_EXT, DS4_ROUTER_SHARED, DS4_ROUTER_PROJECT };
typedef struct {
    uint32_t kind;
    const ds4_metal_hc_pipeline *pipeline;
    ds4_metal_mv_args shared, router;
    ds4_metal_mv_ext_args ext;
    ds4_metal_router_select_one_args select;
    ds4_metal_attention_view gate_weight, up_weight, router_weight, bias, completion;
    ds4_metal_attention_view input, gate, up, mid, logits, probs, selected, weights;
    float clamp;
    bool has_bias;
    uint64_t groups_x, groups_y, threads_x, threads_y, shared_bytes;
} ds4_metal_router_shared_launch;
int ds4_metal_router_shared_encode(void *, const ds4_metal_router_shared_launch *);
/* Cached completion storage is keyed by the actual backing buffer + offset,
 * independently retained and also appended to the existing transient owner. */
ds4_metal_attention_view ds4_metal_router_completion(ds4_metal_attention_view logits);
_Static_assert(sizeof(ds4_metal_router_select_one_args) == 20, "router select packet");
#endif
