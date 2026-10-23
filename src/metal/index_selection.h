#ifndef DS4_METAL_INDEX_SELECTION_H
#define DS4_METAL_INDEX_SELECTION_H
#include "../ds4_gpu.h"

/* Actual shader packets moved once; C fallback and native construction share
 * field widths/padding. Stack packets are borrowed during encode only. */
typedef struct {
    uint32_t n_rows, n_head, head_dim, cache_f16;
    float scale;
} ds4_metal_index_score_args;
typedef struct {
    uint32_t n_rows, n_tokens, n_head, head_dim;
    uint32_t pos0, cache_f16, row_group_size, pad0;
    uint64_t q_token_stride, q_head_stride, weights_token_stride, score_token_stride;
    float scale;
} ds4_metal_index_batch_args;
typedef struct {
    int32_t ne00, ne01, ne02, ne03;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne0, ne1, ne2, ne3, top_k;
    uint32_t causal_start, causal_ratio;
} ds4_metal_index_sort_args;
typedef struct {
    int64_t ne00, ne01, ne02, ne03;
    uint64_t nb00, nb01, nb02, nb03;
    int32_t ne0, ne1, ne2, ne3, top_k, len;
    uint32_t causal_start, causal_ratio, block_width, block_top_k;
} ds4_metal_index_merge_args;
typedef struct {
    int64_t ne00, ne01;
    uint64_t nb00, nb01;
    int64_t ne0, ne1;
    uint64_t nb0, nb1;
} ds4_metal_index_rows_args;
_Static_assert(sizeof(ds4_metal_index_score_args) == 20, "Metal index score packet");
_Static_assert(sizeof(ds4_metal_index_batch_args) == 72, "Metal index batch packet");
_Static_assert(sizeof(ds4_metal_index_sort_args) == 80, "Metal index sort packet");
_Static_assert(sizeof(ds4_metal_index_merge_args) == 104, "Metal index merge packet");
_Static_assert(sizeof(ds4_metal_index_rows_args) == 64, "Metal index row packet");

struct ds4_metal_hc_pipeline;
typedef struct {
    const struct ds4_metal_hc_pipeline *handle;
    uint64_t max_threads;
} ds4_metal_index_pipeline;
typedef struct {
    bool quality;
    uint64_t sort_threads, max_shared_bytes;
} ds4_metal_index_facts;
enum {
    DS4_METAL_INDEX_SCORE = 1, DS4_METAL_INDEX_DIRECT,
    DS4_METAL_INDEX_BATCH, DS4_METAL_INDEX_TILED, DS4_METAL_INDEX_TILED_F32,
    DS4_METAL_INDEX_PACK, DS4_METAL_INDEX_PACKED_SCORE,
    DS4_METAL_INDEX_SORT, DS4_METAL_INDEX_SORT_CAUSAL, DS4_METAL_INDEX_SORT_SHUFFLE,
    DS4_METAL_INDEX_MERGE, DS4_METAL_INDEX_MERGE_CAUSAL, DS4_METAL_INDEX_MERGE_PREFIX,
    DS4_METAL_INDEX_MASK_FILL, DS4_METAL_INDEX_MASK_SCATTER, DS4_METAL_INDEX_ROW_SORT,
};
ds4_metal_index_pipeline ds4_metal_index_get_pipeline(uint32_t kind);
ds4_metal_index_facts ds4_metal_index_get_facts(void);
int ds4_metal_index_reserve_scratch(uint64_t bytes);

typedef union {
    ds4_metal_index_score_args one;
    ds4_metal_index_batch_args batch;
} ds4_metal_index_score_packet;
typedef struct {
    ds4_metal_index_pipeline pipeline;
    bool batch;
    ds4_metal_index_score_packet args;
    const ds4_gpu_tensor *q, *weights, *keys;
    ds4_gpu_tensor *scores;
    uint64_t groups_x, groups_y, threads_x, threads_y, shared_bytes;
} ds4_metal_index_score_launch;

typedef union {
    ds4_metal_index_sort_args sort;
    ds4_metal_index_merge_args merge;
} ds4_metal_index_topk_packet;
typedef struct {
    ds4_metal_index_pipeline pipeline;
    bool merge, to_selected;
    ds4_metal_index_topk_packet args;
    const ds4_gpu_tensor *scores;
    ds4_gpu_tensor *selected;
    uint64_t source_offset, target_offset;
    uint64_t groups, threads, shared_bytes;
} ds4_metal_index_topk_launch;

typedef struct {
    ds4_metal_index_pipeline pipeline;
    ds4_metal_index_rows_args args;
    const ds4_gpu_tensor *source;
    ds4_gpu_tensor *destination;
    uint64_t groups, threads, shared_bytes;
} ds4_metal_index_rows_launch;

typedef struct {
    ds4_metal_index_pipeline pipeline;
    bool scores;
    uint32_t args[4], range[2];
    const ds4_gpu_tensor *q, *weights, *keys;
    ds4_gpu_tensor *output;
    const ds4_gpu_tensor *packed_source;
    uint64_t flags_offset, keys_offset;
    uint64_t groups_x, groups_y, shared_bytes;
} ds4_metal_index_pack_launch;

/* Fixed index-family resource bindings only. The existing command token owns
 * any independent wait; a borrowed batch is never committed by these leaves. */
int ds4_metal_index_score_encode(void *, const ds4_metal_index_score_launch *);
int ds4_metal_index_topk_encode(void *, const ds4_metal_index_topk_launch *);
int ds4_metal_index_rows_encode(void *, const ds4_metal_index_rows_launch *);
int ds4_metal_index_pack_encode(void *, const ds4_metal_index_pack_launch *);
#endif
