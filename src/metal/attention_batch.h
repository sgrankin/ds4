#ifndef DS4_METAL_ATTENTION_BATCH_H
#define DS4_METAL_ATTENTION_BATCH_H
#include "attention_decode.h"
#include "index_selection.h"
#include "../ds4_image.h"
/* Original shader records, shared once with the flag-off C implementation. */
typedef struct {
    int32_t ne01, ne30, ne31, ne32, ne33;
    uint64_t nb31, nb32, nb33;
} ds4_metal_attention_block_args;
typedef struct {
    uint32_t n_tokens, n_head, n_raw, raw_cap, raw_start, n_comp;
    uint32_t top_k, pos0, window, ratio, comp_kv_f16, n_splits;
    uint64_t q_token_stride, q_head_stride, raw_row_stride, comp_row_stride;
    uint64_t topk_token_stride, dst_token_stride, dst_head_stride;
    float scale;
} ds4_metal_attention_indexed_args;
typedef struct {
    uint32_t n_comp, n_tokens, pos0, ratio, n_keys;
} ds4_metal_attention_causal_args;
_Static_assert(sizeof(ds4_metal_attention_block_args) == 48, "Metal attention block packet");
_Static_assert(sizeof(ds4_metal_attention_indexed_args) == 112, "Metal indexed attention packet");
_Static_assert(sizeof(ds4_metal_attention_causal_args) == 20, "Metal causal mask packet");

enum {
    DS4_GPU_PREFILL_MASK_CACHE_RAW = 1,
    DS4_GPU_PREFILL_MASK_CACHE_RATIO4 = 2,
    DS4_GPU_PREFILL_MASK_CACHE_RATIO128 = 3,
    DS4_GPU_PREFILL_MASK_CACHE_SLOTS = 3,
};

/* Existing zero-prefix cache remains one resource owner. Views returned here
 * are separately retained through encoding; filling/publication stays native. */
typedef struct {
    uint32_t kind, n_tokens, n_comp, n_keys, window, ratio, nqptg, ncpsg;
    bool has_kvpad, bc_mask;
    uint64_t mask_bytes, blk_bytes;
} ds4_metal_attention_cache_key;
typedef struct {
    ds4_metal_attention_view mask, block;
    void *entry;
    bool created, block_ready;
} ds4_metal_attention_cache;
ds4_metal_attention_cache ds4_metal_attention_cache_acquire(const ds4_metal_attention_cache_key *);
void ds4_metal_attention_cache_publish(void *entry, bool mask_valid, bool block_ready);
enum { DS4_ATTN_BATCH_BLOCK, DS4_ATTN_BATCH_TOPK };
ds4_metal_attention_view ds4_metal_attention_batch_scratch(uint32_t kind, uint64_t bytes, const char *label);
enum {
    DS4_ATTN_BATCH_PAD, DS4_ATTN_BATCH_BLOCK_MAP, DS4_ATTN_BATCH_VECTOR,
    DS4_ATTN_BATCH_NONVECTOR, DS4_ATTN_BATCH_CAUSAL, DS4_ATTN_BATCH_SORT,
    DS4_ATTN_BATCH_INDEX_SPLIT, DS4_ATTN_BATCH_INDEX_REDUCE,
    DS4_ATTN_BATCH_INDEX_DECODE, DS4_ATTN_BATCH_INDEX_DUAL, DS4_ATTN_BATCH_INDEX,
};
ds4_metal_attention_pipeline ds4_metal_attention_batch_pipeline(uint32_t kind,
    uint32_t nsg, uint32_t nwg, bool pad, bool boundary_mask, bool has_sinks);
bool ds4_metal_attention_batch_mpp(void);
/* Replaces the caller's retained command after the existing synchronized
 * diagnostic boundary. Zero means failure; the previous retain stays owned. */
int ds4_metal_attention_profile_begin(void **command, double *started);
int ds4_metal_attention_profile_stage(void **command, const char *mode,
    const char *stage, uint32_t rows, uint32_t comp, uint32_t keys,
    uint32_t heads, uint32_t width, uint32_t window, uint32_t ratio, double *started);
bool ds4_metal_attention_profile_enabled(void);
int ds4_metal_attention_batch_pad(void *, ds4_metal_attention_pipeline,
    const ds4_metal_attention_pad_args *, ds4_metal_attention_view kv,
    ds4_metal_attention_view mask, ds4_metal_attention_view pad, uint64_t groups);
int ds4_metal_attention_block_map(void *, ds4_metal_attention_pipeline,
    const ds4_metal_attention_block_args *, ds4_metal_attention_view mask,
    ds4_metal_attention_view block, uint64_t gx, uint64_t gy);
int ds4_metal_attention_nonvector(void *, ds4_metal_attention_pipeline,
    const ds4_metal_attention_vec_args *, ds4_metal_attention_view q,
    ds4_metal_attention_view kv, ds4_metal_attention_view mask,
    ds4_metal_attention_view sinks, ds4_metal_attention_view pad,
    ds4_metal_attention_view block, ds4_metal_attention_view dst,
    uint64_t gx, uint64_t gy, uint64_t nsg, uint64_t shared);
int ds4_metal_attention_causal_mask(void *, ds4_metal_attention_pipeline,
    const ds4_metal_attention_causal_args *, ds4_metal_attention_view src,
    ds4_metal_attention_view dst, uint64_t threads);
int ds4_metal_attention_index_sort(void *, ds4_metal_attention_pipeline,
    const ds4_metal_index_rows_args *, ds4_metal_attention_view src,
    ds4_metal_attention_view dst, uint64_t groups, uint64_t threads, uint64_t shared);
int ds4_metal_attention_index_run(void *, ds4_metal_attention_pipeline,
    const ds4_metal_attention_indexed_args *, ds4_metal_attention_view q,
    ds4_metal_attention_view raw, ds4_metal_attention_view comp,
    ds4_metal_attention_view topk, ds4_metal_attention_view sinks,
    ds4_metal_attention_view dst, bool split,
    uint64_t gx, uint64_t gy, uint64_t gz, uint64_t shared);
int ds4_metal_attention_index_reduce(void *, ds4_metal_attention_pipeline,
    const ds4_metal_attention_indexed_args *, ds4_metal_attention_view src,
    ds4_metal_attention_view sinks, ds4_metal_attention_view dst, uint64_t rows);
#endif
