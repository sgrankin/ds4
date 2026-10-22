#ifndef DS4_METAL_ATTENTION_OUTPUT_H
#define DS4_METAL_ATTENTION_OUTPUT_H
#include "attention_decode.h"
#include "dense_matmul.h"
#include "routed_batch.h"
/* The original exported TensorOps attention output is a full 64-token tile. */
enum { DS4_METAL_ATTN_OUT_MPP_TILE_N = 64 };
enum {
    DS4_OUT_LOW_DIRECT,
    DS4_OUT_LOW_IDS,
    DS4_OUT_MAP,
    DS4_OUT_MAPPED,
    DS4_OUT_MPP,
    DS4_OUT_KSLICE_MV,
    DS4_OUT_KSLICE_MM
};
typedef union {
    ds4_gpu_mul_mv_id_args low;
    ds4_gpu_mul_mm_id_args grouped;
    ds4_gpu_mul_mm_id_map_args map;
    ds4_metal_mv_args vector;
    ds4_metal_mm_args matrix;
} ds4_metal_output_args;
/* One fixed dispatch borrows named buffer regions and the actual shader packet.
 * View offsets are already absolute; no tensor/property offset is added again. */
typedef struct {
    uint32_t kind;
    ds4_metal_output_args args;
    const ds4_metal_hc_pipeline *pipeline;
    ds4_metal_attention_view weights, input, output, ids, workmap;
    uint64_t work_ids_offset, work_list_offset;
    uint64_t groups_x, groups_y, groups_z, threads_x, threads_y, shared_bytes;
    bool hint_weights, fold;
    uint32_t fold_slot, fold_value;
} ds4_metal_output_launch;
typedef struct {
    bool active;
    uint32_t slot, value;
} ds4_metal_output_fold;
enum { DS4_OUT_IDS_SCRATCH, DS4_OUT_WORKMAP_SCRATCH };
ds4_metal_attention_view ds4_metal_output_scratch(uint32_t kind, uint64_t bytes, bool transient,
                                                  const char *label);
int ds4_metal_output_encode(void *command, const ds4_metal_output_launch *);
ds4_metal_output_fold ds4_metal_output_fold_take(const ds4_gpu_tensor *, uint64_t bytes);
void ds4_metal_output_fold_publish(void);
#endif
