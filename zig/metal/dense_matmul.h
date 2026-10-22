#ifndef DS4_METAL_DENSE_MATMUL_H
#define DS4_METAL_DENSE_MATMUL_H
#include "bridge_types.h"
#include <stdio.h>
#include <stdlib.h>

/* Reuse the actual 112-byte MV/EXT and 88-byte MM packets defined once in
 * bridge.h. Policy is native; these fixed leaves only borrow arguments and
 * bind ARC buffers/PSOs for one ordinary dense dispatch or paired matvec. */
enum {
    DS4_DENSE_MV, DS4_DENSE_EXT, DS4_DENSE_MM,
    DS4_DENSE_MPP = 1, DS4_DENSE_QUALITY = 2, DS4_DENSE_M5 = 4,
    DS4_DENSE_VERIFY_EXACT = 8, DS4_DENSE_TP_TWO = 16,
    DS4_DENSE_M3_ULTRA = 32, DS4_DENSE_PAIR_NR4 = 64,
};
typedef union {
    ds4_metal_mv_args mv;
    ds4_metal_mv_ext_args ext;
    ds4_metal_mm_args mm;
} ds4_metal_dense_args;
typedef struct {
    uint32_t kind;
    ds4_metal_dense_args args;
    const ds4_metal_hc_pipeline *pipeline;
    const ds4_metal_model_buffer *weights;
    uint64_t weight_offset;
    const ds4_gpu_tensor *x;
    ds4_gpu_tensor *out;
    uint64_t x_offset, out_offset;
    uint64_t groups_x, groups_y, threads_x, threads_y;
    uint64_t repeats, x_step, out_step;
    uint32_t shared_bytes;
} ds4_metal_dense_launch;
typedef struct {
    ds4_metal_mv_args args0, args1;
    bool two_args;
    const ds4_metal_hc_pipeline *pipeline;
    const ds4_metal_model_buffer *weights0, *weights1;
    uint64_t weight_offset0, weight_offset1;
    const ds4_gpu_tensor *x;
    ds4_gpu_tensor *out0, *out1;
    uint64_t groups, simdgroups;
    uint32_t shared_bytes;
} ds4_metal_dense_pair_launch;
uint32_t ds4_metal_dense_facts(bool paired);
const ds4_metal_hc_pipeline *ds4_metal_dense_pipeline(uint32_t kind,
    const char *name, int16_t nsg, int16_t nxpsg, bool bc_in, bool bc_out);
int ds4_metal_dense_encode(void *, const ds4_metal_dense_launch *);
int ds4_metal_dense_pair_encode(void *, const ds4_metal_dense_pair_launch *);
/* Retained scratch is released with the existing model-buffer release. Its
 * cache slot/resize behavior and unretained-command ownership are unchanged. */
ds4_metal_model_buffer *ds4_metal_dense_retain_scratch(uint64_t bytes);
int ds4_metal_dense_unpack(void *, const ds4_metal_hc_pipeline *,
    const ds4_metal_model_buffer *weights, uint64_t weight_offset,
    const ds4_metal_model_buffer *scratch, uint32_t blocks, uint64_t groups);
const ds4_metal_hc_pipeline *ds4_metal_dense_unpack_pipeline(void);
void ds4_metal_dense_warn_mpp(void);
double ds4_metal_dense_now_ms(void);

/* The remaining general-quant C dispatcher delegates its Q8 arm to this whole
 * action. No native controller calls back into the old Q8 body. */
int ds4_native_metal_q8_legacy(ds4_gpu_tensor *, const void *, uint64_t,
    uint64_t, uint64_t, uint64_t, const ds4_gpu_tensor *, uint64_t, bool, bool);
#endif
