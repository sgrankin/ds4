#ifndef DS4_METAL_PRIMITIVES_H
#define DS4_METAL_PRIMITIVES_H
#include "compressor.h"
#include "scheduling.h"
/* Original FP8/QAT packets, shared with all C fallback callers once. */
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
  int32_t n_rot;
} ds4_metal_fp8_quant_args;
typedef struct {
  uint32_t n_rows;
  uint32_t head_dim;
  uint64_t row_stride;
} ds4_metal_qat_args;

_Static_assert(sizeof(ds4_metal_fp8_quant_args) == 104, "FP8 packet");
_Static_assert(sizeof(ds4_metal_qat_args) == 16, "QAT packet");
enum {
  DS4_PRIM_FP8,
  DS4_PRIM_QAT,
  DS4_PRIM_FP8_STORE,
  DS4_PRIM_ROPE_STORE,
  DS4_PRIM_ADD,
  DS4_PRIM_ADD3,
  DS4_PRIM_SWIGLU,
  DS4_PRIM_COPY_F32,
  DS4_PRIM_COPY_F32_F16,
  DS4_PRIM_COPY_F16,
  DS4_PRIM_COPY_F16_F32,
  DS4_PRIM_COPY_F32_F16_CONTIG,
  DS4_PRIM_COPY_F16_F32_CONTIG,
  DS4_PRIM_COPY_F16_CONTIG,
  DS4_PRIM_COPY_F16_BLIT
};
ds4_metal_attention_pipeline ds4_metal_primitive_pipeline(uint32_t kind);
/* Independent retained scratch aliases; callers release through ModelBuffer. */
ds4_metal_attention_view ds4_metal_primitive_round_scratch(bool half,
                                                           uint64_t bytes);
typedef union {
  ds4_metal_fp8_quant_args fp8;
  ds4_metal_qat_args qat;
  ds4_metal_kv_store_args store;
  ds4_gpu_add_flat_args add;
  ds4_gpu_glu_args glu;
} ds4_metal_primitive_packet;
typedef struct {
  uint32_t kind;
  ds4_metal_attention_pipeline pipeline;
  ds4_metal_primitive_packet args;
  ds4_metal_attention_rope_args rope;
  ds4_metal_attention_view input, other, third, out;
  uint64_t groups, threads, shared;
} ds4_metal_primitive_launch;
int ds4_metal_primitive_encode(void *command,
                               const ds4_metal_primitive_launch *);
int ds4_metal_primitive_copy(void *command, uint32_t kind,
                             ds4_metal_attention_pipeline,
                             const ds4_metal_copy_args *, uint32_t count,
                             ds4_metal_attention_view source,
                             ds4_metal_attention_view out, uint64_t groups_x,
                             uint64_t groups_y, uint64_t threads);
/* Borrowed-copy compatibility for remaining C-family callers. Native builds
 * all packets/geometry; these actions never call an old C encoder. */
int ds4_native_metal_copy_1d(void *command, ds4_metal_attention_view source,
                             ds4_metal_attention_view out, uint32_t count,
                             uint32_t kind, bool zero_ok);
int ds4_native_metal_copy_3d(void *command, ds4_metal_attention_view source,
                             ds4_metal_attention_view out, uint32_t columns,
                             uint32_t rows, uint32_t planes,
                             uint64_t source_col_stride,
                             uint64_t source_row_stride,
                             uint64_t source_plane_stride,
                             uint64_t out_row_stride, uint64_t out_plane_stride,
                             uint32_t kind);
int ds4_native_metal_set_rows(void *command, ds4_metal_attention_view source,
                              ds4_metal_attention_view out, const int32_t *rows,
                              uint32_t count, uint32_t out_rows,
                              uint32_t width);
int ds4_native_metal_add_1d(void *command, ds4_metal_attention_view a,
                            ds4_metal_attention_view b,
                            ds4_metal_attention_view out, uint32_t count);
#endif
