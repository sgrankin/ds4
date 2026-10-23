#ifndef DS4_METAL_EMBEDDING_H
#define DS4_METAL_EMBEDDING_H
#include "attention_decode.h"
#include "routed_batch.h"
/* The actual original shader packets, shared once with unchanged C users. */
typedef struct {
  int32_t ne00t;
  int32_t ne00;
  uint64_t nb01;
  uint64_t nb02;
  uint64_t nb03;
  int32_t ne10;
  uint64_t nb10;
  uint64_t nb11;
  uint64_t nb12;
  uint64_t nb1;
  uint64_t nb2;
  uint64_t nb3;
} ds4_metal_embedding_rows_args;
typedef struct {
  int32_t n_embd;
  int32_t n_vocab;
  int32_t n_tokens;
  uint64_t src_row_bytes;
  uint64_t dst_row_bytes;
  uint64_t token_stride;
} ds4_metal_embedding_quant_args;
typedef struct {
  int32_t ne00;
  int32_t ne01;
  int32_t ne02;
  int32_t ne03;
  uint64_t nb00;
  uint64_t nb01;
  uint64_t nb02;
  uint64_t nb03;
  int32_t ne0;
  int32_t ne1;
  int32_t ne2;
  int32_t ne3;
  uint64_t nb0;
  uint64_t nb1;
  uint64_t nb2;
  uint64_t nb3;
} ds4_metal_embedding_repeat_args;
typedef struct {
  uint32_t in_dim;
  uint32_t out_dim;
  uint32_t n_rows;
} ds4_metal_embedding_bf16_args;

_Static_assert(sizeof(ds4_metal_embedding_rows_args) == 88, "GetRows packet");
_Static_assert(sizeof(ds4_metal_embedding_quant_args) == 40,
               "Quant GetRows packet");
_Static_assert(sizeof(ds4_metal_embedding_repeat_args) == 96,
               "HC repeat packet");
_Static_assert(sizeof(ds4_metal_embedding_bf16_args) == 12, "GLM BF16 packet");
enum {
  DS4_EMBED_F16,
  DS4_EMBED_Q8,
  DS4_EMBED_Q4_0,
  DS4_EMBED_Q4_K,
  DS4_EMBED_REPEAT,
  DS4_EMBED_BF16,
  DS4_EMBED_F32_GATHER,
  DS4_EMBED_PACKED_GATHER
};
typedef struct {
  const ds4_metal_hc_pipeline *pipeline;
  uint64_t max_threads;
} ds4_metal_embedding_pipeline;
ds4_metal_embedding_pipeline ds4_metal_embedding_pipeline_get(uint32_t kind);
/* Each returned view owns one reference; use the ordinary ModelBuffer release.
 */
ds4_metal_attention_view ds4_metal_embedding_scratch(uint64_t bytes);
ds4_metal_model_buffer *ds4_metal_embedding_exact_view(const void *model,
                                                       uint64_t size,
                                                       uint64_t offset,
                                                       uint64_t bytes,
                                                       uint64_t *inner);
typedef union {
  ds4_metal_embedding_rows_args rows;
  ds4_metal_embedding_quant_args quant;
  ds4_metal_embedding_repeat_args repeat;
  ds4_metal_embedding_bf16_args bf16;
  uint32_t packed_rows[2];
} ds4_metal_embedding_packet;
typedef struct {
  uint32_t kind;
  const ds4_metal_hc_pipeline *pipeline;
  ds4_metal_embedding_packet args;
  ds4_metal_attention_view source, ids, out;
  bool inline_token, dispatch_threads;
  int32_t token;
  uint64_t groups_x, groups_y, threads;
} ds4_metal_embedding_launch;
int ds4_metal_embedding_encode(void *command,
                               const ds4_metal_embedding_launch *);
#endif
