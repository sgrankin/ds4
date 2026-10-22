#ifndef DS4_METAL_HC_CONTROL_H
#define DS4_METAL_HC_CONTROL_H
#include "attention_output.h"
/* Authoritative original shader layouts, shared with C fallback. */
typedef struct {
  int32_t n_hc;
  int32_t sinkhorn_iters;
  int64_t n_rows;
  int64_t mix_hc;
  uint64_t nb01;
  uint64_t nb1;
  float eps;
} ds4_gpu_hc_split_args;
typedef struct {
  int64_t n_embd;
  int64_t n_hc;
  int64_t n_tokens;
  uint64_t nb_x0;
  uint64_t nb_x1;
  uint64_t nb_x2;
  uint64_t nb_w0;
  uint64_t nb_w1;
  uint64_t nb0;
  uint64_t nb1;
} ds4_gpu_hc_weighted_sum_args;
typedef struct {
  float post_scale;
  float eps;
} ds4_gpu_output_hc_weights4_args;
typedef struct {
  int64_t n_embd;
  int32_t n_hc;
  int32_t sinkhorn_iters;
  int64_t n_rows;
  int64_t mix_hc;
  uint64_t nb_mix1;
  uint64_t nb_split1;
  uint64_t nb_x0;
  uint64_t nb_x1;
  uint64_t nb_x2;
  uint64_t nb0;
  uint64_t nb1;
  float eps;
} ds4_gpu_hc_split_weighted_sum_args;
typedef struct {
  int64_t n_embd;
  int32_t n_hc;
  int32_t sinkhorn_iters;
  int64_t n_rows;
  int64_t mix_hc;
  uint64_t nb_mix1;
  uint64_t nb_split1;
  uint64_t nb_x0;
  uint64_t nb_x1;
  uint64_t nb_x2;
  uint64_t nb0;
  uint64_t nb1;
  uint64_t nb_norm1;
  float eps;
  float norm_eps;
} ds4_gpu_hc_split_weighted_sum_norm_args;
typedef struct {
  int64_t n_embd;
  int64_t n_hc;
  int64_t n_tokens;
  uint64_t nb_block0;
  uint64_t nb_block1;
  uint64_t nb_add0;
  uint64_t nb_add1;
  uint64_t nb_res0;
  uint64_t nb_res1;
  uint64_t nb_res2;
  uint64_t nb_post0;
  uint64_t nb_post1;
  uint64_t nb_comb0;
  uint64_t nb_comb1;
  uint64_t nb_comb2;
  uint64_t nb0;
  uint64_t nb1;
  uint64_t nb2;
  int32_t has_add;
} ds4_gpu_hc_expand_args;
typedef struct {
  int32_t n;
  int32_t out_dim;
  float eps;
} ds4_gpu_hc_norm_mix_args;
enum {
  DS4_HC_SPLIT,
  DS4_HC_SUM,
  DS4_HC_SPLIT_SUM,
  DS4_HC_SPLIT_NORM,
  DS4_HC_NORM_MIX,
  DS4_HC_PRODUCER,
  DS4_HC_EXPAND_PRODUCER,
  DS4_HC_OUTPUT4,
  DS4_HC_OUTPUT_STAGES,
  DS4_HC_EXPAND,
  DS4_HC_SHARED_DOWN,
  DS4_HC_MATMUL,
  DS4_HC_EXPAND4,
  DS4_HC_NORM_MIX_CLUSTER,
  DS4_HC_OUTPUT_MUL,
  DS4_HC_OUTPUT_ADD,
  DS4_HC_OUTPUT_SIGMOID,
  DS4_HC_OUTPUT_SCALE
};
/* Named fixed binding sets. Native supplies every packet/PSO/grid decision. */
typedef struct {
  uint32_t kind;
  ds4_metal_attention_pipeline pipeline;
  ds4_gpu_hc_split_args split_args;
  ds4_gpu_hc_weighted_sum_args sum_args;
  ds4_gpu_hc_split_weighted_sum_args split_sum_args;
  ds4_gpu_hc_split_weighted_sum_norm_args split_norm_args;
  ds4_gpu_hc_norm_mix_args norm_mix_args;
  ds4_gpu_hc_expand_args expand_args;
  ds4_gpu_output_hc_weights4_args output4_args;
  ds4_metal_mv_args mv_args;
  ds4_gpu_bin_args mul_args, add_args;
  ds4_metal_attention_unary_args sigmoid_args, scale_args;
  ds4_metal_attention_pipeline mul, add_pipeline, sigmoid, scale_pipeline;
  ds4_metal_attention_view out, norm_out, mix, split, residual, block,
      block_add;
  ds4_metal_attention_view previous, post, comb, input, weights, scale, base,
      norm_weights;
  ds4_metal_attention_view completion, barrier;
  uint32_t barrier_target;
  uint64_t gx, gy, gz, tx, ty, shared_bytes, mul_threads, add_threads,
      unary_groups, unary_threads;
} ds4_metal_hc_control_launch;
/* Retained completion/barrier storage remains in its actual ObjC cache. Native
 * computes generation wrap and six-group target, then publishes the generation.
 */
typedef struct {
  ds4_metal_attention_view completion, barrier;
  uint32_t generation;
} ds4_metal_hc_completion;
ds4_metal_attention_pipeline ds4_metal_hc_control_pipeline(uint32_t kind);
bool ds4_metal_hc_control_output4_test(void);
bool ds4_metal_hc_control_tp_two(void);
bool ds4_metal_hc_control_completion(ds4_metal_attention_view mix,
                                     bool expanded,
                                     ds4_metal_hc_completion *out);
void ds4_metal_hc_control_generation(ds4_metal_attention_view mix,
                                     uint32_t generation);
int ds4_metal_hc_control_encode(void *command,
                                const ds4_metal_hc_control_launch *launch);

#endif
