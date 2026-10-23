#ifndef DS4_DSPARK_WEIGHTS_H
#define DS4_DSPARK_WEIGHTS_H

#include "ds4_model_weights.h"

#define DS4_DSPARK_MAX_TARGET_LAYERS 8
#define DS4_DSPARK_MAX_STAGES 8

/* Actual borrowed draft-weight records. The C metadata/binding path still
 * constructs these; the draft model mapping outlives their graph owner. */
typedef struct {
    ds4_tensor *main_proj;
    ds4_tensor *main_norm;
    ds4_tensor *norm;
    ds4_tensor *hc_head_base;
    ds4_tensor *hc_head_fn;
    ds4_tensor *hc_head_scale;
    ds4_tensor *markov_w1;
    ds4_tensor *markov_w2;
    ds4_tensor *confidence_proj;
    ds4_layer_weights block;
} ds4_dspark_stage_weights;

typedef struct {
    uint32_t n_stages;
    uint32_t block_size;
    uint32_t markov_rank;
    uint32_t noise_token_id;
    uint32_t target_layer_count;
    uint32_t target_layers[DS4_DSPARK_MAX_TARGET_LAYERS];
    uint32_t present_tensors;
    uint32_t missing_tensors;
    uint32_t invalid_tensors;
    uint32_t metadata_errors;
    uint32_t n_experts, n_experts_used;
    bool v41;
    bool has_block_size;
    bool has_markov_rank;
    bool has_noise_token_id;
    bool has_target_layers;
    ds4_dspark_stage_weights stage[DS4_DSPARK_MAX_STAGES];
} ds4_dspark_weights;

#endif
