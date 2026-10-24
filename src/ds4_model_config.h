#ifndef DS4_MODEL_CONFIG_H
#define DS4_MODEL_CONFIG_H

#include "ds4_gguf.h"
#include <stddef.h>

/* Actual model-shape records shared with the remaining C engine. Maxima size
 * fixed backend workspaces; they are not the dimensions of every loaded model.
 * Runtime configuration is per model. The legacy engine publishes a copy only
 * after successful validation, until its hot loops take explicit config values.
 */
enum {
    DS4_MAX_LAYER            = 79,
    DS4_MAX_EMBD             = 7168,
    DS4_MAX_VOCAB            = 248320,
    DS4_MAX_HEAD             = 128,
    DS4_MAX_HEAD_KV          = 2,
    DS4_MAX_HEAD_DIM         = 576,
    DS4_MAX_VALUE_DIM        = 512,
    DS4_MAX_ROT              = 64,
    DS4_MAX_OUT_GROUP        = 16,
    DS4_MAX_LORA_Q           = 2048,
    DS4_MAX_LORA_O           = 1024,
    DS4_MAX_EXPERT           = 512,
    DS4_MAX_EXPERT_USED      = 10,
    DS4_MAX_EXPERT_SHARED    = 1,
    DS4_MAX_FF_EXP           = 3072,
    DS4_MAX_HASH_LAYER       = 3,
    DS4_MAX_SWA              = 128,
    DS4_MAX_INDEXER_HEAD     = 64,
    DS4_MAX_INDEXER_HEAD_DIM = 128,
    DS4_MAX_INDEXER_TOP_K    = 2048,
    DS4_MAX_HC               = 4,
    DS4_MAX_HC_SINKHORN_ITER = 20,
    DS4_MAX_KDA_HEAD         = 64,
    DS4_MAX_KDA_HEAD_DIM     = 128,
    DS4_MAX_KDA_CONV         = 4,
    DS4_MAX_PLE_NGRAM        = 8,
    DS4_MAX_PLE_HEADS        = 64,
};

typedef enum {
    DS4_MODEL_FAMILY_DEEPSEEK4 = 0,
    DS4_MODEL_FAMILY_GLM_DSA   = 1,
    DS4_MODEL_FAMILY_DEEPSEEK41 = 2,
    DS4_MODEL_FAMILY_QWEN4_EXP = 3,
} ds4_model_family;

typedef enum {
    /* Value 0 is reserved for the retired original V4 Flash profile. */
    DS4_VARIANT_PRO   = 1,
    DS4_VARIANT_GLM53_FULL = 2, /* Retained full DSA layout; preserve public variant ID. */
    DS4_VARIANT_GLM53 = 3,
    DS4_VARIANT_FLASH41 = 4,
    DS4_VARIANT_QWEN4_EXP = 5,
    DS4_VARIANT_QWEN4_MINI = 6,
} ds4_variant;

typedef struct {
    const char *name;
    ds4_model_family family;
    ds4_variant variant;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_head_dim;
    uint32_t n_value_dim;
    uint32_t n_rot;
    uint32_t n_out_group;
    uint32_t n_lora_q;
    uint32_t n_lora_o;
    uint32_t n_expert;
    uint32_t n_expert_used;
    uint32_t n_expert_shared;
    uint32_t n_ff_exp;
    uint32_t n_ff_dense;
    uint32_t n_hash_layer;
    uint32_t n_swa;
    uint32_t n_indexer_head;
    uint32_t n_indexer_head_dim;
    uint32_t n_indexer_top_k;
    uint32_t n_hc;
    uint32_t n_hc_sinkhorn_iter;
    uint32_t n_nextn_predict;
    uint32_t n_leading_dense;
    uint32_t n_kv_lora;
    uint32_t n_key_mla;
    uint32_t n_value_mla;
    uint32_t n_kda_head;
    uint32_t n_kda_head_dim;
    uint32_t n_kda_conv;
    uint32_t n_hc_lowrank;
    uint32_t n_lin_k_head;
    uint32_t n_lin_v_head;
    uint32_t n_lin_head_dim;
    uint32_t n_lin_conv;
    uint32_t n_full_attn_interval;
    uint32_t n_ple_layer;
    uint32_t n_ple_ngram;
    uint32_t n_ple_heads_per_ngram;
    uint32_t n_ple_conv;
    uint32_t n_ple_head_dim;
    int32_t ple_eos_id;
    float rms_eps;
    float hc_eps;
    float expert_weight_scale;
    float swiglu_clamp_exp;
    float rope_freq_base;
    float rope_scale_factor;
    float rope_yarn_beta_fast;
    float rope_yarn_beta_slow;
    float compress_rope_freq_base;
    float kda_gate_lower_bound;
    uint64_t rope_orig_ctx;
} ds4_shape;

/* By-value compatibility boundary: no engine/session layouts, pointers to
 * metadata, or allocations. name points to static model-profile text. Ratios
 * beyond shape.n_layer are zero. The native owner additionally derives typed
 * layer policy from this validated value rather than process globals.
 */
typedef struct {
    ds4_shape shape;
    uint32_t compress_ratios[DS4_MAX_LAYER];
} ds4_model_config;

/* DeepSeek V4.1 metadata and disk-only Engram layout validation. Borrow model
 * descriptors/mapping only for this call; retain no references to their bytes.
 * On success write the complete config. On failure leave output unchanged and
 * report an ordinary message without the caller's "ds4: " prefix. The C caller
 * retains its fatal-error policy. Other model families keep their C validators.
 */
bool ds4_model_config_deepseek41(const ds4_model *model,
                                 ds4_model_config *output,
                                 char *error, size_t error_capacity);

#endif
