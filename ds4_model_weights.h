#ifndef DS4_MODEL_WEIGHTS_H
#define DS4_MODEL_WEIGHTS_H

#include "ds4_model_config.h"

/* Actual GGUF tensor tags and borrowed weight records, shared once with the C
 * fallback. A bound table owns no tensor bytes: its model mapping must outlive
 * every use of these pointers. Other model families retain their C binders. */
enum {
    DS4_TENSOR_F32      = 0,
    DS4_TENSOR_F16      = 1,
    DS4_TENSOR_Q4_0     = 2,
    DS4_TENSOR_Q4_1     = 3,
    DS4_TENSOR_Q8_0     = 8,
    DS4_TENSOR_Q2_K     = 10,
    DS4_TENSOR_Q4_K     = 12,
    DS4_TENSOR_Q5_K     = 13,
    DS4_TENSOR_Q6_K     = 14,
    DS4_TENSOR_Q8_K     = 15,
    DS4_TENSOR_IQ2_XXS  = 16,
    DS4_TENSOR_I8       = 24,
    DS4_TENSOR_I32      = 26,
    DS4_TENSOR_BF16     = 30,
    DS4_TENSOR_MXFP4    = 39,
};

typedef struct {
    ds4_tensor *hc_attn_fn;
    ds4_tensor *hc_attn_scale;
    ds4_tensor *hc_attn_base;
    ds4_tensor *attn_norm;
    ds4_tensor *attn_q_a;
    ds4_tensor *attn_q_a_norm;
    ds4_tensor *attn_q_b;
    ds4_tensor *attn_kv;
    ds4_tensor *attn_kv_a_mqa;
    ds4_tensor *attn_kv_a_norm;
    ds4_tensor *attn_k_b;
    ds4_tensor *attn_v_b;
    ds4_tensor *attn_sinks;
    ds4_tensor *attn_output;
    ds4_tensor *attn_output_a;
    ds4_tensor *attn_output_b;
    ds4_tensor *kda_q;
    ds4_tensor *kda_k;
    ds4_tensor *kda_v;
    ds4_tensor *kda_q_conv;
    ds4_tensor *kda_k_conv;
    ds4_tensor *kda_v_conv;
    ds4_tensor *kda_f_a;
    ds4_tensor *kda_f_b;
    ds4_tensor *kda_dt_bias;
    ds4_tensor *kda_a_log;
    ds4_tensor *kda_beta;
    ds4_tensor *kda_g_a;
    ds4_tensor *kda_g_b;
    ds4_tensor *kda_o_norm;
    ds4_tensor *kda_output;
    ds4_tensor *attn_compressor_ape;
    ds4_tensor *attn_compressor_kv;
    ds4_tensor *attn_compressor_gate;
    ds4_tensor *attn_compressor_norm;
    ds4_tensor *indexer_attn_q_b;
    ds4_tensor *indexer_attn_k;
    ds4_tensor *indexer_k_norm;
    ds4_tensor *indexer_k_norm_b;
    ds4_tensor *indexer_proj;
    ds4_tensor *indexer_compressor_ape;
    ds4_tensor *indexer_compressor_kv;
    ds4_tensor *indexer_compressor_gate;
    ds4_tensor *indexer_compressor_norm;
    ds4_tensor *hc_ffn_fn;
    ds4_tensor *hc_ffn_scale;
    ds4_tensor *hc_ffn_base;
    ds4_tensor *ffn_norm;
    ds4_tensor *ffn_gate_tid2eid;
    ds4_tensor *ffn_gate;
    ds4_tensor *ffn_up;
    ds4_tensor *ffn_down;
    ds4_tensor *ffn_gate_inp;
    ds4_tensor *ffn_exp_probs_b;
    ds4_tensor *ffn_exp_probs_vl;
    ds4_tensor *ffn_gate_exps;
    ds4_tensor *ffn_up_exps;
    ds4_tensor *ffn_down_exps;
    ds4_tensor *ffn_gate_shexp;
    ds4_tensor *ffn_up_shexp;
    ds4_tensor *ffn_down_shexp;
    ds4_tensor *nextn_eh_proj;
    ds4_tensor *nextn_enorm;
    ds4_tensor *nextn_hnorm;
    ds4_tensor *nextn_shared_head_norm;
    ds4_tensor *engram_kv;
    ds4_tensor *engram_q_norm;
    ds4_tensor *engram_k_norm;
    /* Qwen3.8-Flash-Next */
    ds4_tensor *hc_attn_norm;
    ds4_tensor *hc_attn_down;
    ds4_tensor *hc_attn_up;
    ds4_tensor *hc_attn_inject;
    ds4_tensor *hc_ffn_norm;
    ds4_tensor *hc_ffn_down;
    ds4_tensor *hc_ffn_up;
    ds4_tensor *hc_ffn_inject;
    ds4_tensor *attn_q;
    ds4_tensor *attn_k;
    ds4_tensor *attn_v;
    ds4_tensor *attn_q_norm;
    ds4_tensor *attn_k_norm;
    ds4_tensor *indexer_q_proj;
    ds4_tensor *indexer_k_proj;
    ds4_tensor *indexer_q_norm;
    ds4_tensor *lin_qkv;
    ds4_tensor *lin_gate;
    ds4_tensor *lin_conv;
    ds4_tensor *lin_dt_bias;
    ds4_tensor *lin_a;
    ds4_tensor *lin_beta;
    ds4_tensor *lin_alpha;
    ds4_tensor *lin_norm;
    ds4_tensor *lin_out;
    ds4_tensor *ple_key;
    ds4_tensor *ple_value;
    ds4_tensor *ple_norm_key;
    ds4_tensor *ple_norm_query;
    ds4_tensor *ple_norm_conv;
    ds4_tensor *ple_conv;
    ds4_tensor *ffn_gate_inp_shexp;
    ds4_tensor *nextn_hc_head_norm;
    ds4_tensor *nextn_hc_head_down;
    ds4_tensor *nextn_hc_head_up;
} ds4_layer_weights;

typedef struct {
    ds4_tensor *token_embd;
    ds4_tensor *output_hc_base;
    ds4_tensor *output_hc_fn;
    ds4_tensor *output_hc_scale;
    ds4_tensor *output_norm;
    ds4_tensor *output;
    ds4_tensor *output_hc_norm;
    ds4_tensor *output_hc_down;
    ds4_tensor *output_hc_up;
    ds4_tensor *ple_embd;
    ds4_layer_weights layer[DS4_MAX_LAYER];
} ds4_weights;

/* Bind and validate the known V4.1 profile using explicit validated config.
 * Full loads require embedding/output regardless of slice-only flags. A slice
 * may omit embedding after layer zero and may request an optional output head,
 * but a partially present head remains invalid. Failure reports the original
 * message without the caller's "ds4: " prefix; partial bindings may remain.
 */
bool ds4_native_weights_bind41(ds4_weights *weights, const ds4_model *model,
                               const ds4_model_config *config, bool load_slice,
                               uint32_t start, uint32_t end,
                               bool require_output, bool optional_output,
                               char *error, size_t error_capacity);

#endif
