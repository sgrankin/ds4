#ifndef DS4_ENGINE_CORE_H
#define DS4_ENGINE_CORE_H
#include "ds4.h"
#include "ds4_engine_memory.h"
#include "ds4_engine_configuration.h"
#include "ds4_model_weights.h"
#include "ds4_dspark_weights.h"
#include "ds4_support_weights.h"
#include "ds4_vocab.h"
#include "ds4_session_sampling.h"
#ifndef DS4_NO_GPU
#include "ds4_gpu.h"
#include "ds4_qwen4_vision.h"
#endif

/* Actual existing metadata/binding records, shared once with C fallback.
 * Support and vision tables borrow Model; their owners share these records. */
typedef enum {
    DS4_VISION_NONE = 0,
    DS4_VISION_GLM53,
    DS4_VISION_DEEPSEEK4,
    DS4_VISION_QWEN4,
} ds4_vision_kind;

/* The existing common Engine fields, embedded ONCE in Engine. Maps own the
 * borrowed tensor/vocabulary text, so bound tables/vocab retire before maps.
 * Other-family workspace graphs, placement and TP transport stay in Engine.
 * Sampling and Memory are their actual shared records, not mirrored layouts.
 * A native Core is a borrowed owner view over this sole storage location. */
typedef struct {
    char *model_path;
    ds4_engine_configuration configuration;
    ds4_engine_memory memory;
    ds4_model model;
    ds4_model mtp_model;
    ds4_model vision_model;
    ds4_vocab vocab;
    ds4_overthink_policy overthink;
    ds4_weights weights;
    ds4_mtp_weights mtp_weights;
    ds4_dspark_weights dspark_weights;
#ifndef DS4_NO_GPU
    ds4_glm53_vision_weights vision_weights;
    ds4_deepseek4_vision_weights deepseek4_vision_weights;
    ds4_qwen4_vision_weights qwen4_vision_weights;
#endif
    ds4_vision_kind vision_kind;
    int vision_image_token;
    int vision_start_token;
    int vision_end_token;
    ds4_backend backend;
    ds4_support_kind support_kind;
    int dspark_exec_tier;
    uint32_t support_stages;
    int mtp_draft_tokens;
    float mtp_margin;
    float dspark_confidence_threshold;
    bool dspark_confidence_threshold_set;
    char *directional_steering_file;
    float *directional_steering_dirs;
    float directional_steering_attn_scale;
    float directional_steering_ffn_scale;
    int power_percent;
    uint32_t prefill_chunk;
    ds4_ssd_memory_lock simulated_memory;
    bool quality;
    bool glm_mtp;
    bool glm_mtp_timing;
    bool dspark;
    bool dspark_strict;
    bool dspark_exact_sampling;
    bool cuda_tensor_parallel;
    bool glm_tp_token_prefill;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool ssd_streaming_full_layers_set;
    bool ssd_streaming_static_decode_map;
    ds4_distributed_options distributed;
    bool metal_ready;
    bool mtp_ready;
    bool vision_ready;
    bool vision_map_ready;
    bool share_session_prefill_workspace;
} ds4_engine_core;

bool ds4_native_core_finish41(ds4_engine_core *, const ds4_model_config *,
    const ds4_engine_options *, uint64_t *shard_model_bytes);
void ds4_native_core_free41(ds4_engine_core *);
#endif
