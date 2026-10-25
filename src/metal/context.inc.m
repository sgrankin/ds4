/* Actual ARC storage for one DS4 Metal owner. The native Core keeps the +1
 * handle; worker jobs borrow it only until their existing finish/join boundary. */
enum {
    DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS = 64,
    DS4_METAL_DECODE_PIPELINE_FAST_NAME_BYTES = 96,
};
/*
 * Sentinel nxpsg stored by the single-constant ds4_gpu_get_mul_mv_pipeline
 * path, which has no second SIMD-group constant. The extended
 * ds4_gpu_get_mul_mv_ext_pipeline path derives nxpsg as a positive divisor of
 * the 32-row output tile, so INT16_MIN can never collide with a real value.
 */
#define DS4_METAL_DECODE_PIPELINE_FAST_NXPSG_NONE INT16_MIN
_Static_assert(
    (DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS &
     (DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS - 1u)) == 0,
    "decode pipeline fast cache slot count must be a power of two");

typedef struct {
    id<MTLComputePipelineState> __strong pipeline;
    uint64_t hash;
    int16_t nsg;
    int16_t nxpsg;
    uint16_t name_len;
    bool used;
    char name[DS4_METAL_DECODE_PIPELINE_FAST_NAME_BYTES];
} ds4_gpu_decode_pipeline_fast_cache_entry;

typedef struct {
    id<MTLBuffer> __strong mask;
    id<MTLBuffer> __strong blk;
    NSUInteger mask_bytes;
    NSUInteger blk_bytes;
    uint32_t kind;
    uint32_t n_tokens;
    uint32_t n_comp;
    uint32_t n_keys;
    uint32_t window;
    uint32_t ratio;
    uint32_t nqptg;
    uint32_t ncpsg;
    bool has_kvpad;
    bool bc_mask;
    bool valid;
    bool blk_ready;
} ds4_gpu_zero_prefix_prefill_mask_cache_entry;

enum {
    QWEN4_K_HC_NORM_F16 = 0,
    QWEN4_K_HC_NORM_F32,
    QWEN4_K_HC_NORM_Q8,
    QWEN4_K_HC_NORM_REUSE_F16,
    QWEN4_K_HC_NORM_REUSE_F32,
    QWEN4_K_HC_NORM_REUSE_Q8,
    QWEN4_K_HC_GATE_MIX_F16,
    QWEN4_K_HC_GATE_MIX_F16_PF,
    QWEN4_K_HC_GATE_MIX_F32,
    QWEN4_K_HC_GATE_MIX_Q8,
    QWEN4_K_HC_GATE_MIX_PAIR_F16,
    QWEN4_K_HC_GATE_MIX_PAIR_F16_PF,
    QWEN4_K_HC_GATE_MIX_PAIR_F32,
    QWEN4_K_HC_GATE_MIX_PAIR_Q8,
    QWEN4_K_MULTI_GEMV,
    QWEN4_K_HC_COMBINE,
    QWEN4_K_CONV_STREAM,
    QWEN4_K_GDN_PREP,
    QWEN4_K_GDN_SCAN,
    QWEN4_K_GDN_SCAN_R4,
    QWEN4_K_GDN_OUT,
    QWEN4_K_PLE_GATE,
    QWEN4_K_PLE_CONV,
    QWEN4_K_ROUTER,
    QWEN4_K_ATTN_PREP,
    QWEN4_K_IDX_BLOCK_KEY,
    QWEN4_K_IDX_SCORE,
    QWEN4_K_IDX_SCORE_VEC,
    QWEN4_K_IDX_SELECT,
    QWEN4_K_IDX_SELECT_PRE,
    QWEN4_K_IDX_SCORE_MM,
    QWEN4_K_IDX_EXPAND,
    QWEN4_K_ATTN_DECODE_NPT8,
    QWEN4_K_ATTN_DECODE_NPT4,
    QWEN4_K_ATTN_DECODE_NPT1,
    QWEN4_K_ATTN_MERGE_NPT8,
    QWEN4_K_ATTN_MERGE_NPT4,
    QWEN4_K_ATTN_MERGE_NPT1,
    QWEN4_K_ATTN_MERGE_WIDE_NPT8,
    QWEN4_K_ATTN_MERGE_WIDE_NPT4,
    QWEN4_K_ATTN_MM,
    QWEN4_K_MOE_MID,
    QWEN4_K_MOE_MID_Q4K,
    QWEN4_K_MOE_MID_Q4K_NR1,
    QWEN4_K_MOE_DOWN,
    QWEN4_K_MOE_DOWN_MXFP4_PF,
    QWEN4_K_MOE_MID_Q4K_GROUPED,
    QWEN4_K_MOE_DOWN_MXFP4_GROUPED,
    QWEN4_K_MOE_REDUCE,
    QWEN4_K_HC_COMBINE_NORM,
    QWEN4_K_ARGMAX,
    QWEN4_K_MTP_STAGE,
    QWEN4_K_MTP_COMBINE,
    QWEN4_K_GDN_FRONT,
    QWEN4_K_MOE_BUILD_LISTS,
    QWEN4_K_MOE_MM_MID,
    QWEN4_K_MOE_MM_MID_NT1,
    QWEN4_K_MOE_MM_MID_NT2,
    QWEN4_K_MOE_MM_DOWN,
    QWEN4_K_MOE_MM_DOWN_NT1,
    QWEN4_K_MOE_MM_DOWN_NT2,
    QWEN4_K_MOE_MM_DOWN_NT8,
    QWEN4_K_MOE_MM_MID_NT8,
    QWEN4_K_MOE_MM_MID_NAX,
    QWEN4_K_MOE_MM_DOWN_NAX,
    QWEN4_K_MOE_MM_MID_NAX64,
    QWEN4_K_MOE_MM_DOWN_NAX64,
    QWEN4_K_MOE_MM_MID_NAXF,
    QWEN4_K_MOE_MM_DOWN_NAXF,
    QWEN4_K_MOE_MM_MID_NAXF64,
    QWEN4_K_MOE_MM_DOWN_NAXF64,
    QWEN4_K_MOE_MM_MID_NAXC,
    QWEN4_K_MOE_MM_DOWN_NAXC,
    QWEN4_K_MOE_MM_MID_NAXC64,
    QWEN4_K_MOE_MM_DOWN_NAXC64,
    QWEN4_K_ROWS_F32_TO_F16,
    QWEN4_K_DENSE_MM,
    QWEN4_K_DENSE_MM_REDUCE,
    QWEN4_K_BATCH_MM_Q8_T1,
    QWEN4_K_BATCH_MM_Q8_T2,
    QWEN4_K_GDN_SCAN_ROWS,
    QWEN4_K_CONV_STREAM_ROWS,
    QWEN4_K_GDN_SCAN_ROWS2,
    QWEN4_K_CONV_STREAM_ROWS2,
    QWEN4_K_ATTN_PREP_ROWS,
    QWEN4_K_IDX_BLOCK_KEY_ROWS,
    QWEN4_K_IDX_SCORE_ROWS,
    QWEN4_K_IDX_SELECT_ROWS,
    QWEN4_K_IDX_EXPAND_ROWS,
    QWEN4_K_ATTN_DECODE_ROWS_NPT8,
    QWEN4_K_ATTN_DECODE_ROWS_NPT4,
    QWEN4_K_ATTN_MERGE_ROWS_NPT8,
    QWEN4_K_ATTN_MERGE_ROWS_NPT4,
    QWEN4_K_HC_LO_ACT,
    QWEN4_K_HC_MIX_ROWS,
    QWEN4_K_VIS_PATCH_FINISH,
    QWEN4_K_VIS_LAYERNORM,
    QWEN4_K_VIS_QKV_ROPE,
    QWEN4_K_VIS_ATTENTION,
    QWEN4_K_VIS_BIAS_RESIDUAL,
    QWEN4_K_VIS_BIAS_ACT,
    QWEN4_K_COUNT,
};

@class DS4TimelineBatch;
typedef struct {
    ds4_metal_context *metal;
    ds4_host_context *host;
} ds4_metal_thread_owners;
static ds4_metal_thread_owners ds4_metal_thread_capture(void) {
    return (ds4_metal_thread_owners){ds4_metal_context_selected(), ds4_host_context_selected()};
}

@interface DS4MetalContext : NSObject {
@public
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    ds4_metal_scheduling schedule;
    ds4_metal_bootstrap_state bootstrap;
    ds4_metal_model_views model_views;
    ds4_metal_expert_cache_state expert_cache;
    ds4_metal_expert_load_state expert_load;
    int model_fd;
    int streaming;
    int glm;
    int quality;
    int prefill_building;
    int glm_building;
    pthread_t service_thread;
    int service_thread_set;
    bool runtime_owned;
    void *native_exact_cache;
    void *native_resources;
    void *native_diagnostics;
    ds4_metal_deferred_rope deferred_rope;
    ds4_metal_selected_override selected_override;
    /* Per-model pipelines, resource caches and reusable dispatch scratch. */
    id<MTLSharedEvent> selected_readback_event;
    ds4_metal_tp_policy tp_native_policy;
    id<MTLComputePipelineState> set_rows_f32_i32_pipeline;
    id<MTLComputePipelineState> get_rows_f32_pipeline;
    id<MTLComputePipelineState> get_rows_f16_pipeline;
    id<MTLComputePipelineState> get_rows_i32_pipeline;
    id<MTLComputePipelineState> get_rows_q8_0_pipeline;
    id<MTLComputePipelineState> get_rows_q4_0_pipeline;
    id<MTLComputePipelineState> get_rows_q4_K_pipeline;
    id<MTLComputePipelineState> repeat_f32_pipeline;
    id<MTLComputePipelineState> concat_pipeline;
    id<MTLComputePipelineState> cpy_f32_f32_pipeline;
    id<MTLComputePipelineState> cpy_f32_f16_pipeline;
    id<MTLComputePipelineState> cpy_contig_f32_f16_pipeline;
    id<MTLComputePipelineState> cpy_f16_f32_pipeline;
    id<MTLComputePipelineState> cpy_f16_f16_pipeline;
    id<MTLComputePipelineState> cpy_contig_f16_f32_pipeline;
    id<MTLComputePipelineState> cpy_contig_f16_f16_pipeline;
    id<MTLComputePipelineState> flash_kv_stage_f16_pipeline;
    id<MTLComputePipelineState> swiglu_pipeline;
    id<MTLComputePipelineState> swiglu_flat_pipeline;
    id<MTLComputePipelineState> add_pipeline;
    id<MTLComputePipelineState> add2_pipeline;
    id<MTLComputePipelineState> add3_pipeline;
    id<MTLComputePipelineState> moe_sum6_pipeline;
    id<MTLComputePipelineState> moe_sum8_pipeline;
    id<MTLComputePipelineState> mul_pipeline;
    id<MTLComputePipelineState> rms_norm_pipeline;
    id<MTLComputePipelineState> rms_norm_plain_pipeline;
    id<MTLComputePipelineState> add_rms_norm_pipeline;
    id<MTLComputePipelineState> rms_norm_scale_pipeline;
    id<MTLComputePipelineState> dsv4_qkv_rms_norm_pipeline;
    id<MTLComputePipelineState> dsv4_head_rms_norm_rope_tail_pipeline;
    id<MTLComputePipelineState> hc_split_sinkhorn_pipeline;
    id<MTLComputePipelineState> hc_split_weighted_sum_pipeline;
    id<MTLComputePipelineState> hc_split_weighted_sum_norm_pipeline;
    id<MTLComputePipelineState> dsv4_hc_producer_pre_norm_pipeline;
    id<MTLComputePipelineState> dsv4_hc_expand_producer_pre_norm_pipeline;
    NSMutableDictionary<NSString *, id<MTLBuffer>> *dsv4_hc_barrier_cache;
    NSMutableDictionary<NSString *, NSNumber *> *dsv4_hc_barrier_gen;
    id<MTLComputePipelineState> hc_weighted_sum_pipeline;
    id<MTLComputePipelineState> output_hc_weights4_pipeline;
    uint32_t test_flags;
    id<MTLComputePipelineState> hc_expand_pipeline;
    id<MTLComputePipelineState> unary_sigmoid_pipeline;
    id<MTLComputePipelineState> unary_silu_pipeline;
    id<MTLComputePipelineState> unary_softplus_pipeline;
    id<MTLComputePipelineState> unary_sqrt_pipeline;
    id<MTLComputePipelineState> unary_clamp_pipeline;
    id<MTLComputePipelineState> unary_scale_pipeline;
    id<MTLComputePipelineState> unary_fill_pipeline;
    id<MTLComputePipelineState> unary_fill_f16_pipeline;
    id<MTLComputePipelineState> bin_mul_scalar_pipeline;
    id<MTLComputePipelineState> bin_div_row_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_id_iq2_xxs_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_id_iq2_xxs_pair_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_id_iq2_xxs_pair_swiglu_pack2_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_id_q2_k_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_id_q2_k_sum6_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_id_iq2_xxs_sum6_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_id_q4_k_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_id_q4_k_pair_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_id_q4_k_pair_swiglu_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_id_q4_k_sum6_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_group_q4_k_pair_swiglu_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_group_q4_k_sum6_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_group6_q4_k_pair_swiglu_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_group6_q4_k_sum6_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_group8_q4_k_pair_swiglu_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_group8_q4_k_sum6_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_group24_q4_k_id_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_group24_q4_k_sum6_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_slots6_iq2_xxs_pair_swiglu_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_slots6_q2_k_sum6_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_slots6_q4_k_pair_swiglu_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_slots6_q4_k_sum6_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_id_mxfp4_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_id_mxfp4_pair_swiglu_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_id_mxfp4_sum6_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_id_mxfp4_pair_swiglu_pipeline_nsg1;
    id<MTLComputePipelineState> moe_mul_mv_id_mxfp4_sum6_pipeline_nsg1;
    id<MTLComputePipelineState> moe_mul_mv_id_mxfp4_pair_swiglu_pipeline_nsg1_tg_multiple;
    id<MTLComputePipelineState> moe_mul_mv_id_mxfp4_sum6_pipeline_nsg1_tg_multiple;
    id<MTLComputePipelineState> moe_mul_mv_id_mxfp4_pair_swiglu_fixed_route_pipeline_nsg1;
    id<MTLComputePipelineState> moe_mul_mv_id_mxfp4_sum6_fixed_route_pipeline_nsg1;
    id<MTLComputePipelineState> moe_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_pipeline_nsg1;
    id<MTLComputePipelineState> moe_mul_mv_id_mxfp4_pair_swiglu_fixed_route_static_pipeline_nsg1;
    id<MTLComputePipelineState> moe_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_static_pipeline_nsg1;
    id<MTLComputePipelineState> moe_mul_mv_id_mxfp4_pair_swiglu_tp_static_pipeline_nsg1;
    id<MTLComputePipelineState> moe_mul_mv_id_mxfp4_sum6_tp_full_rows_static_pipeline_nsg1;
    id<MTLComputePipelineState> moe_mul_mv_slots6_mxfp4_pair_swiglu_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_slots6_mxfp4_sum6_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_addr_iq2_xxs_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_addr_q2_k_sum6_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_addr_iq2_xxs_pair_swiglu_masked_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_addr_q2_k_sum6_masked_pipeline;
    id<MTLComputePipelineState> moe_stream_expert_cache_validate_pipeline;
    id<MTLComputePipelineState> moe_q4_gather_slots6_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_table_q4_k_pair_swiglu_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_table_q4_k_sum6_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_addr_q4_k_pair_swiglu_pipeline;
    id<MTLComputePipelineState> moe_mul_mv_addr_q4_k_sum6_pipeline;
    id<MTLArgumentEncoder> moe_table_q4_pair_gate_encoder;
    id<MTLArgumentEncoder> moe_table_q4_pair_up_encoder;
    id<MTLArgumentEncoder> moe_table_q4_sum_down_encoder;
    id<MTLComputePipelineState> rope_tail_batch_pipeline;
    id<MTLComputePipelineState> rope_tail_inplace_pair_pipeline;
    id<MTLComputePipelineState> rope_tail_inplace_pair_shared4_pipeline;
    id<MTLComputePipelineState> rope_tail_inplace_pair_affine_pipeline;
    id<MTLComputePipelineState> dsv4_fp8_kv_quantize_pipeline;
    id<MTLComputePipelineState> dsv4_indexer_qat_pipeline;
    id<MTLComputePipelineState> dsv4_kv_fp8_store_pipeline;
    id<MTLComputePipelineState> dsv4_kv_rope_fp8_store_pipeline;
    id<MTLComputePipelineState> dsv4_ratio4_shift_pipeline;
    id<MTLComputePipelineState> dsv4_compressor_pack_ratio4_pipeline;
    id<MTLComputePipelineState> dsv4_compressor_pack_ratio4_decode_ggml_pipeline;
    id<MTLComputePipelineState> dsv4_compressor_exact_softmax_product_pipeline;
    id<MTLComputePipelineState> dsv4_compressor_exact_pool_ratio4_pipeline;
    id<MTLComputePipelineState> dsv4_softmax_pool_ratio4_direct_pipeline;
    id<MTLComputePipelineState> dsv4_softmax_pool_pipeline;
    id<MTLComputePipelineState> soft_max_f32_pipeline;
    id<MTLComputePipelineState> soft_max_f32_4_pipeline;
    id<MTLComputePipelineState> argsort_f32_i32_desc_pipeline;
    id<MTLComputePipelineState> argsort_merge_f32_i32_desc_pipeline;
    id<MTLComputePipelineState> sum_rows_f32_f32_pipeline;
    id<MTLComputePipelineState> dsv4_topk_mask_pipeline;
    id<MTLComputePipelineState> dsv4_topk_mask_scatter_pipeline;
    id<MTLComputePipelineState> dsv4_indexer_weighted_sum_pipeline;
    id<MTLComputePipelineState> dsv4_indexer_score_one_direct_pipeline;
    id<MTLComputePipelineState> dsv4_compressor_store_one_pipeline;
    id<MTLComputePipelineState> dsv4_sort_i32_rows_asc_pipeline;
    id<MTLComputePipelineState> dsv4_indexed_attention_heads8_pipeline;
    id<MTLComputePipelineState> dsv4_indexed_attention_heads8_rb16_pipeline;
    id<MTLComputePipelineState> dsv4_indexed_attention_heads16_dual_pipeline;
    id<MTLComputePipelineState> dsv4_indexed_attention_heads8_split_pipeline;
    id<MTLComputePipelineState> dsv4_indexed_attention_heads8_split_reduce_pipeline;
    id<MTLComputePipelineState> dsv4_softplus_sqrt_pipeline;
    id<MTLComputePipelineState> dsv4_router_finalize_one_pipeline;
    id<MTLComputePipelineState> dsv4_router_finalize_one_simd_pipeline;
    id<MTLComputePipelineState> dsv4_router_finalize_weights_one_simd_pipeline;
    id<MTLComputePipelineState> dsv4_router_transform_finalize_weights_one_simd_pipeline;
    id<MTLComputePipelineState> dsv4_router_select_visual_batch_pipeline;
    NSCache<NSString *, id<MTLBuffer>> *dsv4_completion_cache;
    __weak id<MTLBuffer> dsv4_hc_producer_last_mix_buffer;
    NSUInteger dsv4_hc_producer_last_mix_offset;
    id<MTLBuffer> dsv4_hc_producer_last_completion;
    id<MTLComputePipelineState> dsv4_router_weights_one_pipeline;
    id<MTLComputePipelineState> glm_router_select_one_pipeline;
    id<MTLComputePipelineState> glm_kv_lora_rms_norm_pipeline;
    id<MTLComputePipelineState> glm_k_b_project_pipeline;
    id<MTLComputePipelineState> glm_store_compact_kv_pipeline;
    id<MTLComputePipelineState> glm_qkv_norm_store_compact_kv_pipeline;
    id<MTLComputePipelineState> glm_store_indexer_k_pipeline;
    id<MTLComputePipelineState> glm53_indexer_pool_update_pipeline;
    id<MTLComputePipelineState> glm_build_kv_cache_pipeline;
    id<MTLComputePipelineState> glm_build_kv_cache_decode_group4_pipeline;
    id<MTLComputePipelineState> glm_build_kv_cache_flash_pipeline;
    id<MTLComputePipelineState> glm_attention_full_pipeline;
    id<MTLComputePipelineState> glm_fill_selected_range_pipeline;
    id<MTLComputePipelineState> glm_fill_selected_range_batch_pipeline;
    id<MTLComputePipelineState> glm53_expand_pool_selection_pipeline;
    id<MTLComputePipelineState> glm_indexer_rope_tail_pipeline;
    id<MTLComputePipelineState> glm_indexer_score_one_pipeline;
    id<MTLComputePipelineState> glm_indexer_score_one_direct_pipeline;
    id<MTLComputePipelineState> glm_indexer_scores_batch_pipeline;
    id<MTLComputePipelineState> glm_indexer_scores_tiled_pipeline;
    id<MTLComputePipelineState> glm_indexer_scores_tiled_f32_pipeline;
    id<MTLComputePipelineState> glm_qk_lowrank_pipeline;
    id<MTLComputePipelineState> glm_qk_lowrank_glm52_pipeline;
    id<MTLComputePipelineState> glm_qk_lowrank_glm52_sg_pipeline;
    id<MTLComputePipelineState> glm_qk_lowrank_batch_pipeline;
    id<MTLComputePipelineState> glm_qk_lowrank_batch_glm52_t4_pipeline;
    id<MTLComputePipelineState> glm_value_project_q8_0_pipeline;
    id<MTLComputePipelineState> glm_value_project_q8_0_batch_heads_pipeline;
    id<MTLComputePipelineState> glm_value_project_q8_0_batch_heads_mma_pipeline;
    id<MTLComputePipelineState> glm_attention_indexed_decode_pipeline;
    id<MTLComputePipelineState> glm_attention_indexed_decode_split_group8_partial_pipeline;
    id<MTLComputePipelineState> glm_attention_indexed_decode_split_group8_partial_valid_fullheads_pipeline;
    id<MTLComputePipelineState> glm_attention_indexed_decode_split_group8_reduce_pipeline;
    id<MTLComputePipelineState> glm_attention_indexed_decode_split_group8_reduce16_pipeline;
    id<MTLComputePipelineState> glm_attention_indexed_batch_pipeline;
    id<MTLComputePipelineState> glm_attention_indexed_batch_group2_pipeline;
    id<MTLComputePipelineState> glm_attention_indexed_batch_q2_group4_pipeline;
    id<MTLComputePipelineState> glm_attention_indexed_batch_group8_pipeline;
    id<MTLComputePipelineState> glm_attention_indexed_batch_lora_group8_vec_pipeline;
    id<MTLComputePipelineState> glm_attention_indexed_batch_lora_group8_vec_valid_pipeline;
    id<MTLComputePipelineState> glm_attention_indexed_batch_lora_group8_vec_valid_fullheads_pipeline;
    id<MTLComputePipelineState> glm_attention_indexed_batch_lora_group8_vec_causal_pipeline;
    id<MTLComputePipelineState> glm_attention_indexed_batch_lora_group8_vec_causal_fullheads_pipeline;
    id<MTLComputePipelineState> glm_q4_k_pair_swiglu_f32_pipeline;
    id<MTLComputePipelineState> glm_q4_k_pair_swiglu2_f32_pipeline;
    id<MTLComputePipelineState> glm_q4_k_pair_swiglu4_f32_pipeline;
    id<MTLComputePipelineState> glm_q4_k_pair_swiglu2_mapped_f32_pipeline;
    id<MTLComputePipelineState> glm_q4_k_pair_swiglu2_mapped_row_f32_pipeline;
    id<MTLComputePipelineState> glm_q2_k_pair_swiglu_f32_pipeline;
    id<MTLComputePipelineState> glm_q2_k_addr_pair_swiglu2_f32_pipeline;
    id<MTLComputePipelineState> glm_q2_k_addr_pair_swiglu2_masked_f32_pipeline;
    id<MTLComputePipelineState> glm_q4_k_addr_pair_swiglu_f32_pipeline;
    id<MTLComputePipelineState> glm_q4_k_addr_pair_swiglu_masked_f32_pipeline;
    id<MTLComputePipelineState> glm_q2_k_down_f32_pipeline;
    id<MTLComputePipelineState> glm_q4_k_down_f32_pipeline;
    id<MTLComputePipelineState> glm_q2_k_addr_down_f32_pipeline;
    id<MTLComputePipelineState> glm_q4_k_addr_down_f32_pipeline;
    id<MTLComputePipelineState> glm_q5_k_pair_swiglu_f32_pipeline;
    id<MTLComputePipelineState> glm_q5_k_pair_swiglu_mapped_f32_pipeline;
    id<MTLComputePipelineState> glm_q5_k_pair_swiglu_mapped_row_f32_pipeline;
    id<MTLComputePipelineState> glm_q5_k_down_f32_pipeline;
    id<MTLComputePipelineState> glm_q6_k_down_f32_pipeline;
    id<MTLComputePipelineState> dsv4_router_weights_batch_pipeline;
    id<MTLComputePipelineState> dsv4_hc_expand4_pipeline;
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipeline_cache;
    ds4_gpu_decode_pipeline_fast_cache_entry
    decode_pipeline_fast_cache[DS4_METAL_DECODE_PIPELINE_FAST_CACHE_SLOTS];
    bool decode_pipeline_fast_lookup_active;
    uint32_t decode_pipeline_fast_cache_entries;
    NSMutableDictionary<NSString *, DS4MetalQ4ExpertTable *> *q4_expert_table_cache;
    NSMutableDictionary<NSString *, id> *q4_expert_layer_residency_cache;
    NSMutableArray<id<MTLBuffer>> *transient_buffers;
    ds4_gpu_zero_prefix_prefill_mask_cache_entry
    zero_prefix_prefill_mask_cache[DS4_GPU_PREFILL_MASK_CACHE_SLOTS];
    id<MTLBuffer> flash_attn_mask_buffer;
    id<MTLBuffer> flash_attn_zero_mask_buffer;
    id<MTLBuffer> flash_attn_pad_buffer;
    id<MTLBuffer> flash_attn_tmp_buffer;
    id<MTLBuffer> flash_attn_blk_buffer;
    id<MTLBuffer> flash_attn_ring_buffer;
    id<MTLBuffer> flash_attn_kv_buffer;
    id<MTLBuffer> glm_flash_attn_mask_buffer;
    id<MTLBuffer> compressor_pool_kv_buffer;
    id<MTLBuffer> compressor_pool_score_buffer;
    id<MTLBuffer> compressor_pool_score_cont_buffer;
    id<MTLBuffer> compressor_pool_softmax_buffer;
    id<MTLBuffer> compressor_pool_product_buffer;
    id<MTLBuffer> compressor_store_ape_buffer;
    id<MTLBuffer> compressor_store_score_buffer;
    id<MTLBuffer> embed_rows_buffer;
    id<MTLBuffer> router_selection_buffer;
    id<MTLBuffer> router_weight_sum_buffer;
    id<MTLBuffer> indexer_head_scores_buffer;
    id<MTLBuffer> indexer_topk_buffer;
    id<MTLBuffer> indexed_topk_buffer;
    id<MTLBuffer> f16_round_scratch_buffer;
    id<MTLBuffer> q8_prefill_scratch_buffer;
    id<MTLBuffer> qwen4_conv_halo_buffer;
    NSUInteger qwen4_conv_halo_bytes;
    id<MTLBuffer> raw_store_round_buffer;
    id<MTLBuffer> moe_gate_scratch_buffer;
    id<MTLBuffer> moe_down_scratch_buffer;
    id<MTLBuffer> moe_id_map_buffer;
    id<MTLBuffer> moe_packed_rhs_buffer;
    NSUInteger moe_packed_rhs_bytes;
    id<MTLBuffer> moe_q4_gate_slots_buffer;
    id<MTLBuffer> moe_q4_up_slots_buffer;
    id<MTLBuffer> moe_q4_down_slots_buffer;
    id<MTLBuffer> attn_out_group_ids_buffer;
    const ds4_gpu_tensor *stream_expert_batch_ready;
    int glm_streaming_prefill_full_layer_runtime;
    NSUInteger flash_attn_mask_bytes;
    NSUInteger flash_attn_zero_mask_bytes;
    NSUInteger flash_attn_pad_bytes;
    NSUInteger flash_attn_tmp_bytes;
    NSUInteger flash_attn_blk_bytes;
    NSUInteger flash_attn_ring_bytes;
    NSUInteger flash_attn_kv_bytes;
    NSUInteger glm_flash_attn_mask_bytes;
    uint32_t glm_flash_attn_mask_pos0;
    uint32_t glm_flash_attn_mask_tokens;
    uint32_t glm_flash_attn_mask_cache_len;
    int glm_flash_attn_mask_valid;
    NSUInteger compressor_pool_kv_bytes;
    NSUInteger compressor_pool_score_bytes;
    NSUInteger compressor_pool_score_cont_bytes;
    NSUInteger compressor_pool_softmax_bytes;
    NSUInteger compressor_pool_product_bytes;
    NSUInteger compressor_store_ape_bytes;
    NSUInteger compressor_store_score_bytes;
    NSUInteger embed_rows_bytes;
    NSUInteger router_selection_bytes;
    NSUInteger router_weight_sum_bytes;
    NSUInteger indexer_head_scores_bytes;
    NSUInteger indexer_topk_bytes;
    NSUInteger indexed_topk_bytes;
    NSUInteger f16_round_scratch_bytes;
    NSUInteger q8_prefill_scratch_bytes;
    NSUInteger raw_store_round_bytes;
    NSUInteger moe_gate_scratch_bytes;
    NSUInteger moe_down_scratch_bytes;
    NSUInteger moe_id_map_bytes;
    NSUInteger moe_q4_gate_slots_bytes;
    NSUInteger moe_q4_up_slots_bytes;
    NSUInteger moe_q4_down_slots_bytes;
    NSUInteger attn_out_group_ids_bytes;
    int mpp_invalid_env_reported;
    /* Model-specific Qwen/TP leaves share the same explicit owner. */
    DS4TimelineBatch *timeline_batch;
    double gpu_busy_accum;
    uint64_t gpu_busy_cbs;
    bool qwen4_verify_rows_exact;
    id<MTLSharedEvent> tp_gpu_event;
    id<MTLSharedEvent> tp_cpu_event;
    id<MTLSharedEvent> tp_batch_gpu_event;
    id<MTLSharedEvent> tp_batch_cpu_event;
    int32_t tp_split_rank;
    int32_t tp_split_world;
    int32_t tp_attn_head_split;
    id<MTLBuffer> tp_poll_buffer;
    volatile uint32_t *tp_poll_region;
    volatile uint32_t *tp_poll_status;
    int tp_poll_prev_valid;
    uint32_t tp_poll_prev_slot;
    uint64_t tp_poll_prev_seq;
    ds4_metal_tp_prefetch_policy tp_native_prefetch;
    id<MTLBuffer> tp_fold_ctl;
    id<MTLBuffer> tp_prefetch_scratch;
    uint64_t tp_stat_poll_lines;
    uint64_t tp_stat_poll_max_line;
    uint64_t tp_stat_poll_count;
    uint64_t tp_stat_encode_lead;
    double tp_stat_release_ms;
    double tp_stat_cbwait_ms;
    id<MTLBuffer> tp_check_buffer;
    volatile uint32_t *tp_check_words;
    volatile uint8_t *tp_slab_cpu;
    id<MTLBuffer> tp_slab_buffer;
    volatile uint32_t *tp_gpu_flags;
    ds4_gpu_tp_exchange_fn tp_exchange_fn;
    ds4_gpu_tp_batch_exchange_fn tp_batch_exchange_fn;
    ds4_gpu_tp_big_exchange_fn tp_big_exchange_fn;
    void *tp_exchange_ud;
    pthread_t tp_thread;
    int tp_thread_running;
    int tp_shutdown;
    int tp_failed_flag;
    pthread_mutex_t tp_mutex;
    pthread_cond_t tp_cond;
    ds4_gpu_tp_request tp_queue[DS4_GPU_TP_QUEUE];
    uint32_t tp_queue_head;
    uint32_t tp_queue_count;
    uint64_t tp_stat_gates;
    uint64_t tp_stat_batch_gates;
    double tp_stat_batch_gpu_wait_ms;
    double tp_stat_batch_exchange_ms;
    double tp_stat_batch_release_ms;
    double tp_stat_gpu_wait_ms;
    double tp_stat_exchange_ms;
    uint64_t tp_stat_gate_count[DS4_GPU_TP_GATES_PER_LAYER];
    double tp_stat_gate_gpu_wait_ms[DS4_GPU_TP_GATES_PER_LAYER];
    double tp_stat_gate_exchange_ms[DS4_GPU_TP_GATES_PER_LAYER];
    id<MTLCommandQueue> tp_keepalive_queue;
    id<MTLBuffer> tp_keepalive_buffer;
    pthread_t tp_keepalive_thread;
    int tp_keepalive_running;
    volatile int tp_keepalive_paused;
    id<MTLComputePipelineState> qwen4_pipelines[QWEN4_K_COUNT];
    ds4_gpu_tensor *qwen4_gdn_slots;
    float qwen4_rope_freq[32];
    float qwen4_rope_mscale;
    bool qwen4_rope_set;
    ds4_gpu_tensor *qwen4_nax_half_x, *qwen4_nax_half_mid, *qwen4_nax_half_midr;
    uint64_t qwen4_nax_half_x_bytes, qwen4_nax_half_mid_bytes, qwen4_nax_half_midr_bytes;
    const ds4_gpu_tensor *qwen4_nax_half_mid_for;
    uint64_t qwen4_nax_half_mid_count;
    ds4_gpu_tensor *qwen4_dense_mm_partials;
    uint64_t qwen4_dense_mm_partials_bytes;
    ds4_metal_thread_owners queue_thread_owners, tp_thread_owners, tp_keepalive_owners;
    pthread_t keepalive_thread;
    volatile int keepalive_running;
    volatile int keepalive_stop;
}
@end
@implementation DS4MetalContext
- (instancetype)init {
    self = [super init];
    if (self) {
        model_fd = -1;
        tp_split_world = 1;
        pthread_mutex_init(&tp_mutex, NULL);
        pthread_cond_init(&tp_cond, NULL);
        qwen4_rope_mscale = 1.0f;
        qwen4_rope_set = false;
        expert_cache.cache_first_decode_layer = UINT32_MAX;
        pthread_mutex_init(&expert_load.pool.mutex, NULL);
        pthread_cond_init(&expert_load.pool.start_cond, NULL);
        pthread_cond_init(&expert_load.pool.done_cond, NULL);
    }
    return self;
}
- (void)dealloc {
    // Retirement must finish before the Core drops its last owner reference.
    assert(!bootstrap.initialized && !expert_load.pool.initialized && !runtime_owned);
    assert(!native_resources && !native_exact_cache && !native_diagnostics);
    assert(!tp_thread_running && !tp_keepalive_running);
    pthread_cond_destroy(&tp_cond);
    pthread_mutex_destroy(&tp_mutex);
    pthread_cond_destroy(&expert_load.pool.done_cond);
    pthread_cond_destroy(&expert_load.pool.start_cond);
    pthread_mutex_destroy(&expert_load.pool.mutex);
}
@end

static _Thread_local __unsafe_unretained DS4MetalContext *selected_context;
static DS4MetalContext *legacy_context;
static pthread_once_t legacy_context_once = PTHREAD_ONCE_INIT;
static void ds4_metal_legacy_context_init(void) {
    legacy_context = [DS4MetalContext new];
    if (!legacy_context) abort();
}
static inline DS4MetalContext *ds4_metal_context_current(void) {
    if (selected_context) return selected_context;
    // Retained C/diagnostic callers still have one process-owned context.
    pthread_once(&legacy_context_once, ds4_metal_legacy_context_init);
    return legacy_context;
}
ds4_metal_context *ds4_metal_context_create(void) {
    return (ds4_metal_context *)CFBridgingRetain([DS4MetalContext new]);
}
void ds4_metal_context_destroy(ds4_metal_context *opaque) {
    if (selected_context == (__bridge DS4MetalContext *)opaque) selected_context = nil;
    if (opaque) { id released = CFBridgingRelease(opaque); (void)released; }
}
ds4_metal_context *ds4_metal_context_select(ds4_metal_context *opaque) {
    DS4MetalContext *previous = selected_context;
    selected_context = (__bridge DS4MetalContext *)opaque;
    return (__bridge ds4_metal_context *)previous;
}
ds4_metal_context *ds4_metal_context_selected(void) {
    return (__bridge ds4_metal_context *)selected_context;
}

typedef struct {
    ds4_metal_context *previous;
    ds4_host_context *previous_host;
} ds4_metal_thread_scope;
static ds4_metal_thread_scope ds4_metal_thread_enter(const ds4_metal_thread_owners *owners) {
    return (ds4_metal_thread_scope){ds4_metal_context_select(owners->metal), ds4_host_context_select(owners->host)};
}
static void ds4_metal_thread_leave(ds4_metal_thread_scope *scope) {
    ds4_host_context_select(scope->previous_host);
    ds4_metal_context_select(scope->previous);
}
#define DS4_METAL_THREAD_SCOPE(owners) \
    ds4_metal_thread_scope metal_thread_scope __attribute__((cleanup(ds4_metal_thread_leave))) = \
        ds4_metal_thread_enter((const ds4_metal_thread_owners *)(owners))

#define g_device (ds4_metal_context_current()->device)
#define g_queue (ds4_metal_context_current()->queue)
#define g_library (ds4_metal_context_current()->library)
#define g_schedule (ds4_metal_context_current()->schedule)
#define g_bootstrap (ds4_metal_context_current()->bootstrap)
#define g_model_views (ds4_metal_context_current()->model_views)
#define g_expert_cache (ds4_metal_context_current()->expert_cache)
#define g_expert_load (ds4_metal_context_current()->expert_load)
#define g_model_fd (ds4_metal_context_current()->model_fd)
#define g_ssd_streaming_mode (ds4_metal_context_current()->streaming)
#define g_glm_model_mode (ds4_metal_context_current()->glm)
#define g_quality_mode (ds4_metal_context_current()->quality)
#define g_stream_prefill_batch_selected_addr_building (ds4_metal_context_current()->prefill_building)
#define g_glm_stream_expert_addr_table_building (ds4_metal_context_current()->glm_building)
#define g_stream_expert_service_thread (ds4_metal_context_current()->service_thread)
#define g_stream_expert_service_thread_set (ds4_metal_context_current()->service_thread_set)
#define g_runtime_owned (ds4_metal_context_current()->runtime_owned)
#define g_queue_keepalive_thread (ds4_metal_context_current()->keepalive_thread)
#define g_queue_keepalive_running (ds4_metal_context_current()->keepalive_running)
#define g_queue_keepalive_stop (ds4_metal_context_current()->keepalive_stop)

/* Compatibility names select fields in the actual engine owner. */
#define g_selected_readback_event (ds4_metal_context_current()->selected_readback_event)
#define g_tp_native_policy (ds4_metal_context_current()->tp_native_policy)
#define g_set_rows_f32_i32_pipeline (ds4_metal_context_current()->set_rows_f32_i32_pipeline)
#define g_get_rows_f32_pipeline (ds4_metal_context_current()->get_rows_f32_pipeline)
#define g_get_rows_f16_pipeline (ds4_metal_context_current()->get_rows_f16_pipeline)
#define g_get_rows_i32_pipeline (ds4_metal_context_current()->get_rows_i32_pipeline)
#define g_get_rows_q8_0_pipeline (ds4_metal_context_current()->get_rows_q8_0_pipeline)
#define g_get_rows_q4_0_pipeline (ds4_metal_context_current()->get_rows_q4_0_pipeline)
#define g_get_rows_q4_K_pipeline (ds4_metal_context_current()->get_rows_q4_K_pipeline)
#define g_repeat_f32_pipeline (ds4_metal_context_current()->repeat_f32_pipeline)
#define g_concat_pipeline (ds4_metal_context_current()->concat_pipeline)
#define g_cpy_f32_f32_pipeline (ds4_metal_context_current()->cpy_f32_f32_pipeline)
#define g_cpy_f32_f16_pipeline (ds4_metal_context_current()->cpy_f32_f16_pipeline)
#define g_cpy_contig_f32_f16_pipeline (ds4_metal_context_current()->cpy_contig_f32_f16_pipeline)
#define g_cpy_f16_f32_pipeline (ds4_metal_context_current()->cpy_f16_f32_pipeline)
#define g_cpy_f16_f16_pipeline (ds4_metal_context_current()->cpy_f16_f16_pipeline)
#define g_cpy_contig_f16_f32_pipeline (ds4_metal_context_current()->cpy_contig_f16_f32_pipeline)
#define g_cpy_contig_f16_f16_pipeline (ds4_metal_context_current()->cpy_contig_f16_f16_pipeline)
#define g_flash_kv_stage_f16_pipeline (ds4_metal_context_current()->flash_kv_stage_f16_pipeline)
#define g_swiglu_pipeline (ds4_metal_context_current()->swiglu_pipeline)
#define g_swiglu_flat_pipeline (ds4_metal_context_current()->swiglu_flat_pipeline)
#define g_add_pipeline (ds4_metal_context_current()->add_pipeline)
#define g_add2_pipeline (ds4_metal_context_current()->add2_pipeline)
#define g_add3_pipeline (ds4_metal_context_current()->add3_pipeline)
#define g_moe_sum6_pipeline (ds4_metal_context_current()->moe_sum6_pipeline)
#define g_moe_sum8_pipeline (ds4_metal_context_current()->moe_sum8_pipeline)
#define g_mul_pipeline (ds4_metal_context_current()->mul_pipeline)
#define g_rms_norm_pipeline (ds4_metal_context_current()->rms_norm_pipeline)
#define g_rms_norm_plain_pipeline (ds4_metal_context_current()->rms_norm_plain_pipeline)
#define g_add_rms_norm_pipeline (ds4_metal_context_current()->add_rms_norm_pipeline)
#define g_rms_norm_scale_pipeline (ds4_metal_context_current()->rms_norm_scale_pipeline)
#define g_dsv4_qkv_rms_norm_pipeline (ds4_metal_context_current()->dsv4_qkv_rms_norm_pipeline)
#define g_dsv4_head_rms_norm_rope_tail_pipeline (ds4_metal_context_current()->dsv4_head_rms_norm_rope_tail_pipeline)
#define g_hc_split_sinkhorn_pipeline (ds4_metal_context_current()->hc_split_sinkhorn_pipeline)
#define g_hc_split_weighted_sum_pipeline (ds4_metal_context_current()->hc_split_weighted_sum_pipeline)
#define g_hc_split_weighted_sum_norm_pipeline (ds4_metal_context_current()->hc_split_weighted_sum_norm_pipeline)
#define g_dsv4_hc_producer_pre_norm_pipeline (ds4_metal_context_current()->dsv4_hc_producer_pre_norm_pipeline)
#define g_dsv4_hc_expand_producer_pre_norm_pipeline (ds4_metal_context_current()->dsv4_hc_expand_producer_pre_norm_pipeline)
#define g_dsv4_hc_barrier_cache (ds4_metal_context_current()->dsv4_hc_barrier_cache)
#define g_dsv4_hc_barrier_gen (ds4_metal_context_current()->dsv4_hc_barrier_gen)
#define g_hc_weighted_sum_pipeline (ds4_metal_context_current()->hc_weighted_sum_pipeline)
#define g_output_hc_weights4_pipeline (ds4_metal_context_current()->output_hc_weights4_pipeline)
#define g_test_flags (ds4_metal_context_current()->test_flags)
#define g_hc_expand_pipeline (ds4_metal_context_current()->hc_expand_pipeline)
#define g_unary_sigmoid_pipeline (ds4_metal_context_current()->unary_sigmoid_pipeline)
#define g_unary_silu_pipeline (ds4_metal_context_current()->unary_silu_pipeline)
#define g_unary_softplus_pipeline (ds4_metal_context_current()->unary_softplus_pipeline)
#define g_unary_sqrt_pipeline (ds4_metal_context_current()->unary_sqrt_pipeline)
#define g_unary_clamp_pipeline (ds4_metal_context_current()->unary_clamp_pipeline)
#define g_unary_scale_pipeline (ds4_metal_context_current()->unary_scale_pipeline)
#define g_unary_fill_pipeline (ds4_metal_context_current()->unary_fill_pipeline)
#define g_unary_fill_f16_pipeline (ds4_metal_context_current()->unary_fill_f16_pipeline)
#define g_bin_mul_scalar_pipeline (ds4_metal_context_current()->bin_mul_scalar_pipeline)
#define g_bin_div_row_pipeline (ds4_metal_context_current()->bin_div_row_pipeline)
#define g_moe_mul_mv_id_iq2_xxs_pipeline (ds4_metal_context_current()->moe_mul_mv_id_iq2_xxs_pipeline)
#define g_moe_mul_mv_id_iq2_xxs_pair_pipeline (ds4_metal_context_current()->moe_mul_mv_id_iq2_xxs_pair_pipeline)
#define g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline (ds4_metal_context_current()->moe_mul_mv_id_iq2_xxs_pair_swiglu_pipeline)
#define g_moe_mul_mv_id_iq2_xxs_pair_swiglu_pack2_pipeline (ds4_metal_context_current()->moe_mul_mv_id_iq2_xxs_pair_swiglu_pack2_pipeline)
#define g_moe_mul_mv_id_q2_k_pipeline (ds4_metal_context_current()->moe_mul_mv_id_q2_k_pipeline)
#define g_moe_mul_mv_id_q2_k_sum6_pipeline (ds4_metal_context_current()->moe_mul_mv_id_q2_k_sum6_pipeline)
#define g_moe_mul_mv_id_iq2_xxs_sum6_pipeline (ds4_metal_context_current()->moe_mul_mv_id_iq2_xxs_sum6_pipeline)
#define g_moe_mul_mv_id_q4_k_pipeline (ds4_metal_context_current()->moe_mul_mv_id_q4_k_pipeline)
#define g_moe_mul_mv_id_q4_k_pair_pipeline (ds4_metal_context_current()->moe_mul_mv_id_q4_k_pair_pipeline)
#define g_moe_mul_mv_id_q4_k_pair_swiglu_pipeline (ds4_metal_context_current()->moe_mul_mv_id_q4_k_pair_swiglu_pipeline)
#define g_moe_mul_mv_id_q4_k_sum6_pipeline (ds4_metal_context_current()->moe_mul_mv_id_q4_k_sum6_pipeline)
#define g_moe_mul_mv_group_q4_k_pair_swiglu_pipeline (ds4_metal_context_current()->moe_mul_mv_group_q4_k_pair_swiglu_pipeline)
#define g_moe_mul_mv_group_q4_k_sum6_pipeline (ds4_metal_context_current()->moe_mul_mv_group_q4_k_sum6_pipeline)
#define g_moe_mul_mv_group6_q4_k_pair_swiglu_pipeline (ds4_metal_context_current()->moe_mul_mv_group6_q4_k_pair_swiglu_pipeline)
#define g_moe_mul_mv_group6_q4_k_sum6_pipeline (ds4_metal_context_current()->moe_mul_mv_group6_q4_k_sum6_pipeline)
#define g_moe_mul_mv_group8_q4_k_pair_swiglu_pipeline (ds4_metal_context_current()->moe_mul_mv_group8_q4_k_pair_swiglu_pipeline)
#define g_moe_mul_mv_group8_q4_k_sum6_pipeline (ds4_metal_context_current()->moe_mul_mv_group8_q4_k_sum6_pipeline)
#define g_moe_mul_mv_group24_q4_k_id_pipeline (ds4_metal_context_current()->moe_mul_mv_group24_q4_k_id_pipeline)
#define g_moe_mul_mv_group24_q4_k_sum6_pipeline (ds4_metal_context_current()->moe_mul_mv_group24_q4_k_sum6_pipeline)
#define g_moe_mul_mv_slots6_iq2_xxs_pair_swiglu_pipeline (ds4_metal_context_current()->moe_mul_mv_slots6_iq2_xxs_pair_swiglu_pipeline)
#define g_moe_mul_mv_slots6_q2_k_sum6_pipeline (ds4_metal_context_current()->moe_mul_mv_slots6_q2_k_sum6_pipeline)
#define g_moe_mul_mv_slots6_q4_k_pair_swiglu_pipeline (ds4_metal_context_current()->moe_mul_mv_slots6_q4_k_pair_swiglu_pipeline)
#define g_moe_mul_mv_slots6_q4_k_sum6_pipeline (ds4_metal_context_current()->moe_mul_mv_slots6_q4_k_sum6_pipeline)
#define g_moe_mul_mv_id_mxfp4_pipeline (ds4_metal_context_current()->moe_mul_mv_id_mxfp4_pipeline)
#define g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline (ds4_metal_context_current()->moe_mul_mv_id_mxfp4_pair_swiglu_pipeline)
#define g_moe_mul_mv_id_mxfp4_sum6_pipeline (ds4_metal_context_current()->moe_mul_mv_id_mxfp4_sum6_pipeline)
#define g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline_nsg1 (ds4_metal_context_current()->moe_mul_mv_id_mxfp4_pair_swiglu_pipeline_nsg1)
#define g_moe_mul_mv_id_mxfp4_sum6_pipeline_nsg1 (ds4_metal_context_current()->moe_mul_mv_id_mxfp4_sum6_pipeline_nsg1)
#define g_moe_mul_mv_id_mxfp4_pair_swiglu_pipeline_nsg1_tg_multiple (ds4_metal_context_current()->moe_mul_mv_id_mxfp4_pair_swiglu_pipeline_nsg1_tg_multiple)
#define g_moe_mul_mv_id_mxfp4_sum6_pipeline_nsg1_tg_multiple (ds4_metal_context_current()->moe_mul_mv_id_mxfp4_sum6_pipeline_nsg1_tg_multiple)
#define g_moe_mul_mv_id_mxfp4_pair_swiglu_fixed_route_pipeline_nsg1 (ds4_metal_context_current()->moe_mul_mv_id_mxfp4_pair_swiglu_fixed_route_pipeline_nsg1)
#define g_moe_mul_mv_id_mxfp4_sum6_fixed_route_pipeline_nsg1 (ds4_metal_context_current()->moe_mul_mv_id_mxfp4_sum6_fixed_route_pipeline_nsg1)
#define g_moe_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_pipeline_nsg1 (ds4_metal_context_current()->moe_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_pipeline_nsg1)
#define g_moe_mul_mv_id_mxfp4_pair_swiglu_fixed_route_static_pipeline_nsg1 (ds4_metal_context_current()->moe_mul_mv_id_mxfp4_pair_swiglu_fixed_route_static_pipeline_nsg1)
#define g_moe_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_static_pipeline_nsg1 (ds4_metal_context_current()->moe_mul_mv_id_mxfp4_sum6_fixed_route_full_rows_static_pipeline_nsg1)
#define g_moe_mul_mv_id_mxfp4_pair_swiglu_tp_static_pipeline_nsg1 (ds4_metal_context_current()->moe_mul_mv_id_mxfp4_pair_swiglu_tp_static_pipeline_nsg1)
#define g_moe_mul_mv_id_mxfp4_sum6_tp_full_rows_static_pipeline_nsg1 (ds4_metal_context_current()->moe_mul_mv_id_mxfp4_sum6_tp_full_rows_static_pipeline_nsg1)
#define g_moe_mul_mv_slots6_mxfp4_pair_swiglu_pipeline (ds4_metal_context_current()->moe_mul_mv_slots6_mxfp4_pair_swiglu_pipeline)
#define g_moe_mul_mv_slots6_mxfp4_sum6_pipeline (ds4_metal_context_current()->moe_mul_mv_slots6_mxfp4_sum6_pipeline)
#define g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline (ds4_metal_context_current()->moe_mul_mv_addr_iq2_xxs_pair_swiglu_pipeline)
#define g_moe_mul_mv_addr_iq2_xxs_pipeline (ds4_metal_context_current()->moe_mul_mv_addr_iq2_xxs_pipeline)
#define g_moe_mul_mv_addr_q2_k_sum6_pipeline (ds4_metal_context_current()->moe_mul_mv_addr_q2_k_sum6_pipeline)
#define g_moe_mul_mv_addr_iq2_xxs_pair_swiglu_masked_pipeline (ds4_metal_context_current()->moe_mul_mv_addr_iq2_xxs_pair_swiglu_masked_pipeline)
#define g_moe_mul_mv_addr_q2_k_sum6_masked_pipeline (ds4_metal_context_current()->moe_mul_mv_addr_q2_k_sum6_masked_pipeline)
#define g_moe_stream_expert_cache_validate_pipeline (ds4_metal_context_current()->moe_stream_expert_cache_validate_pipeline)
#define g_moe_q4_gather_slots6_pipeline (ds4_metal_context_current()->moe_q4_gather_slots6_pipeline)
#define g_moe_mul_mv_table_q4_k_pair_swiglu_pipeline (ds4_metal_context_current()->moe_mul_mv_table_q4_k_pair_swiglu_pipeline)
#define g_moe_mul_mv_table_q4_k_sum6_pipeline (ds4_metal_context_current()->moe_mul_mv_table_q4_k_sum6_pipeline)
#define g_moe_mul_mv_addr_q4_k_pair_swiglu_pipeline (ds4_metal_context_current()->moe_mul_mv_addr_q4_k_pair_swiglu_pipeline)
#define g_moe_mul_mv_addr_q4_k_sum6_pipeline (ds4_metal_context_current()->moe_mul_mv_addr_q4_k_sum6_pipeline)
#define g_moe_table_q4_pair_gate_encoder (ds4_metal_context_current()->moe_table_q4_pair_gate_encoder)
#define g_moe_table_q4_pair_up_encoder (ds4_metal_context_current()->moe_table_q4_pair_up_encoder)
#define g_moe_table_q4_sum_down_encoder (ds4_metal_context_current()->moe_table_q4_sum_down_encoder)
#define g_rope_tail_batch_pipeline (ds4_metal_context_current()->rope_tail_batch_pipeline)
#define g_rope_tail_inplace_pair_pipeline (ds4_metal_context_current()->rope_tail_inplace_pair_pipeline)
#define g_rope_tail_inplace_pair_shared4_pipeline (ds4_metal_context_current()->rope_tail_inplace_pair_shared4_pipeline)
#define g_rope_tail_inplace_pair_affine_pipeline (ds4_metal_context_current()->rope_tail_inplace_pair_affine_pipeline)
#define g_dsv4_fp8_kv_quantize_pipeline (ds4_metal_context_current()->dsv4_fp8_kv_quantize_pipeline)
#define g_dsv4_indexer_qat_pipeline (ds4_metal_context_current()->dsv4_indexer_qat_pipeline)
#define g_dsv4_kv_fp8_store_pipeline (ds4_metal_context_current()->dsv4_kv_fp8_store_pipeline)
#define g_dsv4_kv_rope_fp8_store_pipeline (ds4_metal_context_current()->dsv4_kv_rope_fp8_store_pipeline)
#define g_dsv4_ratio4_shift_pipeline (ds4_metal_context_current()->dsv4_ratio4_shift_pipeline)
#define g_dsv4_compressor_pack_ratio4_pipeline (ds4_metal_context_current()->dsv4_compressor_pack_ratio4_pipeline)
#define g_dsv4_compressor_pack_ratio4_decode_ggml_pipeline (ds4_metal_context_current()->dsv4_compressor_pack_ratio4_decode_ggml_pipeline)
#define g_dsv4_compressor_exact_softmax_product_pipeline (ds4_metal_context_current()->dsv4_compressor_exact_softmax_product_pipeline)
#define g_dsv4_compressor_exact_pool_ratio4_pipeline (ds4_metal_context_current()->dsv4_compressor_exact_pool_ratio4_pipeline)
#define g_dsv4_softmax_pool_ratio4_direct_pipeline (ds4_metal_context_current()->dsv4_softmax_pool_ratio4_direct_pipeline)
#define g_dsv4_softmax_pool_pipeline (ds4_metal_context_current()->dsv4_softmax_pool_pipeline)
#define g_soft_max_f32_pipeline (ds4_metal_context_current()->soft_max_f32_pipeline)
#define g_soft_max_f32_4_pipeline (ds4_metal_context_current()->soft_max_f32_4_pipeline)
#define g_argsort_f32_i32_desc_pipeline (ds4_metal_context_current()->argsort_f32_i32_desc_pipeline)
#define g_argsort_merge_f32_i32_desc_pipeline (ds4_metal_context_current()->argsort_merge_f32_i32_desc_pipeline)
#define g_sum_rows_f32_f32_pipeline (ds4_metal_context_current()->sum_rows_f32_f32_pipeline)
#define g_dsv4_topk_mask_pipeline (ds4_metal_context_current()->dsv4_topk_mask_pipeline)
#define g_dsv4_topk_mask_scatter_pipeline (ds4_metal_context_current()->dsv4_topk_mask_scatter_pipeline)
#define g_dsv4_indexer_weighted_sum_pipeline (ds4_metal_context_current()->dsv4_indexer_weighted_sum_pipeline)
#define g_dsv4_indexer_score_one_direct_pipeline (ds4_metal_context_current()->dsv4_indexer_score_one_direct_pipeline)
#define g_dsv4_compressor_store_one_pipeline (ds4_metal_context_current()->dsv4_compressor_store_one_pipeline)
#define g_dsv4_sort_i32_rows_asc_pipeline (ds4_metal_context_current()->dsv4_sort_i32_rows_asc_pipeline)
#define g_dsv4_indexed_attention_heads8_pipeline (ds4_metal_context_current()->dsv4_indexed_attention_heads8_pipeline)
#define g_dsv4_indexed_attention_heads8_rb16_pipeline (ds4_metal_context_current()->dsv4_indexed_attention_heads8_rb16_pipeline)
#define g_dsv4_indexed_attention_heads16_dual_pipeline (ds4_metal_context_current()->dsv4_indexed_attention_heads16_dual_pipeline)
#define g_dsv4_indexed_attention_heads8_split_pipeline (ds4_metal_context_current()->dsv4_indexed_attention_heads8_split_pipeline)
#define g_dsv4_indexed_attention_heads8_split_reduce_pipeline (ds4_metal_context_current()->dsv4_indexed_attention_heads8_split_reduce_pipeline)
#define g_dsv4_softplus_sqrt_pipeline (ds4_metal_context_current()->dsv4_softplus_sqrt_pipeline)
#define g_dsv4_router_finalize_one_pipeline (ds4_metal_context_current()->dsv4_router_finalize_one_pipeline)
#define g_dsv4_router_finalize_one_simd_pipeline (ds4_metal_context_current()->dsv4_router_finalize_one_simd_pipeline)
#define g_dsv4_router_finalize_weights_one_simd_pipeline (ds4_metal_context_current()->dsv4_router_finalize_weights_one_simd_pipeline)
#define g_dsv4_router_transform_finalize_weights_one_simd_pipeline (ds4_metal_context_current()->dsv4_router_transform_finalize_weights_one_simd_pipeline)
#define g_dsv4_router_select_visual_batch_pipeline (ds4_metal_context_current()->dsv4_router_select_visual_batch_pipeline)
#define g_dsv4_completion_cache (ds4_metal_context_current()->dsv4_completion_cache)
#define g_dsv4_hc_producer_last_mix_buffer (ds4_metal_context_current()->dsv4_hc_producer_last_mix_buffer)
#define g_dsv4_hc_producer_last_mix_offset (ds4_metal_context_current()->dsv4_hc_producer_last_mix_offset)
#define g_dsv4_hc_producer_last_completion (ds4_metal_context_current()->dsv4_hc_producer_last_completion)
#define g_dsv4_router_weights_one_pipeline (ds4_metal_context_current()->dsv4_router_weights_one_pipeline)
#define g_glm_router_select_one_pipeline (ds4_metal_context_current()->glm_router_select_one_pipeline)
#define g_glm_kv_lora_rms_norm_pipeline (ds4_metal_context_current()->glm_kv_lora_rms_norm_pipeline)
#define g_glm_k_b_project_pipeline (ds4_metal_context_current()->glm_k_b_project_pipeline)
#define g_glm_store_compact_kv_pipeline (ds4_metal_context_current()->glm_store_compact_kv_pipeline)
#define g_glm_qkv_norm_store_compact_kv_pipeline (ds4_metal_context_current()->glm_qkv_norm_store_compact_kv_pipeline)
#define g_glm_store_indexer_k_pipeline (ds4_metal_context_current()->glm_store_indexer_k_pipeline)
#define g_glm53_indexer_pool_update_pipeline (ds4_metal_context_current()->glm53_indexer_pool_update_pipeline)
#define g_glm_build_kv_cache_pipeline (ds4_metal_context_current()->glm_build_kv_cache_pipeline)
#define g_glm_build_kv_cache_decode_group4_pipeline (ds4_metal_context_current()->glm_build_kv_cache_decode_group4_pipeline)
#define g_glm_build_kv_cache_flash_pipeline (ds4_metal_context_current()->glm_build_kv_cache_flash_pipeline)
#define g_glm_attention_full_pipeline (ds4_metal_context_current()->glm_attention_full_pipeline)
#define g_glm_fill_selected_range_pipeline (ds4_metal_context_current()->glm_fill_selected_range_pipeline)
#define g_glm_fill_selected_range_batch_pipeline (ds4_metal_context_current()->glm_fill_selected_range_batch_pipeline)
#define g_glm53_expand_pool_selection_pipeline (ds4_metal_context_current()->glm53_expand_pool_selection_pipeline)
#define g_glm_indexer_rope_tail_pipeline (ds4_metal_context_current()->glm_indexer_rope_tail_pipeline)
#define g_glm_indexer_score_one_pipeline (ds4_metal_context_current()->glm_indexer_score_one_pipeline)
#define g_glm_indexer_score_one_direct_pipeline (ds4_metal_context_current()->glm_indexer_score_one_direct_pipeline)
#define g_glm_indexer_scores_batch_pipeline (ds4_metal_context_current()->glm_indexer_scores_batch_pipeline)
#define g_glm_indexer_scores_tiled_pipeline (ds4_metal_context_current()->glm_indexer_scores_tiled_pipeline)
#define g_glm_indexer_scores_tiled_f32_pipeline (ds4_metal_context_current()->glm_indexer_scores_tiled_f32_pipeline)
#define g_glm_qk_lowrank_pipeline (ds4_metal_context_current()->glm_qk_lowrank_pipeline)
#define g_glm_qk_lowrank_glm52_pipeline (ds4_metal_context_current()->glm_qk_lowrank_glm52_pipeline)
#define g_glm_qk_lowrank_glm52_sg_pipeline (ds4_metal_context_current()->glm_qk_lowrank_glm52_sg_pipeline)
#define g_glm_qk_lowrank_batch_pipeline (ds4_metal_context_current()->glm_qk_lowrank_batch_pipeline)
#define g_glm_qk_lowrank_batch_glm52_t4_pipeline (ds4_metal_context_current()->glm_qk_lowrank_batch_glm52_t4_pipeline)
#define g_glm_value_project_q8_0_pipeline (ds4_metal_context_current()->glm_value_project_q8_0_pipeline)
#define g_glm_value_project_q8_0_batch_heads_pipeline (ds4_metal_context_current()->glm_value_project_q8_0_batch_heads_pipeline)
#define g_glm_value_project_q8_0_batch_heads_mma_pipeline (ds4_metal_context_current()->glm_value_project_q8_0_batch_heads_mma_pipeline)
#define g_glm_attention_indexed_decode_pipeline (ds4_metal_context_current()->glm_attention_indexed_decode_pipeline)
#define g_glm_attention_indexed_decode_split_group8_partial_pipeline (ds4_metal_context_current()->glm_attention_indexed_decode_split_group8_partial_pipeline)
#define g_glm_attention_indexed_decode_split_group8_partial_valid_fullheads_pipeline (ds4_metal_context_current()->glm_attention_indexed_decode_split_group8_partial_valid_fullheads_pipeline)
#define g_glm_attention_indexed_decode_split_group8_reduce_pipeline (ds4_metal_context_current()->glm_attention_indexed_decode_split_group8_reduce_pipeline)
#define g_glm_attention_indexed_decode_split_group8_reduce16_pipeline (ds4_metal_context_current()->glm_attention_indexed_decode_split_group8_reduce16_pipeline)
#define g_glm_attention_indexed_batch_pipeline (ds4_metal_context_current()->glm_attention_indexed_batch_pipeline)
#define g_glm_attention_indexed_batch_group2_pipeline (ds4_metal_context_current()->glm_attention_indexed_batch_group2_pipeline)
#define g_glm_attention_indexed_batch_q2_group4_pipeline (ds4_metal_context_current()->glm_attention_indexed_batch_q2_group4_pipeline)
#define g_glm_attention_indexed_batch_group8_pipeline (ds4_metal_context_current()->glm_attention_indexed_batch_group8_pipeline)
#define g_glm_attention_indexed_batch_lora_group8_vec_pipeline (ds4_metal_context_current()->glm_attention_indexed_batch_lora_group8_vec_pipeline)
#define g_glm_attention_indexed_batch_lora_group8_vec_valid_pipeline (ds4_metal_context_current()->glm_attention_indexed_batch_lora_group8_vec_valid_pipeline)
#define g_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads_pipeline (ds4_metal_context_current()->glm_attention_indexed_batch_lora_group8_vec_valid_fullheads_pipeline)
#define g_glm_attention_indexed_batch_lora_group8_vec_causal_pipeline (ds4_metal_context_current()->glm_attention_indexed_batch_lora_group8_vec_causal_pipeline)
#define g_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads_pipeline (ds4_metal_context_current()->glm_attention_indexed_batch_lora_group8_vec_causal_fullheads_pipeline)
#define g_glm_q4_k_pair_swiglu_f32_pipeline (ds4_metal_context_current()->glm_q4_k_pair_swiglu_f32_pipeline)
#define g_glm_q4_k_pair_swiglu2_f32_pipeline (ds4_metal_context_current()->glm_q4_k_pair_swiglu2_f32_pipeline)
#define g_glm_q4_k_pair_swiglu4_f32_pipeline (ds4_metal_context_current()->glm_q4_k_pair_swiglu4_f32_pipeline)
#define g_glm_q4_k_pair_swiglu2_mapped_f32_pipeline (ds4_metal_context_current()->glm_q4_k_pair_swiglu2_mapped_f32_pipeline)
#define g_glm_q4_k_pair_swiglu2_mapped_row_f32_pipeline (ds4_metal_context_current()->glm_q4_k_pair_swiglu2_mapped_row_f32_pipeline)
#define g_glm_q2_k_pair_swiglu_f32_pipeline (ds4_metal_context_current()->glm_q2_k_pair_swiglu_f32_pipeline)
#define g_glm_q2_k_addr_pair_swiglu2_f32_pipeline (ds4_metal_context_current()->glm_q2_k_addr_pair_swiglu2_f32_pipeline)
#define g_glm_q2_k_addr_pair_swiglu2_masked_f32_pipeline (ds4_metal_context_current()->glm_q2_k_addr_pair_swiglu2_masked_f32_pipeline)
#define g_glm_q4_k_addr_pair_swiglu_f32_pipeline (ds4_metal_context_current()->glm_q4_k_addr_pair_swiglu_f32_pipeline)
#define g_glm_q4_k_addr_pair_swiglu_masked_f32_pipeline (ds4_metal_context_current()->glm_q4_k_addr_pair_swiglu_masked_f32_pipeline)
#define g_glm_q2_k_down_f32_pipeline (ds4_metal_context_current()->glm_q2_k_down_f32_pipeline)
#define g_glm_q4_k_down_f32_pipeline (ds4_metal_context_current()->glm_q4_k_down_f32_pipeline)
#define g_glm_q2_k_addr_down_f32_pipeline (ds4_metal_context_current()->glm_q2_k_addr_down_f32_pipeline)
#define g_glm_q4_k_addr_down_f32_pipeline (ds4_metal_context_current()->glm_q4_k_addr_down_f32_pipeline)
#define g_glm_q5_k_pair_swiglu_f32_pipeline (ds4_metal_context_current()->glm_q5_k_pair_swiglu_f32_pipeline)
#define g_glm_q5_k_pair_swiglu_mapped_f32_pipeline (ds4_metal_context_current()->glm_q5_k_pair_swiglu_mapped_f32_pipeline)
#define g_glm_q5_k_pair_swiglu_mapped_row_f32_pipeline (ds4_metal_context_current()->glm_q5_k_pair_swiglu_mapped_row_f32_pipeline)
#define g_glm_q5_k_down_f32_pipeline (ds4_metal_context_current()->glm_q5_k_down_f32_pipeline)
#define g_glm_q6_k_down_f32_pipeline (ds4_metal_context_current()->glm_q6_k_down_f32_pipeline)
#define g_dsv4_router_weights_batch_pipeline (ds4_metal_context_current()->dsv4_router_weights_batch_pipeline)
#define g_dsv4_hc_expand4_pipeline (ds4_metal_context_current()->dsv4_hc_expand4_pipeline)
#define g_pipeline_cache (ds4_metal_context_current()->pipeline_cache)
#define g_decode_pipeline_fast_cache (ds4_metal_context_current()->decode_pipeline_fast_cache)
#define g_decode_pipeline_fast_lookup_active (ds4_metal_context_current()->decode_pipeline_fast_lookup_active)
#define g_decode_pipeline_fast_cache_entries (ds4_metal_context_current()->decode_pipeline_fast_cache_entries)
#define g_q4_expert_table_cache (ds4_metal_context_current()->q4_expert_table_cache)
#define g_q4_expert_layer_residency_cache (ds4_metal_context_current()->q4_expert_layer_residency_cache)
#define g_transient_buffers (ds4_metal_context_current()->transient_buffers)
#define g_zero_prefix_prefill_mask_cache (ds4_metal_context_current()->zero_prefix_prefill_mask_cache)
#define g_flash_attn_mask_buffer (ds4_metal_context_current()->flash_attn_mask_buffer)
#define g_flash_attn_zero_mask_buffer (ds4_metal_context_current()->flash_attn_zero_mask_buffer)
#define g_flash_attn_pad_buffer (ds4_metal_context_current()->flash_attn_pad_buffer)
#define g_flash_attn_tmp_buffer (ds4_metal_context_current()->flash_attn_tmp_buffer)
#define g_flash_attn_blk_buffer (ds4_metal_context_current()->flash_attn_blk_buffer)
#define g_flash_attn_ring_buffer (ds4_metal_context_current()->flash_attn_ring_buffer)
#define g_flash_attn_kv_buffer (ds4_metal_context_current()->flash_attn_kv_buffer)
#define g_glm_flash_attn_mask_buffer (ds4_metal_context_current()->glm_flash_attn_mask_buffer)
#define g_compressor_pool_kv_buffer (ds4_metal_context_current()->compressor_pool_kv_buffer)
#define g_compressor_pool_score_buffer (ds4_metal_context_current()->compressor_pool_score_buffer)
#define g_compressor_pool_score_cont_buffer (ds4_metal_context_current()->compressor_pool_score_cont_buffer)
#define g_compressor_pool_softmax_buffer (ds4_metal_context_current()->compressor_pool_softmax_buffer)
#define g_compressor_pool_product_buffer (ds4_metal_context_current()->compressor_pool_product_buffer)
#define g_compressor_store_ape_buffer (ds4_metal_context_current()->compressor_store_ape_buffer)
#define g_compressor_store_score_buffer (ds4_metal_context_current()->compressor_store_score_buffer)
#define g_embed_rows_buffer (ds4_metal_context_current()->embed_rows_buffer)
#define g_router_selection_buffer (ds4_metal_context_current()->router_selection_buffer)
#define g_router_weight_sum_buffer (ds4_metal_context_current()->router_weight_sum_buffer)
#define g_indexer_head_scores_buffer (ds4_metal_context_current()->indexer_head_scores_buffer)
#define g_indexer_topk_buffer (ds4_metal_context_current()->indexer_topk_buffer)
#define g_indexed_topk_buffer (ds4_metal_context_current()->indexed_topk_buffer)
#define g_f16_round_scratch_buffer (ds4_metal_context_current()->f16_round_scratch_buffer)
#define g_q8_prefill_scratch_buffer (ds4_metal_context_current()->q8_prefill_scratch_buffer)
#define g_qwen4_conv_halo_buffer (ds4_metal_context_current()->qwen4_conv_halo_buffer)
#define g_qwen4_conv_halo_bytes (ds4_metal_context_current()->qwen4_conv_halo_bytes)
#define g_raw_store_round_buffer (ds4_metal_context_current()->raw_store_round_buffer)
#define g_moe_gate_scratch_buffer (ds4_metal_context_current()->moe_gate_scratch_buffer)
#define g_moe_down_scratch_buffer (ds4_metal_context_current()->moe_down_scratch_buffer)
#define g_moe_id_map_buffer (ds4_metal_context_current()->moe_id_map_buffer)
#define g_moe_packed_rhs_buffer (ds4_metal_context_current()->moe_packed_rhs_buffer)
#define g_moe_packed_rhs_bytes (ds4_metal_context_current()->moe_packed_rhs_bytes)
#define g_moe_q4_gate_slots_buffer (ds4_metal_context_current()->moe_q4_gate_slots_buffer)
#define g_moe_q4_up_slots_buffer (ds4_metal_context_current()->moe_q4_up_slots_buffer)
#define g_moe_q4_down_slots_buffer (ds4_metal_context_current()->moe_q4_down_slots_buffer)
#define g_attn_out_group_ids_buffer (ds4_metal_context_current()->attn_out_group_ids_buffer)
#define g_stream_expert_batch_ready (ds4_metal_context_current()->stream_expert_batch_ready)
#define g_glm_streaming_prefill_full_layer_runtime (ds4_metal_context_current()->glm_streaming_prefill_full_layer_runtime)
#define g_flash_attn_mask_bytes (ds4_metal_context_current()->flash_attn_mask_bytes)
#define g_flash_attn_zero_mask_bytes (ds4_metal_context_current()->flash_attn_zero_mask_bytes)
#define g_flash_attn_pad_bytes (ds4_metal_context_current()->flash_attn_pad_bytes)
#define g_flash_attn_tmp_bytes (ds4_metal_context_current()->flash_attn_tmp_bytes)
#define g_flash_attn_blk_bytes (ds4_metal_context_current()->flash_attn_blk_bytes)
#define g_flash_attn_ring_bytes (ds4_metal_context_current()->flash_attn_ring_bytes)
#define g_flash_attn_kv_bytes (ds4_metal_context_current()->flash_attn_kv_bytes)
#define g_glm_flash_attn_mask_bytes (ds4_metal_context_current()->glm_flash_attn_mask_bytes)
#define g_glm_flash_attn_mask_pos0 (ds4_metal_context_current()->glm_flash_attn_mask_pos0)
#define g_glm_flash_attn_mask_tokens (ds4_metal_context_current()->glm_flash_attn_mask_tokens)
#define g_glm_flash_attn_mask_cache_len (ds4_metal_context_current()->glm_flash_attn_mask_cache_len)
#define g_glm_flash_attn_mask_valid (ds4_metal_context_current()->glm_flash_attn_mask_valid)
#define g_compressor_pool_kv_bytes (ds4_metal_context_current()->compressor_pool_kv_bytes)
#define g_compressor_pool_score_bytes (ds4_metal_context_current()->compressor_pool_score_bytes)
#define g_compressor_pool_score_cont_bytes (ds4_metal_context_current()->compressor_pool_score_cont_bytes)
#define g_compressor_pool_softmax_bytes (ds4_metal_context_current()->compressor_pool_softmax_bytes)
#define g_compressor_pool_product_bytes (ds4_metal_context_current()->compressor_pool_product_bytes)
#define g_compressor_store_ape_bytes (ds4_metal_context_current()->compressor_store_ape_bytes)
#define g_compressor_store_score_bytes (ds4_metal_context_current()->compressor_store_score_bytes)
#define g_embed_rows_bytes (ds4_metal_context_current()->embed_rows_bytes)
#define g_router_selection_bytes (ds4_metal_context_current()->router_selection_bytes)
#define g_router_weight_sum_bytes (ds4_metal_context_current()->router_weight_sum_bytes)
#define g_indexer_head_scores_bytes (ds4_metal_context_current()->indexer_head_scores_bytes)
#define g_indexer_topk_bytes (ds4_metal_context_current()->indexer_topk_bytes)
#define g_indexed_topk_bytes (ds4_metal_context_current()->indexed_topk_bytes)
#define g_f16_round_scratch_bytes (ds4_metal_context_current()->f16_round_scratch_bytes)
#define g_q8_prefill_scratch_bytes (ds4_metal_context_current()->q8_prefill_scratch_bytes)
#define g_raw_store_round_bytes (ds4_metal_context_current()->raw_store_round_bytes)
#define g_moe_gate_scratch_bytes (ds4_metal_context_current()->moe_gate_scratch_bytes)
#define g_moe_down_scratch_bytes (ds4_metal_context_current()->moe_down_scratch_bytes)
#define g_moe_id_map_bytes (ds4_metal_context_current()->moe_id_map_bytes)
#define g_moe_q4_gate_slots_bytes (ds4_metal_context_current()->moe_q4_gate_slots_bytes)
#define g_moe_q4_up_slots_bytes (ds4_metal_context_current()->moe_q4_up_slots_bytes)
#define g_moe_q4_down_slots_bytes (ds4_metal_context_current()->moe_q4_down_slots_bytes)
#define g_attn_out_group_ids_bytes (ds4_metal_context_current()->attn_out_group_ids_bytes)
#define g_mpp_invalid_env_reported (ds4_metal_context_current()->mpp_invalid_env_reported)

#define g_timeline_batch (ds4_metal_context_current()->timeline_batch)
#define g_gpu_busy_accum (ds4_metal_context_current()->gpu_busy_accum)
#define g_gpu_busy_cbs (ds4_metal_context_current()->gpu_busy_cbs)
#define g_qwen4_verify_rows_exact (ds4_metal_context_current()->qwen4_verify_rows_exact)
#define g_tp_gpu_event (ds4_metal_context_current()->tp_gpu_event)
#define g_tp_cpu_event (ds4_metal_context_current()->tp_cpu_event)
#define g_tp_batch_gpu_event (ds4_metal_context_current()->tp_batch_gpu_event)
#define g_tp_batch_cpu_event (ds4_metal_context_current()->tp_batch_cpu_event)
#define g_tp_split_rank (ds4_metal_context_current()->tp_split_rank)
#define g_tp_split_world (ds4_metal_context_current()->tp_split_world)
#define g_tp_attn_head_split (ds4_metal_context_current()->tp_attn_head_split)
#define g_tp_poll_buffer (ds4_metal_context_current()->tp_poll_buffer)
#define g_tp_poll_region (ds4_metal_context_current()->tp_poll_region)
#define g_tp_poll_status (ds4_metal_context_current()->tp_poll_status)
#define g_tp_poll_prev_valid (ds4_metal_context_current()->tp_poll_prev_valid)
#define g_tp_poll_prev_slot (ds4_metal_context_current()->tp_poll_prev_slot)
#define g_tp_poll_prev_seq (ds4_metal_context_current()->tp_poll_prev_seq)
#define g_tp_native_prefetch (ds4_metal_context_current()->tp_native_prefetch)
#define g_tp_fold_ctl (ds4_metal_context_current()->tp_fold_ctl)
#define g_tp_prefetch_scratch (ds4_metal_context_current()->tp_prefetch_scratch)
#define g_tp_stat_poll_lines (ds4_metal_context_current()->tp_stat_poll_lines)
#define g_tp_stat_poll_max_line (ds4_metal_context_current()->tp_stat_poll_max_line)
#define g_tp_stat_poll_count (ds4_metal_context_current()->tp_stat_poll_count)
#define g_tp_stat_encode_lead (ds4_metal_context_current()->tp_stat_encode_lead)
#define g_tp_stat_release_ms (ds4_metal_context_current()->tp_stat_release_ms)
#define g_tp_stat_cbwait_ms (ds4_metal_context_current()->tp_stat_cbwait_ms)
#define g_tp_check_buffer (ds4_metal_context_current()->tp_check_buffer)
#define g_tp_check_words (ds4_metal_context_current()->tp_check_words)
#define g_tp_slab_cpu (ds4_metal_context_current()->tp_slab_cpu)
#define g_tp_slab_buffer (ds4_metal_context_current()->tp_slab_buffer)
#define g_tp_gpu_flags (ds4_metal_context_current()->tp_gpu_flags)
#define g_tp_exchange_fn (ds4_metal_context_current()->tp_exchange_fn)
#define g_tp_batch_exchange_fn (ds4_metal_context_current()->tp_batch_exchange_fn)
#define g_tp_big_exchange_fn (ds4_metal_context_current()->tp_big_exchange_fn)
#define g_tp_exchange_ud (ds4_metal_context_current()->tp_exchange_ud)
#define g_tp_thread (ds4_metal_context_current()->tp_thread)
#define g_tp_thread_running (ds4_metal_context_current()->tp_thread_running)
#define g_tp_shutdown (ds4_metal_context_current()->tp_shutdown)
#define g_tp_failed_flag (ds4_metal_context_current()->tp_failed_flag)
#define g_tp_mutex (ds4_metal_context_current()->tp_mutex)
#define g_tp_cond (ds4_metal_context_current()->tp_cond)
#define g_tp_queue (ds4_metal_context_current()->tp_queue)
#define g_tp_queue_head (ds4_metal_context_current()->tp_queue_head)
#define g_tp_queue_count (ds4_metal_context_current()->tp_queue_count)
#define g_tp_stat_gates (ds4_metal_context_current()->tp_stat_gates)
#define g_tp_stat_batch_gates (ds4_metal_context_current()->tp_stat_batch_gates)
#define g_tp_stat_batch_gpu_wait_ms (ds4_metal_context_current()->tp_stat_batch_gpu_wait_ms)
#define g_tp_stat_batch_exchange_ms (ds4_metal_context_current()->tp_stat_batch_exchange_ms)
#define g_tp_stat_batch_release_ms (ds4_metal_context_current()->tp_stat_batch_release_ms)
#define g_tp_stat_gpu_wait_ms (ds4_metal_context_current()->tp_stat_gpu_wait_ms)
#define g_tp_stat_exchange_ms (ds4_metal_context_current()->tp_stat_exchange_ms)
#define g_tp_stat_gate_count (ds4_metal_context_current()->tp_stat_gate_count)
#define g_tp_stat_gate_gpu_wait_ms (ds4_metal_context_current()->tp_stat_gate_gpu_wait_ms)
#define g_tp_stat_gate_exchange_ms (ds4_metal_context_current()->tp_stat_gate_exchange_ms)
#define g_tp_keepalive_queue (ds4_metal_context_current()->tp_keepalive_queue)
#define g_tp_keepalive_buffer (ds4_metal_context_current()->tp_keepalive_buffer)
#define g_tp_keepalive_thread (ds4_metal_context_current()->tp_keepalive_thread)
#define g_tp_keepalive_running (ds4_metal_context_current()->tp_keepalive_running)
#define g_tp_keepalive_paused (ds4_metal_context_current()->tp_keepalive_paused)
#define g_qwen4_pipelines (ds4_metal_context_current()->qwen4_pipelines)
#define g_qwen4_gdn_slots (ds4_metal_context_current()->qwen4_gdn_slots)
#define g_qwen4_rope_freq (ds4_metal_context_current()->qwen4_rope_freq)
#define g_qwen4_rope_mscale (ds4_metal_context_current()->qwen4_rope_mscale)
#define g_qwen4_rope_set (ds4_metal_context_current()->qwen4_rope_set)
#define g_qwen4_nax_half_x (ds4_metal_context_current()->qwen4_nax_half_x)
#define g_qwen4_nax_half_mid (ds4_metal_context_current()->qwen4_nax_half_mid)
#define g_qwen4_nax_half_midr (ds4_metal_context_current()->qwen4_nax_half_midr)
#define g_qwen4_nax_half_x_bytes (ds4_metal_context_current()->qwen4_nax_half_x_bytes)
#define g_qwen4_nax_half_mid_bytes (ds4_metal_context_current()->qwen4_nax_half_mid_bytes)
#define g_qwen4_nax_half_midr_bytes (ds4_metal_context_current()->qwen4_nax_half_midr_bytes)
#define g_qwen4_nax_half_mid_for (ds4_metal_context_current()->qwen4_nax_half_mid_for)
#define g_qwen4_nax_half_mid_count (ds4_metal_context_current()->qwen4_nax_half_mid_count)
#define g_qwen4_dense_mm_partials (ds4_metal_context_current()->qwen4_dense_mm_partials)
#define g_qwen4_dense_mm_partials_bytes (ds4_metal_context_current()->qwen4_dense_mm_partials_bytes)
