/* Test-only declarations. The compatibility shell supplies Engine ownership;
 * every candidate graph operation below calls its current native Zig owner. */
bool ds4_test_qwen4_dense_rows(ds4_gpu_tensor *, const ds4_model *, const ds4_tensor *, const ds4_gpu_tensor *, uint32_t, uint64_t);
bool ds4_test_qwen4_hc_mix(ds4_qwen4_gpu_graph *, const ds4_model *, const ds4_tensor *, const ds4_tensor *, const ds4_tensor *, const ds4_tensor *, uint32_t);
bool ds4_test_qwen4_attention(ds4_qwen4_gpu_graph *, const ds4_model *, const ds4_layer_weights *, uint32_t, uint32_t, uint32_t);
bool ds4_test_qwen4_dispatch(ds4_qwen4_gpu_graph *, uint32_t, ds4_gpu_tensor *, ds4_gpu_tensor *, ds4_gpu_tensor *, uint32_t, uint32_t, bool, uint32_t);
bool ds4_test_qwen4_moe(ds4_qwen4_gpu_graph *, const ds4_model *, const ds4_layer_weights *, uint32_t, uint32_t);
bool ds4_test_qwen4_cache_steps(ds4_qwen4_gpu_graph *, const ds4_model *, const ds4_weights *, uint32_t, const int *, uint32_t, uint32_t);
bool ds4_test_qwen4_chain(ds4_qwen4_gpu_graph *, const ds4_model *, const ds4_weights *, int, uint32_t, int *);
bool ds4_test_qwen4_fused(const ds4_qwen4_gpu_graph *, uint32_t);
uint32_t ds4_test_qwen4_draft_rows(const ds4_qwen4_gpu_graph *);
bool ds4_test_qwen4_draft_load(ds4_qwen4_gpu_graph *, const ds4_model *, const ds4_tensor *);
#define qwen4_gemv_rows ds4_test_qwen4_dense_rows
#define qwen4_gemv(o,m,w,x,t) ds4_test_qwen4_dense_rows(o,m,w,x,t,0)
#define qwen4_graph_hc_mix ds4_test_qwen4_hc_mix
#define qwen4_graph_attention ds4_test_qwen4_attention
#define qwen4_graph_attention_dispatch ds4_test_qwen4_dispatch
#define qwen4_graph_moe ds4_test_qwen4_moe
#define qwen4_graph_mtp_cache_steps ds4_test_qwen4_cache_steps
#define qwen4_graph_mtp_chain_step ds4_test_qwen4_chain
#define qwen4_graph_fused ds4_test_qwen4_fused
#define qwen4_mtp_draft_head_load ds4_test_qwen4_draft_load
static uint32_t qwen4_mtp_draft_rows(void) { ds4_qwen4_gpu_graph g = {.config = qwen4_native_config()}; return ds4_test_qwen4_draft_rows(&g); }
static uint64_t qwen4_payload_kv_bytes(uint32_t n) { return (uint64_t)n * DS4_N_HEAD_KV * DS4_N_HEAD_DIM * 2; }
static uint64_t qwen4_payload_ik_bytes(uint32_t n) { return (uint64_t)n * DS4_N_INDEXER_HEAD_DIM * 4; }
static uint64_t qwen4_payload_block_key_bytes(uint32_t n) { return (uint64_t)(n / 4) * DS4_N_INDEXER_HEAD_DIM * 2; }
#define qwen4_prefill_chunk_resolve ds4_native_qwen4_prefill_chunk_resolve
static bool qwen4_streaming_memory_admit(ds4_engine *e, uint64_t bytes, bool fit) { return ds4_native_qwen4_memory_admit(&e->core, &ds4_engine_native_config(e)->qwen, bytes, fit); }
