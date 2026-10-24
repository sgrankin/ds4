#ifndef DS4_ENGINE_GRAPH_API_H
#define DS4_ENGINE_GRAPH_API_H

/* Phase-2 declaration input. Root shares this generated gguf_c Module with
 * input/config/weights when activating the graph owner; no SDK dump is kept. */
#include "../model_api.h"
#include "../core_api.h"
#include "../diagnostics_api.h"
#include "../payload_io.h"
#include "../qwen4/api.h"
#include "../qwen4/model_api.h"
#include "../qwen4/vision_api.h"
#include "../glm53/api.h"
#include "../glm53/model_api.h"
#include "../glm53/vision_api.h"
#include "../glm53/session_api.h"
#include "ds4_deepseek41_graph.h"
#include "ds4_session_frontier.h"
#include "ds4_model_speculation.h"
#include "ds4_engine_memory.h"
#include "ds4_model_residency.h"
#include "ds4_ssd.h"
#include "ds4_session_resources.h"
#include "ds4_engine_tp_state.h"
#include "ds4_sampling.h"
#include <sys/sysctl.h>
#include "ds4_deepseek41_dspark_state.h"
#include "ds4_deepseek41_gpu.h"
#include "ds4_tp.h"
#include "ds4_gpu_tp.h"
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <limits.h>
#include "executor_services.h"

bool ds4_native_graph_alloc41(ds41_gpu_graph *graph, const ds4_model *model,
    const ds4_weights *weights, const ds4_model_config *config, const char *path,
    uint32_t context, uint32_t prefill_chunk, bool streaming);
void ds4_native_graph_free41(ds41_gpu_graph *graph);
void ds4_native_graph_reset41(ds41_gpu_graph *graph);
bool ds4_native_graph_workspace_grow41(ds41_gpu_graph *graph,
    const ds4_model_config *config, uint32_t capacity);
uint64_t ds4_native_graph_bytes41(const ds4_model_config *config,
    uint32_t context, uint32_t prefill_chunk);
ds4_context_memory ds4_native_graph_memory41(const ds4_model_config *config,
    uint32_t context, uint32_t prefill_chunk);

/* Complete scalar/batch attention controllers. The batch caller has already
 * projected its Q/KV rows; precision scope is passed explicitly from C. */
bool ds4_native_graph_attention41(ds41_gpu_graph *graph, const ds4_model *model,
    const ds4_layer_weights *layer, const ds4_model_config *config,
    uint32_t layer_index, bool projected);
bool ds4_native_graph_attention_batch41(ds41_gpu_graph *graph, const ds4_model *model,
    const ds4_layer_weights *layer, const ds4_model_config *config,
    uint32_t layer_index, uint32_t count, bool exact_prefill_rows);
/* Existing optional trace I/O; ordinary execution returns true. */
bool ds4_native_graph_trace_row(ds4_gpu_tensor *tensor, uint32_t width,
    uint32_t start, uint32_t count, uint32_t layer, const char *name);

bool ds4_native_graph_step41(ds41_gpu_graph *graph, const ds4_model *model,
    const ds4_weights *weights, const ds4_model_config *config, int token,
    float *logits, bool exact_prefill_rows);
bool ds4_native_graph_logits41(ds41_gpu_graph *graph, const ds4_model *model,
    const ds4_weights *weights, const ds4_model_config *config, float *logits,
    bool exact_prefill_rows);
uint32_t ds4_native_graph_prefill_count41(const ds41_gpu_graph *graph,
    const ds4_weights *weights, const ds4_model_config *config, uint32_t remaining);
bool ds4_native_graph_prefill41(ds41_gpu_graph *graph, const ds4_model *model,
    const ds4_weights *weights, const ds4_model_config *config,
    const int *tokens, uint32_t count, ds4_session_progress_fn progress,
    void *progress_userdata, int total, ds4_session_cancel_fn cancel,
    void *cancel_userdata, bool encoder_only, bool resume_encoder,
    bool exact_prefill_rows);
ds41_gpu_graph *ds4_native_graph_batch_workspace41(ds41_gpu_graph *const *graphs,
    int count);
bool ds4_native_graph_step_batch41(ds41_gpu_graph *const *graphs, const int *tokens,
    int count, uint32_t prefill_rows, const ds4_model *model,
    const ds4_weights *weights, const ds4_model_config *config,
    bool exact_prefill_rows);
bool ds4_native_graph_short_prefill41(ds41_gpu_graph *graph, const ds4_model *model,
    const ds4_weights *weights, const ds4_model_config *config,
    const int *tokens, uint32_t count, bool exact_prefill_rows);

/* Canonical DSV4 v2 numerical state. Session policy/envelopes remain in C. */
uint64_t ds4_native_graph_payload_body_bytes41(ds41_gpu_graph *graph,
    const ds4_model_config *config, uint32_t position);
int ds4_native_graph_save_payload41(ds41_gpu_graph *graph,
    const ds4_model_config *config, FILE *stream, const ds4_tokens *tokens,
    const float *logits, char *error, size_t error_size);
int ds4_native_graph_load_payload41(ds41_gpu_graph *graph,
    const ds4_model_config *config, FILE *stream,
    const uint32_t header[DS4_SESSION_PAYLOAD_U32_FIELDS], uint64_t remaining,
    float *logits, ds4_tokens *out, bool *mutated, char *error, size_t error_size);

/* Stale suffix graph operations. Session identity and policy stay opaque. */
uint64_t ds4_native_graph_prerope_bytes41(uint32_t end);
bool ds4_native_graph_prerope_enable41(ds41_gpu_graph *, const ds4_model_config *);
bool ds4_native_graph_prerope_capture41(ds41_gpu_graph *, const ds4_model_config *, ds4_suffix_cache *);
int ds4_native_graph_suffix_validate_prefix41(ds41_gpu_graph *, const ds4_model_config *,
    const ds4_suffix_cache *, uint32_t *end, char *error, size_t error_size);
bool ds4_native_graph_prerope_restore41(ds41_gpu_graph *, const ds4_model_config *,
    const ds4_suffix_cache *, uint32_t end, ds4_session_cancel_fn, void *userdata);
int ds4_native_graph_graft_suffix41(ds41_gpu_graph *, const ds4_model_config *,
    const ds4_suffix_cache *, const ds4_tokens *, uint32_t source_start, uint32_t start,
    ds4_suffix_relocation_stats *out, bool *mutated, ds4_session_cancel_fn,
    void *userdata, char *error, size_t error_size);

/* V4.1 drafter math and bounded target verification over the actual owner. */
ds41_dspark *ds4_native_dspark_alloc41(const ds4_model_config *, const ds4_dspark_weights *);
void ds4_native_dspark_free41(ds41_dspark *);
void ds4_native_dspark_reset41(ds41_dspark *);
bool ds4_native_dspark_capture41(const ds4_model_config *, ds41_dspark *, uint32_t, uint32_t, const ds4_gpu_tensor *, uint32_t);
bool ds4_native_dspark_seed41(ds41_dspark *, const ds4_model *, const ds4_model_config *, uint32_t, bool);
bool ds4_native_dspark_stage41(ds41_dspark *, const ds4_model *, const ds4_model_config *, uint32_t, uint32_t, bool);
bool ds4_native_dspark_forward41(ds41_dspark *, const ds4_model *, const ds4_weights *, const ds4_model *, const ds4_model_config *, uint32_t, int, ds41_gpu_graph *, bool);
bool ds4_native_dspark_markov41(ds41_dspark *, const ds4_model *, const ds4_model_config *, uint32_t, int, float *, bool);
bool ds4_native_dspark_verify41(ds41_dspark *, ds41_gpu_graph *, const ds4_model *, const ds4_weights *, const ds4_model_config *, ds4_tp *, const int *, uint32_t, bool);
bool ds4_native_dspark_commit41(ds41_dspark *, ds41_gpu_graph *, const ds4_model_config *, uint32_t, float *);

/* Complete V4.1 speculative cycle over borrowed actual session owners. */
#include "ds4_dspark_cycle.h"
#include "ds4_sampling.h"

bool ds4_native_dspark_cycle_draft41(const ds4_dspark_cycle_args *,
    ds41_gpu_graph *, const ds4_model *, const ds4_weights *, const ds4_model *,
    const ds4_model_config *, int token, bool exact_prefill_rows);
bool ds4_native_dspark_cycle_verify41(const ds4_dspark_cycle_args *,
    ds41_gpu_graph *, const ds4_model *, const ds4_weights *,
    const ds4_model_config *, const int *tokens, uint32_t count, bool exact_prefill_rows);
bool ds4_native_dspark_cycle_commit41(const ds4_dspark_cycle_args *,
    ds41_gpu_graph *, const ds4_model *, const ds4_weights *,
    const ds4_model_config *, const int *tokens, uint32_t count, uint32_t keep,
    bool exact_prefill_rows);
int ds4_native_dspark_cycle_run41(const ds4_dspark_cycle_args *, ds41_gpu_graph *,
    const ds4_model *, const ds4_weights *, const ds4_model *, const ds4_model_config *,
    int first_token, int cap, int eos, bool ignore_eos, ds4_think_mode,
    float temperature, int top_k, float top_p, float min_p, uint64_t *rng,
    int *accepted, char *error, size_t error_size, bool exact_prefill_rows);

/* Append to the existing generated graph_api.h declaration input. Include the
 * real ds4_session_frontier.h there; no Session/Engine definition is exposed. */
int ds4_native_session_sync41(ds4_session_frontier *frontier,
    ds41_gpu_graph *graph, const ds4_model *model, const ds4_weights *weights,
    const ds4_model_config *config, ds4_engine_memory *memory,
    const ds4_engine_memory_inputs *inputs, bool graph_ready,
    const ds4_tokens *prompt, ds4_session_progress_fn progress, void *progress_ud,
    ds4_session_cancel_fn cancel, void *cancel_ud, bool exact_prefill_rows,
    char *error, size_t error_size);
int ds4_native_session_eval41(ds4_session_frontier *frontier,
    ds41_gpu_graph *graph, const ds4_model *model, const ds4_weights *weights,
    const ds4_model_config *config, ds4_engine_memory *memory,
    const ds4_engine_memory_inputs *inputs, bool graph_ready,
    int token, bool exact_prefill_rows, char *error, size_t error_size);
void ds4_native_session_invalidate41(ds4_session_frontier *frontier,
    ds41_gpu_graph *ready_graph);

/* Admission is shared by production and the simulated-host fixture. */
bool ds4_native_memory_admit_for_host41(ds4_engine_memory *memory,
    const ds4_weights *weights, const ds4_model_config *config,
    const ds4_engine_memory_inputs *inputs, uint64_t graph_bytes,
    bool fit_cache, uint64_t host_bytes, uint64_t recommended_bytes);
bool ds4_native_memory_admit41(ds4_engine_memory *memory,
    const ds4_weights *weights, const ds4_model_config *config,
    const ds4_engine_memory_inputs *inputs, uint64_t graph_bytes, bool fit_cache);

bool ds4_native_frontier_vision_prefix(const ds4_session_frontier *frontier,
    const ds4_vision_span *images, size_t count);
bool ds4_native_frontier_vision_state(const ds4_session_frontier *frontier,
    const ds4_vision_span *images, size_t count);
bool ds4_native_frontier_vision_fingerprints(const ds4_session_frontier *frontier,
    const ds4_vision_span *images, size_t count);
bool ds4_native_frontier_rebase_vision(const ds4_session_frontier *frontier,
    ds4_vision_span *images, size_t count);
bool ds4_native_frontier_vision_overlap(const ds4_session_frontier *frontier,
    uint32_t start, uint32_t count);
bool ds4_native_frontier_remember_images(ds4_session_frontier *frontier);

/* Whole encoder lease actions; C fixtures use the production native owner. */
void ds4_native_encoder_acquire41(ds41_encoder_residency *lease,
    ds41_gpu_graph *graph, const ds4_model *model, const ds4_weights *weights,
    const ds4_model_config *config, uint32_t remaining,
    ds4_session_cancel_fn cancel, void *userdata);
void ds4_native_encoder_release41(ds41_encoder_residency *lease,
    ds41_gpu_graph *graph, const ds4_model *model);

/* Canonical local V4.1 checkpoints and private DRP1 replay envelopes. */
uint64_t ds4_native_checkpoint_bytes41(ds4_session_frontier *frontier,
    ds41_gpu_graph *graph, const ds4_model_config *config, bool ready);
int ds4_native_checkpoint_save_payload41(ds4_session_frontier *frontier,
    ds41_gpu_graph *graph, const ds4_model_config *config, bool ready, FILE *stream, char *error, size_t error_size);
int ds4_native_checkpoint_load_payload41(ds4_session_frontier *frontier,
    ds41_gpu_graph *graph, const ds4_model_config *config, bool ready, FILE *stream, uint64_t bytes, char *error, size_t error_size);
int ds4_native_checkpoint_save_snapshot41(ds4_session_frontier *frontier,
    ds41_gpu_graph *graph, const ds4_model_config *config, bool ready, ds4_session_snapshot *snapshot, char *error, size_t error_size);
int ds4_native_checkpoint_load_snapshot41(ds4_session_frontier *frontier,
    ds41_gpu_graph *graph, const ds4_model_config *config, bool ready, const ds4_session_snapshot *snapshot, char *error, size_t error_size);
int ds4_native_checkpoint_stage41(ds4_session_frontier *frontier,
    ds41_gpu_graph *graph, const ds4_model_config *config, bool ready, ds4_session_payload_file *output, char *error, size_t error_size);
uint64_t ds4_native_checkpoint_replay_bytes41(ds4_session_frontier *frontier,
    ds41_gpu_graph *graph, const ds4_model_config *config, bool ready);
int ds4_native_checkpoint_save_replay41(ds4_session_frontier *frontier,
    ds41_gpu_graph *graph, const ds4_model_config *config, bool ready, ds4_session_snapshot *snapshot, char *error, size_t error_size);
int ds4_native_checkpoint_load_replay41(ds4_session_frontier *frontier,
    ds41_gpu_graph *graph, const ds4_model_config *config, bool ready, const ds4_session_snapshot *snapshot, bool *invalidate, char *error, size_t error_size);
int ds4_native_checkpoint_write_staged41(const ds4_session_payload_file *staged,
    FILE *stream, char *error, size_t error_size);
void ds4_native_checkpoint_free_staged41(ds4_session_payload_file *staged);
void ds4_native_checkpoint_free_snapshot41(ds4_session_snapshot *snapshot);

/* Whole V4.1 device/host-resource lifetime. All inputs are actual records;
 * caller-owned Session shell and TP transport registration remain outside. */
bool ds4_native_session_resources_init41(ds4_deepseek41_session_device *,
    ds4_session_frontier *, ds4_engine_memory *, const ds4_engine_memory_inputs *,
    const ds4_model *, const ds4_weights *, const ds4_model_config *, const char *,
    uint32_t context, uint32_t chunk, bool quality, bool streaming,
    const ds4_dspark_weights *drafter, const ds4_engine_tp_state *);
void ds4_native_session_resources_free41(ds4_deepseek41_session_device *,
    ds4_session_frontier *, ds4_engine_memory *);

/* Complete DS41 Metal startup cache phases; resources/inputs are borrowed. */
void ds4_native_memory_cap_requested41(ds4_engine_memory *, const ds4_weights *,
    const ds4_model_config *, const ds4_engine_memory_inputs *, int context, uint32_t chunk);
bool ds4_native_memory_automatic_cache41(ds4_engine_memory *, const ds4_weights *,
    const ds4_model_config *, const ds4_engine_memory_inputs *, int context, uint32_t chunk);
bool ds4_native_memory_reserve_prefill41(ds4_engine_memory *, const ds4_weights *,
    const ds4_model_config *, const ds4_engine_memory_inputs *, bool full_layers_set);

/* Initial V4.1 Metal target views/cache policy. On success, selection owns
 * libc span arrays until the optional-cache phase retires them. Failure leaves
 * caller selection untouched; the old startup byte charge may already be set. */
bool ds4_native_target_residency_install41(ds4_engine_memory *, const ds4_weights *,
    const ds4_model_config *, const ds4_engine_memory_inputs *, const ds4_model *,
    const ds4_model_residency_inputs *, ds4_model_map_selection *);
void ds4_native_target_selection_free41(ds4_model_map_selection *);

/* Append to the SAME graph_api.h generated input. */
#include "ds4_session_control.h"
int ds4_native_session_request_sync41(ds4_session_frontier *,
    ds4_deepseek41_session_device *, ds4_engine_core *, ds4_engine_tp_state *,
    ds4_session_control *, const ds4_model_config *,
    const ds4_engine_memory_inputs *, const ds4_tokens *, bool exact,
    char *, size_t);
int ds4_native_session_multimodal_sync41(ds4_session_frontier *,
    ds4_deepseek41_session_device *, ds4_engine_core *, ds4_engine_tp_state *,
    ds4_session_control *, const ds4_model_config *,
    const ds4_engine_memory_inputs *, const ds4_tokens *,
    const ds4_vision_span *, size_t image_count, bool exact, char *, size_t);
void ds4_native_session_prefill_logit_dump(const float *, uint32_t vocabulary);

/* Append to the SAME graph_api.h after Coordinator prototypes. */
void ds4_native_session_suffix_free(ds4_suffix_cache *);
int ds4_native_session_suffix_enable41(ds4_session_frontier *, ds4_deepseek41_session_device *,
    ds4_engine_core *, ds4_engine_tp_state *, ds4_session_control *, ds4_engine *,
    const ds4_model_config *, const ds4_engine_memory_inputs *, bool exact,
    char *, size_t);
uint64_t ds4_native_session_suffix_archive_bytes41(ds4_session_frontier *, ds4_deepseek41_session_device *,
    ds4_engine_core *, ds4_engine_tp_state *, ds4_session_control *, ds4_engine *,
    const ds4_model_config *, const ds4_engine_memory_inputs *, bool exact);
uint64_t ds4_native_session_suffix_bytes41(ds4_session_frontier *, ds4_deepseek41_session_device *,
    ds4_engine_core *, ds4_engine_tp_state *, ds4_session_control *, ds4_engine *,
    const ds4_model_config *, const ds4_engine_memory_inputs *, bool exact);
int ds4_native_session_suffix_capture41(ds4_session_frontier *, ds4_deepseek41_session_device *,
    ds4_engine_core *, ds4_engine_tp_state *, ds4_session_control *, ds4_engine *,
    const ds4_model_config *, const ds4_engine_memory_inputs *, bool exact,
    ds4_suffix_cache *, char *, size_t);
int ds4_native_session_suffix_load_prefix41(ds4_session_frontier *, ds4_deepseek41_session_device *,
    ds4_engine_core *, ds4_engine_tp_state *, ds4_session_control *, ds4_engine *,
    const ds4_model_config *, const ds4_engine_memory_inputs *, bool exact,
    const ds4_suffix_cache *, char *, size_t);
int ds4_native_session_suffix_diagnostic41(ds4_session_frontier *, ds4_deepseek41_session_device *,
    ds4_engine_core *, ds4_engine_tp_state *, ds4_session_control *, ds4_engine *,
    const ds4_model_config *, const ds4_engine_memory_inputs *, bool exact,
    ds4_session_snapshot *, char *, size_t);
int ds4_native_session_suffix_graft41(ds4_session_frontier *, ds4_deepseek41_session_device *,
    ds4_engine_core *, ds4_engine_tp_state *, ds4_session_control *, ds4_engine *,
    const ds4_model_config *, const ds4_engine_memory_inputs *, bool exact,
    const ds4_suffix_cache *, const ds4_tokens *, uint32_t source_start, uint32_t start,
    ds4_suffix_reuse_stats *, ds4_suffix_relocation_stats *, char *, size_t);

/* Include the real borrowed view in the SAME graph_api.h / gguf_c Module. */
#include "ds4_session_batch.h"
int ds4_native_session_request_eval41(ds4_session_frontier *,
    ds4_deepseek41_session_device *, ds4_engine_core *, ds4_engine_tp_state *,
    ds4_session_control *, ds4_session *, ds4_dist_session *,
    const ds4_model_config *, const ds4_engine_memory_inputs *, int token,
    bool exact, char *, size_t);
int ds4_native_session_batch41(ds4_session_frontier *,
    ds4_deepseek41_session_device *, ds4_engine_core *, ds4_engine_tp_state *,
    ds4_session_control *, ds4_engine *, const ds4_model_config *,
    const ds4_engine_memory_inputs *, const ds4_session_batch_view *, int count,
    const ds4_session_batch_view *prefill, const ds4_tokens *prompt, bool exact,
    bool *native_batch, bool *mixed_batch, char *, size_t);

/* Append AFTER the Batch prototypes in the SAME graph_api.h. */
int ds4_native_session_speculate_argmax41(const ds4_session_batch_view *, ds4_engine_core *, ds4_engine_tp_state *,
    const ds4_model_config *, const ds4_engine_memory_inputs *, ds4_tp *verify_transport,
    int, int, int, bool, ds4_think_mode, int *, int, bool, char *, size_t);
int ds4_native_session_speculate_sample41(const ds4_session_batch_view *, ds4_engine_core *, ds4_engine_tp_state *,
    const ds4_model_config *, const ds4_engine_memory_inputs *, ds4_tp *verify_transport,
    int, int, int, float, int, float, float, uint64_t *, int *, int, bool, char *, size_t);
int ds4_native_session_restore_prefix41(const ds4_session_batch_view *, ds4_engine_core *, ds4_engine_tp_state *,
    const ds4_model_config *, const ds4_engine_memory_inputs *, ds4_tp *verify_transport,
    int, bool);
bool ds4_native_session_rewind_speculative41(const ds4_session_batch_view *, ds4_engine_core *, ds4_engine_tp_state *,
    const ds4_model_config *, const ds4_engine_memory_inputs *, ds4_tp *verify_transport,
    int, bool);
void ds4_native_session_rewind41(const ds4_session_batch_view *, ds4_engine_core *, ds4_engine_tp_state *,
    const ds4_model_config *, const ds4_engine_memory_inputs *, ds4_tp *verify_transport,
    int, bool);
int ds4_native_session_speculative_batch41(const ds4_session_batch_view *,
    ds4_engine_core *, ds4_engine_tp_state *, const ds4_model_config *,
    const ds4_engine_memory_inputs *, ds4_tp *, const ds4_session_batch_view *,
    int count, int (*accepted)[2], int *counts, bool exact,
    bool *native_batch, bool *mixed_batch, char *, size_t);

#include "ds4_engine_vision.h"

int ds4_native_graph_route_batch41(ds41_gpu_graph *, const ds4_model *, const ds4_layer_weights *, const ds4_model_config *, uint32_t);
#endif
