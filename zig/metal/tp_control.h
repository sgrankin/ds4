#ifndef DS4_METAL_TP_CONTROL_H
#define DS4_METAL_TP_CONTROL_H
#include "attention_decode.h"
/* Existing scalar policy storage moves once; transport/ARC resources remain
 * in their actual process owners. Both C fallback and native borrow this. */
typedef struct {
  uint64_t row_sequence, batch_sequence, prepublished_sequence;
  int32_t session_batch_mode;
  bool flag_gates, poll_gates;
  uint64_t out_offset, vector_bytes, slab_offset, flags_offset;
  int fold_requested;
  uint32_t fold_layer, fold_gate;
  uint64_t selected_sequence;
} ds4_metal_tp_policy;
/* Immutable borrowed resource/OS facts for one complete control action. */
typedef struct {
  bool initialized, batch, service_running, ssd, checks, shared_events,
      selected_event;
  uint64_t pending_count;
  ds4_metal_attention_view slab, checks_buffer, poll;
} ds4_metal_tp_facts;
/* The original transport queue item, not another native work queue. */
typedef struct {
  uint32_t layer, gate;
  uint32_t rows; /* zero=row gate; positive=verify/prefill batch */
  uint32_t event_arrival;
  uint64_t seq;
  /* Big gates borrow CPU-visible bounce buffers instead of slab slots. */
  const void *big_out;
  void *big_in;
  uint64_t big_bytes;
  uint32_t poll; /* release through the poll region */
} ds4_gpu_tp_request;
/* Original transport/gate constants, authoritative for both implementations. */
#define DS4_TP_BATCH_FLAG_TAG 0x80000000u
#define DS4_TP_POLL_LINES 8192u
#define DS4_TP_POLL_LINE_BYTES 128u
#define DS4_TP_POLL_MAX_SLOTS 1024u
#define DS4_TP_POLL_RING                                                       \
  16u /* cycle regions so no line is re-read while cached */
#define DS4_TP_POLL_MAX_INFLIGHT 32u

enum {
  DS4_GPU_TP_QUEUE = 1024,
  DS4_GPU_TP_GATE_ATTN = 0,
  DS4_GPU_TP_GATE_FFN = 1,
  DS4_GPU_TP_GATES_PER_LAYER = 2
};
enum {
  DS4_TP_FLAG,
  DS4_TP_CHECKED_FLAG,
  DS4_TP_POLL,
  DS4_TP_FOLD_ADD,
  DS4_TP_EVENT_ROW,
  DS4_TP_EVENT_BATCH,
  DS4_TP_EVENT_SELECTED
};
typedef struct {
  uint32_t kind, value, words, count, groups;
  uint64_t threads, shared_bytes;
  bool end_encoder;
  const ds4_metal_hc_pipeline *pipeline;
  ds4_metal_attention_view flags, checks, payload, control, a, b, out;
} ds4_metal_tp_launch;
ds4_metal_tp_policy *ds4_metal_tp_policy_state(void);
ds4_metal_tp_facts ds4_metal_tp_get_facts(void);
ds4_metal_attention_view ds4_metal_tp_fold_control(void);
int ds4_metal_tp_encode(void *command, const ds4_metal_tp_launch *);
void ds4_metal_tp_signal(uint32_t event, uint64_t sequence);
void ds4_metal_tp_wait(uint32_t event, uint64_t sequence);
bool ds4_metal_tp_selected_ready(void);
bool ds4_metal_tp_wait_selected(uint64_t sequence);
bool ds4_metal_tp_enqueue(const ds4_gpu_tp_request *);
void ds4_metal_tp_wait_pending_index(uint64_t index);
void *ds4_metal_tp_take_batch(uint32_t work_mode);
void ds4_metal_tp_signal_detached(void *command, uint64_t sequence);
void ds4_metal_tp_commit(void *command);
void ds4_metal_tp_append_pending(void *command);
bool ds4_metal_tp_command_error(void *command, const char *label);
void ds4_metal_tp_drop(void *command);
void ds4_metal_tp_clear_transients(void);
bool ds4_metal_tp_new_batch(void);
int ds4_metal_tp_prefetch(uint32_t gate);
int ds4_native_metal_tp_fold_take(const ds4_gpu_tensor *, uint64_t, uint32_t *,
                                  uint32_t *);
int ds4_native_metal_tp_signal_batch_wait(const char *);
#endif
