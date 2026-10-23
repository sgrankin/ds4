#ifndef DS4_ENGINE_TP_STATE_H
#define DS4_ENGINE_TP_STATE_H

#include "ds4_tp.h"
#include "ds4_gpu_tp.h"
#include <stdbool.h>

/* Engine-side tensor-parallel state.  The transport context is owned by the
 * frontend (CLI leader or ds4_tp_worker_run); the engine owns the GPU slab,
 * the per-slot views and the gate machinery lifetime. */
typedef struct {
    struct ds4_tp *ctx;
    ds4_gpu_tensor *slab;
    void *host_slab;             /* CUDA's registered network staging */
    ds4_gpu_tensor **out_views;
    ds4_gpu_tensor **in_views;
    ds4_gpu_tensor **batch_out_views;   /* [layer] verify-block row partials */
    ds4_gpu_tensor **batch_in_views;
    ds4_gpu_tensor *zero_vec;
    uint64_t eval_seq;          /* leader: mirrored eval counter */
    uint64_t next_session_id;   /* leader: stable worker-session handle */
    int rank;
    bool vocab_split;           /* DeepSeek: logits halves cross the wire */
    bool active;
    /* Lockstep TP drafter: this rank built its DSpark expert halves at open
     * (dspark_split_built); both ranks agreed at bind (dspark_split); repeated
     * state disagreement between the ranks switched it off (dspark_split_off). */
    bool dspark_split_built;
    bool dspark_split;
    bool dspark_split_off;
    uint32_t dspark_preflight_mismatches;
    uint64_t dspark_split_bytes; /* this rank's drafter expert halves */
} ds4_engine_tp_state;

#endif
