#ifndef DS4_SESSION_CONTROL_H
#define DS4_SESSION_CONTROL_H
#include "ds4.h"

/* Actual existing request/transport/callback fields, embedded ONCE in Session.
 * Other-family C fallback and the native DS41 coordinator share this storage.
 * Callback user data is borrowed only while its synchronous operation runs.
 * No Engine, graph, CPU backend or Session layout is repeated here. */
typedef struct {
    uint64_t tp_session_id;
    int dspark_rewind_end;
    uint32_t dspark_draft_len;
    uint32_t dspark_sched_cycles;
    uint32_t dspark_sched_accepted;
    uint32_t dspark_sched_no_draft;
    uint32_t dspark_sched_skip;
    uint32_t dspark_sched_lifetime_accepted;
    double dspark_sched_life_extra_ms;
    double dspark_sched_life_saved_ms;
    double dspark_sched_extra_ms;
    double dspark_sched_saved_ms;
    bool dspark_draft_valid;
    /* The current proposal came from a lockstep TP DRAFT on both ranks. */
    bool dspark_tp_proposal;
    bool dspark_sched_skipped_cycle;
    bool dspark_sched_long_accept_seen;
    bool dspark_sched_bypass;
    ds4_session_progress_fn progress;
    void *progress_ud;
    ds4_session_progress_fn display_progress;
    void *display_progress_ud;
    ds4_session_cancel_fn cancel;
    void *cancel_ud;
} ds4_session_control;
#endif
