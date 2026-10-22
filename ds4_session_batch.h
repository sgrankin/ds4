#ifndef DS4_SESSION_BATCH_H
#define DS4_SESSION_BATCH_H
#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_session_frontier.h"
#include "ds4_session_resources.h"
#include "ds4_session_control.h"
#include "ds4_dspark_cycle.h"

/* One synchronous borrowed row, gathered from the actual Session. Identities
 * are used only for admission; all mutable state remains in its actual owner.
 * A NULL Session has NULL owner pointers. A non-NULL Session has all three
 * owner pointers, even when its graph is not ready. No view owns or copies KV,
 * callbacks, transport state or a frontier. */
typedef struct {
    ds4_session *session;
    ds4_engine *engine;
    ds4_dist_session *distributed;
    ds4_session_frontier *frontier;
    ds4_deepseek41_session_device *device;
    ds4_session_control *control;
    ds4_dspark_spec_stats *statistics;
    int token;
} ds4_session_batch_view;
#endif
