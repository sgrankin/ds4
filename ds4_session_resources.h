#ifndef DS4_SESSION_RESOURCES_H
#define DS4_SESSION_RESOURCES_H

#include "ds4_deepseek41_graph.h"

/* The two actual existing Session fields, embedded ONCE. This record owns
 * the graph (including its Engram descriptors, aliases and optional drafter).
 * It is not a Session mirror; the common host frontier remains its own actual
 * shared record. Only the ready owning device releases graph resources. */
typedef struct {
    ds41_gpu_graph graph;
    bool ready;
} ds4_deepseek41_session_device;

#endif
