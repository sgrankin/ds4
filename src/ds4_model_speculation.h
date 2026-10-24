#ifndef DS4_MODEL_SPECULATION_H
#define DS4_MODEL_SPECULATION_H

#include <stdbool.h>
#include <stdint.h>

/* Actual pending/rollback state shared by GLM and Qwen model controllers.
 * Embedded once in Session. Host buffers keep their existing libc ownership;
 * draft parent identifies the boundary token that conditioned the proposal. */
typedef struct {
    int draft;
    int draft2;
    bool have2;
    int parent;
    int have;
    bool rollback_valid;
    uint32_t rollback_pos;
    uint32_t rollback_dense_len;
    int rollback_first_token;
    int inside;
    uint32_t min_pos;
    float *hc;
    float *logits0;
} ds4_model_speculation_state;

#endif
