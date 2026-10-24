#ifndef DS4_STREAMING_POLICY_H
#define DS4_STREAMING_POLICY_H
#include <stdbool.h>
#include <stdint.h>

/* The existing process-local GLM decode-map policy, stored once. Engine
 * admission temporarily varies resident_layers while measuring candidates;
 * model span selection and execution read this same published record. */
typedef struct {
    uint32_t resident_start;
    uint32_t resident_layers;
    bool static_decode_map;
} ds4_streaming_policy;
#endif
