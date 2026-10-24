#ifndef DS4_ENGINE_CONFIGURATION_H
#define DS4_ENGINE_CONFIGURATION_H
#include "ds4_model_config.h"
#include "ds4_qwen4_graph.h"
#include "ds4_glm_graph.h"

/* The selected immutable configuration, embedded once in the actual Core.
 * Every model configuration begins with Shape. Its family/variant identifies
 * the active union member; graph owners capture immutable values from it.
 * Legacy global publication is a compatibility snapshot, never an owner. */
typedef union {
    ds4_shape shape;
    ds4_model_config ds41;
    ds4_qwen4_config qwen;
    ds4_glm_model_config glm;
} ds4_engine_configuration;
#endif
