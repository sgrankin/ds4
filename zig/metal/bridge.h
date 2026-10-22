#ifndef DS4_ZIG_METAL_BRIDGE_H
#define DS4_ZIG_METAL_BRIDGE_H
#include "bridge_types.h"

/* Shared actual index packets and fixed encoder leaves. */
#include "index_selection.h"

/* Ordinary Q8/F16 dense controller fixed bindings. */
#include "dense_matmul.h"

/* Shared actual scalar-attention packets and fixed encoder leaves. */
#include "attention_decode.h"

/* Shared actual rectangular/batch attention packets and fixed leaves. */
#include "attention_batch.h"

/* Actual routed-MoE packets and fixed ARC/resource leaves. */
#include "routed_batch.h"

/* Shared single-token packets and fixed resource leaves. */
#include "routed_single.h"

#include "rope.h"

#include "embedding.h"

#include "expert_cache.h"
#include "expert_tables.h"

#include "expert_load.h"

#include "attention_output.h"
#include "compressor.h"

#include "scheduling.h"


#include "hc_control.h"
#include "router_shared.h"
#include "router.h"
#include "primitives.h"
#include "supported_primitives.h"
#include "tp_control.h"
#include "tp_prefetch.h"

#include "bootstrap.h"

#endif
