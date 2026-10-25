#ifndef DS4_METAL_CONTEXT_H
#define DS4_METAL_CONTEXT_H
#include "engine/host_context.h"
/* One native GGUF engine owns this opaque Metal state. Selection is scoped
 * at API and worker entry; it does not transfer ownership. Calls sharing one
 * engine retain their existing encoding/scratch serialization contract. */
typedef struct ds4_metal_context ds4_metal_context;
ds4_metal_context *ds4_metal_context_create(void);
void ds4_metal_context_destroy(ds4_metal_context *context);
ds4_metal_context *ds4_metal_context_select(ds4_metal_context *context);
ds4_metal_context *ds4_metal_context_selected(void);
#endif
