/* True ARC/driver leaves for native model-view policy. The ONE shared state
 * owns strong buffer fields; native code changes only metadata and ordering.
 * A local pool drains transient labels without releasing installed views. */
#include "model_views.h"

uint64_t ds4_metal_model_view_max_buffer_length(void) {
    @autoreleasepool {
        return (uint64_t)[g_device maxBufferLength];
    }
}

int ds4_metal_model_view_install(ds4_gpu_model_view *view,
        const void *address, uint64_t bytes, uint32_t index) {
    @autoreleasepool {
        id<MTLBuffer> buffer = [g_device newBufferWithBytesNoCopy:(void *)address
                                                       length:(NSUInteger)bytes
                                                      options:ds4_gpu_model_resource_options()
                                                  deallocator:nil];
        if (!buffer) return 0;
        buffer.label = [NSString stringWithFormat:@"ds4_model_view_%u", index];
        view->buffer = buffer;
        return 1;
    }
}

void ds4_metal_model_view_retire(ds4_gpu_model_view *view) {
    @autoreleasepool {
        view->buffer = nil;
        view->model_map = NULL;
        view->model_size = 0;
        view->model_offset = 0;
        view->bytes = 0;
    }
}

void ds4_metal_model_view_move(ds4_gpu_model_view *destination,
        const ds4_gpu_model_view *source) {
    @autoreleasepool {
        /* ARC struct assignment retains the kept view before the owner retires
         * duplicate trailing entries. Do not memcpy a strong buffer field. */
        *destination = *source;
    }
}
