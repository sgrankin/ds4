void **ds4_metal_resource_state_slot(void) { return &ds4_metal_context_current()->native_resources; }
/* True ARC/Metal leaves for the native resource owner. Included at the end of
 * ds4_metal.m; the public handle remains a retained DS4MetalTensor instance. */
#include "resources.h"

ds4_gpu_tensor *ds4_metal_resource_new(uint64_t bytes) {
    @autoreleasepool {
        DS4MetalTensor *tensor = [DS4MetalTensor new];
        tensor.buffer = [g_device newBufferWithLength:(NSUInteger)bytes
                                              options:MTLResourceStorageModeShared];
        if (!tensor.buffer) return NULL;
        tensor.offset = 0;
        tensor.bytes = bytes;
        tensor.owner = 1;
        return (__bridge_retained ds4_gpu_tensor *)tensor;
    }
}

ds4_gpu_tensor *ds4_metal_resource_view(const ds4_gpu_tensor *base,
        uint64_t absolute_offset, uint64_t bytes) {
    @autoreleasepool {
        DS4MetalTensor *view = [DS4MetalTensor new];
        view.buffer = ds4_gpu_tensor_const_obj(base).buffer;
        view.offset = absolute_offset;
        view.bytes = bytes;
        view.owner = 0;
        return (__bridge_retained ds4_gpu_tensor *)view;
    }
}

ds4_metal_tensor_region ds4_metal_resource_region(const ds4_gpu_tensor *tensor) {
    return [ds4_gpu_tensor_const_obj(tensor) ds4ResourceRegion];
}

void ds4_metal_resource_dispose(ds4_gpu_tensor *tensor) {
    @autoreleasepool {
        DS4MetalTensor *obj = (__bridge_transfer DS4MetalTensor *)tensor;
        obj.buffer = nil;
        obj.offset = 0;
        obj.bytes = 0;
        obj.owner = 0;
    }
}

void *ds4_metal_resource_contents(ds4_gpu_tensor *tensor) {
    /* NS_RETURNS_INNER_POINTER can autorelease-retain its MTLBuffer receiver.
     * Drain that temporary retain here. The tensor still owns the buffer and
     * the returned pointer is borrowed until the caller frees that tensor. */
    @autoreleasepool {
        DS4MetalTensor *obj = ds4_gpu_tensor_obj(tensor);
        return (uint8_t *)[obj.buffer contents] + obj.offset;
    }
}

int ds4_metal_resource_blit(ds4_gpu_tensor *dst, uint64_t absolute_dst,
        const ds4_gpu_tensor *src, uint64_t absolute_src, uint64_t bytes) {
    if (!g_batch_cb) return 0;
    /* C callers have no outer pool. Validation-layer blit bookkeeping must not
     * accumulate across tokens, and the caller's command ownership is intact. */
    @autoreleasepool {
        ds4_gpu_close_batch_encoder();
        g_batch_has_work = YES;
        id<MTLBlitCommandEncoder> blit = ds4_gpu_blit_encoder(g_batch_cb, "tensor_copy", 1u);
        if (!blit) return 0;
        [blit copyFromBuffer:ds4_gpu_tensor_const_obj(src).buffer
                sourceOffset:(NSUInteger)absolute_src
                    toBuffer:ds4_gpu_tensor_obj(dst).buffer
           destinationOffset:(NSUInteger)absolute_dst
                        size:(NSUInteger)bytes];
        [blit endEncoding];
    }
    return 1;
}
