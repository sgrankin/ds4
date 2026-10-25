void **ds4_metal_exact_cache_state_slot(void) { return &ds4_metal_context_current()->native_exact_cache; }
/* Only driver allocation/ARC reference operations live here. Exact cache keys,
 * admission, retirement policy and accounting belong to the native owner. */
void *ds4_metal_exact_view_create(const void *address, uint64_t bytes,
        ds4_gpu_exact_view_lifetime lifetime) {
    @autoreleasepool {
        id<MTLBuffer> buffer = [g_device newBufferWithBytesNoCopy:(void *)address
            length:(NSUInteger)bytes options:ds4_gpu_model_resource_options()
            deallocator:nil];
        if (!buffer) return NULL;
        buffer.label = lifetime == DS4_GPU_EXACT_VIEW_CACHED ? @"ds4_model_exact_view" :
            lifetime == DS4_GPU_EXACT_VIEW_TRANSIENT ? @"ds4_model_exact_transient_view" :
            @"ds4_model_exact_owned_view";
        if (lifetime == DS4_GPU_EXACT_VIEW_TRANSIENT)
            [g_transient_buffers addObject:buffer];
        return (__bridge_retained void *)buffer;
    }
}
void ds4_metal_exact_view_release(void *handle) {
    @autoreleasepool {
        id<MTLBuffer> buffer = (__bridge_transfer id<MTLBuffer>)handle;
        (void)buffer;
    }
}
