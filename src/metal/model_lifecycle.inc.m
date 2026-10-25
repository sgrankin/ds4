/* Fixed resource leaves. Native code owns map/replacement/publication policy,
 * residency gates, warmup stride and each range's immutable dispatch plan. */
void ds4_metal_model_view_residency_clear(ds4_metal_model_views *state) {
    @autoreleasepool {
#if TARGET_OS_OSX
        if (@available(macOS 15.0, *)) {
            if (state->residency_set) {
                if (state->residency_added_to_queue && g_queue &&
                    [g_queue respondsToSelector:@selector(removeResidencySet:)])
                    [g_queue removeResidencySet:state->residency_set];
                [state->residency_set endResidency];
                [state->residency_set removeAllAllocations];
                state->residency_set = nil;
            }
        }
#endif
        state->residency_count = 0;
        state->residency_added_to_queue = 0;
    }
}

int ds4_metal_model_view_residency_install(ds4_metal_model_views *state,
                                           bool attach_queue) {
    @autoreleasepool {
#if TARGET_OS_OSX
        if (@available(macOS 15.0, *)) {
            /* A budgeting hint, not a request to fault the whole model. */
            MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
            desc.label = @"ds4_model";
            desc.initialCapacity = state->count;
            NSError *error = nil;
            state->residency_set = [g_device newResidencySetWithDescriptor:desc error:&error];
            if (!state->residency_set) {
                fprintf(stderr, "ds4: Metal model residency set creation failed: %s\n",
                        [[error localizedDescription] UTF8String]);
                return 0;
            }
            for (uint32_t i = 0; i < state->count; i++)
                [state->residency_set addAllocation:state->entries[i].buffer];
            [state->residency_set commit];
            [state->residency_set requestResidency];
            if (attach_queue && g_queue && [g_queue respondsToSelector:@selector(addResidencySet:)]) {
                [g_queue addResidencySet:state->residency_set];
                state->residency_added_to_queue = 1;
            }
            state->residency_count = state->count;
        }
#endif
        return 1;
    }
}

const ds4_metal_model_warm_pipeline *ds4_metal_model_view_warm_pipeline(void) {
    @autoreleasepool {
        return (__bridge const ds4_metal_model_warm_pipeline *)
            ds4_gpu_get_pipeline("kernel_touch_u8_stride");
    }
}
int ds4_metal_model_view_warm_encode(const ds4_metal_model_views *state,
        const ds4_metal_model_warm_pipeline *pipeline,
        const ds4_metal_model_warm_range *ranges, uint64_t total, uint64_t stride) {
    @autoreleasepool {
        id<MTLBuffer> out = [g_device newBufferWithLength:(NSUInteger)total
                                                options:MTLResourceStorageModeShared];
        if (!out) {
            fprintf(stderr, "ds4: Metal model warmup scratch allocation failed\n");
            return 0;
        }
        out.label = @"ds4_model_warmup";
        id<MTLCommandBuffer> cb = ds4_gpu_new_command_buffer();
        if (!cb) {
            fprintf(stderr, "ds4: Metal model warmup command buffer allocation failed\n");
            return 0;
        }
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pipeline];
        for (uint32_t i = 0; i < state->count; i++) {
            [enc setBuffer:state->entries[i].buffer offset:0 atIndex:0];
            [enc setBuffer:out offset:0 atIndex:1];
            [enc setBytes:&stride length:sizeof(stride) atIndex:2];
            [enc setBytes:&ranges[i].bytes length:sizeof(ranges[i].bytes) atIndex:3];
            [enc setBytes:&ranges[i].destination length:sizeof(ranges[i].destination) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(ranges[i].groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        ds4_gpu_end_compute_encoder(cb, enc);
        ds4_runtime_metal_commit((__bridge void *)cb, &g_bootstrap, false);
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "ds4: Metal model warmup failed: %s\n",
                    [[cb.error localizedDescription] UTF8String]);
            return 0;
        }
        return 1;
    }
}
double ds4_metal_model_view_now_ms(void) { return ds4_gpu_now_ms(); }
void ds4_metal_model_view_keepalive(void) { ds4_gpu_queue_keepalive_start(); }
void ds4_metal_model_view_progress(uint32_t stage, const char *text) {
    if (stage == 0) ds4_gpu_progress_begin(text);
    else if (stage == 1) ds4_gpu_progress_done();
    else ds4_gpu_progress_failed();
}
