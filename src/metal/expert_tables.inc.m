/* Fixed ARC and Metal leaves. Native owns table construction policy and every
 * expert/group iteration. These methods touch the existing table objects and
 * actual strong cache fields only; borrowed results keep their original owner.
 */
#include "expert_tables.h"
void ds4_metal_tables_facts(ds4_metal_table_facts *out) {
    @autoreleasepool {
        memset(out, 0, sizeof(*out));
        out->device = g_device != nil;
        out->table_cache = g_q4_expert_table_cache != nil;
        out->layer_cache = g_q4_expert_layer_residency_cache != nil;
        out->streaming = g_ssd_streaming_mode;
        out->glm = g_glm_model_mode;
        out->address_pipelines =
            g_moe_mul_mv_addr_q4_k_pair_swiglu_pipeline && g_moe_mul_mv_addr_q4_k_sum6_pipeline;
        out->encoders[0] = g_moe_table_q4_pair_gate_encoder;
        out->encoders[1] = g_moe_table_q4_pair_up_encoder;
        out->encoders[2] = g_moe_table_q4_sum_down_encoder;
#if TARGET_OS_OSX
        if (@available(macOS 13.0, *))
            out->gpu_addresses = true;
        if (@available(macOS 15.0, *)) {
            out->residency_sets = true;
            out->queue_residency =
                g_queue && [g_queue respondsToSelector:@selector(addResidencySet:)];
        }
#endif
    }
}
ds4_metal_expert_buffer_ref ds4_metal_tables_ensure_buffer(ds4_metal_expert_cache_state *s,
                                                           uint32_t kind, uint32_t layer,
                                                           uint64_t bytes, bool zero,
                                                           const char *label) {
    @autoreleasepool {
        id<MTLBuffer> __strong *slot = NULL;
        switch (kind) {
        case DS4_TABLE_CACHE_GATE:
            slot = &s->cache_gate_addr_buffers[layer];
            break;
        case DS4_TABLE_CACHE_UP:
            slot = &s->cache_up_addr_buffers[layer];
            break;
        case DS4_TABLE_CACHE_DOWN:
            slot = &s->cache_down_addr_buffers[layer];
            break;
        case DS4_TABLE_COMPACT_GATE:
            slot = &s->compact_gate_addr_buffers[layer];
            break;
        case DS4_TABLE_COMPACT_UP:
            slot = &s->compact_up_addr_buffers[layer];
            break;
        case DS4_TABLE_COMPACT_DOWN:
            slot = &s->compact_down_addr_buffers[layer];
            break;
        case DS4_TABLE_COMPACT_IDS:
            slot = &s->compact_selected_buffers[layer];
            break;
        case DS4_TABLE_HOST_IDS:
            slot = &s->selected_id_buffers[layer];
            break;
        }
        if (*slot)
            return *slot;
        id<MTLBuffer> buffer = [g_device newBufferWithLength:(NSUInteger)bytes
                                                     options:MTLResourceStorageModeShared];
        if (!buffer)
            return nil;
        buffer.label = [NSString stringWithUTF8String:label];
        if (zero)
            memset([buffer contents], 0, (NSUInteger)bytes);
        *slot = buffer;
        return *slot;
    }
}
uint64_t ds4_metal_tables_address(ds4_metal_expert_buffer_ref buffer, uintptr_t inner) {
    return ds4_gpu_buffer_address(buffer, (NSUInteger)inner);
}
ds4_metal_table_wrap ds4_metal_tables_wrap(const void *map, uint64_t size, uint64_t off,
                                           uint64_t bytes, bool exact) {
    @autoreleasepool {
        uint64_t inner = 0;
        id<MTLBuffer> buffer =
            exact ? ds4_gpu_wrap_model_exact_range_owned(map, size, off, bytes, &inner)
                  : ds4_gpu_wrap_model_range(map, size, off, bytes, &inner);
        /* Returning an explicit +1 token is independent of this pool. Native
         * releases it only after the actual entry/table has retained it. */
        return (ds4_metal_table_wrap){(__bridge id<MTLBuffer>)(__bridge_retained void *)buffer,
                                      inner};
    }
}
void ds4_metal_tables_release_buffer(ds4_metal_expert_buffer_ref buffer) {
    @autoreleasepool {
        id held = (__bridge_transfer id)(__bridge void *)buffer;
        (void)held;
    }
}
uint64_t ds4_metal_tables_encoded_bytes(ds4_metal_argument_ref encoder) {
    return [encoder encodedLength];
}
static NSString *ds4_metal_table_key_string(const ds4_metal_table_key *key) {
    if (key->address)
        return [NSString
            stringWithFormat:@"addr:%p:%llu:%llu:%llu:%u:%u", key->model,
                             (unsigned long long)key->model_bytes, (unsigned long long)key->offset,
                             (unsigned long long)key->expert_bytes, key->experts, key->group];
    return [NSString stringWithFormat:@"%p:%llu:%llu:%llu:%u:%llu:%u", key->model,
                                      (unsigned long long)key->model_bytes,
                                      (unsigned long long)key->offset,
                                      (unsigned long long)key->expert_bytes, key->experts,
                                      (unsigned long long)key->encoded_bytes, key->group];
}
ds4_metal_table_ref ds4_metal_tables_lookup(const ds4_metal_table_key *key) {
    @autoreleasepool {
        return [g_q4_expert_table_cache objectForKey:ds4_metal_table_key_string(key)];
    }
}
ds4_metal_table_ref ds4_metal_tables_create(const ds4_metal_table_key *key,
                                            ds4_metal_argument_ref encoder, uint32_t capacity) {
    @autoreleasepool {
        DS4MetalQ4ExpertTable *table = [DS4MetalQ4ExpertTable new];
        uint64_t size =
            key->address ? (uint64_t)key->experts * sizeof(uint64_t) : key->encoded_bytes;
        id<MTLBuffer> buffer = [g_device newBufferWithLength:(NSUInteger)size
                                                     options:MTLResourceStorageModeShared];
        if (!buffer) {
            fprintf(stderr, "%s\n",
                    key->address ? "ds4: Metal Q4 expert address table allocation failed"
                                 : "ds4: Metal Q4 expert table argument buffer allocation failed");
            return nil;
        }
        buffer.label = key->address ? @"ds4_q4_expert_address_table" : @"ds4_q4_expert_table";
        if (key->address)
            table.addressBuffer = buffer;
        else {
            table.argumentBuffer = buffer;
            [encoder setArgumentBuffer:buffer offset:0];
        }
        table.expertBuffers = [NSMutableArray arrayWithCapacity:capacity];
        if (!table.expertBuffers)
            return nil;
        table.nExpert = key->experts;
        table.expertBytes = key->expert_bytes;
        return (__bridge DS4MetalQ4ExpertTable *)(__bridge_retained void *)table;
    }
}
void ds4_metal_tables_drop(ds4_metal_table_ref table) {
    @autoreleasepool {
        id held = (__bridge_transfer id)(__bridge void *)table;
        (void)held;
    }
}
void ds4_metal_tables_publish(const ds4_metal_table_key *key, ds4_metal_table_ref table) {
    @autoreleasepool {
        [g_q4_expert_table_cache setObject:table forKey:ds4_metal_table_key_string(key)];
    }
}
ds4_metal_table_regions ds4_metal_tables_regions(ds4_metal_table_ref table) {
    return (ds4_metal_table_regions){table.argumentBuffer, table.addressBuffer, table.residencySet,
                                     (uint32_t)[table.expertBuffers count],
                                     table.expertBuffers != nil};
}
void *ds4_metal_tables_address_bytes(ds4_metal_table_ref table) {
    return [table.addressBuffer contents];
}
void ds4_metal_tables_append(ds4_metal_table_ref table, ds4_metal_expert_buffer_ref buffer) {
    [table.expertBuffers addObject:buffer];
}
void ds4_metal_tables_argument_buffer(ds4_metal_argument_ref encoder,
                                      ds4_metal_expert_buffer_ref buffer, uint64_t offset,
                                      uint32_t index) {
    [encoder setBuffer:buffer offset:(NSUInteger)offset atIndex:index];
}
uint32_t ds4_metal_tables_buffers(ds4_metal_table_ref table,
                                  ds4_metal_expert_buffer_ref __unsafe_unretained *out) {
    /* One borrowed region snapshot avoids per-resource property callbacks.
     * The actual table remains strongly owned throughout the synchronous call. */
    uint32_t count = (uint32_t)[table.expertBuffers count];
    for (uint32_t i = 0; i < count; i++)
        out[i] = [table.expertBuffers objectAtIndex:i];
    return count;
}
ds4_metal_residency_ref ds4_metal_tables_residency_create(uint32_t capacity, bool layer) {
    @autoreleasepool {
#if TARGET_OS_OSX
        if (@available(macOS 15.0, *)) {
            MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
            desc.label = layer ? @"ds4_q4_expert_layer" : @"ds4_q4_expert_table";
            desc.initialCapacity = capacity;
            NSError *error = nil;
            id residency = [g_device newResidencySetWithDescriptor:desc error:&error];
            if (!residency) {
                fprintf(stderr, "%s: %s\n",
                        layer ? "ds4: Metal Q4 expert layer residency set creation failed"
                              : "ds4: Metal Q4 expert table residency set creation failed",
                        [[error localizedDescription] UTF8String]);
                return nil;
            }
            return (__bridge id)(__bridge_retained void *)residency;
        }
#endif
        return nil;
    }
}
void ds4_metal_tables_residency_add(ds4_metal_residency_ref residency,
                                    ds4_metal_expert_buffer_ref buffer) {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *))
        [residency addAllocation:buffer];
#endif
}
void ds4_metal_tables_residency_commit(ds4_metal_residency_ref residency) {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        [residency commit];
        [residency requestResidency];
    }
#endif
}
void ds4_metal_tables_residency_drop(ds4_metal_residency_ref residency) {
    @autoreleasepool {
        id held = (__bridge_transfer id)(__bridge void *)residency;
        (void)held;
    }
}
void ds4_metal_tables_tensor_residency(ds4_metal_table_ref table,
                                       ds4_metal_residency_ref residency) {
    table.residencySet = residency;
}
static NSString *ds4_metal_table_layer_key(ds4_metal_table_ref gate, ds4_metal_table_ref up,
                                           ds4_metal_table_ref down) {
    return [NSString stringWithFormat:@"%p:%p:%p", gate, up, down];
}
ds4_metal_residency_ref ds4_metal_tables_layer_lookup(ds4_metal_table_ref gate,
                                                      ds4_metal_table_ref up,
                                                      ds4_metal_table_ref down) {
    @autoreleasepool {
        return ((DS4MetalQ4LayerResidency *)[g_q4_expert_layer_residency_cache
                    objectForKey:ds4_metal_table_layer_key(gate, up, down)])
            .residencySet;
    }
}
void ds4_metal_tables_layer_publish(ds4_metal_table_ref gate, ds4_metal_table_ref up,
                                    ds4_metal_table_ref down, ds4_metal_residency_ref residency,
                                    bool queue) {
    @autoreleasepool {
        DS4MetalQ4LayerResidency *entry = [DS4MetalQ4LayerResidency new];
        entry.residencySet = residency;
#if TARGET_OS_OSX
        if (@available(macOS 15.0, *)) {
            if (queue) {
                [g_queue addResidencySet:residency];
                entry.addedToQueue = YES;
            }
        }
#endif
        [g_q4_expert_layer_residency_cache setObject:entry
                                              forKey:ds4_metal_table_layer_key(gate, up, down)];
    }
}
bool ds4_metal_tables_use_model(ds4_metal_table_command_ref command) {
#if TARGET_OS_OSX
    if (@available(macOS 15.0, *)) {
        if (command && g_model_views.residency_set &&
            [command respondsToSelector:@selector(useResidencySet:)]) {
            [command useResidencySet:g_model_views.residency_set];
            return true;
        }
    }
#endif
    return false;
}
void ds4_metal_tables_use_residency(ds4_metal_table_command_ref command,
                                    ds4_metal_residency_ref residency) {
    if ([command respondsToSelector:@selector(useResidencySet:)])
        [command useResidencySet:residency];
}
void ds4_metal_tables_use_buffer(ds4_metal_table_encoder_ref encoder,
                                 ds4_metal_expert_buffer_ref buffer) {
    [encoder useResource:buffer usage:MTLResourceUsageRead];
}
void ds4_metal_tables_use_buffers(ds4_metal_table_encoder_ref encoder,
                                  const ds4_metal_expert_buffer_ref __unsafe_unretained *buffers,
                                  uint32_t count) {
    __unsafe_unretained id<MTLResource> resources[64];
    for (uint32_t i = 0; i < count; i++)
        resources[i] = buffers[i];
    [encoder useResources:resources count:count usage:MTLResourceUsageRead];
}
void ds4_metal_tables_bind_buffer(ds4_metal_table_encoder_ref encoder,
                                  ds4_metal_expert_buffer_ref buffer, uintptr_t index) {
    [encoder setBuffer:buffer offset:0 atIndex:(NSUInteger)index];
}
