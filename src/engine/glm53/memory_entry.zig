//! CPU-safe model geometry ABI. These exports do not import graph execution,
//! allocator/device services or the hot-model entry point.
const c = @import("gguf_c");
const Planning = @import("planning.zig");
const Memory = @import("memory.zig");

export fn ds4_native_glm_full_attention_cap(context: u32, streaming: bool) u32 {
    return Planning.fullAttentionCap(context, streaming);
}
export fn ds4_native_glm_context_request(shape: *const c.ds4_shape, context: c_int, output: *u32) bool {
    return Planning.contextRequest(shape, context, output);
}
export fn ds4_native_glm_indexed_chunk(compact_capacity: u32) u32 {
    return Planning.indexedChunk(compact_capacity);
}
export fn ds4_native_glm_score_rows(indexed_rows: u32, compact_capacity: u32) u32 {
    return Planning.scoreRows(indexed_rows, compact_capacity);
}
export fn ds4_native_glm_batch_cap(work_capacity: u32, indexed_rows: u32) u32 {
    return Planning.batchCap(work_capacity, indexed_rows);
}
export fn ds4_native_glm_compact_bytes(shape: *const c.ds4_shape, attention_layers: u32, indexer_layers: u32, capacity: u32) u64 {
    const config = Memory.geometry(shape);
    return Memory.compactBytes(&config, attention_layers, indexer_layers, capacity);
}
export fn ds4_native_glm_indexed_scratch_bytes(shape: *const c.ds4_shape, work_capacity: u32, compact_capacity: u32) u64 {
    const config = Memory.geometry(shape);
    return Memory.indexedScratch(&config, work_capacity, compact_capacity);
}
export fn ds4_native_glm_workspace_bytes(shape: *const c.ds4_shape, work_capacity: u32, compact_capacity: u32) u64 {
    const config = Memory.geometry(shape);
    return Memory.workspace(&config, work_capacity, compact_capacity);
}
export fn ds4_native_glm_fixed_state_bytes(shape: *const c.ds4_shape, first: u32, last: u32) u64 {
    const config = Memory.geometry(shape);
    return Memory.fixedState(&config, first, last);
}
export fn ds4_native_glm_indexer_layers(shape: *const c.ds4_shape, first: u32, last: u32) u32 {
    const config = Memory.geometry(shape);
    return Memory.indexerLayers(&config, first, last);
}
export fn ds4_native_glm_memory(shape: *const c.ds4_shape, context: u32, work_capacity: u32, compact_capacity: u32, first: u32, last: u32) c.ds4_context_memory {
    const config = Memory.geometry(shape);
    return Memory.estimate(&config, context, work_capacity, compact_capacity, first, last);
}
