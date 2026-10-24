//! Cold size accounting entry. Model-free fixtures must not pull hot startup,
//! admission device queries or the serving Engine into their link.
const c = @import("gguf_c");
const sizes = @import("engine_cache_sizes.zig");
export fn ds4_native_cache_per_expert(shape: *const c.ds4_shape, weights: *const c.ds4_weights, output: *u64) bool {
    output.* = sizes.sizeClass(shape, weights) orelse 0;
    return output.* != 0;
}
export fn ds4_native_cacheable_count(shape: *const c.ds4_shape, weights: *const c.ds4_weights, output: *u64, layers: [*c]u32) bool {
    output.* = 0;
    if (layers != null) layers.* = 0;
    const count = sizes.cacheable(shape, weights) orelse return false;
    output.* = count.experts;
    if (layers != null) layers.* = count.layers;
    return true;
}
export fn ds4_native_cache_prefill_headroom(shape: *const c.ds4_shape, weights: *const c.ds4_weights, output: *u64) bool {
    output.* = 0;
    output.* = sizes.prefillHeadroom(shape, weights) orelse return false;
    return true;
}
export fn ds4_native_cache_dynamic_bytes(memory: *const c.ds4_engine_memory, streaming: bool, shape: *const c.ds4_shape, weights: *const c.ds4_weights) u64 {
    return sizes.dynamicBytes(memory, streaming, shape, weights);
}
