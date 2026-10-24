//! Shared routed-expert slab sizes. Reuse the already-qualified native tensor
//! format accounting; this policy chooses the dominant local class and reserves
//! at most two cacheable layers for selected-address prefill.
const std = @import("std");
const c = @import("gguf_c");
const experts = @import("deepseek41/expert_cache.zig");
const Config = @import("config.zig").ModelConfig;
pub const rowBytes = experts.rowBytes;
pub const layerBytes = experts.layerBytes;

pub fn sizeClass(shape: *const c.ds4_shape, weights: *const c.ds4_weights) ?u64 {
    const config = Config{ .shape = shape.*, .compress_ratios = @splat(0) };
    return experts.sizeClass(&config, weights);
}
pub const Count = struct { experts: u64, layers: u32 };
pub fn cacheable(shape: *const c.ds4_shape, weights: *const c.ds4_weights) ?Count {
    if (shape.n_expert == 0) return null;
    const slab = sizeClass(shape, weights) orelse return null;
    var layers: u32 = 0;
    for (weights.layer[0..shape.n_layer]) |*layer| {
        if (layerBytes(layer)) |bytes| if (bytes == slab) {
            layers += 1;
        };
    }
    if (layers == 0) return null;
    return .{ .experts = @as(u64, layers) * shape.n_expert, .layers = layers };
}
pub fn prefillHeadroom(shape: *const c.ds4_shape, weights: *const c.ds4_weights) ?u64 {
    const slab = sizeClass(shape, weights) orelse return null;
    const counts = cacheable(shape, weights) orelse return null;
    const layers = @min(counts.layers, c.DS4_STREAMING_PREFILL_HEADROOM_LAYERS);
    const bytes = std.math.mul(u64, slab, shape.n_expert) catch return null;
    return std.math.mul(u64, bytes, layers) catch null;
}
pub fn dynamicBytes(memory: *const c.ds4_engine_memory, streaming: bool, shape: *const c.ds4_shape, weights: *const c.ds4_weights) u64 {
    if (!streaming) return 0;
    if (memory.ssd_streaming_cache_bytes != 0) return memory.ssd_streaming_cache_bytes;
    if (memory.ssd_streaming_cache_experts == 0) return 0;
    const slab = sizeClass(shape, weights) orelse return 0;
    return slab *| memory.ssd_streaming_cache_experts;
}
