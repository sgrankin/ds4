const c = @import("exact_cache_c");
const cache = @import("exact_cache.zig");
export fn ds4_metal_exact_cache_init() void {
    cache.init();
}
export fn ds4_metal_exact_cache_cleanup() void {
    cache.cleanup();
}
export fn ds4_metal_exact_cache_clear(reason: [*c]const u8) void {
    cache.clear(reason);
}
export fn ds4_metal_exact_cache_maybe_evict(reason: [*c]const u8) void {
    cache.maybeEvict(reason);
}
export fn ds4_metal_exact_cache_report() void {
    cache.report();
}
export fn ds4_metal_exact_cache_wrap(model: ?*const anyopaque, model_size: u64, offset: u64, bytes: u64, inner_offset: [*c]u64, lifetime: c.ds4_gpu_exact_view_lifetime) ?*anyopaque {
    return cache.wrap(model, model_size, offset, bytes, inner_offset, lifetime);
}
