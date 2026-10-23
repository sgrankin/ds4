//! Sole C facade; ContextRuntime has no exports or duplicate owner objects.
const c = @import("agent_runtime_c");
const Runtime = @import("context_runtime.zig").Runtime;
fn runtime(worker: *c.agent_worker) Runtime {
    return .{ .worker = worker };
}
fn errors(pointer: [*c]u8, length: usize) []u8 {
    return if (pointer == null or length == 0) &.{} else pointer[0..length];
}
export fn ds4_agent_runtime_images_clear(worker: ?*c.agent_worker) void {
    if (worker) |value| runtime(value).clearImages();
}
export fn ds4_agent_runtime_images_append(worker: *c.agent_worker, images: [*c]c.ds4_vision_span, count: usize) void {
    runtime(worker).appendImages(images, count);
}
export fn ds4_agent_runtime_images_fit(worker: ?*const c.agent_worker, tokens: [*c]const c.ds4_tokens) bool {
    if (worker == null or tokens == null) return false;
    return runtime(@constCast(worker.?)).imagesFit(@ptrCast(tokens));
}
export fn ds4_agent_runtime_cache_allowed(worker: *c.agent_worker) bool {
    return runtime(worker).cacheAllowed();
}
export fn ds4_agent_runtime_cache_materialize(worker: *c.agent_worker, reference: ?*c.ds4_state_cache_ref, output: *c.ds4_suffix_cache, kind: [*c]c.ds4_state_cache_kind, error_text: [*c]u8, capacity: usize) bool {
    return runtime(worker).materialize(reference, output, kind, errors(error_text, capacity));
}
export fn ds4_agent_runtime_cache_put(worker: *c.agent_worker, source: *const c.ds4_suffix_cache, key: [*c]u8) bool {
    return runtime(worker).putSource(source, key);
}
export fn ds4_agent_runtime_cache_endpoint(worker: *c.agent_worker) void {
    runtime(worker).storeEndpoint();
}
export fn ds4_agent_runtime_cache_restore(worker: *c.agent_worker, tokens: *const c.ds4_tokens) bool {
    return runtime(worker).restoreCachePrefix(tokens);
}
export fn ds4_agent_runtime_compact_prefix(worker: *c.agent_worker, system: *const c.ds4_tokens) bool {
    return runtime(worker).restoreCompactPrefix(system);
}
export fn ds4_agent_runtime_frontier_drop(worker: *c.agent_worker, index: c_int) void {
    runtime(worker).dropFrontier(@intCast(index));
}
export fn ds4_agent_runtime_frontiers_clear(worker: *c.agent_worker) void {
    runtime(worker).clearFrontiers();
}
export fn ds4_agent_runtime_frontier_slots(worker: *c.agent_worker, limit: *u64) c_int {
    return runtime(worker).frontierSlots(limit);
}
export fn ds4_agent_runtime_capture(worker: *c.agent_worker) bool {
    return runtime(worker).captureFrontier();
}
export fn ds4_agent_runtime_restore(worker: *c.agent_worker, tokens: *const c.ds4_tokens) bool {
    return runtime(worker).restoreFrontier(tokens);
}
export fn ds4_agent_runtime_context_clear(worker: *c.agent_worker) void {
    runtime(worker).clearContext();
}
export fn ds4_agent_runtime_render(worker: *c.agent_worker, length: [*c]usize) [*c]u8 {
    return runtime(worker).render(length);
}
export fn ds4_agent_runtime_export(worker: *c.agent_worker, error_text: [*c]u8, capacity: usize) bool {
    return runtime(worker).exportContext(errors(error_text, capacity));
}
export fn ds4_agent_runtime_try_suffix(worker: *c.agent_worker, edited: *const c.ds4_tokens, system: *const c.ds4_tokens, error_text: [*c]u8, capacity: usize) c_int {
    return runtime(worker).trySuffix(edited, system, errors(error_text, capacity));
}
export fn ds4_agent_runtime_import(worker: *c.agent_worker, error_text: [*c]u8, capacity: usize) c_int {
    return runtime(worker).importContext(errors(error_text, capacity));
}
