//! Hot common-session batch ABI; actual views/pool/policy remain caller-owned.
const c = @import("gguf_c");
const MetalScope = @import("metal_scope.zig").Scope(c);
const Batch = @import("session_batch.zig");

fn errors(pointer: [*c]u8, capacity: usize) []u8 {
    return if (capacity == 0) &.{} else pointer[0..capacity];
}
fn invalid(message: [*c]u8, capacity: usize, text: [:0]const u8) c_int {
    if (capacity != 0) _ = c.snprintf(message, capacity, "%s", text.ptr);
    return 1;
}
export fn ds4_native_sessions_eval_batch(views: [*c]c.ds4_session_view, tokens: [*c]const c_int, count: c_int, pool: [*c]c.ds4_session_resource_pool, message: [*c]u8, capacity: usize) c_int {
    if (count <= 0 or views == null or tokens == null or pool == null) return invalid(message, capacity, "empty decode batch");
    const metal_scope = MetalScope.enter(views[0].core);
    defer metal_scope.leave();
    return Batch.eval(views[0..@intCast(count)], tokens[0..@intCast(count)], pool, errors(message, capacity));
}
export fn ds4_native_sessions_speculative_batch(views: [*c]c.ds4_session_view, tokens: [*c]const c_int, count: c_int, pool: [*c]c.ds4_session_resource_pool, policy: [*c]c.ds4_qwen4_batch_policy, accepted: [*c][2]c_int, counts: [*c]c_int, message: [*c]u8, capacity: usize) c_int {
    if (count <= 0 or views == null or tokens == null or pool == null or policy == null or accepted == null or counts == null) return invalid(message, capacity, "empty speculative decode batch");
    const metal_scope = MetalScope.enter(views[0].core);
    defer metal_scope.leave();
    const length: usize = @intCast(count);
    return Batch.speculate(views[0..length], tokens[0..length], pool, policy, accepted[0..length], counts[0..length], errors(message, capacity));
}
export fn ds4_native_sessions_mixed_batch(views: [*c]c.ds4_session_view, tokens: [*c]const c_int, count: c_int, pool: [*c]c.ds4_session_resource_pool, prefill: [*c]c.ds4_session_view, prompt: [*c]const c.ds4_tokens, message: [*c]u8, capacity: usize) c_int {
    if (prefill == null or prompt == null or pool == null or (count > 0 and (views == null or tokens == null))) return invalid(message, capacity, "invalid mixed model batch");
    const metal_scope = MetalScope.enter(prefill.*.core);
    defer metal_scope.leave();
    const length: usize = @intCast(@max(count, 0));
    return Batch.mixed(if (length == 0) &.{} else views[0..length], if (length == 0) &.{} else tokens[0..length], pool, prefill, prompt, errors(message, capacity));
}
