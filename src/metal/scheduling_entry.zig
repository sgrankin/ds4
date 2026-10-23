//! One C facade for the three pure scheduling owners. All native imports use
//! the same generated metal_c module and the same process-owned ARC records.
const c = @import("metal_c");
const command = @import("command_schedule.zig");
const parallel = @import("parallel_ffn.zig");
const kv = @import("deferred_kv.zig");
export fn ds4_native_metal_new_command() ?*anyopaque {
    return command.newCommand();
}
export fn ds4_native_metal_command_acquire(owned: [*c]c_int) ?*anyopaque {
    return command.acquire(@ptrCast(owned));
}
export fn ds4_native_metal_command_release(token: ?*anyopaque, owned: c_int, ok: c_int, label: [*c]const u8, caller: usize) c_int {
    return command.release(token, owned, ok, label, caller);
}
export fn ds4_native_metal_finish_command(token: ?*anyopaque, owned: c_int, label: [*c]const u8, caller: usize) c_int {
    return command.finish(token, owned, label, caller);
}
export fn ds4_native_metal_compute_encoder(token: ?*anyopaque, caller: usize) ?*anyopaque {
    return command.encoder(token, caller);
}
export fn ds4_native_metal_end_encoder(token: ?*anyopaque, encoder: ?*anyopaque) void {
    command.endEncoder(token, encoder);
}
export fn ds4_native_metal_close_encoder() void {
    command.closeEncoder();
}
export fn ds4_native_metal_wait_pending(label: [*c]const u8) c_int {
    return command.waitPending(label);
}
export fn ds4_native_metal_begin_commands() c_int {
    return command.begin();
}
export fn ds4_native_metal_flush_encoder() c_int {
    return command.flushEncoder();
}
export fn ds4_native_metal_flush_commands() c_int {
    return command.flush();
}
export fn ds4_native_metal_end_commands() c_int {
    return command.end();
}
export fn ds4_native_metal_synchronize() c_int {
    return command.synchronize();
}
export fn ds4_native_metal_commands_active() c_int {
    return @intFromBool(command.state().commands.batch != null);
}
export fn ds4_native_metal_parallel_reset(close: bool) void {
    command.resetParallel(close);
}
export fn ds4_native_metal_parallel_start(request: [*c]const c.ds4_metal_parallel_request) c_int {
    return parallel.start(@ptrCast(request));
}
export fn ds4_native_metal_parallel_first(token: ?*anyopaque, mid: ?*anyopaque) c_int {
    return parallel.first(token, mid);
}
export fn ds4_native_metal_parallel_second(token: ?*anyopaque) c_int {
    return parallel.second(token);
}
export fn ds4_native_metal_parallel_finish() c_int {
    return parallel.finish();
}
export fn ds4_native_metal_defer_kv_next() void {
    kv.deferNext();
}
export fn ds4_native_metal_kv_flush() c_int {
    return kv.flush();
}
export fn ds4_native_metal_kv_begin() c_int {
    return kv.beginConcurrent();
}
export fn ds4_native_metal_kv_end() void {
    kv.endConcurrent();
}
export fn ds4_native_metal_fused_kv(request: [*c]const c.ds4_metal_fused_kv_request) c_int {
    return kv.fused(@ptrCast(request));
}
