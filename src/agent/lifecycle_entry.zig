//! Sole C facade; composed session/thread modules have no exports or globals.
const c = @import("agent_runtime_c");
const sessions = @import("worker_sessions.zig");
const Sessions = sessions.Sessions;
const life = @import("lifecycle.zig");
const Lifecycle = life.Lifecycle;
fn errors(value: [*c]u8, length: usize) []u8 {
    return if (value == null or length == 0) &.{} else value[0..length];
}
export fn ds4_agent_worker_load(worker: *c.agent_worker, path: [*c]const u8, sha: [*c]const u8, expected: [*c]const u8, expected_length: usize, output: *c.ds4_tokens, metadata: [*c]c.agent_kv_session_meta, error_text: [*c]u8, capacity: usize) bool {
    return (Sessions{ .worker = worker }).load(path, sha, expected, expected_length, output, metadata, errors(error_text, capacity));
}
export fn ds4_agent_worker_reset(worker: *c.agent_worker, error_text: [*c]u8, capacity: usize) bool {
    return (Sessions{ .worker = worker }).reset(errors(error_text, capacity));
}
export fn ds4_agent_worker_has_user(worker: *c.agent_worker) bool {
    return (Sessions{ .worker = worker }).hasUser();
}
export fn ds4_agent_worker_needs_save(worker: *c.agent_worker) bool {
    return (Sessions{ .worker = worker }).needsSave();
}
export fn ds4_agent_worker_save_now(worker: *c.agent_worker, sha: [*c]u8, count: [*c]c_int, error_text: [*c]u8, capacity: usize) bool {
    return (Sessions{ .worker = worker }).saveNow(sha, count, errors(error_text, capacity));
}
export fn ds4_agent_worker_save(worker: *c.agent_worker, error_text: [*c]u8, capacity: usize) bool {
    return (Sessions{ .worker = worker }).save(errors(error_text, capacity));
}
export fn ds4_agent_worker_history(worker: *c.agent_worker, turns: c_int, error_text: [*c]u8, capacity: usize) bool {
    return (Sessions{ .worker = worker }).showHistory(turns, errors(error_text, capacity));
}
export fn ds4_agent_worker_switch(worker: *c.agent_worker, prefix: [*c]const u8, turns: c_int, error_text: [*c]u8, capacity: usize) bool {
    return (Sessions{ .worker = worker }).switchTo(prefix, turns, errors(error_text, capacity));
}
export fn ds4_agent_worker_numeric_prefix(engine: ?*c.ds4_engine, mode: c.ds4_think_mode, output: *c.ds4_tokens) void {
    sessions.numericPrefix(engine, mode, output);
}
export fn ds4_agent_worker_apply_thinking(worker: *c.agent_worker) void {
    (Sessions{ .worker = worker }).applyThinking();
}
export fn ds4_agent_worker_wait_route(worker: ?*c.agent_worker, error_text: [*c]u8, capacity: usize) bool {
    return if (worker) |owner| (Lifecycle{ .worker = owner }).waitRoute(errors(error_text, capacity)) else true;
}
export fn ds4_agent_worker_deferred_save(worker: *c.agent_worker) void {
    (Lifecycle{ .worker = worker }).deferredSave();
}
export fn ds4_agent_worker_deferred_compact(worker: *c.agent_worker) void {
    (Lifecycle{ .worker = worker }).deferredCompact();
}
export fn ds4_agent_worker_thread(context: ?*anyopaque) ?*anyopaque {
    return life.thread(context);
}
export fn ds4_agent_worker_init(worker: *c.agent_worker, engine: ?*c.ds4_engine, config: *c.agent_config) c_int {
    return Lifecycle.init(worker, engine, config);
}
export fn ds4_agent_worker_free(worker: *c.agent_worker) void {
    (Lifecycle{ .worker = worker }).deinit();
}
