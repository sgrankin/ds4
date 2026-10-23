//! Sole C facade. All owner modules are pure and share agent_runtime_c.
const c = @import("agent_runtime_c");
const Turn = @import("turn.zig").Turn;
const Prompts = @import("prompts.zig");
const Trace = @import("trace.zig").Trace;
fn bytes(value: [*c]const u8, length: usize) []const u8 {
    return if (length == 0) &.{} else value[0..length];
}
fn errors(value: [*c]u8, length: usize) []u8 {
    return if (value == null or length == 0) &.{} else value[0..length];
}
export fn ds4_agent_turn_run(worker: *c.agent_worker, user: [*c]const u8) c_int {
    return (Turn{ .worker = worker }).run(user);
}
export fn ds4_agent_turn_raw(worker: *c.agent_worker, user: [*c]const u8) c_int {
    return (Turn{ .worker = worker }).raw(user);
}
export fn ds4_agent_turn_sync(worker: *c.agent_worker, tokens: *const c.ds4_tokens, progress: bool, error_text: [*c]u8, capacity: usize) c_int {
    return (Turn{ .worker = worker }).sync(tokens, progress, errors(error_text, capacity));
}
export fn ds4_agent_turn_tools_prompt(engine: ?*c.ds4_engine, allow_upto: bool) [*c]u8 {
    return Prompts.tools(engine, allow_upto);
}
export fn ds4_agent_turn_dsml_prompt(allow_upto: bool, vision: bool) [*c]u8 {
    return Prompts.dsml(allow_upto, vision);
}
export fn ds4_agent_turn_glm_prompt(allow_upto: bool, vision: bool) [*c]u8 {
    return Prompts.glm(allow_upto, vision);
}
export fn ds4_agent_turn_qwen_prompt(allow_upto: bool, vision: bool) [*c]u8 {
    return Prompts.qwen(allow_upto, vision);
}
export fn ds4_agent_turn_dsml41_prompt(value: [*c]const u8) [*c]u8 {
    return Prompts.dsml41(value);
}
export fn ds4_agent_turn_hints_note(value: *c.agent_hints) [*c]const u8 {
    return Prompts.hintsNote(value);
}
export fn ds4_agent_turn_hints_compacted(value: *c.agent_hints) void {
    Prompts.hintsCompacted(value);
}
export fn ds4_agent_turn_hints_context(worker: *c.agent_worker) void {
    Prompts.hintsContext(worker);
}
export fn ds4_agent_turn_append_system(engine: ?*c.ds4_engine, tokens: *c.ds4_tokens, extra: [*c]const u8, allow_upto: bool) void {
    Prompts.appendSystem(engine, tokens, extra, allow_upto);
}
export fn ds4_agent_turn_system_status(worker: *c.agent_worker, value: [*c]const u8) void {
    Prompts.publishSystemStatus(worker, value);
}
export fn ds4_agent_turn_datetime(worker: *c.agent_worker) void {
    Prompts.datetimeContext(worker);
}
export fn ds4_agent_turn_system_reminder(worker: *c.agent_worker) void {
    Prompts.systemReminder(worker);
}
export fn ds4_agent_turn_trace_message(worker: ?*c.agent_worker, value: [*c]const u8) void {
    if (worker) |state| (Trace{ .worker = state }).log("%s", .{value});
}
export fn ds4_agent_turn_trace_text(worker: ?*c.agent_worker, label: [*c]const u8, value: [*c]const u8, length: usize) void {
    if (worker) |state| (Trace{ .worker = state }).text(label, bytes(value, length));
}
export fn ds4_agent_turn_trace_token(worker: ?*c.agent_worker, token: c_int, value: [*c]const u8, length: usize, index: c_int) void {
    if (worker) |state| (Trace{ .worker = state }).token(token, bytes(value, length), index);
}
export fn ds4_agent_turn_trace_tokens(worker: ?*c.agent_worker, label: [*c]const u8, value: [*c]const c.ds4_tokens, start: c_int) void {
    if (worker != null and value != null) (Trace{ .worker = worker.? }).tokens(label, @ptrCast(value), start);
}
