//! Sole C facade; imported owner declarations are pure.
const c = @import("agent_tool_actions_c");
const owner = @import("executor.zig");
fn errors(pointer: [*c]u8, length: usize) []u8 {
    return if (pointer == null or length == 0) &.{} else pointer[0..length];
}
export fn ds4_agent_executor_call(runtime: *owner.Runtime, context_size: c_int, allow_upto: bool, non_interactive: bool, use_color: bool, context: ?*anyopaque, emit: c.ds4_agent_presentation_emit_fn, cancelled: c.ds4_session_cancel_fn, terminal_changed: c.ds4_agent_bash_terminal_fn, call: *const c.agent_tool_call) [*c]u8 {
    return (owner.Executor{ .runtime = runtime, .context_size = context_size, .allow_upto = allow_upto, .non_interactive = non_interactive, .use_color = use_color, .context = context, .emit = emit, .cancelled = cancelled, .terminal_changed = terminal_changed }).execute(call);
}
export fn ds4_agent_executor_start(runtime: *owner.Runtime, command: [*c]const u8, seconds: c_int, terminal_changed: c.ds4_agent_bash_terminal_fn, context: ?*anyopaque, error_text: [*c]u8, capacity: usize) ?*owner.Job {
    return owner.startJob(runtime, command, seconds, terminal_changed, context, errors(error_text, capacity));
}
export fn ds4_agent_executor_jobs_free(runtime: *owner.Runtime) void {
    owner.freeJobs(runtime);
}
export fn ds4_agent_executor_find(runtime: *owner.Runtime, id: c_int, pid: c.pid_t) ?*owner.Job {
    return owner.findJob(runtime, id, pid);
}
export fn ds4_agent_executor_remove(runtime: *owner.Runtime, job: *owner.Job) void {
    owner.removeJob(runtime, job);
}
export fn ds4_agent_executor_compaction_observation(runtime: *owner.Runtime) [*c]u8 {
    return owner.compactionObservation(runtime);
}
export fn ds4_agent_executor_terminal_safe(pointer: [*c]const u8, length: usize) [*c]u8 {
    return owner.terminalSafe(if (length == 0) &.{} else pointer[0..length]);
}
