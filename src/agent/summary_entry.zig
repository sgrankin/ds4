//! The three existing whole-action C symbols now enter pure native owners.
const c = @import("agent_runtime_c");
const System = @import("system.zig");
const Summary = @import("summary.zig").Summary;
fn errors(value: [*c]u8, length: usize) []u8 {
    return if (value == null or length == 0) &.{} else value[0..length];
}
export fn ds4_agent_runtime_build_system(worker: *c.agent_worker, output: *c.ds4_tokens) void {
    System.build(worker, output);
}
export fn ds4_agent_turn_compact_private(worker: *c.agent_worker, reason: [*c]const u8, open_assistant: [*c]c_int, error_text: [*c]u8, capacity: usize) bool {
    return (Summary{ .worker = worker }).compact(reason, open_assistant, errors(error_text, capacity));
}
export fn ds4_agent_result_compact(worker: *c.agent_worker, reason: [*c]const u8, error_text: [*c]u8, capacity: usize) bool {
    return (Summary{ .worker = worker }).compactAndExport(reason, errors(error_text, capacity));
}
