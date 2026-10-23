//! The one C facade. The owner is importable without exporting symbols twice.
const std = @import("std");
const c = @import("server_tool_memory_c");
const Memory = @import("tool_memory.zig").Memory;

fn bytes(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn memory(state: *c.tool_memory, mutex: *c.pthread_mutex_t) Memory {
    return .{ .state = state, .mutex = mutex };
}
export fn ds4_server_tool_memory_free(state: *c.tool_memory) void {
    Memory.deinit(state);
}
export fn ds4_server_tool_memory_contains(state: *c.tool_memory, mutex: *c.pthread_mutex_t, id: [*c]const u8) bool {
    return memory(state, mutex).contains(bytes(id));
}
export fn ds4_server_tool_memory_put(state: *c.tool_memory, mutex: *c.pthread_mutex_t, id: [*c]const u8, text: [*c]const u8, source: c.tool_memory_source) void {
    memory(state, mutex).put(bytes(id), bytes(text), source);
}
export fn ds4_server_tool_memory_remember(state: *c.tool_memory, mutex: *c.pthread_mutex_t, calls: *const c.tool_calls) void {
    memory(state, mutex).remember(calls);
}
export fn ds4_server_tool_memory_attach(state: *c.tool_memory, mutex: *c.pthread_mutex_t, messages: *c.chat_msgs, stats: ?*c.tool_replay_stats) void {
    memory(state, mutex).attach(messages, stats);
}
export fn ds4_server_tool_memory_size(state: *c.tool_memory, mutex: *c.pthread_mutex_t, text: [*c]const u8, size: ?*u64) bool {
    if (size) |value| value.* = 0;
    const result = memory(state, mutex).serializedSize(bytes(text)) orelse return false;
    if (size) |value| value.* = result;
    return true;
}
export fn ds4_server_tool_memory_write(state: *c.tool_memory, mutex: *c.pthread_mutex_t, file: *c.FILE, text: [*c]const u8, written: ?*u64) bool {
    return memory(state, mutex).write(file, bytes(text), written);
}
export fn ds4_server_tool_memory_load(state: *c.tool_memory, mutex: *c.pthread_mutex_t, file: *c.FILE, wanted: ?*const c.stop_list) c_int {
    return memory(state, mutex).load(file, wanted);
}
