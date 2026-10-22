//! Sole C facade; pure output owners can be imported without duplicate exports.
const std = @import("std");
const c = @import("server_output_c");
const owner = @import("output.zig");
const Stop = *const volatile c.sig_atomic_t;
fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn output(fd: c_int, stop: Stop) owner.Output {
    return .{ .descriptor = fd, .stop_requested = stop };
}
export fn ds4_server_output_random_id(destination: [*c]u8, capacity: usize, prefix: [*c]const u8) void {
    owner.randomId(if (capacity == 0) &.{} else destination[0..capacity], text(prefix));
}
export fn ds4_server_output_init(request: *const c.request, stream: *c.responses_stream) void {
    owner.init(request, stream);
}
export fn ds4_server_output_free(stream: ?*c.responses_stream) void {
    owner.deinit(stream orelse return);
}
export fn ds4_server_output_append_item(buffer: *c.buf, call: *const c.tool_call, item: *const c.responses_tool_item, status: [*c]const u8, with_args: bool, orders: ?*const c.tool_schema_orders) void {
    owner.appendFunctionCall(buffer, call, item, text(status), with_args, orders);
}
export fn ds4_server_output_append_usage(buffer: *c.buf, request: ?*const c.request, input: c_int, completion: c_int) void {
    owner.appendUsage(buffer, request, input, completion);
}
export fn ds4_server_output_created(fd: c_int, request: *const c.request, stream: *c.responses_stream, created_at: c_long, stop: Stop) bool {
    return output(fd, stop).created(request, stream, created_at);
}
export fn ds4_server_output_update(fd: c_int, request: *const c.request, stream: *c.responses_stream, raw: [*c]const u8, length: usize, final: bool, stop: Stop) bool {
    if (raw == null) return true;
    return output(fd, stop).update(request, stream, raw[0..length], final);
}
export fn ds4_server_output_completed(fd: c_int, request: *c.request, stream: *c.responses_stream, calls: ?*const c.tool_calls, items: [*c]const c.responses_tool_item, finish: [*c]const u8, input: c_int, completion: c_int, created_at: c_long, stop: Stop) bool {
    const values: []const c.responses_tool_item = if (items == null or calls == null or calls.?.len == 0) &.{} else items[0..@intCast(calls.?.len)];
    return output(fd, stop).completed(request, stream, calls, values, text(finish), input, completion, created_at);
}
export fn ds4_server_output_finish(fd: c_int, request: *c.request, stream: *c.responses_stream, raw: [*c]const u8, length: usize, recovered: [*c]const u8, calls: ?*const c.tool_calls, finish: [*c]const u8, input: c_int, completion: c_int, created_at: c_long, stop: Stop) bool {
    return output(fd, stop).finishLive(request, stream, if (raw == null) null else raw[0..length], recovered != null, calls, text(finish), input, completion, created_at);
}
export fn ds4_server_output_final(fd: c_int, cors: bool, request: *c.request, content: [*c]const u8, reasoning: [*c]const u8, calls: ?*const c.tool_calls, finish: [*c]const u8, input: c_int, completion: c_int, stop: Stop) bool {
    return output(fd, stop).respond(cors, request, text(content), text(reasoning), calls, text(finish), input, completion);
}
