//! Sole C facade for complete OpenAI projection actions and shared byte policy.
const c = @import("server_protocol_c");
const shared = @import("stream_bytes.zig");
const OpenAI = @import("openai.zig");
const Anthropic = @import("anthropic.zig");
const Stop = *const volatile c.sig_atomic_t;
const text = shared.text;
fn output(fd: c_int, stop: Stop, memory: ?*c.tool_memory, mutex: ?*c.pthread_mutex_t) OpenAI.Output {
    return .{ .descriptor = fd, .stop = stop, .memory = memory, .mutex = mutex };
}
fn optional(pointer: [*c]const u8) ?[]const u8 {
    return if (pointer == null) null else text(pointer);
}
fn sampled(pointer: [*c]const u8, length: usize) ?[]const u8 {
    return if (pointer == null) null else pointer[0..length];
}
export fn ds4_server_protocol_utf8_limit(raw: [*c]const u8, start: usize, limit: usize, final: bool) usize {
    return shared.utf8Limit(raw, start, limit, final);
}
export fn ds4_server_protocol_text_limit(raw: [*c]const u8, start: usize, length: usize, has_tools: bool, final: bool) usize {
    return shared.textLimit(raw, start, length, has_tools, final);
}
export fn ds4_server_protocol_parameter_limit(raw: [*c]const u8, start: usize, length: usize, closing: [*c]const u8, string: bool) usize {
    return shared.parameterLimit(raw, start, length, closing, string);
}
export fn ds4_server_protocol_random_tool_id(out: [*c]u8, capacity: usize, api: c.api_style) void {
    shared.randomToolId(if (capacity == 0) &.{} else out[0..capacity], api);
}
export fn ds4_server_openai_init(request: *const c.request, stream: *c.openai_stream) void {
    OpenAI.init(request, stream);
}
export fn ds4_server_openai_free(stream: ?*c.openai_stream) void {
    OpenAI.deinit(stream);
}
export fn ds4_server_openai_apply_ids(calls: ?*c.tool_calls, stream: ?*const c.openai_stream) void {
    OpenAI.applyIds(calls, stream);
}
export fn ds4_server_openai_calls(out: *c.buf, calls: *const c.tool_calls, prefix: [*c]const u8, deltas: bool) void {
    OpenAI.appendCalls(out, calls, prefix, deltas);
}
export fn ds4_server_openai_usage(out: *c.buf, request: ?*const c.request, input: c_int, completion: c_int) void {
    OpenAI.appendUsage(out, request, input, completion);
}
export fn ds4_server_openai_chunk(fd: c_int, request: *const c.request, id: [*c]const u8, content: [*c]const u8, finish: [*c]const u8, stop: Stop) bool {
    return output(fd, stop, null, null).chunk(request, text(id), optional(content), optional(finish));
}
export fn ds4_server_openai_done(fd: c_int, request: *const c.request, id: [*c]const u8, input: c_int, completion: c_int, stop: Stop) bool {
    return output(fd, stop, null, null).done(request, text(id), input, completion);
}
export fn ds4_server_openai_chat_finish(fd: c_int, request: *const c.request, id: [*c]const u8, content: [*c]const u8, reasoning: [*c]const u8, calls: ?*const c.tool_calls, finish: [*c]const u8, input: c_int, completion: c_int, stop: Stop) bool {
    return output(fd, stop, null, null).chatFinish(request, text(id), text(content), text(reasoning), calls, text(finish), input, completion);
}
export fn ds4_server_openai_update(fd: c_int, memory: ?*c.tool_memory, mutex: ?*c.pthread_mutex_t, request: *const c.request, id: [*c]const u8, stream: *c.openai_stream, raw: [*c]const u8, length: usize, final: bool, stop: Stop) bool {
    return output(fd, stop, memory, mutex).update(request, text(id), stream, sampled(raw, length), final);
}
export fn ds4_server_openai_finish(fd: c_int, memory: ?*c.tool_memory, mutex: ?*c.pthread_mutex_t, request: *const c.request, id: [*c]const u8, stream: *c.openai_stream, raw: [*c]const u8, length: usize, calls: ?*const c.tool_calls, finish: [*c]const u8, input: c_int, completion: c_int, stop: Stop) bool {
    return output(fd, stop, memory, mutex).finishLive(request, text(id), stream, sampled(raw, length), calls, text(finish), input, completion);
}
export fn ds4_server_openai_final(fd: c_int, cors: bool, request: *const c.request, id: [*c]const u8, content: [*c]const u8, reasoning: [*c]const u8, calls: ?*const c.tool_calls, finish: [*c]const u8, input: c_int, completion: c_int, stop: Stop) bool {
    return output(fd, stop, null, null).respond(cors, request, text(id), text(content), text(reasoning), calls, text(finish), input, completion);
}

fn anthropic(fd: c_int, stop: Stop, memory: ?*c.tool_memory, mutex: ?*c.pthread_mutex_t) Anthropic.Output {
    return .{ .descriptor = fd, .stop = stop, .memory = memory, .mutex = mutex };
}
export fn ds4_server_anthropic_content(out: *c.buf, content: [*c]const u8, reasoning: [*c]const u8, calls: ?*const c.tool_calls, prefix: [*c]const u8, orders: ?*const c.tool_schema_orders) void {
    _ = orders;
    Anthropic.appendContent(out, text(content), text(reasoning), calls, prefix);
}
export fn ds4_server_anthropic_usage(out: *c.buf, request: ?*const c.request, input: c_int, completion: c_int) void {
    Anthropic.appendUsage(out, request, input, completion);
}
export fn ds4_server_anthropic_free(stream: ?*c.anthropic_stream) void {
    Anthropic.deinit(stream);
}
export fn ds4_server_anthropic_apply_ids(calls: ?*c.tool_calls, stream: ?*const c.anthropic_stream) void {
    Anthropic.applyIds(calls, stream);
}
export fn ds4_server_anthropic_final(fd: c_int, cors: bool, request: *const c.request, id: [*c]const u8, content: [*c]const u8, reasoning: [*c]const u8, calls: ?*const c.tool_calls, finish: [*c]const u8, input: c_int, completion: c_int, stop: Stop) bool {
    return anthropic(fd, stop, null, null).respond(cors, request, id, text(content), text(reasoning), calls, text(finish), input, completion);
}
export fn ds4_server_anthropic_start(fd: c_int, request: *const c.request, id: [*c]const u8, input: c_int, stream: *c.anthropic_stream, stop: Stop) bool {
    return anthropic(fd, stop, null, null).start(request, text(id), input, stream);
}
export fn ds4_server_anthropic_update(fd: c_int, memory: ?*c.tool_memory, mutex: ?*c.pthread_mutex_t, request: *const c.request, id: [*c]const u8, stream: *c.anthropic_stream, raw: [*c]const u8, length: usize, final: bool, stop: Stop) bool {
    return anthropic(fd, stop, memory, mutex).update(request, text(id), stream, sampled(raw, length), final);
}
export fn ds4_server_anthropic_finish(fd: c_int, memory: ?*c.tool_memory, mutex: ?*c.pthread_mutex_t, request: *const c.request, id: [*c]const u8, stream: *c.anthropic_stream, raw: [*c]const u8, length: usize, calls: ?*const c.tool_calls, finish: [*c]const u8, completion: c_int, stop: Stop) bool {
    return anthropic(fd, stop, memory, mutex).finishLive(request, id, stream, sampled(raw, length), calls, text(finish), completion);
}
