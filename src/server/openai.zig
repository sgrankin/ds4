//! OpenAI completion/chat responses and the live sampled-DSML projection.
//! Final reply parsing remains authoritative. This owner only decides which
//! already-sampled bytes become content/reasoning/argument wire events.
const std = @import("std");
const shared = @import("stream_bytes.zig");
const c = shared.c;
const Text = shared.Text;
const text = shared.text;
const Stream = c.openai_stream;
const ToolStream = c.openai_tool_stream;
const Stop = *const volatile c.sig_atomic_t;

fn callsSlice(calls: ?*const c.tool_calls) []const c.tool_call {
    const value = calls orelse return &.{};
    return shared.items(c.tool_call, value.v, @intCast(@max(value.len, 0)));
}
pub fn init(request: *const c.request, stream: *Stream) void {
    stream.* = std.mem.zeroes(Stream);
    stream.active = true;
    stream.mode = if (request.model_syntax != c.SERVER_MODEL_SYNTAX_FLINT and c.ds4_think_mode_enabled(request.think_mode)) c.OPENAI_STREAM_THINKING else c.OPENAI_STREAM_TEXT;
    stream.guard_second_reasoning = shared.guardSecondReasoning(request);
}
fn freeTool(stream: *ToolStream) void {
    for (shared.items([*c]u8, stream.ids, @intCast(stream.ids_cap))) |id| c.free(id);
    c.free(@ptrCast(stream.ids));
    stream.ids = null;
    stream.ids_cap = 0;
}
pub fn deinit(stream: ?*Stream) void {
    freeTool(&(stream orelse return).tool);
}
pub fn applyIds(calls: ?*c.tool_calls, stream: ?*const Stream) void {
    const parsed = calls orelse return;
    const live = stream orelse return;
    const count: usize = @intCast(@min(parsed.len, live.tool.ids_cap));
    for (shared.items(c.tool_call, parsed.v, count), shared.items([*c]u8, live.tool.ids, count)) |*call, id| {
        if (call.id != null and call.id[0] != 0) continue;
        if (id != null and id[0] != 0) call.id = shared.duplicate(text(id));
    }
}

fn toolCalls(out: *Text, calls: ?*const c.tool_calls, prefix: [*c]const u8, deltas: bool) void {
    out.byte('[');
    for (callsSlice(calls), 0..) |call, position| {
        if (position != 0) out.byte(',');
        var fallback: [128]u8 = undefined;
        _ = c.snprintf(&fallback, fallback.len, "%s_tool_%d", prefix, @as(c_int, @intCast(position)));
        out.byte('{');
        if (deltas) {
            out.append("\"index\":");
            out.number(position);
            out.byte(',');
        }
        out.append("\"id\":");
        out.quoted(text(if (call.id == null) @as([*c]const u8, &fallback) else call.id));
        out.append(",\"type\":\"function\",\"function\":{\"name\":");
        out.quoted(text(call.name));
        out.append(",\"arguments\":");
        out.objectString(call.arguments);
        out.append("}}");
    }
    out.byte(']');
}
pub fn appendCalls(out: *c.buf, calls: *const c.tool_calls, prefix: [*c]const u8, deltas: bool) void {
    var wire: Text = .{};
    defer wire.deinit();
    toolCalls(&wire, calls, prefix, deltas);
    wire.appendTo(out);
}
fn clamp(value: c_int, maximum: c_int) c_int {
    return if (value < 0) 0 else if (maximum >= 0 and value > maximum) maximum else value;
}
fn usage(out: *Text, request: ?*const c.request, input: c_int, output: c_int) void {
    const cached = clamp(if (request) |r| r.cache_read_tokens else 0, input);
    const written = clamp(if (request) |r| r.cache_write_tokens else 0, input -% cached);
    // OpenAI cached_tokens excludes newly-prefilled tokens. The latter remains
    // a separate DS4 extension rather than inflating the client's cache hits.
    out.append("{\"prompt_tokens\":");
    out.number(input);
    out.append(",\"completion_tokens\":");
    out.number(output);
    out.append(",\"total_tokens\":");
    out.number(input +% output);
    out.append(",\"prompt_tokens_details\":{\"cached_tokens\":");
    out.number(cached);
    out.append(",\"cache_write_tokens\":");
    out.number(written);
    out.append("}}");
}
pub fn appendUsage(out: *c.buf, request: ?*const c.request, input: c_int, output: c_int) void {
    var wire: Text = .{};
    defer wire.deinit();
    usage(&wire, request, input, output);
    wire.appendTo(out);
}
fn chunkHeader(out: *Text, request: *const c.request, id: []const u8, chat: bool, now: c_long) void {
    out.append("data: {\"id\":\"");
    out.append(id);
    out.append(if (chat) "\",\"object\":\"chat.completion.chunk\",\"created\":" else "\",\"object\":\"text_completion\",\"created\":");
    out.number(now);
    out.append(",\"model\":");
    out.quoted(text(request.model));
}

pub const Output = struct {
    descriptor: c_int,
    stop: Stop,
    memory: ?*c.tool_memory = null,
    mutex: ?*c.pthread_mutex_t = null,

    fn send(self: Output, wire: *const Text) bool {
        return c.ds4_server_http_send_all(self.descriptor, wire.list.items.ptr, wire.list.items.len, self.stop);
    }
    pub fn chunk(self: Output, request: *const c.request, id: []const u8, content: ?[]const u8, finish: ?[]const u8) bool {
        var wire: Text = .{};
        defer wire.deinit();
        const chat = request.kind == c.REQ_CHAT;
        chunkHeader(&wire, request, id, chat, c.time(null));
        if (chat) {
            wire.append(",\"choices\":[{\"index\":0,\"delta\":");
            if (content) |value| {
                wire.append("{\"content\":");
                wire.quoted(value);
                wire.byte('}');
            } else wire.append(if (finish != null) "{}" else "{\"role\":\"assistant\"}");
            wire.append(",\"finish_reason\":");
        } else {
            wire.append(",\"choices\":[{\"text\":");
            wire.quoted(content orelse "");
            wire.append(",\"index\":0,\"finish_reason\":");
        }
        if (finish) |value| wire.quoted(value) else wire.append("null");
        wire.append("}]}\n\n");
        return self.send(&wire);
    }
    fn usageChunk(self: Output, request: *const c.request, id: []const u8, input: c_int, output: c_int) bool {
        if (!request.stream_include_usage) return true;
        var wire: Text = .{};
        defer wire.deinit();
        chunkHeader(&wire, request, id, request.kind == c.REQ_CHAT, c.time(null));
        wire.append(",\"choices\":[],\"usage\":");
        usage(&wire, request, input, output);
        wire.append("}\n\n");
        return self.send(&wire);
    }
    pub fn done(self: Output, request: *const c.request, id: []const u8, input: c_int, output: c_int) bool {
        return self.usageChunk(request, id, input, output) and
            c.ds4_server_http_send_all(self.descriptor, "data: [DONE]\n\n", 14, self.stop);
    }
    fn delta(self: Output, request: *const c.request, id: []const u8, field: []const u8, value: []const u8) bool {
        if (value.len == 0) return true;
        var wire: Text = .{};
        defer wire.deinit();
        chunkHeader(&wire, request, id, true, c.time(null));
        wire.append(",\"choices\":[{\"index\":0,\"delta\":{");
        wire.quoted(field);
        wire.byte(':');
        wire.quoted(value);
        wire.append("},\"finish_reason\":null}]}\n\n");
        return self.send(&wire);
    }
    fn toolStart(self: Output, request: *const c.request, id: []const u8, position: c_int, tool_id: []const u8, name: []const u8) bool {
        var wire: Text = .{};
        defer wire.deinit();
        chunkHeader(&wire, request, id, true, c.time(null));
        wire.append(",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":");
        wire.number(position);
        wire.append(",\"id\":");
        wire.quoted(tool_id);
        wire.append(",\"type\":\"function\",\"function\":{\"name\":");
        wire.quoted(name);
        wire.append(",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n");
        return self.send(&wire);
    }
    fn arguments(self: Output, request: *const c.request, id: []const u8, position: c_int, value: []const u8) bool {
        if (value.len == 0) return true;
        var wire: Text = .{};
        defer wire.deinit();
        chunkHeader(&wire, request, id, true, c.time(null));
        wire.append(",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":");
        wire.number(position);
        wire.append(",\"function\":{\"arguments\":");
        wire.quoted(value);
        wire.append("}}]},\"finish_reason\":null}]}\n\n");
        return self.send(&wire);
    }
    fn stringValue(self: Output, request: *const c.request, id: []const u8, stream: *ToolStream, value: []const u8) bool {
        if (value.len == 0) return true;
        const raw = shared.duplicate(value);
        defer c.free(raw);
        c.ds4_tool_text_unescape(raw, stream.param_end);
        var fragment: Text = .{};
        defer fragment.deinit();
        fragment.fragment(text(raw));
        return self.arguments(request, id, stream.index, fragment.list.items);
    }
    fn parameterPrefix(self: Output, request: *const c.request, id: []const u8, stream: *ToolStream, name: []const u8, string: bool) bool {
        var fragment: Text = .{};
        defer fragment.deinit();
        if (stream.first_param) stream.first_param = false else fragment.byte(',');
        fragment.quoted(name);
        fragment.byte(':');
        if (string) fragment.byte('"');
        return self.arguments(request, id, stream.index, fragment.list.items);
    }
    fn toolId(self: Output, stream: *ToolStream, position: c_int) []const u8 {
        if (position < 0) return "";
        if (position >= stream.ids_cap) {
            const previous: usize = @intCast(stream.ids_cap);
            var capacity: c_int = if (previous == 0) 4 else stream.ids_cap;
            while (capacity <= position) capacity *= 2;
            const memory = c.realloc(@ptrCast(stream.ids), @as(usize, @intCast(capacity)) * @sizeOf([*c]u8)) orelse shared.fatal();
            stream.ids = @ptrCast(@alignCast(memory));
            @memset(shared.items([*c]u8, stream.ids, @intCast(capacity))[previous..], null);
            stream.ids_cap = capacity;
        }
        const slot = &stream.ids[@intCast(position)];
        if (slot.* == null) {
            var generated: [64]u8 = undefined;
            while (true) {
                shared.randomToolId(&generated, c.API_OPENAI);
                const value = text(&generated);
                var collision = false;
                for (shared.items([*c]u8, stream.ids, @intCast(position))) |old| {
                    if (old != null and std.mem.eql(u8, text(old), value)) {
                        collision = true;
                        break;
                    }
                }
                if (collision) continue;
                if (self.memory) |memory| {
                    if (c.ds4_server_tool_memory_contains(memory, self.mutex.?, &generated)) continue;
                }
                slot.* = shared.duplicate(value);
                break;
            }
        }
        return text(slot.*);
    }
    fn startInvoke(self: Output, request: *const c.request, id: []const u8, stream: *ToolStream, raw: []const u8) bool {
        const end = std.mem.findScalarPos(u8, raw, stream.parse_pos, '>') orelse return true;
        const tag = shared.duplicate(raw[stream.parse_pos .. end + 1]);
        defer c.free(tag);
        const name = c.ds4_server_reply_attribute(tag, "name");
        if (name == null) return failTool(stream);
        defer c.free(name);
        if (!self.toolStart(request, id, stream.index, self.toolId(stream, stream.index), text(name)) or
            !self.arguments(request, id, stream.index, "{")) return false;
        stream.emitted_any = true;
        stream.args_open = true;
        stream.first_param = true;
        stream.parse_pos = end + 1;
        stream.state = c.DSML_TOOL_BETWEEN_PARAMS;
        return true;
    }
    fn startParameter(self: Output, request: *const c.request, id: []const u8, stream: *ToolStream, raw: []const u8) bool {
        const end = std.mem.findScalarPos(u8, raw, stream.parse_pos, '>') orelse return true;
        const tag = shared.duplicate(raw[stream.parse_pos .. end + 1]);
        defer c.free(tag);
        const name = c.ds4_server_reply_attribute(tag, "name");
        defer c.free(name);
        const string = c.ds4_server_reply_attribute(tag, "string");
        defer c.free(string);
        if (name == null or string == null) return failTool(stream);
        const is_string = std.mem.eql(u8, text(string), "true");
        if (!self.parameterPrefix(request, id, stream, text(name), is_string)) return false;
        stream.param_is_string = is_string;
        stream.parse_pos = end + 1;
        stream.state = c.DSML_TOOL_PARAM_VALUE;
        return true;
    }
    fn finishParameter(self: Output, request: *const c.request, id: []const u8, stream: *ToolStream, raw: []const u8, end: usize) bool {
        if (end > stream.parse_pos) {
            const value = raw[stream.parse_pos..end];
            const ok = if (stream.param_is_string) self.stringValue(request, id, stream, value) else self.arguments(request, id, stream.index, value);
            if (!ok) return false;
        }
        if (stream.param_is_string and !self.arguments(request, id, stream.index, "\"")) return false;
        stream.parse_pos = end + text(stream.param_end).len;
        stream.state = c.DSML_TOOL_BETWEEN_PARAMS;
        return true;
    }
    fn updateTool(self: Output, request: *const c.request, id: []const u8, stream: *ToolStream, raw: []const u8) bool {
        while (stream.active and stream.parse_pos < raw.len) {
            switch (stream.state) {
                c.DSML_TOOL_BETWEEN_INVOKES, c.DSML_TOOL_BETWEEN_PARAMS => {
                    stream.parse_pos = shared.whitespace(raw, stream.parse_pos);
                    if (stream.parse_pos >= raw.len) return true;
                    const between_calls = stream.state == c.DSML_TOOL_BETWEEN_INVOKES;
                    const ending = text(if (between_calls) stream.tool_calls_end else stream.invoke_end);
                    const opening = text(if (between_calls) stream.invoke_start else stream.param_start);
                    if (shared.at(raw, stream.parse_pos, ending)) {
                        if (between_calls) {
                            stream.parse_pos += ending.len;
                            stream.active = false;
                            stream.state = c.DSML_TOOL_DONE;
                            return true;
                        }
                        if (stream.args_open and !self.arguments(request, id, stream.index, "}")) return false;
                        stream.args_open = false;
                        stream.parse_pos += ending.len;
                        stream.index += 1;
                        stream.state = c.DSML_TOOL_BETWEEN_INVOKES;
                        continue;
                    }
                    if (shared.partial(raw, stream.parse_pos, ending) or shared.partial(raw, stream.parse_pos, opening)) return true;
                    if (!shared.at(raw, stream.parse_pos, opening)) return failTool(stream);
                    const before = stream.parse_pos;
                    const previous = stream.state;
                    const ok = if (between_calls) self.startInvoke(request, id, stream, raw) else self.startParameter(request, id, stream, raw);
                    if (!ok) return false;
                    if (stream.parse_pos == before and stream.state == previous) return true;
                },
                c.DSML_TOOL_PARAM_VALUE => {
                    if (std.mem.findPos(u8, raw, stream.parse_pos, text(stream.param_end))) |end| {
                        if (!self.finishParameter(request, id, stream, raw, end)) return false;
                        continue;
                    }
                    const limit = shared.parameterLimit(raw.ptr, stream.parse_pos, raw.len, stream.param_end, stream.param_is_string);
                    if (limit > stream.parse_pos) {
                        const value = raw[stream.parse_pos..limit];
                        const ok = if (stream.param_is_string) self.stringValue(request, id, stream, value) else self.arguments(request, id, stream.index, value);
                        if (!ok) return false;
                        stream.parse_pos = limit;
                    }
                    return true;
                },
                else => return true,
            }
        }
        return true;
    }
    pub fn update(self: Output, request: *const c.request, id: []const u8, stream: *Stream, raw: ?[]const u8, final: bool) bool {
        const sampled = raw orelse return true;
        if (!stream.active) return true;
        if (request.model_syntax == c.SERVER_MODEL_SYNTAX_FLINT and !stream.checked_think_prefix) {
            if (!final and sampled.len < 7 and std.mem.startsWith(u8, "<think>", sampled)) return true;
            stream.checked_think_prefix = true;
            if (std.mem.startsWith(u8, sampled, "<think>")) {
                stream.mode = c.OPENAI_STREAM_THINKING;
                stream.emit_pos = 7;
            }
        }
        if (stream.mode == c.OPENAI_STREAM_THINKING) {
            if (!stream.checked_think_prefix) {
                if (sampled.len < "<think>".len and std.mem.startsWith(u8, "<think>", sampled) and !final) return true;
                if (std.mem.startsWith(u8, sampled, "<think>")) stream.emit_pos = "<think>".len;
                stream.checked_think_prefix = true;
            }
            const closing = findTerminated(sampled, stream.emit_pos, "</think>");
            const tool = if (request.has_tools) findTool(sampled, stream.emit_pos) else null;
            const tool_before_close = tool != null and (closing == null or tool.? < closing.?);
            const complete_tool = tool_before_close and c.ds4_server_reply_tool_end(sampled.ptr + tool.?) != null;
            const limit = if (tool_before_close)
                c.ds4_server_reply_trim_separator(sampled.ptr, stream.emit_pos, tool.?)
            else if (closing) |position| position else if (final) sampled.len else shared.utf8Limit(sampled.ptr, stream.emit_pos, if (sampled.len > 7) sampled.len - 7 else stream.emit_pos, false);
            if (limit > stream.emit_pos) {
                if (!self.delta(request, id, "reasoning_content", sampled[stream.emit_pos..limit])) return false;
                stream.sent_reasoning = true;
                stream.emit_pos = limit;
            }
            if (tool_before_close) {
                if (complete_tool) {
                    stream.emit_pos = tool.?;
                    stream.mode = c.OPENAI_STREAM_SUPPRESS;
                }
                return true;
            }
            if (closing) |position| {
                stream.emit_pos = position + "</think>".len;
                stream.mode = c.OPENAI_STREAM_TEXT;
            } else {
                if (final) stream.mode = c.OPENAI_STREAM_SUPPRESS;
                return true;
            }
        }
        if (stream.mode == c.OPENAI_STREAM_TEXT) {
            if (stream.guard_second_reasoning) {
                const closing = findTerminated(sampled, stream.emit_pos, "</think>");
                const tool = if (request.has_tools) findTool(sampled, stream.emit_pos) else null;
                if (closing != null and (tool == null or closing.? < tool.?)) {
                    const limit = closing.?;
                    if (limit > stream.emit_pos) {
                        if (!self.delta(request, id, "reasoning_content", sampled[stream.emit_pos..limit])) return false;
                        stream.sent_reasoning = true;
                    }
                    stream.emit_pos = limit + "</think>".len;
                    stream.guard_second_reasoning = false;
                } else if (tool == null and !final) return true else stream.guard_second_reasoning = false;
            }
            const tool = if (request.has_tools) findTool(sampled, stream.emit_pos) else null;
            const limit = shared.textLimit(sampled.ptr, stream.emit_pos, sampled.len, request.has_tools, final);
            if (limit > stream.emit_pos) {
                if (!self.delta(request, id, "content", sampled[stream.emit_pos..limit])) return false;
                stream.sent_content = true;
                stream.emit_pos = limit;
            }
            if (tool) |position| {
                stream.emit_pos = position;
                stream.mode = if (initTool(&stream.tool, sampled, position)) c.OPENAI_STREAM_TOOL else c.OPENAI_STREAM_SUPPRESS;
            } else if (final) stream.mode = c.OPENAI_STREAM_SUPPRESS;
        }
        if (stream.mode == c.OPENAI_STREAM_TOOL) {
            if (!self.updateTool(request, id, &stream.tool, sampled)) return false;
            if (!stream.tool.active) stream.mode = c.OPENAI_STREAM_SUPPRESS;
        }
        return true;
    }
    fn terminal(self: Output, request: *const c.request, id: []const u8, calls: ?*const c.tool_calls, finish: []const u8, input: c_int, output: c_int, now: c_long) bool {
        var wire: Text = .{};
        defer wire.deinit();
        if (callsSlice(calls).len != 0) {
            chunkHeader(&wire, request, id, true, now);
            wire.append(",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":");
            toolCalls(&wire, calls, id.ptr, true);
            wire.append("},\"finish_reason\":null}]}\n\n");
        }
        chunkHeader(&wire, request, id, true, now);
        wire.append(",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":");
        wire.quoted(finish);
        wire.append("}]}\n\n");
        return self.send(&wire) and self.done(request, id, input, output);
    }
    pub fn finishLive(self: Output, request: *const c.request, id: []const u8, stream: *Stream, raw: ?[]const u8, calls: ?*const c.tool_calls, finish: []const u8, input: c_int, output: c_int) bool {
        if (!self.update(request, id, stream, raw, true)) return false;
        return self.terminal(request, id, if (stream.tool.emitted_any) null else calls, finish, input, output, c.time(null));
    }
    pub fn chatFinish(self: Output, request: *const c.request, id: []const u8, content: []const u8, reasoning: []const u8, calls: ?*const c.tool_calls, finish: []const u8, input: c_int, output: c_int) bool {
        if (!self.chunk(request, id, null, null)) return false;
        // C captures one timestamp for these concatenated events. Separate
        // sends would change both failure boundaries and the wire grouping.
        var wire: Text = .{};
        defer wire.deinit();
        const now = c.time(null);
        const values = [_][]const u8{ reasoning, content };
        const fields = [_][]const u8{ "reasoning_content", "content" };
        for (values, fields) |value, field| {
            if (value.len == 0) continue;
            chunkHeader(&wire, request, id, true, now);
            wire.append(",\"choices\":[{\"index\":0,\"delta\":{");
            wire.quoted(field);
            wire.byte(':');
            wire.quoted(value);
            wire.append("},\"finish_reason\":null}]}\n\n");
        }
        if (callsSlice(calls).len != 0) {
            chunkHeader(&wire, request, id, true, now);
            wire.append(",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":");
            toolCalls(&wire, calls, id.ptr, true);
            wire.append("},\"finish_reason\":null}]}\n\n");
        }
        chunkHeader(&wire, request, id, true, now);
        wire.append(",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":");
        wire.quoted(finish);
        wire.append("}]}\n\n");
        return self.send(&wire) and self.done(request, id, input, output);
    }
    pub fn respond(self: Output, cors: bool, request: *const c.request, id: []const u8, content: []const u8, reasoning: []const u8, calls: ?*const c.tool_calls, finish: []const u8, input: c_int, output: c_int) bool {
        var wire: Text = .{};
        defer wire.deinit();
        const chat = request.kind == c.REQ_CHAT;
        wire.append("{\"id\":\"");
        wire.append(id);
        wire.append(if (chat) "\",\"object\":\"chat.completion\",\"created\":" else "\",\"object\":\"text_completion\",\"created\":");
        wire.number(c.time(null));
        wire.append(",\"model\":");
        wire.quoted(text(request.model));
        if (chat) {
            wire.append(",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
            wire.quoted(content);
            if (reasoning.len != 0) {
                wire.append(",\"reasoning_content\":");
                wire.quoted(reasoning);
            }
            if (callsSlice(calls).len != 0) {
                wire.append(",\"tool_calls\":");
                toolCalls(&wire, calls, id.ptr, false);
            }
            wire.append("},\"finish_reason\":");
        } else {
            wire.append(",\"choices\":[{\"text\":");
            wire.quoted(content);
            wire.append(",\"index\":0,\"finish_reason\":");
        }
        wire.quoted(finish);
        wire.append("}],\"usage\":");
        usage(&wire, request, input, output);
        wire.append("}\n");
        return c.ds4_server_http_response(self.descriptor, cors, 200, "application/json", wire.terminated(), self.stop);
    }
};

fn findTerminated(raw: []const u8, start: usize, needle: [*:0]const u8) ?usize {
    // The original structural searches use strstr, not the supplied raw length.
    const found = c.strstr(raw.ptr + start, needle);
    return if (found == null) null else @intFromPtr(found) - @intFromPtr(raw.ptr);
}
fn findTool(raw: []const u8, start: usize) ?usize {
    const found = c.ds4_server_reply_tool_start(raw.ptr + start);
    return if (found == null) null else @intFromPtr(found) - @intFromPtr(raw.ptr);
}
fn failTool(stream: *ToolStream) bool {
    stream.active = false;
    stream.state = c.DSML_TOOL_ERROR;
    return true;
}
fn initTool(stream: *ToolStream, raw: []const u8, position: usize) bool {
    freeTool(stream);
    stream.* = std.mem.zeroes(ToolStream);
    stream.active = true;
    stream.state = c.DSML_TOOL_BETWEEN_INVOKES;
    stream.parse_pos = position;
    const variants = [_][6][*c]const u8{
        .{ c.DS41_TOOL_CALLS_START, c.DS41_TOOL_CALLS_END, c.DS41_INVOKE_START, c.DS41_INVOKE_END, c.DS41_PARAM_START, c.DS41_PARAM_END },
        .{ c.DS4_TOOL_CALLS_START, c.DS4_TOOL_CALLS_END, c.DS4_INVOKE_START, c.DS4_INVOKE_END, c.DS4_PARAM_START, c.DS4_PARAM_END },
        .{ c.DS4_TOOL_CALLS_START_SHORT, c.DS4_TOOL_CALLS_END_SHORT, c.DS4_INVOKE_START_SHORT, c.DS4_INVOKE_END_SHORT, c.DS4_PARAM_START_SHORT, c.DS4_PARAM_END_SHORT },
        .{ "<tool_calls>", "</tool_calls>", "<invoke", "</invoke>", "<parameter", "</parameter>" },
    };
    for (variants) |variant| {
        if (!shared.at(raw, position, text(variant[0]))) continue;
        stream.parse_pos += text(variant[0]).len;
        stream.tool_calls_end = variant[1];
        stream.invoke_start = variant[2];
        stream.invoke_end = variant[3];
        stream.param_start = variant[4];
        stream.param_end = variant[5];
        return true;
    }
    stream.active = false;
    stream.state = c.DSML_TOOL_ERROR;
    return false;
}
