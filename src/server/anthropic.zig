//! Anthropic message envelopes and sampled-text content-block projection.
//! This has its own block/signature/stop lifecycle; sharing a callback parser
//! with OpenAI would hide the protocol-specific failure and commit boundaries.
const std = @import("std");
const shared = @import("stream_bytes.zig");
const c = shared.c;
const Text = shared.Text;
const text = shared.text;
const Stream = c.anthropic_stream;
const ToolStream = c.anthropic_tool_stream;
const Stop = *const volatile c.sig_atomic_t;

fn callsSlice(calls: ?*const c.tool_calls) []const c.tool_call {
    const value = calls orelse return &.{};
    return shared.items(c.tool_call, value.v, @intCast(@max(value.len, 0)));
}
fn stopReason(finish: []const u8) []const u8 {
    if (std.mem.eql(u8, finish, "tool_calls")) return "tool_use";
    if (std.mem.eql(u8, finish, "length")) return "max_tokens";
    return "end_turn";
}
fn toolUse(out: *Text, call: *const c.tool_call, prefix: [*c]const u8, position: c_int) void {
    var fallback: [128]u8 = undefined;
    _ = c.snprintf(&fallback, fallback.len, "toolu_%s_%d", prefix, position);
    out.append("{\"type\":\"tool_use\",\"id\":");
    out.quoted(if (call.id != null and call.id[0] != 0) text(call.id) else text(&fallback));
    out.append(",\"name\":");
    out.quoted(text(call.name));
    out.append(",\"input\":");
    var object = std.mem.zeroes(c.buf);
    defer c.free(object.ptr);
    c.ds4_server_render_object(&object, call.arguments);
    out.append(text(object.ptr));
    out.byte('}');
}
fn content(out: *Text, answer: []const u8, reasoning: []const u8, calls: ?*const c.tool_calls, id: [*c]const u8) void {
    out.byte('[');
    var wrote = false;
    var after_thinking = false;
    if (reasoning.len != 0) {
        out.append("{\"type\":\"thinking\",\"thinking\":");
        out.quoted(reasoning);
        out.append(",\"signature\":");
        out.quoted(text(id));
        out.byte('}');
        wrote = true;
    }
    if (answer.len != 0) {
        if (wrote) out.byte(',');
        out.append("{\"type\":\"text\",\"text\":");
        out.quoted(answer);
        out.byte('}');
        wrote = true;
        after_thinking = true;
    }
    for (callsSlice(calls), 0..) |*call, position| {
        if (wrote) out.byte(',');
        toolUse(out, call, id, @intCast(position));
        wrote = true;
        after_thinking = true;
    }
    if (!wrote or (reasoning.len != 0 and !after_thinking)) {
        if (wrote) out.byte(',');
        out.append("{\"type\":\"text\",\"text\":\"\"}");
    }
    out.byte(']');
}
pub fn appendContent(out: *c.buf, answer: []const u8, reasoning: []const u8, calls: ?*const c.tool_calls, id: [*c]const u8) void {
    var wire: Text = .{};
    defer wire.deinit();
    content(&wire, answer, reasoning, calls, id);
    wire.appendTo(out);
}
fn clamp(value: c_int, maximum: c_int) c_int {
    return if (value < 0) 0 else if (maximum >= 0 and value > maximum) maximum else value;
}
fn usage(out: *Text, request: ?*const c.request, input: c_int, output: c_int) void {
    const cached = clamp(if (request) |r| r.cache_read_tokens else 0, input);
    const written = clamp(if (request) |r| r.cache_write_tokens else 0, input -% cached);
    const fresh = @max(@as(c_int, 0), input -% cached -% written);
    out.append("{\"input_tokens\":");
    out.number(fresh);
    out.append(",\"output_tokens\":");
    out.number(output);
    out.append(",\"cache_read_input_tokens\":");
    out.number(cached);
    out.append(",\"cache_creation_input_tokens\":");
    out.number(written);
    out.byte('}');
}
pub fn appendUsage(out: *c.buf, request: ?*const c.request, input: c_int, output: c_int) void {
    var wire: Text = .{};
    defer wire.deinit();
    usage(&wire, request, input, output);
    wire.appendTo(out);
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
    // The exposed live IDs must reach final parsed calls before assignment and
    // tool-memory storage, or returned tool_result IDs lose their live KV path.
    const count: usize = @intCast(@min(parsed.len, live.tool.ids_cap));
    for (shared.items(c.tool_call, parsed.v, count), shared.items([*c]u8, live.tool.ids, count)) |*call, id| {
        if (call.id != null and call.id[0] != 0) continue;
        if (id != null and id[0] != 0) call.id = shared.duplicate(text(id));
    }
}

pub const Output = struct {
    descriptor: c_int,
    stop: Stop,
    memory: ?*c.tool_memory = null,
    mutex: ?*c.pthread_mutex_t = null,

    fn event(self: Output, name: [*:0]const u8, wire: *Text) bool {
        return c.ds4_server_http_sse_event(self.descriptor, name, wire.terminated(), self.stop);
    }
    pub fn respond(self: Output, cors: bool, request: *const c.request, id: [*c]const u8, answer: []const u8, reasoning: []const u8, calls: ?*const c.tool_calls, finish: []const u8, input: c_int, output: c_int) bool {
        var wire: Text = .{};
        defer wire.deinit();
        wire.append("{\"id\":\"");
        wire.append(text(id));
        wire.append("\",\"type\":\"message\",\"role\":\"assistant\",\"model\":");
        wire.quoted(text(request.model));
        wire.append(",\"content\":");
        content(&wire, answer, reasoning, calls, id);
        wire.append(",\"stop_reason\":");
        wire.quoted(stopReason(finish));
        wire.append(",\"stop_sequence\":null,\"usage\":");
        usage(&wire, request, input, output);
        wire.append("}\n");
        return c.ds4_server_http_response(self.descriptor, cors, 200, "application/json", wire.terminated(), self.stop);
    }
    pub fn start(self: Output, request: *const c.request, id: []const u8, input: c_int, stream: *Stream) bool {
        var wire: Text = .{};
        defer wire.deinit();
        wire.append("{\"type\":\"message_start\",\"message\":{\"id\":\"");
        wire.append(id);
        wire.append("\",\"type\":\"message\",\"role\":\"assistant\",\"model\":");
        wire.quoted(text(request.model));
        wire.append(",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":");
        usage(&wire, request, input, 0);
        wire.append("}}");
        const ok = self.event("message_start", &wire);
        stream.* = std.mem.zeroes(Stream);
        stream.active = ok;
        stream.mode = if (c.ds4_think_mode_enabled(request.think_mode)) c.ANTH_STREAM_THINKING else c.ANTH_STREAM_TEXT;
        stream.guard_second_reasoning = shared.guardSecondReasoning(request);
        return ok;
    }
    // Thinking/text have fixed shapes. A tool block waits for the complete
    // invoke name, so its opener remains separate and protocol-specific.
    fn openBlock(self: Output, stream: *Stream, kind: c.anthropic_block_type) bool {
        if (stream.open_block == kind) return true;
        if (stream.open_block != c.ANTH_BLOCK_NONE) return false;
        var wire: Text = .{};
        defer wire.deinit();
        wire.append("{\"type\":\"content_block_start\",\"index\":");
        wire.number(stream.next_index);
        wire.append(if (kind == c.ANTH_BLOCK_THINKING)
            ",\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}"
        else
            ",\"content_block\":{\"type\":\"text\",\"text\":\"\"}}");
        const ok = self.event("content_block_start", &wire);
        if (ok) stream.open_block = kind;
        return ok;
    }
    fn openTool(self: Output, stream: *Stream, id: []const u8, name: []const u8) bool {
        if (stream.open_block == c.ANTH_BLOCK_TOOL) return true;
        if (stream.open_block != c.ANTH_BLOCK_NONE) return false;
        var wire: Text = .{};
        defer wire.deinit();
        wire.append("{\"type\":\"content_block_start\",\"index\":");
        wire.number(stream.next_index);
        wire.append(",\"content_block\":{\"type\":\"tool_use\",\"id\":");
        wire.quoted(id);
        wire.append(",\"name\":");
        wire.quoted(name);
        wire.append(",\"input\":{}}}");
        const ok = self.event("content_block_start", &wire);
        if (ok) stream.open_block = c.ANTH_BLOCK_TOOL;
        return ok;
    }
    fn delta(self: Output, stream: *const Stream, kind: c.anthropic_block_type, value: []const u8) bool {
        if (value.len == 0) return true;
        var wire: Text = .{};
        defer wire.deinit();
        wire.append("{\"type\":\"content_block_delta\",\"index\":");
        wire.number(stream.next_index);
        wire.append(if (kind == c.ANTH_BLOCK_THINKING)
            ",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":"
        else
            ",\"delta\":{\"type\":\"text_delta\",\"text\":");
        wire.quoted(value);
        wire.append("}}");
        return self.event("content_block_delta", &wire);
    }
    // input_json_delta is a JSON string containing a fragment of the same
    // object the final parser builds, not a replacement object or new grammar.
    fn arguments(self: Output, stream: *const Stream, value: []const u8) bool {
        if (value.len == 0) return true;
        var wire: Text = .{};
        defer wire.deinit();
        wire.append("{\"type\":\"content_block_delta\",\"index\":");
        wire.number(stream.next_index);
        wire.append(",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":");
        wire.quoted(value);
        wire.append("}}");
        return self.event("content_block_delta", &wire);
    }
    fn closeBlock(self: Output, id: []const u8, stream: *Stream) bool {
        if (stream.open_block == c.ANTH_BLOCK_NONE) return true;
        if (stream.open_block == c.ANTH_BLOCK_THINKING) {
            var signature: Text = .{};
            defer signature.deinit();
            signature.append("{\"type\":\"content_block_delta\",\"index\":");
            signature.number(stream.next_index);
            signature.append(",\"delta\":{\"type\":\"signature_delta\",\"signature\":");
            signature.quoted(id);
            signature.append("}}");
            if (!self.event("content_block_delta", &signature)) return false;
        }
        var wire: Text = .{};
        defer wire.deinit();
        wire.append("{\"type\":\"content_block_stop\",\"index\":");
        wire.number(stream.next_index);
        wire.byte('}');
        if (!self.event("content_block_stop", &wire)) return false;
        stream.open_block = c.ANTH_BLOCK_NONE;
        stream.next_index += 1;
        return true;
    }
    fn stringValue(self: Output, stream: *Stream, value: []const u8) bool {
        if (value.len == 0) return true;
        const raw = shared.duplicate(value);
        defer c.free(raw);
        c.ds4_tool_text_unescape(raw, stream.tool.syn.*.param_end);
        var fragment: Text = .{};
        defer fragment.deinit();
        fragment.fragment(text(raw));
        return self.arguments(stream, fragment.list.items);
    }
    fn parameterPrefix(self: Output, stream: *Stream, name: []const u8, string: bool) bool {
        var fragment: Text = .{};
        defer fragment.deinit();
        if (stream.tool.first_param) stream.tool.first_param = false else fragment.byte(',');
        fragment.quoted(name);
        fragment.byte(':');
        if (string) fragment.byte('"');
        return self.arguments(stream, fragment.list.items);
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
                // The same shared action is used by OpenAI and remaining C
                // callers: one counter, with Anthropic's original prefix.
                shared.randomToolId(&generated, c.API_ANTHROPIC);
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
    fn startInvoke(self: Output, stream: *Stream, raw: []const u8) bool {
        const tool = &stream.tool;
        const end = std.mem.findScalarPos(u8, raw, tool.parse_pos, '>') orelse return true;
        const tag = shared.duplicate(raw[tool.parse_pos .. end + 1]);
        defer c.free(tag);
        const name = c.ds4_server_reply_attribute(tag, "name");
        if (name == null) return failTool(tool);
        defer c.free(name);
        const id = self.toolId(tool, tool.index);
        if (!self.openTool(stream, id, text(name)) or !self.arguments(stream, "{")) return false;
        tool.emitted_any = true;
        tool.args_open = true;
        tool.first_param = true;
        tool.parse_pos = end + 1;
        tool.state = c.DSML_TOOL_BETWEEN_PARAMS;
        return true;
    }
    fn startParameter(self: Output, stream: *Stream, raw: []const u8) bool {
        const tool = &stream.tool;
        const end = std.mem.findScalarPos(u8, raw, tool.parse_pos, '>') orelse return true;
        const tag = shared.duplicate(raw[tool.parse_pos .. end + 1]);
        defer c.free(tag);
        const name = c.ds4_server_reply_attribute(tag, "name");
        defer c.free(name);
        const string = c.ds4_server_reply_attribute(tag, "string");
        defer c.free(string);
        if (name == null or string == null) return failTool(tool);
        const is_string = std.mem.eql(u8, text(string), "true");
        if (!self.parameterPrefix(stream, text(name), is_string)) return false;
        tool.param_is_string = is_string;
        tool.parse_pos = end + 1;
        tool.state = c.DSML_TOOL_PARAM_VALUE;
        return true;
    }
    fn finishParameter(self: Output, stream: *Stream, raw: []const u8, end: usize) bool {
        const tool = &stream.tool;
        if (end > tool.parse_pos) {
            const value = raw[tool.parse_pos..end];
            const ok = if (tool.param_is_string) self.stringValue(stream, value) else self.arguments(stream, value);
            if (!ok) return false;
        }
        if (tool.param_is_string and !self.arguments(stream, "\"")) return false;
        tool.parse_pos = end + text(tool.syn.*.param_end).len;
        tool.state = c.DSML_TOOL_BETWEEN_PARAMS;
        return true;
    }
    fn updateTool(self: Output, id: []const u8, stream: *Stream, raw: []const u8) bool {
        const tool = &stream.tool;
        while (tool.active and tool.parse_pos < raw.len) {
            switch (tool.state) {
                c.DSML_TOOL_BETWEEN_INVOKES, c.DSML_TOOL_BETWEEN_PARAMS => {
                    tool.parse_pos = shared.whitespace(raw, tool.parse_pos);
                    if (tool.parse_pos >= raw.len) return true;
                    const between_calls = tool.state == c.DSML_TOOL_BETWEEN_INVOKES;
                    const syntax = tool.syn.*;
                    const ending = text(if (between_calls) syntax.tool_calls_end else syntax.invoke_end);
                    const opening = text(if (between_calls) syntax.invoke_start else syntax.param_start);
                    if (shared.at(raw, tool.parse_pos, ending)) {
                        if (between_calls) {
                            tool.parse_pos += ending.len;
                            tool.active = false;
                            tool.state = c.DSML_TOOL_DONE;
                            return true;
                        }
                        if (tool.args_open and !self.arguments(stream, "}")) return false;
                        tool.args_open = false;
                        if (!self.closeBlock(id, stream)) return false;
                        tool.parse_pos += ending.len;
                        tool.index += 1;
                        tool.state = c.DSML_TOOL_BETWEEN_INVOKES;
                        continue;
                    }
                    if (shared.partial(raw, tool.parse_pos, ending) or shared.partial(raw, tool.parse_pos, opening)) return true;
                    if (!shared.at(raw, tool.parse_pos, opening)) return failTool(tool);
                    const before = tool.parse_pos;
                    const previous = tool.state;
                    const ok = if (between_calls) self.startInvoke(stream, raw) else self.startParameter(stream, raw);
                    if (!ok) return false;
                    if (tool.parse_pos == before and tool.state == previous) return true;
                },
                c.DSML_TOOL_PARAM_VALUE => {
                    const closing = tool.syn.*.param_end;
                    if (std.mem.findPos(u8, raw, tool.parse_pos, text(closing))) |end| {
                        if (!self.finishParameter(stream, raw, end)) return false;
                        continue;
                    }
                    const limit = shared.parameterLimit(raw.ptr, tool.parse_pos, raw.len, closing, tool.param_is_string);
                    if (limit > tool.parse_pos) {
                        const value = raw[tool.parse_pos..limit];
                        const ok = if (tool.param_is_string) self.stringValue(stream, value) else self.arguments(stream, value);
                        if (!ok) return false;
                        tool.parse_pos = limit;
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
        if (stream.mode == c.ANTH_STREAM_THINKING) {
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
                if (!self.openBlock(stream, c.ANTH_BLOCK_THINKING) or
                    !self.delta(stream, c.ANTH_BLOCK_THINKING, sampled[stream.emit_pos..limit])) return false;
                stream.sent_thinking = true;
                stream.emit_pos = limit;
            }
            if (tool_before_close) {
                if (complete_tool) {
                    if (!self.closeBlock(id, stream)) return false;
                    stream.emit_pos = tool.?;
                    stream.mode = c.ANTH_STREAM_SUPPRESS;
                }
                return true;
            }
            if (closing != null or final) {
                if (!self.closeBlock(id, stream)) return false;
                if (closing) |position| {
                    stream.emit_pos = position + "</think>".len;
                    stream.mode = c.ANTH_STREAM_TEXT;
                } else {
                    stream.mode = c.ANTH_STREAM_SUPPRESS;
                    return true;
                }
            } else return true;
        }
        if (stream.mode == c.ANTH_STREAM_TEXT) {
            if (stream.guard_second_reasoning) {
                const closing = findTerminated(sampled, stream.emit_pos, "</think>");
                const tool = if (request.has_tools) findTool(sampled, stream.emit_pos) else null;
                if (closing != null and (tool == null or closing.? < tool.?)) {
                    const limit = closing.?;
                    if (limit > stream.emit_pos) {
                        if (!self.openBlock(stream, c.ANTH_BLOCK_THINKING) or
                            !self.delta(stream, c.ANTH_BLOCK_THINKING, sampled[stream.emit_pos..limit])) return false;
                        stream.sent_thinking = true;
                    }
                    if (!self.closeBlock(id, stream)) return false;
                    stream.emit_pos = limit + "</think>".len;
                    stream.guard_second_reasoning = false;
                } else if (tool == null and !final) return true else stream.guard_second_reasoning = false;
            }
            const tool = if (request.has_tools) findTool(sampled, stream.emit_pos) else null;
            const limit = shared.textLimit(sampled.ptr, stream.emit_pos, sampled.len, request.has_tools, final);
            if (limit > stream.emit_pos) {
                if (!self.openBlock(stream, c.ANTH_BLOCK_TEXT) or
                    !self.delta(stream, c.ANTH_BLOCK_TEXT, sampled[stream.emit_pos..limit])) return false;
                stream.sent_text = true;
                stream.emit_pos = limit;
            }
            if (tool) |position| {
                if (!self.closeBlock(id, stream)) return false;
                stream.emit_pos = position;
                // Final catch-up from plain text leaves tools for the existing
                // final emitter. Ordinary token updates start the live parser.
                stream.mode = if (!final and initTool(&stream.tool, sampled, position)) c.ANTH_STREAM_TOOL else c.ANTH_STREAM_SUPPRESS;
            } else if (final) {
                if (!self.closeBlock(id, stream)) return false;
                stream.mode = c.ANTH_STREAM_SUPPRESS;
            }
        }
        if (stream.mode == c.ANTH_STREAM_TOOL) {
            if (!self.updateTool(id, stream, sampled)) return false;
            if (!stream.tool.active) stream.mode = c.ANTH_STREAM_SUPPRESS;
        }
        return true;
    }
    fn tailTools(self: Output, id: [*c]const u8, stream: *Stream, calls: ?*const c.tool_calls) bool {
        const values = callsSlice(calls);
        const already: usize = @intCast(@min(if (stream.tool.emitted_any) stream.tool.index else 0, @as(c_int, @intCast(values.len))));
        // Completed live calls already emitted start/delta/stop. Only catch-up
        // tail calls use these three events; commit their index after all pass.
        for (values[already..], already..) |*call, position| {
            var fallback: [128]u8 = undefined;
            _ = c.snprintf(&fallback, fallback.len, "toolu_%s_%d", id, @as(c_int, @intCast(position)));
            var begin: Text = .{};
            defer begin.deinit();
            begin.append("{\"type\":\"content_block_start\",\"index\":");
            begin.number(stream.next_index);
            begin.append(",\"content_block\":{\"type\":\"tool_use\",\"id\":");
            begin.quoted(if (call.id != null and call.id[0] != 0) text(call.id) else text(&fallback));
            begin.append(",\"name\":");
            begin.quoted(text(call.name));
            begin.append(",\"input\":{}}}");
            if (!self.event("content_block_start", &begin)) return false;
            var fragment: Text = .{};
            defer fragment.deinit();
            fragment.append("{\"type\":\"content_block_delta\",\"index\":");
            fragment.number(stream.next_index);
            fragment.append(",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":");
            fragment.objectString(call.arguments);
            fragment.append("}}");
            if (!self.event("content_block_delta", &fragment)) return false;
            var end: Text = .{};
            defer end.deinit();
            end.append("{\"type\":\"content_block_stop\",\"index\":");
            end.number(stream.next_index);
            end.byte('}');
            if (!self.event("content_block_stop", &end)) return false;
            stream.next_index += 1;
        }
        return true;
    }
    fn finishMessage(self: Output, finish: []const u8, output: c_int) bool {
        var wire: Text = .{};
        defer wire.deinit();
        wire.append("{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":");
        wire.quoted(stopReason(finish));
        wire.append(",\"stop_sequence\":null},\"usage\":{\"output_tokens\":");
        wire.number(output);
        wire.append("}}");
        return self.event("message_delta", &wire) and
            c.ds4_server_http_sse_event(self.descriptor, "message_stop", "{\"type\":\"message_stop\"}", self.stop);
    }
    pub fn finishLive(self: Output, request: *const c.request, id: [*c]const u8, stream: *Stream, raw: ?[]const u8, calls: ?*const c.tool_calls, finish: []const u8, output: c_int) bool {
        if (!self.update(request, text(id), stream, raw, true)) return false;
        if (stream.sent_thinking and !stream.sent_text and callsSlice(calls).len == 0) {
            if (!self.openBlock(stream, c.ANTH_BLOCK_TEXT) or !self.closeBlock(text(id), stream)) return false;
        }
        return self.tailTools(id, stream, calls) and self.finishMessage(finish, output);
    }
};

fn findTerminated(raw: []const u8, start: usize, needle: [*:0]const u8) ?usize {
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
// Persist only a pointer to an immutable actual syntax record, as C does.
const syntaxes = [_]c.dsml_syntax{
    .{ .tool_calls_start = c.DS41_TOOL_CALLS_START, .tool_calls_end = c.DS41_TOOL_CALLS_END, .invoke_start = c.DS41_INVOKE_START, .invoke_end = c.DS41_INVOKE_END, .param_start = c.DS41_PARAM_START, .param_end = c.DS41_PARAM_END },
    .{ .tool_calls_start = c.DS4_TOOL_CALLS_START, .tool_calls_end = c.DS4_TOOL_CALLS_END, .invoke_start = c.DS4_INVOKE_START, .invoke_end = c.DS4_INVOKE_END, .param_start = c.DS4_PARAM_START, .param_end = c.DS4_PARAM_END },
    .{ .tool_calls_start = c.DS4_TOOL_CALLS_START_SHORT, .tool_calls_end = c.DS4_TOOL_CALLS_END_SHORT, .invoke_start = c.DS4_INVOKE_START_SHORT, .invoke_end = c.DS4_INVOKE_END_SHORT, .param_start = c.DS4_PARAM_START_SHORT, .param_end = c.DS4_PARAM_END_SHORT },
    .{ .tool_calls_start = "<tool_calls>", .tool_calls_end = "</tool_calls>", .invoke_start = "<invoke", .invoke_end = "</invoke>", .param_start = "<parameter", .param_end = "</parameter>" },
};
fn initTool(stream: *ToolStream, raw: []const u8, position: usize) bool {
    freeTool(stream);
    stream.* = std.mem.zeroes(ToolStream);
    stream.active = true;
    stream.state = c.DSML_TOOL_BETWEEN_INVOKES;
    for (&syntaxes) |*syntax| {
        if (!shared.at(raw, position, text(syntax.tool_calls_start))) continue;
        stream.syn = syntax;
        stream.parse_pos = position + text(syntax.tool_calls_start).len;
        return true;
    }
    stream.active = false;
    stream.state = c.DSML_TOOL_ERROR;
    return false;
}
