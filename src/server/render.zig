//! Server prompt/replay and tool argument formatting over the real records.
//! Tolerant primitive JSON parsing stays in its existing library. Rendering,
//! argument ownership, Anthropic split views and call-order policy live here.
const std = @import("std");
const c = @import("server_render_c");
const Canonical = @import("canonical.zig");
const Prompt = @import("ds41_prompt");
const FlintPrompt = @import("flint_prompt");
const QwenPrompt = @import("qwen_prompt");
const GlmPrompt = @import("glm_prompt");
const allocator = std.heap.c_allocator;
const Invalid = error{InvalidJson};

fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
    c.exit(1);
}
fn bytes(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) "" else std.mem.span(pointer);
}
fn equal(pointer: [*c]const u8, expected: []const u8) bool {
    return std.mem.eql(u8, bytes(pointer), expected);
}
fn items(comptime T: type, pointer: [*c]T, count: usize) []T {
    // A nonnull empty buffer still owns its capacity. Keep that pointer when
    // temporarily adopting a C allocation; only NULL/zero is truly empty.
    return if (count == 0 and pointer == null) &.{} else @as([*]T, @ptrCast(pointer))[0..count];
}
fn duplicate(value: []const u8) [*c]u8 {
    return (allocator.dupeSentinel(u8, value, 0) catch fatal()).ptr;
}

const Text = struct {
    list: std.ArrayList(u8) = .empty,

    fn deinit(self: *Text) void {
        self.list.deinit(allocator);
    }
    fn append(self: *Text, value: []const u8) void {
        self.list.appendSlice(allocator, value) catch fatal();
    }
    fn byte(self: *Text, value: u8) void {
        self.list.append(allocator, value) catch fatal();
    }
    fn string(self: *Text, value: [*c]const u8) void {
        self.append(bytes(value));
    }
    fn quoted(self: *Text, value: [*c]const u8) void {
        const hex = "0123456789abcdef";
        self.byte('"');
        for (bytes(value)) |value_byte| switch (value_byte) {
            '"', '\\' => self.append(&.{ '\\', value_byte }),
            '\n' => self.append("\\n"),
            '\r' => self.append("\\r"),
            '\t' => self.append("\\t"),
            0...8, 11, 12, 14...31 => self.append(&.{ '\\', 'u', '0', '0', hex[value_byte >> 4], hex[value_byte & 15] }),
            else => self.byte(value_byte),
        };
        self.byte('"');
    }
    fn take(self: *Text) [*c]u8 {
        return (self.list.toOwnedSliceSentinel(allocator, 0) catch fatal()).ptr;
    }
    fn adopt(buffer: *c.buf) Text {
        var text: Text = .{};
        text.list.items = items(u8, buffer.ptr, buffer.len);
        text.list.capacity = buffer.cap;
        return text;
    }
    fn store(self: *Text, buffer: *c.buf) void {
        if (self.list.capacity != 0) {
            self.list.ensureUnusedCapacity(allocator, 1) catch fatal();
            self.list.items.ptr[self.list.items.len] = 0;
            buffer.* = .{ .ptr = self.list.items.ptr, .len = self.list.items.len, .cap = self.list.capacity };
        }
        self.list = .empty; // The original C buffer owns this allocation again.
    }
};

const Reader = struct {
    cursor: [*c]const u8,

    fn peek(self: *Reader) u8 {
        c.ds4_server_json_ws(&self.cursor);
        return self.cursor[0];
    }
    fn consume(self: *Reader, value: u8) Invalid!void {
        if (self.peek() != value) return error.InvalidJson;
        self.cursor += 1;
    }
    fn string(self: *Reader) Invalid![*c]u8 {
        var value: [*c]u8 = null;
        if (!c.ds4_server_json_string(&self.cursor, &value)) return error.InvalidJson;
        return value;
    }
    fn raw(self: *Reader) Invalid![*c]u8 {
        var value: [*c]u8 = null;
        if (!c.ds4_server_json_raw(&self.cursor, &value)) return error.InvalidJson;
        return value;
    }
};

fn pushArgument(args: *c.json_args, argument: c.json_arg) void {
    var list: std.ArrayList(c.json_arg) = .empty;
    list.items = items(c.json_arg, args.v, @intCast(args.len));
    list.capacity = @intCast(args.cap);
    list.append(allocator, argument) catch fatal();
    args.* = .{ .v = list.items.ptr, .len = @intCast(list.items.len), .cap = @intCast(list.capacity) };
}
fn parseArgument(reader: *Reader) Invalid!c.json_arg {
    const key = try reader.string();
    errdefer c.free(key);
    try reader.consume(':');
    const string = reader.peek() == '"';
    const value = if (string) try reader.string() else value: {
        const raw = try reader.raw();
        defer c.free(raw);
        break :value c.ds4_server_json_minify(raw);
    };
    return .{ .key = key, .value = value, .is_string = string, .used = false };
}
pub export fn ds4_server_render_arguments_free(args: *c.json_args) void {
    for (items(c.json_arg, args.v, @intCast(args.len))) |argument| {
        c.free(argument.key);
        c.free(argument.value);
    }
    c.free(@ptrCast(args.v));
    args.* = std.mem.zeroes(c.json_args);
}
pub export fn ds4_server_render_arguments_find(args: *c.json_args, key: [*c]const u8) c_int {
    if (key == null) return -1;
    for (items(c.json_arg, args.v, @intCast(args.len)), 0..) |argument, position| {
        if (!argument.used and argument.key != null and equal(argument.key, bytes(key))) return @intCast(position);
    }
    return -1;
}
fn parseArguments(raw: [*c]const u8, args: *c.json_args) Invalid!void {
    var reader: Reader = .{ .cursor = if (raw == null) "" else raw };
    // A non-object fails without changing an existing vector, as in C.
    try reader.consume('{');
    errdefer ds4_server_render_arguments_free(args);
    while (reader.peek() != 0 and reader.peek() != '}') {
        pushArgument(args, try parseArgument(&reader));
        if (reader.peek() == ',') reader.cursor += 1;
    }
    if (reader.peek() != '}') return error.InvalidJson;
    // Existing callers allow a trailing suffix; this parser does not consume it.
}
pub export fn ds4_server_render_arguments_parse(raw: [*c]const u8, args: *c.json_args) bool {
    parseArguments(raw, args) catch return false;
    return true;
}

fn escapedClose(tail: []const u8, end: []const u8) bool {
    if (tail.len == 0 or tail[0] != '&') return false;
    var remaining = tail[1..];
    while (std.mem.startsWith(u8, remaining, "amp;")) remaining = remaining[4..];
    return std.mem.startsWith(u8, remaining, "lt;") and std.mem.startsWith(u8, remaining[3..], end[1..]);
}
fn body(out: *Text, value: [*c]const u8, end: []const u8) void {
    // Tool bodies are not HTML. Escape only their closing sentinel and the
    // spellings which would otherwise collide with that escape.
    const source = bytes(value);
    for (source, 0..) |value_byte, position| {
        const tail = source[position..];
        if (std.mem.startsWith(u8, tail, end) or escapedClose(tail, end)) out.append(if (value_byte == '<') "&lt;" else "&amp;") else out.byte(value_byte);
    }
}
fn attribute(out: *Text, value: [*c]const u8) void {
    for (bytes(value)) |value_byte| switch (value_byte) {
        '&' => out.append("&amp;"),
        '<' => out.append("&lt;"),
        '>' => out.append("&gt;"),
        '"' => out.append("&quot;"),
        else => out.byte(value_byte),
    };
}
const calls_open = "<｜DSML｜ calls>";
const calls_close = "</｜DSML｜ calls>";
const invoke_open = "<｜DSML｜ invoke";
const invoke_close = "</｜DSML｜ invoke>";
const parameter_open = "<｜DSML｜ parameter";
const parameter_close = "</｜DSML｜ parameter>";

fn dsmlArgument(out: *Text, argument: *const c.json_arg, v41: bool) void {
    out.append(if (v41) parameter_open else "<｜DSML｜parameter");
    out.append(" name=\"");
    attribute(out, argument.key);
    out.append(if (argument.is_string) "\" string=\"true\">" else "\" string=\"false\">");
    const end = if (v41) parameter_close else "</｜DSML｜parameter>";
    if (argument.is_string) body(out, argument.value, end) else {
        const value = bytes(argument.value);
        for (value, 0..) |value_byte, position| {
            if (std.mem.startsWith(u8, value[position..], end)) out.append("\\u003c") else out.byte(value_byte);
        }
    }
    out.append(end);
    out.byte('\n');
}
fn orderedArguments(out: *Text, raw: [*c]const u8, v41: bool) bool {
    var args = std.mem.zeroes(c.json_args);
    if (!ds4_server_render_arguments_parse(raw, &args)) return false;
    defer ds4_server_render_arguments_free(&args);
    for (items(c.json_arg, args.v, @intCast(args.len))) |*argument| {
        dsmlArgument(out, argument, v41);
    }
    return true;
}
fn jsonPair(out: *Text, argument: *const c.json_arg) void {
    out.quoted(argument.key);
    out.byte(':');
    if (argument.is_string) out.quoted(argument.value) else out.string(argument.value);
}
fn object(out: *Text, raw: [*c]const u8) void {
    var args = std.mem.zeroes(c.json_args);
    if (!ds4_server_render_arguments_parse(raw, &args)) {
        out.append("{}");
        return;
    }
    defer ds4_server_render_arguments_free(&args);
    out.byte('{');
    for (items(c.json_arg, args.v, @intCast(args.len)), 0..) |*argument, position| {
        if (position != 0) out.byte(',');
        jsonPair(out, argument);
    }
    out.byte('}');
}
fn callSlice(calls: [*c]const c.tool_calls) []const c.tool_call {
    return if (calls == null) &.{} else items(c.tool_call, calls.*.v, @intCast(calls.*.len));
}
fn dsmlCalls(out: *Text, calls: [*c]const c.tool_calls, v41: bool) void {
    const list = callSlice(calls);
    if (list.len == 0) return;
    if (bytes(calls.*.raw_tool_text).len != 0) {
        out.string(calls.*.raw_tool_text);
        return;
    }
    out.append(if (v41) "\n\n" ++ calls_open ++ "\n" else "\n\n<｜DSML｜tool_calls>\n");
    for (list) |call| {
        out.append(if (v41) invoke_open else "<｜DSML｜invoke");
        out.append(" name=\"");
        attribute(out, call.name);
        out.append("\">\n");
        if (!orderedArguments(out, call.arguments, v41)) {
            out.append(if (v41) parameter_open else "<｜DSML｜parameter");
            out.append(" name=\"arguments\" string=\"true\">");
            body(out, call.arguments, if (v41) parameter_close else "</｜DSML｜parameter>");
            out.append(if (v41) parameter_close ++ "\n" else "</｜DSML｜parameter>\n");
        }
        out.append(if (v41) invoke_close ++ "\n" else "</｜DSML｜invoke>\n");
    }
    out.append(if (v41) calls_close else "</｜DSML｜tool_calls>");
}
const v41_tools_preamble = Prompt.tools_preamble;
const legacy_tools_preamble =
    "## Tools\n\n" ++
    "You have access to a set of tools to help answer the user question. " ++
    "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n" ++
    "<｜DSML｜tool_calls>\n" ++
    "<｜DSML｜invoke name=\"$TOOL_NAME\">\n" ++
    "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n" ++
    "...\n" ++
    "</｜DSML｜invoke>\n" ++
    "<｜DSML｜invoke name=\"$TOOL_NAME2\">\n" ++
    "...\n" ++
    "</｜DSML｜invoke>\n" ++
    "</｜DSML｜tool_calls>\n\n" ++
    "String parameters should be specified as raw text and set `string=\"true\"`. " ++
    "Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal string characters with XML or HTML entity escapes. " ++
    "Only if a string value itself contains the exact closing parameter tag `</｜DSML｜parameter>`, write that tag as `&lt;/｜DSML｜parameter>` inside the value. " ++
    "For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n" ++
    "When thinking mode is enabled, finish reasoning with </think> before any tool calls or final response.\n\n" ++
    "Otherwise, output directly after </think> with tool calls or final response.\n\n" ++
    "### Available Tool Schemas\n\n";
const tools_footer = Prompt.tools_footer;

fn tools(out: *Text, schemas: [*c]const u8, v41: bool) void {
    if (bytes(schemas).len == 0) return;
    out.append(if (v41) v41_tools_preamble else legacy_tools_preamble);
    out.string(schemas);
    out.append(tools_footer);
}
fn messageSlice(messages: [*c]const c.chat_msgs) []const c.chat_msg {
    return if (messages == null) &.{} else items(c.chat_msg, messages.*.v, @intCast(messages.*.len));
}
/// Compatibility entry for callers still owning the C logical records. The
/// model implementation sees only native borrowed conversation values.
pub export fn ds4_server_render_deepseek41(messages: [*c]const c.chat_msgs, start: c_int, schemas: [*c]const u8, mode: c.ds4_think_mode, live_tail: bool, initial: [*c]const c.stop_list) [*c]u8 {
    var view = Canonical.View.init(allocator, c, messages, null) catch fatal();
    defer view.deinit();
    const initial_order = view.strings(initial) catch fatal();
    const prompt = Prompt.render(allocator, view.messages, .{
        .start = @intCast(start),
        .schemas = if (schemas == null) null else bytes(schemas),
        .reasoning = Canonical.reasoning(c, mode),
        .live_tail = live_tail,
        .initial_tool_order = initial_order,
    }) catch fatal();
    return prompt.ptr;
}

pub export fn ds4_server_render_flint(messages: [*c]const c.chat_msgs, start: c_int, schemas: [*c]const u8) [*c]u8 {
    var view = Canonical.View.init(allocator, c, messages, null) catch fatal();
    defer view.deinit();
    const prompt = FlintPrompt.render(allocator, view.messages, .{
        .start = @intCast(@max(start, 0)),
        .schemas = if (schemas == null) null else bytes(schemas),
    }) catch fatal();
    return prompt.ptr;
}

pub export fn ds4_server_render_qwen(messages: [*c]const c.chat_msgs, start: c_int, schemas: [*c]const u8, orders: [*c]const c.tool_schema_orders, mode: c.ds4_think_mode, live_tail: bool) [*c]u8 {
    var view = Canonical.View.init(allocator, c, messages, orders) catch fatal();
    defer view.deinit();
    return (QwenPrompt.render(allocator, view.messages, .{
        .start = @intCast(@max(start, 0)),
        .schemas = if (schemas == null) null else bytes(schemas),
        .tools = view.tools,
        .reasoning = Canonical.reasoning(c, mode),
        .live_tail = live_tail,
    }) catch fatal()).ptr;
}

pub export fn ds4_server_render_glm(messages: [*c]const c.chat_msgs, start: c_int, schemas: [*c]const u8, orders: [*c]const c.tool_schema_orders, mode: c.ds4_think_mode, live_tail: bool) [*c]u8 {
    var view = Canonical.View.init(allocator, c, messages, orders) catch fatal();
    defer view.deinit();
    return (GlmPrompt.render(allocator, view.messages, .{
        .start = @intCast(@max(start, 0)),
        .schemas = if (schemas == null) null else bytes(schemas),
        .tools = view.tools,
        .reasoning = Canonical.reasoning(c, mode),
        .live_tail = live_tail,
    }) catch fatal()).ptr;
}

pub export fn ds4_server_render_flint_calls(out: *c.buf, calls: [*c]const c.tool_calls, orders: [*c]const c.tool_schema_orders) void {
    if (calls == null or calls.*.len == 0) return;
    var message = std.mem.zeroes(c.chat_msg);
    message.role = @constCast("assistant");
    message.calls = calls.*;
    const messages: c.chat_msgs = .{ .v = &message, .len = 1, .cap = 1 };
    var view = Canonical.View.init(allocator, c, &messages, orders) catch fatal();
    defer view.deinit();
    const rendered = FlintPrompt.render(allocator, view.messages, .{ .add_generation_prompt = false }) catch fatal();
    defer allocator.free(rendered);
    var text = Text.adopt(out);
    text.append(rendered["<|assistant|>".len..]);
    text.store(out);
}

fn replayMessage(out: *Text, message: *const c.chat_msg) void {
    out.append("{\"role\":");
    out.quoted(if (message.role == null) "user" else message.role);
    out.append(",\"content\":");
    out.quoted(if (message.content == null) "" else message.content);
    if (bytes(message.reasoning).len != 0) {
        out.append(",\"reasoning_content\":");
        out.quoted(message.reasoning);
    }
    if (message.tool_call_id != null) {
        out.append(",\"tool_call_id\":");
        out.quoted(message.tool_call_id);
    }
    if (message.calls.len != 0) {
        out.append(",\"tool_calls\":[");
        for (callSlice(&message.calls), 0..) |call, position| {
            if (position != 0) out.byte(',');
            out.append("{\"id\":");
            out.quoted(if (call.id == null) "" else call.id);
            out.append(",\"type\":\"function\",\"function\":{\"name\":");
            out.quoted(if (call.name == null) "" else call.name);
            out.append(",\"arguments\":");
            out.quoted(if (call.arguments == null) "{}" else call.arguments);
            out.append("}}");
        }
        out.byte(']');
    }
    if (message.calls.raw_tool_text != null) {
        out.append(",\"raw_tool_text\":");
        out.quoted(message.calls.raw_tool_text);
    }
    if (message.discovered_tools != null) {
        out.append(",\"ds4_discovered_tools\":");
        out.string(message.discovered_tools);
    }
    out.byte('}');
}
pub export fn ds4_server_render_replay_history(messages: [*c]const c.chat_msgs, skip: c_int) [*c]u8 {
    var out: Text = .{};
    defer out.deinit();
    out.byte('[');
    const source = messageSlice(messages);
    const begin: usize = @intCast(skip);
    if (begin < source.len) for (source[begin..], 0..) |*message, position| {
        if (position != 0) out.byte(',');
        replayMessage(&out, message);
    };
    out.byte(']');
    return out.take();
}

// Narrow C append entries adopt the one actual buf allocation temporarily.
// They transfer it back on return; there is no second persistent text mirror.
pub export fn ds4_server_render_tools(buffer: *c.buf, schemas: [*c]const u8, v41: bool) void {
    var out = Text.adopt(buffer);
    defer out.store(buffer);
    tools(&out, schemas, v41);
}
pub export fn ds4_server_render_glm_body(buffer: *c.buf, value: [*c]const u8, end: [*c]const u8) void {
    var out = Text.adopt(buffer);
    defer out.store(buffer);
    body(&out, value, bytes(end));
}
pub export fn ds4_server_render_glm_result(buffer: *c.buf, value: [*c]const u8) void {
    var out = Text.adopt(buffer);
    defer out.store(buffer);
    body(&out, value, "</tool_response>");
}
pub export fn ds4_server_render_tool_result(buffer: *c.buf, value: [*c]const u8) void {
    var out = Text.adopt(buffer);
    defer out.store(buffer);
    body(&out, value, "</tool_result>");
}
pub export fn ds4_server_render_object(buffer: *c.buf, raw: [*c]const u8) void {
    var out = Text.adopt(buffer);
    defer out.store(buffer);
    object(&out, raw);
}
pub export fn ds4_server_render_dsml_calls(buffer: *c.buf, calls: [*c]const c.tool_calls, v41: bool) void {
    var out = Text.adopt(buffer);
    defer out.store(buffer);
    dsmlCalls(&out, calls, v41);
}
pub export fn ds4_server_render_glm_calls(buffer: *c.buf, calls: [*c]const c.tool_calls, orders: [*c]const c.tool_schema_orders) void {
    if (calls == null or calls.*.len == 0) return;
    var message = std.mem.zeroes(c.chat_msg);
    message.role = @constCast("assistant");
    message.calls = calls.*;
    const messages: c.chat_msgs = .{ .v = &message, .len = 1, .cap = 1 };
    var view = Canonical.View.init(allocator, c, &messages, orders) catch fatal();
    defer view.deinit();
    const rendered = GlmPrompt.renderCalls(allocator, view.messages[0].calls, view.messages[0].raw_tool_text, view.tools) catch fatal();
    defer allocator.free(rendered);
    var out = Text.adopt(buffer);
    defer out.store(buffer);
    out.append(rendered);
}
pub export fn ds4_server_render_qwen_calls(buffer: *c.buf, calls: [*c]const c.tool_calls, has_content: bool, orders: [*c]const c.tool_schema_orders) void {
    if (calls == null or calls.*.len == 0) return;
    var message = std.mem.zeroes(c.chat_msg);
    message.role = @constCast("assistant");
    message.calls = calls.*;
    const messages: c.chat_msgs = .{ .v = &message, .len = 1, .cap = 1 };
    var view = Canonical.View.init(allocator, c, &messages, orders) catch fatal();
    defer view.deinit();
    const rendered = QwenPrompt.renderCalls(allocator, view.messages[0], view.tools, has_content) catch fatal();
    defer allocator.free(rendered);
    var out = Text.adopt(buffer);
    defer out.store(buffer);
    out.append(rendered);
}
pub export fn ds4_server_render_replay_message(buffer: *c.buf, message: *const c.chat_msg) void {
    var out = Text.adopt(buffer);
    defer out.store(buffer);
    replayMessage(&out, message);
}
