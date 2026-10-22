//! Server prompt/replay and tool argument formatting over the real records.
//! Tolerant primitive JSON parsing stays in its existing library. Rendering,
//! argument ownership, Anthropic split views and call-order policy live here.
const std = @import("std");
const c = @import("server_render_c");
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
fn glmArgument(out: *Text, argument: *const c.json_arg) void {
    out.append("<arg_key>");
    body(out, argument.key, "</arg_key>");
    out.append("</arg_key><arg_value>");
    body(out, argument.value, "</arg_value>");
    out.append("</arg_value>");
}
fn orderedArguments(out: *Text, raw: [*c]const u8, order: [*c]const c.tool_schema_order, v41: bool, glm: bool) bool {
    var args = std.mem.zeroes(c.json_args);
    if (!ds4_server_render_arguments_parse(raw, &args)) return false;
    defer ds4_server_render_arguments_free(&args);
    if (order != null) for (items([*c]u8, order.*.prop, @intCast(order.*.len))) |property| {
        const found = ds4_server_render_arguments_find(&args, property);
        if (found < 0) continue;
        const argument: *c.json_arg = @ptrCast(&args.v[@intCast(found)]);
        if (glm) glmArgument(out, argument) else dsmlArgument(out, argument, v41);
        argument.used = true;
    };
    for (items(c.json_arg, args.v, @intCast(args.len))) |*argument| {
        if (argument.used) continue;
        if (glm) glmArgument(out, argument) else dsmlArgument(out, argument, v41);
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
fn findOrder(orders: [*c]const c.tool_schema_orders, name: [*c]const u8) [*c]const c.tool_schema_order {
    if (orders == null or name == null) return null;
    for (items(c.tool_schema_order, orders.*.v, @intCast(orders.*.len))) |*order| {
        if (order.name != null and equal(order.name, bytes(name))) return order;
    }
    return null;
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
        if (!orderedArguments(out, call.arguments, null, v41, false)) {
            out.append(if (v41) parameter_open else "<｜DSML｜parameter");
            out.append(" name=\"arguments\" string=\"true\">");
            body(out, call.arguments, if (v41) parameter_close else "</｜DSML｜parameter>");
            out.append(if (v41) parameter_close ++ "\n" else "</｜DSML｜parameter>\n");
        }
        out.append(if (v41) invoke_close ++ "\n" else "</｜DSML｜invoke>\n");
    }
    out.append(if (v41) calls_close else "</｜DSML｜tool_calls>");
}
fn glmCalls(out: *Text, calls: [*c]const c.tool_calls, orders: [*c]const c.tool_schema_orders) void {
    const list = callSlice(calls);
    if (list.len == 0) return;
    if (bytes(calls.*.raw_tool_text).len != 0) {
        out.string(calls.*.raw_tool_text);
        return;
    }
    // GLM emits this inline. Added newlines would change the replayed prefix.
    for (list) |call| {
        out.append("<tool_call>");
        out.string(call.name);
        if (!orderedArguments(out, call.arguments, findOrder(orders, call.name), false, true)) {
            out.append("<arg_key>arguments</arg_key><arg_value>");
            body(out, call.arguments, "</arg_value>");
            out.append("</arg_value>");
        }
        out.append("</tool_call>");
    }
}
fn qwenCalls(out: *Text, calls: [*c]const c.tool_calls, has_content: bool, orders: [*c]const c.tool_schema_orders) void {
    const list = callSlice(calls);
    if (list.len == 0) return;
    if (bytes(calls.*.raw_tool_text).len != 0) {
        out.string(calls.*.raw_tool_text);
        return;
    }
    for (list, 0..) |call, position| {
        out.append(if (position != 0) "\n<tool_call>\n<function=" else if (has_content) "\n\n<tool_call>\n<function=" else "<tool_call>\n<function=");
        out.string(call.name);
        out.append(">\n");
        var args = std.mem.zeroes(c.json_args);
        if (ds4_server_render_arguments_parse(call.arguments, &args)) {
            defer ds4_server_render_arguments_free(&args);
            const order = findOrder(orders, call.name);
            for (0..2) |pass| {
                const count: usize = if (pass == 0) (if (order == null) 0 else @intCast(order.*.len)) else @intCast(args.len);
                for (0..count) |property| {
                    const found: c_int = if (pass == 0) ds4_server_render_arguments_find(&args, order.*.prop[property]) else @intCast(property);
                    if (found < 0) continue;
                    const argument: *c.json_arg = @ptrCast(&args.v[@intCast(found)]);
                    if (pass != 0 and argument.used) continue;
                    argument.used = true;
                    out.append("<parameter=");
                    body(out, argument.key, ">");
                    out.append(">\n");
                    if (argument.is_string) body(out, argument.value, "</parameter>") else {
                        const raw = c.ds4_server_json_minify(if (argument.value == null) "null" else argument.value);
                        defer c.free(raw);
                        body(out, if (bytes(raw).len == 0) "null" else raw, "</parameter>");
                    }
                    out.append("\n</parameter>\n");
                }
            }
        } else if (bytes(call.arguments).len != 0) {
            out.append("<parameter=arguments>\n");
            body(out, call.arguments, "</parameter>");
            out.append("\n</parameter>\n");
        }
        out.append("</function>\n</tool_call>");
    }
}

const v41_tools_preamble =
    "## Tools\n\n" ++
    "You can invoke tools using this format:\n\n" ++
    "<｜DSML｜ calls>" ++
    "\n" ++
    "<｜DSML｜ invoke" ++
    " name=\"$TOOL_NAME\">\n" ++
    "<｜DSML｜ parameter" ++
    " name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE" ++
    "</｜DSML｜ parameter>" ++
    "\n" ++
    "</｜DSML｜ invoke>" ++
    "\n" ++
    "</｜DSML｜ calls>" ++
    "\n\n" ++
    "String values use string=\"true\"; all other values use JSON and string=\"false\". " ++
    "Inside string values only, escape a literal </｜DSML｜ parameter> as &lt;/｜DSML｜ parameter>. " ++
    "To write that escaped spelling literally, use &amp;lt;/｜DSML｜ parameter>. Other HTML entities are unchanged.\n\n" ++
    "Finish reasoning with </think> before tool calls or a final response.\n\n" ++
    "### Available Tool Schemas\n\n";
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
const tools_footer =
    "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. " ++
    "Use the exact parameter names from the schemas.";

fn tools(out: *Text, schemas: [*c]const u8, v41: bool) void {
    if (bytes(schemas).len == 0) return;
    out.append(if (v41) v41_tools_preamble else legacy_tools_preamble);
    out.string(schemas);
    out.append(tools_footer);
}
fn glmSchema(out: *Text, raw: [*c]const u8, emitted: *bool) bool {
    emitted.* = false;
    var args = std.mem.zeroes(c.json_args);
    if (!ds4_server_render_arguments_parse(raw, &args)) return false;
    defer ds4_server_render_arguments_free(&args);
    const arguments = items(c.json_arg, args.v, @intCast(args.len));
    for (arguments) |argument| {
        if (equal(argument.key, "defer_loading") and !argument.is_string and equal(argument.value, "true")) return true;
    }
    out.byte('{');
    var wrote = false;
    for (arguments) |argument| {
        if (equal(argument.key, "defer_loading") or equal(argument.key, "strict")) continue;
        if (wrote) out.append(", ");
        out.quoted(argument.key);
        out.append(": ");
        if (argument.is_string) out.quoted(argument.value) else out.string(argument.value);
        wrote = true;
    }
    out.byte('}');
    emitted.* = true;
    return true;
}
fn glmTools(out: *Text, schemas: [*c]const u8) void {
    if (bytes(schemas).len == 0) return;
    out.append("\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" ++
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n\n");
    var reader: Reader = .{ .cursor = schemas };
    var any = false;
    while (reader.peek() != 0) {
        const raw = reader.raw() catch break;
        defer c.free(raw);
        var emitted = false;
        if (!glmSchema(out, raw, &emitted)) {
            out.string(raw);
            emitted = true;
        }
        if (emitted) {
            out.append("\n\n\n");
            any = true;
        }
    }
    if (!any) out.byte('\n');
    out.append("</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n" ++
        "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>" ++
        "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>");
}

fn messageSlice(messages: [*c]const c.chat_msgs) []const c.chat_msg {
    return if (messages == null) &.{} else items(c.chat_msg, messages.*.v, @intCast(messages.*.len));
}
fn userLike(message: *const c.chat_msg) bool {
    return equal(message.role, "user") or equal(message.role, "tool") or equal(message.role, "function");
}
fn systemLike(message: *const c.chat_msg) bool {
    return equal(message.role, "system") or equal(message.role, "developer");
}
fn toolContext(messages: []const c.chat_msg, schemas: [*c]const u8) bool {
    if (bytes(schemas).len != 0) return true;
    for (messages) |message| {
        if ((equal(message.role, "assistant") and message.calls.len > 0) or equal(message.role, "tool") or equal(message.role, "function")) return true;
    }
    return false;
}

const Views = struct {
    messages: std.ArrayList(c.chat_msg) = .empty,
    strings: std.ArrayList([*c]u8) = .empty,
    start: usize = 0,

    fn deinit(self: *Views) void {
        for (self.strings.items) |value| c.free(value);
        self.strings.deinit(allocator);
        self.messages.deinit(allocator);
    }
    fn owned(self: *Views, value: []const u8) [*c]u8 {
        const result = duplicate(value);
        self.strings.append(allocator, result) catch fatal();
        return result;
    }
    fn append(self: *Views, message: c.chat_msg) void {
        self.messages.append(allocator, message) catch fatal();
    }
    fn init(source: []const c.chat_msg, start: usize) Views {
        var views: Views = .{};
        for (source, 0..) |message, position| {
            if (position == start) views.start = views.messages.items.len;
            if (!equal(message.role, "user") or message.tool_results_len == 0) {
                views.append(message);
                continue;
            }
            // Most views borrow the original message. Only split Anthropic
            // blocks own strings, leaving the source history/images untouched.
            const content = bytes(message.content);
            const spans = items(c.tool_result_span, message.tool_results, @intCast(message.tool_results_len));
            var begin: usize = 0;
            for (0..spans.len + 1) |part| {
                const span: ?*const c.tool_result_span = if (part < spans.len) &spans[part] else null;
                const end = if (span) |result| result.begin else content.len;
                if (end > begin) {
                    var text = std.mem.zeroes(c.chat_msg);
                    text.role = views.owned("user");
                    text.content = views.owned(content[begin..end]);
                    views.append(text);
                }
                if (span) |result| {
                    var tool = std.mem.zeroes(c.chat_msg);
                    tool.role = views.owned("tool");
                    tool.content = views.owned(bytes(result.content));
                    tool.tool_call_id = if (result.id == null) null else views.owned(bytes(result.id));
                    views.append(tool);
                    begin = result.end;
                }
            }
        }
        if (start >= source.len) views.start = views.messages.items.len;
        return views;
    }
};

const RankedResult = struct {
    message: *const c.chat_msg,
    order: usize,
    position: usize,

    fn before(_: void, left: RankedResult, right: RankedResult) bool {
        return if (left.order != right.order) left.order < right.order else left.position < right.position;
    }
};
fn orderedMessages(messages: []const c.chat_msg, initial: [*c]const c.stop_list) []*const c.chat_msg {
    const ordered = allocator.alloc(*const c.chat_msg, messages.len) catch fatal();
    var results: std.ArrayList(RankedResult) = .empty;
    defer results.deinit(allocator);
    var order: std.StringHashMap(usize) = .init(allocator);
    defer order.deinit();
    if (initial != null) for (items([*c]u8, initial.*.v, @intCast(initial.*.len)), 0..) |id, rank| {
        order.put(bytes(id), rank) catch fatal();
    };
    var position: usize = 0;
    while (position < messages.len) {
        const message = &messages[position];
        ordered[position] = message;
        if (equal(message.role, "assistant") and message.calls.len != 0) {
            order.clearRetainingCapacity();
            for (callSlice(&message.calls), 0..) |call, rank| {
                if (bytes(call.id).len != 0) order.put(bytes(call.id), rank) catch fatal();
            }
        }
        if (!userLike(message)) {
            position += 1;
            continue;
        }
        const begin = position;
        results.clearRetainingCapacity();
        while (position < messages.len and userLike(&messages[position])) : (position += 1) {
            const result = &messages[position];
            ordered[position] = result;
            if (!equal(result.role, "user")) results.append(allocator, .{
                .message = result,
                // Missing IDs sort at rank zero, matching the original rax
                // lookup; original positions break ties deterministically.
                .order = if (result.tool_call_id == null) 0 else order.get(bytes(result.tool_call_id)) orelse 0,
                .position = position,
            }) catch fatal();
        }
        std.mem.sortUnstable(RankedResult, results.items, {}, RankedResult.before);
        var next: usize = 0;
        for (begin..position) |slot| {
            if (equal(messages[slot].role, "user")) continue;
            ordered[slot] = results.items[next].message;
            next += 1;
        }
    }
    return ordered;
}

pub export fn ds4_server_render_deepseek41(messages: [*c]const c.chat_msgs, start: c_int, schemas: [*c]const u8, mode: c.ds4_think_mode, live_tail: bool, initial: [*c]const c.stop_list) [*c]u8 {
    var views = Views.init(messageSlice(messages), @intCast(start));
    defer views.deinit();
    const ordered = orderedMessages(views.messages.items, initial);
    defer allocator.free(ordered);
    const think = c.ds4_think_mode_enabled(mode);
    const keep_reasoning = live_tail or toolContext(views.messages.items, schemas);
    const effort = c.ds4_deepseek41_reasoning_effort_text(mode);
    var out: Text = .{};
    defer out.deinit();
    var initial_system = false;
    var pending = false;
    var user_open = false;
    var last_user: ?usize = null;
    for (views.messages.items, 0..) |*message, position| {
        if (userLike(message) or (position > 0 and systemLike(message))) last_user = position;
    }
    if (live_tail) out.append("<｜end▁of▁sentence｜>") else {
        out.append("<｜begin▁of▁sentence｜>");
        if (effort != null or bytes(schemas).len != 0) {
            out.append("<｜System｜>");
            initial_system = true;
            out.string(effort);
            if (bytes(schemas).len != 0) tools(&out, schemas, true);
        }
    }
    // Full requests and live tails share these separators and transitions;
    // they must agree exactly at the authoritative live-KV boundary.
    for (views.start..ordered.len) |position| {
        const message = ordered[position];
        if (systemLike(message)) {
            if (position == 0 and initial_system) {
                if (bytes(schemas).len != 0) out.append("\n\n");
            } else out.append("<｜System｜>");
            out.string(message.content);
            pending = position > 0 or live_tail;
            user_open = false;
        } else if (userLike(message)) {
            out.append(if (user_open) "\n\n" else "<｜User｜>");
            if (!equal(message.role, "user")) {
                out.append("<tool_result>");
                body(&out, message.content, "</tool_result>");
                out.append("</tool_result>");
            } else out.string(message.content);
            pending = true;
            user_open = true;
        } else if (equal(message.role, "assistant")) {
            if (pending) {
                out.append("<｜Assistant｜>");
                if (think and (keep_reasoning or last_user == null or position > last_user.?)) {
                    out.append("<think>");
                    out.string(message.reasoning);
                }
                out.append("</think>");
            }
            out.string(message.content);
            dsmlCalls(&out, &message.calls, true);
            out.append("<｜end▁of▁sentence｜>");
            pending = false;
            user_open = false;
        }
    }
    if (pending) {
        out.append("<｜Assistant｜>");
        out.append(if (think) "<think>" else "</think>");
    }
    return out.take();
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
pub export fn ds4_server_render_glm_schema(buffer: *c.buf, raw: [*c]const u8, emitted: *bool) bool {
    var out = Text.adopt(buffer);
    defer out.store(buffer);
    return glmSchema(&out, raw, emitted);
}
pub export fn ds4_server_render_glm_tools(buffer: *c.buf, schemas: [*c]const u8) void {
    var out = Text.adopt(buffer);
    defer out.store(buffer);
    glmTools(&out, schemas);
}
pub export fn ds4_server_render_glm_body(buffer: *c.buf, value: [*c]const u8, end: [*c]const u8) void {
    var out = Text.adopt(buffer);
    defer out.store(buffer);
    body(&out, value, bytes(end));
}
pub export fn ds4_server_render_glm_value(buffer: *c.buf, value: [*c]const u8) void {
    var out = Text.adopt(buffer);
    defer out.store(buffer);
    body(&out, value, "</arg_value>");
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
    var out = Text.adopt(buffer);
    defer out.store(buffer);
    glmCalls(&out, calls, orders);
}
pub export fn ds4_server_render_qwen_calls(buffer: *c.buf, calls: [*c]const c.tool_calls, has_content: bool, orders: [*c]const c.tool_schema_orders) void {
    var out = Text.adopt(buffer);
    defer out.store(buffer);
    qwenCalls(&out, calls, has_content, orders);
}
pub export fn ds4_server_render_replay_message(buffer: *c.buf, message: *const c.chat_msg) void {
    var out = Text.adopt(buffer);
    defer out.store(buffer);
    replayMessage(&out, message);
}
