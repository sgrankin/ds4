//! Completed generated-reply decoding. Tool-shaped text inside argument values
//! stays literal data; executable calls are taken from the post-reasoning surface.
//! All strings/call vectors use libc ownership and the actual server records.
const std = @import("std");
const c = @import("server_reply_c");
const allocator = std.heap.c_allocator;
const Invalid = error{InvalidToolCall};

fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
    c.exit(1);
}
fn bytes(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) "" else std.mem.span(pointer);
}
fn duplicate(value: []const u8) [*c]u8 {
    return (allocator.dupeSentinel(u8, value, 0) catch fatal()).ptr;
}
fn items(comptime T: type, pointer: [*c]T, count: usize) []T {
    // A cleared vector may still own capacity; retain its allocation address.
    return if (count == 0 and pointer == null) &.{} else @as([*]T, @ptrCast(pointer))[0..count];
}
fn find(text: []const u8, start: usize, needle: []const u8) ?usize {
    return std.mem.findPos(u8, text, start, needle);
}
fn at(text: []const u8, position: usize, value: []const u8) bool {
    return position <= text.len and std.mem.startsWith(u8, text[position..], value);
}
fn skipWhitespace(text: []const u8, start: usize) usize {
    var position = start;
    while (position < text.len and c.isspace(text[position]) != 0) position += 1;
    return position;
}
pub fn trimSeparator(text: [*c]const u8, start: usize, limit: usize) usize {
    var end = limit;
    while (end > start and c.isspace(text[end - 1]) != 0) end -= 1;
    return end;
}
fn trim(text: []const u8) []const u8 {
    const begin = skipWhitespace(text, 0);
    var end = text.len;
    while (end > begin and c.isspace(text[end - 1]) != 0) end -= 1;
    return text[begin..end];
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
    fn quoted(self: *Text, value: [*c]const u8) void {
        const hex = "0123456789abcdef";
        self.byte('"');
        for (bytes(value)) |character| switch (character) {
            '"', '\\' => self.append(&.{ '\\', character }),
            '\n' => self.append("\\n"),
            '\r' => self.append("\\r"),
            '\t' => self.append("\\t"),
            0...8, 11, 12, 14...31 => self.append(&.{ '\\', 'u', '0', '0', hex[character >> 4], hex[character & 15] }),
            else => self.byte(character),
        };
        self.byte('"');
    }
    fn terminated(self: *Text) [*c]const u8 {
        self.list.ensureUnusedCapacity(allocator, 1) catch fatal();
        self.list.items.ptr[self.list.items.len] = 0;
        return self.list.items.ptr;
    }
    fn take(self: *Text) [*c]u8 {
        return (self.list.toOwnedSliceSentinel(allocator, 0) catch fatal()).ptr;
    }
    fn argument(self: *Text, name: [*c]const u8, value: [*c]const u8, string: bool) void {
        if (self.list.items.len != 0) self.append(", ");
        self.quoted(name);
        self.append(": ");
        if (string) self.quoted(value) else {
            const raw = c.ds4_server_json_minify(if (value == null) "null" else value);
            defer c.free(raw);
            self.append(if (bytes(raw).len == 0) "null" else bytes(raw));
        }
    }
};

const Style = struct {
    open: []const u8,
    close: []const u8,
    invoke: []const u8,
    invoke_end: []const u8,
    parameter: []const u8,
    parameter_end: []const u8,
};
fn styles() [4]Style {
    return .{
        .{ .open = bytes(c.DS4_TOOL_CALLS_START), .close = bytes(c.DS4_TOOL_CALLS_END), .invoke = bytes(c.DS4_INVOKE_START), .invoke_end = bytes(c.DS4_INVOKE_END), .parameter = bytes(c.DS4_PARAM_START), .parameter_end = bytes(c.DS4_PARAM_END) },
        .{ .open = "<tool_calls>", .close = "</tool_calls>", .invoke = "<invoke", .invoke_end = "</invoke>", .parameter = "<parameter", .parameter_end = "</parameter>" },
        .{ .open = bytes(c.DS4_TOOL_CALLS_START_SHORT), .close = bytes(c.DS4_TOOL_CALLS_END_SHORT), .invoke = bytes(c.DS4_INVOKE_START_SHORT), .invoke_end = bytes(c.DS4_INVOKE_END_SHORT), .parameter = bytes(c.DS4_PARAM_START_SHORT), .parameter_end = bytes(c.DS4_PARAM_END_SHORT) },
        .{ .open = bytes(c.DS41_TOOL_CALLS_START), .close = bytes(c.DS41_TOOL_CALLS_END), .invoke = bytes(c.DS41_INVOKE_START), .invoke_end = bytes(c.DS41_INVOKE_END), .parameter = bytes(c.DS41_PARAM_START), .parameter_end = bytes(c.DS41_PARAM_END) },
    };
}
fn firstTool(text: []const u8, end: bool) ?usize {
    const formats = styles();
    const candidates = if (end) [_][]const u8{ formats[3].close, formats[0].close, formats[2].close, "</tool_calls>", "</tool_call>" } else [_][]const u8{ formats[3].open, formats[0].open, formats[2].open, "<tool_calls>", "<tool_call>" };
    var best: ?usize = null;
    for (candidates) |candidate| {
        if (find(text, 0, candidate)) |position| {
            if (best == null or position < best.?) best = position;
        }
    }
    return best;
}
pub fn toolStart(text: [*c]const u8) [*c]const u8 {
    return if (firstTool(bytes(text), false)) |position| text + position else null;
}
pub fn toolEnd(text: [*c]const u8) [*c]const u8 {
    return if (firstTool(bytes(text), true)) |position| text + position else null;
}
pub fn lastSubstring(text: [*c]const u8, needle: [*c]const u8) [*c]const u8 {
    if (text == null or needle == null or bytes(needle).len == 0) return null;
    return if (std.mem.findLast(u8, bytes(text), bytes(needle))) |position| text + position else null;
}
fn structural(text: []const u8, start: usize, needle: []const u8, last: bool) ?usize {
    const formats = styles();
    const wrappers = [_][2][]const u8{
        .{ formats[3].parameter, formats[3].parameter_end },
        .{ formats[0].parameter, formats[0].parameter_end },
        .{ formats[2].parameter, formats[2].parameter_end },
        .{ "<parameter", "</parameter>" },
        .{ "<arg_key>", "</arg_key>" },
        .{ "<arg_value>", "</arg_value>" },
    };
    var position = start;
    var found: ?usize = null;
    while (position < text.len) {
        if (at(text, position, needle)) {
            found = position;
            if (!last) return found;
        }
        var skipped = false;
        for (wrappers) |wrapper| {
            if (!at(text, position, wrapper[0])) continue;
            const tag_end = std.mem.findScalarPos(u8, text, position, '>') orelse return found;
            const end = find(text, tag_end + 1, wrapper[1]) orelse return found;
            position = end + wrapper[1].len;
            skipped = true;
            break;
        }
        if (!skipped) position += 1;
    }
    return found;
}
fn unescapeAttribute(value: []const u8) [*c]u8 {
    const entities = [_][]const u8{ "&amp;", "&lt;", "&gt;", "&quot;", "&apos;" };
    const decoded = "&<>\"'";
    var out: Text = .{};
    defer out.deinit();
    var position: usize = 0;
    while (position < value.len) {
        var consumed = false;
        if (value[position] == '&') for (entities, decoded) |entity, replacement| {
            if (!at(value, position, entity)) continue;
            out.byte(replacement);
            position += entity.len;
            consumed = true;
            break;
        };
        if (!consumed) {
            out.byte(value[position]);
            position += 1;
        }
    }
    return out.take();
}
fn attribute(tag: []const u8, name: [*c]const u8) ?[*c]u8 {
    var pattern: [64]u8 = undefined;
    _ = c.snprintf(&pattern, pattern.len, "%s=\"", name);
    const wanted = std.mem.span(@as([*:0]const u8, @ptrCast(&pattern)));
    const begin = (find(tag, 0, wanted) orelse return null) + wanted.len;
    const end = std.mem.findScalarPos(u8, tag, begin, '"') orelse return null;
    return unescapeAttribute(tag[begin..end]);
}
pub fn attr(tag: [*c]const u8, name: [*c]const u8) [*c]u8 {
    return attribute(bytes(tag), name) orelse null;
}
fn escapedClose(value: []const u8, end: []const u8) bool {
    if (value.len == 0 or value[0] != '&') return false;
    var tail = value[1..];
    while (std.mem.startsWith(u8, tail, "amp;")) tail = tail[4..];
    return std.mem.startsWith(u8, tail, "lt;") and std.mem.startsWith(u8, tail[3..], end[1..]);
}
fn unescapeValue(value: []const u8, end: []const u8) [*c]u8 {
    var out: Text = .{};
    defer out.deinit();
    var position: usize = 0;
    while (position < value.len) {
        if (escapedClose(value[position..], end)) {
            if (at(value, position, "&amp;")) {
                out.byte('&');
                position += 5;
            } else {
                out.byte('<');
                position += 4;
            }
        } else {
            out.byte(value[position]);
            position += 1;
        }
    }
    return out.take();
}

const Outputs = struct {
    content: *[*c]u8,
    reasoning: *[*c]u8,
    calls: *c.tool_calls,

    fn split(self: Outputs, text: []const u8) void {
        const body = if (std.mem.startsWith(u8, text, "<think>")) text[7..] else text;
        if (find(body, 0, "</think>")) |end| {
            self.reasoning.* = duplicate(body[0..end]);
            self.content.* = duplicate(body[end + 8 ..]);
        } else {
            self.reasoning.* = null;
            self.content.* = duplicate(text);
        }
    }
    fn unterminated(self: Outputs, text: []const u8) void {
        self.reasoning.* = duplicate(if (std.mem.startsWith(u8, text, "<think>")) text[7..] else text);
        self.content.* = duplicate("");
    }
    fn finish(self: Outputs, text: []const u8, prefix: usize, recovered: bool) void {
        if (recovered) self.unterminated(text[0..@min(prefix, text.len)]) else self.split(text[0..prefix]);
    }
    fn qwenTrim(self: Outputs) void {
        if (self.reasoning.* != null) {
            var length = bytes(self.reasoning.*).len;
            while (length > 0 and self.reasoning.*[length - 1] == '\n') {
                length -= 1;
                self.reasoning.*[length] = 0;
            }
        }
        if (self.content.* != null) {
            const value = bytes(self.content.*);
            var begin: usize = 0;
            while (begin < value.len and value[begin] == '\n') begin += 1;
            if (begin != 0) {
                const replacement = duplicate(value[begin..]);
                c.free(self.content.*);
                self.content.* = replacement;
            }
        }
    }
    fn push(self: Outputs, name: [*c]u8, arguments: *Text) void {
        var wrapped: Text = .{};
        defer wrapped.deinit();
        wrapped.byte('{');
        wrapped.append(arguments.list.items);
        wrapped.byte('}');
        var calls: std.ArrayList(c.tool_call) = .empty;
        calls.items = items(c.tool_call, self.calls.v, @intCast(self.calls.len));
        calls.capacity = @intCast(self.calls.cap);
        calls.append(allocator, .{ .id = null, .name = name, .arguments = wrapped.take() }) catch fatal();
        self.calls.v = calls.items.ptr;
        self.calls.len = @intCast(calls.items.len);
        self.calls.cap = @intCast(calls.capacity);
    }
    fn raw(self: Outputs, value: []const u8) void {
        c.free(self.calls.raw_tool_text);
        self.calls.raw_tool_text = duplicate(value);
    }
};
const Kind = enum { deepseek, glm, qwen, flint };
const Search = struct { start: usize, recovered: bool };
fn toolSearch(text: []const u8, require_closed: bool, kind: Kind, out: Outputs) ?Search {
    if (!require_closed) return .{ .start = 0, .recovered = false };
    if (structural(text, 0, "</think>", true)) |end| return .{ .start = end + 8, .recovered = false };
    const candidate = if (kind == .deepseek) firstTool(text, false) else find(text, 0, "<tool_call>");
    const complete = if (candidate) |start| (if (kind == .deepseek) firstTool(text[start..], true) != null else find(text, start, "</tool_call>") != null) else false;
    if (!complete) {
        const label: [:0]const u8 = switch (kind) {
            .deepseek => "DSML",
            .glm => "GLM tool calls",
            .qwen => "Qwen tool calls",
            .flint => "Flint tool calls",
        };
        _ = c.fprintf(c.__stderrp, "ds4-server: thinking not closed, ignoring incomplete %s in reasoning\n", label.ptr);
        out.unterminated(text);
        return null;
    }
    return .{ .start = candidate.?, .recovered = true };
}
fn preferredStart(text: []const u8, start: usize, tag: []const u8) ?usize {
    // Preserve the template's preference for a two-newline separator, even if
    // a bare occurrence of this same tag appears earlier.
    const first = find(text, start, tag) orelse return null;
    var position = first;
    while (true) {
        if (position >= start + 2 and std.mem.eql(u8, text[position - 2 .. position], "\n\n")) return position - 2;
        position = find(text, position + 1, tag) orelse return first;
    }
}
fn leaf(text: []const u8, cursor: *usize, format: Style, out: *Text) Invalid!void {
    var position = cursor.*;
    if (!at(text, position, format.parameter)) return error.InvalidToolCall;
    const tag_end = std.mem.findScalarPos(u8, text, position, '>') orelse return error.InvalidToolCall;
    const name = attribute(text[position .. tag_end + 1], "name") orelse return error.InvalidToolCall;
    defer c.free(name);
    const type_name = attribute(text[position .. tag_end + 1], "string");
    defer if (type_name) |value| c.free(value);
    position = tag_end + 1;
    const end = find(text, position, format.parameter_end) orelse return error.InvalidToolCall;
    const string = type_name == null or std.mem.eql(u8, bytes(type_name.?), "true");
    const value = if (string) unescapeValue(text[position..end], format.parameter_end) else duplicate(text[position..end]);
    defer c.free(value);
    out.argument(name, value, string);
    cursor.* = end + format.parameter_end.len;
}
fn nested(text: []const u8, cursor: *usize, format: Style, out: *Text) Invalid!void {
    var members: Text = .{};
    defer members.deinit();
    var position = cursor.*;
    var any = false;
    while (true) {
        position = skipWhitespace(text, position);
        if (!at(text, position, format.parameter)) break;
        try leaf(text, &position, format, &members);
        any = true;
    }
    if (!any) return error.InvalidToolCall;
    out.byte('{');
    out.append(members.list.items);
    out.byte('}');
    cursor.* = position;
}
fn parseDeepseek(text: []const u8, require_closed: bool, out: Outputs) Invalid!void {
    const search = toolSearch(text, require_closed, .deepseek, out) orelse return;
    const formats = styles();
    var format = formats[0];
    var start = preferredStart(text, search.start, format.open);
    if (start == null) {
        format = formats[2];
        start = preferredStart(text, search.start, format.open);
    }
    if (start == null) {
        format = formats[1];
        start = preferredStart(text, search.start, format.open);
    }
    if (preferredStart(text, search.start, formats[3].open)) |v41| {
        if (start == null or v41 < start.?) {
            start = v41;
            format = formats[3];
        }
    }
    if (start == null) {
        out.split(text);
        return;
    }
    const raw_start = start.?;
    const content_length = trimSeparator(text.ptr, 0, raw_start);
    var position = (find(text, raw_start, format.open) orelse return error.InvalidToolCall) + format.open.len;
    while (true) {
        position = skipWhitespace(text, position);
        if (at(text, position, format.close)) {
            out.raw(text[raw_start .. position + format.close.len]);
            out.finish(text, content_length, search.recovered);
            return;
        }
        if (!at(text, position, format.invoke)) return error.InvalidToolCall;
        const tag_end = std.mem.findScalarPos(u8, text, position, '>') orelse return error.InvalidToolCall;
        var name = attribute(text[position .. tag_end + 1], "name") orelse return error.InvalidToolCall;
        defer if (name != null) c.free(name);
        position = tag_end + 1;
        var arguments: Text = .{};
        defer arguments.deinit();
        while (true) {
            position = skipWhitespace(text, position);
            if (at(text, position, format.invoke_end)) {
                position += format.invoke_end.len;
                break;
            }
            if (!at(text, position, format.parameter)) return error.InvalidToolCall;
            const end_tag = std.mem.findScalarPos(u8, text, position, '>') orelse return error.InvalidToolCall;
            const parameter = attribute(text[position .. end_tag + 1], "name") orelse return error.InvalidToolCall;
            defer c.free(parameter);
            const type_name = attribute(text[position .. end_tag + 1], "string");
            defer if (type_name) |value| c.free(value);
            const value_start = end_tag + 1;
            if (type_name == null and at(text, skipWhitespace(text, value_start), format.parameter)) {
                var child: Text = .{};
                defer child.deinit();
                var child_position = value_start;
                try nested(text, &child_position, format, &child);
                arguments.argument(parameter, child.terminated(), false);
                position = skipWhitespace(text, child_position);
                if (at(text, position, format.parameter_end)) position += format.parameter_end.len;
                continue;
            }
            const value_end = find(text, value_start, format.parameter_end) orelse return error.InvalidToolCall;
            const string = type_name == null or std.mem.eql(u8, bytes(type_name.?), "true");
            const value = if (string) unescapeValue(text[value_start..value_end], format.parameter_end) else duplicate(text[value_start..value_end]);
            defer c.free(value);
            arguments.argument(parameter, value, string);
            position = value_end + format.parameter_end.len;
        }
        out.push(name, &arguments);
        name = null;
    }
}

fn schemaAccepts(schema: std.json.Value, kind: []const u8) bool {
    if (schema != .object) return false;
    if (schema.object.get("type")) |value| {
        if (value == .string and std.mem.eql(u8, value.string, kind)) return true;
        if (value == .array) for (value.array.items) |item| {
            if (item == .string and std.mem.eql(u8, item.string, kind)) return true;
        };
    }
    if (std.mem.eql(u8, kind, "string")) {
        if (schema.object.get("enum")) |value| {
            if (value == .array) for (value.array.items) |item| if (item == .string) return true;
        }
    }
    for ([_][]const u8{ "anyOf", "oneOf" }) |key| {
        if (schema.object.get(key)) |value| {
            if (value == .array) for (value.array.items) |item| if (schemaAccepts(item, kind)) return true;
        }
    }
    return false;
}

fn flintString(orders: [*c]const c.tool_schema_orders, name: [*c]const u8, key: [*c]const u8, value: [*c]const u8) bool {
    // Flint's XML strings are raw; other arguments attempt JSON decoding.
    // Nullable string schemas give the exact spelling "null" its JSON meaning.
    if (orders != null and name != null) {
        for (items(c.tool_schema_order, orders.*.v, @intCast(orders.*.len))) |order| {
            if (order.name == null or !std.mem.eql(u8, bytes(order.name), bytes(name))) continue;
            for (items([*c]u8, order.prop, @intCast(order.len)), 0..) |property, index| {
                if (!std.mem.eql(u8, bytes(property), bytes(key))) continue;
                const schema = std.json.parseFromSlice(std.json.Value, allocator, bytes(order.prop_schema[index]), .{}) catch break;
                defer schema.deinit();
                if (std.mem.eql(u8, bytes(value), "null") and schemaAccepts(schema.value, "null")) return false;
                if (schemaAccepts(schema.value, "string")) return true;
                break;
            }
            break;
        }
    }
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes(value), .{}) catch return true;
    parsed.deinit();
    return false;
}

fn parseXml(text: []const u8, require_closed: bool, out: Outputs, flint: bool, orders: [*c]const c.tool_schema_orders) Invalid!void {
    const search = toolSearch(text, require_closed, if (flint) .flint else .glm, out) orelse return;
    const start = find(text, search.start, "<tool_call>") orelse {
        out.split(text);
        return;
    };
    const raw_start = if (start >= 2 and std.mem.eql(u8, text[start - 2 .. start], "\n\n")) start - 2 else start;
    const content_length = trimSeparator(text.ptr, 0, raw_start);
    var position = start;
    while (true) {
        position = skipWhitespace(text, position);
        if (!at(text, position, "<tool_call>")) break;
        position += "<tool_call>".len;
        const close = structural(text, position, "</tool_call>", false) orelse return error.InvalidToolCall;
        const argument = find(text, position, "<arg_key>");
        const name_end = if (argument != null and argument.? <= close) argument.? else close;
        const name_text = trim(text[position..name_end]);
        if (name_text.len == 0) return error.InvalidToolCall;
        if (flint) for (name_text) |byte| {
            if (byte == '<' or std.ascii.isWhitespace(byte)) return error.InvalidToolCall;
        };
        var name: [*c]u8 = duplicate(name_text);
        defer if (name != null) c.free(name);
        position = @intFromPtr(name_text.ptr) + name_text.len - @intFromPtr(text.ptr);
        var arguments: Text = .{};
        defer arguments.deinit();
        while (true) {
            position = skipWhitespace(text, position);
            if (at(text, position, "</tool_call>")) {
                position += "</tool_call>".len;
                break;
            }
            if (!at(text, position, "<arg_key>")) return error.InvalidToolCall;
            position += "<arg_key>".len;
            const key_end = find(text, position, "</arg_key>") orelse return error.InvalidToolCall;
            if (key_end > close) return error.InvalidToolCall;
            const key = if (flint) duplicate(text[position..key_end]) else unescapeValue(trim(text[position..key_end]), "</arg_key>");
            defer c.free(key);
            position = skipWhitespace(text, key_end + "</arg_key>".len);
            if (!at(text, position, "<arg_value>")) return error.InvalidToolCall;
            position += "<arg_value>".len;
            const value_end = find(text, position, "</arg_value>") orelse return error.InvalidToolCall;
            if (value_end > close) return error.InvalidToolCall;
            const value = if (flint) duplicate(text[position..value_end]) else unescapeValue(text[position..value_end], "</arg_value>");
            defer c.free(value);
            arguments.argument(key, value, !flint or flintString(orders, name, key, value));
            position = value_end + "</arg_value>".len;
        }
        out.push(name, &arguments);
        name = null;
        position = skipWhitespace(text, position);
        if (!at(text, position, "<tool_call>")) break;
    }
    if (out.calls.len == 0) return error.InvalidToolCall;
    out.raw(text[raw_start..position]);
    out.finish(text, content_length, search.recovered);
}
fn jsonValue(value: [*c]const u8) bool {
    var cursor = value;
    c.ds4_server_json_ws(&cursor);
    if (cursor[0] == 0 or cursor[0] == '"') return false;
    var raw: [*c]u8 = null;
    if (!c.ds4_server_json_raw(&cursor, &raw)) return false;
    c.free(raw);
    c.ds4_server_json_ws(&cursor);
    return cursor[0] == 0;
}
fn declaredString(orders: [*c]const c.tool_schema_orders, name: [*c]const u8, key: [*c]const u8) bool {
    if (orders == null or name == null) return false;
    for (items(c.tool_schema_order, orders.*.v, @intCast(orders.*.len))) |order| {
        if (order.name == null or !std.mem.eql(u8, bytes(order.name), bytes(name))) continue;
        for (items([*c]u8, order.prop, @intCast(order.len)), 0..) |property, position| {
            if (!std.mem.eql(u8, bytes(property), bytes(key))) continue;
            var schema = std.mem.zeroes(c.json_args);
            defer c.ds4_server_render_arguments_free(&schema);
            var string = false;
            if (c.ds4_server_render_arguments_parse(order.prop_schema[position], &schema)) {
                for (items(c.json_arg, schema.v, @intCast(schema.len))) |argument| {
                    if (std.mem.eql(u8, bytes(argument.key), "type") and argument.is_string and std.mem.eql(u8, bytes(argument.value), "string")) string = true;
                }
            }
            return string;
        }
        return false;
    }
    return false;
}
fn stripValueNewlines(value: []const u8) []const u8 {
    var begin: usize = 0;
    var end = value.len;
    if (begin < end and value[begin] == '\n') begin += 1;
    if (end > begin and value[end - 1] == '\n') end -= 1;
    return value[begin..end];
}
fn parseQwen(text: []const u8, require_closed: bool, out: Outputs, orders: [*c]const c.tool_schema_orders) Invalid!void {
    const search = toolSearch(text, require_closed, .qwen, out) orelse return;
    const start = find(text, search.start, "<tool_call>") orelse {
        out.split(text);
        out.qwenTrim();
        return;
    };
    var raw_start = start;
    while (raw_start > 0 and (text[raw_start - 1] == '\n' or text[raw_start - 1] == ' ')) raw_start -= 1;
    const content_length = trimSeparator(text.ptr, 0, raw_start);
    var position = start;
    while (true) {
        position = skipWhitespace(text, position);
        if (!at(text, position, "<tool_call>")) break;
        position = skipWhitespace(text, position + "<tool_call>".len);
        if (!at(text, position, "<function=")) return error.InvalidToolCall;
        position += "<function=".len;
        const name_end = std.mem.findAnyPos(u8, text, position, "<>") orelse return error.InvalidToolCall;
        if (text[name_end] != '>') return error.InvalidToolCall;
        const name_text = trim(text[position..name_end]);
        if (name_text.len == 0) return error.InvalidToolCall;
        var name: [*c]u8 = duplicate(name_text);
        defer if (name != null) c.free(name);
        position = name_end + 1;
        var arguments: Text = .{};
        defer arguments.deinit();
        while (true) {
            const parameter = skipWhitespace(text, position);
            if (at(text, parameter, "</function>")) {
                position = parameter + "</function>".len;
                break;
            }
            if (!at(text, parameter, "<parameter=")) return error.InvalidToolCall;
            const key_start = parameter + "<parameter=".len;
            const key_end = std.mem.findAnyPos(u8, text, key_start, "<>") orelse return error.InvalidToolCall;
            if (text[key_end] != '>') return error.InvalidToolCall;
            const key = duplicate(trim(text[key_start..key_end]));
            defer c.free(key);
            const value_start = key_end + 1;
            // Only the parameter delimiter ends a value. Tool/function tags
            // inside argument data are literal, not another call boundary.
            const value_end = find(text, value_start, "</parameter>") orelse return error.InvalidToolCall;
            const raw = duplicate(stripValueNewlines(text[value_start..value_end]));
            defer c.free(raw);
            const string = declaredString(orders, name, key) or !jsonValue(raw);
            const value = if (string) unescapeValue(bytes(raw), "</parameter>") else duplicate(bytes(raw));
            defer c.free(value);
            arguments.argument(key, value, string);
            position = value_end + "</parameter>".len;
        }
        position = skipWhitespace(text, position);
        if (!at(text, position, "</tool_call>")) return error.InvalidToolCall;
        position += "</tool_call>".len;
        out.push(name, &arguments);
        name = null;
        position = skipWhitespace(text, position);
        if (!at(text, position, "<tool_call>")) break;
    }
    if (out.calls.len == 0) return error.InvalidToolCall;
    out.raw(text[raw_start..position]);
    out.finish(text, content_length, search.recovered);
    out.qwenTrim();
}

fn parseWithOrders(syntax: c.server_model_syntax, text: [*c]const u8, require_closed: bool, content: *[*c]u8, reasoning: *[*c]u8, calls: *c.tool_calls, orders: [*c]const c.tool_schema_orders) bool {
    const out: Outputs = .{ .content = content, .reasoning = reasoning, .calls = calls };
    switch (syntax) {
        c.SERVER_MODEL_SYNTAX_GLM => parseXml(bytes(text), require_closed, out, false, orders) catch return false,
        c.SERVER_MODEL_SYNTAX_FLINT => parseXml(bytes(text), require_closed and std.mem.startsWith(u8, bytes(text), "<think>"), out, true, orders) catch return false,
        c.SERVER_MODEL_SYNTAX_QWEN => parseQwen(bytes(text), require_closed, out, orders) catch return false,
        else => parseDeepseek(bytes(text), require_closed, out) catch return false,
    }
    return true;
}
pub fn parse(syntax: c.server_model_syntax, text: [*c]const u8, require_closed: bool, content: *[*c]u8, reasoning: *[*c]u8, calls: *c.tool_calls) bool {
    return parseWithOrders(syntax, text, require_closed, content, reasoning, calls, null);
}

pub fn repair(raw: [*c]const u8, length: usize, output: *c.buf) bool {
    if (raw == null or length == 0) return false;
    const text = bytes(raw);
    const scan = if (structural(text, 0, "</think>", true)) |end| end + 8 else 0;
    const first = firstTool(text[scan..], false) orelse return false;
    const formats = styles();
    var selected: ?Style = null;
    for ([_]usize{ 3, 0, 2, 1 }) |candidate| {
        if (at(text, scan + first, formats[candidate].open)) {
            selected = formats[candidate];
            break;
        }
    }
    const format = selected orelse return false;
    const tags = [_][]const u8{ format.open, format.close, format.invoke, format.invoke_end, format.parameter, format.parameter_end };
    var counts: [6]usize = @splat(0);
    var position = scan;
    // Count tags only after the last reasoning close. Supply outer wrappers
    // alone; closing an unfinished invocation/value could invent an action.
    while (position < length) {
        var matched = false;
        for (tags, 0..) |tag, kind| {
            if (!at(text, position, tag)) continue;
            counts[kind] += 1;
            position += tag.len;
            matched = true;
            break;
        }
        if (!matched) position += 1;
    }
    if (counts[2] != counts[3] or counts[4] != counts[5]) return false;
    if (counts[0] == counts[1]) return false;
    // Refuse orphan closing tags before subtracting unsigned counters.
    if (counts[1] > counts[0] or counts[3] > counts[2] or counts[5] > counts[4]) return false;
    var list: std.ArrayList(u8) = .empty;
    list.items = items(u8, output.ptr, output.len);
    list.capacity = output.cap;
    list.appendSlice(allocator, text) catch fatal();
    for (0..counts[0] - counts[1]) |_| list.appendSlice(allocator, format.close) catch fatal();
    list.ensureUnusedCapacity(allocator, 1) catch fatal();
    list.items.ptr[list.items.len] = 0;
    output.* = .{ .ptr = list.items.ptr, .len = list.items.len, .cap = list.capacity };
    return true;
}
fn freeCalls(calls: *c.tool_calls) void {
    for (items(c.tool_call, calls.v, @intCast(calls.len))) |call| {
        c.free(call.id);
        c.free(call.name);
        c.free(call.arguments);
    }
    c.free(@ptrCast(calls.v));
    c.free(calls.raw_tool_text);
    calls.* = std.mem.zeroes(c.tool_calls);
}
pub fn parseForResponse(syntax: c.server_model_syntax, text: [*c]const u8, has_tools: bool, saw_tool_start: bool, require_closed: bool, finish_io: [*c][*c]const u8, error_text: [*c]u8, error_capacity: usize, content: *[*c]u8, reasoning: *[*c]u8, calls: *c.tool_calls, recovered: [*c]bool, orders: [*c]const c.tool_schema_orders) bool {
    if (recovered != null) recovered.* = false;
    if (parseWithOrders(syntax, text, require_closed, content, reasoning, calls, orders)) return true;
    c.free(content.*);
    c.free(reasoning.*);
    content.* = duplicate(bytes(text));
    reasoning.* = null;
    freeCalls(calls);
    // A malformed tool block is model output, not a server failure. Keep its
    // raw text as fallback while the worker may append a tool error and retry.
    const finish: [*c]const u8 = if (finish_io != null and finish_io.* != null) finish_io.* else "stop";
    if (has_tools and saw_tool_start and !std.mem.eql(u8, bytes(finish), "error")) {
        if (finish_io != null) finish_io.* = if (std.mem.eql(u8, bytes(finish), "length")) "length" else "stop";
        if (error_text != null and error_capacity != 0) {
            const message = "invalid tool call";
            const count = @min(message.len, error_capacity - 1);
            @memcpy(error_text[0..count], message[0..count]);
            error_text[count] = 0;
        }
        if (recovered != null) recovered.* = true;
    }
    return false;
}
