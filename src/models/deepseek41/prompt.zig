//! DeepSeek V4.1 prompt formatting over canonical conversation values.
//! Inputs are borrowed for this call. Only the returned prompt is retained;
//! splitting tool-result messages and ordering parallel calls use local views.
const std = @import("std");
const types = @import("serving_types");
const Allocator = std.mem.Allocator;
const Message = types.Message;

pub const Options = struct {
    start: usize = 0,
    schemas: ?[]const u8 = null,
    reasoning: types.Reasoning = .high,
    live_tail: bool = false,
    initial_tool_order: []const []const u8 = &.{},
};

const Text = struct {
    allocator: Allocator,
    list: std.ArrayList(u8) = .empty,

    fn append(self: *Text, value: []const u8) Allocator.Error!void {
        try self.list.appendSlice(self.allocator, value);
    }
    fn byte(self: *Text, value: u8) Allocator.Error!void {
        try self.list.append(self.allocator, value);
    }
    fn string(self: *Text, value: ?[]const u8) Allocator.Error!void {
        try self.append(value orelse "");
    }
};

fn equal(value: ?[]const u8, expected: []const u8) bool {
    return std.mem.eql(u8, value orelse "", expected);
}
fn nonempty(value: ?[]const u8) bool {
    return if (value) |text| text.len != 0 else false;
}
fn userLike(message: *const Message) bool {
    return equal(message.role, "user") or equal(message.role, "tool") or equal(message.role, "function");
}
fn systemLike(message: *const Message) bool {
    return equal(message.role, "system") or equal(message.role, "developer");
}
fn toolContext(messages: []const Message, schemas: ?[]const u8) bool {
    if (nonempty(schemas)) return true;
    for (messages) |message| {
        if ((equal(message.role, "assistant") and message.calls.len > 0) or equal(message.role, "tool") or equal(message.role, "function")) return true;
    }
    return false;
}

const Views = struct {
    messages: []const Message,
    start: usize,

    fn init(allocator: Allocator, source: []const Message, start: usize) Allocator.Error!Views {
        var messages: std.ArrayList(Message) = .empty;
        var begin_message: usize = 0;
        for (source, 0..) |message, position| {
            if (position == start) begin_message = messages.items.len;
            if (!equal(message.role, "user") or message.tool_results.len == 0) {
                try messages.append(allocator, message);
                continue;
            }
            // Anthropic text/tool blocks become ordinary borrowed messages.
            // Their content and call identities still belong to the request.
            const content = message.content orelse "";
            var begin: usize = 0;
            for (0..message.tool_results.len + 1) |part| {
                const span = if (part < message.tool_results.len) &message.tool_results[part] else null;
                const end = if (span) |result| result.begin else content.len;
                if (end > begin) try messages.append(allocator, .{ .role = "user", .content = content[begin..end] });
                if (span) |result| {
                    try messages.append(allocator, .{ .role = "tool", .content = result.content orelse "", .tool_call_id = result.id });
                    begin = result.end;
                }
            }
        }
        if (start >= source.len) begin_message = messages.items.len;
        return .{ .messages = messages.items, .start = begin_message };
    }
};

const RankedResult = struct {
    message: *const Message,
    order: usize,
    position: usize,

    fn before(_: void, left: RankedResult, right: RankedResult) bool {
        return if (left.order != right.order) left.order < right.order else left.position < right.position;
    }
};

fn orderedMessages(allocator: Allocator, messages: []const Message, initial: []const []const u8) Allocator.Error![]*const Message {
    const ordered = try allocator.alloc(*const Message, messages.len);
    var results: std.ArrayList(RankedResult) = .empty;
    var order: std.StringHashMap(usize) = .init(allocator);
    for (initial, 0..) |id, rank| try order.put(id, rank);
    var position: usize = 0;
    while (position < messages.len) {
        const message = &messages[position];
        ordered[position] = message;
        if (equal(message.role, "assistant") and message.calls.len != 0) {
            order.clearRetainingCapacity();
            for (message.calls, 0..) |call, rank| {
                if (nonempty(call.id)) try order.put(call.id.?, rank);
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
            if (!equal(result.role, "user")) try results.append(allocator, .{
                .message = result,
                // Missing IDs retain the old rank-zero behavior; source
                // position deterministically breaks equal-rank ties.
                .order = if (result.tool_call_id) |id| order.get(id) orelse 0 else 0,
                .position = position,
            });
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

fn escapedClose(tail: []const u8, end: []const u8) bool {
    if (tail.len == 0 or tail[0] != '&') return false;
    var remaining = tail[1..];
    while (std.mem.startsWith(u8, remaining, "amp;")) remaining = remaining[4..];
    return std.mem.startsWith(u8, remaining, "lt;") and std.mem.startsWith(u8, remaining[3..], end[1..]);
}
fn body(out: *Text, source: []const u8, end: []const u8) Allocator.Error!void {
    // Escape the closing sentinel and spellings that collide with that escape;
    // ordinary ampersands and HTML entities remain literal model input.
    for (source, 0..) |value, position| {
        const tail = source[position..];
        if (std.mem.startsWith(u8, tail, end) or escapedClose(tail, end)) {
            try out.append(if (value == '<') "&lt;" else "&amp;");
        } else try out.byte(value);
    }
}
fn attribute(out: *Text, value: ?[]const u8) Allocator.Error!void {
    for (value orelse "") |byte| switch (byte) {
        '&' => try out.append("&amp;"),
        '<' => try out.append("&lt;"),
        '>' => try out.append("&gt;"),
        '"' => try out.append("&quot;"),
        else => try out.byte(byte),
    };
}

const calls_open = "<｜DSML｜ calls>";
const calls_close = "</｜DSML｜ calls>";
const invoke_open = "<｜DSML｜ invoke";
const invoke_close = "</｜DSML｜ invoke>";
const parameter_open = "<｜DSML｜ parameter";
const parameter_close = "</｜DSML｜ parameter>";

fn parameter(out: *Text, argument: types.Parameter) Allocator.Error!void {
    try out.append(parameter_open ++ " name=\"");
    try attribute(out, argument.name);
    try out.append(if (argument.is_string) "\" string=\"true\">" else "\" string=\"false\">");
    if (argument.is_string) {
        try body(out, argument.value, parameter_close);
    } else {
        for (argument.value, 0..) |byte, position| {
            if (std.mem.startsWith(u8, argument.value[position..], parameter_close)) {
                try out.append("\\u003c");
            } else try out.byte(byte);
        }
    }
    try out.append(parameter_close ++ "\n");
}

fn calls(out: *Text, values: []const types.ToolCall, raw_tool_text: ?[]const u8) Allocator.Error!void {
    if (values.len == 0) return;
    if (nonempty(raw_tool_text)) {
        try out.string(raw_tool_text);
        return;
    }
    try out.append("\n\n" ++ calls_open ++ "\n");
    for (values) |call| {
        try out.append(invoke_open ++ " name=\"");
        try attribute(out, call.name);
        try out.append("\">\n");
        if (call.parameters) |parameters| {
            for (parameters) |argument| try parameter(out, argument);
        } else {
            try out.append(parameter_open ++ " name=\"arguments\" string=\"true\">");
            try body(out, call.arguments orelse "", parameter_close);
            try out.append(parameter_close ++ "\n");
        }
        try out.append(invoke_close ++ "\n");
    }
    try out.append(calls_close);
}

pub const tools_preamble =
    "## Tools\n\n" ++
    "You can invoke tools using this format:\n\n" ++
    "<｜DSML｜ calls>\n" ++
    "<｜DSML｜ invoke name=\"$TOOL_NAME\">\n" ++
    "<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n" ++
    "</｜DSML｜ invoke>\n" ++
    "</｜DSML｜ calls>\n\n" ++
    "String values use string=\"true\"; all other values use JSON and string=\"false\". " ++
    "Inside string values only, escape a literal </｜DSML｜ parameter> as &lt;/｜DSML｜ parameter>. " ++
    "To write that escaped spelling literally, use &amp;lt;/｜DSML｜ parameter>. Other HTML entities are unchanged.\n\n" ++
    "Finish reasoning with </think> before tool calls or a final response.\n\n" ++
    "### Available Tool Schemas\n\n";
pub const tools_footer =
    "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. " ++
    "Use the exact parameter names from the schemas.";

pub fn render(allocator: Allocator, messages: []const Message, options: Options) Allocator.Error![:0]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const views = try Views.init(arena.allocator(), messages, options.start);
    const ordered = try orderedMessages(arena.allocator(), views.messages, options.initial_tool_order);
    const think = switch (options.reasoning) {
        .none => false,
        .level => |level| level > 0,
        else => true,
    };
    const keep_reasoning = options.live_tail or toolContext(views.messages, options.schemas);
    const level = switch (options.reasoning) {
        .high => @as(u8, 75),
        .max => @as(u8, 100),
        .level => |value| value,
        else => @as(u8, 0),
    };
    var out = Text{ .allocator = allocator };
    errdefer out.list.deinit(allocator);
    var initial_system = false;
    var pending = false;
    var user_open = false;
    var last_user: ?usize = null;
    for (views.messages, 0..) |*message, position| {
        if (userLike(message) or (position > 0 and systemLike(message))) last_user = position;
    }
    if (options.live_tail) {
        try out.append("<｜end▁of▁sentence｜>");
    } else {
        try out.append("<｜begin▁of▁sentence｜>");
        if (level > 0 or nonempty(options.schemas)) {
            try out.append("<｜System｜>");
            initial_system = true;
            if (level > 0) try out.list.print(allocator, "Reasoning Effort: {d} (range 1-100, the higher the value, the more thorough the reasoning)\n\n", .{level});
            if (nonempty(options.schemas)) {
                try out.append(tools_preamble);
                try out.string(options.schemas);
                try out.append(tools_footer);
            }
        }
    }
    // Full requests and live tails use the same separator transitions at the
    // authoritative cached frontier. A tail starts after an assistant turn.
    for (views.start..ordered.len) |position| {
        const message = ordered[position];
        if (systemLike(message)) {
            if (position == 0 and initial_system) {
                if (nonempty(options.schemas)) try out.append("\n\n");
            } else try out.append("<｜System｜>");
            try out.string(message.content);
            pending = position > 0 or options.live_tail;
            user_open = false;
        } else if (userLike(message)) {
            try out.append(if (user_open) "\n\n" else "<｜User｜>");
            if (!equal(message.role, "user")) {
                try out.append("<tool_result>");
                try body(&out, message.content orelse "", "</tool_result>");
                try out.append("</tool_result>");
            } else try out.string(message.content);
            pending = true;
            user_open = true;
        } else if (equal(message.role, "assistant")) {
            if (pending) {
                try out.append("<｜Assistant｜>");
                if (think and (keep_reasoning or last_user == null or position > last_user.?)) {
                    try out.append("<think>");
                    try out.string(message.reasoning);
                }
                try out.append("</think>");
            }
            try out.string(message.content);
            try calls(&out, message.calls, message.raw_tool_text);
            try out.append("<｜end▁of▁sentence｜>");
            pending = false;
            user_open = false;
        }
    }
    if (pending) {
        try out.append("<｜Assistant｜>");
        try out.append(if (think) "<think>" else "</think>");
    }
    return out.list.toOwnedSliceSentinel(allocator, 0);
}
