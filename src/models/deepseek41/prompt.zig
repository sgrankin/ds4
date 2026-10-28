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
    prompt_map: ?*types.PromptMap = null,
    response_format: types.ResponseFormat = .text,
};

const Text = struct {
    allocator: Allocator,
    list: std.ArrayList(u8) = .empty,
    map: ?*types.PromptMap = null,

    fn boundary(self: *Text) Allocator.Error!void {
        if (self.map) |map| {
            if (!map.capture_context) return;
            const position = self.list.items.len;
            if (map.context_boundaries.items.len == 0 or map.context_boundaries.items[map.context_boundaries.items.len - 1] != position)
                try map.context_boundaries.append(self.allocator, position);
        }
    }

    fn append(self: *Text, value: []const u8) Allocator.Error!void {
        try self.list.appendSlice(self.allocator, value);
    }
    fn byte(self: *Text, value: u8) Allocator.Error!void {
        try self.list.append(self.allocator, value);
    }
    fn string(self: *Text, value: ?[]const u8) Allocator.Error!void {
        try self.append(value orelse "");
    }
    fn control(self: *Text, value: []const u8) Allocator.Error!void {
        const begin = self.list.items.len;
        try self.append(value);
        if (self.map) |map| try map.special_ranges.append(self.allocator, .{ .begin = begin, .end = self.list.items.len });
    }
    /// Only template-owned fragments enter this scanner. Payload is appended
    /// with string/body/attribute and never gains control meaning by spelling.
    fn template(self: *Text, value: []const u8) Allocator.Error!void {
        const markers = [_][]const u8{
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<｜System｜>",
            "<｜User｜>",
            "<｜Assistant｜>",
            "<think>",
            "</think>",
            "｜DSML｜",
        };
        var position: usize = 0;
        var plain: usize = 0;
        while (position < value.len) {
            var matched = false;
            for (markers) |marker| {
                if (!std.mem.startsWith(u8, value[position..], marker)) continue;
                try self.append(value[plain..position]);
                try self.control(marker);
                position += marker.len;
                plain = position;
                matched = true;
                break;
            }
            if (!matched) position += 1;
        }
        try self.append(value[plain..]);
    }
    fn assistant(self: *Text, value: ?[]const u8) Allocator.Error!void {
        const source = value orelse return;
        const grounding = [_][]const u8{
            "<｜ref｜>",
            "<｜/ref｜>",
            "<｜box｜>",
            "<｜/box｜>",
            "<｜point｜>",
            "<｜/point｜>",
            "<｜polygon｜>",
            "<｜/polygon｜>",
        };
        var position: usize = 0;
        var plain: usize = 0;
        while (position < source.len) {
            var matched = false;
            for (grounding) |marker| {
                if (!std.mem.startsWith(u8, source[position..], marker)) continue;
                try self.append(source[plain..position]);
                try self.control(marker);
                position += marker.len;
                plain = position;
                matched = true;
                break;
            }
            if (!matched) position += 1;
        }
        try self.append(source[plain..]);
    }
    fn content(self: *Text, message: *const Message, assistant_mode: bool) Allocator.Error!void {
        const source = message.content orelse "";
        var begin: usize = 0;
        for (message.cache_breakpoints) |end| {
            std.debug.assert(end >= begin and end <= source.len);
            if (assistant_mode) try self.assistant(source[begin..end]) else try self.append(source[begin..end]);
            if (self.map) |map| {
                const position = self.list.items.len;
                if (map.cache_breakpoints.items.len == 0 or map.cache_breakpoints.items[map.cache_breakpoints.items.len - 1] != position)
                    try map.cache_breakpoints.append(self.allocator, position);
            }
            begin = end;
        }
        if (assistant_mode) try self.assistant(source[begin..]) else try self.append(source[begin..]);
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

const calls_open = "<｜DSML｜ calls>";
const calls_close = "</｜DSML｜ calls>";
const invoke_open = "<｜DSML｜ invoke";
const invoke_close = "</｜DSML｜ invoke>";
const parameter_open = "<｜DSML｜ parameter";
const parameter_close = "</｜DSML｜ parameter>";

fn parameter(out: *Text, argument: types.Parameter) Allocator.Error!void {
    try out.template(parameter_open ++ " name=\"");
    try out.append(argument.name);
    try out.append(if (argument.is_string) "\" string=\"true\">" else "\" string=\"false\">");
    try out.append(argument.value);
    try out.template(parameter_close ++ "\n");
}

fn calls(out: *Text, values: []const types.ToolCall, raw_tool_text: ?[]const u8, raw_ranges: ?[]const types.ByteRange) Allocator.Error!void {
    if (values.len == 0) return;
    if (nonempty(raw_tool_text) and raw_ranges != null) {
        const base = out.list.items.len;
        try out.string(raw_tool_text);
        if (out.map) |map| for (raw_ranges.?) |range| {
            try map.special_ranges.append(out.allocator, .{ .begin = base + range.begin, .end = base + range.end });
        };
        return;
    }
    try out.template("\n\n" ++ calls_open ++ "\n");
    for (values) |call| {
        try out.template(invoke_open ++ " name=\"");
        try out.string(call.name);
        try out.append("\">\n");
        if (call.parameters) |parameters| {
            for (parameters) |argument| try parameter(out, argument);
            if (parameters.len == 0) try out.append("\n");
        } else {
            try out.template(parameter_open ++ " name=\"arguments\" string=\"true\">");
            try out.string(call.arguments);
            try out.template(parameter_close ++ "\n");
        }
        try out.template(invoke_close ++ "\n");
        try out.boundary();
    }
    try out.template(calls_close);
}

// Reference instruction text: DeepSeek's MIT-licensed recipe encoder,
// deepseek-recipe-encoding/src/v4/dsv41.rs and v4/mod.rs. See LICENSE.recipe.
pub const tools_preamble =
    \\## Tools
    \\
    \\You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:
    \\
    \\<｜DSML｜ calls>
    \\<｜DSML｜ invoke name="$TOOL_NAME">
    \\<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>
    \\...
    \\</｜DSML｜ invoke>
    \\<｜DSML｜ invoke name="$TOOL_NAME2">
    \\...
    \\</｜DSML｜ invoke>
    \\</｜DSML｜ calls>
    \\
    \\String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.
    \\
    \\If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.
    \\
    \\Otherwise, output directly after </think> with tool calls or final response.
    \\
    \\### Available Tool Schemas
    \\
    \\
;
pub const tools_footer = "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

fn extras(out: *Text, schemas: ?[]const u8, format: types.ResponseFormat) Allocator.Error!void {
    if (nonempty(schemas)) {
        try out.append("\n\n");
        try out.template(tools_preamble);
        try out.string(schemas);
        try out.append(tools_footer);
    }
    const schema = switch (format) {
        .json_object => "{\"type\": \"json_object\"}",
        .json_schema => |value| value,
        else => return,
    };
    try out.append("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
    try out.append(schema);
}

fn hashControl(hash: *std.crypto.hash.Sha1, role: []const u8, payload: []const u8) void {
    var length: [8]u8 = undefined;
    hash.update(role);
    hash.update("\x00");
    std.mem.writeInt(u64, &length, payload.len, .little);
    hash.update(&length);
    hash.update(payload);
}

/// A tool-result edit can preserve an ongoing task. Changed actual controls or
/// declarations require ordinary freshness policy even if an older task matches.
fn controlKey(messages: []const Message, options: Options, level: u8) ?[41]u8 {
    var hash = std.crypto.hash.Sha1.init(.{});
    hash.update("ongoing-controls-v3\x00");
    var has_user = false;
    for (messages) |message| {
        if (!systemLike(&message) and !equal(message.role, "user")) continue;
        has_user = has_user or equal(message.role, "user");
        hashControl(&hash, message.role.?, message.content orelse "");
    }
    if (!has_user) return null;
    hashControl(&hash, "reasoning", &.{level});
    if (nonempty(options.schemas)) hashControl(&hash, "tools", options.schemas.?);
    switch (options.response_format) {
        .json_object => hashControl(&hash, "response_format", "{\"type\": \"json_object\"}"),
        .json_schema => |schema| hashControl(&hash, "response_format", schema),
        else => {},
    }
    var digest: [20]u8 = undefined;
    hash.final(&digest);
    var result: [41]u8 = undefined;
    const alphabet = "0123456789abcdef";
    for (digest, 0..) |byte, index| {
        result[2 * index] = alphabet[byte >> 4];
        result[2 * index + 1] = alphabet[byte & 15];
    }
    result[40] = 0;
    return result;
}

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
    const keep_reasoning = options.live_tail or nonempty(options.schemas);
    const level = switch (options.reasoning) {
        .low => @as(u8, 50),
        .high => @as(u8, 75),
        .max => @as(u8, 100),
        .level => |value| value,
        else => @as(u8, 0),
    };
    if (options.prompt_map) |map| {
        map.special_ranges.clearRetainingCapacity();
        map.cache_breakpoints.clearRetainingCapacity();
        map.context_boundaries.clearRetainingCapacity();
        map.user_ranges.clearRetainingCapacity();
        map.last_user_begin = 0;
        map.control_end = 0;
        map.control_key = if (map.capture_context) controlKey(views.messages, options, level) else null;
    }
    var out = Text{ .allocator = allocator, .map = options.prompt_map };
    errdefer out.list.deinit(allocator);
    var initial_system = false;
    var pending = false;
    var user_open = false;
    if (options.live_tail) {
        try out.control("<｜end▁of▁sentence｜>");
    } else {
        try out.control("<｜begin▁of▁sentence｜>");
        if (level > 0 or nonempty(options.schemas) or options.response_format == .json_object or options.response_format == .json_schema) {
            try out.control("<｜System｜>");
            initial_system = true;
            if (level > 0) try out.list.print(allocator, "Reasoning Effort: {d} (range 1-100, the higher the value, the more thorough the reasoning)\n\n", .{level});
            if (views.messages.len == 0 or !systemLike(&views.messages[0])) try extras(&out, options.schemas, options.response_format);
        }
    }
    // Full requests and live tails use the same separator transitions at the
    // authoritative cached frontier. A tail starts after an assistant turn.
    for (views.start..ordered.len) |position| {
        const message = ordered[position];
        if (systemLike(message)) {
            if (!(position == 0 and initial_system)) try out.control("<｜System｜>");
            try out.content(message, false);
            if (position == 0 and !options.live_tail) try extras(&out, options.schemas, options.response_format);
            if (out.map) |map| {
                if (map.capture_context) map.control_end = out.list.items.len;
            }
            pending = true;
            user_open = false;
        } else if (userLike(message)) {
            const turn_begin = out.list.items.len;
            try out.template(if (user_open) "\n\n" else "<｜User｜>");
            const body_begin = out.list.items.len;
            if (!equal(message.role, "user")) {
                try out.append("<tool_result>");
                try out.content(message, false);
                try out.append("</tool_result>");
            } else try out.content(message, false);
            if (equal(message.role, "user")) if (out.map) |map| {
                if (map.capture_context) {
                    try map.user_ranges.append(allocator, .{ .begin = body_begin, .end = out.list.items.len });
                    map.last_user_begin = turn_begin;
                    map.control_end = out.list.items.len;
                }
            };
            pending = true;
            user_open = true;
        } else if (equal(message.role, "assistant")) {
            try out.control("<｜Assistant｜>");
            if (think and keep_reasoning) {
                try out.control("<think>");
                try out.assistant(message.reasoning);
            }
            try out.control("</think>");
            try out.content(message, true);
            try calls(&out, message.calls, message.raw_tool_text, message.raw_tool_special_ranges);
            try out.control("<｜end▁of▁sentence｜>");
            pending = false;
            user_open = false;
        }
        try out.boundary();
    }
    if (pending) {
        try out.control("<｜Assistant｜>");
        try out.control(if (think) "<think>" else "</think>");
    }
    return out.list.toOwnedSliceSentinel(allocator, 0);
}
