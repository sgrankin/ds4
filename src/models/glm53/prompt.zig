//! GLM's chat/tool template over borrowed logical serving values. The model
//! owns its control tokens, reasoning retention and inline tool-call spelling;
//! HTTP records and the inference engine do not enter this module.
const std = @import("std");
const types = @import("serving_types");
const Schema = @import("tool_schema");
const Allocator = std.mem.Allocator;
const Message = types.Message;

pub const Options = struct {
    /// A live tail begins at this message after the cached assistant turn.
    start: usize = 0,
    schemas: ?[]const u8 = null,
    tools: []const types.ToolSchema = &.{},
    reasoning: types.Reasoning = .high,
    live_tail: bool = false,
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
fn systemLike(message: Message) bool {
    return equal(message.role, "system") or equal(message.role, "developer");
}
fn userLike(message: Message) bool {
    return equal(message.role, "user") or equal(message.role, "tool") or equal(message.role, "function");
}
fn toolResult(message: Message) bool {
    // Anthropic's parser records call IDs and wraps each user-role result in
    // <tool_result>. A mixed message follows that existing wrapper convention.
    return equal(message.role, "tool") or equal(message.role, "function") or
        (equal(message.role, "user") and message.tool_call_ids.len != 0);
}
fn toolContext(messages: []const Message, schemas: ?[]const u8) bool {
    if (nonempty(schemas)) return true;
    for (messages) |message| {
        if ((equal(message.role, "assistant") and message.calls.len != 0) or
            equal(message.role, "tool") or equal(message.role, "function")) return true;
    }
    return false;
}

fn escapedClose(tail: []const u8, end: []const u8) bool {
    if (tail.len == 0 or tail[0] != '&') return false;
    var remaining = tail[1..];
    while (std.mem.startsWith(u8, remaining, "amp;")) remaining = remaining[4..];
    return std.mem.startsWith(u8, remaining, "lt;") and std.mem.startsWith(u8, remaining[3..], end[1..]);
}
fn body(out: *Text, value: []const u8, end: []const u8) Allocator.Error!void {
    // Tool bodies are not HTML. Escape only the closing sentinel and text
    // which would collide with that escape; leave ordinary entities literal.
    for (value, 0..) |byte, position| {
        const tail = value[position..];
        if (std.mem.startsWith(u8, tail, end) or escapedClose(tail, end))
            try out.append(if (byte == '<') "&lt;" else "&amp;")
        else
            try out.byte(byte);
    }
}

fn argument(out: *Text, value: types.Parameter) Allocator.Error!void {
    try out.append("<arg_key>");
    try body(out, value.name, "</arg_key>");
    try out.append("</arg_key><arg_value>");
    try body(out, value.value, "</arg_value>");
    try out.append("</arg_value>");
}
fn orderFor(tools: []const types.ToolSchema, name: ?[]const u8) []const types.Property {
    const wanted = name orelse return &.{};
    for (tools) |tool| {
        if (tool.name) |candidate| if (std.mem.eql(u8, candidate, wanted)) return tool.properties;
    }
    return &.{};
}
fn calls(out: *Text, arena: Allocator, values: []const types.ToolCall, raw: ?[]const u8, tools: []const types.ToolSchema) Allocator.Error!void {
    if (values.len == 0) return;
    if (nonempty(raw)) return out.string(raw);
    // GLM emits these inline. Added separators would change the live prefix.
    for (values) |call| {
        try out.append("<tool_call>");
        try out.string(call.name);
        if (call.parameters) |parameters| {
            const used = try arena.alloc(bool, parameters.len);
            @memset(used, false);
            for (orderFor(tools, call.name)) |property| {
                const wanted = property.name orelse continue;
                for (parameters, 0..) |parameter, index| {
                    if (!used[index] and std.mem.eql(u8, parameter.name, wanted)) {
                        try argument(out, parameter);
                        used[index] = true;
                        break;
                    }
                }
            }
            for (parameters, used) |parameter, emitted| {
                if (!emitted) try argument(out, parameter);
            }
        } else {
            // The wire boundary's tolerant parser supplies descriptors. A null
            // vector means parsing failed, retaining the literal raw fallback.
            try argument(out, .{ .name = "arguments", .value = call.arguments orelse "", .is_string = true });
        }
        try out.append("</tool_call>");
    }
}

const tools_before =
    "\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n" ++
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n\n";
const tools_after =
    "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n" ++
    "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>" ++
    "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";

fn toolsPrompt(out: *Text, arena: Allocator, schemas: []const u8) Allocator.Error!void {
    if (schemas.len == 0) return;
    try out.append(tools_before);
    var remaining = schemas;
    var any = false;
    while (try Schema.takeValue(arena, remaining)) |value| {
        if (try Schema.renderOne(arena, value.raw)) |schema| {
            try out.append(schema);
            try out.append("\n\n\n");
            any = true;
        }
        remaining = remaining[value.consumed..];
    }
    if (!any) try out.byte('\n');
    try out.append(tools_after);
}

fn response(out: *Text, content: []const u8) Allocator.Error!void {
    try out.append("<tool_response>");
    try body(out, content, "</tool_response>");
    try out.append("</tool_response>");
}
fn resultMessage(out: *Text, message: Message) Allocator.Error!void {
    const content = message.content orelse "";
    if (equal(message.role, "user") and message.tool_call_ids.len != 0) {
        var remaining = content;
        var found = false;
        while (std.mem.indexOf(u8, remaining, "<tool_result>")) |open| {
            const after_open = remaining[open + "<tool_result>".len ..];
            const close = std.mem.indexOf(u8, after_open, "</tool_result>") orelse break;
            try response(out, after_open[0..close]);
            found = true;
            remaining = after_open[close + "</tool_result>".len ..];
        }
        if (found) return;
    }
    try response(out, content);
}
fn assistantPrefix(out: *Text, message: Message, preserve: bool) Allocator.Error!void {
    const content = message.content orelse "";
    // Test the original text before trimming, as the existing template does.
    if (std.mem.startsWith(u8, content, "<think>") or std.mem.startsWith(u8, content, "</think>")) return;
    try out.append("<think>");
    if (preserve) try out.string(message.reasoning);
    try out.append("</think>");
}

pub fn render(allocator: Allocator, messages: []const Message, options: Options) Allocator.Error![:0]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var out = Text{ .allocator = allocator };
    errdefer out.list.deinit(allocator);
    const think = options.reasoning.enabled();
    const context = toolContext(messages, options.schemas);
    var last_user: ?usize = null;
    for (messages, 0..) |message, index| if (userLike(message)) {
        last_user = index;
    };

    if (!options.live_tail) {
        try out.append("[gMASK]<sop>");
        if (think) try out.append(switch (options.reasoning) {
            .low, .medium, .high => "<|system|>Reasoning Effort: High",
            else => "<|system|>Reasoning Effort: Max",
        });
        if (nonempty(options.schemas)) {
            try out.append("<|system|>");
            try toolsPrompt(&out, arena.allocator(), options.schemas.?);
        }
        // Client system/developer messages are hoisted after the model's
        // effort/tool instructions, retaining their original relative order.
        for (messages) |message| {
            if (!systemLike(message)) continue;
            try out.append("<|system|>");
            try out.string(message.content);
        }
    }

    var pending = false;
    var observation_open = false;
    const start = if (options.live_tail) @min(options.start, messages.len) else 0;
    for (start..messages.len) |index| {
        const message = messages[index];
        if (systemLike(message)) {
            observation_open = false;
        } else if (toolResult(message)) {
            if (!observation_open) try out.append("<|observation|>");
            try resultMessage(&out, message);
            observation_open = true;
            pending = true;
        } else if (equal(message.role, "user")) {
            observation_open = false;
            try out.append("<|user|>");
            try out.string(message.content);
            pending = true;
        } else if (equal(message.role, "assistant")) {
            observation_open = false;
            try out.append("<|assistant|>");
            try assistantPrefix(&out, message, think and (options.live_tail or context or last_user == null or index > last_user.?));
            try out.append(std.mem.trim(u8, message.content orelse "", &std.ascii.whitespace));
            try calls(&out, arena.allocator(), message.calls, message.raw_tool_text, options.tools);
            pending = false;
        }
    }
    if (pending) {
        try out.append("<|assistant|>");
        try out.append(if (think) "<think>" else "<think></think>");
    }
    return out.list.toOwnedSliceSentinel(allocator, 0);
}

/// Small compatibility leaves can reuse the same native owner while their
/// surrounding callers still hold C buffers. Returned bytes belong to caller.
pub fn renderCalls(allocator: Allocator, values: []const types.ToolCall, raw: ?[]const u8, tools: []const types.ToolSchema) Allocator.Error![:0]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var out = Text{ .allocator = allocator };
    errdefer out.list.deinit(allocator);
    try calls(&out, arena.allocator(), values, raw, tools);
    return out.list.toOwnedSliceSentinel(allocator, 0);
}
pub fn renderTools(allocator: Allocator, schemas: []const u8) Allocator.Error![:0]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var out = Text{ .allocator = allocator };
    errdefer out.list.deinit(allocator);
    try toolsPrompt(&out, arena.allocator(), schemas);
    return out.list.toOwnedSliceSentinel(allocator, 0);
}

test "GLM hoists systems and retains exact tools template" {
    const rendered = try render(std.testing.allocator, &.{
        .{ .role = "user", .content = "Hello" },
        .{ .role = "developer", .content = "You are terse." },
    }, .{ .schemas = "{\"name\":\"hidden\",\"defer_loading\":true,\"parameters\":{}}\n{\"name\":\"bash\",\"parameters\":{ \"type\": \"object\" },\"strict\":true}" });
    defer std.testing.allocator.free(rendered);
    try std.testing.expectEqualStrings("[gMASK]<sop><|system|>Reasoning Effort: High<|system|>" ++ tools_before ++
        "{\"name\": \"bash\", \"parameters\": {\"type\":\"object\"}}\n\n\n" ++ tools_after ++
        "<|system|>You are terse.<|user|>Hello<|assistant|><think>", rendered);
}

test "GLM reasoning retention and assistant prefix use original content" {
    const messages = [_]Message{
        .{ .role = "user", .content = "first" },
        .{ .role = "assistant", .content = "  answer \n", .reasoning = "old reasoning" },
        .{ .role = "user", .content = "second" },
        .{ .role = "assistant", .content = "</think>direct" },
    };
    const plain = try render(std.testing.allocator, &messages, .{});
    defer std.testing.allocator.free(plain);
    try std.testing.expectEqualStrings("[gMASK]<sop><|system|>Reasoning Effort: High<|user|>first<|assistant|><think></think>answer<|user|>second<|assistant|></think>direct", plain);
    const tools = try render(std.testing.allocator, &messages, .{ .schemas = "{}" });
    defer std.testing.allocator.free(tools);
    try std.testing.expect(std.mem.indexOf(u8, tools, "<think>old reasoning</think>answer") != null);
    const live = try render(std.testing.allocator, &messages, .{ .live_tail = true, .start = 1 });
    defer std.testing.allocator.free(live);
    try std.testing.expectEqualStrings("<|assistant|><think>old reasoning</think>answer<|user|>second<|assistant|></think>direct", live);
}

test "GLM inline calls preserve schema order duplicate parameters and literal bodies" {
    const rendered = try renderCalls(std.testing.allocator, &.{.{ .name = "lookup", .parameters = &.{
        .{ .name = "count", .value = "1e+02", .is_string = false },
        .{ .name = "query", .value = "a > b && &lt; </arg_value>", .is_string = true },
        .{ .name = "count", .value = "-0", .is_string = false },
    } }}, null, &.{.{ .name = "lookup", .properties = &.{ .{ .name = "query" }, .{ .name = "count" } } }});
    defer std.testing.allocator.free(rendered);
    try std.testing.expectEqualStrings("<tool_call>lookup<arg_key>query</arg_key><arg_value>a > b && &lt; &lt;/arg_value></arg_value><arg_key>count</arg_key><arg_value>1e+02</arg_value><arg_key>count</arg_key><arg_value>-0</arg_value></tool_call>", rendered);
}

test "GLM raw replay and invalid arguments retain exact fallback" {
    const values = [_]types.ToolCall{.{ .name = "lookup", .arguments = "bad </arg_value> &lt;/arg_value>" }};
    const fallback = try renderCalls(std.testing.allocator, &values, null, &.{});
    defer std.testing.allocator.free(fallback);
    try std.testing.expectEqualStrings("<tool_call>lookup<arg_key>arguments</arg_key><arg_value>bad &lt;/arg_value> &amp;lt;/arg_value></arg_value></tool_call>", fallback);
    const raw = try renderCalls(std.testing.allocator, &values, "\n<tool_call>EXACT</tool_call>", &.{});
    defer std.testing.allocator.free(raw);
    try std.testing.expectEqualStrings("\n<tool_call>EXACT</tool_call>", raw);
}

test "GLM observations group Anthropic results and live tails have no extra terminator" {
    const rendered = try render(std.testing.allocator, &.{
        .{ .role = "assistant", .content = "cached" },
        .{ .role = "user", .content = "outside<tool_result>one</tool_result><tool_result>two</tool_result>outside", .tool_call_ids = &.{ "a", "b" } },
        .{ .role = "function", .content = "three</tool_response>" },
    }, .{ .start = 1, .live_tail = true, .reasoning = .none });
    defer std.testing.allocator.free(rendered);
    try std.testing.expectEqualStrings("<|observation|><tool_response>one</tool_response><tool_response>two</tool_response><tool_response>three&lt;/tool_response></tool_response><|assistant|><think></think>", rendered);
}

test "GLM numeric efforts keep current fallback and zero disables thinking" {
    const messages = [_]Message{.{ .role = "user", .content = "Q" }};
    const numeric = try render(std.testing.allocator, &messages, .{ .reasoning = .{ .level = 42 } });
    defer std.testing.allocator.free(numeric);
    try std.testing.expectEqualStrings("[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Q<|assistant|><think>", numeric);
    const disabled = try render(std.testing.allocator, &messages, .{ .reasoning = .{ .level = 0 } });
    defer std.testing.allocator.free(disabled);
    try std.testing.expectEqualStrings("[gMASK]<sop><|user|>Q<|assistant|><think></think>", disabled);
}
