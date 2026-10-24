//! Qwen3.8 ChatML over canonical serving values. Preserve sampled assistant
//! bytes, schema argument order and the missing stop-token boundary used by
//! live tool continuation. No HTTP, C records or model engine enter this owner.
const std = @import("std");
const types = @import("serving_types");
const schema = @import("tool_schema");
const Allocator = std.mem.Allocator;
const Message = types.Message;
const ToolSchema = types.ToolSchema;
const whitespace = " \t\r\n\x0b\x0c";

pub const Options = struct {
    start: usize = 0,
    schemas: ?[]const u8 = null,
    tools: []const ToolSchema = &.{},
    reasoning: types.Reasoning = .high,
    live_tail: bool = false,
};

const Text = struct {
    allocator: Allocator,
    bytes: std.ArrayList(u8) = .empty,

    fn append(self: *Text, value: []const u8) Allocator.Error!void {
        try self.bytes.appendSlice(self.allocator, value);
    }
    fn byte(self: *Text, value: u8) Allocator.Error!void {
        try self.bytes.append(self.allocator, value);
    }
};

fn equal(value: ?[]const u8, expected: []const u8) bool {
    return std.mem.eql(u8, value orelse "", expected);
}
fn trimmed(value: ?[]const u8) []const u8 {
    return std.mem.trim(u8, value orelse "", whitespace);
}
fn systemLike(message: Message) bool {
    return equal(message.role, "system") or equal(message.role, "developer");
}
fn toolResult(message: Message) bool {
    return equal(message.role, "tool") or equal(message.role, "function") or
        (equal(message.role, "user") and message.tool_call_ids.len != 0);
}
fn escapedClose(source: []const u8, closing: []const u8) bool {
    if (source.len == 0 or source[0] != '&') return false;
    var remaining = source[1..];
    while (std.mem.startsWith(u8, remaining, "amp;")) remaining = remaining[4..];
    return std.mem.startsWith(u8, remaining, "lt;") and std.mem.startsWith(u8, remaining[3..], closing[1..]);
}
fn body(out: *Text, source: []const u8, closing: []const u8) !void {
    // Tool text is not HTML. Escape only its closing sentinel and the entity
    // spellings that would collide with this escape when read back.
    for (source, 0..) |byte, index| {
        const tail = source[index..];
        if (std.mem.startsWith(u8, tail, closing) or escapedClose(tail, closing)) {
            try out.append(if (byte == '<') "&lt;" else "&amp;");
        } else try out.byte(byte);
    }
}

const xhigh = "Reasoning effort is set to xhigh. Please think carefully through the task, " ++
    "validate key assumptions, consider plausible alternatives, and prioritize " ++
    "correctness, consistency, and clarity in the final answer.";
const low = "Reasoning effort is set to low. Keep your thinking brief and focused, " ++
    "moving directly to the conclusion without unnecessary elaboration.";

fn effort(reasoning: types.Reasoning) ?[]const u8 {
    // The reference template defaults to xhigh. Medium intentionally adds no
    // instruction; arbitrary shared numeric levels likewise add no Qwen text.
    return switch (reasoning) {
        .high, .max => xhigh,
        .low => low,
        else => null,
    };
}

const tools_before = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const tools_after = "\n</tools>\n\n" ++
    "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n" ++
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" ++
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n" ++
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" ++
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested " ++
    "within <tool_call></tool_call> XML tags\n" ++
    "- Required parameters MUST be specified\n" ++
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, " ++
    "but NOT after\n" ++
    "- If there is no function call available, answer the question like normal with your current knowledge and " ++
    "do not tell the user about function calls\n</IMPORTANT>";

fn tools(out: *Text, source: []const u8) !void {
    try out.append(tools_before);
    var position: usize = 0;
    while (try schema.takeValue(out.allocator, source[position..])) |value| {
        if (try schema.renderOne(out.allocator, value.raw)) |rendered| {
            try out.append("\n{\"type\": \"function\", \"function\": ");
            try out.append(rendered);
            try out.append("}");
        }
        position += value.consumed;
    }
    try out.append(tools_after);
}

fn toolOrder(all: []const ToolSchema, name: ?[]const u8) []const types.Property {
    const requested = name orelse return &.{};
    for (all) |tool| if (equal(tool.name, requested)) return tool.properties;
    return &.{};
}

fn parameter(out: *Text, value: types.Parameter) !void {
    try out.append("<parameter=");
    try body(out, value.name, ">");
    try out.append(">\n");
    const text = if (value.is_string) value.value else try schema.minify(out.allocator, value.value);
    try body(out, if (!value.is_string and text.len == 0) "null" else text, "</parameter>");
    try out.append("\n</parameter>\n");
}

fn calls(out: *Text, message: Message, ordered: []const ToolSchema, has_content: bool) !void {
    if (message.calls.len == 0) return;
    if (message.raw_tool_text) |raw| if (raw.len != 0) return out.append(raw);
    for (message.calls, 0..) |call, index| {
        try out.append(if (index != 0) "\n<tool_call>\n<function=" else if (has_content) "\n\n<tool_call>\n<function=" else "<tool_call>\n<function=");
        try out.append(call.name orelse "");
        try out.append(">\n");
        if (call.parameters) |parameters| {
            const used = try out.allocator.alloc(bool, parameters.len);
            @memset(used, false);
            for (toolOrder(ordered, call.name)) |property| {
                const name = property.name orelse continue;
                for (parameters, 0..) |value, position| {
                    if (!used[position] and std.mem.eql(u8, value.name, name)) {
                        used[position] = true;
                        try parameter(out, value);
                        break;
                    }
                }
            }
            for (parameters, used) |value, already_used| if (!already_used) try parameter(out, value);
        } else if (call.arguments) |raw| if (raw.len != 0) {
            try out.append("<parameter=arguments>\n");
            try body(out, raw, "</parameter>");
            try out.append("\n</parameter>\n");
        };
        try out.append("</function>\n</tool_call>");
    }
}

fn response(out: *Text, text: []const u8) !void {
    try out.append("\n<tool_response>\n");
    try body(out, text, "</tool_response>");
    try out.append("\n</tool_response>");
}

fn resultMessage(out: *Text, message: Message) !void {
    const content = message.content orelse "";
    if (equal(message.role, "user") and message.tool_call_ids.len != 0) {
        // Anthropic user-role result blocks already carry these wrappers.
        // Preserve their order, omitting surrounding ordinary text as before.
        const open = "<tool_result>";
        const close = "</tool_result>";
        var position: usize = 0;
        var found = false;
        while (std.mem.indexOf(u8, content[position..], open)) |offset| {
            const begin = position + offset + open.len;
            const end = begin + (std.mem.indexOf(u8, content[begin..], close) orelse break);
            try response(out, content[begin..end]);
            found = true;
            position = end + close.len;
        }
        if (found) return;
    }
    try response(out, content);
}

fn assistant(out: *Text, message: Message, ordered: []const ToolSchema) !void {
    const content = message.content orelse "";
    const reasoning = message.reasoning orelse "";
    const text = trimmed(content);
    try out.append("<|im_start|>assistant\n");
    if (!std.mem.startsWith(u8, content, "<think>") and !std.mem.startsWith(u8, content, "</think>")) {
        try out.append("<think>\n");
        try out.append(trimmed(reasoning));
        // Sampled tool text already includes its post-think separator. Adding
        // the template separator here would alter the cached replay prefix.
        const sampled = text.len == 0 and reasoning.len != 0 and message.raw_tool_text != null and message.raw_tool_text.?.len != 0;
        try out.append(if (sampled) "\n</think>" else "\n</think>\n\n");
    }
    try out.append(text);
    if (text.len != 0 and message.calls.len == 0) {
        // Plain answers retain trailing whitespace so echoing them yields the
        // same sampled tokens, while still trimming their leading whitespace.
        const end = std.mem.trimEnd(u8, content, whitespace).len;
        try out.append(content[end..]);
    }
    try calls(out, message, ordered, text.len != 0);
    try out.append("<|im_end|>\n");
}

fn conversation(out: *Text, messages: []const Message, start: usize, ordered: []const ToolSchema, thinking: bool) !void {
    var pending = false;
    var tool_open = false;
    for (messages[@min(start, messages.len)..]) |message| {
        if (systemLike(message)) continue;
        if (toolResult(message)) {
            if (!tool_open) try out.append("<|im_start|>user");
            try resultMessage(out, message);
            tool_open = true;
            pending = true;
            continue;
        }
        if (tool_open) {
            try out.append("<|im_end|>\n");
            tool_open = false;
        }
        if (equal(message.role, "user")) {
            try out.append("<|im_start|>user\n");
            try out.append(trimmed(message.content));
            try out.append("<|im_end|>\n");
            pending = true;
        } else if (equal(message.role, "assistant")) {
            try assistant(out, message, ordered);
            pending = false;
        }
    }
    if (tool_open) try out.append("<|im_end|>\n");
    if (pending) try out.append(if (thinking) "<|im_start|>assistant\n<think>\n" else "<|im_start|>assistant\n<think>\n\n</think>\n\n");
}

pub fn render(allocator: Allocator, messages: []const Message, options: Options) ![:0]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var out = Text{ .allocator = arena.allocator() };
    const thinking = options.reasoning.enabled();
    if (options.live_tail) {
        // The previous tool-call turn stopped before its im_end reached KV.
        try out.append("<|im_end|>\n");
        try conversation(&out, messages, options.start, options.tools, thinking);
    } else {
        var system = Text{ .allocator = arena.allocator() };
        for (messages) |message| if (systemLike(message)) {
            if (system.bytes.items.len != 0) try system.append("\n\n");
            try system.append(trimmed(message.content));
        };
        const instruction = if (thinking) effort(options.reasoning) else null;
        const have_tools = if (options.schemas) |source| source.len != 0 else false;
        if (instruction != null or have_tools or system.bytes.items.len != 0) {
            try out.append("<|im_start|>system\n");
            var wrote = false;
            if (instruction) |text| {
                try out.append(text);
                wrote = true;
            }
            if (have_tools) {
                if (wrote) try out.append("\n\n");
                try tools(&out, options.schemas.?);
                wrote = true;
            }
            if (system.bytes.items.len != 0) {
                if (wrote) try out.append("\n\n");
                try out.append(system.bytes.items);
            }
            try out.append("<|im_end|>\n");
        }
        try conversation(&out, messages, 0, options.tools, thinking);
    }
    return allocator.dupeSentinel(u8, out.bytes.items, 0);
}

pub fn renderTools(allocator: Allocator, schemas: []const u8) ![:0]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var out = Text{ .allocator = arena.allocator() };
    try tools(&out, schemas);
    return allocator.dupeSentinel(u8, out.bytes.items, 0);
}

pub fn renderCalls(allocator: Allocator, message: Message, ordered: []const ToolSchema, has_content: bool) ![:0]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var out = Text{ .allocator = arena.allocator() };
    try calls(&out, message, ordered, has_content);
    return allocator.dupeSentinel(u8, out.bytes.items, 0);
}

pub const SuffixOptions = struct {
    reasoning: types.Reasoning = .none,
    include_reasoning: bool = true,
    include_stop: bool = true,
};

/// Complete the assistant prefix already resident in KV. Transport policy
/// decides whether to include private reasoning and whether the sampled stop
/// token belongs to this checkpoint; the model owns all delimiter bytes.
pub fn assistantSuffix(allocator: Allocator, message: Message, ordered: []const ToolSchema, options: SuffixOptions) ![:0]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var out = Text{ .allocator = arena.allocator() };
    if (options.reasoning.enabled()) {
        if (options.include_reasoning) try out.append(trimmed(message.reasoning));
        try out.append("\n</think>\n\n");
    }
    const content = trimmed(message.content);
    try out.append(content);
    try calls(&out, message, ordered, content.len != 0);
    if (options.include_stop) try out.append("<|im_end|>");
    return allocator.dupeSentinel(u8, out.bytes.items, 0);
}

test "Qwen bare no-thinking prompt and plain assistant replay" {
    const allocator = std.testing.allocator;
    const prompt = try render(allocator, &.{ .{ .role = "system", .content = " You are terse. " }, .{ .role = "user", .content = " Hello " } }, .{ .reasoning = .none });
    defer allocator.free(prompt);
    try std.testing.expectEqualStrings("<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", prompt);
    const echoed = try render(allocator, &.{.{ .role = "assistant", .content = "  answer\n " }}, .{ .reasoning = .medium });
    defer allocator.free(echoed);
    try std.testing.expectEqualStrings("<|im_start|>assistant\n<think>\n\n</think>\n\nanswer\n <|im_end|>\n", echoed);
}

test "Qwen ordered calls raw replay and live result boundary" {
    const allocator = std.testing.allocator;
    const args = [_]types.Parameter{ .{ .name = "second", .value = "2.00", .is_string = false }, .{ .name = "first", .value = "a</parameter>&lt;/parameter>", .is_string = true } };
    const ordered = [_]ToolSchema{.{ .name = "read", .properties = &.{ .{ .name = "first" }, .{ .name = "second" } } }};
    const message = Message{ .role = "assistant", .calls = &.{.{ .name = "read", .parameters = &args }} };
    const formatted = try renderCalls(allocator, message, &ordered, false);
    defer allocator.free(formatted);
    try std.testing.expectEqualStrings("<tool_call>\n<function=read>\n<parameter=first>\na&lt;/parameter>&amp;lt;/parameter>\n</parameter>\n<parameter=second>\n2.00\n</parameter>\n</function>\n</tool_call>", formatted);
    const tail = try render(allocator, &.{ .{ .role = "tool", .content = "one" }, .{ .role = "function", .content = "two" } }, .{ .reasoning = .none, .live_tail = true });
    defer allocator.free(tail);
    try std.testing.expectEqualStrings("<|im_end|>\n<|im_start|>user\n<tool_response>\none\n</tool_response>\n<tool_response>\ntwo\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", tail);
    const raw = try renderCalls(allocator, .{ .calls = message.calls, .raw_tool_text = "\n\n<tool_call>raw</tool_call>" }, &ordered, true);
    defer allocator.free(raw);
    try std.testing.expectEqualStrings("\n\n<tool_call>raw</tool_call>", raw);
    const suffix = try assistantSuffix(allocator, .{ .reasoning = " plan ", .calls = message.calls, .raw_tool_text = raw }, &ordered, .{ .reasoning = .high, .include_reasoning = false, .include_stop = false });
    defer allocator.free(suffix);
    try std.testing.expectEqualStrings("\n</think>\n\n\n\n<tool_call>raw</tool_call>", suffix);
}
