//! Native rendering of recursive-oss/rec's Flint chat template (Apache-2.0).
//! Canonical serving values stay independent of HTTP and model engine layouts.
const std = @import("std");
const types = @import("serving_types");
const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub const Options = struct {
    start: usize = 0,
    /// The server's stream of function-schema JSON objects, or a JSON array.
    schemas: ?[]const u8 = null,
    add_generation_prompt: bool = true,
};

const Text = struct {
    allocator: Allocator,
    bytes: std.ArrayList(u8) = .empty,

    fn append(self: *Text, text: []const u8) !void {
        try self.bytes.appendSlice(self.allocator, text);
    }

    /// Python json.dumps(ensure_ascii=False)'s ordinary separator spacing.
    /// String escaping and value serialization remain in the standard library.
    fn json(self: *Text, value: Value) !void {
        switch (value) {
            .array => |array| {
                try self.append("[");
                for (array.items, 0..) |item, index| {
                    if (index > 0) try self.append(", ");
                    try self.json(item);
                }
                try self.append("]");
                return;
            },
            .object => |object| {
                try self.append("{");
                var entries = object.iterator();
                var first = true;
                while (entries.next()) |entry| {
                    if (!first) try self.append(", ");
                    try self.json(.{ .string = entry.key_ptr.* });
                    try self.append(": ");
                    try self.json(entry.value_ptr.*);
                    first = false;
                }
                try self.append("}");
                return;
            },
            .float => |number| {
                if (!std.math.isFinite(number)) return error.InvalidJsonNumber;
                var buffer: [400]u8 = undefined;
                const scientific = @abs(number) >= 1e16 or (number != 0 and @abs(number) < 1e-4);
                const formatted = try std.fmt.float.render(&buffer, number, .{ .mode = if (scientific) .scientific else .decimal });
                if (scientific) {
                    const exponent = std.mem.indexOfScalar(u8, formatted, 'e').?;
                    try self.append(formatted[0 .. exponent + 1]);
                    var digits = formatted[exponent + 1 ..];
                    if (digits[0] == '-') {
                        try self.append("-");
                        digits = digits[1..];
                    } else try self.append("+");
                    if (digits.len == 1) try self.append("0");
                    try self.append(digits);
                } else {
                    try self.append(formatted);
                    if (std.mem.indexOfScalar(u8, formatted, '.') == null) try self.append(".0");
                }
                return;
            },
            else => {},
        }
        const encoded = try std.json.Stringify.valueAlloc(self.allocator, value, .{});
        defer self.allocator.free(encoded);
        try self.append(encoded);
    }
};

const control_tokens = [_][]const u8{
    "<|begin_of_text|>", "<|end_of_text|>", "<|system|>",      "<|developer|>",
    "<|user|>",          "<|assistant|>",   "<|observation|>",
};

fn normalized(a: Allocator, source: []const u8) ![]const u8 {
    var clean = Text{ .allocator = a };
    for (source) |byte| if (byte != 0) try clean.bytes.append(a, byte);
    const without_open = try std.mem.replaceOwned(u8, a, clean.bytes.items, "<think>", "");
    const without_think = try std.mem.replaceOwned(u8, a, without_open, "</think>", "");
    var out = Text{ .allocator = a };
    var position: usize = 0;
    while (position < without_think.len) {
        var control: ?[]const u8 = null;
        for (control_tokens) |token| {
            if (std.mem.startsWith(u8, without_think[position..], token)) {
                control = token;
                break;
            }
        }
        if (control) |token| {
            try out.append("&lt;");
            try out.append(token[1 .. token.len - 1]);
            try out.append("&gt;");
            position += token.len;
        } else {
            try out.bytes.append(a, without_think[position]);
            position += 1;
        }
    }
    return out.bytes.items;
}

const tools_before =
    "<|system|>\n# Tools\n\n" ++
    "You may call one or more functions to assist with the user query.\n\n" ++
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
const tools_after =
    "</tools>\n\n" ++
    "For each function call, output the function name and arguments within the following XML format:\n" ++
    "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>\n";

fn schema(out: *Text, value: Value) !void {
    const function = if (value == .object) value.object.get("function") orelse value else value;
    try out.json(function);
}

fn tools(out: *Text, a: Allocator, source: ?[]const u8) !void {
    const text = std.mem.trim(u8, source orelse "", " \t\r\n");
    if (text.len == 0) return;
    if (text[0] == '[') {
        const parsed = try std.json.parseFromSlice(Value, a, text, .{});
        if (parsed.value != .array) return error.InvalidToolSchemas;
        if (parsed.value.array.items.len == 0) return;
        try out.append(tools_before);
        for (parsed.value.array.items) |value| try schema(out, value);
    } else {
        try out.append(tools_before);
        var position: usize = 0;
        while (position < text.len) {
            // A fresh standard scanner reads one object from the server's
            // concatenated-schema stream, including pretty-printed objects.
            var scanner = std.json.Scanner.initCompleteInput(a, text[position..]);
            defer scanner.deinit();
            const value = try std.json.innerParse(Value, a, &scanner, .{
                .allocate = .alloc_always,
                .max_value_len = text.len - position,
                .duplicate_field_behavior = .use_last,
            });
            if (value != .object or scanner.cursor == 0) return error.InvalidToolSchemas;
            try schema(out, value);
            position += scanner.cursor;
            while (position < text.len and std.ascii.isWhitespace(text[position])) position += 1;
        }
    }
    try out.append(tools_after);
}

fn argument(out: *Text, name: []const u8, value: Value) !void {
    try out.append("<arg_key>");
    try out.append(name);
    try out.append("</arg_key><arg_value>");
    if (value == .string) try out.append(value.string) else try out.json(value);
    try out.append("</arg_value>");
}

fn call(out: *Text, a: Allocator, value: types.ToolCall) !void {
    try out.append("<tool_call>");
    try out.append(value.name orelse "");
    if (value.arguments) |raw| {
        const text = std.mem.trim(u8, raw, " \t\r\n");
        if (text.len > 0) {
            const parsed = try std.json.parseFromSlice(Value, a, text, .{});
            if (parsed.value != .object) return error.InvalidToolArguments;
            var entries = parsed.value.object.iterator();
            while (entries.next()) |entry| try argument(out, entry.key_ptr.*, entry.value_ptr.*);
        }
    } else if (value.parameters) |parameters| {
        for (parameters) |parameter| {
            if (parameter.is_string) {
                try argument(out, parameter.name, .{ .string = parameter.value });
            } else {
                const parsed = try std.json.parseFromSlice(Value, a, parameter.value, .{});
                try argument(out, parameter.name, parsed.value);
            }
        }
    }
    try out.append("</tool_call>");
}

pub fn render(allocator: Allocator, messages: []const types.Message, options: Options) ![:0]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var out = Text{ .allocator = allocator };
    errdefer out.bytes.deinit(allocator);
    var tools_pending = true;
    for (messages[@min(options.start, messages.len)..]) |message| {
        const role = message.role orelse "";
        if (tools_pending and !std.mem.eql(u8, role, "system")) {
            try tools(&out, a, options.schemas);
            tools_pending = false;
        }
        const assistant = std.mem.eql(u8, role, "assistant");
        const content = if (assistant) message.content orelse "" else try normalized(a, message.content orelse "");
        if (std.mem.eql(u8, role, "system") or std.mem.eql(u8, role, "developer") or std.mem.eql(u8, role, "user")) {
            try out.append("<|");
            try out.append(role);
            try out.append("|>");
            try out.append(content);
        } else if (assistant) {
            try out.append("<|assistant|>");
            if (message.reasoning) |reasoning| if (reasoning.len > 0) {
                try out.append("<think>");
                try out.append(reasoning);
                try out.append("</think>");
            };
            try out.append(content);
            if (message.calls.len > 0 and message.raw_tool_text != null and message.raw_tool_text.?.len > 0) {
                // An in-process continuation may retain the exact sampled call
                // spelling. Fresh API histories use the canonical template.
                try out.append(message.raw_tool_text.?);
            } else for (message.calls) |value| try call(&out, a, value);
        } else if (std.mem.eql(u8, role, "tool") or std.mem.eql(u8, role, "function")) {
            try out.append("<|observation|><tool_response>");
            try out.append(content);
            try out.append("</tool_response>");
        }
    }
    if (tools_pending) try tools(&out, a, options.schemas);
    if (options.add_generation_prompt) try out.append("<|assistant|>");
    return out.bytes.toOwnedSliceSentinel(allocator, 0);
}

test "Flint roles and non-assistant controls" {
    const rendered = try render(std.testing.allocator, &.{
        .{ .role = "system", .content = "A\x00<think>B</think><|user|>" },
        .{ .role = "developer", .content = "D" },
        .{ .role = "user", .content = "Q" },
        .{ .role = "assistant", .content = "answer", .reasoning = "reason" },
        .{ .role = "tool", .content = "result" },
    }, .{});
    defer std.testing.allocator.free(rendered);
    try std.testing.expectEqualStrings("<|system|>AB&lt;|user|&gt;<|developer|>D<|user|>Q<|assistant|><think>reason</think>answer<|observation|><tool_response>result</tool_response><|assistant|>", rendered);
}

test "Flint tool schemas and XML call arguments" {
    const rendered = try render(std.testing.allocator, &.{
        .{ .role = "system", .content = "S" },
        .{ .role = "user", .content = "Q" },
        .{ .role = "assistant", .calls = &.{.{ .name = "lookup", .arguments = "{\"q\":\"雪\",\"count\":2,\"options\":{\"ok\":true}}" }} },
    }, .{ .schemas = "{\"name\":\"lookup\",\n\"parameters\":{\"type\":\"object\"}}\n{\"name\":\"second\"}" });
    defer std.testing.allocator.free(rendered);
    try std.testing.expect(std.mem.startsWith(u8, rendered, "<|system|>S" ++ tools_before ++ "{\"name\": \"lookup\", \"parameters\": {\"type\": \"object\"}}{\"name\": \"second\"}"));
    try std.testing.expect(std.mem.endsWith(u8, rendered, "<|user|>Q<|assistant|><tool_call>lookup<arg_key>q</arg_key><arg_value>雪</arg_value><arg_key>count</arg_key><arg_value>2</arg_value><arg_key>options</arg_key><arg_value>{\"ok\": true}</arg_value></tool_call><|assistant|>"));
}
