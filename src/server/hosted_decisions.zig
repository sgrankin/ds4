//! A request-bound hosted decision tool over the existing native registry.
//! The client pins the model; generated arguments supply questions and may
//! replace the bound evidence. No HTTP request or second model owner is made.
const std = @import("std");
const allocator = std.heap.c_allocator;
const Value = std.json.Value;

pub const tool_type = "ds4:decision";
pub const function_name = "ds4_decide";
pub const schema: [:0]const u8 =
    \\{"name":"ds4_decide","description":"Ask the loaded decision model to evaluate bounded questions. Omit input to use the evidence bound to this tool, normally the latest user message; do not copy a long message into arguments. Returns choices and probabilities. Use choices for choice questions, levels for score questions, and neither for predicates.","parameters":{"type":"object","properties":{"questions":{"type":"array","minItems":1,"maxItems":400,"items":{"type":"object","properties":{"type":{"type":"string","enum":["predicate","choice","score"]},"name":{"type":"string"},"instructions":{"type":"string","minLength":1},"choices":{"type":"array","minItems":2,"maxItems":255,"items":{"type":"object","properties":{"value":{"type":["string","boolean"]},"description":{"type":"string"}},"required":["value","description"],"additionalProperties":false}},"levels":{"type":"array","minItems":2,"maxItems":10,"items":{"type":"object","properties":{"label":{"type":"string"},"description":{"type":"string"}},"required":["label","description"],"additionalProperties":false}}},"required":["type","instructions"],"additionalProperties":false}},"input":{"description":"Optional replacement evidence: text or native Decisions input parts. Omit to evaluate the bound evidence.","anyOf":[{"type":"string"},{"type":"array","items":{"type":"object"}}]}},"required":["questions"],"additionalProperties":false}}
;

fn bytes(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) "" else std.mem.span(pointer);
}

fn jsonOwned(value: anytype) ![:0]u8 {
    var encoded = try std.json.Stringify.valueAlloc(allocator, value, .{});
    errdefer allocator.free(encoded);
    const length = encoded.len;
    encoded = try allocator.realloc(encoded, length + 1);
    encoded[length] = 0;
    return encoded[0..length :0];
}

fn inputValue(value: Value) !Value {
    return switch (value) {
        .string, .array => value,
        else => error.InvalidDecisionInput,
    };
}

fn modelValue(value: Value) ![]const u8 {
    if (value != .string or value.string.len == 0 or std.mem.indexOfScalar(u8, value.string, 0) != null)
        return error.InvalidDecisionModel;
    return value.string;
}

/// The two strings are libc-owned and may be adopted into tool_schema_order.
pub const Binding = struct {
    model: [:0]u8,
    input: ?[:0]u8 = null,

    pub fn parse(raw: []const u8) !Binding {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const root = try std.json.parseFromSliceLeaky(Value, arena.allocator(), raw, .{ .allocate = .alloc_always });
        if (root != .object) return error.ExpectedDecisionTool;
        const kind = root.object.get("type") orelse return error.ExpectedDecisionTool;
        if (kind != .string or !std.mem.eql(u8, kind.string, tool_type)) return error.ExpectedDecisionTool;
        const name = try modelValue(root.object.get("model") orelse return error.MissingDecisionModel);
        var result = Binding{ .model = try allocator.dupeSentinel(u8, name, 0) };
        errdefer result.deinit();
        if (root.object.get("input")) |input| result.input = try jsonOwned(try inputValue(input));
        return result;
    }

    pub fn deinit(self: *Binding) void {
        allocator.free(self.model);
        if (self.input) |input| allocator.free(input);
    }
};

/// Verify a request's pinned model without loading, unloading or running one.
pub fn validateLoaded(comptime c: type, registry: ?*c.ds4_decision_registry, model: []const u8) !void {
    const present = registry orelse return error.NoDecisionModelLoaded;
    if (model.len == 0 or std.mem.indexOfScalar(u8, model, 0) != null) return error.InvalidDecisionModel;
    const name = try allocator.dupeSentinel(u8, model, 0);
    defer allocator.free(name);
    const metadata = c.ds4_decision_registry_models(present, name.ptr);
    if (metadata == null) return error.DecisionModelNotLoaded;
    allocator.free(std.mem.span(metadata));
}

fn object() Value {
    return .{ .object = .empty };
}

fn put(a: std.mem.Allocator, target: *Value, key: []const u8, value: Value) !void {
    try target.object.put(a, key, value);
}

fn textPart(a: std.mem.Allocator, text: []const u8) !Value {
    var part = object();
    try put(a, &part, "type", .{ .string = "input_text" });
    try put(a, &part, "text", .{ .string = text });
    return part;
}

fn imagePart(a: std.mem.Allocator, encoded: []const u8) !Value {
    const mime = if (std.mem.startsWith(u8, encoded, "\x89PNG\r\n\x1a\n"))
        "image/png"
    else if (std.mem.startsWith(u8, encoded, "\xff\xd8\xff"))
        "image/jpeg"
    else
        return error.UnsupportedDecisionImage;
    const size = std.base64.standard.Encoder.calcSize(encoded.len);
    const data = try a.alloc(u8, size);
    _ = std.base64.standard.Encoder.encode(data, encoded);
    const uri = try std.fmt.allocPrint(a, "data:{s};base64,{s}", .{ mime, data });
    var part = object();
    try put(a, &part, "type", .{ .string = "input_image" });
    try put(a, &part, "image_url", .{ .string = uri });
    return part;
}

/// Resolve once from the materialized conversation, before generated tool
/// calls are appended. Preserve image placement using the owned logical bytes.
/// The returned JSON is a Decisions input value, not an HTTP request body.
pub fn boundInput(comptime c: type, messages: [*c]const c.chat_msgs) ![:0]u8 {
    if (messages == null or messages.*.len <= 0 or messages.*.v == null) return error.NoDecisionUserInput;
    var index: usize = @intCast(messages.*.len);
    while (index != 0) {
        index -= 1;
        const message = messages.*.v[index];
        if (!std.mem.eql(u8, bytes(message.role), "user")) continue;
        const content = bytes(message.content);
        if (message.images.len == 0) {
            if (content.len == 0) return error.NoDecisionUserInput;
            return jsonOwned(content);
        }
        if (message.images.v == null) return error.InvalidDecisionImage;
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var parts = Value{ .array = std.json.Array.init(a) };
        var position: usize = 0;
        for (message.images.v[0..message.images.len]) |image| {
            if (image.encoded == null or image.encoded_len == 0) return error.InvalidDecisionImage;
            const marker = std.mem.sliceTo(&image.marker, 0);
            if (marker.len == 0) return error.UnsupportedDecisionImagePlacement;
            const found = std.mem.indexOfPos(u8, content, position, marker) orelse return error.UnsupportedDecisionImagePlacement;
            if (found != position) try parts.array.append(try textPart(a, content[position..found]));
            try parts.array.append(try imagePart(a, image.encoded[0..image.encoded_len]));
            position = found + marker.len;
        }
        if (position < content.len) try parts.array.append(try textPart(a, content[position..]));
        return jsonOwned(parts);
    }
    return error.NoDecisionUserInput;
}

pub const Invocation = struct {
    status: i32,
    output: [:0]u8,
    /// Resolved evidence for the provider call receipt; owned independently.
    input: ?[:0]u8 = null,
    /// Exact native Decisions request, useful for a reproducible trace.
    request: ?[:0]u8 = null,

    pub fn deinit(self: *Invocation) void {
        allocator.free(self.output);
        if (self.input) |input| allocator.free(input);
        if (self.request) |request| allocator.free(request);
    }
};

fn failure(status: i32, code: []const u8, message: []const u8, input: ?[:0]u8, request: ?[:0]u8) !Invocation {
    return .{
        .status = status,
        .output = try jsonOwned(.{ .@"error" = .{ .code = code, .message = message }, .status = status }),
        .input = input,
        .request = request,
    };
}

/// Client-bound model selection is authoritative. Tool arguments cannot add
/// a model or other endpoint controls. The native parser validates questions.
pub fn invoke(comptime c: type, registry: ?*c.ds4_decision_registry, model: []const u8, bound_input_json: ?[]const u8, args_json: []const u8, cancel: c.ds4_decision_cancel_fn, context: ?*anyopaque) !Invocation {
    if (cancel) |check| if (check(context)) return failure(499, "cancelled", "Decision request cancelled", null, null);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const arguments = std.json.parseFromSliceLeaky(Value, a, args_json, .{ .allocate = .alloc_always }) catch |err|
        return failure(400, "invalid_arguments", @errorName(err), null, null);
    if (arguments != .object) return failure(400, "invalid_arguments", "Expected an object with questions and optional input", null, null);
    for (arguments.object.keys()) |key| {
        if (!std.mem.eql(u8, key, "questions") and !std.mem.eql(u8, key, "input"))
            return failure(400, "invalid_arguments", "Only questions and input are accepted; the client fixes the decision model", null, null);
    }
    const questions = arguments.object.get("questions") orelse return failure(400, "invalid_arguments", "Missing questions", null, null);
    const input = if (arguments.object.get("input")) |provided|
        inputValue(provided) catch return failure(400, "invalid_arguments", "Input must be text or an array", null, null)
    else if (bound_input_json) |bound| blk: {
        const value = std.json.parseFromSliceLeaky(Value, a, bound, .{ .allocate = .alloc_always }) catch |err|
            return failure(400, "invalid_input", @errorName(err), null, null);
        break :blk inputValue(value) catch return failure(400, "invalid_input", "Bound input must be text or an array", null, null);
    } else return failure(400, "missing_input", "Supply input or a user message for the decision tool", null, null);
    var body = object();
    try put(a, &body, "model", .{ .string = model });
    try put(a, &body, "input", input);
    try put(a, &body, "questions", questions);
    const resolved_input = try jsonOwned(input);
    errdefer allocator.free(resolved_input);
    const request = try jsonOwned(body);
    errdefer allocator.free(request);
    const selected = registry orelse return failure(404, "model_not_loaded", "No decision model is loaded", resolved_input, request);
    var output: [*c]u8 = null;
    var message: [512]u8 = @splat(0);
    const status = c.ds4_decision_registry_request(selected, request.ptr, true, &output, &message, message.len, cancel, context);
    if (status == 200 and output != null) return .{ .status = status, .output = std.mem.span(output), .input = resolved_input, .request = request };
    if (output != null) allocator.free(std.mem.span(output));
    const detail = if (status == 499) "Decision request cancelled" else std.mem.sliceTo(&message, 0);
    return failure(if (status == 200) 500 else status, if (status == 499) "cancelled" else "decision_error", detail, resolved_input, request);
}

test "hosted binding owns a pinned model and optional JSON evidence" {
    var binding = try Binding.parse("{\"type\":\"ds4:decision\",\"model\":\"clef-flash\",\"input\":\"Invoice evidence\"}");
    defer binding.deinit();
    try std.testing.expectEqualStrings("clef-flash", binding.model);
    try std.testing.expectEqualStrings("\"Invoice evidence\"", binding.input.?);
    try std.testing.expectError(error.MissingDecisionModel, Binding.parse("{\"type\":\"ds4:decision\"}"));
    var parsed = try std.json.parseFromSlice(Value, std.testing.allocator, schema, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(function_name, parsed.value.object.get("name").?.string);
}

const TestC = struct {
    pub const ds4_decision_registry = opaque {};
    pub const ds4_decision_cancel_fn = ?*const fn (?*anyopaque) callconv(.c) bool;
    const Image = extern struct { marker: [64]u8 = @splat(0), encoded: [*c]const u8, encoded_len: usize };
    const Images = extern struct { v: [*c]const Image = null, len: usize = 0 };
    const Message = extern struct { role: [*c]const u8, content: [*c]const u8, images: Images = .{} };
    pub const chat_msgs = extern struct { v: [*c]const Message, len: c_int };
    const State = struct { request: ?[:0]u8 = null };

    pub fn ds4_decision_registry_request(registry: *ds4_decision_registry, body: [*:0]const u8, openai: bool, output: [*c][*c]u8, _: [*c]u8, _: usize, _: ds4_decision_cancel_fn, _: ?*anyopaque) c_int {
        std.debug.assert(openai);
        const state: *State = @ptrCast(@alignCast(registry));
        state.request = allocator.dupeSentinel(u8, std.mem.span(body), 0) catch unreachable;
        output[0] = (allocator.dupeSentinel(u8, "{\"answers\":[]}", 0) catch unreachable).ptr;
        return 200;
    }
};

test "bound input takes the latest user and preserves inline image placement" {
    const marker = "[[image:test]]";
    const png = "\x89PNG\r\n\x1a\n";
    var image = TestC.Image{ .encoded = png, .encoded_len = png.len };
    @memcpy(image.marker[0..marker.len], marker);
    const messages = [_]TestC.Message{
        .{ .role = "user", .content = "Earlier evidence" },
        .{ .role = "user", .content = "Before " ++ marker ++ " after", .images = .{ .v = &image, .len = 1 } },
        .{ .role = "assistant", .content = "Later assistant output" },
    };
    const input = try boundInput(TestC, &.{ .v = &messages, .len = messages.len });
    defer allocator.free(input);
    var parsed = try std.json.parseFromSlice(Value, std.testing.allocator, input, .{});
    defer parsed.deinit();
    const parts = parsed.value.array.items;
    try std.testing.expectEqual(@as(usize, 3), parts.len);
    try std.testing.expectEqualStrings("Before ", parts[0].object.get("text").?.string);
    try std.testing.expect(std.mem.startsWith(u8, parts[1].object.get("image_url").?.string, "data:image/png;base64,"));
    try std.testing.expectEqualStrings(" after", parts[2].object.get("text").?.string);
}

test "invocation uses bound evidence, permits override and rejects model switching" {
    var state = TestC.State{};
    defer if (state.request) |request| allocator.free(request);
    const registry: *TestC.ds4_decision_registry = @ptrCast(&state);
    var result = try invoke(TestC, registry, "clef-flash", "\"Bound user evidence\"", "{\"questions\":[]}", null, null);
    defer result.deinit();
    try std.testing.expectEqual(@as(i32, 200), result.status);
    try std.testing.expectEqualStrings("\"Bound user evidence\"", result.input.?);
    var parsed = try std.json.parseFromSlice(Value, std.testing.allocator, result.request.?, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("clef-flash", parsed.value.object.get("model").?.string);
    allocator.free(state.request.?);
    state.request = null;
    var overridden = try invoke(TestC, registry, "clef-flash", "\"Bound user evidence\"", "{\"input\":\"Short override\",\"questions\":[]}", null, null);
    defer overridden.deinit();
    try std.testing.expectEqualStrings("\"Short override\"", overridden.input.?);
    var forbidden = try invoke(TestC, registry, "clef-flash", null, "{\"model\":\"clef\",\"questions\":[]}", null, null);
    defer forbidden.deinit();
    try std.testing.expectEqual(@as(i32, 400), forbidden.status);
    try std.testing.expect(forbidden.request == null);
}
