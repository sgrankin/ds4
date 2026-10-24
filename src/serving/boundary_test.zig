//! Exercise the model boundary without HTTP, C server records or Metal.
//! Existing ds4_test --server covers the wire-to-model compatibility path.
const std = @import("std");
const types = @import("serving_types");
const prompt = @import("ds41_prompt");
const policy = @import("ds4_model");
const allocator = std.heap.page_allocator;

fn equal(expected: []const u8, actual: []const u8) !void {
    if (!std.mem.eql(u8, expected, actual)) {
        std.debug.print("expected: {s}\nactual:   {s}\n", .{ expected, actual });
        return error.PromptChanged;
    }
}

fn contains(actual: []const u8, expected: []const u8) !void {
    if (std.mem.find(u8, actual, expected) == null) {
        std.debug.print("missing: {s}\nactual:  {s}\n", .{ expected, actual });
        return error.PromptChanged;
    }
}

fn simpleConversation() !void {
    const messages = [_]types.Message{.{ .role = "user", .content = "Hi" }};
    const model: policy.Model = .{};
    const request: types.Request = .{ .messages = &messages, .reasoning = .{ .effort = .none } };
    const rendered = try model.prepare(allocator, request, .{}, 8192);
    defer allocator.free(rendered);
    try equal("<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜></think>", rendered);
}

fn sampledToolsAndReasoning() !void {
    // Spacing and argument order in sampled tool text belong to the stored
    // model transcript. Reconstructing it from parsed arguments changes KV.
    const sampled = "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"probe\">\n" ++
        "<｜DSML｜ parameter name=\"z\" string=\"false\"> 2 </｜DSML｜ parameter>\n" ++
        "<｜DSML｜ parameter name=\"a\" string=\"true\">x</｜DSML｜ parameter>\n" ++
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>";
    const calls = [_]types.ToolCall{.{ .id = "a", .name = "probe", .arguments = "{\"a\":\"x\",\"z\":2}" }};
    const messages = [_]types.Message{
        .{ .role = "user", .content = "Inspect" },
        .{ .role = "assistant", .content = "Before.", .reasoning = "Plan.", .calls = &calls, .raw_tool_text = sampled },
        .{ .role = "tool", .tool_call_id = "a", .content = "OK" },
        .{ .role = "user", .content = "Continue" },
    };
    const rendered = try prompt.render(allocator, &messages, .{ .reasoning = .high });
    defer allocator.free(rendered);
    try contains(rendered, "<think>Plan.</think>Before." ++ sampled ++ "<｜end▁of▁sentence｜>");
    try contains(rendered, "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n");
}

fn liveResultOrder() !void {
    const outputs = [_]types.Message{
        .{ .role = "tool", .tool_call_id = "b", .content = "SECOND" },
        .{ .role = "tool", .tool_call_id = "a", .content = "FIRST" },
    };
    const order = [_][]const u8{ "a", "b" };
    const rendered = try prompt.render(allocator, &outputs, .{
        .reasoning = .high,
        .live_tail = true,
        .initial_tool_order = &order,
    });
    defer allocator.free(rendered);
    try equal("<｜end▁of▁sentence｜><｜User｜><tool_result>FIRST</tool_result>\n\n" ++
        "<tool_result>SECOND</tool_result><｜Assistant｜><think>", rendered);
}

fn mixedResultsKeepUserText() !void {
    const calls = [_]types.ToolCall{
        .{ .id = "a", .name = "read", .arguments = "{}" },
        .{ .id = "b", .name = "read", .arguments = "{}" },
    };
    const spans = [_]types.ToolResultSpan{
        .{ .begin = 0, .end = 1, .id = "b", .content = "SECOND </tool_result> &lt;" },
        .{ .begin = 5, .end = 6, .id = "a", .content = "" },
    };
    const messages = [_]types.Message{
        .{ .role = "user", .content = "Inspect" },
        .{ .role = "assistant", .calls = &calls },
        .{ .role = "user", .content = "xNOTEy", .tool_results = &spans },
        .{ .role = "assistant", .content = "Done" },
        .{ .role = "user", .content = "Continue" },
    };
    const full = try prompt.render(allocator, &messages, .{ .reasoning = .high });
    defer allocator.free(full);
    try contains(full, "<tool_result></tool_result>\n\nNOTE\n\n" ++
        "<tool_result>SECOND &lt;/tool_result> &lt;</tool_result>");
    const tail = try prompt.render(allocator, &messages, .{ .reasoning = .high, .live_tail = true, .start = 2 });
    defer allocator.free(tail);
    const boundary = std.mem.find(u8, full, "<｜end▁of▁sentence｜>") orelse return error.MissingAssistantBoundary;
    try equal(full[boundary..], tail);
}

fn modelOwnsReasoningPolicy() !void {
    const model: policy.Model = .{};
    switch (model.reasoning(.{}, "deepseek-chat", 8192)) {
        .none => {},
        else => return error.AliasReasoningChanged,
    }
    // An explicit client control wins over the model-name alias, including
    // a numeric effort that must survive canonical request handling intact.
    switch (model.reasoning(.{ .enabled = true, .effort = .{ .level = 75 } }, "deepseek-chat", 8192)) {
        .level => |value| if (value != 75) return error.NumericReasoningChanged,
        else => return error.NumericReasoningChanged,
    }
}

fn qwenPreparation() !void {
    const model: policy.Model = .{ .family = .qwen };
    const messages = [_]types.Message{.{ .role = "user", .content = "Hi" }};
    const rendered = try model.prepare(allocator, .{
        .messages = &messages,
        .reasoning = .{ .enabled = false },
    }, .{}, 8192);
    defer allocator.free(rendered);
    try equal("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", rendered);
}

fn glmPreparation() !void {
    const model: policy.Model = .{ .family = .glm53 };
    const messages = [_]types.Message{.{ .role = "user", .content = "Hi" }};
    const rendered = try model.prepare(allocator, .{
        .messages = &messages,
        .reasoning = .{ .enabled = false },
    }, .{}, 8192);
    defer allocator.free(rendered);
    try equal("[gMASK]<sop><|user|>Hi<|assistant|><think></think>", rendered);
}

pub fn main() !void {
    try simpleConversation();
    try sampledToolsAndReasoning();
    try liveResultOrder();
    try mixedResultsKeepUserText();
    try modelOwnsReasoningPolicy();
    try qwenPreparation();
    try glmPreparation();
    std.debug.print("PASS: native serving/model boundary (7 cases)\n", .{});
}
