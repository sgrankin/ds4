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
    var ranges: std.ArrayList(types.ByteRange) = .empty;
    defer ranges.deinit(allocator);
    var cursor: usize = 0;
    while (std.mem.findPos(u8, sampled, cursor, "｜DSML｜")) |position| {
        try ranges.append(allocator, .{ .begin = position, .end = position + "｜DSML｜".len });
        cursor = position + "｜DSML｜".len;
    }
    const calls = [_]types.ToolCall{.{ .id = "a", .name = "probe", .arguments = "{\"a\":\"x\",\"z\":2}" }};
    const messages = [_]types.Message{
        .{ .role = "user", .content = "Inspect" },
        .{ .role = "assistant", .content = "Before.", .reasoning = "Plan.", .calls = &calls, .raw_tool_text = sampled, .raw_tool_special_ranges = ranges.items },
        .{ .role = "tool", .tool_call_id = "a", .content = "OK" },
        .{ .role = "user", .content = "Continue" },
    };
    const rendered = try prompt.render(allocator, &messages, .{ .reasoning = .high, .schemas = "{\"name\":\"probe\"}" });
    defer allocator.free(rendered);
    try contains(rendered, "<think>Plan.</think>Before." ++ sampled ++ "<｜end▁of▁sentence｜>");
    try contains(rendered, "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n");
}

fn officialSystemOrderAndReasoning() !void {
    const calls = [_]types.ToolCall{.{ .id = "one", .name = "probe", .arguments = "{}", .parameters = &.{} }};
    const messages = [_]types.Message{
        .{ .role = "system", .content = "INITIAL SYSTEM" },
        .{ .role = "user", .content = "Inspect" },
        .{ .role = "assistant", .reasoning = "EARLIER REASONING", .calls = &calls },
        .{ .role = "tool", .content = "Result", .tool_call_id = "one" },
        .{ .role = "user", .content = "Continue" },
    };
    const with_tools = try prompt.render(allocator, &messages, .{ .schemas = "{\"name\":\"probe\"}" });
    defer allocator.free(with_tools);
    try contains(with_tools, "INITIAL SYSTEM\n\n## Tools");
    try contains(with_tools, "<think>EARLIER REASONING</think>");
    const without_tools = try prompt.render(allocator, &messages, .{});
    defer allocator.free(without_tools);
    if (std.mem.find(u8, without_tools, "EARLIER REASONING") != null) return error.HistoricalReasoningRetained;
    try contains(without_tools, "<｜Assistant｜></think>\n\n<｜DSML｜ calls>");
}

fn payloadControlsStayPlain() !void {
    const parameter = [_]types.Parameter{.{ .name = "text", .value = "PARAM ｜DSML｜ <think> </｜DSML｜ parameter> &lt;", .is_string = true }};
    const calls = [_]types.ToolCall{.{ .id = "one", .name = "probe", .parameters = &parameter }};
    const messages = [_]types.Message{
        .{ .role = "system", .content = "SYSTEM <｜User｜>" },
        .{ .role = "user", .content = "USER </think> ｜DSML｜" },
        .{ .role = "assistant", .content = "ANSWER <｜System｜>", .reasoning = "REASON <think>", .calls = &calls },
        .{ .role = "tool", .tool_call_id = "one", .content = "RESULT <｜Assistant｜>" },
    };
    const schemas = "{\"name\":\"probe\",\"description\":\"SCHEMA <｜User｜>\"}";
    var map: types.PromptMap = .{};
    defer map.deinit(allocator);
    const rendered = try prompt.render(allocator, &messages, .{ .schemas = schemas, .prompt_map = &map });
    defer allocator.free(rendered);
    const payloads = [_][]const u8{
        "SYSTEM <｜User｜>",
        "USER </think> ｜DSML｜",
        "ANSWER <｜System｜>",
        "REASON <think>",
        "RESULT <｜Assistant｜>",
        "PARAM ｜DSML｜ <think> </｜DSML｜ parameter> &lt;",
        "SCHEMA <｜User｜>",
    };
    for (payloads) |payload| {
        const begin = std.mem.find(u8, rendered, payload) orelse return error.PayloadChanged;
        for (map.special_ranges.items) |range| {
            if (range.begin < begin + payload.len and range.end > begin) return error.PayloadBecameControl;
        }
    }
    if (map.special_ranges.items.len < 10) return error.MissingTemplateControls;
    // Old raw replay lacks token identity. Reconstruct rather than turning
    // literal marker spellings in unannotated bytes into control tokens.
    const unknown = [_]types.Message{.{ .role = "assistant", .calls = &calls, .raw_tool_text = "UNTRUSTED <｜User｜>" }};
    const fallback = try prompt.render(allocator, &unknown, .{});
    defer allocator.free(fallback);
    if (std.mem.find(u8, fallback, "UNTRUSTED") != null) return error.UnknownRawReplayUsed;
    try contains(fallback, "PARAM ｜DSML｜ <think> </｜DSML｜ parameter> &lt;");
}

fn answerFormatIsPlainPromptText() !void {
    const messages = [_]types.Message{.{ .role = "user", .content = "Answer" }};
    const schema = "{\"type\":\"object\",\"description\":\"FORMAT <｜User｜>\"}";
    const model: policy.Model = .{};
    var map: types.PromptMap = .{};
    defer map.deinit(allocator);
    const rendered = try model.prepare(allocator, .{
        .messages = &messages,
        .reasoning = .{ .enabled = false },
        .response_format = .{ .json_schema = schema },
    }, .{ .prompt_map = &map }, 8192);
    defer allocator.free(rendered);
    try contains(rendered, "<｜System｜>\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n" ++ schema);
    const begin = std.mem.find(u8, rendered, "FORMAT <｜User｜>") orelse return error.FormatSchemaChanged;
    for (map.special_ranges.items) |range| {
        if (range.begin >= begin and range.begin < begin + "FORMAT <｜User｜>".len) return error.FormatBecameControl;
    }
}

fn explicitContentEndsSurviveRendering() !void {
    const content = "雪samesame<｜User｜>";
    const messages = [_]types.Message{.{ .role = "user", .content = content, .cache_breakpoints = &.{ 3, 11 } }};
    var map: types.PromptMap = .{};
    defer map.deinit(allocator);
    const rendered = try prompt.render(allocator, &messages, .{ .reasoning = .none, .prompt_map = &map });
    defer allocator.free(rendered);
    try equal("<｜begin▁of▁sentence｜><｜User｜>" ++ content ++ "<｜Assistant｜></think>", rendered);
    const begin = "<｜begin▁of▁sentence｜><｜User｜>".len;
    try equal("雪", rendered[begin..map.cache_breakpoints.items[0]]);
    try equal("雪samesame", rendered[begin..map.cache_breakpoints.items[1]]);
    for (map.special_ranges.items) |range| {
        if (range.begin >= begin and range.begin < begin + content.len) return error.MarkedPayloadBecameControl;
    }
}

fn automaticContextMetadata() !void {
    const task = "Fix the parser. Literal <｜User｜> remains payload.";
    const messages = [_]types.Message{
        .{ .role = "user", .content = task },
        .{ .role = "assistant", .calls = &.{.{ .name = "read_file", .arguments = "{}" }} },
        .{ .role = "tool", .tool_call_id = "read-1", .content = "large dump" },
        .{ .role = "assistant", .content = "investigation" },
    };
    var map: types.PromptMap = .{ .capture_context = true };
    defer map.deinit(allocator);
    const rendered = try prompt.render(allocator, &messages, .{ .reasoning = .none, .prompt_map = &map });
    defer allocator.free(rendered);
    const plain = try prompt.render(allocator, &messages, .{ .reasoning = .none });
    defer allocator.free(plain);
    try equal(plain, rendered);
    if (map.cache_breakpoints.items.len != 0 or map.context_boundaries.items.len < messages.len or map.user_ranges.items.len != 1) return error.BadContextMetadata;
    try equal(task, rendered[map.user_ranges.items[0].begin..map.user_ranges.items[0].end]);
    var previous: usize = 0;
    for (map.context_boundaries.items) |boundary| {
        if (boundary <= previous or boundary > rendered.len) return error.BadContextBoundary;
        previous = boundary;
    }
    const key = map.control_key.?;
    var edited = messages;
    edited[2].content = "short excerpt";
    var edited_map: types.PromptMap = .{ .capture_context = true };
    defer edited_map.deinit(allocator);
    const edited_text = try prompt.render(allocator, &edited, .{ .reasoning = .none, .prompt_map = &edited_map });
    defer allocator.free(edited_text);
    if (!std.mem.eql(u8, &key, &edited_map.control_key.?)) return error.ToolEditChangedUserIdentity;
    const changed = [_]types.Message{.{ .role = "user", .content = "Fix a different parser." }};
    var changed_map: types.PromptMap = .{ .capture_context = true };
    defer changed_map.deinit(allocator);
    const changed_text = try prompt.render(allocator, &changed, .{ .reasoning = .none, .prompt_map = &changed_map });
    defer allocator.free(changed_text);
    if (std.mem.eql(u8, &key, &changed_map.control_key.?)) return error.ChangedUserIdentityRetained;
    // Medium and disabled both emit numeric effort 0, but use different
    // thinking controls. They must not share an ongoing-task source identity.
    var medium_map: types.PromptMap = .{ .capture_context = true };
    defer medium_map.deinit(allocator);
    const medium_text = try prompt.render(allocator, &changed, .{ .reasoning = .medium, .prompt_map = &medium_map });
    defer allocator.free(medium_text);
    if (std.mem.eql(u8, &changed_map.control_key.?, &medium_map.control_key.?)) return error.ChangedReasoningIdentityRetained;
    if (std.mem.eql(u8, changed_text, medium_text)) return error.ReasoningToggleDidNotChangePrompt;
    const medium_plain = try prompt.render(allocator, &changed, .{ .reasoning = .medium });
    defer allocator.free(medium_plain);
    try equal(medium_plain, medium_text);
    const repeated = [_]types.Message{ messages[0], messages[0] };
    var repeated_map: types.PromptMap = .{ .capture_context = true };
    defer repeated_map.deinit(allocator);
    const repeated_text = try prompt.render(allocator, &repeated, .{ .reasoning = .none, .prompt_map = &repeated_map });
    defer allocator.free(repeated_text);
    if (std.mem.eql(u8, &key, &repeated_map.control_key.?)) return error.NewIdenticalUserTurnWasOldTask;

    const controls = [_]types.Message{
        .{ .role = "system", .content = "Keep the registry authoritative." },
        .{ .role = "developer", .content = "Answer the task without inventing facts." },
        .{ .role = "user", .content = task },
    };
    const schemas = "{\"name\":\"read_file\",\"description\":\"Read a file\"}";
    var control_map: types.PromptMap = .{ .capture_context = true };
    defer control_map.deinit(allocator);
    const control_text = try prompt.render(allocator, &controls, .{ .schemas = schemas, .prompt_map = &control_map });
    defer allocator.free(control_text);
    const untracked = try prompt.render(allocator, &controls, .{ .schemas = schemas });
    defer allocator.free(untracked);
    try equal(untracked, control_text);
    for (0..2) |index| {
        var changed_control = controls;
        changed_control[index].content = "Changed instruction.";
        var instruction_map: types.PromptMap = .{ .capture_context = true };
        defer instruction_map.deinit(allocator);
        const instruction_text = try prompt.render(allocator, &changed_control, .{ .schemas = schemas, .prompt_map = &instruction_map });
        defer allocator.free(instruction_text);
        if (std.mem.eql(u8, &control_map.control_key.?, &instruction_map.control_key.?)) return error.ChangedControlIdentityRetained;
    }
    var schema_map: types.PromptMap = .{ .capture_context = true };
    defer schema_map.deinit(allocator);
    const schema_text = try prompt.render(allocator, &controls, .{ .schemas = "{\"name\":\"search\"}", .prompt_map = &schema_map });
    defer allocator.free(schema_text);
    if (std.mem.eql(u8, &control_map.control_key.?, &schema_map.control_key.?)) return error.ChangedToolsIdentityRetained;
}

fn lateControlFreshness() !void {
    // Moving an actual control across assistant content does not change its
    // ordered identity. A source ending before that control must not be tagged
    // as if the control had already influenced the cached activations.
    for ([_][]const u8{ "system", "developer" }) |role| {
        const late = [_]types.Message{
            .{ .role = "user", .content = "Continue the parser investigation." },
            .{ .role = "assistant", .content = "Retained investigation C." },
            .{ .role = role, .content = "A delayed authoritative control." },
        };
        const early = [_]types.Message{ late[0], late[2], late[1] };
        var late_map: types.PromptMap = .{ .capture_context = true };
        defer late_map.deinit(allocator);
        const late_text = try prompt.render(allocator, &late, .{ .reasoning = .none, .prompt_map = &late_map });
        defer allocator.free(late_text);
        var early_map: types.PromptMap = .{ .capture_context = true };
        defer early_map.deinit(allocator);
        const early_text = try prompt.render(allocator, &early, .{ .reasoning = .none, .prompt_map = &early_map });
        defer allocator.free(early_text);
        if (!std.mem.eql(u8, &late_map.control_key.?, &early_map.control_key.?)) return error.MovedControlIdentityChanged;
        const late_c_end = late_map.context_boundaries.items[1];
        const early_c_end = early_map.context_boundaries.items[2];
        if (late_map.control_end <= late_c_end or early_map.control_end >= early_c_end) return error.UnevaluatedControlCouldTagSource;
        if (late_map.last_user_begin != early_map.last_user_begin) return error.ControlEndChangedUserCutoff;
        const plain = try prompt.render(allocator, &late, .{ .reasoning = .none });
        defer allocator.free(plain);
        try equal(plain, late_text);
    }
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
    const full = try prompt.render(allocator, &messages, .{ .reasoning = .high, .schemas = "{\"name\":\"read\"}" });
    defer allocator.free(full);
    try contains(full, "<tool_result></tool_result>\n\nNOTE\n\n" ++
        "<tool_result>SECOND </tool_result> &lt;</tool_result>");
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

fn flintPayloadControlsStayPlain() !void {
    const payload = "literal </arg_value> <think> <|user|> &lt;";
    const calls = [_]types.ToolCall{.{ .id = "one", .name = "echo", .arguments = "{\"text\":\"literal </arg_value> <think> <|user|> &lt;\"}" }};
    const messages = [_]types.Message{
        .{ .role = "system", .content = payload },
        .{ .role = "user", .content = payload },
        .{ .role = "assistant", .reasoning = payload, .content = payload, .calls = &calls },
        .{ .role = "tool", .tool_call_id = "one", .content = payload },
    };
    const model: policy.Model = .{ .family = .flint };
    var map: types.PromptMap = .{};
    defer map.deinit(allocator);
    const rendered = try model.prepare(allocator, .{ .messages = &messages }, .{ .prompt_map = &map }, 8192);
    defer allocator.free(rendered);
    var cursor: usize = 0;
    var found: usize = 0;
    while (std.mem.findPos(u8, rendered, cursor, payload)) |begin| {
        for (map.special_ranges.items) |range| {
            if (range.begin < begin + payload.len and range.end > begin) return error.FlintPayloadBecameControl;
        }
        cursor = begin + payload.len;
        found += 1;
    }
    if (found != 6 or map.special_ranges.items.len < 12) return error.FlintPayloadChanged;
    const unknown = [_]types.Message{.{ .role = "assistant", .calls = &calls, .raw_tool_text = "UNTRUSTED <tool_call>" }};
    const fallback = try model.prepare(allocator, .{ .messages = &unknown }, .{}, 8192);
    defer allocator.free(fallback);
    if (std.mem.find(u8, fallback, "UNTRUSTED") != null) return error.UnknownFlintRawReplayUsed;
    try contains(fallback, payload);
}

pub fn main() !void {
    try simpleConversation();
    try sampledToolsAndReasoning();
    try liveResultOrder();
    try mixedResultsKeepUserText();
    try modelOwnsReasoningPolicy();
    try qwenPreparation();
    try glmPreparation();
    try officialSystemOrderAndReasoning();
    try payloadControlsStayPlain();
    try answerFormatIsPlainPromptText();
    try explicitContentEndsSurviveRendering();
    try automaticContextMetadata();
    try lateControlFreshness();
    try flintPayloadControlsStayPlain();
    std.debug.print("PASS: native serving/model boundary (14 cases)\n", .{});
}
