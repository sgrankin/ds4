//! Production-linked token provenance, Responses streaming and SSD tool replay.
const std = @import("std");
const c = @import("ds41_test_c");
const Projection = @import("ds41_output.zig");
const FlintProjection = @import("flint_output.zig");
const a = std.heap.c_allocator;
const expect = std.testing.expect;
const equal = std.testing.expectEqualStrings;

const Capture = struct {
    events: std.ArrayList([]u8) = .empty,
    fn emit(context: ?*anyopaque, raw: [*c]const u8, length: usize) callconv(.c) bool {
        const self: *Capture = @ptrCast(@alignCast(context.?));
        const copy = a.dupe(u8, raw[0..length]) catch return false;
        self.events.append(a, copy) catch {
            a.free(copy);
            return false;
        };
        return true;
    }
    fn deinit(self: *Capture) void {
        for (self.events.items) |event| a.free(event);
        self.events.deinit(a);
    }
};
fn text(value: [*c]const u8) []const u8 {
    return if (value == null) "" else std.mem.span(value);
}
fn freeCalls(calls: *c.tool_calls) void {
    if (calls.len > 0) for (calls.v[0..@intCast(calls.len)]) |call| {
        c.free(call.id);
        c.free(call.name);
        c.free(call.arguments);
    };
    c.free(calls.v);
    c.free(calls.raw_tool_text);
    c.free(calls.raw_special_ranges);
    calls.* = std.mem.zeroes(c.tool_calls);
}
fn field(value: std.json.Value, key: []const u8) !std.json.Value {
    return if (value == .object) value.object.get(key) orelse error.MissingField else error.ExpectedObject;
}
fn stringField(value: std.json.Value, key: []const u8) ![]const u8 {
    const result = try field(value, key);
    return if (result == .string) result.string else error.ExpectedString;
}
fn flintStream() !void {
    const raw = "<tool_call>echo<arg_key>text</arg_key><arg_value>λ literal </arg_value> data</arg_value></tool_call>";
    const markers = [_][]const u8{ "<tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "</tool_call>" };
    var ranges: std.ArrayList(c.ds4_prompt_range) = .empty;
    defer ranges.deinit(a);
    var cursor: usize = 0;
    for (markers, 0..) |marker, index| {
        const begin = if (index == 4) std.mem.findLast(u8, raw, marker).? else std.mem.findPos(u8, raw, cursor, marker).?;
        try ranges.append(a, .{ .begin = begin, .end = begin + marker.len });
        cursor = begin + marker.len;
    }
    var request = std.mem.zeroes(c.request);
    request.api = c.API_RESPONSES;
    request.kind = c.REQ_CHAT;
    request.model_syntax = c.SERVER_MODEL_SYNTAX_FLINT;
    request.has_tools = true;
    request.model = @constCast("flint-test");
    request.sampled_special_ranges = ranges.items.ptr;
    request.sampled_special_count = ranges.items.len;
    var key: [*c]u8 = @constCast("text");
    var schema: [*c]u8 = @constCast("{\"type\":\"string\"}");
    var order = std.mem.zeroes(c.tool_schema_order);
    order.name = @constCast("echo");
    order.prop = &key;
    order.prop_schema = &schema;
    order.len = 1;
    request.tool_orders = .{ .v = &order, .len = 1, .cap = 1 };
    var capture = Capture{};
    defer capture.deinit();
    const sink = c.server_response_sink{ .context = &capture, .emit = Capture.emit, .load_parent = null };
    request.response_sink = &sink;
    var stream: c.responses_stream = undefined;
    c.ds4_server_output_init(&request, &stream);
    defer c.ds4_server_output_free(&stream);
    stream.active = true;
    var stop: c.sig_atomic_t = 0;
    try expect(c.ds4_server_output_created(-1, &request, &stream, 1234, &stop));
    var prefix: usize = 0;
    while (prefix < raw.len) {
        prefix += 1;
        for (ranges.items) |range| if (prefix > range.begin and prefix < range.end) {
            prefix = range.end;
        };
        try expect(c.ds4_server_output_update(-1, &request, &stream, raw, prefix, false, &stop));
    }
    var calls = std.mem.zeroes(c.tool_calls);
    defer freeCalls(&calls);
    var content: [*c]u8 = null;
    var reasoning: [*c]u8 = null;
    defer {
        c.free(content);
        c.free(reasoning);
    }
    var finish: [*c]const u8 = "tool_calls";
    var errors: [160]u8 = @splat(0);
    var recovered = false;
    try expect(c.ds4_server_reply_for_sampled_response(&request, raw, true, &finish, &errors, errors.len, &content, &reasoning, &calls, &recovered));
    c.ds4_server_output_apply_ids(&calls, &stream);
    try expect(c.ds4_server_output_finish(-1, &request, &stream, raw, raw.len, null, &calls, "tool_calls", 10, 25, 1234, &stop));
    var arguments: std.ArrayList(u8) = .empty;
    defer arguments.deinit(a);
    var deltas: usize = 0;
    for (capture.events.items) |event| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, event, .{});
        defer parsed.deinit();
        const kind = try stringField(parsed.value, "type");
        if (std.mem.eql(u8, kind, "response.function_call_arguments.delta")) {
            try arguments.appendSlice(a, try stringField(parsed.value, "delta"));
            deltas += 1;
        } else if (std.mem.eql(u8, kind, "response.function_call_arguments.done")) {
            try equal(arguments.items, try stringField(parsed.value, "arguments"));
        }
    }
    try expect(deltas > 10);
    try equal("{\"text\":\"λ literal </arg_value> data\"}", arguments.items);
}
fn flintSchemaCoercion() !void {
    var request = std.mem.zeroes(c.request);
    request.api = c.API_RESPONSES;
    request.kind = c.REQ_CHAT;
    request.model_syntax = c.SERVER_MODEL_SYNTAX_FLINT;
    request.has_tools = true;
    request.model = (try a.dupeSentinel(u8, "flint-test", 0)).ptr;
    defer c.ds4_server_input_request_free(&request);
    c.ds4_server_input_schema_order_add(&request.tool_orders,
        \\{"name":"select","parameters":{"type":"object","$defs":{
        \\ "truth":{"const":"true"}},"properties":{
        \\ "ref":{"$ref":"#/$defs/truth"},"mixed":{"enum":[1,"x"]},
        \\ "nullable":{"type":["string","null"]}}}}
    , null, null, false);
    try expect(request.tool_orders.len == 1 and request.tool_orders.v[0].parameters_json != null);
    const raw = "<tool_call>select<arg_key>ref</arg_key><arg_value>true</arg_value><arg_key>mixed</arg_key><arg_value>1</arg_value><arg_key>nullable</arg_key><arg_value>null</arg_value></tool_call>";
    const markers = [_][]const u8{ "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>" };
    var ranges: std.ArrayList(c.ds4_prompt_range) = .empty;
    defer ranges.deinit(a);
    var cursor: usize = 0;
    while (cursor < raw.len) {
        var matched = false;
        for (markers) |marker| if (std.mem.startsWith(u8, raw[cursor..], marker)) {
            try ranges.append(a, .{ .begin = cursor, .end = cursor + marker.len });
            cursor += marker.len;
            matched = true;
            break;
        };
        if (!matched) cursor += 1;
    }
    request.sampled_special_ranges = ranges.items.ptr;
    request.sampled_special_count = ranges.items.len;
    var capture = Capture{};
    defer capture.deinit();
    const sink = c.server_response_sink{ .context = &capture, .emit = Capture.emit, .load_parent = null };
    request.response_sink = &sink;
    var stream: c.responses_stream = undefined;
    c.ds4_server_output_init(&request, &stream);
    defer c.ds4_server_output_free(&stream);
    stream.active = true;
    var stop: c.sig_atomic_t = 0;
    cursor = 0;
    while (cursor < raw.len) {
        cursor += 1;
        for (ranges.items) |range| if (cursor > range.begin and cursor < range.end) {
            cursor = range.end;
        };
        try expect(c.ds4_server_output_update(-1, &request, &stream, raw, cursor, false, &stop));
    }
    var calls = std.mem.zeroes(c.tool_calls);
    defer freeCalls(&calls);
    var content: [*c]u8 = null;
    var reasoning: [*c]u8 = null;
    defer {
        c.free(content);
        c.free(reasoning);
    }
    var finish: [*c]const u8 = "tool_calls";
    var errors: [160]u8 = @splat(0);
    var recovered = false;
    try expect(c.ds4_server_reply_for_sampled_response(&request, raw, true, &finish, &errors, errors.len, &content, &reasoning, &calls, &recovered));
    try expect(calls.len == 1);
    const expected = "{\"ref\":\"true\",\"mixed\":1,\"nullable\":null}";
    try equal(expected, text(calls.v[0].arguments));
    const arguments = try std.json.parseFromSlice(std.json.Value, a, text(calls.v[0].arguments), .{});
    defer arguments.deinit();
    try expect((try field(arguments.value, "ref")) == .string);
    try expect((try field(arguments.value, "mixed")) == .integer);
    try expect((try field(arguments.value, "nullable")) == .null);
    var emitted: std.ArrayList(u8) = .empty;
    defer emitted.deinit(a);
    for (capture.events.items) |event| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, event, .{});
        defer parsed.deinit();
        const kind = try stringField(parsed.value, "type");
        if (std.mem.eql(u8, kind, "response.function_call_arguments.delta")) try emitted.appendSlice(a, try stringField(parsed.value, "delta"));
        if (std.mem.eql(u8, kind, "response.function_call_arguments.done")) try equal(emitted.items, try stringField(parsed.value, "arguments"));
    }
    try equal(expected, emitted.items);
}
fn otherProtocols(request: *c.request, raw: [:0]const u8, calls: *c.tool_calls, expected: []const ExpectedCall) !void {
    var chat = std.mem.zeroes(c.buf);
    defer c.free(chat.ptr);
    c.ds4_server_openai_calls(&chat, calls, "fixture", false);
    const encoded = try std.json.parseFromSlice(std.json.Value, a, text(chat.ptr), .{});
    defer encoded.deinit();
    try expect(encoded.value.array.items.len == expected.len);
    for (encoded.value.array.items, expected) |item, want| {
        try equal(want.arguments, try stringField(try field(item, "function"), "arguments"));
    }
    // Anthropic's nonstream tool_use input is an object, so it omits partial
    // calls rather than inventing an empty or repaired input object.
    var blocks = std.mem.zeroes(c.buf);
    defer c.free(blocks.ptr);
    c.ds4_server_anthropic_content(&blocks, "", "", calls, "fixture", &request.tool_orders);
    const content = try std.json.parseFromSlice(std.json.Value, a, text(blocks.ptr), .{});
    defer content.deinit();
    var complete: usize = 0;
    for (expected) |want| if (!want.incomplete) {
        complete += 1;
    };
    var uses: usize = 0;
    for (content.value.array.items) |item| if (std.mem.eql(u8, try stringField(item, "type"), "tool_use")) {
        uses += 1;
    };
    try expect(uses == complete);

    // A local socket pair exercises the real Anthropic SSE lifecycle without
    // a server, inference, model load or network listener.
    var sockets: [2]c_int = undefined;
    try expect(c.socketpair(c.AF_UNIX, c.SOCK_STREAM, 0, &sockets) == 0);
    defer _ = c.close(sockets[1]);
    var sender_open = true;
    defer if (sender_open) {
        _ = c.close(sockets[0]);
    };
    const old_api = request.api;
    request.api = c.API_ANTHROPIC;
    defer request.api = old_api;
    var stream: c.anthropic_stream = undefined;
    defer c.ds4_server_anthropic_free(&stream);
    var stop: c.sig_atomic_t = 0;
    try expect(c.ds4_server_anthropic_start(sockets[0], request, "msg_fixture", 10, &stream, &stop));
    try expect(c.ds4_server_anthropic_update(sockets[0], null, null, request, "msg_fixture", &stream, raw.ptr, raw.len, false, &stop));
    c.ds4_server_anthropic_apply_ids(calls, &stream);
    try expect(c.ds4_server_anthropic_finish(sockets[0], null, null, request, "msg_fixture", &stream, raw.ptr, raw.len, calls, "length", 25, &stop));
    _ = c.close(sockets[0]);
    sender_open = false;
    var wire: std.ArrayList(u8) = .empty;
    defer wire.deinit(a);
    var buffer: [4096]u8 = undefined;
    while (true) {
        const count = c.read(sockets[1], &buffer, buffer.len);
        try expect(count >= 0);
        if (count == 0) break;
        try wire.appendSlice(a, buffer[0..@intCast(count)]);
    }
    var fragments: [2]std.ArrayList(u8) = @splat(.empty);
    defer for (&fragments) |*fragment| fragment.deinit(a);
    var opened: usize = 0;
    var closed: usize = 0;
    var ended = false;
    var lines = std.mem.splitScalar(u8, wire.items, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "data: ")) continue;
        const json = try std.json.parseFromSlice(std.json.Value, a, line[6..], .{});
        defer json.deinit();
        const kind = try stringField(json.value, "type");
        if (std.mem.eql(u8, kind, "content_block_start")) {
            opened += 1;
        } else if (std.mem.eql(u8, kind, "content_block_stop")) {
            closed += 1;
        } else if (std.mem.eql(u8, kind, "content_block_delta")) {
            const index: usize = @intCast((try field(json.value, "index")).integer);
            try fragments[index].appendSlice(a, try stringField(try field(json.value, "delta"), "partial_json"));
        } else if (std.mem.eql(u8, kind, "message_delta")) {
            try equal("max_tokens", try stringField(try field(json.value, "delta"), "stop_reason"));
            ended = true;
        }
    }
    try expect(opened == expected.len and closed == expected.len and ended);
    for (expected, 0..) |want, i| try equal(want.arguments, fragments[i].items);
}

const ExpectedCall = struct { arguments: []const u8, incomplete: bool = true };
fn truncatedTool(raw: [:0]const u8, flint: bool, expected: ?[]const ExpectedCall) !void {
    var ranges: std.ArrayList(c.ds4_prompt_range) = .empty;
    defer ranges.deinit(a);
    const markers = if (flint)
        &[_][]const u8{ "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>" }
    else
        &[_][]const u8{"｜DSML｜"};
    var offset: usize = 0;
    while (offset < raw.len) {
        var matched = false;
        for (markers) |marker| if (std.mem.startsWith(u8, raw[offset..], marker)) {
            try ranges.append(a, .{ .begin = offset, .end = offset + marker.len });
            offset += marker.len;
            matched = true;
            break;
        };
        if (!matched) offset += 1;
    }
    var request = std.mem.zeroes(c.request);
    request.api = c.API_RESPONSES;
    request.kind = c.REQ_CHAT;
    request.model_syntax = if (flint) c.SERVER_MODEL_SYNTAX_FLINT else c.SERVER_MODEL_SYNTAX_DEEPSEEK41;
    request.has_tools = true;
    request.model = @constCast("truncation-test");
    request.think_mode = c.DS4_THINK_NONE;
    request.sampled_special_ranges = ranges.items.ptr;
    request.sampled_special_count = ranges.items.len;
    var key: [*c]u8 = @constCast("text");
    var schema: [*c]u8 = @constCast("{\"type\":\"string\"}");
    var order = std.mem.zeroes(c.tool_schema_order);
    order.name = @constCast("echo");
    order.prop = &key;
    order.prop_schema = &schema;
    order.len = 1;
    request.tool_orders = .{ .v = &order, .len = 1, .cap = 1 };
    var calls = std.mem.zeroes(c.tool_calls);
    defer freeCalls(&calls);
    var content: [*c]u8 = null;
    var reasoning: [*c]u8 = null;
    defer {
        c.free(content);
        c.free(reasoning);
    }
    var finish: [*c]const u8 = "length";
    var errors: [160]u8 = @splat(0);
    var recovered = false;
    const parsed = c.ds4_server_reply_for_sampled_response(&request, raw.ptr, true, &finish, &errors, errors.len, &content, &reasoning, &calls, &recovered);
    if (expected == null) {
        try expect(!parsed and std.mem.eql(u8, text(finish), "error"));
        try expect(std.mem.find(u8, text(content), "｜DSML｜") == null);
        try expect(calls.len == 0);
        return;
    }
    try expect(parsed and !recovered);
    // The same bytes at ordinary EOS are malformed, not a completed call.
    var stopped_finish: [*c]const u8 = "stop";
    var stopped_calls = std.mem.zeroes(c.tool_calls);
    defer freeCalls(&stopped_calls);
    var stopped_content: [*c]u8 = null;
    var stopped_reasoning: [*c]u8 = null;
    defer {
        c.free(stopped_content);
        c.free(stopped_reasoning);
    }
    try expect(!c.ds4_server_reply_for_sampled_response(&request, raw.ptr, true, &stopped_finish, &errors, errors.len, &stopped_content, &stopped_reasoning, &stopped_calls, &recovered));
    try equal("error", text(stopped_finish));
    try expect(stopped_calls.len == 0);
    try equal("length", text(finish));
    try equal("", text(content));
    try expect(@as(usize, @intCast(calls.len)) == expected.?.len);
    // Partial turns never acquire a raw DSML replay record, even if an earlier
    // invocation within the same turn was complete.
    try expect(calls.raw_tool_text == null and calls.raw_special_count == 0);
    var partial = false;
    for (expected.?, 0..) |want, i| {
        try equal(want.arguments, text(calls.v[i].arguments));
        try expect(calls.v[i].incomplete == want.incomplete);
        partial = partial or want.incomplete;
    }
    if (partial) try expect(!c.ds4_server_generation_canonical_needed(null, &calls));
    var memory = std.mem.zeroes(c.tool_memory);
    defer c.ds4_server_tool_memory_free(&memory);
    var mutex: c.pthread_mutex_t = undefined;
    try expect(c.pthread_mutex_init(&mutex, null) == 0);
    defer _ = c.pthread_mutex_destroy(&mutex);
    c.ds4_server_tool_memory_remember(&memory, &mutex, &calls);
    try expect(memory.entries == 0);
    if (expected.?.len != 0) try otherProtocols(&request, raw, &calls, expected.?);

    var capture = Capture{};
    defer capture.deinit();
    const sink = c.server_response_sink{ .context = &capture, .emit = Capture.emit, .load_parent = null };
    request.response_sink = &sink;
    var stream: c.responses_stream = undefined;
    c.ds4_server_output_init(&request, &stream);
    defer c.ds4_server_output_free(&stream);
    stream.active = true;
    var stop: c.sig_atomic_t = 0;
    try expect(c.ds4_server_output_created(-1, &request, &stream, 1234, &stop));
    var prefix: usize = 0;
    while (prefix < raw.len) {
        prefix += 1;
        for (ranges.items) |range| if (prefix > range.begin and prefix < range.end) {
            prefix = range.end;
        };
        try expect(c.ds4_server_output_update(-1, &request, &stream, raw.ptr, prefix, false, &stop));
    }
    c.ds4_server_output_apply_ids(&calls, &stream);
    try expect(c.ds4_server_output_finish(-1, &request, &stream, raw.ptr, raw.len, null, &calls, finish, 10, 25, 1234, &stop));
    var fragments: [2]std.ArrayList(u8) = @splat(.empty);
    defer for (&fragments) |*fragment| fragment.deinit(a);
    var argument_done: [2]usize = @splat(0);
    var item_done: [2]usize = @splat(0);
    var terminal = false;
    for (capture.events.items) |event| {
        const json = try std.json.parseFromSlice(std.json.Value, a, event, .{});
        defer json.deinit();
        const kind = try stringField(json.value, "type");
        if (std.mem.eql(u8, kind, "response.function_call_arguments.delta")) {
            const index: usize = @intCast((try field(json.value, "output_index")).integer);
            try fragments[index].appendSlice(a, try stringField(json.value, "delta"));
        } else if (std.mem.eql(u8, kind, "response.function_call_arguments.done")) {
            const index: usize = @intCast((try field(json.value, "output_index")).integer);
            try equal(fragments[index].items, try stringField(json.value, "arguments"));
            argument_done[index] += 1;
        } else if (std.mem.eql(u8, kind, "response.output_item.done")) {
            const index: usize = @intCast((try field(json.value, "output_index")).integer);
            const item = try field(json.value, "item");
            try equal(if (expected.?[index].incomplete) "incomplete" else "completed", try stringField(item, "status"));
            try equal(text(calls.v[index].id), try stringField(item, "call_id"));
            try equal(fragments[index].items, try stringField(item, "arguments"));
            item_done[index] += 1;
        } else if (std.mem.eql(u8, kind, "response.incomplete")) {
            const response = try field(json.value, "response");
            try equal("incomplete", try stringField(response, "status"));
            try equal("max_output_tokens", try stringField(try field(response, "incomplete_details"), "reason"));
            const output = try field(response, "output");
            try expect(output.array.items.len == expected.?.len);
            for (output.array.items, 0..) |item, index| {
                try equal(expected.?[index].arguments, try stringField(item, "arguments"));
                try equal(if (expected.?[index].incomplete) "incomplete" else "completed", try stringField(item, "status"));
                try equal(text(&stream.tool_items[index].fc_id), try stringField(item, "id"));
                try equal(text(calls.v[index].id), try stringField(item, "call_id"));
            }
            terminal = true;
        } else try expect(!std.mem.eql(u8, kind, "response.failed") and !std.mem.eql(u8, kind, "response.output_text.delta"));
    }
    try expect(terminal);
    for (expected.?, 0..) |want, i| {
        try equal(want.arguments, fragments[i].items);
        try expect(argument_done[i] == 1 and item_done[i] == 1);
    }
}
fn truncationCases() !void {
    const open = "<｜DSML｜ calls><｜DSML｜ invoke name=\"echo\"><｜DSML｜ parameter name=\"text\" string=\"true\">";
    try truncatedTool(open ++ "λ & <", false, &.{.{ .arguments = "{\"text\":\"λ & <" }});
    try truncatedTool(open ++ "λ \xe2\x82", false, &.{.{ .arguments = "{\"text\":\"λ " }});
    try truncatedTool(open ++ "value</｜DSML｜ parameter>", false, &.{.{ .arguments = "{\"text\":\"value\"" }});
    try truncatedTool("<｜DSML｜ calls><｜DSML｜ invoke name=\"echo\"><｜DSML｜ parameter name=\"n\" string=\"false\">tru", false, &.{.{ .arguments = "{\"n\":tru" }});
    try truncatedTool(open ++ "first</｜DSML｜ parameter></｜DSML｜ invoke><｜DSML｜ invoke name=\"echo\"><｜DSML｜ parameter name=\"text\" string=\"true\">second", false, &.{
        .{ .arguments = "{\"text\":\"first\"}", .incomplete = false },
        .{ .arguments = "{\"text\":\"second" },
    });
    try truncatedTool("<｜DSML｜ calls><｜DSML｜ invoke name=\"echo", false, &.{});
    try truncatedTool("<｜DSML｜ calls><｜DSML｜ invoke name=\"echo\">garbage", false, null);
    try truncatedTool("<｜DSML｜ calls><｜DSML｜ invoke name=\"echo\"><｜DSML｜ parameter name=\"n\" string=\"false\">truX", false, null);
    try truncatedTool("<tool_call>echo<arg_key>text</arg_key><arg_value>λ partial", true, &.{.{ .arguments = "{\"text\":\"λ partial" }});
    try truncatedTool("<tool_call>echo<arg_key>text</arg_key><arg_value>value</arg_value>garbage", true, null);
    std.debug.print("Typed DSML/Flint truncation and incomplete-stream cases: PASS\n", .{});
}

fn run() !void {
    var request = std.mem.zeroes(c.request);
    request.api = c.API_RESPONSES;
    request.kind = c.REQ_CHAT;
    request.model_syntax = c.SERVER_MODEL_SYNTAX_DEEPSEEK41;
    request.has_tools = true;
    request.think_mode = c.DS4_THINK_NONE;
    request.temperature = 1;
    request.top_p = 1;
    request.model = @constCast("deepseek-test");
    var calls = std.mem.zeroes(c.tool_calls);
    defer freeCalls(&calls);
    var content: [*c]u8 = null;
    var reasoning: [*c]u8 = null;
    defer {
        c.free(content);
        c.free(reasoning);
    }
    var finish: [*c]const u8 = "stop";
    var errors: [160]u8 = @splat(0);
    var recovered = false;
    const literal = "A literal </think> and <｜DSML｜ calls> is documentation.";
    try expect(c.ds4_server_reply_for_sampled_response(&request, literal, false, &finish, &errors, errors.len, &content, &reasoning, &calls, &recovered));
    try equal(literal, text(content));
    try expect(calls.len == 0 and reasoning == null);
    c.free(content);
    content = null;

    const raw = "reasoning </think> is literal</think><｜DSML｜ calls><｜DSML｜ invoke name=\"echo\"><｜DSML｜ parameter name=\"text\" string=\"true\">λ raw </think> </｜DSML｜ parameter> &lt; &amp;</｜DSML｜ parameter><｜DSML｜ parameter name=\"n\" string=\"false\">[ 1, 2 ]</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
    const marker = "｜DSML｜";
    var ranges: std.ArrayList(c.ds4_prompt_range) = .empty;
    defer ranges.deinit(a);
    const close = std.mem.indexOf(u8, raw, "literal</think>").? + "literal".len;
    try ranges.append(a, .{ .begin = close, .end = close + 8 });
    var offset: usize = close + 8;
    var count: usize = 0;
    while (std.mem.findPos(u8, raw, offset, marker)) |position| {
        if (count != 3) try ranges.append(a, .{ .begin = position, .end = position + marker.len });
        count += 1;
        offset = position + marker.len;
    }
    request.think_mode = c.DS4_THINK_HIGH;
    request.reasoning_summary_emit = true;
    request.sampled_special_ranges = ranges.items.ptr;
    request.sampled_special_count = ranges.items.len;
    try expect(c.ds4_server_reply_for_sampled_response(&request, raw, true, &finish, &errors, errors.len, &content, &reasoning, &calls, &recovered));
    try equal("reasoning </think> is literal", text(reasoning));
    try expect(calls.len == 1 and calls.raw_special_count > 0);
    const expected = "{\"text\":\"λ raw </think> </｜DSML｜ parameter> &lt; &amp;\",\"n\":[1,2]}";
    try equal(expected, text(calls.v[0].arguments));

    var capture = Capture{};
    defer capture.deinit();
    const sink = c.server_response_sink{ .context = &capture, .emit = Capture.emit, .load_parent = null };
    request.response_sink = &sink;
    var stream: c.responses_stream = undefined;
    c.ds4_server_output_init(&request, &stream);
    defer c.ds4_server_output_free(&stream);
    stream.active = true;
    var stop: c.sig_atomic_t = 0;
    try expect(c.ds4_server_output_created(-1, &request, &stream, 1234, &stop));
    // Send successive ordinary-byte prefixes, keeping each marked token atomic.
    var prefix: usize = 0;
    while (prefix < raw.len) {
        prefix += 1;
        for (ranges.items) |range| if (prefix > range.begin and prefix < range.end) {
            prefix = range.end;
        };
        try expect(c.ds4_server_output_update(-1, &request, &stream, raw, prefix, false, &stop));
    }
    c.ds4_server_output_apply_ids(&calls, &stream);
    try expect(c.ds4_server_output_finish(-1, &request, &stream, raw, raw.len, null, &calls, "tool_calls", 10, 25, 1234, &stop));
    var arguments: std.ArrayList(u8) = .empty;
    defer arguments.deinit(a);
    var item_id: ?[]u8 = null;
    defer if (item_id) |value| a.free(value);
    var call_id: ?[]u8 = null;
    defer if (call_id) |value| a.free(value);
    var deltas: usize = 0;
    var complete = false;
    for (capture.events.items) |event| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, event, .{});
        defer parsed.deinit();
        const kind = try stringField(parsed.value, "type");
        if (std.mem.eql(u8, kind, "response.output_item.added")) {
            const item = try field(parsed.value, "item");
            if (std.mem.eql(u8, try stringField(item, "type"), "function_call")) {
                item_id = try a.dupe(u8, try stringField(item, "id"));
                call_id = try a.dupe(u8, try stringField(item, "call_id"));
                try equal("echo", try stringField(item, "name"));
                try equal("", try stringField(item, "arguments"));
            }
        } else if (std.mem.eql(u8, kind, "response.function_call_arguments.delta")) {
            try equal(item_id.?, try stringField(parsed.value, "item_id"));
            try arguments.appendSlice(a, try stringField(parsed.value, "delta"));
            deltas += 1;
        } else if (std.mem.eql(u8, kind, "response.function_call_arguments.done")) {
            try equal(arguments.items, try stringField(parsed.value, "arguments"));
        } else if (std.mem.eql(u8, kind, "response.completed")) {
            const response = try field(parsed.value, "response");
            const output = try field(response, "output");
            const item = output.array.items[output.array.items.len - 1];
            try equal(item_id.?, try stringField(item, "id"));
            try equal(call_id.?, try stringField(item, "call_id"));
            try equal(arguments.items, try stringField(item, "arguments"));
            complete = true;
        }
    }
    try expect(complete and deltas > 2);
    try equal(expected, arguments.items);

    var memory = std.mem.zeroes(c.tool_memory);
    defer c.ds4_server_tool_memory_free(&memory);
    var restored = std.mem.zeroes(c.tool_memory);
    defer c.ds4_server_tool_memory_free(&restored);
    var mutex: c.pthread_mutex_t = undefined;
    try expect(c.pthread_mutex_init(&mutex, null) == 0);
    defer _ = c.pthread_mutex_destroy(&mutex);
    c.ds4_server_tool_memory_remember(&memory, &mutex, &calls);
    const path = ".workspaces/interaction-catchup-20261027/output-token-replay.ktm";
    const file = c.fopen(path, "w+b") orelse return error.OpenFile;
    defer {
        _ = c.fclose(file);
        _ = c.unlink(path);
    }
    var written: u64 = 0;
    try expect(c.ds4_server_tool_memory_write(&memory, &mutex, file, raw, &written));
    try expect(written > 0);
    c.rewind(file);
    try expect(c.ds4_server_tool_memory_load(&restored, &mutex, file, null) == 1);
    var message = std.mem.zeroes(c.chat_msg);
    var borrowed_call = c.tool_call{ .id = calls.v[0].id, .name = calls.v[0].name, .arguments = calls.v[0].arguments, .incomplete = false };
    message.calls.v = &borrowed_call;
    message.calls.len = 1;
    var messages = c.chat_msgs{ .v = &message, .len = 1, .cap = 1 };
    c.ds4_server_tool_memory_attach(&restored, &mutex, &messages, null);
    defer {
        c.free(message.calls.raw_tool_text);
        c.free(message.calls.raw_special_ranges);
    }
    try equal(text(calls.raw_tool_text), text(message.calls.raw_tool_text));
    try expect(calls.raw_special_count == message.calls.raw_special_count);
    for (0..calls.raw_special_count) |index| {
        try expect(calls.raw_special_ranges[index].begin == message.calls.raw_special_ranges[index].begin);
        try expect(calls.raw_special_ranges[index].end == message.calls.raw_special_ranges[index].end);
    }
    try flintStream();
    try flintSchemaCoercion();
    try truncationCases();
    std.debug.print("DS41/Flint output: literal control spellings, incremental Responses arguments/IDs and KTM2 token-range restart pass.\n", .{});
}

fn timestamp() f64 {
    var stamp: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &stamp);
    return @as(f64, @floatFromInt(stamp.tv_sec)) + @as(f64, @floatFromInt(stamp.tv_nsec)) * 1e-9;
}
fn measure(raw: [:0]u8, ranges: []const Projection.Range, thinking: bool, flint: bool, mode: u2) !f64 {
    var tracker: c.dsml_decode_tracker = undefined;
    c.ds4_server_decode_init(&tracker);
    tracker.model_syntax = if (flint) c.SERVER_MODEL_SYNTAX_FLINT else c.SERVER_MODEL_SYNTAX_DEEPSEEK41;
    var state = std.mem.zeroes(c.thinking_state);
    state.inside = thinking;
    var recovery_cursor: usize = 0;
    var previous: usize = 0;
    var checksum: usize = 0;
    var key: [*c]u8 = @constCast("text");
    var schema: [*c]u8 = @constCast("{\"type\":\"string\"}");
    var order = std.mem.zeroes(c.tool_schema_order);
    order.name = @constCast("echo");
    order.prop = &key;
    order.prop_schema = &schema;
    order.len = 1;
    const orders = c.tool_schema_orders{ .v = &order, .len = 1, .cap = 1 };
    const started = timestamp();
    while (previous < raw.len) {
        var end = @min(raw.len, previous + 5);
        for (ranges) |range| if (end > range.begin and end < range.end) {
            end = range.end;
        };
        const saved = raw[end];
        raw[end] = 0;
        if (mode != 0) {
            const view = Projection.View{ .raw = raw[0..end], .ranges = ranges };
            var projected = if (flint) (if (mode == 2) try FlintProjection.projectState(a, view, &orders, false) else try FlintProjection.project(a, view, &orders, false)) else (if (mode == 2) try Projection.projectState(a, view, thinking, false) else try Projection.project(a, view, thinking, false));
            checksum +%= (projected.tool_start orelse 0) + (projected.tool_end orelse 0) + @intFromBool(projected.in_payload) + @intFromBool(projected.invalid);
            checksum +%= projected.reasoning.len + projected.content.len;
            for (projected.calls.items) |call| checksum +%= call.arguments.len;
            projected.deinit(a);
        } else {
            if (!c.ds4_server_decode_is_tool(tracker.decode)) c.ds4_server_generation_thinking_feed(&state, raw.ptr + previous, end - previous);
            if (state.inside) {
                _ = c.ds4_server_generation_complete_thinking(tracker.model_syntax, raw.ptr, end, &recovery_cursor);
            } else c.ds4_server_decode_update(&tracker, raw.ptr, end);
            checksum +%= end + @as(usize, @intCast(tracker.decode));
        }
        raw[end] = saved;
        previous = end;
    }
    const elapsed = timestamp() - started;
    std.mem.doNotOptimizeAway(checksum);
    return elapsed * 1000;
}
fn bench() !void {
    for ([_]bool{ false, true }) |flint| for ([_]bool{ true, false }) |thinking| for ([_]usize{ 2048, 8192 }) |tokens| {
        var buffer: std.ArrayList(u8) = .empty;
        defer buffer.deinit(a);
        if (thinking and flint) try buffer.appendSlice(a, "<think>");
        if (!thinking) try buffer.appendSlice(a, if (flint) "<tool_call>echo<arg_key>text</arg_key><arg_value>" else "<｜DSML｜ calls><｜DSML｜ invoke name=\"echo\"><｜DSML｜ parameter name=\"text\" string=\"true\">");
        for (0..tokens) |_| try buffer.appendSlice(a, "word ");
        const body_end = buffer.items.len;
        if (thinking) try buffer.appendSlice(a, "</think>OK") else try buffer.appendSlice(a, if (flint) "</arg_value></tool_call>" else "</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>");
        const raw = try buffer.toOwnedSliceSentinel(a, 0);
        defer a.free(raw);
        var ranges: std.ArrayList(Projection.Range) = .empty;
        defer ranges.deinit(a);
        _ = body_end;
        const markers = [_][]const u8{ "<think>", "</think>", "｜DSML｜", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>" };
        var cursor: usize = 0;
        while (cursor < raw.len) {
            var matched = false;
            for (markers) |marker| if (std.mem.startsWith(u8, raw[cursor..], marker)) {
                try ranges.append(a, .{ .begin = cursor, .end = cursor + marker.len });
                cursor += marker.len;
                matched = true;
                break;
            };
            if (!matched) cursor += 1;
        }
        // Verify the old byte parser and typed projection agree on this plain
        // fixture before timing them. Literal-control behavior is tested above.
        var old_content: [*c]u8 = null;
        var old_reasoning: [*c]u8 = null;
        var old_calls = std.mem.zeroes(c.tool_calls);
        defer {
            c.free(old_content);
            c.free(old_reasoning);
            freeCalls(&old_calls);
        }
        try expect(c.ds4_server_reply_parse(if (flint) c.SERVER_MODEL_SYNTAX_FLINT else c.SERVER_MODEL_SYNTAX_DEEPSEEK41, raw.ptr, thinking, &old_content, &old_reasoning, &old_calls));
        const view = Projection.View{ .raw = raw, .ranges = ranges.items };
        var typed = if (flint) try FlintProjection.project(a, view, null, true) else try Projection.project(a, view, thinking, true);
        defer typed.deinit(a);
        try equal(text(old_content), typed.content);
        try equal(text(old_reasoning), typed.reasoning);
        try expect(@as(usize, @intCast(old_calls.len)) == typed.calls.items.len);
        if (!thinking) {
            const old_args = try std.json.parseFromSlice(std.json.Value, a, text(old_calls.v[0].arguments), .{});
            defer old_args.deinit();
            const new_args = try std.json.parseFromSlice(std.json.Value, a, typed.calls.items[0].arguments, .{});
            defer new_args.deinit();
            try equal(try stringField(old_args.value, "text"), try stringField(new_args.value, "text"));
        }
        _ = try measure(raw, ranges.items, thinking, flint, 0);
        _ = try measure(raw, ranges.items, thinking, flint, 1);
        var old_samples: [5]f64 = undefined;
        var new_samples: [5]f64 = undefined;
        var state_samples: [5]f64 = undefined;
        _ = try measure(raw, ranges.items, thinking, flint, 2);
        for (0..5) |iteration| {
            state_samples[iteration] = try measure(raw, ranges.items, thinking, flint, 2);
            if (iteration % 2 == 0) {
                old_samples[iteration] = try measure(raw, ranges.items, thinking, flint, 0);
                new_samples[iteration] = try measure(raw, ranges.items, thinking, flint, 1);
            } else {
                new_samples[iteration] = try measure(raw, ranges.items, thinking, flint, 1);
                old_samples[iteration] = try measure(raw, ranges.items, thinking, flint, 0);
            }
        }
        std.mem.sort(f64, &old_samples, {}, std.sort.asc(f64));
        std.mem.sort(f64, &new_samples, {}, std.sort.asc(f64));
        std.mem.sort(f64, &state_samples, {}, std.sort.asc(f64));
        std.debug.print("{{\"model\":\"{s}\",\"case\":\"{s}\",\"five_byte_tokens\":{d},\"bytes\":{d},\"legacy_median_ms\":{d:.4},\"typed_median_ms\":{d:.4},\"state_median_ms\":{d:.4},\"typed_us_per_token\":{d:.4},\"state_us_per_token\":{d:.4}}}\n", .{ if (flint) "flint" else "ds41", if (thinking) "ordinary_reasoning" else "string_tool_payload", tokens, raw.len, old_samples[2], new_samples[2], state_samples[2], new_samples[2] * 1000 / @as(f64, @floatFromInt(tokens)), state_samples[2] * 1000 / @as(f64, @floatFromInt(tokens)) });
    };
}
export fn main(argc: c_int, argv: [*c][*c]const u8) c_int {
    if (argc == 2 and std.mem.eql(u8, text(argv[1]), "--bench")) {
        bench() catch |err| {
            std.debug.print("Output projection benchmark: {s}\n", .{@errorName(err)});
            return 1;
        };
        return 0;
    }
    run() catch |err| {
        std.debug.print("DS41 output: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}
