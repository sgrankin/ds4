//! Production-linked receipts, mixed-call replay and stream identity checks.
const std = @import("std");
const c = @import("hosted_test_c");
const a = std.testing.allocator;

fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) "" else std.mem.span(pointer);
}
fn receipt(status: []const u8, call_id: []const u8, call_ids: []const u8) ![:0]u8 {
    return std.fmt.allocPrintSentinel(a, "{{\"type\":\"ds4:decision_call\",\"id\":\"dc_{s}\",\"status\":\"{s}\",\"call_id\":\"{s}\",\"call_ids\":{s},\"model\":\"clef-flash\",\"arguments\":\"{{\\\"questions\\\":[]}}\",\"input\":\"original state\",\"result\":{{\"model\":\"clef-flash\",\"answers\":[]}}}}", .{ call_id, status, call_id, call_ids }, 0);
}

test "decision receipts preserve original arguments, typed input and result" {
    const arguments = "{ \"questions\": [], \"note\": \"line\\nvalue\" }";
    const call = c.tool_call{ .id = @constCast("call_decision"), .name = @constCast("ds4_decide"), .arguments = @constCast(arguments) };
    var batch = [_]c.tool_call{call};
    const group = c.tool_calls{ .v = &batch, .len = 1, .cap = 1, .raw_tool_text = null };
    var identity = std.mem.zeroes(c.responses_tool_item);
    @memcpy(identity.fc_id[0..11], "dc_original");
    @memcpy(identity.call_id[0..13], "call_decision");
    for ([_][*:0]const u8{ "completed", "failed" }) |status| {
        var out = std.mem.zeroes(c.buf);
        defer c.free(out.ptr);
        c.ds4_server_output_append_decision_item(&out, &call, &group, &identity, status, "clef-flash", "[{\"role\":\"user\",\"content\":\"state\"}]", "{\"answers\":[],\"usage\":{\"input_tokens\":123}}");
        const parsed = try std.json.parseFromSlice(std.json.Value, a, out.ptr[0..out.len], .{});
        defer parsed.deinit();
        const object = parsed.value.object;
        try std.testing.expectEqualStrings("ds4:decision_call", object.get("type").?.string);
        try std.testing.expectEqualStrings(text(status), object.get("status").?.string);
        try std.testing.expectEqualStrings("dc_original", object.get("id").?.string);
        try std.testing.expectEqualStrings(arguments, object.get("arguments").?.string);
        try std.testing.expectEqualStrings("call_decision", object.get("call_ids").?.array.items[0].string);
        try std.testing.expect(object.get("input").? == .array);
        try std.testing.expectEqual(@as(i64, 123), object.get("result").?.object.get("usage").?.object.get("input_tokens").?.integer);
    }
}

test "mixed hosted receipts and client calls restore one parallel call group" {
    const external = "{\"type\":\"function_call\",\"id\":\"fc_client\",\"status\":\"completed\",\"call_id\":\"call_client\",\"name\":\"lookup_record\",\"arguments\":\"{}\"}";
    const external_result = "{\"type\":\"function_call_output\",\"call_id\":\"call_client\",\"output\":\"client result\"}";
    for ([_][]const u8{ "completed", "failed" }) |status| {
        for ([_]bool{ false, true }) |client_first| {
            const hosted = try receipt(status, "call_decision", if (client_first) "[\"call_client\",\"call_decision\"]" else "[\"call_decision\",\"call_client\"]");
            defer a.free(hosted);
            const raw = try std.fmt.allocPrintSentinel(a, "[{s},{s},{s}]", .{ if (client_first) external else hosted, if (client_first) hosted else external, external_result }, 0);
            defer a.free(raw);
            var cursor: [*c]const u8 = raw.ptr;
            var messages = std.mem.zeroes(c.chat_msgs);
            defer c.ds4_server_input_messages_free(&messages);
            try std.testing.expect(c.ds4_server_json_responses_input(&cursor, &messages, null, null, false));
            try std.testing.expectEqual(@as(c_int, 3), messages.len);
            try std.testing.expectEqualStrings("assistant", text(messages.v[0].role));
            try std.testing.expectEqual(@as(c_int, 2), messages.v[0].calls.len);
            try std.testing.expectEqualStrings(if (client_first) "lookup_record" else "ds4_decide", text(messages.v[0].calls.v[0].name));
            try std.testing.expectEqualStrings(if (client_first) "ds4_decide" else "lookup_record", text(messages.v[0].calls.v[1].name));
            const hosted_call = messages.v[0].calls.v[if (client_first) @as(usize, 1) else 0];
            try std.testing.expectEqualStrings("{\"questions\":[]}", text(hosted_call.arguments));
            try std.testing.expectEqualStrings("tool", text(messages.v[1].role));
            try std.testing.expectEqualStrings("call_decision", text(messages.v[1].tool_call_id));
            try std.testing.expectEqualStrings("call_client", text(messages.v[2].tool_call_id));
        }
    }
}

test "adjacent hosted receipts retain sequential rounds without visible text" {
    const first = try receipt("completed", "call_one", "[\"call_one\"]");
    defer a.free(first);
    for ([_][]const u8{ "completed", "failed" }) |status| {
        const second = try receipt(status, "call_two", "[\"call_two\"]");
        defer a.free(second);
        const raw = try std.fmt.allocPrintSentinel(a, "[{s},{s}]", .{ first, second }, 0);
        defer a.free(raw);
        var cursor: [*c]const u8 = raw.ptr;
        var messages = std.mem.zeroes(c.chat_msgs);
        defer c.ds4_server_input_messages_free(&messages);
        try std.testing.expect(c.ds4_server_json_responses_input(&cursor, &messages, null, null, false));
        try std.testing.expectEqual(@as(c_int, 4), messages.len);
        for ([_][]const u8{ "call_one", "call_two" }, 0..) |id, round| {
            try std.testing.expectEqualStrings("assistant", text(messages.v[round * 2].role));
            try std.testing.expectEqual(@as(c_int, 1), messages.v[round * 2].calls.len);
            try std.testing.expectEqualStrings(id, text(messages.v[round * 2].calls.v[0].id));
            try std.testing.expectEqualStrings("tool", text(messages.v[round * 2 + 1].role));
            try std.testing.expectEqualStrings(id, text(messages.v[round * 2 + 1].tool_call_id));
        }
    }
}

test "adjacent hosted receipts from one batch retain their parallel group" {
    const first = try receipt("completed", "call_one", "[\"call_one\",\"call_two\"]");
    defer a.free(first);
    const second = try receipt("failed", "call_two", "[\"call_one\",\"call_two\"]");
    defer a.free(second);
    const raw = try std.fmt.allocPrintSentinel(a, "[{s},{s}]", .{ first, second }, 0);
    defer a.free(raw);
    var cursor: [*c]const u8 = raw.ptr;
    var messages = std.mem.zeroes(c.chat_msgs);
    defer c.ds4_server_input_messages_free(&messages);
    try std.testing.expect(c.ds4_server_json_responses_input(&cursor, &messages, null, null, false));
    try std.testing.expectEqual(@as(c_int, 3), messages.len);
    try std.testing.expectEqual(@as(c_int, 2), messages.v[0].calls.len);
    try std.testing.expectEqualStrings("call_one", text(messages.v[1].tool_call_id));
    try std.testing.expectEqualStrings("call_two", text(messages.v[2].tool_call_id));
}

test "hosted receipt followed by a later client round does not merge assistant batches" {
    const hosted = try receipt("completed", "call_decision", "[\"call_decision\"]");
    defer a.free(hosted);
    const raw = try std.fmt.allocPrintSentinel(a, "[{s},{{\"type\":\"function_call\",\"status\":\"completed\",\"call_id\":\"call_client\",\"name\":\"lookup_record\",\"arguments\":\"{{}}\"}},{{\"type\":\"function_call_output\",\"call_id\":\"call_client\",\"output\":\"client result\"}}]", .{hosted}, 0);
    defer a.free(raw);
    var cursor: [*c]const u8 = raw.ptr;
    var messages = std.mem.zeroes(c.chat_msgs);
    defer c.ds4_server_input_messages_free(&messages);
    try std.testing.expect(c.ds4_server_json_responses_input(&cursor, &messages, null, null, false));
    try std.testing.expectEqual(@as(c_int, 4), messages.len);
    try std.testing.expectEqualStrings("call_decision", text(messages.v[0].calls.v[0].id));
    try std.testing.expectEqualStrings("call_decision", text(messages.v[1].tool_call_id));
    try std.testing.expectEqualStrings("call_client", text(messages.v[2].calls.v[0].id));
    try std.testing.expectEqualStrings("call_client", text(messages.v[3].tool_call_id));
}

test "partial hosted receipt is rejected without accepting failed ordinary calls" {
    const partial = try receipt("in_progress", "call_decision", "[\"call_decision\"]");
    defer a.free(partial);
    const raw = try std.fmt.allocPrintSentinel(a, "[{s}]", .{partial}, 0);
    defer a.free(raw);
    var cursor: [*c]const u8 = raw.ptr;
    var messages = std.mem.zeroes(c.chat_msgs);
    defer c.ds4_server_input_messages_free(&messages);
    try std.testing.expect(!c.ds4_server_json_responses_input(&cursor, &messages, null, null, false));
    cursor = "[{\"type\":\"function_call\",\"status\":\"failed\",\"call_id\":\"x\",\"name\":\"lookup_record\",\"arguments\":\"{}\"}]";
    try std.testing.expect(!c.ds4_server_json_responses_input(&cursor, &messages, null, null, false));
}

test "SDK message-shaped receipt dumps replay their null optional fields" {
    const hosted = try receipt("completed", "call_decision", "[\"call_decision\"]");
    defer a.free(hosted);
    const raw = try std.fmt.allocPrintSentinel(a, "[{{\"role\":null,\"content\":null,\"phase\":null,{s}]", .{hosted[1..]}, 0);
    defer a.free(raw);
    var cursor: [*c]const u8 = raw.ptr;
    var messages = std.mem.zeroes(c.chat_msgs);
    defer c.ds4_server_input_messages_free(&messages);
    try std.testing.expect(c.ds4_server_json_responses_input(&cursor, &messages, null, null, false));
    try std.testing.expectEqual(@as(c_int, 2), messages.len);
    try std.testing.expectEqualStrings("ds4_decide", text(messages.v[0].calls.v[0].name));
    try std.testing.expectEqualStrings("call_decision", text(messages.v[1].tool_call_id));
}

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
        for (self.events.items) |value| a.free(value);
        self.events.deinit(a);
    }
};

test "hosted rounds keep one stream identity and mixed call source order" {
    for ([_]bool{ false, true }) |client_first| {
        var capture = Capture{};
        defer capture.deinit();
        const sink = c.server_response_sink{ .context = &capture, .emit = Capture.emit, .load_parent = null };
        var request = std.mem.zeroes(c.request);
        request.api = c.API_RESPONSES;
        request.model_syntax = c.SERVER_MODEL_SYNTAX_FLINT;
        request.response_sink = &sink;
        request.stream = true;
        request.model = @constCast("fixture");
        var stream = std.mem.zeroes(c.responses_stream);
        c.ds4_server_output_init(&request, &stream);
        stream.active = true;
        defer c.ds4_server_output_free(&stream);
        defer c.free(request.responses_output_prefix);
        const response_id = stream.response_id;
        var stop: c.sig_atomic_t = 0;
        try std.testing.expect(c.ds4_server_output_created(-1, &request, &stream, 1, &stop));
        try std.testing.expect(c.ds4_server_output_update(-1, &request, &stream, "Checking.", 9, false, &stop));
        try std.testing.expect(c.ds4_server_output_round_close(-1, &request, &stream, "Checking.", 9, "Checking.", null, false, "tool_calls", &stop));
        try std.testing.expectEqual(@as(c_int, 1), request.responses_output_prefix_count);
        const hosted = c.tool_call{ .id = @constCast("call_decision"), .name = @constCast("ds4_decide"), .arguments = @constCast("{\"questions\":[]}") };
        const external = c.tool_call{ .id = @constCast("call_client"), .name = @constCast("lookup_record"), .arguments = @constCast("{}") };
        var batch = [_]c.tool_call{ if (client_first) external else hosted, if (client_first) hosted else external };
        const group = c.tool_calls{ .v = &batch, .len = 2, .cap = 2, .raw_tool_text = null };
        var hosted_item = std.mem.zeroes(c.responses_tool_item);
        var external_item = std.mem.zeroes(c.responses_tool_item);
        if (client_first) try std.testing.expect(c.ds4_server_output_external_call(-1, &request, &stream, &external, &external_item, "tool_calls", &stop));
        try std.testing.expect(c.ds4_server_output_hosted_call(-1, &request, &stream, &hosted, &group, &hosted_item, "in_progress", "clef-flash", null, null, &stop));
        const stable_id = hosted_item.fc_id;
        const stable_index = hosted_item.output_index;
        try std.testing.expect(c.ds4_server_output_hosted_call(-1, &request, &stream, &hosted, &group, &hosted_item, "completed", "clef-flash", "\"state\"", "{\"answers\":[]}", &stop));
        if (!client_first) try std.testing.expect(c.ds4_server_output_external_call(-1, &request, &stream, &external, &external_item, "tool_calls", &stop));
        try std.testing.expectEqualSlices(u8, &stable_id, &hosted_item.fc_id);
        try std.testing.expectEqual(stable_index, hosted_item.output_index);
        try std.testing.expectEqualSlices(u8, &response_id, &stream.response_id);
        try std.testing.expectEqual(@as(c_int, 3), request.responses_output_prefix_count);
        const prefix = try std.fmt.allocPrint(a, "[{s}]", .{text(request.responses_output_prefix)});
        defer a.free(prefix);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, prefix, .{});
        defer parsed.deinit();
        const items = parsed.value.array.items;
        try std.testing.expectEqualStrings("message", items[0].object.get("type").?.string);
        try std.testing.expectEqualStrings(if (client_first) "function_call" else "ds4:decision_call", items[1].object.get("type").?.string);
        try std.testing.expectEqualStrings(if (client_first) "ds4:decision_call" else "function_call", items[2].object.get("type").?.string);
        const call_ids = items[if (client_first) @as(usize, 2) else 1].object.get("call_ids").?.array.items;
        try std.testing.expectEqual(@as(usize, 2), call_ids.len);
        try std.testing.expectEqualStrings(if (client_first) "call_client" else "call_decision", call_ids[0].string);
        try std.testing.expectEqualStrings(if (client_first) "call_decision" else "call_client", call_ids[1].string);
        for (capture.events.items, 0..) |raw_event, index| {
            const event = try std.json.parseFromSlice(std.json.Value, a, raw_event, .{});
            defer event.deinit();
            const object = event.value.object;
            try std.testing.expectEqual(@as(i64, @intCast(index)), object.get("sequence_number").?.integer);
            const kind = object.get("type").?.string;
            try std.testing.expect(!std.mem.eql(u8, kind, "response.completed"));
            try std.testing.expect(!std.mem.eql(u8, kind, "response.failed"));
        }
    }
}

test "two hosted rounds retain all text and receipt items in final output" {
    var capture = Capture{};
    defer capture.deinit();
    const sink = c.server_response_sink{ .context = &capture, .emit = Capture.emit, .load_parent = null };
    var request = std.mem.zeroes(c.request);
    request.api = c.API_RESPONSES;
    request.model_syntax = c.SERVER_MODEL_SYNTAX_FLINT;
    request.response_sink = &sink;
    request.stream = true;
    request.model = @constCast("fixture");
    var stream = std.mem.zeroes(c.responses_stream);
    c.ds4_server_output_init(&request, &stream);
    stream.active = true;
    defer c.ds4_server_output_free(&stream);
    defer c.free(request.responses_output_prefix);
    const response_id = stream.response_id;
    var stop: c.sig_atomic_t = 0;
    try std.testing.expect(c.ds4_server_output_created(-1, &request, &stream, 1, &stop));
    for ([_][*:0]const u8{ "Checking.", "Rechecking." }, 0..) |content, round| {
        const length = text(content).len;
        try std.testing.expect(c.ds4_server_output_round_close(-1, &request, &stream, content, length, content, null, false, "tool_calls", &stop));
        const call = c.tool_call{ .id = @constCast(if (round == 0) "call_one" else "call_two"), .name = @constCast("ds4_decide"), .arguments = @constCast("{\"questions\":[]}") };
        var batch = [_]c.tool_call{call};
        const group = c.tool_calls{ .v = &batch, .len = 1, .cap = 1, .raw_tool_text = null };
        var item = std.mem.zeroes(c.responses_tool_item);
        try std.testing.expect(c.ds4_server_output_hosted_call(-1, &request, &stream, &call, &group, &item, "in_progress", "clef-flash", null, null, &stop));
        try std.testing.expectEqual(@as(c_int, @intCast(round * 2 + 1)), item.output_index);
        try std.testing.expect(c.ds4_server_output_hosted_call(-1, &request, &stream, &call, &group, &item, "completed", "clef-flash", "\"state\"", "{\"answers\":[]}", &stop));
        c.ds4_server_output_round_reset(&request, &stream);
    }
    try std.testing.expect(c.ds4_server_output_finish(-1, &request, &stream, "Done.", 5, null, null, "stop", 42, 12, 1, &stop));
    const terminal = try std.json.parseFromSlice(std.json.Value, a, capture.events.items[capture.events.items.len - 1], .{});
    defer terminal.deinit();
    try std.testing.expectEqualStrings("response.completed", terminal.value.object.get("type").?.string);
    const response = terminal.value.object.get("response").?.object;
    try std.testing.expectEqualStrings(text(&response_id), response.get("id").?.string);
    const items = response.get("output").?.array.items;
    try std.testing.expectEqual(@as(usize, 5), items.len);
    for ([_][]const u8{ "Checking.", "Rechecking.", "Done." }, 0..) |expected, index| {
        try std.testing.expectEqualStrings(expected, items[index * 2].object.get("content").?.array.items[0].object.get("text").?.string);
    }
    var next_added: i64 = 0;
    var next_done: i64 = 0;
    for (capture.events.items, 0..) |raw_event, index| {
        const event = try std.json.parseFromSlice(std.json.Value, a, raw_event, .{});
        defer event.deinit();
        const object = event.value.object;
        try std.testing.expectEqual(@as(i64, @intCast(index)), object.get("sequence_number").?.integer);
        const kind = object.get("type").?.string;
        if (std.mem.eql(u8, kind, "response.output_item.added") or std.mem.eql(u8, kind, "response.output_item.done")) {
            const output_index = object.get("output_index").?.integer;
            const counter = if (std.mem.eql(u8, kind, "response.output_item.added")) &next_added else &next_done;
            try std.testing.expectEqual(counter.*, output_index);
            counter.* += 1;
            try std.testing.expectEqualStrings(items[@intCast(output_index)].object.get("id").?.string, object.get("item").?.object.get("id").?.string);
        }
        if (index + 1 < capture.events.items.len) try std.testing.expect(!std.mem.eql(u8, kind, "response.completed"));
    }
    try std.testing.expectEqual(@as(i64, 5), next_added);
    try std.testing.expectEqual(@as(i64, 5), next_done);
}
