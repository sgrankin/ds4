//! Model-free ownership and replay checks exercise the production wire codecs.
const std = @import("std");
const c = @import("hosted_test_c");

comptime {
    _ = @import("server/cache_controls.zig");
    _ = @import("server/prompt_cache_policy.zig");
}

fn decode(endpoint: c.ds4_server_input_endpoint, body: [:0]const u8) !c.ds4_server_input {
    const defaults: c.ds4_server_input_defaults = .{
        .max_tokens = 128,
        .ctx_size = 8192,
        .syntax = c.SERVER_MODEL_SYNTAX_DEEPSEEK41,
        .responses_store_dir = null,
        .incremental_tool_schemas = false,
    };
    var input = std.mem.zeroes(c.ds4_server_input);
    var errors: [512]u8 = @splat(0);
    if (!c.ds4_server_input_decode(endpoint, body, &defaults, &input, &errors, errors.len)) return error.InvalidRequest;
    return input;
}
fn offsets(message: [*c]const c.chat_msg) []const usize {
    const value = message[0];
    return if (value.cache_breakpoints_len == 0) &.{} else value.cache_breakpoints[0..@as(usize, @intCast(value.cache_breakpoints_len))];
}

const content =
    \\"content":[{"type":"input_text","text":"雪","prompt_cache_breakpoint":{"mode":"explicit"}},{"type":"input_text","text":"same"},{"type":"input_text","text":"same","prompt_cache_breakpoint":{"mode":"explicit"}}]
;

test "Responses cache markers retain byte endpoints without entering model text" {
    var input = try decode(c.DS4_SERVER_INPUT_RESPONSES, "{\"input\":[{" ++ content ++ ",\"role\":\"developer\"}],\"prompt_cache_options\":{\"mode\":\"explicit\"}}");
    defer c.ds4_server_input_deinit(&input);
    try std.testing.expect(input.req.prompt_cache_explicit);
    try std.testing.expect(input.req.prompt_cache_controls_set);
    try std.testing.expectEqualStrings("雪samesame", std.mem.span(input.messages.v[0].content));
    try std.testing.expectEqualSlices(usize, &.{ 3, 11 }, offsets(&input.messages.v[0]));
    try std.testing.expectEqual(@as(c_int, 0), input.req.cache_breakpoints.len);
}

test "Chat cache markers use parsed block boundaries and duplicate content replaces offsets" {
    var input = try decode(c.DS4_SERVER_INPUT_CHAT, "{\"messages\":[{" ++ content ++ ",\"role\":\"user\"}]}");
    defer c.ds4_server_input_deinit(&input);
    try std.testing.expectEqualSlices(usize, &.{ 3, 11 }, offsets(&input.messages.v[0]));
    var replaced = try decode(c.DS4_SERVER_INPUT_CHAT, "{\"messages\":[{" ++ content ++ ",\"content\":\"replacement\"}]}");
    defer c.ds4_server_input_deinit(&replaced);
    try std.testing.expectEqualStrings("replacement", std.mem.span(replaced.messages.v[0].content));
    try std.testing.expectEqual(@as(usize, 0), offsets(&replaced.messages.v[0]).len);
}

test "private replay preserves marked blocks and exact literal data" {
    var input = try decode(c.DS4_SERVER_INPUT_RESPONSES, "{\"input\":[{" ++ content ++ ",\"role\":\"user\"}]}");
    defer c.ds4_server_input_deinit(&input);
    const replay = c.ds4_server_render_replay_history(&input.messages, 0);
    defer c.free(replay);
    var restored = std.mem.zeroes(c.ds4_server_input);
    defer c.ds4_server_input_deinit(&restored);
    var cursor: [*c]const u8 = replay;
    try std.testing.expect(c.ds4_server_json_messages(&cursor, &restored.messages, true));
    try std.testing.expectEqualStrings("雪samesame", std.mem.span(restored.messages.v[0].content));
    try std.testing.expectEqualSlices(usize, &.{ 3, 11 }, offsets(&restored.messages.v[0]));
}

test "historical marker count is independent of the per-request cache write limit" {
    const part = "{\"type\":\"input_text\",\"text\":\"x\",\"prompt_cache_breakpoint\":{\"mode\":\"explicit\"}}";
    var input = try decode(c.DS4_SERVER_INPUT_RESPONSES, "{\"input\":[{\"role\":\"user\",\"content\":[" ++ part ++ "," ++ part ++ "," ++ part ++ "," ++ part ++ "," ++ part ++ "]}]}");
    defer c.ds4_server_input_deinit(&input);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3, 4, 5 }, offsets(&input.messages.v[0]));
}

test "cache mode can explicitly disable prompt caching without marked content" {
    var input = try decode(c.DS4_SERVER_INPUT_RESPONSES, "{\"input\":\"hello\",\"prompt_cache_options\":{\"mode\":\"explicit\"}}");
    defer c.ds4_server_input_deinit(&input);
    try std.testing.expect(input.req.prompt_cache_explicit);
    var implicit = try decode(c.DS4_SERVER_INPUT_CHAT, "{\"messages\":[],\"prompt_cache_options\":null}");
    defer c.ds4_server_input_deinit(&implicit);
    try std.testing.expect(!implicit.req.prompt_cache_explicit);
}

test "public JSON validates the complete envelope with the standard parser" {
    var ordinary = try decode(c.DS4_SERVER_INPUT_RESPONSES, "{\"input\":\"hello\"}");
    defer c.ds4_server_input_deinit(&ordinary);
    try std.testing.expectError(error.InvalidRequest, decode(c.DS4_SERVER_INPUT_RESPONSES, "{\"input\":\"hello\"} trailing"));
}

test "C-string payloads reject decoded NUL instead of silently truncating" {
    var literal = try decode(c.DS4_SERVER_INPUT_RESPONSES, "{\"input\":\"literal \\\\u0000\"}");
    defer c.ds4_server_input_deinit(&literal);
    try std.testing.expectError(error.InvalidRequest, decode(c.DS4_SERVER_INPUT_RESPONSES, "{\"input\":\"before\\u0000after\"}"));
    try std.testing.expectError(error.InvalidRequest, decode(c.DS4_SERVER_INPUT_RESPONSES, "{\"input\":[{\"type\":\"function_call\",\"call_id\":\"one\",\"name\":\"test\",\"arguments\":\"{\\\"text\\\":\\\"before\\\\u0000after\\\"}\"}]}"));
}

test "unsupported cache controls and marked modalities reject rather than disappear" {
    try std.testing.expectError(error.InvalidRequest, decode(c.DS4_SERVER_INPUT_RESPONSES, "{\"input\":\"x\",\"prompt_cache_options\":{\"ttl\":\"12h\"}}"));
    try std.testing.expectError(error.InvalidRequest, decode(c.DS4_SERVER_INPUT_CHAT, "{\"messages\":[{\"content\":[{\"type\":\"text\",\"text\":\"x\",\"prompt_cache_breakpoint\":{\"mode\":\"implicit\"}}]}]}"));
    try std.testing.expectError(error.InvalidRequest, decode(c.DS4_SERVER_INPUT_RESPONSES, "{\"input\":[{\"content\":[{\"type\":\"input_image\",\"prompt_cache_breakpoint\":{\"mode\":\"explicit\"}}]}]}"));
}

test "custom stops cannot truncate a promised structured answer" {
    var input = try decode(c.DS4_SERVER_INPUT_CHAT, "{\"messages\":[{\"role\":\"user\",\"content\":\"answer\"}],\"response_format\":{\"type\":\"json_object\"},\"stop\":[\"x\"]}");
    defer c.ds4_server_input_deinit(&input);
    var request = std.mem.zeroes(c.request);
    defer c.ds4_server_input_request_free(&request);
    var errors: [512]u8 = @splat(0);
    try std.testing.expect(!c.ds4_server_materialize(null, null, c.DS4_SERVER_INPUT_CHAT, &input, 8192, false, &request, &errors, errors.len));
    try std.testing.expect(std.mem.find(u8, std.mem.sliceTo(&errors, 0), "Custom stop strings cannot combine") != null);
}
