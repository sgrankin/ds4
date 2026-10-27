//! CPU-only production BPE check: framing controls, literal payload and replay.
const std = @import("std");
const c = @import("flint_prompt_test_c");
const types = @import("serving_types");
const model = @import("ds4_model");
const allocator = std.heap.c_allocator;
const expect = std.testing.expect;
const equal = std.testing.expectEqualStrings;

fn encodeAndCheck(tokenizer: *c.flint_tokenizer, rendered: [:0]const u8, map: *const types.PromptMap) !void {
    const ranges = try allocator.alloc(c.ds4_prompt_range, map.special_ranges.items.len);
    defer allocator.free(ranges);
    for (map.special_ranges.items, ranges) |range, *target| target.* = .{ .begin = range.begin, .end = range.end };
    var encoded = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&encoded);
    try expect(c.flint_tokenizer_encode_prompt_span(tokenizer, rendered.ptr, 0, rendered.len, ranges.ptr, ranges.len, &encoded));
    var decoded: std.ArrayList(u8) = .empty;
    defer decoded.deinit(allocator);
    const first_added = c.flint_tokenizer_special_id(tokenizer, "<|begin_of_text|>");
    var controls: usize = 0;
    for (encoded.v[0..@intCast(encoded.len)]) |token| {
        var length: usize = 0;
        const piece = c.flint_tokenizer_piece(tokenizer, token, &length);
        try decoded.appendSlice(allocator, piece[0..length]);
        if (token >= first_added) {
            try expect(controls < ranges.len);
            const range = ranges[controls];
            try equal(rendered[range.begin..range.end], piece[0..length]);
            controls += 1;
        }
    }
    try expect(controls == ranges.len);
    try equal(rendered, decoded.items);
}
fn run(path: [*:0]const u8) !void {
    const tokenizer = c.flint_tokenizer_load(path) orelse return error.LoadTokenizer;
    defer c.flint_tokenizer_free(tokenizer);
    const literal = "literal <think> </think> <|user|> </arg_value> </tool_response> <|reserved_150019|> 雪 &lt;";
    var empty_map: types.PromptMap = .{};
    try encodeAndCheck(tokenizer, literal, &empty_map);

    const calls = [_]types.ToolCall{.{ .id = "one", .name = "echo", .arguments = "{\"text\":\"literal <think> </think> <|user|> </arg_value> </tool_response> <|reserved_150019|> 雪 &lt;\"}" }};
    const policy: model.Model = .{ .family = .flint };
    const messages = [_]types.Message{
        .{ .role = "system", .content = literal },
        .{ .role = "user", .content = literal },
        .{ .role = "assistant", .reasoning = literal, .content = literal, .calls = &calls },
        .{ .role = "tool", .tool_call_id = "one", .content = literal },
        .{ .role = "user", .content = "Continue" },
    };
    var map: types.PromptMap = .{};
    defer map.deinit(allocator);
    const rendered = try policy.prepare(allocator, .{ .messages = &messages }, .{ .prompt_map = &map }, 8192);
    defer allocator.free(rendered);
    try encodeAndCheck(tokenizer, rendered, &map);
    var found: usize = 0;
    var position: usize = 0;
    while (std.mem.findPos(u8, rendered, position, literal)) |begin| {
        for (map.special_ranges.items) |range| try expect(range.end <= begin or range.begin >= begin + literal.len);
        found += 1;
        position = begin + literal.len;
    }
    try expect(found == 6);

    const raw = "<tool_call>echo<arg_key>text</arg_key><arg_value>literal </arg_value> <|user|> data</arg_value></tool_call>";
    const markers = [_][]const u8{ "<tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "</tool_call>" };
    var raw_ranges: std.ArrayList(types.ByteRange) = .empty;
    defer raw_ranges.deinit(allocator);
    position = 0;
    for (markers, 0..) |marker, index| {
        const begin = if (index == 4) std.mem.findLast(u8, raw, marker).? else std.mem.findPos(u8, raw, position, marker).?;
        try raw_ranges.append(allocator, .{ .begin = begin, .end = begin + marker.len });
        position = begin + marker.len;
    }
    const history = [_]types.Message{
        .{ .role = "user", .content = "Inspect" },
        .{ .role = "assistant", .calls = &calls, .raw_tool_text = raw, .raw_tool_special_ranges = raw_ranges.items },
        .{ .role = "tool", .tool_call_id = "one", .content = literal },
    };
    var replay_map: types.PromptMap = .{};
    defer replay_map.deinit(allocator);
    const replay = try policy.prepare(allocator, .{ .messages = &history }, .{ .prompt_map = &replay_map }, 8192);
    defer allocator.free(replay);
    try expect(std.mem.indexOf(u8, replay, raw) != null);
    try encodeAndCheck(tokenizer, replay, &replay_map);

    const chat = [_]types.Message{ .{ .role = "system", .content = "System" }, .{ .role = "user", .content = literal } };
    var cli_map: types.PromptMap = .{};
    defer cli_map.deinit(allocator);
    const cli = try policy.prepare(allocator, .{ .messages = &chat }, .{ .prompt_map = &cli_map }, 8192);
    defer allocator.free(cli);
    var cli_tokens = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&cli_tokens);
    try expect(c.flint_tokenizer_encode_chat(tokenizer, "System", literal, &cli_tokens));
    const ranges = try allocator.alloc(c.ds4_prompt_range, cli_map.special_ranges.items.len);
    defer allocator.free(ranges);
    for (cli_map.special_ranges.items, ranges) |range, *target| target.* = .{ .begin = range.begin, .end = range.end };
    var direct_tokens = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&direct_tokens);
    try expect(c.flint_tokenizer_encode_prompt_span(tokenizer, cli.ptr, 0, cli.len, ranges.ptr, ranges.len, &direct_tokens));
    try expect(cli_tokens.len == direct_tokens.len);
    try expect(std.mem.eql(c_int, cli_tokens.v[0..@intCast(cli_tokens.len)], direct_tokens.v[0..@intCast(direct_tokens.len)]));
    std.debug.print("PASS: Flint plain payload BPE, actual framing IDs, canonical tool turns, mapped raw replay and CLI chat parity.\n", .{});
}
export fn main(argc: c_int, argv: [*c][*c]const u8) c_int {
    if (argc != 2) return 2;
    run(@ptrCast(argv[1])) catch |err| {
        std.debug.print("Flint prompt tokens: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}
