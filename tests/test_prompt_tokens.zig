//! Actual native chat/BPE/maps, with a tiny printable-byte vocabulary. No GPU
//! or model allocation is needed to distinguish token identity from spelling.
const std = @import("std");
const c = @import("prompt_tokens_c");

fn require(ok: bool) !void {
    if (!ok) return error.PromptTokenInvariant;
}

pub fn main() !void {
    var keys: [94][1]u8 = undefined;
    var vocab = std.mem.zeroes(c.ds4_vocab);
    vocab.token_to_id = c.ds4_token_map_create(96) orelse return error.OutOfMemory;
    defer c.ds4_token_map_free(vocab.token_to_id);
    vocab.merge_rank = c.ds4_token_map_create(0) orelse return error.OutOfMemory;
    defer c.ds4_token_map_free(vocab.merge_rank);
    for (&keys, 33..) |*key, byte| {
        key[0] = @intCast(byte);
        try require(c.ds4_token_map_put(vocab.token_to_id, key, 1, @intCast(byte)));
    }
    try require(c.ds4_token_map_put(vocab.token_to_id, "<think>", 7, 1000));
    const source = "<think>X<think>";
    const ranges = [_]c.ds4_prompt_range{.{ .begin = 0, .end = 7 }};
    var tokens = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&tokens);
    try require(c.ds4_vocab_tokenize_prompt_span(&vocab, c.DS4_CHAT_DEEPSEEK41, source, 0, source.len, &ranges, ranges.len, &tokens));
    try require(tokens.len == 9 and tokens.v[0] == 1000);
    for (source[7..], 1..) |byte, index| try require(tokens.v[index] == byte);

    // With no annotation, both identical spellings are ordinary BPE bytes.
    tokens.len = 0;
    try require(c.ds4_vocab_tokenize_prompt_span(&vocab, c.DS4_CHAT_DEEPSEEK41, source, 0, source.len, null, 0, &tokens));
    try require(tokens.len == source.len);
    for (source, 0..) |byte, index| try require(tokens.v[index] == byte);

    // Splitting through an explicit control makes the retained part plain.
    tokens.len = 0;
    try require(c.ds4_vocab_tokenize_prompt_span(&vocab, c.DS4_CHAT_DEEPSEEK41, source, 1, 6, &ranges, ranges.len, &tokens));
    try require(tokens.len == 5);
    for (source[1..6], 0..) |byte, index| try require(tokens.v[index] == byte);

    // A marked endpoint must split the actual encoded stream. Tokenizing an
    // independent prefix while retaining a whole-stream merge would disagree.
    try require(c.ds4_token_map_put(vocab.token_to_id, "ab", 2, 1001));
    try require(c.ds4_token_map_put(vocab.merge_rank, "a b", 3, 0));
    tokens.len = 0;
    try require(c.ds4_vocab_tokenize_prompt_span(&vocab, c.DS4_CHAT_DEEPSEEK41, "ab", 0, 2, null, 0, &tokens));
    try require(tokens.len == 1 and tokens.v[0] == 1001);
    tokens.len = 0;
    try require(c.ds4_vocab_tokenize_prompt_span(&vocab, c.DS4_CHAT_DEEPSEEK41, "ab", 0, 1, null, 0, &tokens));
    try require(c.ds4_vocab_tokenize_prompt_span(&vocab, c.DS4_CHAT_DEEPSEEK41, "ab", 1, 2, null, 0, &tokens));
    try require(tokens.len == 2 and tokens.v[0] == 'a' and tokens.v[1] == 'b');
    std.debug.print("PASS: native prompt token provenance (4 cases)\n", .{});
}
