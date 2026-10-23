//! Vocabulary data ownership over borrowed GGUF strings. The C model owner
//! locates metadata and owns the mapping; this module owns only the token array
//! and two opaque maps. Closing the vocabulary never frees borrowed text.
const std = @import("std");
const c = @import("vocab_c");
const Vocab = c.ds4_vocab;
const Source = c.ds4_vocab_source;
const String = c.ds4_str;
const allocator = std.heap.c_allocator;
const BuildError = error{ OutOfMemory, InvalidGGUF, MissingToken };

fn errorText(buffer: *[256]u8, message: []const u8) void {
    const count = @min(message.len, buffer.len - 1);
    @memcpy(buffer[0..count], message[0..count]);
    buffer[count] = 0;
}
fn optional(vocab: *const Vocab, spelling: []const u8) c_int {
    var token: c_int = -1;
    _ = c.ds4_token_map_get(vocab.token_to_id, spelling.ptr, spelling.len, &token);
    return token;
}
fn required(vocab: *const Vocab, spelling: []const u8, message: *[256]u8) BuildError!c_int {
    const token = optional(vocab, spelling);
    if (token < 0) {
        _ = std.mem.printSentinel(message, "required tokenizer token is missing: {s}", .{spelling}, 0) catch unreachable;
        return error.MissingToken;
    }
    return token;
}
fn put(map: ?*c.ds4_token_map, key: String, value: c_int) BuildError!void {
    c.ds4_vocab_admit_allocation("token map put", key.len);
    if (!c.ds4_token_map_put(map, key.ptr, key.len, value)) return error.OutOfMemory;
}
fn load(vocab: *Vocab, source: *const Source, message: *[256]u8) BuildError!void {
    if (source.token_count > std.math.maxInt(c_int)) {
        errorText(message, "GGUF tokenizer token table is missing or invalid");
        return error.InvalidGGUF;
    }
    const count: usize = @intCast(source.token_count);
    c.ds4_vocab_admit_allocation("calloc", count * @sizeOf(String));
    // Keep a real libc-owned pointer for the zero-token case too. C consumers
    // may release this record through the same public ownership contract.
    const tokens = allocator.alloc(String, @max(count, 1)) catch return error.OutOfMemory;
    @memset(tokens, std.mem.zeroes(String));
    vocab.token = tokens.ptr;
    vocab.n_vocab = @intCast(count);
    c.ds4_vocab_admit_allocation("token map create", source.token_count);
    vocab.token_to_id = c.ds4_token_map_create(source.token_count);
    if (vocab.token_to_id == null) return error.OutOfMemory;
    var cursor = source.tokens;
    for (tokens[0..count], 0..) |*token, index| {
        if (!c.ds4_gguf_string(&cursor, token)) {
            errorText(message, std.mem.span(@as([*:0]const u8, @ptrCast(&cursor.@"error"))));
            return error.InvalidGGUF;
        }
        try put(vocab.token_to_id, token.*, @intCast(index));
    }
    c.ds4_vocab_admit_allocation("token map create", source.merge_count);
    vocab.merge_rank = c.ds4_token_map_create(source.merge_count);
    if (vocab.merge_rank == null) return error.OutOfMemory;
    cursor = source.merges;
    var index: u64 = 0;
    while (index < source.merge_count) : (index += 1) {
        var merge: String = undefined;
        if (!c.ds4_gguf_string(&cursor, &merge)) {
            errorText(message, std.mem.span(@as([*:0]const u8, @ptrCast(&cursor.@"error"))));
            return error.InvalidGGUF;
        }
        try put(vocab.merge_rank, merge, @bitCast(@as(u32, @truncate(index))));
    }

    // Every absent marker is -1. Unlike the old memset-plus-many-assignments,
    // the family branches below only state the markers they actually provide.
    vocab.bos_id = -1;
    vocab.eos_id = -1;
    vocab.system_id = -1;
    vocab.user_id = -1;
    vocab.assistant_id = -1;
    vocab.observation_id = -1;
    vocab.sop_id = -1;
    vocab.think_start_id = -1;
    vocab.think_end_id = -1;
    vocab.tool_call_start_id = -1;
    vocab.tool_call_end_id = -1;
    vocab.tool_response_start_id = -1;
    vocab.tool_response_end_id = -1;
    vocab.arg_key_start_id = -1;
    vocab.arg_key_end_id = -1;
    vocab.arg_value_start_id = -1;
    vocab.arg_value_end_id = -1;
    vocab.dsml_id = -1;
    vocab.im_start_id = -1;
    vocab.im_end_id = -1;
    vocab.endoftext_id = -1;
    if (source.family == c.DS4_CHAT_QWEN35) {
        // ChatML has no BOS; endoftext is the document separator and a second
        // generation stop, while im_end closes an assistant conversation turn.
        vocab.im_start_id = try required(vocab, "<|im_start|>", message);
        vocab.im_end_id = try required(vocab, "<|im_end|>", message);
        vocab.endoftext_id = try required(vocab, "<|endoftext|>", message);
        vocab.eos_id = vocab.im_end_id;
        vocab.think_start_id = try required(vocab, "<think>", message);
        vocab.think_end_id = try required(vocab, "</think>", message);
        vocab.tool_call_start_id = optional(vocab, "<tool_call>");
        vocab.tool_call_end_id = optional(vocab, "</tool_call>");
        vocab.tool_response_start_id = optional(vocab, "<tool_response>");
        vocab.tool_response_end_id = optional(vocab, "</tool_response>");
        return;
    }
    if (source.family == c.DS4_CHAT_GLM4) {
        vocab.bos_id = if (source.glm_bos_id >= 0) source.glm_bos_id else optional(vocab, "<sop>");
        vocab.eos_id = if (source.glm_eos_id >= 0) source.glm_eos_id else optional(vocab, "<|endoftext|>");
        vocab.system_id = optional(vocab, "<|system|>");
        vocab.user_id = optional(vocab, "<|user|>");
        vocab.assistant_id = optional(vocab, "<|assistant|>");
        vocab.observation_id = optional(vocab, "<|observation|>");
        vocab.sop_id = optional(vocab, "<sop>");
        vocab.think_start_id = optional(vocab, "<think>");
        vocab.think_end_id = optional(vocab, "</think>");
        vocab.tool_call_start_id = optional(vocab, "<tool_call>");
        vocab.tool_call_end_id = optional(vocab, "</tool_call>");
        vocab.tool_response_start_id = optional(vocab, "<tool_response>");
        vocab.tool_response_end_id = optional(vocab, "</tool_response>");
        vocab.arg_key_start_id = optional(vocab, "<arg_key>");
        vocab.arg_key_end_id = optional(vocab, "</arg_key>");
        vocab.arg_value_start_id = optional(vocab, "<arg_value>");
        vocab.arg_value_end_id = optional(vocab, "</arg_value>");
        return;
    }
    vocab.bos_id = try required(vocab, "<｜begin▁of▁sentence｜>", message);
    vocab.eos_id = try required(vocab, "<｜end▁of▁sentence｜>", message);
    if (source.family == c.DS4_CHAT_DEEPSEEK41) vocab.system_id = try required(vocab, "<｜System｜>", message);
    vocab.user_id = try required(vocab, "<｜User｜>", message);
    vocab.assistant_id = try required(vocab, "<｜Assistant｜>", message);
    vocab.think_start_id = try required(vocab, "<think>", message);
    vocab.think_end_id = try required(vocab, "</think>", message);
    vocab.dsml_id = try required(vocab, "｜DSML｜", message);
}

export fn ds4_vocab_build(out: *Vocab, source: *const Source, message: *[256]u8) bool {
    out.* = std.mem.zeroes(Vocab);
    message[0] = 0;
    load(out, source, message) catch |failure| {
        if (failure == error.OutOfMemory) errorText(message, "out of memory");
        ds4_vocab_release(out);
        return false;
    };
    return true;
}
export fn ds4_vocab_release(vocab: *Vocab) void {
    if (vocab.token != null) allocator.free(vocab.token[0..@max(@as(usize, @intCast(vocab.n_vocab)), 1)]);
    c.ds4_token_map_free(vocab.token_to_id);
    c.ds4_token_map_free(vocab.merge_rank);
    vocab.* = std.mem.zeroes(Vocab);
}

// GPT-2 inverse of its printable-byte alphabet. Unknown codepoints are skipped
// rather than copied or replaced; this is byte decoding, not normalization.
const decoded_bytes: [324]i16 = inverse: {
    var result: [324]i16 = @splat(-1);
    var next: usize = 256;
    for (0..256) |byte| {
        if ((byte >= 33 and byte <= 126) or (byte >= 161 and byte <= 172) or byte >= 174) {
            result[byte] = @intCast(byte);
        } else {
            result[next] = @intCast(byte);
            next += 1;
        }
    }
    break :inverse result;
};

fn codepoint(bytes: []const u8, position: *usize) u32 {
    const tail = bytes[position.*..];
    const first = tail[0];
    const width: usize = if (first & 0xe0 == 0xc0 and tail.len >= 2) 2 else if (first & 0xf0 == 0xe0 and tail.len >= 3) 3 else if (first & 0xf8 == 0xf0 and tail.len >= 4) 4 else 1;
    position.* += width;
    const decoded = switch (width) {
        2 => std.unicode.utf8Decode2(tail[0..2].*),
        3 => std.unicode.utf8Decode3(tail[0..3].*),
        4 => std.unicode.utf8Decode4(tail[0..4].*),
        else => return first,
    };
    if (decoded) |valid| return valid else |_| {}
    return switch (width) {
        // The C decoder deliberately masks continuation bytes without checking
        // them, including overlong/surrogate encodings. Keep that compatibility
        // for malformed GGUF token text; valid UTF-8 uses the standard decoder.
        2 => (@as(u32, first & 0x1f) << 6) | (tail[1] & 0x3f),
        3 => (@as(u32, first & 0x0f) << 12) | (@as(u32, tail[1] & 0x3f) << 6) | (tail[2] & 0x3f),
        4 => (@as(u32, first & 0x07) << 18) | (@as(u32, tail[1] & 0x3f) << 12) | (@as(u32, tail[2] & 0x3f) << 6) | (tail[3] & 0x3f),
        else => first,
    };
}
export fn ds4_vocab_decode_token(vocab: *const Vocab, token: c_int, length: [*c]usize) [*c]u8 {
    const bytes: []const u8 = if (token < 0 or token >= vocab.n_vocab) "" else vocab.token[@intCast(token)].ptr[0..@intCast(vocab.token[@intCast(token)].len)];
    c.ds4_vocab_admit_allocation("malloc", bytes.len + 1);
    const output = allocator.alloc(u8, bytes.len + 1) catch return null;
    var used: usize = 0;
    if (std.mem.indexOf(u8, bytes, "\xef\xbd\x9c") != null) {
        // U+FF5C marks literal DeepSeek special spellings. Preserve every byte.
        @memcpy(output[0..bytes.len], bytes);
        used = bytes.len;
    } else {
        var position: usize = 0;
        while (position < bytes.len) {
            const cp = codepoint(bytes, &position);
            if (cp < decoded_bytes.len and decoded_bytes[cp] >= 0) {
                output[used] = @intCast(decoded_bytes[cp]);
                used += 1;
            }
        }
    }
    output[used] = 0;
    if (length != null) length[0] = used;
    return output.ptr;
}
