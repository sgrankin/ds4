//! Borrowed vocabulary maps and byte-level BPE pieces. Vocabulary ownership,
//! ordered splitting and chat policy live in vocab.zig, pretokenizer.zig and
//! chat.zig; the C engine coordinates their public ABI.
const std = @import("std");
const c = @import("tokenizer_c");
const Handle = c.ds4_token_map;
const Tokens = c.ds4_tokens;
const Map = std.StringHashMap(c_int);
const allocator = std.heap.c_allocator;

fn mutableMap(handle: *Handle) *Map {
    return @ptrCast(@alignCast(handle));
}

fn readableMap(handle: *const Handle) *const Map {
    return @ptrCast(@alignCast(handle));
}

fn borrowedBytes(ptr: [*c]const u8, len: u64) []const u8 {
    return if (len == 0) "" else ptr[0..@as(usize, @intCast(len))];
}

export fn ds4_token_map_create(max_entries: u64) ?*Handle {
    // Leave room for the standard map's load factor and power-of-two bucket
    // count. Beyond this limit its u32 capacity cannot represent the reservation.
    const largest_capacity = (@as(u64, std.math.maxInt(Map.Size)) + 1) / 2;
    const entry_limit = (largest_capacity * std.hash_map.default_max_load_percentage - 1) / 100;
    if (max_entries > entry_limit) return null;
    const map = allocator.create(Map) catch return null;
    map.* = Map.init(allocator);
    map.ensureTotalCapacity(@intCast(max_entries)) catch {
        map.deinit();
        allocator.destroy(map);
        return null;
    };
    return @ptrCast(map);
}

export fn ds4_token_map_free(maybe_handle: ?*Handle) void {
    const handle = maybe_handle orelse return;
    const map = mutableMap(handle);
    // StringHashMap deinit releases its table, never the borrowed key bytes.
    map.deinit();
    allocator.destroy(map);
}

export fn ds4_token_map_put(handle: *Handle, key: [*c]const u8, len: u64, value: c_int) bool {
    mutableMap(handle).put(borrowedBytes(key, len), value) catch return false;
    return true;
}

export fn ds4_token_map_get(handle: *const Handle, key: [*c]const u8, len: u64, value: *c_int) bool {
    const found = readableMap(handle).get(borrowedBytes(key, len)) orelse return false;
    value.* = found;
    return true;
}

// GPT-2 maps printable bytes to themselves; the remaining bytes receive code
// points starting at 256 in byte order. This is a format table, not Unicode
// normalization. Every resulting code point is below 2048 (one or two UTF-8 bytes).
const byte_codepoints: [256]u16 = table: {
    var result: [256]u16 = undefined;
    var next: u16 = 256;
    for (&result, 0..) |*codepoint, byte| {
        if ((byte >= 33 and byte <= 126) or (byte >= 161 and byte <= 172) or byte >= 174) {
            codepoint.* = @intCast(byte);
        } else {
            codepoint.* = next;
            next += 1;
        }
    }
    break :table result;
};

// Symbols always cover adjacent regions of one immutable encoded buffer. A
// merge changes only the region boundary; no symbol owns or copies payload bytes.
const Span = struct { start: usize, end: usize };

fn mergeRank(map: *const Map, left: []const u8, right: []const u8, scratch: []u8) c_int {
    // GGUF ranks use "left right", not their concatenation. Reuse one buffer
    // large enough for any pair in this piece; map lookup borrows it only here.
    @memcpy(scratch[0..left.len], left);
    scratch[left.len] = ' ';
    @memcpy(scratch[left.len + 1 ..][0..right.len], right);
    return map.get(scratch[0 .. left.len + 1 + right.len]) orelse -1;
}

export fn ds4_bpe_emit_piece(token_handle: *const Handle, merge_handle: *const Handle, raw: [*c]const u8, len: u64, out: *Tokens) bool {
    if (len == 0) return true;
    const input = borrowedBytes(raw, len);
    if (input.len > std.math.maxInt(usize) / 2) return false;
    const tokens = readableMap(token_handle);
    const merges = readableMap(merge_handle);

    // One encoding allocation and one symbol vector, both bounded by the raw
    // piece length. Each input byte creates one initial code-point symbol.
    const encoded = allocator.alloc(u8, input.len * 2) catch return false;
    defer allocator.free(encoded);
    var symbols = std.ArrayList(Span).initCapacity(allocator, input.len) catch return false;
    defer symbols.deinit(allocator);
    var encoded_len: usize = 0;
    for (input) |byte| {
        const start = encoded_len;
        const codepoint = byte_codepoints[byte];
        if (codepoint <= 0x7f) {
            encoded[encoded_len] = @intCast(codepoint);
            encoded_len += 1;
        } else {
            encoded[encoded_len] = @intCast(0xc0 | (codepoint >> 6));
            encoded[encoded_len + 1] = @intCast(0x80 | (codepoint & 0x3f));
            encoded_len += 2;
        }
        symbols.appendAssumeCapacity(.{ .start = start, .end = encoded_len });
    }
    const bytes = encoded[0..encoded_len];
    const scratch = allocator.alloc(u8, encoded_len + 1) catch return false;
    defer allocator.free(scratch);

    while (true) {
        var best_index: ?usize = null;
        var best_rank: c_int = std.math.maxInt(c_int);
        for (0..symbols.items.len - 1) |index| {
            const left = symbols.items[index];
            const right = symbols.items[index + 1];
            const rank = mergeRank(merges, bytes[left.start..left.end], bytes[right.start..right.end], scratch);
            // Strict comparison preserves the first (leftmost) equal-rank pair.
            // INT32_MAX is the C sentinel and is deliberately never selected.
            if (rank >= 0 and rank < best_rank) {
                best_rank = rank;
                best_index = index;
            }
        }
        const index = best_index orelse break;
        symbols.items[index].end = symbols.items[index + 1].end;
        _ = symbols.orderedRemove(index + 1);
    }

    for (symbols.items) |symbol| {
        const text = bytes[symbol.start..symbol.end];
        if (tokens.get(text)) |token| {
            c.ds4_tokens_push(out, token);
        } else {
            // Preserve the original fallback: look up individual ENCODED bytes,
            // silently omitting a byte absent from the token table. Do not turn
            // this into raw-byte fallback or another Unicode decoding pass.
            for (0..text.len) |index| {
                if (tokens.get(text[index .. index + 1])) |token| c.ds4_tokens_push(out, token);
            }
        }
    }
    return true;
}
