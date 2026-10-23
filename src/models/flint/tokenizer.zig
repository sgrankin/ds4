//! Flint's tokenizer.json byte-level BPE. Reuses the shared native BPE and
//! macOS ICU for the checkpoint's three ordered Unicode regular expressions.
//! Prompt/control policy follows recursive-oss/rec (Apache-2.0).
const std = @import("std");
const c = @import("flint_tokenizer_c");
const prompt = @import("flint_prompt");
const icu = c;
const allocator = std.heap.c_allocator;
const Handle = c.flint_tokenizer;
const Added = struct { content: []const u8, id: c_int };
const Regex = *icu.URegularExpression;

const Tokenizer = struct {
    arena: std.heap.ArenaAllocator,
    tokens: *c.ds4_token_map,
    merges: *c.ds4_token_map,
    pieces: []const []const u8,
    added: []const Added,
    added_start: [256]bool,
    patterns: [3]Regex,

    fn deinit(self: *Tokenizer) void {
        for (self.patterns) |pattern| icu.uregex_close(pattern);
        c.ds4_token_map_free(self.tokens);
        c.ds4_token_map_free(self.merges);
        self.arena.deinit();
    }
};

fn get(handle: *const Handle) *const Tokenizer {
    return @ptrCast(@alignCast(handle));
}

fn field(value: std.json.Value, key: []const u8) !std.json.Value {
    if (value != .object) return error.InvalidTokenizer;
    return value.object.get(key) orelse error.InvalidTokenizer;
}

fn string(value: std.json.Value) ![]const u8 {
    return if (value == .string) value.string else error.InvalidTokenizer;
}

// The inverse GPT-2 byte alphabet. Normal vocabulary strings contain only
// these code points; added tokens are intentionally decoded as literal UTF-8.
fn decodePiece(a: std.mem.Allocator, encoded: []const u8) ![]const u8 {
    var inverse: [512]?u8 = @splat(null);
    var next: usize = 256;
    for (0..256) |byte| {
        const printable = (byte >= 33 and byte <= 126) or (byte >= 161 and byte <= 172) or byte >= 174;
        const codepoint = if (printable) byte else value: {
            defer next += 1;
            break :value next;
        };
        inverse[codepoint] = @intCast(byte);
    }
    var result: std.ArrayList(u8) = .empty;
    var iterator = (try std.unicode.Utf8View.init(encoded)).iterator();
    while (iterator.nextCodepoint()) |cp| {
        if (cp >= inverse.len) return error.InvalidTokenizer;
        try result.append(a, inverse[cp] orelse return error.InvalidTokenizer);
    }
    return result.toOwnedSlice(a);
}

fn load(path: []const u8) !*Tokenizer {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    const terminated_path = try a.dupeSentinel(u8, path, 0);
    const file = c.fopen(terminated_path.ptr, "rb") orelse return error.OpenTokenizer;
    defer _ = c.fclose(file);
    if (c.fseek(file, 0, c.SEEK_END) != 0) return error.ReadTokenizer;
    const length = c.ftell(file);
    if (length < 0 or length > 64 * 1024 * 1024 or c.fseek(file, 0, c.SEEK_SET) != 0) return error.ReadTokenizer;
    const bytes = try a.alloc(u8, @intCast(length));
    if (c.fread(bytes.ptr, 1, bytes.len, file) != bytes.len) return error.ReadTokenizer;
    const json = try std.json.parseFromSlice(std.json.Value, a, bytes, .{ .allocate = .alloc_always });
    const model = try field(json.value, "model");
    if (!std.mem.eql(u8, try string(try field(model, "type")), "BPE")) return error.InvalidTokenizer;
    const vocab = try field(model, "vocab");
    const merge_values = try field(model, "merges");
    const added_values = try field(json.value, "added_tokens");
    if (vocab != .object or merge_values != .array or added_values != .array) return error.InvalidTokenizer;
    const count = vocab.object.count() + added_values.array.items.len;
    const pieces = try a.alloc([]const u8, count);
    @memset(pieces, "");
    const tokens = c.ds4_token_map_create(count) orelse return error.OutOfMemory;
    errdefer c.ds4_token_map_free(tokens);
    const merges = c.ds4_token_map_create(merge_values.array.items.len) orelse return error.OutOfMemory;
    errdefer c.ds4_token_map_free(merges);
    var entries = vocab.object.iterator();
    while (entries.next()) |entry| {
        if (entry.value_ptr.* != .integer) return error.InvalidTokenizer;
        const id = entry.value_ptr.integer;
        if (id < 0 or id >= count) return error.InvalidTokenizer;
        pieces[@intCast(id)] = try decodePiece(a, entry.key_ptr.*);
        if (!c.ds4_token_map_put(tokens, entry.key_ptr.ptr, entry.key_ptr.len, @intCast(id))) return error.OutOfMemory;
    }
    for (merge_values.array.items, 0..) |merge, rank| {
        if (merge != .array or merge.array.items.len != 2) return error.InvalidTokenizer;
        const key = try std.fmt.allocPrint(a, "{s} {s}", .{
            try string(merge.array.items[0]), try string(merge.array.items[1]),
        });
        if (!c.ds4_token_map_put(merges, key.ptr, key.len, @intCast(rank))) return error.OutOfMemory;
    }
    const added = try a.alloc(Added, added_values.array.items.len);
    var added_start: [256]bool = @splat(false);
    for (added_values.array.items, added) |value, *token| {
        const id = try field(value, "id");
        const content = try string(try field(value, "content"));
        if (id != .integer or id.integer < 0 or id.integer >= count or content.len == 0) return error.InvalidTokenizer;
        // The shipped tokenizer needs no added-token whitespace/word rules.
        for ([_][]const u8{ "single_word", "lstrip", "rstrip", "normalized" }) |key| {
            const flag = try field(value, key);
            if (flag != .bool or flag.bool) return error.InvalidTokenizer;
        }
        token.* = .{ .content = content, .id = @intCast(id.integer) };
        pieces[@intCast(id.integer)] = content;
        added_start[content[0]] = true;
    }
    const sequence = try field(try field(json.value, "pre_tokenizer"), "pretokenizers");
    if (sequence != .array or sequence.array.items.len != 4) return error.InvalidTokenizer;
    var patterns: [3]Regex = undefined;
    var compiled: usize = 0;
    errdefer for (patterns[0..compiled]) |pattern| icu.uregex_close(pattern);
    for (sequence.array.items[0..3], 0..) |split, index| {
        if (!std.mem.eql(u8, try string(try field(split, "behavior")), "Isolated")) return error.InvalidTokenizer;
        const pattern = try string(try field(try field(split, "pattern"), "Regex"));
        const utf16 = try std.unicode.utf8ToUtf16LeAlloc(a, pattern);
        var status: icu.UErrorCode = icu.U_ZERO_ERROR;
        patterns[index] = icu.uregex_open(utf16.ptr, @intCast(utf16.len), 0, null, &status) orelse return error.InvalidRegex;
        compiled += 1;
        if (status > icu.U_ZERO_ERROR) return error.InvalidRegex;
    }
    const byte_level = sequence.array.items[3];
    if (!std.mem.eql(u8, try string(try field(byte_level, "type")), "ByteLevel")) return error.InvalidTokenizer;
    for ([_][]const u8{ "add_prefix_space", "use_regex" }) |key| {
        const flag = try field(byte_level, key);
        if (flag != .bool or flag.bool) return error.InvalidTokenizer;
    }
    const result = try allocator.create(Tokenizer);
    result.* = .{ .arena = arena, .tokens = tokens, .merges = merges, .pieces = pieces, .added = added, .added_start = added_start, .patterns = patterns };
    return result;
}

export fn flint_tokenizer_load(path: [*:0]const u8) ?*Handle {
    return @ptrCast(load(std.mem.span(path)) catch |err| {
        std.debug.print("Flint tokenizer: {s}: {s}\n", .{ path, @errorName(err) });
        return null;
    });
}

export fn flint_tokenizer_free(handle: ?*Handle) void {
    const value = handle orelse return;
    const tokenizer: *Tokenizer = @constCast(get(value));
    tokenizer.deinit();
    allocator.destroy(tokenizer);
}

const Encoding = struct {
    tokenizer: *const Tokenizer,
    patterns: [3]Regex,
    out: *c.ds4_tokens,

    fn split(self: *const Encoding, text: []const u8, stage: usize) !void {
        if (text.len == 0) return;
        if (stage == self.patterns.len) {
            if (!c.ds4_bpe_emit_piece(self.tokenizer.tokens, self.tokenizer.merges, text.ptr, text.len, self.out)) return error.OutOfMemory;
            return;
        }
        var status: icu.UErrorCode = icu.U_ZERO_ERROR;
        const view = icu.utext_openUTF8(null, text.ptr, @intCast(text.len), &status) orelse return error.InvalidText;
        defer _ = icu.utext_close(view);
        const regex = self.patterns[stage];
        icu.uregex_setUText(regex, view, &status);
        var position: usize = 0;
        while (icu.uregex_findNext(regex, &status) != 0) {
            const begin: usize = @intCast(icu.uregex_start64(regex, 0, &status));
            const end: usize = @intCast(icu.uregex_end64(regex, 0, &status));
            if (status > icu.U_ZERO_ERROR or end <= begin) return error.InvalidText;
            try self.split(text[position..begin], stage + 1);
            try self.split(text[begin..end], stage + 1);
            position = end;
        }
        if (status > icu.U_ZERO_ERROR) return error.InvalidText;
        try self.split(text[position..], stage + 1);
    }
};

fn encode(tokenizer: *const Tokenizer, text: []const u8, out: *c.ds4_tokens) !void {
    var encoding = Encoding{ .tokenizer = tokenizer, .patterns = undefined, .out = out };
    var cloned: usize = 0;
    defer for (encoding.patterns[0..cloned]) |pattern| icu.uregex_close(pattern);
    for (tokenizer.patterns, 0..) |pattern, index| {
        var status: icu.UErrorCode = icu.U_ZERO_ERROR;
        encoding.patterns[index] = icu.uregex_clone(pattern, &status) orelse return error.OutOfMemory;
        cloned += 1;
        if (status > icu.U_ZERO_ERROR) return error.InvalidRegex;
    }
    var begin: usize = 0;
    var position: usize = 0;
    while (position < text.len) {
        if (!tokenizer.added_start[text[position]]) {
            position += 1;
            continue;
        }
        var longest: ?Added = null;
        for (tokenizer.added) |token| {
            if (std.mem.startsWith(u8, text[position..], token.content) and (longest == null or token.content.len > longest.?.content.len)) longest = token;
        }
        if (longest) |token| {
            try encoding.split(text[begin..position], 0);
            c.ds4_tokens_push(out, token.id);
            position += token.content.len;
            begin = position;
        } else position += 1;
    }
    try encoding.split(text[begin..], 0);
}

export fn flint_tokenizer_encode(handle: *const Handle, raw: [*c]const u8, len: usize, out: *c.ds4_tokens) bool {
    const text = if (len == 0) "" else raw[0..len];
    encode(get(handle), text, out) catch return false;
    return true;
}

export fn flint_tokenizer_piece(handle: *const Handle, id: c_int, len: *usize) [*c]const u8 {
    len.* = 0;
    const tokenizer = get(handle);
    if (id < 0 or id >= tokenizer.pieces.len) return null;
    const piece = tokenizer.pieces[@intCast(id)];
    len.* = piece.len;
    return piece.ptr;
}

export fn flint_tokenizer_vocab_size(handle: *const Handle) c_int {
    return @intCast(get(handle).pieces.len);
}

export fn flint_tokenizer_special_id(handle: *const Handle, spelling: [*:0]const u8) c_int {
    const text = std.mem.span(spelling);
    for (get(handle).added) |token| if (std.mem.eql(u8, token.content, text)) return token.id;
    return -1;
}

export fn flint_tokenizer_is_stop(handle: *const Handle, id: c_int) bool {
    for ([_][:0]const u8{ "<|end_of_text|>", "<|user|>", "<|observation|>", "<|developer|>" }) |spelling| {
        if (flint_tokenizer_special_id(handle, spelling) == id) return true;
    }
    return false;
}

export fn flint_render_message(role: [*:0]const u8, content: [*:0]const u8) [*c]u8 {
    const rendered = prompt.render(allocator, &.{.{ .role = std.mem.span(role), .content = std.mem.span(content) }}, .{ .add_generation_prompt = false }) catch return null;
    return rendered.ptr;
}

export fn flint_render_chat(system: [*c]const u8, user: [*:0]const u8) [*c]u8 {
    const messages = [_]@import("serving_types").Message{
        .{ .role = "system", .content = if (system == null) null else std.mem.span(system) },
        .{ .role = "user", .content = std.mem.span(user) },
    };
    const rendered = prompt.render(allocator, messages[if (system == null or system[0] == 0) @as(usize, 1) else 0..], .{}) catch return null;
    return rendered.ptr;
}
