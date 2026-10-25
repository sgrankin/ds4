//! Clef's added-token → NFC → Qwen Unicode split → byte BPE pipeline.
//! Uses the shared native BPE and ICU, with immutable model vocabulary.
const std = @import("std");
const c = @import("clef_tokenizer_c");
const Allocator = std.mem.Allocator;
const Added = struct { text: []const u8, id: i32 };

pub const Tokenizer = struct {
    arena: std.heap.ArenaAllocator,
    tokens: *c.ds4_token_map,
    merges: *c.ds4_token_map,
    added: []const Added,
    added_start: [256]bool,
    pattern: *c.URegularExpression,
    vocab_size: usize,

    pub fn open(allocator: Allocator, path: []const u8) !Tokenizer {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const a = arena.allocator();
        const bytes = try readFile(a, path);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, bytes, .{ .allocate = .alloc_always });
        const root = parsed.value;
        const model = try field(root, "model");
        if (!std.mem.eql(u8, try string(try field(model, "type")), "BPE")) return error.UnsupportedTokenizer;
        const vocab = try field(model, "vocab");
        const merge_values = try field(model, "merges");
        const added_values = try field(root, "added_tokens");
        if (vocab != .object or merge_values != .array or added_values != .array) return error.InvalidTokenizer;
        const tokens = c.ds4_token_map_create(vocab.object.count()) orelse return error.OutOfMemory;
        errdefer c.ds4_token_map_free(tokens);
        const merges = c.ds4_token_map_create(merge_values.array.items.len) orelse return error.OutOfMemory;
        errdefer c.ds4_token_map_free(merges);
        var vocab_size: usize = 0;
        var entries = vocab.object.iterator();
        while (entries.next()) |entry| {
            const id = try integer(entry.value_ptr.*);
            if (!c.ds4_token_map_put(tokens, entry.key_ptr.ptr, entry.key_ptr.len, @intCast(id))) return error.OutOfMemory;
            vocab_size = @max(vocab_size, id + 1);
        }
        for (merge_values.array.items, 0..) |merge, rank| {
            if (merge != .array or merge.array.items.len != 2) return error.InvalidTokenizer;
            const key = try std.fmt.allocPrint(a, "{s} {s}", .{ try string(merge.array.items[0]), try string(merge.array.items[1]) });
            var existing: c_int = undefined;
            // Hugging Face retains the first rank for a repeated merge.
            if (!c.ds4_token_map_get(merges, key.ptr, key.len, &existing) and
                !c.ds4_token_map_put(merges, key.ptr, key.len, @intCast(rank))) return error.OutOfMemory;
        }
        const added = try a.alloc(Added, added_values.array.items.len);
        var added_start: [256]bool = @splat(false);
        for (added_values.array.items, added) |value, *out| {
            const text = try string(try field(value, "content"));
            const id = try integer(try field(value, "id"));
            if (text.len == 0) return error.InvalidTokenizer;
            for ([_][]const u8{ "single_word", "lstrip", "rstrip", "normalized" }) |key| {
                if (try field(value, key) != .bool or (try field(value, key)).bool) return error.UnsupportedAddedToken;
            }
            out.* = .{ .text = text, .id = @intCast(id) };
            added_start[text[0]] = true;
            vocab_size = @max(vocab_size, id + 1);
        }
        if (!std.mem.eql(u8, try string(try field(try field(root, "normalizer"), "type")), "NFC")) return error.UnsupportedNormalizer;
        const sequence = try field(try field(root, "pre_tokenizer"), "pretokenizers");
        if (sequence != .array or sequence.array.items.len != 2) return error.UnsupportedTokenizer;
        const split = sequence.array.items[0];
        if (!std.mem.eql(u8, try string(try field(split, "behavior")), "Isolated")) return error.UnsupportedTokenizer;
        const regex = try string(try field(try field(split, "pattern"), "Regex"));
        const utf16 = try std.unicode.utf8ToUtf16LeAlloc(a, regex);
        var status: c.UErrorCode = c.U_ZERO_ERROR;
        const pattern = c.uregex_open(utf16.ptr, @intCast(utf16.len), 0, null, &status) orelse return error.InvalidRegex;
        errdefer c.uregex_close(pattern);
        if (status > c.U_ZERO_ERROR) return error.InvalidRegex;
        const level = sequence.array.items[1];
        if (!std.mem.eql(u8, try string(try field(level, "type")), "ByteLevel") or
            (try field(level, "add_prefix_space")).bool or (try field(level, "use_regex")).bool) return error.UnsupportedTokenizer;
        return .{ .arena = arena, .tokens = tokens, .merges = merges, .added = added, .added_start = added_start, .pattern = pattern, .vocab_size = vocab_size };
    }

    pub fn specialId(self: *const Tokenizer, spelling: []const u8) !u32 {
        for (self.added) |token| if (std.mem.eql(u8, token.text, spelling)) return @intCast(token.id);
        return error.MissingMediaToken;
    }

    pub fn deinit(self: *Tokenizer) void {
        c.uregex_close(self.pattern);
        c.ds4_token_map_free(self.tokens);
        c.ds4_token_map_free(self.merges);
        self.arena.deinit();
        self.* = undefined;
    }

    /// Return an owned token slice. Match added tokens before normalizing text,
    /// so decomposed ordinary Unicode is normalized without altering controls.
    pub fn encode(self: *const Tokenizer, a: Allocator, text: []const u8) ![]i32 {
        var output: c.ds4_tokens = .{ .v = null, .len = 0, .cap = 0 };
        defer c.ds4_tokens_free(&output);
        var status: c.UErrorCode = c.U_ZERO_ERROR;
        const regex = c.uregex_clone(self.pattern, &status) orelse return error.OutOfMemory;
        defer c.uregex_close(regex);
        if (status > c.U_ZERO_ERROR) return error.InvalidRegex;
        var begin: usize = 0;
        var position: usize = 0;
        while (position < text.len) {
            if (!self.added_start[text[position]]) {
                position += 1;
                continue;
            }
            var longest: ?Added = null;
            for (self.added) |token| {
                if (std.mem.startsWith(u8, text[position..], token.text) and (longest == null or token.text.len > longest.?.text.len)) longest = token;
            }
            if (longest) |token| {
                try self.ordinary(a, regex, text[begin..position], &output);
                c.ds4_tokens_push(&output, token.id);
                position += token.text.len;
                begin = position;
            } else position += 1;
        }
        try self.ordinary(a, regex, text[begin..], &output);
        if (output.len == 0) return a.alloc(i32, 0);
        return a.dupe(i32, output.v[0..@intCast(output.len)]);
    }

    fn ordinary(self: *const Tokenizer, a: Allocator, regex: *c.URegularExpression, source: []const u8, output: *c.ds4_tokens) !void {
        if (source.len == 0) return;
        const normalized = try nfc(a, source);
        defer a.free(normalized);
        var status: c.UErrorCode = c.U_ZERO_ERROR;
        const view = c.utext_openUTF8(null, normalized.ptr, @intCast(normalized.len), &status) orelse return error.InvalidText;
        defer _ = c.utext_close(view);
        c.uregex_setUText(regex, view, &status);
        var position: usize = 0;
        while (c.uregex_findNext(regex, &status) != 0) {
            const begin: usize = @intCast(c.uregex_start64(regex, 0, &status));
            const end: usize = @intCast(c.uregex_end64(regex, 0, &status));
            if (status > c.U_ZERO_ERROR or end <= begin) return error.InvalidText;
            try self.emit(normalized[position..begin], output);
            try self.emit(normalized[begin..end], output);
            position = end;
        }
        if (status > c.U_ZERO_ERROR) return error.InvalidText;
        try self.emit(normalized[position..], output);
    }

    fn emit(self: *const Tokenizer, text: []const u8, output: *c.ds4_tokens) !void {
        if (text.len > 0 and !c.ds4_bpe_emit_piece(self.tokens, self.merges, text.ptr, text.len, output)) return error.OutOfMemory;
    }
};

fn nfc(a: Allocator, text: []const u8) ![]u8 {
    var length: usize = 0;
    const normalized = c.ds4_clef_nfc(text.ptr, text.len, &length) orelse return error.NormalizeText;
    defer c.free(normalized);
    return a.dupe(u8, normalized[0..length]);
}

fn field(value: std.json.Value, key: []const u8) !std.json.Value {
    if (value != .object) return error.InvalidTokenizer;
    return value.object.get(key) orelse error.InvalidTokenizer;
}
fn string(value: std.json.Value) ![]const u8 {
    return if (value == .string) value.string else error.InvalidTokenizer;
}
fn integer(value: std.json.Value) !usize {
    return if (value == .integer and value.integer >= 0) @intCast(value.integer) else error.InvalidTokenizer;
}
fn readFile(a: Allocator, path: []const u8) ![]u8 {
    const zpath = try a.dupeSentinel(u8, path, 0);
    defer a.free(zpath);
    const file = c.fopen(zpath, "rb") orelse return error.OpenTokenizer;
    defer _ = c.fclose(file);
    if (c.fseek(file, 0, c.SEEK_END) != 0) return error.ReadTokenizer;
    const length = c.ftell(file);
    if (length < 0 or length > 64 * 1024 * 1024 or c.fseek(file, 0, c.SEEK_SET) != 0) return error.ReadTokenizer;
    const bytes = try a.alloc(u8, @intCast(length));
    if (c.fread(bytes.ptr, 1, bytes.len, file) != bytes.len) return error.ReadTokenizer;
    return bytes;
}
