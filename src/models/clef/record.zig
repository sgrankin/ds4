//! Cloudflare's STATE → SCHEMA FIELDS → JOINT SCHEMA DECISIONS record.
const std = @import("std");
const types = @import("types.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const Span = types.Span;

pub const EncodedOption = struct {
    id: []const u8,
    span: Span,
    lexical_span: Span,
    source_index: usize,
};
pub const EncodedQuestion = struct {
    kind: types.Kind,
    span: Span,
    options: []EncodedOption,
};
pub const MediaSpec = struct {
    kind: types.MediaKind,
    token_count: usize,
    grid: [3]u32 = .{ 0, 0, 0 },
    timestamps: []const f64 = &.{},
    label: ?[]const u8 = null,
    /// An audiovisual video's audio feature entry in this same spec list.
    soundtrack: ?usize = null,
};
pub const MediaRows = struct { kind: types.MediaKind, rows: []u32 };
const PositionToken = struct { row: usize, relative: [3]u32 };
const PositionGroup = struct { tokens: []const PositionToken };

pub const Record = struct {
    arena: std.heap.ArenaAllocator,
    ids: []u32,
    schema_start: usize,
    questions: []EncodedQuestion,
    embedding_ids: []u32,
    positions: []u32,
    media_rows: []MediaRows,
    pub fn deinit(self: *Record) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

const system_prompt = "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.";
const prefix_text = "<|im_start|>system\n" ++ system_prompt ++ "<|im_end|>\n<|im_start|>user\nSTATE:\n";
const suffix_text = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:";
const default_noul = [_]types.Option{
    .{ .id = "true", .description = .{ .string = "The proposition is true or the answer is yes." }, .value = .{ .bool = true } },
    .{ .id = "false", .description = .{ .string = "The proposition is false or the answer is no." }, .value = .{ .bool = false } },
};

pub fn encode(tokenizer: *const Tokenizer, allocator: Allocator, request: types.Request, max_tokens: usize) !Record {
    if (request.media.len > 0) return error.MediaNotPrepared;
    return encodeMedia(tokenizer, allocator, request, max_tokens, false, &.{});
}

pub fn encodeMedia(tokenizer: *const Tokenizer, allocator: Allocator, request: types.Request, max_tokens: usize, omni: bool, media: []const MediaSpec) !Record {
    if (request.questions.len == 0) return error.NoQuestions;
    if (request.media.len > 0 and media.len == 0) return error.MediaNotPrepared;
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    var schema = Tokens{ .a = a, .tokenizer = tokenizer };
    try schema.append("\n\nSCHEMA FIELDS:\n");
    const questions = try a.alloc(EncodedQuestion, request.questions.len);
    for (request.questions, questions, 0..) |question, *encoded, qi| {
        const header = try std.fmt.allocPrint(a, "\nFIELD {d}\nID: {s}\nTYPE: {s}\nINSTRUCTION: ", .{ qi + 1, question.id, @tagName(question.kind) });
        try schema.append(header);
        const qstart = schema.ids.items.len;
        const ins = if (question.instructions == .null or (question.instructions == .string and question.instructions.string.len == 0))
            question.id
        else
            try render(a, question.instructions);
        try schema.append(ins);
        if (qstart == schema.ids.items.len) return error.EmptyQuestion;
        encoded.* = .{ .kind = question.kind, .span = .{ .start = qstart, .end = schema.ids.items.len }, .options = undefined };
        try schema.append("\nALLOWED OPTIONS:\n");
        const options: []const types.Option = if (question.kind == .noul and question.options.len == 0) &default_noul else question.options;
        if (options.len == 0) return error.NoOptions;
        if (question.kind == .noul and options.len != 2) return error.InvalidNoulOptions;
        const order = try a.alloc(usize, options.len);
        for (order, 0..) |*index, i| index.* = i;
        if (question.kind == .choice) std.mem.sort(usize, order, options, struct {
            fn less(opts: []const types.Option, lhs: usize, rhs: usize) bool {
                return std.mem.lessThan(u8, opts[lhs].id, opts[rhs].id);
            }
        }.less);
        if (question.kind == .noul and !std.mem.eql(u8, options[order[0]].id, "true")) std.mem.swap(usize, &order[0], &order[1]);
        encoded.options = try a.alloc(EncodedOption, options.len);
        for (order, encoded.options, 0..) |source_index, *option, oi| {
            const input = options[source_index];
            const id = if (question.kind == .score) try std.fmt.allocPrint(a, "{d}", .{oi}) else input.id;
            try schema.append(try std.fmt.allocPrint(a, "OPTION {d}: ", .{oi + 1}));
            const start = schema.ids.items.len;
            var semantics: std.json.ObjectMap = .empty;
            if (input.description != .null) try semantics.put(a, "description", input.description);
            try semantics.put(a, "option_id", .{ .string = id });
            try schema.append(try render(a, .{ .object = semantics }));
            option.* = .{ .id = id, .span = .{ .start = start, .end = schema.ids.items.len }, .lexical_span = undefined, .source_index = source_index };
            try schema.append("\n");
        }
        try schema.append("END FIELD\n");
        if (schema.ids.items.len > max_tokens) return error.ContextTooLong;
    }
    var prefix_tokens = Tokens{ .a = a, .tokenizer = tokenizer };
    try prefix_tokens.append(prefix_text);
    var placement = try MediaPlacement.init(a, &prefix_tokens, media, omni);
    try placement.emit();
    const prefix = prefix_tokens.ids.items;
    const state = try tokenizer.encode(a, try render(a, request.state));
    const suffix = try tokenizer.encode(a, suffix_text);
    const total = prefix.len + state.len + schema.ids.items.len + suffix.len;
    if (total > max_tokens) return error.ContextTooLong;
    const ids = try a.alloc(u32, total);
    const offset = prefix.len + state.len;
    for (prefix, ids[0..prefix.len]) |id, *out| out.* = @intCast(id);
    for (state, ids[prefix.len..][0..state.len]) |id, *out| out.* = @intCast(id);
    for (schema.ids.items, ids[offset..][0..schema.ids.items.len]) |id, *out| out.* = @intCast(id);
    for (suffix, ids[offset + schema.ids.items.len ..]) |id, *out| out.* = @intCast(id);
    var lexical: std.ArrayList(u32) = .empty;
    for (questions) |*question| {
        question.span.start += offset;
        question.span.end += offset;
        for (question.options) |*option| {
            option.span.start += offset;
            option.span.end += offset;
            const start = lexical.items.len;
            try lexical.appendSlice(a, ids[option.span.start..option.span.end]);
            option.lexical_span = .{ .start = start, .end = lexical.items.len };
        }
    }
    return .{ .arena = arena, .ids = ids, .schema_start = offset, .questions = questions, .embedding_ids = try lexical.toOwnedSlice(a), .positions = try placement.positions(total), .media_rows = placement.rows };
}

const Tokens = struct {
    a: Allocator,
    tokenizer: *const Tokenizer,
    ids: std.ArrayList(i32) = .empty,
    fn append(self: *Tokens, text: []const u8) !void {
        const ids = try self.tokenizer.encode(self.a, text);
        try self.ids.appendSlice(self.a, ids);
        self.a.free(ids);
    }
};

/// Python render(): strings verbatim, other JSON compact with sorted keys.
/// Container ordering is model policy; leaf escaping uses the standard parser.
pub fn render(a: Allocator, value: Value) ![]const u8 {
    if (value == .string) return value.string;
    var text: std.ArrayList(u8) = .empty;
    try compact(a, &text, value);
    return text.toOwnedSlice(a);
}
fn compact(a: Allocator, out: *std.ArrayList(u8), value: Value) !void {
    switch (value) {
        .object => |object| {
            const keys = try a.dupe([]const u8, object.keys());
            defer a.free(keys);
            std.mem.sort([]const u8, keys, {}, struct {
                fn less(_: void, x: []const u8, y: []const u8) bool {
                    return std.mem.lessThan(u8, x, y);
                }
            }.less);
            try out.append(a, '{');
            for (keys, 0..) |key, index| {
                if (index > 0) try out.append(a, ',');
                try compact(a, out, .{ .string = key });
                try out.append(a, ':');
                try compact(a, out, object.get(key).?);
            }
            try out.append(a, '}');
        },
        .array => |array| {
            try out.append(a, '[');
            for (array.items, 0..) |item, index| {
                if (index > 0) try out.append(a, ',');
                try compact(a, out, item);
            }
            try out.append(a, ']');
        },
        .float => |number| {
            if (!std.math.isFinite(number)) return error.InvalidJsonNumber;
            var buffer: [400]u8 = undefined;
            const scientific = @abs(number) >= 1e16 or (number != 0 and @abs(number) < 1e-4);
            const formatted = try std.fmt.float.render(&buffer, number, .{ .mode = if (scientific) .scientific else .decimal });
            if (scientific) {
                const exponent = std.mem.indexOfScalar(u8, formatted, 'e').?;
                try out.appendSlice(a, formatted[0 .. exponent + 1]);
                var digits = formatted[exponent + 1 ..];
                if (digits[0] == '-') {
                    try out.append(a, '-');
                    digits = digits[1..];
                } else try out.append(a, '+');
                if (digits.len == 1) try out.append(a, '0');
                try out.appendSlice(a, digits);
            } else {
                try out.appendSlice(a, formatted);
                if (std.mem.indexOfScalar(u8, formatted, '.') == null) try out.appendSlice(a, ".0");
            }
        },
        else => {
            const bytes = try std.json.Stringify.valueAlloc(a, value, .{});
            defer a.free(bytes);
            try out.appendSlice(a, bytes);
        },
    }
}

test "structured model text uses compact sorted JSON and Unicode" {
    const a = std.testing.allocator;
    const value = try std.json.parseFromSlice(Value, a, "{\"z\":[1,true],\"a\":\"顧客\"}", .{});
    defer value.deinit();
    const text = try render(a, value.value);
    defer a.free(text);
    try std.testing.expectEqualStrings("{\"a\":\"顧客\",\"z\":[1,true]}", text);
}

const MediaPlacement = struct {
    a: Allocator,
    prefix: *Tokens,
    specs: []const MediaSpec,
    omni: bool,
    rows: []MediaRows,
    written: []usize,
    groups: std.ArrayList(PositionGroup) = .empty,

    fn init(a: Allocator, prefix: *Tokens, specs: []const MediaSpec, omni: bool) !MediaPlacement {
        const rows = try a.alloc(MediaRows, specs.len);
        const written = try a.alloc(usize, specs.len);
        @memset(written, 0);
        for (specs, rows) |spec, *out| {
            if (spec.token_count == 0) return error.EmptyMedia;
            if (spec.kind != .audio and (spec.grid[0] == 0 or spec.grid[1] % 2 != 0 or spec.grid[2] % 2 != 0 or
                @as(usize, spec.grid[0]) * spec.grid[1] * spec.grid[2] / 4 != spec.token_count)) return error.InvalidMediaGrid;
            if (!omni and spec.kind == .audio) return error.UnsupportedAudio;
            out.* = .{ .kind = spec.kind, .rows = try a.alloc(u32, spec.token_count) };
        }
        return .{ .a = a, .prefix = prefix, .specs = specs, .omni = omni, .rows = rows, .written = written };
    }
    fn control(self: *MediaPlacement, spelling: []const u8) !void {
        try self.prefix.ids.append(self.a, @intCast(try self.prefix.tokenizer.specialId(spelling)));
    }
    fn label(self: *MediaPlacement, spec: MediaSpec) !void {
        if (spec.label) |name| try self.prefix.append(try std.fmt.allocPrint(self.a, "{s}:\n", .{name}));
    }
    fn pad(self: *MediaPlacement, source: usize, spelling: []const u8, relative: [3]u32, group: *std.ArrayList(PositionToken)) !void {
        const row = self.prefix.ids.items.len;
        const index = self.written[source];
        if (index >= self.rows[source].rows.len) return error.MediaPlacementMismatch;
        self.rows[source].rows[index] = @intCast(row);
        self.written[source] += 1;
        try group.append(self.a, .{ .row = row, .relative = relative });
        try self.control(spelling);
    }
    fn visualFrame(self: *MediaPlacement, source: usize, frame: u32, group: *std.ArrayList(PositionToken)) !void {
        const spec = self.specs[source];
        const time = if (self.omni and spec.kind == .video) frame * 13 else 0;
        for (0..spec.grid[1] / 2) |y| for (0..spec.grid[2] / 2) |x| {
            try self.pad(source, if (spec.kind == .video) "<|video_pad|>" else "<|image_pad|>", .{ time, @intCast(y), @intCast(x) }, group);
        };
    }
    fn finishGroup(self: *MediaPlacement, group: *std.ArrayList(PositionToken)) !void {
        try self.groups.append(self.a, .{ .tokens = try group.toOwnedSlice(self.a) });
    }
    fn linked(self: *const MediaPlacement, source: usize) bool {
        for (self.specs) |spec| if (spec.soundtrack == source) return true;
        return false;
    }
    fn emit(self: *MediaPlacement) !void {
        // This is the published ordering: still images, standalone audio, videos.
        for (self.specs, 0..) |spec, i| if (spec.kind == .image) {
            try self.label(spec);
            try self.control("<|vision_start|>");
            var group: std.ArrayList(PositionToken) = .empty;
            try self.visualFrame(i, 0, &group);
            try self.finishGroup(&group);
            try self.control("<|vision_end|>");
            if (self.omni) try self.prefix.append("\n");
        };
        for (self.specs, 0..) |spec, i| if (spec.kind == .audio and !self.linked(i)) {
            try self.label(spec);
            try self.control("<|audio_start|>");
            var group: std.ArrayList(PositionToken) = .empty;
            for (0..spec.token_count) |j| try self.pad(i, "<|audio_pad|>", @splat(@intCast(j)), &group);
            try self.finishGroup(&group);
            try self.control("<|audio_end|>");
            try self.prefix.append("\n");
        };
        for (self.specs, 0..) |spec, i| if (spec.kind == .video) {
            try self.label(spec);
            try self.control("<|vision_start|>");
            if (!self.omni) {
                if (spec.soundtrack != null) return error.UnsupportedAudio;
                if (spec.timestamps.len != spec.grid[0]) return error.MissingVideoTimestamps;
                for (0..spec.grid[0]) |frame| {
                    try self.prefix.append(try timestamp(self.a, spec.timestamps[frame]));
                    try self.control("<|vision_start|>");
                    var group: std.ArrayList(PositionToken) = .empty;
                    try self.visualFrame(i, @intCast(frame), &group);
                    try self.finishGroup(&group);
                    try self.control("<|vision_end|>");
                }
            } else {
                var group: std.ArrayList(PositionToken) = .empty;
                var audio_index: usize = 0;
                if (spec.soundtrack) |audio| {
                    if (audio >= self.specs.len or self.specs[audio].kind != .audio) return error.InvalidSoundtrack;
                    try self.control("<|audio_start|>");
                }
                for (0..spec.grid[0]) |frame| {
                    if (spec.soundtrack) |audio| while (audio_index < self.specs[audio].token_count and audio_index < frame * 13) : (audio_index += 1)
                        try self.pad(audio, "<|audio_pad|>", @splat(@intCast(audio_index)), &group);
                    try self.visualFrame(i, @intCast(frame), &group);
                }
                if (spec.soundtrack) |audio| {
                    while (audio_index < self.specs[audio].token_count) : (audio_index += 1)
                        try self.pad(audio, "<|audio_pad|>", @splat(@intCast(audio_index)), &group);
                }
                try self.finishGroup(&group);
                if (spec.soundtrack != null) try self.control("<|audio_end|>");
            }
            try self.control("<|vision_end|>");
            if (self.omni) try self.prefix.append("\n");
        };
        if (!self.omni and self.specs.len > 0) try self.prefix.append("\n");
        for (self.written, self.specs) |n, spec| if (n != spec.token_count) return error.MediaPlacementMismatch;
    }
    fn positions(self: *const MediaPlacement, length: usize) ![]u32 {
        const out = try self.a.alloc(u32, length * 3);
        var row: usize = 0;
        var current: u32 = 0;
        for (self.groups.items) |group| {
            const first = group.tokens[0].row;
            while (row < first) : (row += 1) {
                @memset(out[row * 3 ..][0..3], current);
                current += 1;
            }
            const base = current;
            for (group.tokens) |token| {
                if (row != token.row) return error.MediaPlacementMismatch;
                for (token.relative, out[row * 3 ..][0..3]) |relative, *value| value.* = base + relative;
                current = base + @max(@max(token.relative[0], token.relative[1]), token.relative[2]) + 1;
                row += 1;
            }
        }
        while (row < length) : (row += 1) {
            @memset(out[row * 3 ..][0..3], current);
            current += 1;
        }
        return out;
    }
};

// Python's .1f and libc use ties-to-even; Zig's precision formatter rounds
// .25 to .3, which changes actual video prompt tokens at ordinary 2-FPS times.
fn timestamp(a: Allocator, seconds: f64) ![]const u8 {
    const c = @import("clef_tokenizer_c");
    var buffer: [64]u8 = undefined;
    const count = c.snprintf(&buffer, buffer.len, "<%.1f seconds>", seconds);
    if (count < 0 or count >= buffer.len) return error.InvalidVideoTimestamp;
    return a.dupe(u8, buffer[0..@intCast(count)]);
}
