//! CPU-only token/span parity against pinned Cloudflare encode_record fixtures.
const std = @import("std");
const c = @import("clef_tokenizer_c");
const Tokenizer = @import("models/clef/tokenizer.zig").Tokenizer;
const Record = @import("models/clef/record.zig");
const Wire = @import("server/decisions.zig");
const a = std.heap.c_allocator;

fn readFile(path: [*:0]const u8) ![]u8 {
    const file = c.fopen(path, "rb") orelse return error.OpenFixture;
    defer _ = c.fclose(file);
    if (c.fseek(file, 0, c.SEEK_END) != 0) return error.ReadFixture;
    const length = c.ftell(file);
    if (length < 0 or c.fseek(file, 0, c.SEEK_SET) != 0) return error.ReadFixture;
    const bytes = try a.alloc(u8, @intCast(length));
    errdefer a.free(bytes);
    if (c.fread(bytes.ptr, 1, bytes.len, file) != bytes.len) return error.ReadFixture;
    return bytes;
}
fn equal(actual: usize, expected: std.json.Value) !void {
    if (expected != .integer or actual != expected.integer) return error.RecordMismatch;
}
fn span(actual: anytype, expected: std.json.Value) !void {
    try equal(actual.start, expected.array.items[0]);
    try equal(actual.end, expected.array.items[1]);
}
fn run(tokenizer_path: [*:0]const u8, fixtures_path: [*:0]const u8) !void {
    var tokenizer = try Tokenizer.open(a, std.mem.span(tokenizer_path));
    defer tokenizer.deinit();
    const bytes = try readFile(fixtures_path);
    defer a.free(bytes);
    const fixtures = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer fixtures.deinit();
    const cases = fixtures.value.array.items;
    var token_count: usize = 0;
    for (cases, 0..) |fixture, index| {
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const allocator = arena.allocator();
        const request = try std.json.Stringify.valueAlloc(allocator, fixture.object.get("request").?, .{});
        const parsed = try Wire.parse(allocator, request, .systemone);
        const media: []const Record.MediaSpec = if (fixture.object.get("media_specs")) |specs|
            (try std.json.parseFromValue([]Record.MediaSpec, allocator, specs, .{})).value
        else
            &.{};
        const omni = if (fixture.object.get("omni")) |flag| flag.bool else false;
        var encoded = try Record.encodeMedia(&tokenizer, allocator, parsed.request, 16384, omni, media);
        defer encoded.deinit();
        const ids = fixture.object.get("input_ids").?.array.items;
        if (ids.len != encoded.ids.len) {
            std.debug.print("case {d}: expected {d} tokens, got {d}\n", .{ index, ids.len, encoded.ids.len });
            return error.TokenCountMismatch;
        }
        for (encoded.ids, ids, 0..) |actual, expected, ti| {
            equal(actual, expected) catch {
                std.debug.print("case {d} token {d}: expected {d}, got {d}\n", .{ index, ti, expected.integer, actual });
                return error.TokenMismatch;
            };
        }
        const questions = fixture.object.get("questions").?.array.items;
        if (questions.len != encoded.questions.len) return error.QuestionCountMismatch;
        for (encoded.questions, questions) |question, expected| {
            try equal(@backingInt(question.kind), expected.object.get("question_type").?);
            try span(question.span, expected.object.get("question_span").?);
            const options = expected.object.get("option_spans").?.array.items;
            const option_ids = expected.object.get("option_ids").?.array.items;
            if (options.len != question.options.len) return error.OptionCountMismatch;
            for (question.options, options, option_ids) |option, expected_span, id| {
                try span(option.span, expected_span);
                if (!std.mem.eql(u8, option.id, id.string)) return error.OptionIdMismatch;
                if (!std.mem.eql(u32, encoded.ids[option.span.start..option.span.end], encoded.embedding_ids[option.lexical_span.start..option.lexical_span.end])) return error.LexicalSpanMismatch;
            }
        }
        if (fixture.object.get("positions")) |expected| {
            if (expected.array.items.len != encoded.positions.len) return error.PositionCountMismatch;
            for (encoded.positions, expected.array.items) |actual, target| try equal(actual, target);
        } else if (media.len == 0) {
            for (encoded.positions, 0..) |actual, i| if (actual != i / 3) return error.TextPositionMismatch;
        }
        if (fixture.object.get("media_rows")) |expected| {
            if (expected.array.items.len != encoded.media_rows.len) return error.MediaCountMismatch;
            for (encoded.media_rows, expected.array.items) |rows, target| {
                if (rows.rows.len != target.array.items.len) return error.MediaRowCountMismatch;
                for (rows.rows, target.array.items) |actual, value| try equal(actual, value);
            }
        }
        _ = Record.encodeMedia(&tokenizer, allocator, parsed.request, encoded.ids.len - 1, omni, media) catch |err| {
            if (err != error.ContextTooLong) return err;
            token_count += ids.len;
            continue;
        };
        return error.OverlengthWasNotRejected;
    }
    std.debug.print("Clef record parity: {d} cases, {d} exact tokens, spans/options/lexical rows and overlength checks pass.\n", .{ cases.len, token_count });
}
export fn main(argc: c_int, argv: [*c][*c]u8) c_int {
    if (argc != 3) {
        std.debug.print("Usage: test-clef-record TOKENIZER.json FIXTURES.json\n", .{});
        return 2;
    }
    run(@ptrCast(argv[1]), @ptrCast(argv[2])) catch |err| {
        std.debug.print("Clef record: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}
