//! Isolate the native head using saved reference features, without backbone inference.
const std = @import("std");
const c = @import("clef_tokenizer_c");
const Backend = @import("models/clef/backend.zig");
const Head = @import("models/clef/head.zig").Head;
const Tokenizer = @import("models/clef/tokenizer.zig").Tokenizer;
const Record = @import("models/clef/record.zig");
const Wire = @import("server/decisions.zig");
const a = std.heap.c_allocator;

fn readFile(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const zpath = try allocator.dupeSentinel(u8, path, 0);
    defer allocator.free(zpath);
    const file = c.fopen(zpath, "rb") orelse return error.OpenOracle;
    defer _ = c.fclose(file);
    if (c.fseek(file, 0, c.SEEK_END) != 0) return error.ReadOracle;
    const length = c.ftell(file);
    if (length < 0 or c.fseek(file, 0, c.SEEK_SET) != 0) return error.ReadOracle;
    const bytes = try allocator.alignedAlloc(u8, .of(f32), @intCast(length));
    if (c.fread(bytes.ptr, 1, bytes.len, file) != bytes.len) return error.ReadOracle;
    return bytes;
}
fn joined(allocator: std.mem.Allocator, directory: []const u8, name: []const u8) ![]const u8 {
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ directory, name });
}
fn probability(value: std.json.Value) !f64 {
    return switch (value) {
        .float => value.float,
        .integer => @floatFromInt(value.integer),
        else => error.InvalidOracleProbability,
    };
}
fn one(model: *Backend.Engine, head: *const Head, tokenizer: *const Tokenizer, directory: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const allocator = arena.allocator();
    const body = try readFile(allocator, try joined(allocator, directory, "record.json"));
    const parsed = try Wire.parse(allocator, body, .systemone);
    var record = try Record.encode(tokenizer, allocator, parsed.request, 16384);
    defer record.deinit();
    const hidden_bytes = try readFile(allocator, try joined(allocator, directory, "hidden.f32"));
    const lexical_bytes = try readFile(allocator, try joined(allocator, directory, "lexical.f32"));
    const hidden: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, hidden_bytes));
    const lexical: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, lexical_bytes));
    if (hidden.len != record.ids.len * model.config.H) return error.HiddenShapeMismatch;
    const use_gpu = if (c.getenv("DS4_CLEF_HEAD_GPU")) |setting| setting[0] != 0 and setting[0] != '0' else false;
    const result = if (use_gpu)
        try head.runPrepared(allocator, &record, try model.headFeatures(hidden), lexical)
    else
        try head.run(allocator, &record, hidden, lexical);
    const expected_bytes = try readFile(allocator, try joined(allocator, directory, "probabilities.json"));
    const expected = try std.json.parseFromSlice(std.json.Value, allocator, expected_bytes, .{});
    var max_gap: f64 = 0;
    var kl: f64 = 0;
    var choices_equal: usize = 0;
    for (parsed.request.questions, result.answers) |question, answer| {
        const target = expected.value.object.get(question.id) orelse return error.MissingOracleQuestion;
        var total: f64 = 0;
        var best_native: usize = 0;
        var best_reference: usize = 0;
        var reference_max: f64 = -1;
        for (question.options, answer.probabilities, 0..) |option, p, i| {
            if (!std.math.isFinite(p) or p < 0) return error.InvalidNativeProbability;
            const q = try probability(target.object.get(option.id) orelse return error.MissingOracleOption);
            max_gap = @max(max_gap, @abs(q - p));
            if (q > 0) kl += q * @log(q / @max(p, 1e-30));
            total += p;
            if (p > answer.probabilities[best_native]) best_native = i;
            if (q > reference_max) {
                reference_max = q;
                best_reference = i;
            }
        }
        if (@abs(total - 1) > 1e-5) return error.UnnormalizedNativeProbability;
        if (best_native == best_reference) choices_equal += 1;
    }
    const json = try Wire.serialize(allocator, parsed, result, .systemone);
    const line = try allocator.dupeSentinel(u8, json, 0);
    _ = c.puts(line);
    std.debug.print("head oracle {s}: max probability gap {d:.8}, KL {d:.8}, choices {d}/{d}\n", .{ directory, max_gap, kl, choices_equal, result.answers.len });
}
fn run(argc: c_int, argv: [*c][*c]u8) !void {
    const directory = std.mem.span(argv[1]);
    const model = try Backend.Engine.open(directory);
    defer model.close();
    var head = try Head.open(a, model);
    defer head.deinit();
    const tokenizer_path = try joined(a, directory, "tokenizer.json");
    defer a.free(tokenizer_path);
    var tokenizer = try Tokenizer.open(a, tokenizer_path);
    defer tokenizer.deinit();
    for (2..@intCast(argc)) |i| try one(model, &head, &tokenizer, std.mem.span(argv[i]));
}
export fn main(argc: c_int, argv: [*c][*c]u8) c_int {
    if (argc < 3) {
        std.debug.print("Usage: test-clef-head MODEL_DIR ORACLE_CASE_DIR...\n", .{});
        return 2;
    }
    run(argc, argv) catch |err| {
        std.debug.print("Clef head oracle: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}
