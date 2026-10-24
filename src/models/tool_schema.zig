//! Small shared schema transformation used by Qwen and GLM prompt templates.
//! The standard JSON scanner checks boundaries. Source slices retain nested
//! JSON spelling, top-level field order and duplicate keys from the request.
const std = @import("std");
const Allocator = std.mem.Allocator;
const whitespace = " \t\r\n";

pub const ValueSpan = struct { raw: []const u8, consumed: usize };

/// Read one value from the server's concatenated schema stream. A malformed
/// tail stops that stream, just as the established rendering boundary does.
pub fn takeValue(allocator: Allocator, source: []const u8) Allocator.Error!?ValueSpan {
    const begin = source.len - std.mem.trimStart(u8, source, whitespace).len;
    if (begin == source.len) return null;
    var scanner = std.json.Scanner.initCompleteInput(allocator, source[begin..]);
    defer scanner.deinit();
    scanner.skipValue() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    return .{ .raw = source[begin .. begin + scanner.cursor], .consumed = begin + scanner.cursor };
}

/// Remove insignificant whitespace after a standard parser has selected a
/// JSON value. Keep escapes and number lexemes; a DOM rewrite would change
/// those bytes and could collapse duplicate keys.
pub fn minify(allocator: Allocator, source: []const u8) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var quoted = false;
    var escaped = false;
    for (source) |byte| {
        if (!quoted and std.mem.indexOfScalar(u8, whitespace, byte) != null) continue;
        try out.append(allocator, byte);
        if (escaped) escaped = false else if (quoted and byte == '\\') {
            escaped = true;
        } else if (byte == '"') quoted = !quoted;
    }
    return out.toOwnedSlice(allocator);
}

const Field = struct { name: []const u8, value: []const u8, is_string: bool };

fn stringToken(token: std.json.Token) ![]const u8 {
    return switch (token) {
        .string, .allocated_string => |value| value,
        else => error.InvalidSchema,
    };
}

fn fields(allocator: Allocator, source: []const u8) ![]const Field {
    var scanner = std.json.Scanner.initCompleteInput(allocator, source);
    defer scanner.deinit();
    if (try scanner.next() != .object_begin) return error.InvalidSchema;
    var result: std.ArrayList(Field) = .empty;
    while (true) {
        const token = try scanner.nextAllocMax(allocator, .alloc_always, source.len);
        if (token == .object_end) break;
        // The compatibility boundary historically exposed decoded JSON text
        // as C strings. Retain its first-NUL limit while owning it natively.
        const key = std.mem.sliceTo(try stringToken(token), 0);
        var begin = scanner.cursor;
        while (begin < source.len and (source[begin] == ':' or std.mem.indexOfScalar(u8, whitespace, source[begin]) != null)) begin += 1;
        try scanner.skipValue();
        const raw = source[begin..scanner.cursor];
        const is_string = raw.len != 0 and raw[0] == '"';
        const value = if (is_string) value: {
            const parsed = try std.json.parseFromSlice([]const u8, allocator, raw, .{ .allocate = .alloc_always });
            break :value std.mem.sliceTo(parsed.value, 0);
        } else try minify(allocator, raw);
        try result.append(allocator, .{ .name = key, .value = value, .is_string = is_string });
    }
    return result.toOwnedSlice(allocator);
}

fn quote(out: *std.ArrayList(u8), allocator: Allocator, source: []const u8) !void {
    const hex = "0123456789abcdef";
    try out.append(allocator, '"');
    for (source) |byte| switch (byte) {
        '"', '\\' => try out.appendSlice(allocator, &.{ '\\', byte }),
        '\n' => try out.appendSlice(allocator, "\\n"),
        '\r' => try out.appendSlice(allocator, "\\r"),
        '\t' => try out.appendSlice(allocator, "\\t"),
        0...8, 11, 12, 14...31 => try out.appendSlice(allocator, &.{ '\\', 'u', '0', '0', hex[byte >> 4], hex[byte & 15] }),
        else => try out.append(allocator, byte),
    };
    try out.append(allocator, '"');
}

/// Return null for deferred tools. Malformed/non-object values keep the raw
/// fallback. Only the outer object gets ordinary separators; nested values
/// retain their established compact spelling.
pub fn renderOne(allocator: Allocator, source: []const u8) Allocator.Error!?[:0]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const parsed = fields(arena.allocator(), source) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return try allocator.dupeSentinel(u8, source, 0),
    };
    for (parsed) |field| {
        if (std.mem.eql(u8, field.name, "defer_loading") and !field.is_string and std.mem.eql(u8, field.value, "true")) return null;
    }
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.append(allocator, '{');
    var wrote = false;
    for (parsed) |field| {
        if (std.mem.eql(u8, field.name, "strict") or std.mem.eql(u8, field.name, "defer_loading")) continue;
        if (wrote) try out.appendSlice(allocator, ", ");
        try quote(&out, allocator, field.name);
        try out.appendSlice(allocator, ": ");
        if (field.is_string) try quote(&out, allocator, field.value) else try out.appendSlice(allocator, field.value);
        wrote = true;
    }
    try out.append(allocator, '}');
    return try allocator.dupeSentinel(u8, out.items, 0);
}

test "schema transformation retains duplicate fields and raw JSON spelling" {
    const allocator = std.testing.allocator;
    const source = "{\"name\":\"read\",\"strict\":true,\"x\":1.00e+02,\"x\":{ \"a\": [1, 2] },\"description\":\"a\\b\"}";
    const rendered = (try renderOne(allocator, source)).?;
    defer allocator.free(rendered);
    try std.testing.expectEqualStrings("{\"name\": \"read\", \"x\": 1.00e+02, \"x\": {\"a\":[1,2]}, \"description\": \"a\\u0008\"}", rendered);
    try std.testing.expect(try renderOne(allocator, "{\"defer_loading\":true,\"name\":\"later\"}") == null);
    const first = (try takeValue(allocator, " \n{\"name\":\"a\"}\n{\"name\":\"b\"}")).?;
    try std.testing.expectEqualStrings("{\"name\":\"a\"}", first.raw);
    const fallback = (try renderOne(allocator, "[1, 2]")).?;
    defer allocator.free(fallback);
    try std.testing.expectEqualStrings("[1, 2]", fallback);
}
