//! OpenAI-compatible cache controls are request metadata, never prompt text.
const std = @import("std");
const allocator = std.heap.c_allocator;
pub const Error = error{ InvalidCacheControl, UnsupportedCacheControl };

pub fn breakpoint(raw: []const u8) Error!bool {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, raw, .{
        .duplicate_field_behavior = .use_last,
    }) catch return error.InvalidCacheControl;
    defer parsed.deinit();
    if (parsed.value == .null) return false;
    if (parsed.value != .object) return error.InvalidCacheControl;
    const fields = parsed.value.object;
    const mode = fields.get("mode") orelse return error.InvalidCacheControl;
    if (mode != .string or !std.mem.eql(u8, mode.string, "explicit")) return error.InvalidCacheControl;
    for (fields.keys()) |key| {
        if (!std.mem.eql(u8, key, "mode")) return error.UnsupportedCacheControl;
    }
    return true;
}

pub fn explicitMode(raw: []const u8) Error!bool {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, raw, .{
        .duplicate_field_behavior = .use_last,
    }) catch return error.InvalidCacheControl;
    defer parsed.deinit();
    if (parsed.value == .null) return false;
    if (parsed.value != .object) return error.InvalidCacheControl;
    var result = false;
    var fields = parsed.value.object.iterator();
    while (fields.next()) |field| {
        if (!std.mem.eql(u8, field.key_ptr.*, "mode")) return error.UnsupportedCacheControl;
        const mode = field.value_ptr.*;
        if (mode != .string) return error.InvalidCacheControl;
        if (std.mem.eql(u8, mode.string, "explicit")) result = true else if (!std.mem.eql(u8, mode.string, "implicit")) return error.InvalidCacheControl;
    }
    return result;
}

test "cache metadata parses explicit markers and mode defaults" {
    try std.testing.expect(try breakpoint("{\"mode\":\"explicit\"}"));
    try std.testing.expect(!try breakpoint("null"));
    try std.testing.expect(try explicitMode("{\"mode\":\"explicit\"}"));
    try std.testing.expect(!try explicitMode("{\"mode\":\"implicit\"}"));
    try std.testing.expect(!try explicitMode("{}"));
    try std.testing.expect(!try explicitMode("null"));
    try std.testing.expectError(error.InvalidCacheControl, breakpoint("{}"));
    try std.testing.expectError(error.UnsupportedCacheControl, breakpoint("{\"mode\":\"explicit\",\"ttl\":\"12h\"}"));
    try std.testing.expectError(error.UnsupportedCacheControl, explicitMode("{\"prewarm\":true}"));
}
