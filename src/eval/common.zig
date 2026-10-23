const std = @import("std");
const c = @import("eval_c");
pub fn text(value: [*c]const u8) []const u8 {
    return if (value == null) &.{} else std.mem.span(value);
}
pub fn output(pointer: [*c]u8, length: usize) []u8 {
    return if (length == 0) &.{} else pointer[0..length];
}
pub fn copy(target: []u8, value: []const u8) void {
    if (target.len == 0) return;
    const length = @min(target.len - 1, value.len);
    @memcpy(target[0..length], value[0..length]);
    target[length] = 0;
}
pub fn written(value: []const u8) []const u8 {
    const end = std.mem.findScalar(u8, value, 0) orelse value.len;
    return value[0..end];
}
