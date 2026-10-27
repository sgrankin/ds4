//! One optional DSP1 trailer after the unchanged canonical DSV4 payload.
//! Three chronological BF16 target-feature arrays, not projected draft KV.
const std = @import("std");

pub const magic: u32 = 0x31505344;
pub const version: u32 = 1;
pub const header_bytes: usize = 32;
pub const Header = [8]u32;

pub fn rows(position: u32) u32 {
    return @min(position, 128);
}
pub fn wireBytes(position: u32, width: u32) u64 {
    return header_bytes + @as(u64, 3) * rows(position) * width * 2;
}
pub fn header(position: u32, width: u32, layers: [3]u32) Header {
    return .{ magic, version, position, rows(position), width, layers[0], layers[1], layers[2] };
}
pub fn validHeader(value: Header, position: u32, width: u32) bool {
    return position > 0 and width > 0 and value[0] == magic and value[1] == version and
        value[2] == position and value[3] == rows(position) and value[4] == width;
}
pub fn validBytes(bytes: []const u8, position: u32, width: u32) bool {
    if (bytes.len != wireBytes(position, width)) return false;
    var value: Header = undefined;
    for (&value, 0..) |*word, index|
        word.* = std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little);
    return validHeader(value, position, width);
}

test "DSP1 describes one exact absolute target-feature tail" {
    const value = header(257, 5120, .{ 37, 38, 39 });
    try std.testing.expect(validHeader(value, 257, 5120));
    try std.testing.expect(!validHeader(value, 258, 5120));
    try std.testing.expect(!validHeader(value, 257, 4096));
    try std.testing.expectEqual(@as(u64, 3_932_192), wireBytes(257, 5120));
    try std.testing.expectEqual(@as(u32, 127), rows(127));
}

test "DSP1 wire size and little-endian header agree with native cache admission" {
    const a = std.testing.allocator;
    const bytes = try a.alloc(u8, @intCast(wireBytes(1, 5120)));
    defer a.free(bytes);
    @memset(bytes, 0);
    const value = header(1, 5120, .{ 37, 38, 39 });
    for (value, 0..) |word, index|
        std.mem.writeInt(u32, bytes[index * 4 ..][0..4], word, .little);
    try std.testing.expect(validBytes(bytes, 1, 5120));
    try std.testing.expect(!validBytes(bytes[0 .. bytes.len - 1], 1, 5120));
}
