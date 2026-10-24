//! A bank has one hidden-width residual direction per trunk layer. Zero rows
//! leave a layer unselected. Projection uses unit rows; addition retains the
//! learned magnitude and distributes its strength across selected layers.
const std = @import("std");

pub fn effectiveScale(raw: f32, mode: c_int, selected: u32) f32 {
    return if (mode == 1) raw / @as(f32, @floatFromInt(@max(selected, 1))) else raw;
}

pub fn prepare(values: []f32, width: usize, mode: c_int) ?u32 {
    if (width == 0 or values.len % width != 0 or (mode != 0 and mode != 1)) return null;
    var selected: u32 = 0;
    var offset: usize = 0;
    while (offset < values.len) : (offset += width) {
        const row = values[offset..][0..width];
        var square: f64 = 0;
        for (row) |value| {
            if (!std.math.isFinite(value)) return null;
            square += @as(f64, value) * value;
        }
        if (square == 0) continue;
        selected += 1;
        if (mode == 0) {
            const norm = @sqrt(square);
            for (row) |*value| value.* = @floatCast(@as(f64, value.*) / norm);
        }
    }
    return selected;
}

test "project banks normalize selected rows and additive banks preserve magnitude" {
    const original = [_]f32{ 3, 4, 0, 0, 0, -2 };
    var project = original;
    try std.testing.expectEqual(@as(u32, 2), prepare(&project, 2, 0).?);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), project[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), project[1], 1e-6);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, -1 }, project[2..]);

    var add = original;
    try std.testing.expectEqual(@as(u32, 2), prepare(&add, 2, 1).?);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&original), std.mem.sliceAsBytes(&add));
}
