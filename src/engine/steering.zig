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
