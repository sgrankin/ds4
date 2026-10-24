//! Keep bank preparation independent of model/GPU owners for host fixtures.
export fn ds4_steering_prepare_f32(values: [*]f32, layers: u32, width: u32, mode: c_int) c_int {
    const selected = @import("steering.zig").prepare(values[0 .. @as(usize, layers) * width], width, mode) orelse return -1;
    return @intCast(selected);
}
