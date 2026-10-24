//! Optional hidden-width residual capture. HC streams share one direction, so
//! extraction uses their mean. No allocation/readback occurs without a match.
const std = @import("std");
const c = @import("gguf_c");

pub fn capture(tensor: ?*const c.ds4_gpu_tensor, layer: u32, rows: u32, width: u32, streams: u32, label: [*:0]const u8, position: u32) bool {
    const prefix = c.ds4_native_graph_debug_prefix(label, layer, position);
    if (prefix == null) return true;
    if (tensor == null or rows == 0 or width == 0 or streams == 0 or c.ds4_gpu_synchronize() == 0) return false;
    defer _ = c.ds4_gpu_begin_commands();
    const allocator = std.heap.c_allocator;
    const count = @as(usize, width) * streams;
    const values = allocator.alloc(f32, count) catch return false;
    defer allocator.free(values);
    const mean = allocator.alloc(f32, width) catch return false;
    defer allocator.free(mean);
    if (c.ds4_gpu_tensor_read(tensor, @as(u64, rows - 1) * count * @sizeOf(f32), values.ptr, count * @sizeOf(f32)) == 0) return false;
    for (mean, 0..) |*value, column| {
        var sum: f32 = 0;
        for (0..streams) |stream| sum += values[stream * width + column];
        value.* = sum / @as(f32, @floatFromInt(streams));
    }
    var path: [1024]u8 = undefined;
    const length = c.snprintf(&path, path.len, "%s_%s-%u_pos0.bin", prefix, label, layer);
    if (length < 0 or length >= path.len) return false;
    const file = c.fopen(&path, "wb") orelse return false;
    const ok = c.fwrite(mean.ptr, @sizeOf(f32), mean.len, file) == mean.len;
    return c.fclose(file) == 0 and ok;
}
