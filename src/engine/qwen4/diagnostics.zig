//! Optional steering/activation dumps. Shared leaves select the existing output
//! prefix; Qwen owns the last-row selection and HC mean used by its recipes.
const std = @import("std");
const c = @import("gguf_c");
const ops = @import("ops.zig");
const Graph = c.ds4_qwen4_gpu_graph;

fn write(path: [*:0]const u8, values: []const f32) bool {
    const stream = c.fopen(path, "wb") orelse return false;
    const ok = c.fwrite(values.ptr, @sizeOf(f32), values.len, stream) == values.len;
    return c.fclose(stream) == 0 and ok;
}

pub fn lastFfn(graph: *Graph, layer: u32, rows: u32) void {
    const prefix = c.ds4_native_graph_debug_prefix("ffn_out", layer, 0);
    if (graph.R == null or rows == 0 or prefix == null or c.ds4_gpu_synchronize() == 0) return;
    defer _ = ops.beginCommands();
    const shape = &graph.config.shape;
    const width: usize = shape.n_embd;
    const streams: usize = shape.n_hc;
    const values = std.heap.c_allocator.alloc(f32, streams * width) catch c.ds4_engine_model_die("out of memory");
    defer std.heap.c_allocator.free(values);
    const offset = @as(u64, rows - 1) * streams * width * @sizeOf(f32);
    if (c.ds4_gpu_tensor_read(graph.R, offset, values.ptr, values.len * @sizeOf(f32)) == 0) return;
    const mean = std.heap.c_allocator.alloc(f32, width) catch c.ds4_engine_model_die("out of memory");
    defer std.heap.c_allocator.free(mean);
    for (mean, 0..) |*value, column| {
        var sum: f32 = 0;
        for (0..streams) |stream| sum += values[stream * width + column];
        value.* = sum / @as(f32, @floatFromInt(streams));
    }
    var storage: [1024]u8 = undefined;
    _ = c.snprintf(&storage, storage.len, "%s_ffn_out-%u_pos0.bin", prefix, layer);
    if (write(@ptrCast(&storage), mean)) _ = c.fprintf(c.__stderrp, "ds4: dumped ffn_out layer %u last-token HC-mean to %s\n", layer, &storage);
}

pub fn lastAttention(graph: *Graph, layer: u32, rows: u32) void {
    const prefix = c.ds4_native_graph_debug_prefix("attn_out", layer, 0);
    if (graph.blk == null or rows == 0 or prefix == null or c.ds4_gpu_synchronize() == 0) return;
    defer _ = ops.beginCommands();
    const width: usize = graph.config.shape.n_embd;
    const values = std.heap.c_allocator.alloc(f32, width) catch c.ds4_engine_model_die("out of memory");
    defer std.heap.c_allocator.free(values);
    if (c.ds4_gpu_tensor_read(graph.blk, @as(u64, rows - 1) * width * @sizeOf(f32), values.ptr, width * @sizeOf(f32)) == 0) return;
    var storage: [1024]u8 = undefined;
    _ = c.snprintf(&storage, storage.len, "%s_attn_out-%u_pos0.bin", prefix, layer);
    if (write(@ptrCast(&storage), values)) _ = c.fprintf(c.__stderrp, "ds4: dumped attn_out layer %u last-token to %s\n", layer, &storage);
}
