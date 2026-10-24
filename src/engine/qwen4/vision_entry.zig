//! C-compatible sidecar/encoder entries. The model implementation owns no
//! Engine mirror and borrows only explicit GGUF records and tensor pointers.
const c = @import("gguf_c");
const weights = @import("vision_weights.zig");
const vision = @import("vision.zig");

export fn ds4_native_qwen4_vision_weights_bind(output: *c.ds4_qwen4_vision_weights, model: *const c.ds4_model, error_text: [*c]u8, capacity: usize) bool {
    const errors: []u8 = if (capacity == 0) &.{} else error_text[0..capacity];
    if (errors.len != 0) errors[0] = 0;
    weights.bind(output, model, errors) catch return false;
    return true;
}
export fn ds4_gpu_qwen4_vision_encode(out: [*c]f32, patches: [*c]const f32, position: [*c]const f32, rows: u32, grid_width: u32, model_map: ?*const anyopaque, model_size: u64, table: [*c]const c.ds4_qwen4_vision_weights) c_int {
    return vision.encode(out, patches, position, rows, grid_width, model_map, model_size, table);
}
export fn ds4_native_qwen4_vision_pos_embed(table: [*c]const f32, side: u32, width: u32, grid_height: u32, grid_width: u32, out: [*c]f32) void {
    vision.positionEmbeddings(table, side, width, grid_height, grid_width, out);
}
