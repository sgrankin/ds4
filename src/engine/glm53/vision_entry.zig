//! Explicit model records cross this C ABI; the implementation owns the full
//! vision workspace and layer flow without borrowing process-global shapes.
const c = @import("gguf_c");
const weights = @import("vision_weights.zig");
const vision = @import("vision.zig");

export fn ds4_native_glm53_vision_weights_bind(output: *c.ds4_glm53_vision_weights, model: *const c.ds4_model, error_text: [*c]u8, capacity: usize) bool {
    const errors: []u8 = if (capacity == 0) &.{} else error_text[0..capacity];
    if (errors.len != 0) errors[0] = 0;
    weights.bind(output, model, errors) catch return false;
    return true;
}
export fn ds4_gpu_glm53_vision_encode(out: [*c]f32, patches: [*c]const f32, grid_height: u32, grid_width: u32, model_map: ?*const anyopaque, model_size: u64, table: [*c]const c.ds4_glm53_vision_weights) c_int {
    return vision.encode(out, patches, grid_height, grid_width, model_map, model_size, table);
}
