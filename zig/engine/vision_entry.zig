//! Keep the public GPU entry separate from graph/session exports: image-only
//! callers must not pull the engine's transitional diagnostics into their link.
const c = @import("gguf_c");
const vision = @import("vision.zig");

export fn ds4_gpu_deepseek4_vision_encode(out: [*c]f32, patches: [*c]const f32, grid_height: u32, grid_width: u32, model_map: ?*const anyopaque, model_size: u64, weights: [*c]const c.ds4_deepseek4_vision_weights) c_int {
    return vision.encode(out, patches, grid_height, grid_width, model_map, model_size, weights);
}
