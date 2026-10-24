//! Image preparation around the Qwen vision tower. Learned positions use the
//! same align-corners resampling and 2x2 patch order as the native encoder.
const c = @import("gguf_c");

pub fn maxTokens() u32 {
    const value = c.getenv("DS4_QWEN4_IMAGE_MAX_TOKENS");
    const count = if (value == null) 0 else c.atol(value);
    // Four patches per token; the tower requires fewer than 65,536 patches.
    return if (count >= 64 and count <= 16383) @intCast(count) else 1024;
}

pub fn encode(model: *const c.ds4_model, weights: *const c.ds4_qwen4_vision_weights, image: *const c.ds4_image, minimum: u32, maximum: u32, out: *[*c]f32, patches: *c.ds4_image_patches, errors: []u8) bool {
    out.* = null;
    const error_pointer: [*c]u8 = if (errors.len == 0) null else errors.ptr;
    if (c.ds4_image_preprocess_qwen4(patches, image, minimum, maximum, error_pointer, errors.len) == 0) return false;
    const position_memory = c.malloc(@as(usize, patches.patch_count) * weights.n_embd * @sizeOf(f32));
    defer c.free(position_memory);
    const embedding_memory = c.malloc(@as(usize, patches.image_token_count) * weights.n_out * @sizeOf(f32));
    var keep = false;
    defer if (!keep) c.free(embedding_memory);
    if (position_memory != null and embedding_memory != null) {
        const position: [*]f32 = @ptrCast(@alignCast(position_memory.?));
        const embedding: [*]f32 = @ptrCast(@alignCast(embedding_memory.?));
        const table: [*]align(1) const f32 = @ptrCast(model.map + weights.pos_embd);
        // Bound model validation guarantees alignment of the learned table.
        c.ds4_native_qwen4_vision_pos_embed(@alignCast(table), weights.n_pos_side, weights.n_embd, patches.grid_height, patches.grid_width, position);
        if (c.ds4_gpu_qwen4_vision_encode(embedding, patches.patches, position, patches.patch_count, patches.grid_width, model.map, model.size, weights) != 0) {
            out.* = embedding;
            keep = true;
            return true;
        }
    }
    if (errors.len != 0) _ = c.snprintf(error_pointer, errors.len, "Qwen3.8 vision inference failed");
    return false;
}
