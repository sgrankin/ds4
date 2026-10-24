//! Actual Core/Shape are borrowed synchronously. Outputs own libc memory;
//! prompt spans receive ownership only when their complete image block exists.
const c = @import("gguf_c");
const Runtime = @import("vision_runtime.zig");
const Qwen = @import("qwen4/vision_input.zig");

fn errors(pointer: [*c]u8, capacity: usize) []u8 {
    return if (pointer == null or capacity == 0) &.{} else pointer[0..capacity];
}
export fn ds4_native_vision_encode_file(core: *c.ds4_engine_core, shape: *const c.ds4_shape, path: [*c]const u8, out: ?*c.ds4_vision_embedding, message: [*c]u8, capacity: usize) c_int {
    return @intFromBool((Runtime.Vision{ .core = core, .shape = shape }).encodeFile(path, out, errors(message, capacity)));
}
export fn ds4_native_vision_encode_memory(core: *c.ds4_engine_core, shape: *const c.ds4_shape, encoded: [*c]const u8, length: usize, out: ?*c.ds4_vision_embedding, message: [*c]u8, capacity: usize) c_int {
    return @intFromBool((Runtime.Vision{ .core = core, .shape = shape }).encodeMemory(encoded, length, out, errors(message, capacity)));
}
export fn ds4_native_vision_append(core: *c.ds4_engine_core, shape: *const c.ds4_shape, tokens: ?*c.ds4_tokens, span: ?*c.ds4_vision_span, embedding: ?*c.ds4_vision_embedding, message: [*c]u8, capacity: usize) c_int {
    return @intFromBool((Runtime.Vision{ .core = core, .shape = shape }).append(tokens, span, embedding, errors(message, capacity)));
}
export fn ds4_native_vision_message(core: *c.ds4_engine_core, shape: *const c.ds4_shape, tokens: ?*c.ds4_tokens, role: [*c]const u8, parts: [*c]const [*c]const u8, embeddings: [*c]c.ds4_vision_embedding, count: usize, spans: [*c]c.ds4_vision_span, message: [*c]u8, capacity: usize) c_int {
    return @intFromBool((Runtime.Vision{ .core = core, .shape = shape }).message(tokens, role, parts, embeddings, count, spans, errors(message, capacity)));
}
export fn ds4_native_qwen4_vision_image(model: *const c.ds4_model, weights: *const c.ds4_qwen4_vision_weights, image: *const c.ds4_image, minimum: u32, maximum: u32, out: *[*c]f32, patches: *c.ds4_image_patches, message: [*c]u8, capacity: usize) c_int {
    return @intFromBool(Qwen.encode(model, weights, image, minimum, maximum, out, patches, errors(message, capacity)));
}
export fn ds4_native_qwen4_vision_max_tokens() u32 {
    return Qwen.maxTokens();
}
