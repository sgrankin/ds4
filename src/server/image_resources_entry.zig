//! Sole ABI facade; the owner has no exported definitions when imported.
const c = @import("server_images_c");
const owner = @import("image_resources.zig");
const marker_bytes = c.SERVER_IMAGE_MARKER_BYTES;
export fn ds4_server_images_free(images: ?*c.server_image_inputs) void {
    owner.free(images);
}
export fn ds4_server_image_data_uri(images: *c.server_image_inputs, uri: [*c]const u8, marker: *[marker_bytes]u8) bool {
    return owner.dataUri(images, uri, marker);
}
export fn ds4_server_image_base64(images: *c.server_image_inputs, media_type: [*c]const u8, source: [*c]const u8, marker: *[marker_bytes]u8) bool {
    return owner.base64(images, media_type, source, marker);
}
export fn ds4_server_images_cache_clear(cache: *c.server_image_cache) void {
    owner.clear(cache);
}
export fn ds4_server_images_cache_get(cache: *c.server_image_cache, input: *const c.server_image_input, out: *c.ds4_vision_embedding) bool {
    return owner.get(cache, input, out);
}
export fn ds4_server_images_cache_put(cache: *c.server_image_cache, input: *const c.server_image_input, embedding: *const c.ds4_vision_embedding, dimension: u32, budget: usize) void {
    owner.put(cache, input, embedding, dimension, budget);
}
export fn ds4_server_materialize_tokenize(engine: ?*c.ds4_engine, runtime: ?*c.server, request: *c.request, messages: ?*const c.chat_msgs, errors: [*c]u8, capacity: usize) bool {
    return owner.tokenize(engine, runtime, request, messages, if (capacity == 0) &.{} else errors[0..capacity]);
}
