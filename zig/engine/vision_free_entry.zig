//! Independent retirement facade lets CPU fixtures link free without the GPU encoder.
const c = @import("gguf_c");
const VisionRuntime = @import("vision_runtime.zig");
export fn ds4_vision_embedding_free(embedding: ?*c.ds4_vision_embedding) void {
    VisionRuntime.freeEmbedding(embedding);
}
