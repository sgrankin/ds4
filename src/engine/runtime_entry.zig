//! One compilation/archive unit for ordinary hot Metal runtime exports.
//! These entries already share gguf_c. Keep CPU-safe metadata, memory, FILE,
//! sampling and frontier exports separate so model-free fixtures cannot pull
//! this runtime and its engine/platform dependencies from the archive.
comptime {
    _ = @import("graph.zig");
    _ = @import("qwen4_entry.zig");
    _ = @import("glm53_entry.zig");
    _ = @import("glm53_storage_entry.zig");
    _ = @import("vision_runtime_entry.zig");
    _ = @import("session_entry.zig");
    _ = @import("session_batch_entry.zig");
    _ = @import("engine_admission_entry.zig");
    _ = @import("engine_lifecycle_entry.zig");
}
