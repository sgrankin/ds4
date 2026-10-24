//! Descriptor loading remains usable without pulling in the GPU runtime.
comptime {
    _ = @import("glm53/model_entry.zig");
    _ = @import("glm53/memory_entry.zig");
}
