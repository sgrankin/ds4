//! CPU-safe policy exports share one compilation unit and the actual bindings.
//! Keep GPU/session lifecycle out so host fixtures extract only cold services.
comptime {
    _ = @import("engine_cache_sizes_entry.zig");
    _ = @import("model_spans_entry.zig");
    _ = @import("engine_admission_math_entry.zig");
}
