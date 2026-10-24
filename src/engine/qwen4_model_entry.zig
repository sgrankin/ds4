//! Descriptor loading remains usable without pulling in the GPU runtime.
comptime {
    _ = @import("qwen4/model_entry.zig");
    _ = @import("qwen4/memory_entry.zig");
    _ = @import("qwen4/ngrams_entry.zig");
}
