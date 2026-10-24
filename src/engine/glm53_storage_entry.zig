//! GLM SSD policy shares the parent engine module's actual span/I/O owners.
pub const Storage = @import("glm53/storage.zig");
comptime {
    _ = @import("glm53/storage_entry.zig");
}
