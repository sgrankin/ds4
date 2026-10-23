//! Sole C facade for the shared actual cache-runtime library action.
const c = @import("cache_runtime_c");
const owner = @import("cache_runtime.zig");
export fn ds4_state_cache_runtime_open(engine: ?*c.ds4_engine, options: *const c.ds4_engine_options, context: c_int, profile: [*c]u8, errors: [*c]u8, capacity: usize) ?*c.ds4_state_cache {
    return owner.open(engine, options, context, profile, if (capacity == 0 and errors == null) &.{} else @as([*]u8, @ptrCast(errors))[0..capacity]);
}
