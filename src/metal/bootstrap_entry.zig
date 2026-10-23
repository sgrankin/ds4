//! Sole bootstrap facade. Public C shims retain only autorelease scope entry.
const c = @import("metal_c");
const owner = @import("bootstrap.zig");
export fn ds4_native_metal_bootstrap_init(context: *const c.ds4_metal_bootstrap_context) c_int {
    return @intFromBool(owner.init(context));
}
export fn ds4_native_metal_bootstrap_begin_cleanup(context: *const c.ds4_metal_bootstrap_context) bool {
    return owner.beginCleanup(context);
}
export fn ds4_native_metal_bootstrap_cleanup(context: *const c.ds4_metal_bootstrap_context) void {
    owner.cleanup(context);
}
