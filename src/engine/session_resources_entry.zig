//! Hot session-resource ABI. Keep separate from CPU-only frontier/control and
//! snapshot objects: construction requires the actual native model allocators.
const c = @import("session_resources_c");
const MetalScope = @import("metal_scope.zig").Scope(c);
const resources = @import("session_resources.zig");
export fn ds4_native_session_resources_init(
    view: *c.ds4_session_view,
    pool: *c.ds4_session_resource_pool,
    inputs: *const c.ds4_session_resource_inputs,
    context: c_int,
) bool {
    const metal_scope = MetalScope.enter(view.core);
    defer metal_scope.leave();
    return resources.init(view, pool, inputs, context);
}
export fn ds4_native_session_resources_free(
    view: *c.ds4_session_view,
    pool: *c.ds4_session_resource_pool,
) void {
    const metal_scope = MetalScope.enter(view.core);
    defer metal_scope.leave();
    resources.deinit(view, pool);
}
