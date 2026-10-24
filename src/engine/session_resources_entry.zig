//! Hot session-resource ABI. Keep separate from CPU-only frontier/control and
//! snapshot objects: construction requires the actual native model allocators.
const c = @import("session_resources_c");
const resources = @import("session_resources.zig");
export fn ds4_native_session_resources_init(
    view: *c.ds4_session_view,
    pool: *c.ds4_session_resource_pool,
    inputs: *const c.ds4_session_resource_inputs,
    context: c_int,
) bool {
    return resources.init(view, pool, inputs, context);
}
export fn ds4_native_session_resources_free(
    view: *c.ds4_session_view,
    pool: *c.ds4_session_resource_pool,
) void {
    resources.deinit(view, pool);
}
