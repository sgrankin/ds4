//! C only allocates its opaque outer Engine shell and captures these actual
//! component owners. The complete ordinary Metal startup transaction is native.
const c = @import("gguf_c");
const owner = @import("engine_lifecycle.zig");

export fn ds4_native_engine_lifecycle_open(core: *c.ds4_engine_core, configuration: *c.ds4_engine_configuration, pool: *c.ds4_session_resource_pool, flint: *?*c.flint_engine, streaming: *c.ds4_streaming_policy, options: *const c.ds4_engine_options, owned: *bool) c_int {
    var lifecycle = owner.Lifecycle{ .core = core, .configuration = configuration, .pool = pool, .flint = flint, .streaming = streaming };
    return lifecycle.open(options, owned);
}
export fn ds4_native_engine_lifecycle_close(core: *c.ds4_engine_core, pool: *c.ds4_session_resource_pool, flint: *?*c.flint_engine) void {
    owner.closeOwners(core, pool, flint);
}
