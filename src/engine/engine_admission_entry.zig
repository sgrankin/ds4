//! Hot engine policy ABI. Pure budget/format exports stay in cold objects so
//! CPU fixtures cannot extract a second Engine merely to measure metadata.
const c = @import("gguf_c");
const admission = @import("engine_admission.zig");
const hotlist = @import("engine_cache_seed.zig");
const startup_cache = @import("engine_startup_cache.zig");
export fn ds4_native_engine_fit_glm_cache(
    core: *c.ds4_engine_core,
    pool: *const c.ds4_session_resource_pool,
    shape: *const c.ds4_shape,
    policy: *c.ds4_streaming_policy,
    placement: *const c.ds4_model_residency_inputs,
    session_hint: c_int,
) void {
    admission.fit(core, pool, shape, policy, placement, session_hint);
}
export fn ds4_native_engine_admit_glm(
    core: *const c.ds4_engine_core,
    shape: *const c.ds4_shape,
    placement: *const c.ds4_model_residency_inputs,
    compact: u32,
    extra: u64,
    phase: [*c]const u8,
) bool {
    return admission.admit(core, shape, placement, compact, extra, phase);
}
export fn ds4_native_engine_seed_hotlist(core: *const c.ds4_engine_core, shape: *const c.ds4_shape) bool {
    return hotlist.seed(core, shape);
}
export fn ds4_native_engine_auto_cache(core: *c.ds4_engine_core, shape: *const c.ds4_shape, context: c_int, chunk: u32) bool {
    return startup_cache.automatic(core, shape, context, chunk);
}
export fn ds4_native_engine_reserve_cache(core: *c.ds4_engine_core, shape: *const c.ds4_shape, policy: *c.ds4_streaming_policy) bool {
    return startup_cache.reserve(core, shape, policy);
}
export fn ds4_native_engine_manual_safe_bytes(core: *const c.ds4_engine_core, shape: *const c.ds4_shape, context: c_int, chunk: u32) u64 {
    return startup_cache.manualSafe(core, shape, context, chunk);
}
