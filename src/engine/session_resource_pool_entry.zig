//! Cold pool admission entry: no model allocator, graph execution or Engine
//! implementation may enter this object.
const c = @import("session_pool_c");
const accounting = @import("session_resource_accounting.zig");
export fn ds4_native_session_graph_budget(
    pool: *const c.ds4_session_resource_pool,
    session_hint: c_int,
    next_bytes: u64,
) u64 {
    return accounting.graphBudget(pool.glm_session_graph_bytes, pool.glm_session_count, session_hint, next_bytes);
}
