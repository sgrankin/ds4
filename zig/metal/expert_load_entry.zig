//! Sole ABI facade for the complete expert read/preparation transactions.
const c = @import("metal_c");
const Core = @import("expert_load.zig");
const Selected = @import("expert_selected_load.zig");
const Prefill = @import("expert_prefill_load.zig");
const IO = @import("expert_io.zig");
const Policy = @import("expert_load_policy.zig");
const Entry = Core.Entry;
const Reuse = Core.Reuse;
const Table = Core.Table;
export fn ds4_native_expert_pending_finish(entries: [*c][*c]Entry) c_int {
    return @intFromBool(Core.Context.current().finish(entries));
}
export fn ds4_native_expert_pending_clear() void {
    Core.Context.current().clear();
}
export fn ds4_native_expert_pending_matches(table: *const Table, ids: [*c]const i32, count: u32) c_int {
    return @intFromBool(Core.Context.current().matches(table, Core.slice(ids, count)));
}
export fn ds4_native_expert_pending_gate_wait() c_int {
    return IO.gateWait(c.ds4_metal_expert_load_driver());
}
export fn ds4_native_expert_pread_shutdown() void {
    const view = c.ds4_metal_expert_load_driver();
    IO.shutdown(@ptrCast(&view.load.*.pool));
}
export fn ds4_native_expert_locked_slots() u32 {
    return Core.Context.current().cache.lockedCount();
}
export fn ds4_native_expert_load_missing(
    table: *const Table,
    ids: [*c]const i32,
    count: u32,
    gate: [*c]const u64,
    up: [*c]const u64,
    down: [*c]const u64,
    mask: u32,
    entries: [*c][*c]Entry,
) c_int {
    var context = Core.Context.current();
    return @intFromBool(Selected.missing(&context, table, ids, count, gate, up, down, mask, false, entries));
}
export fn ds4_native_expert_prefetch_take(table: *const Table, count: u32, ids: [*c]i32) c_int {
    var context = Core.Context.current();
    return @intFromBool(Selected.prefetchTake(&context, table, count, ids));
}
export fn ds4_native_expert_prepare_batch(
    table: *const Table,
    selected: ?*const c.ds4_gpu_tensor,
    tokens: u32,
    used: u32,
    addresses: *Reuse,
    resources: [*c][*c]Entry,
    resource_count: *u32,
    unique: *u32,
    overflow: *Reuse,
) c_int {
    var context = Core.Context.current();
    return @intFromBool(Prefill.prepare(&context, table, selected, tokens, used, addresses, resources, resource_count, unique, overflow));
}
export fn ds4_gpu_stream_expert_cache_begin_selected_load(table: ?*const Table, ids: [*c]const i32, count: u32) c_int {
    var context = Core.Context.current();
    return @intFromBool(context.begin(table, ids, count));
}
export fn ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(table: ?*const Table, selected: ?*const c.ds4_gpu_tensor, count: u32) c_int {
    var context = Core.Context.current();
    return @intFromBool(Selected.beginTensor(&context, table, selected, count));
}
export fn ds4_gpu_stream_expert_cache_seed_selected(table: ?*const Table, ids: [*c]const i32, count: u32) c_int {
    var context = Core.Context.current();
    return @intFromBool(Selected.seedSelected(&context, table, ids, count));
}
export fn ds4_gpu_stream_expert_cache_seed_experts(table: ?*const Table, ids: [*c]const i32, priorities: [*c]const u32, count: u32) c_int {
    var context = Core.Context.current();
    return @intFromBool(Selected.seedExperts(&context, table, ids, priorities, count, false));
}
export fn ds4_gpu_stream_expert_cache_seed_experts_gpu_copy(table: ?*const Table, ids: [*c]const i32, priorities: [*c]const u32, count: u32) c_int {
    var context = Core.Context.current();
    return @intFromBool(Selected.seedExperts(&context, table, ids, priorities, count, true));
}
export fn ds4_native_expert_address_requested() c_int {
    return @intFromBool(Policy.addressRequested(c.ds4_metal_expert_load_driver()));
}
export fn ds4_native_expert_prefill_enabled(tokens: u32, total: u32, used: u32, gate_type: u32, down_type: u32) c_int {
    return @intFromBool(Policy.prefillEnabled(Core.Context.current(), tokens, total, used, gate_type, down_type));
}
