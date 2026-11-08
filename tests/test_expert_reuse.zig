//! CPU fixture for the production cache policy. Only ARC buffer moves and
//! driver completion hooks are stand-ins; victim selection and accounting are
//! the same native code used by the streamed model.
const std = @import("std");
const c = @import("metal_c");
const cache = @import("expert_cache");
const expect = std.testing.expect;
const equal = std.testing.expectEqual;

export var ds4_native_expert_policy: c.ds4_metal_expert_policy = .{ .timing_checked = true, .timing_enabled = false, .decay_interval = 16 };
var service_thread = false;
export fn ds4_metal_expert_on_service_thread() bool {
    return service_thread;
}
export fn ds4_metal_expert_now_ms() f64 {
    return 0;
}
export fn ds4_metal_expert_wait(_: [*c]const u8) bool {
    return false;
}
export fn ds4_metal_expert_zero_address(_: u32, _: u32) void {}
export fn ds4_metal_expert_buffer_storage(_: cache.Buffer) c.ds4_metal_expert_buffer_memory {
    return .{ .data = null, .bytes = 0 };
}
export fn ds4_metal_expert_reuse_reset(reuse: *cache.Reuse) void {
    reuse.* = std.mem.zeroes(cache.Reuse);
}
export fn ds4_metal_expert_entry_retire(entry: *cache.Entry, reuse: ?*cache.Reuse) void {
    if (reuse) |out| out.* = .{
        .gate_buffer = entry.gate_buffer,
        .up_buffer = entry.up_buffer,
        .down_buffer = entry.down_buffer,
        .gate_inner = entry.gate_inner,
        .up_inner = entry.up_inner,
        .down_inner = entry.down_inner,
    };
    entry.gate_buffer = null;
    entry.up_buffer = null;
    entry.down_buffer = null;
}
fn newState() !*cache.State {
    const state = try std.testing.allocator.create(cache.State);
    @memset(std.mem.asBytes(state), 0);
    state.cache_budget_override = 1;
    return state;
}
fn seed(state: *cache.State, layer: u32, expert: u32, hotness: u32, used: u64) void {
    const entry = &state.entries[layer][expert];
    const identity: usize = 16 + @as(usize, layer) * 4096 + @as(usize, expert) * 4;
    entry.gate_buffer = @ptrFromInt(identity);
    entry.up_buffer = @ptrFromInt(identity + (if (expert % 2 == 0) @as(usize, 0) else 1));
    entry.down_buffer = @ptrFromInt(identity + (if (expert % 2 == 0) @as(usize, 0) else 2));
    entry.valid = 1;
    entry.gate_expert_bytes = 64;
    entry.down_expert_bytes = 32;
    entry.logical_bytes = 160;
    entry.last_used = used;
    state.cache_route_hotness[layer][expert] = hotness;
    state.cache_entry_count += 1;
    state.cache_layer_count[layer] += 1;
    state.cache_bytes += 160;
}

test "batch plan matches scalar eviction order, protected IDs, sizes and counts" {
    const a = try newState();
    defer std.testing.allocator.destroy(a);
    const b = try newState();
    defer std.testing.allocator.destroy(b);
    for (0..4) |layer| for (0..512) |expert| {
        seed(a, @intCast(layer), @intCast(expert), @intCast((expert * 17 + layer) % 13), (expert * 11 + layer) % 19);
    };
    // Wrong-size and scalar sentinel entries never become reusable victims.
    a.entries[0][0].gate_expert_bytes = 96;
    a.cache_route_hotness[0][1] = std.math.maxInt(u32);
    a.entries[0][1].last_used = std.math.maxInt(u64);
    @memcpy(std.mem.asBytes(b), std.mem.asBytes(a));
    const scalar = cache.Cache{ .state = a, .streaming = true };
    const batched = cache.Cache{ .state = b, .streaming = true };
    const protected = [_]i32{ 2, 3, 4, 500 };
    var plan: cache.ReusePlan = .{};
    try expect(batched.planReuse(&plan, 512, 1, &protected, 64, 32));
    try equal(@as(u32, 2048), b.cache_entry_count); // planning is read-only
    for (0..512) |_| {
        var expected = std.mem.zeroes(cache.Reuse);
        var actual = std.mem.zeroes(cache.Reuse);
        try expect(scalar.takeReusable(true, 1, &protected, 64, 32, &expected));
        try expect(batched.takePlanned(&plan, 1, &protected, 64, 32, &actual));
        try equal(expected.gate_buffer, actual.gate_buffer);
        try equal(expected.up_buffer, actual.up_buffer);
        try equal(expected.down_buffer, actual.down_buffer);
    }
    try equal(a.cache_entry_count, b.cache_entry_count);
    try equal(a.cache_bytes, b.cache_bytes);
    try equal(a.cache_evictions, b.cache_evictions);
    try equal(a.cache_buffer_reuses, b.cache_buffer_reuses);
    try std.testing.expectEqualSlices(u32, &a.cache_layer_count, &b.cache_layer_count);
    for (protected) |expert| try expect(b.entries[1][@intCast(expert)].valid != 0);
}

test "partial plans keep unused buffers and stale or in-flight victims invalidate the plan" {
    const state = try newState();
    defer std.testing.allocator.destroy(state);
    const owner = cache.Cache{ .state = state, .streaming = true };
    for (0..16) |expert| seed(state, 0, @intCast(expert), 0, expert);
    var plan: cache.ReusePlan = .{};
    var reuse = std.mem.zeroes(cache.Reuse);
    try expect(owner.planReuse(&plan, 16, 1, &.{}, 64, 32));
    try expect(owner.takePlanned(&plan, 1, &.{}, 64, 32, &reuse));
    try equal(@as(u32, 15), state.cache_entry_count);
    for (1..16) |expert| try expect(state.entries[0][expert].valid != 0);
    state.entries[0][1].last_used += 1;
    try expect(!owner.takePlanned(&plan, 1, &.{}, 64, 32, &reuse));
    try equal(@as(u32, 15), state.cache_entry_count);
    state.entries[0][2].inflight_seq = 1;
    try expect(!owner.planReuse(&plan, 16, 1, &.{}, 64, 32));
    state.cache_done_seq = 1;
    service_thread = true;
    defer service_thread = false;
    try expect(!owner.planReuse(&plan, 16, 1, &.{}, 64, 32));
    service_thread = false;
    try expect(owner.planReuse(&plan, 16, 1, &.{}, 64, 32));
    // A submitted command after planning also protects its victim.
    state.entries[0][1].inflight_seq = 2;
    try expect(!owner.takePlanned(&plan, 1, &.{}, 64, 32, &reuse));
    try equal(@as(u32, 15), state.cache_entry_count);
}
