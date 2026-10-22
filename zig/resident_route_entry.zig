const c = @import("gguf_c");
const route = @import("resident_route.zig");
const Change = c.ds4_resident_route_change;
const Plan = c.ds4_resident_route_plan;
const Drop = c.ds4_resident_drop_change;
export fn ds4_resident_route_weakest(selected: [*c]const i32, affinity: [*c]const f32, total: u32, used: u32, max_mass: f32, maybe_slot: ?*u32, maybe_mass: ?*f32) bool {
    return route.ds4_resident_route_weakest(selected, affinity, total, used, max_mass, maybe_slot, maybe_mass);
}
export fn ds4_resident_route_choose(selected: [*c]const i32, affinity: [*c]const f32, bias: [*c]const f32, resident: [*c]const u8, total: u32, used: u32, max_gap: f32, max_rank: u32, max_mass: f32, maybe_change: ?*Change) bool {
    return route.ds4_resident_route_choose(selected, affinity, bias, resident, total, used, max_gap, max_rank, max_mass, maybe_change);
}
export fn ds4_resident_route_weak_pair(selected: [*c]const i32, affinity: [*c]const f32, total: u32, used: u32, max_mass: f32, slots: [*c]u32, masses: [*c]f32) u32 {
    return route.ds4_resident_route_weak_pair(selected, affinity, total, used, max_mass, slots, masses);
}
export fn ds4_resident_route_weak_triple(selected: [*c]const i32, affinity: [*c]const f32, total: u32, used: u32, max_mass: f32, slots: [*c]u32, masses: [*c]f32) u32 {
    return route.ds4_resident_route_weak_triple(selected, affinity, total, used, max_mass, slots, masses);
}
export fn ds4_resident_route_choose_triple(selected: [*c]const i32, affinity: [*c]const f32, bias: [*c]const f32, resident: [*c]const u8, total: u32, used: u32, max_gap: f32, max_rank: u32, max_mass: f32, max_total_mass: f32, maybe_plan: ?*Plan) bool {
    return route.ds4_resident_route_choose_triple(selected, affinity, bias, resident, total, used, max_gap, max_rank, max_mass, max_total_mass, maybe_plan);
}
export fn ds4_resident_route_choose_plan(selected: [*c]const i32, affinity: [*c]const f32, bias: [*c]const f32, resident: [*c]const u8, total: u32, used: u32, max_gap: f32, max_rank: u32, max_mass: f32, max_subs: u32, max_total_mass: f32, maybe_plan: ?*Plan) bool {
    return route.ds4_resident_route_choose_plan(selected, affinity, bias, resident, total, used, max_gap, max_rank, max_mass, max_subs, max_total_mass, maybe_plan);
}
export fn ds4_resident_route_choose_drop(selected: [*c]const i32, affinity: [*c]const f32, selected_resident: [*c]const u8, total: u32, used: u32, max_mass: f32, maybe_change: ?*Drop) bool {
    return route.ds4_resident_route_choose_drop(selected, affinity, selected_resident, total, used, max_mass, maybe_change);
}
export fn ds4_resident_route_drop_weights(weights: [*c]f32, original: [*c]const i32, affinity: [*c]const f32, used: u32, dropped: u32, scale: f32) void {
    route.ds4_resident_route_drop_weights(weights, original, affinity, used, dropped, scale);
}
export fn ds4_resident_route_weights(weights: [*c]f32, ids: [*c]const i32, affinity: [*c]const f32, used: u32, scale: f32) void {
    route.ds4_resident_route_weights(weights, ids, affinity, used, scale);
}
