//! Bounded resident-expert substitution policy over caller-owned snapshots.
//! Callers own cache admission, GPU commands, commits and metrics.
const std = @import("std");
const c = @import("gguf_c");
const Change = c.ds4_resident_route_change;
const Plan = c.ds4_resident_route_plan;
const Drop = c.ds4_resident_drop_change;

// The caller can reject an ineligible native route before examining the cache.
pub fn ds4_resident_route_weakest(selected: [*c]const i32, affinity: [*c]const f32, total: u32, used: u32, max_mass: f32, maybe_slot: ?*u32, maybe_mass: ?*f32) bool {
    @setFloatMode(.strict);
    if (selected == null or affinity == null or maybe_slot == null or maybe_mass == null or total == 0 or used != 6 or max_mass <= 0) return false;
    const routes = selected[0..used];
    var weakest: usize = 0;
    var sum: f32 = 0;
    for (routes, 0..) |id, slot| {
        if (id < 0 or @as(u32, @intCast(id)) >= total) return false;
        const probability = affinity[@intCast(id)];
        if (!std.math.isFinite(probability) or probability <= 0) return false;
        for (routes[0..slot]) |previous| if (previous == id) return false;
        sum += probability;
        if (probability < affinity[@intCast(routes[weakest])]) weakest = slot;
    }
    const mass = affinity[@intCast(routes[weakest])] / sum;
    if (mass > max_mass) return false;
    // Rejection leaves both caller outputs untouched.
    maybe_slot.?.* = @intCast(weakest);
    maybe_mass.?.* = mass;
    return true;
}

// A score neighbor need not compute the same function: this is intentionally
// lossy. Protect the strongest original contributors and measure task quality.
fn replacement(routes: []const i32, affinity: []const f32, bias: []const f32, resident: []const u8, max_gap: f32, max_rank: u32, slot: u32, mass: f32, proposed: []const Change) ?Change {
    @setFloatMode(.strict);
    const old: usize = @intCast(routes[slot]);
    const old_probability = affinity[old];
    const old_score = old_probability + bias[old];
    var candidate: ?usize = null;
    var best_score: f32 = 0;
    for (resident, 0..) |present, expert| {
        if (present == 0) continue;
        var included = false;
        for (routes) |id| included = included or @as(usize, @intCast(id)) == expert;
        for (proposed) |change| included = included or change.new_expert == expert;
        if (included) continue;
        const probability = affinity[expert];
        const score = probability + bias[expert];
        if (!std.math.isFinite(probability) or !std.math.isFinite(score) or probability <= 0 or
            probability < old_probability * (1.0 - max_gap) or
            probability > old_probability * (1.0 + max_gap) or
            old_score - score > max_gap * old_probability) continue;
        // Ascending expert iteration retains the lower ID when scores tie.
        if (candidate == null or score > best_score) {
            candidate = expert;
            best_score = score;
        }
    }
    const expert = candidate orelse return null;
    var rank: u32 = 1;
    for (affinity, bias, 0..) |probability, correction, other| {
        const score = probability + correction;
        if (score > best_score or (score == best_score and other < expert)) rank += 1;
    }
    if (rank > max_rank) return null;
    return .{
        .slot = slot,
        .old_expert = @intCast(old),
        .new_expert = @intCast(expert),
        .candidate_rank = rank,
        .old_mass = mass,
        .score_gap = c.fmaxf(0, old_score - best_score) / old_probability,
        .affinity_ratio = affinity[expert] / old_probability,
    };
}

// Only the weakest unbiased contribution among the native six can move. The
// other five, correction bias and six unique routes survive a one-slot change.
pub fn ds4_resident_route_choose(selected: [*c]const i32, affinity: [*c]const f32, bias: [*c]const f32, resident: [*c]const u8, total: u32, used: u32, max_gap: f32, max_rank: u32, max_mass: f32, maybe_change: ?*Change) bool {
    @setFloatMode(.strict);
    if (bias == null or resident == null or maybe_change == null or max_gap <= 0 or max_rank <= used) return false;
    var slot: u32 = undefined;
    var mass: f32 = undefined;
    if (!ds4_resident_route_weakest(selected, affinity, total, used, max_mass, &slot, &mass)) return false;
    if (resident[@intCast(selected[slot])] != 0) return false;
    const change = replacement(selected[0..used], affinity[0..total], bias[0..total], resident[0..total], max_gap, max_rank, slot, mass, &.{}) orelse return false;
    maybe_change.?.* = change;
    return true;
}

// Only the two weakest ORIGINAL routes are eligible. Replacement affinities
// must never change which four native contributors this policy protects.
pub fn ds4_resident_route_weak_pair(selected: [*c]const i32, affinity: [*c]const f32, total: u32, used: u32, max_mass: f32, slots: [*c]u32, masses: [*c]f32) u32 {
    @setFloatMode(.strict);
    if (slots == null or masses == null or !ds4_resident_route_weakest(selected, affinity, total, used, max_mass, &slots[0], &masses[0])) return 0;
    var sum: f32 = 0;
    var second: u32 = if (slots[0] == 0) 1 else 0;
    for (selected[0..used], 0..) |id, slot| {
        sum += affinity[@intCast(id)];
        if (slot != slots[0] and affinity[@intCast(id)] < affinity[@intCast(selected[second])]) second = @intCast(slot);
    }
    // These outputs are written even when the second mass is ineligible.
    slots[1] = second;
    masses[1] = affinity[@intCast(selected[second])] / sum;
    return if (masses[1] <= max_mass) 2 else 1;
}

// Extend only eligible ORIGINAL slots. Keep this arithmetic separate from the
// one/two-slot helpers, whose existing decisions and floating bits are retained.
pub fn ds4_resident_route_weak_triple(selected: [*c]const i32, affinity: [*c]const f32, total: u32, used: u32, max_mass: f32, slots: [*c]u32, masses: [*c]f32) u32 {
    @setFloatMode(.strict);
    const pair = ds4_resident_route_weak_pair(selected, affinity, total, used, max_mass, slots, masses);
    if (pair != 2) return pair;
    var third: u32 = 0;
    while (third == slots[0] or third == slots[1]) third += 1;
    var sum: f32 = 0;
    for (selected[0..used], 0..) |id, slot| {
        sum += affinity[@intCast(id)];
        if (slot != slots[0] and slot != slots[1] and affinity[@intCast(id)] < affinity[@intCast(selected[third])]) third = @intCast(slot);
    }
    slots[2] = third;
    masses[2] = affinity[@intCast(selected[third])] / sum;
    return if (masses[2] <= max_mass) 3 else 2;
}

// Two- and three-slot plans have the same replacement loop. Their independent
// eligibility helpers above retain the original summation/slot-selection order.
fn fillPlan(routes: []const i32, affinity: []const f32, bias: []const f32, resident: []const u8, max_gap: f32, max_rank: u32, max_total_mass: f32, slots: []const u32, masses: []const f32, plan: *Plan) bool {
    @setFloatMode(.strict);
    for (slots, masses) |slot, mass| {
        const old: usize = @intCast(routes[slot]);
        // Budget original contribution, not the evolving fallback mixture.
        // Preserve the original skip behavior rather than stopping the scan.
        if (resident[old] != 0 or plan.total_mass + mass > max_total_mass) continue;
        const change = replacement(routes, affinity, bias, resident, max_gap, max_rank, slot, mass, plan.changes[0..plan.count]) orelse continue;
        plan.changes[plan.count] = change;
        plan.count += 1;
        plan.total_mass += mass;
    }
    return plan.count != 0;
}

// Protect the three strongest ORIGINAL contributions while preserving score,
// rank and original-mass guards. Rejection still clears the complete plan.
pub fn ds4_resident_route_choose_triple(selected: [*c]const i32, affinity: [*c]const f32, bias: [*c]const f32, resident: [*c]const u8, total: u32, used: u32, max_gap: f32, max_rank: u32, max_mass: f32, max_total_mass: f32, maybe_plan: ?*Plan) bool {
    @setFloatMode(.strict);
    const plan = maybe_plan orelse return false;
    plan.* = std.mem.zeroes(Plan);
    if (bias == null or resident == null or max_gap <= 0 or max_rank <= used or max_total_mass <= 0) return false;
    var slots: [3]u32 = undefined;
    var masses: [3]f32 = undefined;
    const count = ds4_resident_route_weak_triple(selected, affinity, total, used, max_mass, &slots, &masses);
    if (count == 0) return false;
    return fillPlan(selected[0..used], affinity[0..total], bias[0..total], resident[0..total], max_gap, max_rank, max_total_mass, slots[0..count], masses[0..count], plan);
}

pub fn ds4_resident_route_choose_plan(selected: [*c]const i32, affinity: [*c]const f32, bias: [*c]const f32, resident: [*c]const u8, total: u32, used: u32, max_gap: f32, max_rank: u32, max_mass: f32, max_subs: u32, max_total_mass: f32, maybe_plan: ?*Plan) bool {
    @setFloatMode(.strict);
    const plan = maybe_plan orelse return false;
    plan.* = std.mem.zeroes(Plan);
    if (max_subs == 1) {
        // Preserve the v3 selector and exact arithmetic for the default.
        // In particular, one-slot mode never consults max_total_mass.
        if (!ds4_resident_route_choose(selected, affinity, bias, resident, total, used, max_gap, max_rank, max_mass, &plan.changes[0])) return false;
        plan.count = 1;
        plan.total_mass = plan.changes[0].old_mass;
        return true;
    }
    if (max_subs == 3) return ds4_resident_route_choose_triple(selected, affinity, bias, resident, total, used, max_gap, max_rank, max_mass, max_total_mass, plan);
    if (max_subs != 2 or bias == null or resident == null or max_gap <= 0 or max_rank <= used or max_total_mass <= 0) return false;
    var slots: [2]u32 = undefined;
    var masses: [2]f32 = undefined;
    const count = ds4_resident_route_weak_pair(selected, affinity, total, used, max_mass, &slots, &masses);
    if (count == 0) return false;
    return fillPlan(selected[0..used], affinity[0..total], bias[0..total], resident[0..total], max_gap, max_rank, max_total_mass, slots[0..count], masses[0..count], plan);
}

// Deliberately lossy: alias only a missing weakest ORIGINAL contribution to
// an already resident native ID. The other five IDs and ratios stay unchanged.
pub fn ds4_resident_route_choose_drop(selected: [*c]const i32, affinity: [*c]const f32, selected_resident: [*c]const u8, total: u32, used: u32, max_mass: f32, maybe_change: ?*Drop) bool {
    if (selected_resident == null or maybe_change == null) return false;
    var weakest: u32 = undefined;
    var mass: f32 = undefined;
    if (!ds4_resident_route_weakest(selected, affinity, total, used, max_mass, &weakest, &mass) or selected_resident[weakest] != 0) return false;
    for (selected[0..used], 0..) |id, slot| {
        if (slot == weakest or selected_resident[slot] == 0) continue;
        maybe_change.?.* = .{ .slot = weakest, .old_expert = @intCast(selected[weakest]), .alias_slot = @intCast(slot), .alias_expert = @intCast(id), .old_mass = mass };
        return true;
    }
    return false;
}

pub fn ds4_resident_route_drop_weights(weights: [*c]f32, original: [*c]const i32, affinity: [*c]const f32, used: u32, dropped: u32, scale: f32) void {
    @setFloatMode(.strict);
    const routes: []const i32 = if (used == 0) &.{} else original[0..used];
    var sum: f32 = 0;
    for (routes, 0..) |id, slot| if (slot != dropped) {
        sum += affinity[@intCast(id)];
    };
    sum = c.fmaxf(sum, 6.103515625e-5); // Existing 2^-14 denominator floor.
    for (routes, 0..) |id, slot| weights[slot] = if (slot == dropped) 0 else affinity[@intCast(id)] / sum * scale;
}

// Cache residency affects selection only. Mixture coefficients always use
// unmodified sqrt(softplus(logit)) affinities of the final chosen experts.
pub fn ds4_resident_route_weights(weights: [*c]f32, ids: [*c]const i32, affinity: [*c]const f32, used: u32, scale: f32) void {
    @setFloatMode(.strict);
    const routes: []const i32 = if (used == 0) &.{} else ids[0..used];
    var sum: f32 = 0;
    for (routes) |id| sum += affinity[@intCast(id)];
    sum = c.fmaxf(sum, 6.103515625e-5);
    for (routes, 0..) |id, slot| weights[slot] = affinity[@intCast(id)] / sum * scale;
}
