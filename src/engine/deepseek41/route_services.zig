//! Model-owned optional route diagnostics and resident-cache transactions.
//! Actual graph settings/snapshots remain in Graph; all metrics, files and tape
//! state belong to the selected HostContext. Pure route math is imported directly.
const std = @import("std");
const c = @import("gguf_c");
const route = @import("resident_route");
const oracle_owner = @import("route_oracle.zig");
const Ops = @import("ops.zig").Ops;
const cache = @import("expert_cache.zig");
const Graph = c.ds41_gpu_graph;
const Layer = c.ds4_layer_weights;
const Plan = c.ds4_resident_route_plan;
const Metrics = c.ds4_resident_route_layer_metrics;
const apply_mode = 0;
const native_mode = 1;
const lookup_mode = 2;
const noop_mode = 3;
fn die(message: [*:0]const u8) noreturn {
    c.ds4_engine_model_die(message);
    unreachable;
}
fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn equals(text: [*c]const u8, literal: [*:0]const u8) bool {
    return text != null and c.strcmp(text, literal) == 0;
}
fn now() f64 {
    var time: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &time);
    return (@as(f64, @floatFromInt(time.tv_sec)) + @as(f64, @floatFromInt(time.tv_nsec)) * 1.0e-9) * 1000.0;
}
const RouteStats = struct {
    registered: bool = false,
    layers: u64 = 0,
    substitutions: u64 = 0,
    native_misses: u64 = 0,
    skipped_loads: u64 = 0,
    skipped_bytes: u64 = 0,
    full_snapshots: u64 = 0,
    double_layers: u64 = 0,
    triple_layers: u64 = 0,
    planned: u64 = 0,
    drops: u64 = 0,
    native_weights_requested: bool = false,
    native_substitutions: u64 = 0,
    replaced_mass: f64 = 0,
    score_gap: f64 = 0,
    affinity_ratio: f64 = 0,
    rank: f64 = 0,
    per_layer: [40]u64 = @splat(0),
};
const PrefillStats = struct {
    calls: u64 = 0,
    fallbacks: u64 = 0,
    frozen: u64 = 0,
    row_layers: u64 = 0,
    native_misses: u64 = 0,
    planned: u64 = 0,
    substitutions: u64 = 0,
    skipped_bytes: u64 = 0,
    double_layers: u64 = 0,
    triple_layers: u64 = 0,
    replaced_mass: f64 = 0,
};
const State = struct {
    oracle: oracle_owner.Oracle = .{},
    metrics: c.ds4_resident_route_metrics = std.mem.zeroes(c.ds4_resident_route_metrics),
    route_stats: RouteStats = .{},
    prefill_stats: PrefillStats = .{},
    noop_weights: [c.DS4_MAX_EXPERT_USED]f32 = @splat(0),
    prefill_logit_dumped: bool = false,
    profile_checked: bool = false,
    profile_enabled: bool = false,
    fn profile(self: *State) bool {
        if (!self.profile_checked) {
            const value = c.getenv("DS4_METAL_V41_RESIDENT_ROUTE_PROFILE");
            self.profile_enabled = value != null and value[0] != 0 and !equals(value, "0");
            self.profile_checked = true;
        }
        return self.profile_enabled;
    }
    fn registerReport(self: *State) void {
        if (!self.route_stats.registered) {
            self.route_stats.registered = true;
            if (c.ds4_host_context_selected() == null) _ = c.atexit(report);
        }
    }
};
// The actual diagnostics live once in the selected HostContext. Native Core
// retirement closes them; bare legacy callers retain their atexit report.
fn current() *State {
    const slot = c.ds4_host_context_routes_slot();
    if (slot.* == null) {
        const state = std.heap.c_allocator.create(State) catch die("cannot allocate route service owner");
        state.* = .{};
        slot.* = state;
    }
    return @ptrCast(@alignCast(slot.*.?));
}
pub fn cleanup() void {
    const slot = c.ds4_host_context_routes_slot();
    const state: *State = @ptrCast(@alignCast(slot.* orelse return));
    if (state.route_stats.registered) report();
    state.oracle.close();
    slot.* = null;
    std.heap.c_allocator.destroy(state);
}
pub fn dumpedFlag() *bool {
    return &current().prefill_logit_dumped;
}
fn closeOracle() callconv(.c) void {
    current().oracle.close();
}
fn mean(total: f64, count: u64) f64 {
    return if (count != 0) total / @as(f64, @floatFromInt(count)) else 0;
}
fn report() callconv(.c) void {
    const s = &current().route_stats;
    _ = c.fprintf(c.__stderrp, "ds4: resident-route layers=%llu substitutions=%llu native_misses=%llu skipped_loads=%llu skipped_bytes=%llu mean_mass=%.8f mean_score_gap=%.8f mean_affinity_ratio=%.8f mean_rank=%.3f full_snapshots=%llu double_layers=%llu planned_substitutions=%llu drops=%llu triple_layers=%llu\n", s.layers, s.substitutions, s.native_misses, s.skipped_loads, s.skipped_bytes, mean(s.replaced_mass, s.substitutions), mean(s.score_gap, s.substitutions), mean(s.affinity_ratio, s.substitutions), mean(s.rank, s.substitutions), s.full_snapshots, s.double_layers, s.planned, s.drops, s.triple_layers);
    if (s.native_weights_requested)
        _ = c.fprintf(c.__stderrp, "ds4: resident-route-native-weights requested=1 substitutions=%llu\n", s.native_substitutions);
    const p = &current().prefill_stats;
    if (p.calls != 0 or p.fallbacks != 0)
        _ = c.fprintf(c.__stderrp, "ds4: resident-prefill calls=%llu fallbacks=%llu frozen_residents=%llu row_layers=%llu native_misses=%llu planned_substitutions=%llu substitutions=%llu skipped_bytes=%llu double_layers=%llu triple_layers=%llu mean_mass=%.8f native_weights=1\n", p.calls, p.fallbacks, p.frozen, p.row_layers, p.native_misses, p.planned, p.substitutions, p.skipped_bytes, p.double_layers, p.triple_layers, mean(p.replaced_mass, p.substitutions));
}
pub fn metricsRead(output: ?*c.ds4_resident_route_metrics) void {
    const out = output orelse return;
    out.* = current().metrics;
    out.enabled = current().profile();
}
fn positiveGap(value: [*c]const u8) bool {
    if (value == null or value[0] == 0) return false;
    var end: [*c]u8 = null;
    const gap = c.strtof(value, &end);
    return end != value and end[0] == 0 and std.math.isFinite(gap) and gap > 0 and gap <= 1;
}
fn nativeWeights(value: [*c]const u8) bool {
    if (value == null or equals(value, "0")) return false;
    if (equals(value, "1")) return true;
    die("resident route native weights must be literal 0 or 1");
}
fn mass(value: [*c]const u8, message: [*:0]const u8) f32 {
    var end: [*c]u8 = null;
    const result = c.strtof(value, &end);
    if (end == value or end[0] != 0 or !std.math.isFinite(result) or result < 0 or result > 1) die(message);
    return result;
}
pub const Transaction = struct {
    ops: Ops,
    graph: *Graph,
    fn configure(self: Transaction, decode: bool) void {
        const g = self.graph;
        g.resident_route_gap = 0;
        g.resident_route_max_rank = 12;
        g.resident_route_max_mass = 1.0 / @as(f32, @floatFromInt(self.ops.config.shape.n_expert_used));
        g.resident_route_max_subs = 1;
        g.resident_route_max_total_mass = 0.20;
        g.resident_route_mode = apply_mode;
        g.resident_route_profile = decode and current().profile();
        g.resident_route_drop_weak = false;
        g.resident_route_native_weights = false;
        const native = g.resident_prefill_active or nativeWeights(c.getenv("DS4_METAL_V41_RESIDENT_ROUTE_NATIVE_WEIGHTS"));
        var value = c.getenv(if (g.resident_prefill_active) "DS4_METAL_V41_RESIDENT_PREFILL_GAP" else "DS4_METAL_V41_RESIDENT_ROUTE_GAP");
        if (native) {
            if (equals(c.getenv("DS4_METAL_V41_RESIDENT_ROUTE_POLICY"), "drop")) die("resident route native weights are incompatible with drop policy");
            if (!positiveGap(value)) die("resident route native weights require a positive configured gap");
        }
        if ((!decode and !g.resident_prefill_active) or value == null or value[0] == 0) return;
        const gap = mass(value, "resident route gap must be in [0,1]");
        if (gap == 0) {
            if (native) die("resident route native weights require a positive configured gap");
            return; // Mode-off arithmetic and execution remain native.
        }
        value = c.getenv("DS4_METAL_V41_RESIDENT_ROUTE_MODE");
        if (value != null and value[0] != 0) {
            if (equals(value, "native")) g.resident_route_mode = native_mode else if (equals(value, "lookup")) g.resident_route_mode = lookup_mode else if (equals(value, "noop")) g.resident_route_mode = noop_mode else if (!equals(value, "apply")) die("resident route mode must be native, lookup, noop or apply");
        }
        if (!g.streaming or g.quality or g.imatrix != null or g.image_count != 0 or g.tp_world != 1 or
            present("DS4_METAL_DISABLE_V41_SHORT_OPTIMIZATIONS") or present("DS4_METAL_V41_ASYNC_EXPERT_LOAD") or
            present("DS4_METAL_V41_GATE_PREFETCH") or current().oracle.active or
            present("DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD")) die("resident routing requires ordinary scalar text SSD early loading");
        value = c.getenv("DS4_METAL_V41_RESIDENT_ROUTE_POLICY");
        if (value != null and value[0] != 0) {
            if (equals(value, "drop")) {
                g.resident_route_drop_weak = true;
                g.resident_route_max_mass = 0.10;
            } else if (!equals(value, "replace")) die("resident route policy must be replace or drop");
        }
        value = c.getenv("DS4_METAL_V41_RESIDENT_ROUTE_MAX_RANK");
        if (value != null and value[0] != 0) {
            var end: [*c]u8 = null;
            const rank = c.strtoul(value, &end, 10);
            if (end == value or end[0] != 0 or rank < 7 or rank > self.ops.config.shape.n_expert) die("resident route max rank must be in [7,384]");
            g.resident_route_max_rank = @intCast(rank);
        }
        value = c.getenv("DS4_METAL_V41_RESIDENT_ROUTE_MAX_MASS");
        if (value != null and value[0] != 0) g.resident_route_max_mass = mass(value, "resident route max mass must be in [0,1]");
        value = c.getenv("DS4_METAL_V41_RESIDENT_ROUTE_MAX_SUBS");
        if (value != null and value[0] != 0) {
            var end: [*c]u8 = null;
            const count = c.strtoul(value, &end, 10);
            if (end == value or end[0] != 0 or count < 1 or count > 3) die("resident route max substitutions must be 1, 2 or 3");
            g.resident_route_max_subs = @intCast(count);
        }
        if (g.resident_route_drop_weak and g.resident_route_max_subs != 1) die("resident route drop policy permits one original weak miss only");
        value = c.getenv("DS4_METAL_V41_RESIDENT_ROUTE_MAX_TOTAL_MASS");
        if (g.resident_route_max_subs >= 2 and value != null and value[0] != 0) g.resident_route_max_total_mass = mass(value, "resident route maximum total mass must be in [0,1]");
        g.resident_route_gap = if (g.resident_route_mode == native_mode) 0 else gap;
        g.resident_route_native_weights = native;
        current().route_stats.native_weights_requested = current().route_stats.native_weights_requested or native;
        current().registerReport();
    }
    pub fn beginToken(self: Transaction, token: u32, decode: bool) bool {
        current().oracle.begin(self.ops, self.graph, token, closeOracle);
        self.graph.resident_prefill_active = false;
        self.configure(decode);
        return current().oracle.active;
    }
    pub fn endToken(self: Transaction, success: bool) bool {
        return current().oracle.end(self.ops, self.graph, success);
    }
    pub fn snapshot(self: Transaction, index: u32) c.ds4_native_moe_services {
        return .{ .profile = if (self.graph.resident_route_profile) &current().metrics.layer[index] else null, .oracle_deferred = current().oracle.active and current().oracle.defer_check };
    }
    pub fn probe(self: Transaction, layer: *const Layer, token: u32) bool {
        return current().oracle.probeRoutes(self.ops, self.graph, layer, token);
    }
    pub fn oraclePrefetch(self: Transaction, layer: *const Layer, index: u32) bool {
        return current().oracle.prefetch(self.ops, layer, index);
    }
    pub fn captureDeferred(self: Transaction, index: u32, selected: []i32) bool {
        return current().oracle.captureDeferred(self.ops, self.graph, index, selected);
    }
    pub fn observe(self: Transaction, table: *const c.ds4_gpu_stream_expert_table, index: u32, token: u32, selected: []const i32) void {
        current().oracle.observe(self.ops, table, index, token, selected);
    }
    /// Predicted IDs only warm the cache. Exact router IDs still determine
    /// every expert and coefficient used by model arithmetic.
    pub fn predict(self: Transaction, layer: *const Layer, index: u32, token: u32) bool {
        if (!self.graph.predict_active) return true;
        if (layer.ffn_exp_probs_b == null) return false;
        const table = oracle_owner.routeTable(self.ops, layer, index) orelse return false;
        var event: u64 = 0;
        return oracle_owner.select(self.ops, self.graph, layer, self.graph.predict_selected, token) and
            c.ds4_gpu_signal_selected_readback_ready(&event) != 0 and c.ds4_gpu_flush_commands() != 0 and
            c.ds4_native_moe_selected_start(&self.graph.predict_load, self.graph.predict_selected, self.ops.model, layer, index, event, table.gate_expert_bytes, table.down_expert_bytes);
    }
    /// Freeze all layers before selected-cache prefill can load or evict anything.
    /// Full layer sweeps/scalar fallback retain their native routes.
    pub fn beginPrefill(self: Transaction, weights: *const c.ds4_weights, selected: bool) bool {
        const g = self.graph;
        g.resident_prefill_active = false;
        const value = c.getenv("DS4_METAL_V41_RESIDENT_PREFILL_GAP");
        if (value == null or value[0] == 0) return true;
        const gap = mass(value, "resident prefill gap must be in [0,1]");
        if (gap == 0) return true;
        current().registerReport();
        if (!selected) {
            current().prefill_stats.fallbacks +%= 1;
            return true;
        }
        if (present("DS4_METAL_DISABLE_V41_BATCH_MOE") or present("DS4_METAL_DISABLE_V41_SELECTED_SMALL_MOE")) die("resident prefill routing requires selected batch MoE");
        g.resident_prefill_active = true;
        self.configure(false);
        for (weights.layer[0..self.ops.config.shape.n_layer], 0..) |*layer, index| {
            const table = cache.table(self.ops.config, self.ops.model, layer, @intCast(index), cache.rowBytes(@ptrCast(layer.ffn_gate_exps)) *% self.ops.config.shape.n_ff_exp, cache.rowBytes(@ptrCast(layer.ffn_down_exps)) *% self.ops.config.shape.n_embd);
            if (c.ds4_gpu_stream_expert_cache_snapshot(&table, &g.resident_prefill_snapshot[index]) == 0) return false;
            for (g.resident_prefill_snapshot[index][0..self.ops.config.shape.n_expert]) |resident| current().prefill_stats.frozen +%= @intFromBool(resident != 0);
        }
        current().prefill_stats.calls +%= 1;
        return true;
    }
    fn bias(self: Transaction, layer: *const Layer) ?[*]const f32 {
        if (layer.ffn_exp_probs_b == null) return null;
        return @ptrCast(@alignCast(self.ops.model.map + layer.ffn_exp_probs_b.*.abs_offset));
    }
    /// Only IDs change; native weights remain untouched. Every row uses the
    /// same immutable call-start mask, before any selected-cache mutation.
    pub fn prefill(self: Transaction, layer: *const Layer, index: u32, count: u32) bool {
        const g = self.graph;
        const shape = &self.ops.config.shape;
        if (!g.resident_prefill_active or g.resident_route_gap <= 0) return true;
        if (c.ds4_gpu_end_commands() == 0) return false;
        const raw_ids = c.ds4_gpu_tensor_contents(g.batch.selected) orelse return false;
        const raw_affinity = c.ds4_gpu_tensor_contents(g.batch.route_probs) orelse return false;
        const ids: [*]i32 = @ptrCast(@alignCast(raw_ids));
        const affinity: [*]const f32 = @ptrCast(@alignCast(raw_affinity));
        const bias_ = self.bias(layer) orelse return false;
        const resident = &g.resident_prefill_snapshot[index];
        var substitutions: u64 = 0;
        const expert_bytes = 2 *% cache.rowBytes(@ptrCast(layer.ffn_gate_exps)) *% shape.n_ff_exp +% cache.rowBytes(@ptrCast(layer.ffn_down_exps)) *% shape.n_embd;
        for (0..count) |row| {
            const selected = ids[row * shape.n_expert_used ..][0..shape.n_expert_used];
            current().prefill_stats.row_layers +%= 1;
            for (selected) |id| {
                if (id < 0 or @as(u32, @intCast(id)) >= shape.n_expert) return false;
                current().prefill_stats.native_misses +%= @intFromBool(resident[@intCast(id)] == 0);
            }
            if (g.resident_route_mode == lookup_mode) continue;
            var row_plan: Plan = undefined;
            if (!route.ds4_resident_route_choose_plan(selected.ptr, affinity + row * shape.n_expert, bias_, resident, shape.n_expert, shape.n_expert_used, g.resident_route_gap, g.resident_route_max_rank, g.resident_route_max_mass, g.resident_route_max_subs, g.resident_route_max_total_mass, &row_plan)) continue;
            current().prefill_stats.planned +%= row_plan.count;
            if (g.resident_route_mode == noop_mode) continue;
            for (row_plan.changes[0..row_plan.count]) |change| selected[change.slot] = @intCast(change.new_expert);
            substitutions +%= row_plan.count;
            current().prefill_stats.replaced_mass += row_plan.total_mass;
            current().prefill_stats.double_layers +%= @intFromBool(row_plan.count == 2);
            current().prefill_stats.triple_layers +%= @intFromBool(row_plan.count == 3);
        }
        current().prefill_stats.substitutions +%= substitutions;
        current().prefill_stats.skipped_bytes +%= substitutions *% expert_bytes;
        return c.ds4_gpu_begin_commands() != 0;
    }
    fn profile(self: Transaction, index: u32) ?*Metrics {
        return if (self.graph.resident_route_profile) &current().metrics.layer[index] else null;
    }
    const Decision = struct { plan: Plan, drop: c.ds4_resident_drop_change, affinity: [*]const f32 };
    fn plan(self: Transaction, layer: *const Layer, index: u32, table: *const c.ds4_gpu_stream_expert_table, selected: []const i32) error{Unavailable}!?Decision {
        const g = self.graph;
        const shape = &self.ops.config.shape;
        const metrics = self.profile(index);
        var started = if (metrics != null) now() else 0;
        var resident: [c.DS4_MAX_EXPERT]u8 = undefined;
        var selected_resident: [c.DS4_MAX_EXPERT_USED]u8 = undefined;
        const raw_affinity = c.ds4_gpu_tensor_contents(g.route_probs) orelse return error.Unavailable;
        const affinity: [*]const f32 = @ptrCast(@alignCast(raw_affinity));
        const bias_ = self.bias(layer) orelse return error.Unavailable;
        if (c.ds4_gpu_stream_expert_cache_snapshot_selected(table, selected.ptr, shape.n_expert_used, &selected_resident) == 0) return error.Unavailable;
        if (metrics) |p| p.lookup_ms += now() - started;
        current().route_stats.layers +%= 1;
        for (selected_resident[0..shape.n_expert_used]) |present_| current().route_stats.native_misses +%= @intFromBool(present_ == 0);
        started = if (metrics != null) now() else 0;
        var eligible = false;
        var drop = std.mem.zeroes(c.ds4_resident_drop_change);
        if (g.resident_route_drop_weak) {
            eligible = route.ds4_resident_route_choose_drop(selected.ptr, affinity, &selected_resident, shape.n_expert, shape.n_expert_used, g.resident_route_max_mass, &drop);
        } else if (g.resident_route_max_subs == 1) {
            var slot: u32 = undefined;
            var old_mass: f32 = undefined;
            eligible = route.ds4_resident_route_weakest(selected.ptr, affinity, shape.n_expert, shape.n_expert_used, g.resident_route_max_mass, &slot, &old_mass) and selected_resident[slot] == 0;
        } else {
            var slots: [3]u32 = undefined;
            var masses: [3]f32 = undefined;
            const admitted = if (g.resident_route_max_subs == 2)
                route.ds4_resident_route_weak_pair(selected.ptr, affinity, shape.n_expert, shape.n_expert_used, g.resident_route_max_mass, &slots, &masses)
            else
                route.ds4_resident_route_weak_triple(selected.ptr, affinity, shape.n_expert, shape.n_expert_used, g.resident_route_max_mass, &slots, &masses);
            for (slots[0..admitted], masses[0..admitted]) |slot, mass_| eligible = eligible or (selected_resident[slot] == 0 and mass_ <= g.resident_route_max_total_mass);
        }
        if (metrics) |p| {
            p.guard_ms += now() - started;
            p.eligible +%= @intFromBool(eligible);
        }
        if (!eligible or g.resident_route_mode == lookup_mode) return null;
        if (!g.resident_route_drop_weak) {
            started = if (metrics != null) now() else 0;
            if (c.ds4_gpu_stream_expert_cache_snapshot(table, &resident) == 0) return error.Unavailable;
            if (metrics) |p| {
                p.snapshot_ms += now() - started;
                p.snapshots +%= 1;
            }
            current().route_stats.full_snapshots +%= 1;
        }
        var proposed: Plan = std.mem.zeroes(Plan);
        started = if (metrics != null) now() else 0;
        var planned = true;
        if (g.resident_route_drop_weak) {
            proposed.count = 1;
            proposed.total_mass = drop.old_mass;
            proposed.changes[0] = .{ .slot = drop.slot, .old_expert = drop.old_expert, .new_expert = drop.alias_expert, .candidate_rank = 0, .old_mass = drop.old_mass, .score_gap = 0, .affinity_ratio = 0 };
        } else planned = route.ds4_resident_route_choose_plan(selected.ptr, affinity, bias_, &resident, shape.n_expert, shape.n_expert_used, g.resident_route_gap, g.resident_route_max_rank, g.resident_route_max_mass, g.resident_route_max_subs, g.resident_route_max_total_mass, &proposed);
        if (metrics) |p| {
            p.plan_ms += now() - started;
            p.plans +%= @intFromBool(planned);
            p.planned_substitutions +%= proposed.count;
        }
        if (!planned) return null;
        current().route_stats.planned +%= proposed.count;
        return .{ .plan = proposed, .drop = drop, .affinity = affinity };
    }
    fn noteCommit(self: Transaction, index: u32, table: *const c.ds4_gpu_stream_expert_table, proposed: Plan, native: bool) void {
        const s = &current().route_stats;
        s.substitutions +%= proposed.count;
        if (native) s.native_substitutions +%= proposed.count;
        s.per_layer[index] +%= proposed.count;
        s.double_layers +%= @intFromBool(proposed.count == 2);
        s.triple_layers +%= @intFromBool(proposed.count == 3);
        s.drops +%= @intFromBool(self.graph.resident_route_drop_weak);
        s.skipped_loads +%= proposed.count;
        s.skipped_bytes +%= proposed.count *% (2 *% table.gate_expert_bytes +% table.down_expert_bytes);
        for (proposed.changes[0..proposed.count]) |change| {
            s.replaced_mass += change.old_mass;
            s.score_gap += change.score_gap;
            s.affinity_ratio += change.affinity_ratio;
            s.rank += @floatFromInt(change.candidate_rank);
        }
    }
    // Called after router completion and before any selected loads or routed
    // kernels. The committed replacement belongs to the protected six-ID set;
    // no original read or late asynchronous rewrite is introduced.
    fn legacy(self: Transaction, layer: *const Layer, index: u32, table: *const c.ds4_gpu_stream_expert_table, selected: []i32) bool {
        if (self.graph.resident_route_gap <= 0) return true;
        const decision = (self.plan(layer, index, table, selected) catch return false) orelse return true;
        const proposed = decision.plan;
        const metrics = self.profile(index);
        const started = if (metrics != null) now() else 0;
        const affinity = decision.affinity;
        var final: [c.DS4_MAX_EXPERT_USED]i32 = undefined;
        @memcpy(final[0..selected.len], selected);
        for (proposed.changes[0..proposed.count]) |change| final[change.slot] = @intCast(change.new_expert);
        var weights: [c.DS4_MAX_EXPERT_USED]f32 = undefined;
        if (self.graph.resident_route_drop_weak)
            route.ds4_resident_route_drop_weights(&weights, selected.ptr, affinity, @intCast(selected.len), decision.drop.slot, self.ops.config.shape.expert_weight_scale)
        else
            route.ds4_resident_route_weights(&weights, &final, affinity, @intCast(selected.len), self.ops.config.shape.expert_weight_scale);
        // Preserve the legacy coefficient-normalization boundary. Plan-noop
        // computes the same arithmetic and makes every coefficient observable,
        // without changing the native IDs/weights or loader/cache behavior.
        if (self.graph.resident_route_mode == noop_mode) {
            for (weights[0..selected.len], 0..) |weight, slot| @as(*volatile f32, &current().noop_weights[slot]).* = weight;
            if (metrics) |p| p.commit_ms += now() - started;
            return true;
        }
        @memcpy(selected, final[0..selected.len]);
        const bytes = @as(u64, selected.len) * @sizeOf(i32);
        if (c.ds4_gpu_tensor_write(self.graph.selected, 0, selected.ptr, bytes) == 0 or
            c.ds4_gpu_tensor_write(self.graph.route_weights, 0, &weights, bytes) == 0) return false;
        if (metrics) |p| {
            p.commit_ms += now() - started;
            p.substitutions +%= proposed.count;
        }
        self.noteCommit(index, table, proposed, false);
        return true;
    }
    fn nativeWeightsOnly(self: Transaction, layer: *const Layer, index: u32, table: *const c.ds4_gpu_stream_expert_table, selected: []i32) bool {
        if (self.graph.resident_route_gap <= 0) return true;
        if (self.graph.resident_route_drop_weak) return false;
        const decision = (self.plan(layer, index, table, selected) catch return false) orelse return true;
        const proposed = decision.plan;
        const metrics = self.profile(index);
        const started = if (metrics != null) now() else 0;
        var final: [c.DS4_MAX_EXPERT_USED]i32 = undefined;
        @memcpy(final[0..selected.len], selected);
        for (proposed.changes[0..proposed.count]) |change| final[change.slot] = @intCast(change.new_expert);
        // ID-only noop does not recompute ANY native mixture coefficient.
        if (self.graph.resident_route_mode == noop_mode) {
            if (metrics) |p| p.commit_ms += now() - started;
            return true;
        }
        @memcpy(selected, final[0..selected.len]);
        if (c.ds4_gpu_tensor_write(self.graph.selected, 0, selected.ptr, @as(u64, selected.len) * @sizeOf(i32)) == 0) return false;
        if (metrics) |p| {
            p.commit_ms += now() - started;
            p.substitutions +%= proposed.count;
        }
        self.noteCommit(index, table, proposed, true);
        return true;
    }
    pub fn apply(self: Transaction, layer: *const Layer, index: u32, table: *const c.ds4_gpu_stream_expert_table, selected: []i32) bool {
        return if (self.graph.resident_route_native_weights) self.nativeWeightsOnly(layer, index, table, selected) else self.legacy(layer, index, table, selected);
    }
};
