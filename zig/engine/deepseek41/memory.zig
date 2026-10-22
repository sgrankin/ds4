//! V4.1 Engine memory admission and scoped workspace/cache rebalancing.
//! The actual accounting record is embedded once in the C Engine. Config,
//! weights and per-call resource facts are borrowed; no Engine layout enters
//! this owner. Other-family cache/startup policies remain C.
const std = @import("std");
const c = @import("gguf_c");
const Config = @import("../config.zig").ModelConfig;
const experts = @import("expert_cache.zig");
const streaming = @import("streaming.zig");
const resources = @import("resources.zig");
const Layout = @import("layout.zig").Layout;
const gib: u64 = 1073741824;

fn asGiB(bytes: u64) f64 {
    @setFloatMode(.optimized);
    return @as(f64, @floatFromInt(bytes)) / gib;
}
fn message(destination: []u8, text: []const u8) void {
    if (destination.len == 0) return;
    const count = @min(destination.len - 1, text.len);
    @memcpy(destination[0..count], text[0..count]);
    destination[count] = 0;
}

pub const Memory = struct {
    state: *c.ds4_engine_memory,
    config: *const Config,
    weights: *const c.ds4_weights,
    inputs: c.ds4_engine_memory_inputs,

    /// Explicit NGB budgets name only the routed cache. Reserve the graph
    /// first and round down to GiB, preserving the one-GiB minimum. This is a
    /// working-set pressure limit, deliberately not an mlock-derived cap.
    pub fn safeCacheBytes(self: *const Memory, context: c_int, chunk: u32, recommended: u64) u64 {
        if (recommended == 0) return 0;
        const target = if (recommended > std.math.maxInt(u64) / 7)
            std.math.maxInt(u64)
        else
            recommended * 7 / 8;
        const rows: u32 = if (context > 0) @intCast(context) else 1;
        const layout = Layout.init(self.config, rows, chunk);
        const graph_bytes = layout.bytes();
        const available = if (target > graph_bytes) target - graph_bytes else 0;
        const rounded = available / gib * gib;
        return if (rounded == 0) gib else rounded;
    }

    /// This phase remains before inspect-only return in Engine open. The
    /// existing GPU recommendation API may initialize Metal, just as in C.
    pub fn capRequestedCache(self: *const Memory, context: c_int, chunk: u32) void {
        @setFloatMode(.optimized);
        if (!self.inputs.ssd_streaming or self.state.ssd_streaming_cache_bytes == 0) return;
        const requested = self.state.ssd_streaming_cache_bytes;
        const safe = self.safeCacheBytes(context, chunk, c.ds4_gpu_recommended_working_set_size());
        if (safe != 0 and requested > safe) {
            self.state.ssd_streaming_cache_bytes = safe;
            _ = c.fprintf(c.__stderrp, "ds4: metal SSD streaming cache budget %.2f GiB capped to %.2f GiB to stay below the graph working-set pressure budget\n", asGiB(requested), asGiB(safe));
        }
    }

    // The cache has one slab size. Boosted/outlier layers remain direct mapped
    // reads; only layers matching the dominant local class contribute slots.
    fn cacheableLayers(self: *const Memory, per_expert: u64) ?u32 {
        if (self.config.shape.n_expert == 0) return null;
        var count: u32 = 0;
        for (self.weights.layer[0..self.config.shape.n_layer]) |*layer| {
            if (experts.layerBytes(layer)) |bytes| if (bytes == per_expert) {
                count += 1;
            };
        }
        if (count == 0 or count > std.math.maxInt(u64) / @as(u64, self.config.shape.n_expert)) return null;
        return count;
    }

    pub fn automaticCache(self: *const Memory, context: c_int, chunk: u32) bool {
        @setFloatMode(.optimized);
        if (!self.inputs.ssd_streaming or self.state.ssd_streaming_cache_experts != 0 or
            self.state.ssd_streaming_cache_bytes != 0) return true;
        const recommended = c.ds4_gpu_recommended_working_set_size();
        if (recommended == 0) {
            _ = c.fprintf(c.__stderrp, "ds4: SSD streaming auto cache: recommended working set unavailable; set --ssd-streaming-cache-experts N or NGB explicitly\n");
            return false;
        }
        const non_routed = streaming.staticBytes(self.config, self.weights, self.inputs.include_token, self.inputs.include_output) orelse {
            _ = c.fprintf(c.__stderrp, "ds4: SSD streaming auto cache could not measure non-routed model weights\n");
            return false;
        };
        const per_expert = experts.sizeClass(self.config, self.weights) orelse {
            _ = c.fprintf(c.__stderrp, "ds4: SSD streaming auto cache could not measure routed expert size\n");
            return false;
        };
        const layers = self.cacheableLayers(per_expert) orelse {
            _ = c.fprintf(c.__stderrp, "ds4: SSD streaming auto cache could not count local cacheable experts\n");
            return false;
        };
        const max_experts = @as(u64, layers) * self.config.shape.n_expert;
        // More cached experts help only if fixed weights stay resident too.
        // Share the existing static-pinning budget, including requested context.
        const model_limit = self.safeCacheBytes(if (context > 0) context else 4096, chunk, recommended);
        var plan: c.ds4_ssd_cache_plan = undefined;
        if (!c.ds4_ssd_auto_cache_plan(recommended, 86, model_limit, non_routed, per_expert, max_experts, &plan)) {
            _ = c.fprintf(c.__stderrp, "ds4: SSD streaming auto cache could not compute a valid cache budget\n");
            return false;
        }
        self.state.ssd_streaming_cache_experts = plan.cache_experts;
        self.state.ssd_streaming_cache_bytes = plan.effective_cache_bytes;
        _ = c.fprintf(c.__stderrp, "ds4: SSD streaming auto cache budget\n");
        _ = c.fprintf(c.__stderrp, "ds4:   metal recommends %.2f GiB working set\n", asGiB(recommended));
        _ = c.fprintf(c.__stderrp, "ds4:   using %.0f%% total for model + cached experts: %.2f GiB\n", 100.0 * @as(f64, @floatFromInt(plan.model_target_bytes)) / @as(f64, @floatFromInt(recommended)), asGiB(plan.model_target_bytes));
        _ = c.fprintf(c.__stderrp, "ds4:   non-routed weights: %.2f GiB\n", asGiB(non_routed));
        _ = c.fprintf(c.__stderrp, "ds4:   routed expert size: %.2f MiB\n", @as(f64, @floatFromInt(per_expert)) / 1048576.0);
        _ = c.fprintf(c.__stderrp, "ds4:   expert budget before prefill reserve: %u (%.2f GiB)\n", plan.cache_experts, asGiB(plan.effective_cache_bytes));
        if (plan.model_target_bytes <= non_routed)
            _ = c.fprintf(c.__stderrp, "ds4:   note: non-routed weights already fill the target; keeping a one-expert cache\n");
        return true;
    }

    /// A selected-address batch can touch many experts in one layer, rather
    /// than just one decode token's top-k. Reserve up to two cacheable layers
    /// before converting a byte target into complete dynamic expert slots.
    pub fn reservePrefill(self: *const Memory, full_layers_set: bool) bool {
        @setFloatMode(.optimized);
        self.state.ssd_streaming_decode_map_bytes = 0;
        self.state.ssd_streaming_decode_map_extra_bytes = 0;
        if (!self.inputs.ssd_streaming) return true;
        if (full_layers_set and self.state.ssd_streaming_full_layers != 0) {
            _ = c.fprintf(c.__stderrp, "ds4: --ssd-streaming-full-layers is currently supported only for GLM graph SSD streaming; ignoring\n");
            self.state.ssd_streaming_full_layers = 0;
        }
        const need_expert = self.state.ssd_streaming_cache_experts != 0 or
            self.state.ssd_streaming_cache_bytes != 0 or self.state.ssd_streaming_full_layers != 0;
        const per_expert = if (need_expert) experts.sizeClass(self.config, self.weights) orelse {
            _ = c.fprintf(c.__stderrp, "ds4: SSD streaming could not measure routed expert size\n");
            return false;
        } else 0;
        const layers: u32 = if (need_expert) self.cacheableLayers(per_expert) orelse {
            _ = c.fprintf(c.__stderrp, "ds4: SSD streaming could not count local cacheable experts\n");
            return false;
        } else 0;
        const max_experts: u32 = @intCast(@min(@as(u64, layers) * self.config.shape.n_expert, std.math.maxInt(u32)));
        if (self.state.ssd_streaming_cache_bytes == 0 and max_experts != 0 and
            self.state.ssd_streaming_cache_experts > max_experts)
        {
            _ = c.fprintf(c.__stderrp, "ds4: SSD streaming expert cache capped from %u to %u experts cacheable by this layer slice\n", self.state.ssd_streaming_cache_experts, max_experts);
            self.state.ssd_streaming_cache_experts = max_experts;
        }
        const target = self.state.ssd_streaming_cache_bytes;
        var headroom: u64 = 0;
        if (target != 0) {
            const reserve_layers: u32 = @min(layers, c.DS4_STREAMING_PREFILL_HEADROOM_LAYERS);
            if (per_expert > std.math.maxInt(u64) / @as(u64, self.config.shape.n_expert)) {
                _ = c.fprintf(c.__stderrp, "ds4: SSD streaming prefill headroom byte accounting failed\n");
                return false;
            }
            const layer_bytes = per_expert * self.config.shape.n_expert;
            if (reserve_layers != 0 and layer_bytes > std.math.maxInt(u64) / @as(u64, reserve_layers)) {
                _ = c.fprintf(c.__stderrp, "ds4: SSD streaming prefill headroom byte accounting failed\n");
                return false;
            }
            headroom = layer_bytes * reserve_layers;
            if (headroom >= target) {
                const fitted = if (target > per_expert) target - per_expert else 0;
                _ = c.fprintf(c.__stderrp, "ds4: SSD streaming prefill reserve reduced from %.2f to %.2f GiB to honor the %.2f GiB cache target\n", asGiB(headroom), asGiB(fitted), asGiB(target));
                headroom = fitted;
            }
        }
        self.state.ssd_streaming_prefill_headroom_bytes = headroom;
        // Even a direct caller that omitted the option's `set` bit has zero
        // supported GLM resident layers in a V4.1 configuration, as in C.
        if (self.state.ssd_streaming_full_layers != 0)
            _ = c.fprintf(c.__stderrp, "ds4: GLM SSD streaming full resident layers capped from %u to %u supported routed prefix layers\n", self.state.ssd_streaming_full_layers, @as(c_uint, 0));
        if (target != 0) {
            var budget = c.ds4_ssd_cache_experts_for_byte_budget(target - headroom, per_expert);
            if (max_experts != 0) budget = @min(budget, max_experts);
            var slot_bytes = per_expert;
            if (budget == 0 or slot_bytes == 0) {
                _ = c.fprintf(c.__stderrp, "ds4: SSD streaming cache target leaves no complete expert slot; using direct per-layer reads\n");
                budget = 0;
                slot_bytes = 0;
            }
            self.state.ssd_streaming_cache_experts = budget;
            self.state.ssd_streaming_cache_bytes = @as(u64, budget) * slot_bytes;
            const total = headroom +| self.state.ssd_streaming_cache_bytes;
            _ = c.fprintf(c.__stderrp, "ds4: metal SSD streaming cache target %.2f GiB; effective %.2f GiB = %.2f GiB prefill headroom + %.2f GiB dynamic cache (%u experts, %.2f MiB each)\n", asGiB(target), asGiB(total), asGiB(headroom), asGiB(self.state.ssd_streaming_cache_bytes), budget, @as(f64, @floatFromInt(slot_bytes)) / 1048576.0);
        }
        self.state.ssd_streaming_full_layers = 0;
        self.state.ssd_streaming_full_layer_bytes = 0;
        return true;
    }

    /// Keep the same 7/8 host guard, capped by Metal's recommendation.
    /// fit_cache may shrink the configured cache; failed admission does not
    /// change it. Additional sessions use fit_cache=false to preserve it.
    pub fn admitForHost(self: *const Memory, graph_bytes_: u64, fit_cache: bool, host: u64, recommended: u64) bool {
        var budget = host / 8 * 7;
        if (host == 0 or recommended == 0) {
            _ = c.fprintf(c.__stderrp, "ds4: cannot determine a safe V4.1 memory budget\n");
            return false;
        }
        budget = @min(budget, recommended);
        var weights = self.inputs.target_model_bytes;
        if (self.inputs.ssd_streaming) {
            weights = streaming.staticBytes(self.config, self.weights, self.inputs.include_token, self.inputs.include_output) orelse return false;
        }
        weights +|= self.inputs.vision_model_bytes;
        weights +|= self.inputs.mtp_model_bytes;
        var graph_bytes = graph_bytes_;
        if (self.inputs.dspark) graph_bytes +|= (@as(u64, 64) << 20) *| self.inputs.session_count;
        const fixed = weights +| (graph_bytes +| (2 * gib +% self.state.ssd_streaming_prefill_headroom_bytes));
        const expert = if (self.inputs.ssd_streaming) experts.sizeClass(self.config, self.weights) orelse return false else 0;
        if (fixed >= budget or (expert != 0 and budget - fixed < expert)) {
            _ = c.fprintf(c.__stderrp, "ds4: V4.1 needs %.2f GiB before the expert cache; safe budget %.2f GiB. Use --ssd-streaming or a smaller context.\n", asGiB(fixed), asGiB(budget));
            return false;
        }
        if (expert != 0 and self.state.ssd_streaming_cache_experts > (budget - fixed) / expert) {
            if (!fit_cache) {
                _ = c.fprintf(c.__stderrp, "ds4: no room for another V4.1 session; reduce context or the SSD expert cache\n");
                return false;
            }
            const count: u32 = @intCast((budget - fixed) / expert);
            _ = c.fprintf(c.__stderrp, "ds4: V4.1 SSD cache fitted from %u to %u experts for context/runtime headroom\n", self.state.ssd_streaming_cache_experts, count);
            self.state.ssd_streaming_cache_experts = count;
            self.state.ssd_streaming_cache_bytes = @as(u64, count) *% expert;
        }
        return true;
    }

    pub fn admit(self: *const Memory, graph_bytes: u64, fit_cache: bool) bool {
        var host: u64 = 0;
        var size: usize = @sizeOf(u64);
        if (c.sysctlbyname("hw.memsize", &host, &size, null, 0) != 0) host = 0;
        return self.admitForHost(graph_bytes, fit_cache, host, c.ds4_gpu_recommended_working_set_size());
    }

    /// Return borrowed expert-cache bytes BEFORE allocating larger workspace
    /// buffers, even when the broad host budget still has headroom. Only an
    /// admission failure rolls the cache back; partial allocation is charged
    /// conservatively and invalidates the graph, as in the original controller.
    pub fn growWorkspace(self: *const Memory, graph: *c.ds41_gpu_graph, need: u32, remaining: u32, errors: []u8) bool {
        const old_bytes = graph.allocation_bytes;
        const plan = Layout.init(self.config, graph.ctx, need);
        const new_bytes = plan.bytes();
        const old_budget = self.state.ssd_streaming_cache_experts;
        const old_cache_bytes = self.state.ssd_streaming_cache_bytes;
        if (c.ds4_gpu_synchronize() == 0) {
            message(errors, "cannot drain V4.1 workspace users");
            return false;
        }
        if (graph.streaming and old_budget != 0) {
            const expert = experts.sizeClass(self.config, self.weights) orelse {
                message(errors, "cannot size V4.1 expert cache for workspace growth");
                return false;
            };
            const slots = (new_bytes -% old_bytes +% expert -% 1) / expert;
            self.state.ssd_streaming_cache_experts = if (slots < old_budget) old_budget - @as(u32, @intCast(slots)) else 1;
            self.state.ssd_streaming_cache_bytes = @as(u64, self.state.ssd_streaming_cache_experts) *% expert;
        }
        const next_bytes = (self.state.ds41_session_bytes -% old_bytes) +| new_bytes;
        if (!self.admit(next_bytes, true)) {
            self.state.ssd_streaming_cache_experts = old_budget;
            self.state.ssd_streaming_cache_bytes = old_cache_bytes;
            message(errors, "cannot admit larger V4.1 workspace");
            return false;
        }
        if (self.state.ssd_streaming_cache_experts != old_budget)
            c.ds4_gpu_set_streaming_expert_cache_budget(self.state.ssd_streaming_cache_experts);
        _ = c.fprintf(c.__stderrp, "ds4: V4.1 workspace grows %u -> %u rows before %u-token prefill\n", graph.prefill_cap, need, remaining);
        self.state.ds41_session_bytes = self.state.ds41_session_bytes -% old_bytes +% new_bytes;
        graph.allocation_bytes = new_bytes;
        if (!resources.growRecord(graph, self.config, need)) {
            graph.workspace_failed = true;
            graph.valid = false;
            message(errors, "cannot grow V4.1 workspace");
            return false;
        }
        return true;
    }
};
