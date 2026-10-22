//! Scoped reuse of the admitted decode-cache budget for long encoder sweeps.
//! The model mapping remains borrowed. Every exit releases pinned pages before
//! restoring the cache budget, including cancellation and partial mlock failure.
const std = @import("std");
const c = @import("gguf_c");
const cache = @import("expert_cache.zig");
const Config = @import("../config.zig").ModelConfig;
const Graph = c.ds41_gpu_graph;
const Model = c.ds4_model;
const Weights = c.ds4_weights;

// One actual C lease record is shared with the existing graph fixtures.
// It owns only page locks and the temporarily borrowed expert-cache budget.
pub const Lease = c.ds41_encoder_residency;

pub fn release(lease: *Lease, graph: *Graph, model: *const Model) void {
    if (lease.cache_budget == 0) return;
    for (&lease.span) |*span| {
        if (span.locked == 0) continue;
        const pointer: *anyopaque = @ptrCast(@constCast(model.map + span.off));
        if (c.munlock(pointer, @intCast(span.locked)) != 0)
            c.ds4_engine_model_die("cannot release V4.1 encoder pages safely before restoring the expert cache");
        _ = c.madvise(pointer, @intCast(span.locked), c.MADV_DONTNEED);
        span.locked = 0;
    }
    graph.encoder_resident = false;
    c.ds4_gpu_set_streaming_expert_cache_budget(lease.cache_budget);
    lease.cache_budget = 0;
}

pub fn acquire(lease: *Lease, graph: *Graph, model: *const Model, weights: *const Weights, config: *const Config, remaining: u32, cancel: c.ds4_session_cancel_fn, userdata: ?*anyopaque) void {
    @setFloatMode(.optimized);
    if (!graph.streaming or graph.tp_world != 1 or graph.imatrix != null or remaining < 16384 or
        c.getenv("DS4_METAL_DISABLE_V41_LAYER_PREFILL") != null or
        c.getenv("DS4_METAL_DISABLE_V41_ENCODER_RESIDENCY") != null) return;
    // A wide sweep reads each encoder layer once already, without pinning
    // a second long-lived set of experts or discarding the decode cache.
    if (graph.carry_cap != 0 and c.getenv("DS4_METAL_DISABLE_V41_WIDE_PREFILL") == null) return;
    const budget = c.ds4_gpu_stream_expert_cache_configured_count();
    const expert_bytes = cache.sizeClass(config, weights) orelse return;
    const page: u64 = @intCast(c.sysconf(c._SC_PAGESIZE));
    var bytes: u64 = 0;
    for (0..20) |layer| {
        const tensors = [_][*c]const c.ds4_tensor{ weights.layer[layer].ffn_gate_exps, weights.layer[layer].ffn_up_exps, weights.layer[layer].ffn_down_exps };
        for (tensors, 0..) |raw, member| {
            if (raw == null) return;
            const tensor: *const c.ds4_tensor = @ptrCast(raw);
            // Exclusive whole pages cannot unlock neighboring static
            // weights. The V4.1 converter aligns these tensors to 16KiB.
            if (tensor.abs_offset % page != 0 or tensor.bytes % page != 0 or
                tensor.abs_offset > model.size or tensor.bytes > model.size - tensor.abs_offset or
                tensor.bytes > std.math.maxInt(u64) - bytes) return;
            const span = &lease.span[layer * 3 + member];
            span.off = tensor.abs_offset;
            span.bytes = tensor.bytes;
            bytes += tensor.bytes;
        }
    }
    if (expert_bytes == 0 or bytes > @as(u64, budget) *% expert_bytes or c.ds4_gpu_synchronize() == 0) return;
    lease.cache_budget = budget;
    c.ds4_gpu_set_streaming_expert_cache_budget(0);
    _ = c.fprintf(c.__stderrp, "ds4: V4.1 prefill loading %.2f GiB encoder experts in place of the decode cache\n", @as(f64, @floatFromInt(bytes)) / 1073741824.0);
    for (&lease.span) |*span| {
        while (span.locked < span.bytes) {
            const count: u64 = @min(span.bytes - span.locked, @as(u64, 256) << 20);
            if (cancel) |check| if (check(userdata)) {
                release(lease, graph, model);
                return;
            };
            const pointer: *anyopaque = @ptrCast(@constCast(model.map + span.off + span.locked));
            if (c.mlock(pointer, @intCast(count)) != 0) {
                _ = c.fprintf(c.__stderrp, "ds4: V4.1 encoder residency unavailable (%s); using two-layer prefill\n", c.strerror(c.__error().*));
                release(lease, graph, model);
                return;
            }
            span.locked += count;
        }
    }
    graph.encoder_resident = true;
    _ = c.fprintf(c.__stderrp, "ds4: V4.1 prefill keeps %.2f GiB encoder experts resident in place of the decode cache\n", @as(f64, @floatFromInt(bytes)) / 1073741824.0);
}
