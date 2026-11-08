//! SSD changes routed storage only. Static weights, live session graphs and
//! transient windows are reserved before the remaining budget becomes cache.
const c = @import("gguf_c");
const sizes = @import("../engine_cache_sizes.zig");
const gib: u64 = 1024 * 1024 * 1024;

pub fn staging(shape: *const c.ds4_shape, weights: *const c.ds4_weights, mtp: bool) ?u64 {
    if (shape.n_layer == 0 or shape.n_layer > c.DS4_MAX_LAYER or shape.n_nextn_predict >= shape.n_layer or shape.n_expert == 0 or shape.n_expert_used == 0 or shape.n_expert_used > shape.n_expert) return null;
    const trunk = shape.n_layer - shape.n_nextn_predict;
    const layers = if (mtp) shape.n_layer else trunk;
    const addresses = @as(u64, shape.n_expert) * 3 * 8;
    var reserve: u64 = 0;
    for (weights.layer[0..layers], 0..) |*layer, index| {
        const expert = sizes.layerBytes(layer) orelse return null;
        const full = expert *| shape.n_expert;
        // Trunk prefill can double-buffer full windows. A boosted predictor
        // uses compact selected slots, but addressless devices need one full
        // fallback window, so both paths are admitted on every machine.
        const window = if (index < trunk) full *| 2 else @max(full, (expert *| shape.n_expert_used +| addresses) *| 2);
        reserve = @max(reserve, window);
    }
    return reserve;
}

pub fn budget(host: u64, recommended: u64) u64 {
    return @min(host / 8 * 7, recommended);
}
fn fail(message: [*:0]const u8) bool {
    _ = c.fputs(message, c.__stderrp);
    return false;
}

pub fn admitForHost(core: *c.ds4_engine_core, config: *const c.ds4_qwen4_config, graph_bytes: u64, fit: bool, host: u64, recommended: u64) bool {
    if (!core.ssd_streaming) return true;
    if (host == 0 or recommended == 0) return fail("ds4: cannot determine Qwen SSD memory budget\n");
    const shape = &config.shape;
    var statics: u64 = 0;
    if (!c.ds4_native_model_spans_static_bytes(&core.weights, shape, true, true, &statics)) return false;
    const expert = sizes.sizeClass(shape, &core.weights) orelse return false;
    const maximum = sizes.cacheable(shape, &core.weights) orelse return false;
    const temporary = staging(shape, &core.weights, core.glm_mtp) orelse return false;
    const fixed = statics +| core.vision_model.size +| graph_bytes +| temporary +| (2 * gib);
    const safe = budget(host, recommended);
    if (fixed >= safe or safe - fixed < expert) return fail("ds4: Qwen SSD weights, graph and staging exceed safe memory budget; reduce context or prefill chunk\n");
    var count = @min((safe - fixed) / expert, maximum.experts);
    if (!fit) {
        if (core.memory.ssd_streaming_cache_experts > count) return fail("ds4: no room for another Qwen SSD session; reduce context or expert cache\n");
        return true;
    }
    var requested: u64 = core.memory.ssd_streaming_cache_experts;
    if (core.memory.ssd_streaming_cache_bytes != 0) {
        const target = core.memory.ssd_streaming_cache_bytes;
        if (target <= temporary or target - temporary < expert) return fail("ds4: Qwen SSD cache target must cover staging and at least one expert\n");
        requested = (target - temporary) / expert;
    }
    if (requested != 0) count = @min(count, requested);
    core.memory.ssd_streaming_cache_experts = @intCast(count);
    core.memory.ssd_streaming_cache_bytes = count * expert;
    core.memory.ssd_streaming_prefill_headroom_bytes = temporary;
    core.ssd_streaming_static_decode_map = true;
    core.memory.ssd_streaming_decode_map_bytes = statics;
    _ = c.fprintf(c.__stderrp, "ds4: Qwen SSD: %.2f GiB static weights, %.2f GiB graph reserve, %.2f GiB staging, %u cached experts (%.2f GiB)\n", @as(f64, @floatFromInt(statics)) / gib, @as(f64, @floatFromInt(graph_bytes)) / gib, @as(f64, @floatFromInt(temporary)) / gib, core.memory.ssd_streaming_cache_experts, @as(f64, @floatFromInt(core.memory.ssd_streaming_cache_bytes)) / gib);
    return true;
}

pub fn admit(core: *c.ds4_engine_core, config: *const c.ds4_qwen4_config, graph_bytes: u64, fit: bool) bool {
    var host: u64 = 0;
    var size: usize = @sizeOf(u64);
    if (c.sysctlbyname("hw.memsize", &host, &size, null, 0) != 0) host = 0;
    return admitForHost(core, config, graph_bytes, fit, host, c.ds4_gpu_recommended_working_set_size());
}
