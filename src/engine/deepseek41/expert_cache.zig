//! Expert-cache format and prefill seeding. Model weights stay borrowed; the
//! existing GPU cache owns its slabs. No second copy or whole-layer read is
//! introduced here: seeding uses the layer that is already mapped for prefill.
const std = @import("std");
const c = @import("gguf_c");
const Config = @import("../config.zig").ModelConfig;
const Graph = c.ds41_gpu_graph;
const Model = c.ds4_model;
const Layer = c.ds4_layer_weights;
const Weight = c.ds4_tensor;
const Weights = c.ds4_weights;

pub fn rowBytes(weight: *const Weight) u64 {
    const info = c.ds4_gguf_tensor_type(weight.type);
    if (info == null or info.*.block_elems == 0)
        c.ds4_engine_model_die("unsupported routed expert tensor type");
    if (weight.dim[0] % info.*.block_elems != 0)
        c.ds4_engine_model_die("routed expert row is not quant block aligned");
    switch (weight.type) {
        c.DS4_TENSOR_Q8_0, c.DS4_TENSOR_IQ2_XXS, c.DS4_TENSOR_Q2_K, c.DS4_TENSOR_Q4_K, c.DS4_TENSOR_Q5_K, c.DS4_TENSOR_Q6_K, c.DS4_TENSOR_MXFP4 => {},
        else => c.ds4_engine_model_die("unsupported routed expert tensor type"),
    }
    return (weight.dim[0] / info.*.block_elems) *% info.*.block_bytes;
}

pub fn layerBytes(layer: *const Layer) ?u64 {
    const tensors = [_][*c]const Weight{ layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps };
    var total: u64 = 0;
    for (tensors) |tensor| {
        if (tensor == null) return null;
        const weight: *const Weight = @ptrCast(tensor);
        const bytes = rowBytes(weight);
        if (weight.dim[1] > std.math.maxInt(u64) / bytes) return null;
        const extent = weight.dim[1] * bytes;
        if (extent > std.math.maxInt(u64) - total) return null;
        total += extent;
    }
    return if (total != 0) total else null;
}

pub fn sizeClass(config: *const Config, weights: *const Weights) ?u64 {
    // Mixed-precision distributed slices may start with an outlier format.
    // Use the most common local size class, retaining the first on ties.
    const layers = weights.layer[0..config.shape.n_layer];
    var best_bytes: u64 = 0;
    var best_count: u32 = 0;
    for (layers) |*layer| {
        const candidate = layerBytes(layer) orelse continue;
        var count: u32 = 0;
        for (layers) |*other| {
            if (layerBytes(other)) |bytes| if (bytes == candidate) {
                count += 1;
            };
        }
        if (count > best_count) {
            best_bytes = candidate;
            best_count = count;
        }
    }
    return if (best_count != 0) best_bytes else null;
}

pub fn table(config: *const Config, model: *const Model, layer: *const Layer, index: u32, gate_bytes: u64, down_bytes: u64) c.ds4_gpu_stream_expert_table {
    return .{
        .model_map = model.map,
        .model_size = model.size,
        .layer = index,
        .n_total_expert = config.shape.n_expert,
        .gate_offset = if (layer.ffn_gate_exps == null) 0 else layer.ffn_gate_exps.*.abs_offset,
        .up_offset = if (layer.ffn_up_exps == null) 0 else layer.ffn_up_exps.*.abs_offset,
        .down_offset = if (layer.ffn_down_exps == null) 0 else layer.ffn_down_exps.*.abs_offset,
        .gate_expert_bytes = gate_bytes,
        .down_expert_bytes = down_bytes,
    };
}

pub fn seed(config: *const Config, graph: *Graph, model: *const Model, layer: *const Layer, index: u32, count: u32) bool {
    if (!graph.streaming or c.getenv("DS4_METAL_DISABLE_STREAMING_PREFILL_CACHE_SEED") != null) return true;
    const shape = &config.shape;
    const target: u32 = @min(c.ds4_gpu_stream_expert_cache_configured_count() / shape.n_layer, shape.n_expert);
    if (target == 0) return true;
    const memory = c.ds4_gpu_tensor_contents(graph.batch.selected) orelse return false;
    const selected: [*]const i32 = @ptrCast(@alignCast(memory));
    var frequency: [c.DS4_MAX_EXPERT]u32 = @splat(0);
    const recent: u32 = @min(count, 32);
    for (0..count) |token| {
        for (0..shape.n_expert_used) |slot| {
            const id = selected[token * shape.n_expert_used + slot];
            if (id < 0 or @as(u32, @intCast(id)) >= shape.n_expert) return false;
            frequency[@intCast(id)] +%= 1 +% (if (token >= count - recent) count *% shape.n_expert_used else 0);
        }
    }
    var experts: [c.DS4_MAX_EXPERT]i32 = undefined;
    var priority: [c.DS4_MAX_EXPERT]u32 = undefined;
    var count_seed: u32 = 0;
    while (count_seed < target) : (count_seed += 1) {
        var best: u32 = 0;
        for (1..shape.n_expert) |expert| {
            if (frequency[expert] > frequency[best]) best = @intCast(expert);
        }
        if (frequency[best] == 0) break;
        experts[count_seed] = @intCast(best);
        // Rank is a preload preference, not thousands of observed routes.
        priority[count_seed] = 1 + @as(u32, @intCast(31 * @as(u64, target - count_seed) / target));
        frequency[best] = 0;
    }
    if (count_seed == 0) return true;
    const source = table(config, model, layer, index, rowBytes(layer.ffn_gate_exps) *% shape.n_ff_exp, rowBytes(layer.ffn_down_exps) *% shape.n_embd);
    if (c.ds4_gpu_begin_commands() == 0) return false;
    const ok = c.ds4_gpu_stream_expert_cache_seed_experts_gpu_copy(&source, &experts, &priority, count_seed) != 0;
    const ended = c.ds4_gpu_end_commands() != 0;
    return ended and ok;
}
