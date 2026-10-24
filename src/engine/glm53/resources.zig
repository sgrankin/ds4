//! Own GLM-5.3's bounded mHC/KDA prefill workspace and recurrent-state reset.
//! These handles are distinct allocations, not aliases of scalar decode scratch.
const c = @import("gguf_c");
const std = @import("std");
const Config = @import("config.zig");
const Graph = c.ds4_glm_gpu_graph;
const Tensor = c.ds4_gpu_tensor;

/// Native Metal graphs have no CUDA per-tier scratch mirrors. The remaining
/// C multi-tier owner must restore its original handles before releasing them.
/// Every single Tensor field below owns its handle, including retained views;
/// transport slabs (tp_out/tp_in), weights and imatrix remain borrowed.
pub fn deinit(maybe_graph: ?*Graph) bool {
    const graph = maybe_graph orelse return true;
    if (graph.placement != null) return false;
    const record = @typeInfo(Graph).@"struct";
    inline for (record.field_names, record.field_types) |name, Field| {
        if (Field == ?*Tensor) c.ds4_gpu_tensor_free(@field(graph, name));
    }
    inline for (.{
        "directional_steering_dirs_by_tier", "layer_kda_conv_state", "layer_kda_recurrent_state",
        "layer_indexer_key_cache",           "layer_indexer_tail_k", "layer_indexer_tail_gate",
        "layer_k_rope_cache",                "layer_kv_lora_cache",  "layer_value_cache",
        "layer_key_cache",
    }) |field| {
        for (@field(graph, field)) |tensor| c.ds4_gpu_tensor_free(tensor);
    }
    c.free(graph.mtp_logits_host);
    graph.* = std.mem.zeroes(Graph);
    return true;
}

const prefill_fields = .{
    "batch_hc_cur",          "batch_hc_next", "batch_hc_flat", "batch_hc_mix",      "batch_hc_split",     "batch_hc_after_attn",
    "batch_kda_q",           "batch_kda_k",   "batch_kda_v",   "batch_kda_lowrank", "batch_kda_raw_gate", "batch_kda_raw_beta",
    "batch_kda_output_gate", "batch_kda_out",
};

pub fn freePrefill(graph: *Graph) void {
    inline for (prefill_fields) |field| {
        c.ds4_gpu_tensor_free(@field(graph, field));
        @field(graph, field) = null;
    }
    graph.glm53_prefill_cap = 0;
}

pub fn ensurePrefill(graph: *Graph, rows: u32) bool {
    if (!graph.glm53 or rows == 0 or rows > graph.config.prefill_chunk) return false;
    if (graph.glm53_prefill_cap >= rows) return true;
    var capacity: u32 = 1;
    while (capacity < rows) capacity *= 2;
    const config = &graph.config;
    const hc: u64 = Config.hcWidth(config);
    const mix: u64 = Config.hcMix(config);
    const projection: u64 = Config.projection(config);
    const widths = .{ hc, hc, hc, mix, mix, hc, projection, projection, projection, @as(u64, config.shape.n_kda_head_dim), projection, @as(u64, config.shape.n_kda_head), projection, projection };

    // As in the old owner, a failed growth releases the whole prefill workspace.
    // The caller must invalidate partially advanced live state before retrying.
    freePrefill(graph);
    inline for (prefill_fields, widths) |field, width| {
        @field(graph, field) = c.ds4_gpu_tensor_alloc(@as(u64, capacity) * width * @sizeOf(f32));
        if (@field(graph, field) == null) {
            freePrefill(graph);
            return false;
        }
    }
    graph.glm53_prefill_cap = capacity;
    return true;
}

pub fn ensureBatchLogits(graph: *Graph, rows: u32) bool {
    if (!graph.glm53 or rows == 0) return false;
    if (graph.session_batch_logits_cap >= rows and graph.session_batch_logits != null) return true;
    var capacity: u32 = 1;
    while (capacity < rows) {
        if (capacity > std.math.maxInt(u32) / 2) return false;
        capacity *= 2;
    }
    const tensor = c.ds4_gpu_tensor_alloc(@as(u64, capacity) * graph.config.shape.n_vocab * @sizeOf(f32)) orelse return false;
    c.ds4_gpu_tensor_free(graph.session_batch_logits);
    graph.session_batch_logits = tensor;
    graph.session_batch_logits_cap = capacity;
    return true;
}

pub fn resetKda(maybe_graph: ?*Graph) bool {
    const graph = maybe_graph orelse return true;
    if (!graph.glm53) return true;
    for (graph.layer_start..graph.layer_end + 1) |layer| {
        const conv = graph.layer_kda_conv_state[layer];
        const recurrent = graph.layer_kda_recurrent_state[layer];
        if (conv == null and recurrent == null) continue;
        if (conv == null or recurrent == null or
            c.ds4_gpu_tensor_fill_f32(conv, 0, c.ds4_gpu_tensor_bytes(conv) / @sizeOf(f32)) == 0 or
            c.ds4_gpu_tensor_fill_f32(recurrent, 0, c.ds4_gpu_tensor_bytes(recurrent) / @sizeOf(f32)) == 0) return false;
    }
    return c.ds4_gpu_synchronize() != 0;
}
