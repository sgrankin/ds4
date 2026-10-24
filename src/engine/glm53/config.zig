//! Per-graph dimensions and layer policies. Configuration is captured from the
//! validated checkpoint once, rather than read from the C engine's active model.
const c = @import("gguf_c");
const std = @import("std");
pub const Record = c.ds4_glm_model_config;
pub const Graph = c.ds4_glm_gpu_graph;

pub fn capture(graph: *Graph, shape: *const c.ds4_shape) void {
    const reference = c.getenv("DS4_METAL_DISABLE_HC_FUSION");
    const flush = c.getenv("DS4_GLM_DECODE_FLUSH_INTERVAL");
    graph.config = .{
        .shape = shape.*,
        .prefill_chunk = 2048,
        .index_pool_size = 4,
        .reference_hc = reference != null and reference[0] != 0 and !std.mem.eql(u8, std.mem.span(reference), "0"),
        .fused_bf16_qkv = c.getenv("DS4_METAL_DISABLE_M3_ULTRA_GLM53_DECODE") == null and
            c.getenv("DS4_METAL_DISABLE_GLM53_BF16_QKV") == null,
        .stream_selected_async = c.getenv("DS4_METAL_ENABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD") != null and
            c.getenv("DS4_METAL_DISABLE_STREAMING_SELECTED_ASYNC_LOAD") == null and
            c.getenv("DS4_METAL_DISABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD") == null,
        .flash_prefill = c.getenv("DS4_GLM_DISABLE_FLASH_PREFILL") == null,
        .decode_flush_interval = if (flush != null and flush[0] != 0) @max(c.atoi(flush), 0) else -1,
    };
}

pub fn trunkLayers(config: *const Record) u32 {
    return config.shape.n_layer - config.shape.n_nextn_predict;
}

pub fn isKda(config: *const Record, layer: u32) bool {
    return config.shape.variant == c.DS4_VARIANT_GLM53 and
        layer < trunkLayers(config) and layer % 4 != 3;
}

pub fn hasIndexer(config: *const Record, layer: u32) bool {
    if (config.shape.variant == c.DS4_VARIANT_GLM53) return !isKda(config, layer);
    return layer < config.shape.n_leading_dense or (layer >= 6 and (layer - 6) % 4 == 0);
}

/// Pool storage rounds up, but only full pools have published indexer rows.
/// The incomplete pool is represented by the separate tail tensors.
pub fn poolCapacity(config: *const Record, tokens: u32) u32 {
    return (tokens + config.index_pool_size - 1) / config.index_pool_size;
}

pub fn indexLiveRows(config: *const Record, layer: u32, compact_live: u32) u32 {
    if (compact_live == 0 or !hasIndexer(config, layer)) return 0;
    return if (config.shape.variant == c.DS4_VARIANT_GLM53) compact_live / config.index_pool_size else compact_live;
}

pub fn projection(config: *const Record) u32 {
    return config.shape.n_kda_head * config.shape.n_kda_head_dim;
}

pub fn hcWidth(config: *const Record) u32 {
    return config.shape.n_embd * config.shape.n_hc;
}

pub fn hcMix(config: *const Record) u32 {
    return config.shape.n_hc * (config.shape.n_hc + 2);
}

pub fn convStateBytes(config: *const Record) u64 {
    return 3 * @as(u64, config.shape.n_kda_conv - 1) * projection(config) * @sizeOf(f32);
}

pub fn recurrentStateBytes(config: *const Record) u64 {
    return @as(u64, projection(config)) * config.shape.n_kda_head_dim * @sizeOf(f32);
}
