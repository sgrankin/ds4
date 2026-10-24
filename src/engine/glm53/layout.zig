//! Persistent GLM state is more than KV: KDA has a convolution history and
//! recurrent matrix, while DSA's pooled indexer retains an incomplete tail.
//! Payload sizes describe the established F32 wire representation even when a
//! compact GPU cache uses F16. Changing cache precision must not change the file.
const c = @import("gguf_c");
const Config = @import("config.zig");
const Graph = c.ds4_glm_gpu_graph;

pub fn fullLiveRows(graph: *const Graph, checkpoint: u32, dense: u32) u32 {
    return if (graph.full_kv_cache) @min(graph.ctx_cap, checkpoint, dense) else 0;
}

pub fn compactLiveRows(graph: *const Graph, checkpoint: u32) u32 {
    return @min(graph.compact_cache_cap, checkpoint);
}

pub fn indexerTailBytes(config: *const c.ds4_glm_model_config) u64 {
    return 2 * @as(u64, config.index_pool_size) * config.shape.n_indexer_head_dim * @sizeOf(f32);
}

pub fn payloadLayerBytes(graph: *const Graph, layer: u32, full_live: u32, compact_live: u32) u64 {
    const config = &graph.config;
    const shape = &config.shape;
    if (Config.isKda(config, layer)) return Config.convStateBytes(config) + Config.recurrentStateBytes(config);
    var bytes = @as(u64, full_live) * (graph.q_dim + graph.heads_dim) * @sizeOf(f32);
    if (compact_live == 0) return bytes;
    bytes += @as(u64, compact_live) * (shape.n_kv_lora + shape.n_rot) * @sizeOf(f32);
    if (Config.hasIndexer(config, layer)) {
        bytes += @as(u64, Config.indexLiveRows(config, layer, compact_live)) * shape.n_indexer_head_dim * @sizeOf(f32);
        if (graph.glm53) bytes += indexerTailBytes(config);
    }
    return bytes;
}

pub fn payloadLiveBytes(graph: *const Graph, full_live: u32, compact_live: u32) u64 {
    var bytes: u64 = 0;
    for (0..graph.normal_layers) |layer| bytes += payloadLayerBytes(graph, @intCast(layer), full_live, compact_live);
    return bytes;
}
