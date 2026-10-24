//! Preserve the existing admission estimate while moving its model geometry
//! out of the C engine. This is an estimate, not an allocation census: the old
//! conservative scratch reservations intentionally remain unchanged.
const c = @import("gguf_c");
const std = @import("std");
const Config = @import("config.zig");
const Planning = @import("planning.zig");
const Record = c.ds4_glm_model_config;
const Shape = c.ds4_shape;

pub fn geometry(shape: *const Shape) Record {
    var config = std.mem.zeroes(Record);
    config.shape = shape.*;
    config.prefill_chunk = 2048;
    config.index_pool_size = 4;
    return config;
}

fn countKda(config: *const Record, first: u32, last: u32) u32 {
    if (first > last) return 0;
    var count: u32 = 0;
    for (first..last + 1) |layer| count += @intFromBool(Config.isKda(config, @intCast(layer)));
    return count;
}

pub fn indexerLayers(config: *const Record, first: u32, last: u32) u32 {
    return if (first > last) 0 else last - first + 1 - countKda(config, first, last);
}

pub fn fixedState(config: *const Record, first: u32, last: u32) u64 {
    const shape = &config.shape;
    const hc: u64 = Config.hcWidth(config);
    const mix: u64 = Config.hcMix(config);
    const projection: u64 = Config.projection(config);
    const persistent = @as(u64, countKda(config, first, last)) * (Config.convStateBytes(config) + Config.recurrentStateBytes(config));
    const scratch = (4 * hc + 2 * mix + shape.n_embd + shape.n_hc +
        6 * projection + 2 * shape.n_kda_head_dim + shape.n_kda_head) * @sizeOf(f32);
    return persistent + scratch;
}

pub fn compactBytes(config: *const Record, attention_layers: u32, indexer_layers: u32, capacity: u32) u64 {
    if (capacity == 0) return 0;
    const shape = &config.shape;
    const latent = @as(u64, attention_layers) * capacity * (shape.n_kv_lora + shape.n_rot) * @sizeOf(u16);
    const index = @as(u64, indexer_layers) * Config.poolCapacity(config, capacity) * shape.n_indexer_head_dim * @sizeOf(u16);
    const unfinished = @as(u64, indexer_layers) * 2 * config.index_pool_size * shape.n_indexer_head_dim * @sizeOf(f32);
    return latent + index + unfinished;
}

pub fn indexedScratch(config: *const Record, work_capacity: u32, compact_capacity: u32) u64 {
    _ = work_capacity;
    if (compact_capacity == 0) return 0;
    const shape = &config.shape;
    const indexed: u64 = Planning.indexedChunk(compact_capacity);
    const scores: u64 = Planning.scoreRows(@intCast(indexed), compact_capacity);
    const index_rows: u64 = Config.poolCapacity(config, compact_capacity);
    const query = @as(u64, shape.n_indexer_head) * shape.n_indexer_head_dim;
    const latent = @as(u64, shape.n_head) * shape.n_kv_lora;
    const selected = shape.n_indexer_top_k + config.index_pool_size - 1;
    return (index_rows + scores * index_rows + indexed * query + indexed * shape.n_indexer_head + indexed * selected +
        indexed * shape.n_indexer_head_dim + indexed * (shape.n_indexer_top_k / config.index_pool_size) +
        indexed * latent * 2) * @sizeOf(f32);
}

pub fn workspace(config: *const Record, work_capacity: u32, compact_capacity: u32) u64 {
    const shape = &config.shape;
    const indexed: u64 = if (compact_capacity != 0) Planning.indexedChunk(compact_capacity) else 0;
    const rows: u64 = Planning.batchCap(work_capacity, @intCast(indexed));
    const emb: u64 = shape.n_embd;
    const q_dim = @as(u64, shape.n_head) * shape.n_key_mla;
    const q_nope = shape.n_key_mla -| shape.n_rot;
    const heads = @as(u64, shape.n_head) * shape.n_value_mla;
    const kv_raw = @as(u64, shape.n_kv_lora) + shape.n_rot;
    const dense: u64 = @max(shape.n_ff_dense, shape.n_ff_exp);
    const middle = @max(dense, @as(u64, shape.n_expert_used) * shape.n_ff_exp);
    const lowrank = @as(u64, shape.n_head) * shape.n_kv_lora;
    const split_blocks: u64 = (shape.n_indexer_top_k + config.index_pool_size - 1 + 31) / 32;
    var bytes = indexedScratch(config, work_capacity, compact_capacity);
    // Scalar projection, indexer/attention, FFN/router, language-head and
    // captured router-tape reservations, in the original estimate's order.
    const scalar = [_]u64{
        3 * emb,                                                   2 * @as(u64, shape.n_lora_q), q_dim,
        shape.n_indexer_head_dim,                                  shape.n_indexer_head_dim,     shape.n_indexer_top_k / config.index_pool_size,
        @as(u64, shape.n_indexer_head) * shape.n_indexer_head_dim, shape.n_indexer_head,         work_capacity,
        lowrank,                                                   split_blocks * lowrank,       split_blocks * shape.n_head * 2,
        kv_raw,                                                    shape.n_kv_lora,              @as(u64, shape.n_head) * q_nope,
        2 * heads,                                                 6 * emb,                      2 * dense,
        middle,                                                    @as(u64, shape.n_expert) * 2, shape.n_expert_used,
        shape.n_expert_used,                                       shape.n_vocab,                @as(u64, shape.n_layer) * 64 * shape.n_expert_used,
    };
    for (scalar) |elements| bytes +|= elements *| @sizeOf(f32);
    const batch = [_]u64{
        rows,             rows * shape.n_expert * 2, rows * shape.n_expert_used,   rows * shape.n_expert_used,
        rows * emb * 7,   rows * shape.n_lora_q * 2, rows * q_dim,                 rows * shape.n_indexer_head_dim,
        rows * kv_raw,    rows * shape.n_kv_lora,    rows * shape.n_head * q_nope, rows * heads * 2,
        rows * dense * 2, rows * shape.n_ff_exp,     rows * middle,
    };
    for (batch) |elements| bytes +|= elements *| @sizeOf(f32);
    const hc: u64 = Config.hcWidth(config);
    const mix: u64 = Config.hcMix(config);
    const projection: u64 = Config.projection(config);
    bytes +|= rows * (4 * hc + 2 * mix + 6 * projection + shape.n_kda_head_dim + shape.n_kda_head) * @sizeOf(f32);
    // Retain the additional low-rank reservation of the established guard.
    if (indexed != 0) bytes +|= indexed * lowrank * @sizeOf(f32);
    return bytes;
}

pub fn estimate(config: *const Record, context: u32, work_capacity: u32, compact: u32, first: u32, requested_last: u32) c.ds4_context_memory {
    var result = std.mem.zeroes(c.ds4_context_memory);
    if (config.shape.n_layer <= config.shape.n_nextn_predict) return result;
    const trunk = Config.trunkLayers(config);
    if (first >= trunk or requested_last < first) return result;
    const last = @min(requested_last, trunk - 1);
    const layers = last - first + 1;
    const attention = layers - countKda(config, first, last);
    const capacity = @min(compact, context);
    const indexed = if (capacity != 0) Planning.indexedChunk(capacity) else 0;
    result.prefill_cap = Planning.batchCap(work_capacity, indexed);
    // Flash uses absorbed MLA throughout. Expanded per-head K/V stays off.
    result.raw_cap = 0;
    result.scratch_bytes = workspace(config, work_capacity, capacity) + fixedState(config, first, last);
    if (capacity != 0) {
        result.comp_cap = capacity;
        result.compressed_bytes = compactBytes(config, attention, attention, capacity);
    }
    result.total_bytes = result.compressed_bytes + result.scratch_bytes;
    return result;
}
