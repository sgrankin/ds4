//! Model-specific context and batching policy. OS memory admission and mapped
//! storage residency remain outside this pure graph geometry.
const c = @import("gguf_c");
const std = @import("std");
const Graph = c.ds4_glm_gpu_graph;

pub fn fitsContext(graph: *const Graph, position: u32, rows: u32) bool {
    return rows != 0 and position < graph.ctx_size and rows <= graph.ctx_size - position;
}

pub fn fitsFull(graph: *const Graph, position: u32, rows: u32) bool {
    return rows != 0 and position < graph.ctx_cap and rows <= graph.ctx_cap - position;
}

pub fn denseLimit(graph: *const Graph) u32 {
    return graph.config.shape.n_indexer_top_k + graph.config.index_pool_size - 1;
}

pub fn useIndexedPrefill(graph: *const Graph) bool {
    return graph.glm53 and graph.indexed_prefill_cap != 0 and !graph.full_kv_cache;
}

pub fn indexedAvailable(graph: *const Graph) bool {
    return graph.compact_cache_cap != 0 and graph.indexed_prefill_cap != 0 and graph.indexed_prefill_score_cap != 0 and
        graph.batch_indexer_q != null and graph.batch_indexer_weights != null and graph.batch_indexer_scores != null and
        graph.batch_indexer_selected != null and graph.batch_indexer_gate != null and graph.batch_indexer_pool_selected != null and
        graph.batch_qk_low != null and graph.batch_attn_lora != null;
}

pub fn limitIndexedChunk(graph: *const Graph, position: u32, rows: u32) u32 {
    // One layer-major chunk can cross the dense/sparse boundary. The attention
    // owner divides its slices there; only the explicit isolation knob splits.
    if (graph.glm53 and indexedAvailable(graph) and c.getenv("DS4_GLM53_DISABLE_MIXED_PREFILL") == null) return rows;
    const limit = denseLimit(graph);
    return if (position < limit) @min(rows, limit - position) else rows;
}

pub fn resumePrefillMin(graph: *const Graph) u32 {
    const env = c.getenv("DS4_GLM_RESUME_PREFILL_MIN");
    if (env == null or env[0] == 0) return if (graph.glm53) 2 else 4;
    var end: [*c]u8 = null;
    const value = c.strtol(env, &end, 10);
    if (end == env) return 4;
    return if (value <= 0) std.math.maxInt(u32) else @truncate(@as(c_ulong, @intCast(value)));
}

pub fn fullAttentionCap(context: u32, streaming: bool) u32 {
    var capacity: u32 = if (streaming) 8192 else 4096;
    if (context >= 65536) capacity = @min(capacity, 4096);
    if (context != 0) capacity = @min(capacity, context);
    return @max(capacity, 1);
}

pub fn contextRequest(shape: *const c.ds4_shape, context: c_int, output: *u32) bool {
    if (context <= 0) return false;
    const maximum: u32 = @intCast(@min(shape.rope_orig_ctx, std.math.maxInt(u32)));
    if (@as(u32, @intCast(context)) > maximum) {
        _ = c.fprintf(c.__stderrp, "ds4: GLM context %d exceeds model context %u\n", context, maximum);
        return false;
    }
    output.* = @intCast(context);
    return true;
}

pub fn indexedChunk(compact_capacity: u32) u32 {
    const env = c.getenv("DS4_GLM53_DISABLE_INDEXED_PREFILL");
    if (env != null and env[0] != 0) {
        const value = std.mem.span(env);
        if (!std.mem.eql(u8, value, "0") and !std.ascii.eqlIgnoreCase(value, "false") and
            !std.ascii.eqlIgnoreCase(value, "off") and !std.ascii.eqlIgnoreCase(value, "no")) return 0;
    }
    return if (compact_capacity != 0) @min(2048, compact_capacity) else 2048;
}

pub fn scoreRows(indexed_rows: u32, compact_capacity: u32) u32 {
    if (indexed_rows == 0 or compact_capacity == 0) return 0;
    const columns = (@as(u64, compact_capacity) + 3) / 4;
    return @intCast(@max(1, @min(indexed_rows, (256 * 1024 * 1024) / (columns * @sizeOf(f32)))));
}

pub fn batchCap(work_capacity: u32, indexed_rows: u32) u32 {
    var capacity = @min(work_capacity, 2048);
    if (indexed_rows != 0) capacity = @min(capacity, indexed_rows);
    return @max(capacity, 1);
}

pub fn flushInterval(rows: u32, command_rows: u32, logits: bool) u32 {
    // Tiny verification batches avoid the round-trips of per-layer flushes.
    return @intFromBool(rows > 2048 or command_rows > 2048 or (logits and rows > 8));
}

pub fn fullLayerPrefill(graph: *const Graph, position: u32, rows: u32) bool {
    if (position != 0 or rows < 256 or graph.quality or !graph.ssd_streaming or graph.tp_world >= 2 or
        graph.layer_start != 0 or graph.layer_count != graph.normal_layers or
        c.getenv("DS4_METAL_DISABLE_GLM_STREAMING_PREFILL_FULL_LAYER") != null) return false;
    if (c.getenv("DS4_METAL_GLM_STREAMING_PREFILL_FULL_LAYER") != null) return true;
    var minimum: u32 = 64;
    const env = c.getenv("DS4_METAL_GLM_STREAMING_PREFILL_FULL_LAYER_MIN_TOKENS");
    if (env != null) {
        var end: [*c]u8 = null;
        const value = c.strtoul(env, &end, 10);
        if (end != env and value != 0 and value <= std.math.maxInt(u32)) minimum = @intCast(value);
    }
    return rows >= minimum;
}
