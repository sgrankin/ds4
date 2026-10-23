//! V4.1 batch attention over the graph's owned workspace. The executor chooses
//! the sweep and command lifetime; this controller preserves publication order,
//! causal per-row selection and the existing raw-attention reduction partitions.
const std = @import("std");
const c = @import("gguf_c");
const Ops = @import("ops.zig").Ops;
const scalar = @import("attention.zig");
const Attention = scalar.Attention;
const Graph = c.ds41_gpu_graph;
const Row = c.ds41_prefill_row;
const Layer = c.ds4_layer_weights;
const Weight = c.ds4_tensor;
const Tensor = c.ds4_gpu_tensor;
const score_rows = 32;

fn knownWeight(pointer: [*c]const Weight) *const Weight {
    return @ptrCast(pointer); // Validated bound-layer table; no layout copy.
}
fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn selectRow(graph: *Graph, row: *const Row) void {
    // Execution rows borrow all handles. The graph owner alone retires them.
    inline for (comptime std.meta.fieldNames(Row)) |name| @field(graph, name) = @field(row, name);
}

pub const BatchAttention = struct {
    ops: Ops,
    graph: *Graph,

    fn attention(self: *const BatchAttention, graph: *Graph) Attention {
        return .{ .config = self.ops.config, .model = self.ops.model, .graph = graph };
    }

    pub fn project(self: *const BatchAttention, layer: *const Layer, count: u32) bool {
        const graph = self.graph;
        const batch = &graph.batch;
        const shape = &self.ops.config.shape;
        const model = self.ops.model;
        const query_width = shape.n_head / graph.tp_world * shape.n_head_dim;
        // This path deliberately uses the original row RMS kernels, followed
        // by their explicit BF16 boundaries, rather than the scalar norm fuse.
        return self.ops.matmulBatch(batch.qr, knownWeight(layer.attn_q_a), batch.norm, count, true) and
            c.ds4_gpu_rms_norm_weight_rows_tensor(batch.qr, batch.qr, model.map, model.size, knownWeight(layer.attn_q_a_norm).abs_offset, shape.n_lora_q, count, shape.rms_eps) != 0 and
            c.ds4_gpu_dsv41_quantize(batch.qr, shape.n_lora_q, count, c.DS4_V41_BF16) != 0 and
            self.ops.matmulRowsBatch(batch.q, knownWeight(layer.attn_q_b), batch.qr, graph.tp_rank * query_width, query_width, count) and
            self.ops.matmulBatch(batch.kv, knownWeight(layer.attn_kv), batch.norm, count, true) and
            c.ds4_gpu_rms_norm_weight_rows_tensor(batch.kv, batch.kv, model.map, model.size, knownWeight(layer.attn_kv_a_norm).abs_offset, shape.n_head_dim, count, shape.rms_eps) != 0 and
            c.ds4_gpu_dsv41_quantize(batch.kv, shape.n_head_dim, count, c.DS4_V41_BF16) != 0;
    }

    pub fn projectRows(self: *const BatchAttention, out: ?*Tensor, weight: *const Weight, input: ?*const Tensor, count: u32, round: bool) bool {
        if (weight.type != c.DS4_TENSOR_F16)
            return self.ops.matmulBatch(out, weight, input, count, round);
        return c.ds4_gpu_dsv41_projection_rows(out, self.ops.model.map, self.ops.model.size, weight.abs_offset, @truncate(weight.dim[0]), @truncate(weight.dim[1]), count, input) != 0 and
            (!round or c.ds4_gpu_dsv41_quantize(out, @truncate(weight.dim[1]), count, c.DS4_V41_BF16) != 0);
    }

    // Decoder preparation may publish a view of only part of the workspace.
    // Accept that actual row record instead of assuming graph.batch's extent.
    pub fn publish(self: *const BatchAttention, batch: *Row, layer: *const Layer, layer_index: u32, start: u32, count: u32) bool {
        const graph = self.graph;
        const shape = &self.ops.config.shape;
        const ratio = self.ops.config.layerPolicy(layer_index).?.compression_ratio;
        const owner = scalar.cacheOwner(layer_index);
        const first = start / ratio;
        const rows = (start + count) / ratio - first;
        if (ratio == 2) {
            if (!self.projectRows(batch.pool_kv, knownWeight(layer.attn_compressor_kv), batch.norm, count, false) or
                !self.projectRows(batch.pool_score, knownWeight(layer.attn_compressor_gate), batch.norm, count, false) or
                c.ds4_gpu_dsv41_pool2(batch.latent, batch.pool_kv, batch.pool_score, graph.previous_kv[owner], graph.previous_score[owner], shape.n_head_dim, count, start) == 0) return false;
        } else if (!self.projectRows(batch.latent, knownWeight(layer.attn_compressor_kv), batch.norm, count, true)) return false;
        if (rows == 0) return true;
        // Bulk publication and suffix replay must produce the same keys.
        // Ordinary matrix reductions would make old keys depend on sweep size.
        const originals = self.attention(graph);
        return self.ops.normBatch(batch.latent, batch.latent, knownWeight(layer.attn_compressor_norm), rows) and
            self.projectRows(batch.index_k, knownWeight(layer.indexer_attn_k), batch.latent, rows, true) and
            self.ops.normBatch(batch.index_k, batch.index_k, knownWeight(layer.indexer_k_norm), rows) and
            originals.archivePublished(owner, batch.index_k, first, rows, true) and
            originals.archivePublished(owner, batch.latent, first, rows, false) and
            c.ds4_gpu_dsv41_rope_stride(batch.index_k, shape.n_indexer_head_dim, 1, rows, first * ratio, ratio, true, false) != 0 and
            c.ds4_gpu_dsv41_quantize(batch.index_k, shape.n_indexer_head_dim, rows, c.DS4_V41_FP4_E8M0) != 0 and
            c.ds4_gpu_tensor_copy(graph.index_cache[owner], @as(u64, first) * shape.n_indexer_head_dim * 4, batch.index_k, 0, @as(u64, rows) * shape.n_indexer_head_dim * 4) != 0 and
            c.ds4_gpu_dsv41_rope_stride(batch.latent, shape.n_head_dim, 1, rows, first * ratio, ratio, true, false) != 0 and
            c.ds4_gpu_dsv41_quantize(batch.latent, shape.n_head_dim, rows, c.DS4_V41_FP4_E4M3) != 0 and
            originals.storeCompressed(owner, first, batch.latent, rows);
    }

    fn scoreTile(self: *const BatchAttention, owner: usize, compressed: u32, count: u32, offset: u32, rows: u32, ratio: u32, packed_scores: bool) bool {
        const graph = self.graph;
        const shape = &self.ops.config.shape;
        const query_bytes = @as(u64, shape.n_indexer_head) * shape.n_indexer_head_dim * 4;
        const weight_bytes = @as(u64, shape.n_indexer_head) * 4;
        const queries = c.ds4_gpu_tensor_view(graph.batch.index_q, @as(u64, offset) * query_bytes, @as(u64, rows) * query_bytes);
        const weights = c.ds4_gpu_tensor_view(graph.batch.index_weights, @as(u64, offset) * weight_bytes, @as(u64, rows) * weight_bytes);
        defer c.ds4_gpu_tensor_free(weights);
        defer c.ds4_gpu_tensor_free(queries);
        if (queries == null or weights == null) return false;
        return if (packed_scores)
            c.ds4_gpu_dsv41_indexer_scores_packed(graph.index_scores, queries, weights, graph.index_cache[owner], graph.index_packed, compressed, rows, graph.pos + offset, ratio, count, offset) != 0
        else
            c.ds4_gpu_dsv41_indexer_scores_batch(graph.index_scores, queries, weights, graph.index_cache[owner], compressed, rows, graph.pos + offset, ratio) != 0;
    }

    pub fn index(self: *const BatchAttention, layer: *const Layer, layer_index: u32, count: u32) bool {
        const graph = self.graph;
        const shape = &self.ops.config.shape;
        const start = graph.pos;
        const ratio = self.ops.config.layerPolicy(layer_index).?.compression_ratio;
        const owner = scalar.cacheOwner(layer_index);
        const compressed = (start + count) / ratio;
        if (compressed == 0) return true;
        const batch = &graph.batch;
        if (present("DS4_METAL_DISABLE_V41_BATCH_INDEX_PROJ")) {
            for (0..count) |offset| {
                const row = &graph.rows_view[offset];
                if (!self.ops.matmul(row.index_q, knownWeight(layer.indexer_attn_q_b), row.qr, true) or
                    !self.ops.rope(row.index_q, shape.n_indexer_head, 128, layer_index, start + @as(u32, @intCast(offset)), false) or
                    c.ds4_gpu_dsv41_quantize(row.index_q, 128, shape.n_indexer_head, c.DS4_V41_FP4_E8M0) == 0 or
                    !self.ops.matmul(row.index_weights, knownWeight(layer.indexer_proj), row.norm, true)) return false;
            }
        } else if (!self.projectRows(batch.index_q, knownWeight(layer.indexer_attn_q_b), batch.qr, count, true) or
            c.ds4_gpu_dsv41_rope(batch.index_q, shape.n_indexer_head_dim, shape.n_indexer_head, count, start, true, false) == 0 or
            c.ds4_gpu_dsv41_quantize(batch.index_q, shape.n_indexer_head_dim, count * shape.n_indexer_head, c.DS4_V41_FP4_E8M0) == 0 or
            !self.projectRows(batch.index_weights, knownWeight(layer.indexer_proj), batch.norm, count, true)) return false;

        // This is a borrowed execution copy, never an owning graph. Only 32
        // score rows are live; each retains its own causal width and tie order.
        var row = graph.*;
        const packed_scores = count >= 8 and c.ds4_gpu_dsv41_tensor_ops_available() != 0 and
            !present("DS4_METAL_DISABLE_V41_PACKED_INDEX");
        if (packed_scores and c.ds4_gpu_dsv41_indexer_pack(graph.index_packed, batch.index_q, graph.index_cache[owner], compressed, count) == 0) return false;
        var offset: u32 = 0;
        while (offset < count) : (offset += score_rows) {
            const rows: u32 = @min(count - offset, score_rows);
            if (!self.scoreTile(owner, compressed, count, offset, rows, ratio, packed_scores)) return false;
            const batch_topk = (start + offset + 1) / ratio >= 1024 and !present("DS4_METAL_DISABLE_V41_BATCH_TOPK");
            var token: u32 = 0;
            while ((!batch_topk or layer_index >= 20) and token < rows) : (token += 1) {
                row.pos = start + offset + token;
                row.selected_comp = graph.rows_view[offset + token].selected_comp;
                row.block_mask = graph.rows_view[offset + token].block_mask;
                row.index_scores = c.ds4_gpu_tensor_view(graph.index_scores, @as(u64, token) * compressed * 4, @as(u64, compressed) * 4);
                const attention_row = self.attention(&row);
                const ok = row.index_scores != null and if (batch_topk)
                    attention_row.candidates(layer_index)
                else
                    attention_row.pick(layer_index);
                c.ds4_gpu_tensor_free(row.index_scores);
                if (!ok) return false;
            }
            if (batch_topk) {
                const id_bytes = @as(u64, shape.n_indexer_top_k) * @sizeOf(i32);
                const ids = c.ds4_gpu_tensor_view(batch.selected_comp, @as(u64, offset) * id_bytes, @as(u64, rows) * id_bytes);
                const ok = ids != null and c.ds4_gpu_dsv41_indexer_topk_batch(ids, graph.index_scores, compressed, rows, start + offset, ratio) != 0;
                c.ds4_gpu_tensor_free(ids);
                if (!ok) return false;
            }
        }
        return true;
    }

    fn rawAttention(self: *const BatchAttention, heads: u32, sinks: u64, previous: u32, count: u32) bool {
        const graph = self.graph;
        const model = self.ops.model;
        const width = self.ops.config.shape.n_head_dim;
        const row_bytes = @as(u64, width) * 4;
        const query_bytes = @as(u64, heads) * row_bytes;
        const originals = self.attention(graph);
        // Raw attention rounds probabilities per key block. Keep its original
        // partitions even when the surrounding dense/FFN batch is wider.
        var offset: u32 = 0;
        while (offset < count) : (offset += 2048) {
            const rows: u32 = @min(count - offset, 2048);
            const past = originals.rawPast(graph.pos + offset, 127);
            const raw_offset = previous + offset - past;
            const queries = c.ds4_gpu_tensor_view(graph.batch.q, @as(u64, offset) * query_bytes, @as(u64, rows) * query_bytes);
            const out = c.ds4_gpu_tensor_view(graph.batch.heads, @as(u64, offset) * query_bytes, @as(u64, rows) * query_bytes);
            const raw = c.ds4_gpu_tensor_view(graph.raw_prefill, @as(u64, raw_offset) * row_bytes, @as(u64, past + rows) * row_bytes);
            const ok = queries != null and out != null and raw != null and
                c.ds4_gpu_attention_decode_raw_batch_heads_tensor(out, model.map, model.size, sinks, queries, raw, rows, graph.pos + offset, past + rows, past + rows, 0, 128, heads, width) != 0;
            c.ds4_gpu_tensor_free(raw);
            c.ds4_gpu_tensor_free(out);
            c.ds4_gpu_tensor_free(queries);
            if (!ok) return false;
        }
        return true;
    }

    // As in the original batch controller, the executor projects Q/KV first.
    // This method then owns publication, selection, attention and ring update.
    pub fn run(self: *const BatchAttention, layer: *const Layer, layer_index: u32, count: u32) bool {
        const graph = self.graph;
        const model = self.ops.model;
        const shape = &self.ops.config.shape;
        const start = graph.pos;
        const policy = self.ops.config.layerPolicy(layer_index) orelse return false;
        const ratio = policy.compression_ratio;
        if (graph.decoder_replay_floor != 0 and start < graph.decoder_replay_floor) return false;
        const owner = scalar.cacheOwner(layer_index);
        const compressed = if (ratio != 0) (start + count) / ratio else 0;
        const originals = self.attention(graph);
        const previous = originals.rawPast(start, 127);
        const raw_start = (start - previous) % 128;
        const first: u32 = @min(previous, 128 - raw_start);
        const raw_rows = previous + count;
        const row_bytes = @as(u64, shape.n_head_dim) * 4;
        const heads = shape.n_head / graph.tp_world;
        const sinks = knownWeight(layer.attn_sinks).abs_offset + @as(u64, graph.tp_rank) * heads * 4;
        const batch = &graph.batch;
        if (!originals.archiveRaw(layer_index, batch.kv, start, count) or
            c.ds4_gpu_dsv41_rope(batch.q, shape.n_head_dim, heads, count, start, ratio != 0, false) == 0 or
            c.ds4_gpu_dsv41_rope(batch.kv, shape.n_head_dim, 1, count, start, ratio != 0, false) == 0 or
            c.ds4_gpu_dsv41_quantize(batch.kv, shape.n_head_dim, count, c.DS4_V41_FP8_E8M0) == 0) return false;
        // Save chronological raw keys before replacing the final decode ring.
        if (first != 0 and c.ds4_gpu_tensor_copy(graph.raw_prefill, 0, graph.window[layer_index], @as(u64, raw_start) * row_bytes, @as(u64, first) * row_bytes) == 0) return false;
        if (previous > first and c.ds4_gpu_tensor_copy(graph.raw_prefill, @as(u64, first) * row_bytes, graph.window[layer_index], 0, @as(u64, previous - first) * row_bytes) == 0) return false;
        if (c.ds4_gpu_tensor_copy(graph.raw_prefill, @as(u64, previous) * row_bytes, batch.kv, 0, @as(u64, count) * row_bytes) == 0) return false;

        var row = graph.*; // Borrowed handles only; never deinitialize this copy.
        const batch_index = policy.publishes_index and !present("DS4_METAL_DISABLE_V41_BATCH_INDEX");
        const batch_publish = policy.publishes_kv and ratio == 2 and !present("DS4_METAL_DISABLE_V41_BATCH_COMPRESS");
        if (batch_publish and !self.publish(batch, layer, layer_index, start, count)) return false;
        if (!batch_index or !batch_publish) {
            for (0..count) |offset| {
                row.pos = start + @as(u32, @intCast(offset));
                const row_view: *const Row = @ptrCast(&graph.rows_view[offset]);
                selectRow(&row, row_view);
                const attention_row = self.attention(&row);
                if (!batch_publish and !attention_row.publish(layer, layer_index)) return false;
                if (!batch_index and !attention_row.selectPublished(layer, layer_index)) return false;
            }
        }
        if (batch_index and !self.index(layer, layer_index, count)) return false;
        const ok = if (compressed == 0)
            self.rawAttention(heads, sinks, previous, count)
        else if (compressed < shape.n_indexer_top_k)
            c.ds4_gpu_attention_decode_mixed_batch_heads_tensor(batch.heads, model.map, model.size, sinks, batch.q, graph.raw_prefill, graph.compressed[owner], @intFromBool(graph.compressed_kv_f16), null, 0, count, start, raw_rows, graph.prefill_cap + 128, 0, compressed, 128, ratio, heads, shape.n_head_dim) != 0
        else
            c.ds4_gpu_attention_indexed_mixed_batch_heads_tensor(batch.heads, model.map, model.size, sinks, batch.q, graph.raw_prefill, graph.compressed[owner], @intFromBool(graph.compressed_kv_f16), batch.selected_comp, count, start, raw_rows, graph.prefill_cap + 128, 0, compressed, shape.n_indexer_top_k, 128, ratio, heads, shape.n_head_dim) != 0;
        if (!ok or c.ds4_gpu_dsv41_quantize(batch.heads, heads * shape.n_head_dim, count, c.DS4_V41_BF16) == 0 or
            c.ds4_gpu_dsv41_rope(batch.heads, shape.n_head_dim, heads, count, start, ratio != 0, true) == 0) return false;
        const kept: u32 = @min(raw_rows, 128);
        const slot = (start + count - kept) % 128;
        const part: u32 = @min(kept, 128 - slot);
        return c.ds4_gpu_tensor_copy(graph.window[layer_index], @as(u64, slot) * row_bytes, graph.raw_prefill, @as(u64, raw_rows - kept) * row_bytes, @as(u64, part) * row_bytes) != 0 and
            (kept == part or c.ds4_gpu_tensor_copy(graph.window[layer_index], 0, graph.raw_prefill, @as(u64, raw_rows - kept + part) * row_bytes, @as(u64, kept - part) * row_bytes) != 0);
    }
};
