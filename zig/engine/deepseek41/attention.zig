//! Scalar V4.1 attention over one explicitly configured graph. Command-buffer
//! ownership stays with the executor; this controller preserves the released
//! projection/rounding/publication order and the existing cache representations.
const c = @import("gguf_c");
const Config = @import("../config.zig").ModelConfig;
const Ops = @import("ops.zig").Ops;
const Record = c.ds41_gpu_graph;
const Layer = c.ds4_layer_weights;
const Tensor = c.ds4_gpu_tensor;
const Weight = c.ds4_tensor;
const Archive = c.ds41_prerope_archive;

// Validated source groups share one compressed/index cache. The two raw-only
// layers do not consume owner zero's compressed rows until publication begins.
pub fn cacheOwner(layer: u32) usize {
    return if (layer < 8) 0 else if (layer < 14) 1 else if (layer < 20) 2 else 3;
}
fn knownWeight(pointer: [*c]const Weight) *const Weight {
    // The bound layer table has already passed config-specific completeness
    // and layout validation. Narrow its C field once at the native boundary.
    return @ptrCast(pointer);
}
fn bf16(tensor: ?*Tensor, width: u32) bool {
    return c.ds4_gpu_dsv41_quantize(tensor, width, 1, c.DS4_V41_BF16) != 0;
}

pub const Attention = struct {
    config: *const Config,
    model: *const c.ds4_model,
    graph: *Record,

    fn ops(self: *const Attention) Ops {
        return .{ .config = self.config, .model = self.model };
    }
    fn archive(self: *const Attention) ?*Archive {
        return if (self.graph.prerope == null) null else @ptrCast(self.graph.prerope);
    }

    // A replayed decoder has no local keys before its floor. Mask those rows
    // instead of treating missing cache entries as zero-valued attention keys.
    pub fn rawPast(self: *const Attention, position: u32, limit: u32) u32 {
        var past: u32 = @min(position, limit);
        const floor = self.graph.decoder_replay_floor;
        if (floor != 0) past = @min(past, if (position > floor) position - floor else 0);
        return past;
    }

    pub fn archiveRaw(self: *const Attention, layer: u32, kv: ?*Tensor, start: u32, count: u32) bool {
        const originals = self.archive() orelse return true;
        const width = self.config.shape.n_head_dim;
        // A wide publication keeps only its final live raw ring. Capture
        // normalized BF16 originals BEFORE rotary/FP8 cache quantization.
        var offset: u32 = if (count > 128) count - 128 else 0;
        while (offset < count) {
            const slot = (start + offset) % 128;
            const rows: u32 = @min(count - offset, 128 - slot);
            const view = c.ds4_gpu_tensor_view(kv, @as(u64, offset) * width * 4, @as(u64, rows) * width * 4) orelse return false;
            defer c.ds4_gpu_tensor_free(view);
            if (c.ds4_gpu_dsv41_prerope_copy(originals.raw[layer], slot, view, width, rows, true) == 0) return false;
            for (0..rows) |row| originals.raw_epoch[layer][slot + row] = start + offset + @as(u32, @intCast(row));
            offset += rows;
        }
        return true;
    }

    pub fn archivePublished(self: *const Attention, owner: usize, tensor: ?*Tensor, first: u32, rows: u32, index: bool) bool {
        const originals = self.archive() orelse return true;
        if (rows == 0) return true;
        const capacity = self.graph.ctx / @as(u32, if (owner < 3) 2 else 1);
        const width = if (index) self.config.shape.n_indexer_head_dim else self.config.shape.n_head_dim;
        const destination = if (index) originals.index[owner] else originals.compressed[owner];
        if (first > capacity or rows > capacity + 1 - first or
            c.ds4_gpu_dsv41_prerope_copy(destination, first, tensor, width, rows, true) == 0) return false;
        for (0..rows) |row| originals.published[owner][first + row] |= if (index) @as(u8, 1) else 2;
        return true;
    }

    pub fn storeCompressed(self: *const Attention, owner: usize, first: u32, source: ?*Tensor, rows: u32) bool {
        const width = self.config.shape.n_head_dim;
        const values = @as(u64, rows) * width;
        // Mode-3 quantization has already rounded into BF16 values exactly
        // representable in F16. Storage conversion does not requantize them.
        if (self.graph.compressed_kv_f16)
            return c.ds4_gpu_tensor_copy_f32_to_f16(self.graph.compressed[owner], @as(u64, first) * width * 2, source, 0, values) != 0;
        return c.ds4_gpu_tensor_copy(self.graph.compressed[owner], @as(u64, first) * width * 4, source, 0, values * 4) != 0;
    }

    fn packedGather(self: *const Attention) bool {
        const value = c.getenv("DS4_METAL_V41_PACKED_F16_GATHER");
        return self.graph.compressed_kv_f16 and (value == null or c.strcmp(value, "0") != 0);
    }
    fn gather(self: *const Attention, owner: usize, rows: u32, selected: u32, packed_kv: bool) bool {
        const graph = self.graph;
        if (graph.compressed_kv_f16) {
            if (packed_kv)
                return c.ds4_gpu_dsv41_gather_kv_f16_packed(graph.selected_kv, graph.compressed[owner], graph.selected_comp, rows, selected) != 0;
            return c.ds4_gpu_dsv41_gather_kv_f16(graph.selected_kv, graph.compressed[owner], graph.selected_comp, rows, selected) != 0;
        }
        return c.ds4_gpu_dsv41_gather_kv(graph.selected_kv, graph.compressed[owner], graph.selected_comp, rows, selected) != 0;
    }

    pub fn publish(self: *const Attention, layer: *const Layer, index: u32) bool {
        const graph = self.graph;
        const position = graph.pos;
        const policy = self.config.layerPolicy(index).?;
        if (!policy.publishes_kv) return true;
        const owner = cacheOwner(index);
        const ratio = policy.compression_ratio;
        const count = (position + 1) / ratio;
        const op = self.ops();
        if (!op.matmul(graph.pool_kv, knownWeight(layer.attn_compressor_kv), graph.norm, ratio == 1)) return false;
        if (ratio == 2) {
            if (!op.matmul(graph.pool_score, knownWeight(layer.attn_compressor_gate), graph.norm, false) or
                c.ds4_gpu_dsv41_pool2(graph.latent, graph.pool_kv, graph.pool_score, graph.previous_kv[owner], graph.previous_score[owner], 512, 1, position) == 0) return false;
        } else if (c.ds4_gpu_tensor_copy(graph.latent, 0, graph.pool_kv, 0, 512 * 4) == 0) return false;
        if (graph.verify_prefix and
            (c.ds4_gpu_tensor_copy(graph.verify_previous_kv[owner], 0, graph.previous_kv[owner], 0, 512 * 4) == 0 or
                c.ds4_gpu_tensor_copy(graph.verify_previous_score[owner], 0, graph.previous_score[owner], 0, 512 * 4) == 0)) return false;
        if ((position + 1) % ratio != 0) return true;
        const rope_position = position + 1 - ratio;
        // Source tape precedes RoPE and cache-format rounding. Each archive
        // restore/relocation can start from these originals independently.
        return op.norm(graph.latent, graph.latent, knownWeight(layer.attn_compressor_norm)) and
            op.matmul(graph.index_k, knownWeight(layer.indexer_attn_k), graph.latent, true) and
            op.norm(graph.index_k, graph.index_k, knownWeight(layer.indexer_k_norm)) and
            self.archivePublished(owner, graph.index_k, count - 1, 1, true) and
            self.archivePublished(owner, graph.latent, count - 1, 1, false) and
            op.rope(graph.index_k, 1, self.config.shape.n_indexer_head_dim, index, rope_position, false) and
            c.ds4_gpu_dsv41_quantize(graph.index_k, 128, 1, c.DS4_V41_FP4_E8M0) != 0 and
            c.ds4_gpu_tensor_copy(graph.index_cache[owner], @as(u64, count - 1) * 128 * 4, graph.index_k, 0, 128 * 4) != 0 and
            op.rope(graph.latent, 1, 512, index, rope_position, false) and
            c.ds4_gpu_dsv41_quantize(graph.latent, 512, 1, c.DS4_V41_FP4_E4M3) != 0 and
            self.storeCompressed(owner, count - 1, graph.latent, 1);
    }

    pub fn candidates(self: *const Attention, index: u32) bool {
        const graph = self.graph;
        const policy = self.config.layerPolicy(index).?;
        const count = if (policy.compression_ratio != 0) (graph.pos + 1) / policy.compression_ratio else 0;
        if (count == 0 or !policy.publishes_index) return true;
        if (index == 20) {
            const blocks = (count + 7) / 8;
            const top: u32 = @min(blocks, 2048);
            // Below the cap every block survives. Only the mask is consumed,
            // so sorting its unused all-block selection would add no value.
            if (top == blocks) return c.ds4_gpu_tensor_fill_f32(graph.block_mask, 0, blocks) != 0;
            return c.ds4_gpu_dsv41_candidate_blocks(graph.block_scores, graph.index_scores, count, 1, graph.pos, policy.compression_ratio) != 0 and
                c.ds4_gpu_indexer_topk_tensor(graph.block_selected, graph.block_scores, blocks, 1, top) != 0 and
                c.ds4_gpu_dsv4_topk_mask_tensor(graph.block_mask, graph.block_selected, blocks, 1, top) != 0;
        }
        return index <= 20 or c.ds4_gpu_dsv41_candidate_filter(graph.index_scores, graph.block_mask, count, 1, graph.pos, policy.compression_ratio) != 0;
    }

    pub fn selectPublished(self: *const Attention, layer: *const Layer, index: u32) bool {
        const graph = self.graph;
        const shape = &self.config.shape;
        const policy = self.config.layerPolicy(index).?;
        const count = if (policy.compression_ratio != 0) (graph.pos + 1) / policy.compression_ratio else 0;
        if (count != 0 and policy.publishes_index) {
            const op = self.ops();
            if (!op.matmul(graph.index_q, knownWeight(layer.indexer_attn_q_b), graph.qr, true) or
                !op.rope(graph.index_q, shape.n_indexer_head, 128, index, graph.pos, false) or
                c.ds4_gpu_dsv41_quantize(graph.index_q, 128, shape.n_indexer_head, c.DS4_V41_FP4_E8M0) == 0 or
                !op.matmul(graph.index_weights, knownWeight(layer.indexer_proj), graph.norm, true) or
                c.ds4_gpu_glm_indexer_score_one_tensor(graph.index_scores, graph.index_q, graph.index_weights, graph.index_cache[cacheOwner(index)], count, shape.n_indexer_head, 128, 1.0 / 64.0, false) == 0) return false;
        }
        return self.pick(index);
    }

    pub fn pick(self: *const Attention, index: u32) bool {
        const graph = self.graph;
        const policy = self.config.layerPolicy(index).?;
        const count = if (policy.compression_ratio != 0) (graph.pos + 1) / policy.compression_ratio else 0;
        const top: u32 = @min(count, self.config.shape.n_indexer_top_k);
        return self.candidates(index) and (count == 0 or !policy.publishes_index or
            c.ds4_gpu_indexer_topk_tensor(graph.selected_comp, graph.index_scores, count, 1, top) != 0);
    }

    fn select(self: *const Attention, layer: *const Layer, index: u32) bool {
        return self.publish(layer, index) and self.selectPublished(layer, index);
    }

    pub fn project(self: *const Attention, layer: *const Layer) bool {
        const graph = self.graph;
        const shape = &self.config.shape;
        const query_width = shape.n_head / graph.tp_world * shape.n_head_dim;
        const op = self.ops();
        return op.matmul(graph.qr, knownWeight(layer.attn_q_a), graph.norm, true) and
            op.norm(graph.qr, graph.qr, knownWeight(layer.attn_q_a_norm)) and
            op.matmulRows(graph.q, knownWeight(layer.attn_q_b), graph.qr, graph.tp_rank * query_width, query_width) and
            op.matmul(graph.kv, knownWeight(layer.attn_kv), graph.norm, true) and
            op.norm(graph.kv, graph.kv, knownWeight(layer.attn_kv_a_norm));
    }

    pub fn low(self: *const Attention, layer: *const Layer) bool {
        const graph = self.graph;
        const shape = &self.config.shape;
        const groups = shape.n_out_group / graph.tp_world;
        const group_start = graph.tp_rank * groups;
        const group_width = @as(u64, shape.n_head_dim) * (shape.n_head / shape.n_out_group);
        var row_bytes: u64 = undefined;
        const output_a = knownWeight(layer.attn_output_a);
        return c.ds4_gguf_tensor_nbytes(output_a.type, group_width, &row_bytes) and
            c.ds4_gpu_attention_output_low_q8_tensor(graph.low, self.model.map, self.model.size, output_a.abs_offset + @as(u64, group_start) * shape.n_lora_o * row_bytes, group_width, shape.n_lora_o, groups, graph.heads) != 0 and
            bf16(graph.low, groups * shape.n_lora_o);
    }

    pub fn output(self: *const Attention, layer: *const Layer) bool {
        if (!self.low(layer)) return false;
        const graph = self.graph;
        const shape = &self.config.shape;
        const groups = shape.n_out_group / graph.tp_world;
        const op = self.ops();
        if (graph.tp_world == 2)
            return op.quantKslice(graph.block, knownWeight(layer.attn_output_b), @as(u64, shape.n_out_group) * shape.n_lora_o, @as(u64, graph.tp_rank) * groups * shape.n_lora_o, @as(u64, groups) * shape.n_lora_o, shape.n_embd, graph.low, 0);
        return op.matmul(graph.block, knownWeight(layer.attn_output_b), graph.low, false);
    }

    pub fn run(self: *const Attention, layer: *const Layer, index: u32, projected: bool) bool {
        const graph = self.graph;
        const shape = &self.config.shape;
        const policy = self.config.layerPolicy(index) orelse return false;
        const position = graph.pos;
        if (graph.decoder_replay_floor != 0 and position < graph.decoder_replay_floor) return false;
        const compressed = if (policy.compression_ratio != 0) (position + 1) / policy.compression_ratio else 0;
        const heads = shape.n_head / graph.tp_world;
        const head_start = graph.tp_rank * heads;
        if (!projected and !self.project(layer)) return false;
        if (!self.trace(graph.qr, shape.n_lora_q, index, "1a-qr") or
            !self.trace(graph.q, heads * shape.n_head_dim, index, "1b-q") or
            !self.trace(graph.kv, shape.n_head_dim, index, "1c-kv")) return false;
        const op = self.ops();
        if (!self.archiveRaw(index, graph.kv, position, 1) or
            !op.rope(graph.q, heads, shape.n_head_dim, index, position, false) or
            !op.rope(graph.kv, 1, shape.n_head_dim, index, position, false) or
            c.ds4_gpu_dsv41_quantize(graph.kv, shape.n_head_dim, 1, c.DS4_V41_FP8_E8M0) == 0 or
            c.ds4_gpu_tensor_copy(graph.window[index], @as(u64, position % 128) * 512 * 4, graph.kv, 0, 512 * 4) == 0 or
            !self.select(layer, index)) return false;
        const attended: u32 = @min(compressed, shape.n_indexer_top_k);
        const packed_kv = self.packedGather();
        if (compressed != 0 and !self.gather(cacheOwner(index), compressed, attended, packed_kv)) return false;
        const raw = self.rawPast(position, 127) + 1;
        if (c.ds4_gpu_attention_decode_heads_tensor(graph.heads, self.model.map, self.model.size, knownWeight(layer.attn_sinks).abs_offset + @as(u64, head_start) * 4, graph.q, graph.window[index], raw, 128, (position + 1 - raw) % 128, graph.selected_kv, @intFromBool(packed_kv), attended, null, 0, heads, shape.n_head_dim) == 0 or
            !bf16(graph.heads, heads * shape.n_head_dim) or
            !op.rope(graph.heads, heads, shape.n_head_dim, index, position, true) or
            !self.trace(graph.heads, heads * shape.n_head_dim, index, "1d-heads")) return false;
        if (projected) return true;
        return self.output(layer) and op.sumPartial(graph, graph.block, index, c.DS4_TP_GATE_ATTN) and bf16(graph.block, shape.n_embd);
    }

    fn trace(self: *const Attention, tensor: ?*Tensor, width: u32, layer: u32, name: [*:0]const u8) bool {
        // Existing diagnostic I/O only: default builds return true. The graph
        // controller and every GPU/math decision above remain native.
        return c.ds4_native_graph_trace_row(tensor, width, self.graph.pos, 1, layer, name);
    }
};
