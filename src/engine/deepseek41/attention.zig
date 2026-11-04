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
const Narrow = @import("decode_weights.zig");

/// Rebuild a disposable decode view after a load/graft. An older checkpoint
/// may hold F32 keys that are not BF16-exact; never narrow those silently.
pub fn restoreIndexViews(graph: *Record, position: u32) void {
    graph.index_bf16_ok = false;
    graph.block_list_mask = null;
    if (graph.index_fp4 or graph.index_bf16_flag == null) return;
    const zero: u32 = 0;
    if (c.ds4_gpu_tensor_write(graph.index_bf16_flag, 0, &zero, 4) == 0) return;
    for (0..4) |owner| {
        const live = position / @as(u32, if (owner < 3) 2 else 1);
        if (live != 0 and c.ds4_gpu_dsv41_index_keys_bf16(graph.index_bf16[owner], graph.index_cache[owner], 0, live, graph.index_bf16_flag) == 0) return;
    }
    var flag: u32 = 1;
    if (c.ds4_gpu_synchronize() == 0 or c.ds4_gpu_tensor_read(graph.index_bf16_flag, 0, &flag, 4) == 0) return;
    graph.index_bf16_ok = flag == 0;
}

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
        if (self.graph.compressed_kv_fp4) {
            const destination = c.ds4_gpu_tensor_view(self.graph.compressed[owner], @as(u64, first) * 288, @as(u64, rows) * 288);
            defer c.ds4_gpu_tensor_free(destination);
            return destination != null and c.ds4_gpu_dsv41_main_pack_fp4(destination, source, rows) != 0;
        }
        // Mode-3 quantization has already rounded into BF16 values exactly
        // representable in F16. Storage conversion does not requantize them.
        if (self.graph.compressed_kv_f16)
            return c.ds4_gpu_tensor_copy_f32_to_f16(self.graph.compressed[owner], @as(u64, first) * width * 2, source, 0, values) != 0;
        return c.ds4_gpu_tensor_copy(self.graph.compressed[owner], @as(u64, first) * width * 4, source, 0, values * 4) != 0;
    }

    /// Publish original post-RoPE keys directly to their physical cache. FP4
    /// retains the original scale; no full-sized rounded F32 shadow is kept.
    pub fn storeIndex(self: *const Attention, owner: usize, first: u32, source: ?*Tensor, rows: u32) bool {
        if (rows == 0) return true;
        if (self.graph.index_fp4) {
            const destination = c.ds4_gpu_tensor_view(self.graph.index_cache[owner], @as(u64, first) * 68, @as(u64, rows) * 68);
            defer c.ds4_gpu_tensor_free(destination);
            return destination != null and c.ds4_gpu_dsv41_indexer_pack_fp4(destination, source, rows) != 0;
        }
        return c.ds4_gpu_dsv41_quantize(source, 128, rows, c.DS4_V41_FP4_E8M0) != 0 and
            c.ds4_gpu_tensor_copy(self.graph.index_cache[owner], @as(u64, first) * 512, source, 0, @as(u64, rows) * 512) != 0 and
            (self.graph.index_bf16[owner] == null or c.ds4_gpu_dsv41_index_keys_bf16(self.graph.index_bf16[owner], self.graph.index_cache[owner], first, rows, null) != 0);
    }

    fn packedGather(self: *const Attention) bool {
        const value = c.getenv("DS4_METAL_V41_PACKED_F16_GATHER");
        // Main FP4 changes storage only by default. Its half scratch/M5
        // attention path remains an explicit experiment; F16 keeps its default.
        if (self.graph.compressed_kv_fp4) return value != null and c.strcmp(value, "1") == 0;
        return self.graph.compressed_kv_f16 and (value == null or c.strcmp(value, "0") != 0);
    }
    fn gather(self: *const Attention, owner: usize, rows: u32, selected: u32, packed_kv: bool) bool {
        const graph = self.graph;
        if (graph.compressed_kv_fp4)
            return c.ds4_gpu_dsv41_main_gather_fp4(graph.selected_kv, graph.compressed[owner], graph.selected_comp, rows, selected, packed_kv) != 0;
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
        // Cached cumulative state and C2 carries remain immutable during raw-only replay.
        if (position < graph.immutable_cache_end) return true;
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
            self.storeIndex(owner, count - 1, graph.index_k, 1) and
            op.rope(graph.latent, 1, 512, index, rope_position, false) and
            (graph.compressed_kv_fp4 or c.ds4_gpu_dsv41_quantize(graph.latent, 512, 1, c.DS4_V41_FP4_E4M3) != 0) and
            self.storeCompressed(owner, count - 1, graph.latent, 1);
    }

    pub fn candidates(self: *const Attention, index: u32) bool {
        const graph = self.graph;
        const policy = self.config.layerPolicy(index).?;
        const count = if (policy.compression_ratio != 0) (graph.pos + 1) / policy.compression_ratio else 0;
        if (count == 0 or !policy.publishes_index) return true;
        if (index == 20) {
            graph.block_list_mask = null;
            const blocks = (count + 7) / 8;
            const top: u32 = @min(blocks, 2048);
            // Below the cap every block survives. Only the mask is consumed,
            // so sorting its unused all-block selection would add no value.
            if (top == blocks) return c.ds4_gpu_tensor_fill_f32(graph.block_mask, 0, blocks) != 0;
            return c.ds4_gpu_dsv41_candidate_blocks(graph.block_scores, graph.index_scores, count, 1, graph.pos, policy.compression_ratio) != 0 and
                c.ds4_gpu_dsv41_indexer_topk_tensor(graph.block_selected, graph.block_scores, blocks, 1, top) != 0 and
                c.ds4_gpu_dsv4_topk_mask_tensor(graph.block_mask, graph.block_selected, blocks, 1, top) != 0;
        }
        return index <= 20 or c.ds4_gpu_dsv41_candidate_filter(graph.index_scores, graph.block_mask, count, 1, graph.pos, policy.compression_ratio, 0) != 0;
    }

    pub fn selectPublished(self: *const Attention, layer: *const Layer, index: u32) bool {
        const graph = self.graph;
        const shape = &self.config.shape;
        const policy = self.config.layerPolicy(index).?;
        const count = if (policy.compression_ratio != 0) (graph.pos + 1) / policy.compression_ratio else 0;
        if (count != 0 and policy.publishes_index) {
            const op = self.ops();
            const candidate_setting = c.getenv("DS4_V41_CANDIDATE_INDEX");
            // Layer20 still scores every key to choose the trained candidate
            // set, including its forced newest block. Later source layers can
            // skip discarded dot products once that set excludes any keys.
            const candidate_only = index > 20 and count > 2048 * 8 and
                candidate_setting != null and c.strcmp(candidate_setting, "1") == 0;
            const copies = Narrow.Copies.fromOpaque(graph.decode_weights);
            const copied = if (copies) |weights| !graph.quality and Narrow.project(weights.indexer[index], graph.index_q, graph.qr, layer.indexer_attn_q_b) else false;
            if ((!copied and !op.matmul(graph.index_q, knownWeight(layer.indexer_attn_q_b), graph.qr, true)) or
                !self.indexQueryRope(index) or
                !op.matmul(graph.index_weights, knownWeight(layer.indexer_proj), graph.norm, true)) return false;
            const simd_setting = c.getenv("DS4_V41_INDEX_SIMD");
            // The register layout wins at the measured 1K+ key points; the
            // old layout wins at 128/256 keys. Keep small work on that path.
            const simd_enabled = if (simd_setting == null) graph.index_fp4 else c.strcmp(simd_setting, "1") == 0;
            const simd = count >= 1024 and simd_enabled;
            // The trained later-layer mask excludes these blocks exactly.
            // The native F32 path combines candidate scoring and selection;
            // packed keys retain their measured SIMD implementation.
            if (!graph.index_fp4 and index > 20 and (count + 7) / 8 > 2048 and
                c.ds4_gpu_dsv41_indexer_candidates_tensor(graph.selected_comp, graph.index_scores, graph.block_list, graph.index_q, graph.index_weights, graph.index_cache[cacheOwner(index)], graph.block_mask, count, 1.0 / 64.0, shape.n_indexer_top_k, @intFromBool(graph.block_list_mask == graph.block_mask)) != 0)
            {
                graph.block_list_mask = graph.block_mask;
                return true;
            }
            const scored = if (simd)
                c.ds4_gpu_dsv41_indexer_score_simd(graph.index_scores, graph.index_q, graph.index_weights, graph.index_cache[cacheOwner(index)], if (candidate_only) graph.block_selected else null, count, if (candidate_only) 2048 else 0, graph.index_fp4)
            else if (graph.index_fp4)
                c.ds4_gpu_dsv41_indexer_score_fp4(graph.index_scores, graph.index_q, graph.index_weights, graph.index_cache[cacheOwner(index)], if (candidate_only) graph.block_selected else null, count, if (candidate_only) 2048 else 0)
            else if (candidate_only)
                c.ds4_gpu_dsv41_indexer_score_candidates(graph.index_scores, graph.index_q, graph.index_weights, graph.index_cache[cacheOwner(index)], graph.block_selected, count, 2048)
            else
                c.ds4_gpu_dsv41_indexer_score_row(graph.index_scores, graph.index_q, graph.index_weights, graph.index_cache[cacheOwner(index)], if (graph.index_bf16_ok) graph.index_bf16[cacheOwner(index)] else null, count);
            if (scored == 0) return false;
        }
        return self.pick(index);
    }

    fn indexQueryRope(self: *const Attention, index: u32) bool {
        const graph = self.graph;
        const heads = self.config.shape.n_indexer_head;
        const compressed = self.config.compress_ratios[index] != 0;
        if (compressed and graph.tp_world == 1 and graph.rope_tab_pos == graph.pos + 1 and
            c.ds4_gpu_dsv41_rope_quantize_heads_tab(graph.index_q, 128, heads, graph.pos, true, c.DS4_V41_FP4_E8M0, graph.rope_tab) != 0) return true;
        const op = self.ops();
        return op.rope(graph.index_q, heads, 128, index, graph.pos, false) and
            c.ds4_gpu_dsv41_quantize(graph.index_q, 128, heads, c.DS4_V41_FP4_E8M0) != 0;
    }

    pub fn pick(self: *const Attention, index: u32) bool {
        const graph = self.graph;
        const policy = self.config.layerPolicy(index).?;
        const count = if (policy.compression_ratio != 0) (graph.pos + 1) / policy.compression_ratio else 0;
        const top: u32 = @min(count, self.config.shape.n_indexer_top_k);
        return self.candidates(index) and (count == 0 or !policy.publishes_index or
            c.ds4_gpu_dsv41_indexer_topk_tensor(graph.selected_comp, graph.index_scores, count, 1, top) != 0);
    }

    fn select(self: *const Attention, layer: *const Layer, index: u32) bool {
        return self.publish(layer, index) and self.selectPublished(layer, index);
    }

    pub fn project(self: *const Attention, layer: *const Layer, index: u32, roped: *bool) bool {
        const graph = self.graph;
        const shape = &self.config.shape;
        const query_width = shape.n_head / graph.tp_world * shape.n_head_dim;
        const op = self.ops();
        roped.* = false;
        const paired = layer.attn_q_a.*.type == c.DS4_TENSOR_Q8_0 and layer.attn_kv.*.type == c.DS4_TENSOR_Q8_0 and
            c.ds4_gpu_dsv41_project_pair_q8(graph.qr, graph.kv, self.model.map, self.model.size, layer.attn_q_a.*.abs_offset, layer.attn_kv.*.abs_offset, shape.n_embd, @intCast(layer.attn_q_a.*.dim[1]), @intCast(layer.attn_kv.*.dim[1]), graph.norm) != 0;
        if (!paired and (!op.matmul(graph.qr, knownWeight(layer.attn_q_a), graph.norm, true) or
            !op.matmul(graph.kv, knownWeight(layer.attn_kv), graph.norm, true))) return false;
        const norms = c.ds4_gpu_dsv41_norm_pair_rows(graph.qr, graph.qr, layer.attn_q_a_norm.*.abs_offset, shape.n_lora_q, graph.kv, graph.kv, layer.attn_kv_a_norm.*.abs_offset, shape.n_head_dim, self.model.map, self.model.size, shape.rms_eps, 1) != 0;
        if (!norms and (!op.norm(graph.qr, graph.qr, knownWeight(layer.attn_q_a_norm)) or !op.norm(graph.kv, graph.kv, knownWeight(layer.attn_kv_a_norm)))) return false;
        if (graph.hc_attn_deferred) {
            // The fused publisher consumes KV after its BF16 norm. Preserve
            // the immutable pre-RoPE tape before it publishes the raw ring.
            if (!self.archiveRaw(index, graph.kv, graph.pos, 1) or
                c.ds4_gpu_dsv41_query_deferred_hc(graph.q, graph.mix, graph.attn_split, graph.qr, graph.residual, self.model.map, self.model.size, layer.attn_q_b.*.abs_offset, layer.hc_attn_fn.*.abs_offset, layer.hc_attn_scale.*.abs_offset, layer.hc_attn_base.*.abs_offset, graph.pos, self.config.compress_ratios[index] != 0, shape.n_hc_sinkhorn_iter, shape.hc_eps, shape.rms_eps, graph.x, graph.norm, graph.pre, layer.attn_norm.*.abs_offset, graph.kv, graph.window[index], @as(u64, graph.pos % 128) * shape.n_head_dim * 4) == 0) return false;
            graph.kv_stored = true;
            graph.hc_attn_deferred = false;
            roped.* = true;
            return true;
        }
        if (layer.attn_q_b.*.type == c.DS4_TENSOR_Q8_0) {
            var bytes: u64 = 0;
            if (c.ds4_gguf_tensor_nbytes(layer.attn_q_b.*.type, layer.attn_q_b.*.dim[0], &bytes)) {
                const table_setting = c.getenv("DS4_METAL_V41_SHORT_Q8_QUERY_TABLE");
                // The scalar executor publishes validity only after encoding
                // this position's table. The graph owns its storage across
                // streaming commits; step boundaries clear the validity marker.
                const table = table_setting != null and c.strcmp(table_setting, "1") == 0 and
                    graph.rope_tab != null and graph.rope_tab_pos == graph.pos + 1;
                const offset = layer.attn_q_b.*.abs_offset + @as(u64, graph.tp_rank) * query_width * bytes;
                const projected = if (table)
                    c.ds4_gpu_dsv41_project_q_tab(graph.q, self.model.map, self.model.size, offset, @intCast(layer.attn_q_b.*.dim[0]), query_width, graph.qr, graph.pos, self.config.compress_ratios[index] != 0, graph.rope_tab)
                else
                    c.ds4_gpu_dsv41_project_q(graph.q, self.model.map, self.model.size, offset, @intCast(layer.attn_q_b.*.dim[0]), query_width, graph.qr, graph.pos, self.config.compress_ratios[index] != 0);
                if (projected != 0) {
                    roped.* = true;
                    return true;
                }
            }
        }
        return op.matmulRows(graph.q, knownWeight(layer.attn_q_b), graph.qr, graph.tp_rank * query_width, query_width);
    }

    pub fn low(self: *const Attention, layer: *const Layer) bool {
        // Row-by-row prefill callers consume a BF16 materialization, unlike
        // scalar outputB whose Q8 reader performs that rounding on load.
        return self.lowProjected(layer, null, false) and bf16(self.graph.low, self.config.shape.n_out_group / self.graph.tp_world * self.config.shape.n_lora_o);
    }

    fn lowProjected(self: *const Attention, layer: *const Layer, index: ?u32, transformed: bool) bool {
        const graph = self.graph;
        const shape = &self.config.shape;
        const groups = shape.n_out_group / graph.tp_world;
        const width = shape.n_head_dim * (shape.n_head / shape.n_out_group);
        var row_bytes: u64 = 0;
        if (!c.ds4_gguf_tensor_nbytes(layer.attn_output_a.*.type, width, &row_bytes)) return false;
        const offset = layer.attn_output_a.*.abs_offset + @as(u64, graph.tp_rank) * groups * shape.n_lora_o * row_bytes;
        graph.low_unrounded = false;
        if (transformed) {
            if (c.ds4_gpu_dsv41_attention_low_plain(graph.low, self.model.map, self.model.size, offset, width, shape.n_lora_o, groups, graph.heads) == 0) return false;
        } else {
            const fused = if (index) |il| c.ds4_gpu_dsv41_attention_low(graph.low, self.model.map, self.model.size, offset, width, shape.n_lora_o, groups, graph.heads, graph.pos, self.config.compress_ratios[il] != 0) != 0 else false;
            if (!fused) {
                if (index) |il| {
                    if (c.ds4_gpu_dsv41_rope_bf16(graph.heads, shape.n_head_dim, shape.n_head / graph.tp_world, 1, graph.pos, self.config.compress_ratios[il] != 0, true) == 0) return false;
                }
                if (c.ds4_gpu_attention_output_low_q8_tensor(graph.low, self.model.map, self.model.size, offset, width, shape.n_lora_o, groups, graph.heads) == 0) return false;
            }
        }
        // Q8 readers perform BF16 rounding as they load low. TP needs the
        // corresponding K-slice reader; otherwise retain the separate round.
        graph.low_unrounded = graph.tp_world == 2 and layer.attn_output_b.*.type == c.DS4_TENSOR_Q8_0 and c.getenv("DS4_METAL_DISABLE_V41_TP_KSLICE_ROUND_IN") == null;
        return graph.low_unrounded or (graph.tp_world == 1 and layer.attn_output_b.*.type == c.DS4_TENSOR_Q8_0) or bf16(graph.low, groups * shape.n_lora_o);
    }

    pub fn output(self: *const Attention, layer: *const Layer) bool {
        return self.low(layer) and self.outputB(layer, self.graph.block);
    }

    fn outputB(self: *const Attention, layer: *const Layer, out: ?*Tensor) bool {
        const graph = self.graph;
        const shape = &self.config.shape;
        const groups = shape.n_out_group / graph.tp_world;
        const op = self.ops();
        if (graph.tp_world == 2) {
            if (graph.low_unrounded) {
                graph.low_unrounded = false;
                if (c.ds4_gpu_matmul_q8_0_kslice_bf16in_tensor(out, self.model.map, self.model.size, layer.attn_output_b.*.abs_offset, @as(u64, shape.n_out_group) * shape.n_lora_o, @as(u64, graph.tp_rank) * groups * shape.n_lora_o, @as(u64, groups) * shape.n_lora_o, shape.n_embd, graph.low) != 0) return true;
                if (!bf16(graph.low, groups * shape.n_lora_o)) return false;
            }
            return op.quantKslice(out, knownWeight(layer.attn_output_b), @as(u64, shape.n_out_group) * shape.n_lora_o, @as(u64, graph.tp_rank) * groups * shape.n_lora_o, @as(u64, groups) * shape.n_lora_o, shape.n_embd, graph.low, 0);
        }
        if (layer.attn_output_b.*.type == c.DS4_TENSOR_Q8_0)
            return c.ds4_gpu_matmul_q8_0_bf16io_tensor(out, self.model.map, self.model.size, layer.attn_output_b.*.abs_offset, layer.attn_output_b.*.dim[0], layer.attn_output_b.*.dim[1], graph.low, 1) != 0;
        return op.matmul(out, knownWeight(layer.attn_output_b), graph.low, true);
    }

    fn expandOutput(self: *const Attention, layer: *const Layer) bool {
        const graph = self.graph;
        if (c.ds4_native_graph_trace_active() or graph.steering_attn_scale != 0 or c.getenv("DS4_METAL_DISABLE_V41_EXPAND_FUSION") != null or layer.attn_output_b.*.type != c.DS4_TENSOR_Q8_0) return false;
        graph.attn_expanded = c.ds4_gpu_dsv41_matmul_expand_rows(graph.after_attn, self.model.map, self.model.size, layer.attn_output_b.*.abs_offset, @intCast(layer.attn_output_b.*.dim[0]), @intCast(layer.attn_output_b.*.dim[1]), graph.low, null, graph.residual, graph.attn_split, self.config.shape.n_hc, 1) != 0;
        return graph.attn_expanded;
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
        var query_roped = false;
        graph.kv_stored = false;
        graph.attn_expanded = false;
        if (!projected and !self.project(layer, index, &query_roped)) return false;
        if (!self.trace(graph.qr, shape.n_lora_q, index, "1a-qr") or
            !self.trace(graph.q, heads * shape.n_head_dim, index, "1b-q") or
            !self.trace(graph.kv, shape.n_head_dim, index, "1c-kv")) return false;
        const op = self.ops();
        if ((!graph.kv_stored and !self.archiveRaw(index, graph.kv, position, 1)) or
            (!query_roped and !op.rope(graph.q, heads, shape.n_head_dim, index, position, false)) or
            (!graph.kv_stored and !self.storeRaw(graph.kv, index, position)) or
            !self.select(layer, index)) return false;
        const attended: u32 = @min(compressed, shape.n_indexer_top_k);
        const packed_kv = self.packedGather();
        const raw = self.rawPast(position, 127) + 1;
        const tracing = c.ds4_native_graph_trace_active();
        const arm = !projected and !tracing and graph.tp_world == 1 and graph.rope_tab_pos == position + 1;
        if (arm) c.ds4_gpu_dsv41_heads_once_arm(graph.rope_tab, policy.compression_ratio != 0);
        const sinks = knownWeight(layer.attn_sinks).abs_offset + @as(u64, head_start) * 4;
        // This upstream staging path reads expanded F32 rows. Packed local
        // caches use their own exact gather before the same decode operation.
        const staged = compressed != 0 and !graph.compressed_kv_fp4 and !graph.compressed_kv_f16 and
            c.ds4_gpu_dsv41_attention_decode(graph.heads, self.model.map, self.model.size, sinks, graph.q, graph.window[index], raw, 128, (position + 1 - raw) % 128, graph.compressed[cacheOwner(index)], graph.selected_comp, graph.selected_kv, compressed, attended, heads, shape.n_head_dim) != 0;
        const decoded = staged or
            ((compressed == 0 or self.gather(cacheOwner(index), compressed, attended, packed_kv)) and
                c.ds4_gpu_attention_decode_heads_tensor(graph.heads, self.model.map, self.model.size, sinks, graph.q, graph.window[index], raw, 128, (position + 1 - raw) % 128, graph.selected_kv, @intFromBool(packed_kv), attended, null, 0, heads, shape.n_head_dim) != 0);
        const transformed = arm and c.ds4_gpu_dsv41_heads_once_take() != 0;
        if (!decoded) return false;
        if (projected) return c.ds4_gpu_dsv41_rope_bf16(graph.heads, shape.n_head_dim, heads, 1, position, policy.compression_ratio != 0, true) != 0;
        if (tracing) {
            if (c.ds4_gpu_dsv41_rope_bf16(graph.heads, shape.n_head_dim, heads, 1, position, policy.compression_ratio != 0, true) == 0 or
                !self.trace(graph.heads, heads * shape.n_head_dim, index, "1d-heads") or !self.low(layer)) return false;
        } else if (!self.lowProjected(layer, index, transformed)) return false;
        if (graph.tp_world == 1 and self.expandOutput(layer)) return true;
        const destination = if (graph.tp_world == 2 and !tracing and c.getenv("DS4_METAL_DISABLE_V41_TP_SLAB_WRITE") == null)
            graph.tp_out[index * c.DS4_TP_GATES_PER_LAYER + c.DS4_TP_GATE_ATTN]
        else
            graph.block;
        if (destination != graph.block) c.ds4_gpu_tp_flag_fold_request(index, c.DS4_TP_GATE_ATTN);
        return self.outputB(layer, destination) and (graph.tp_world == 1 or
            (if (graph.tp_gate_fused) op.gatePartial(graph, destination, index, c.DS4_TP_GATE_ATTN) else op.sumPartialInto(graph, destination, graph.block, index, c.DS4_TP_GATE_ATTN) and bf16(graph.block, shape.n_embd)));
    }

    fn storeRaw(self: *const Attention, key: ?*Tensor, layer: u32, position: u32) bool {
        const graph = self.graph;
        const width = self.config.shape.n_head_dim;
        const offset = @as(u64, position % 128) * width * 4;
        if (c.ds4_gpu_dsv41_rope_quantize(key, graph.window[layer], offset, width, position, self.config.compress_ratios[layer] != 0, c.DS4_V41_FP8_E8M0) != 0) return true;
        const op = self.ops();
        return op.rope(key, 1, width, layer, position, false) and
            c.ds4_gpu_dsv41_quantize(key, width, 1, c.DS4_V41_FP8_E8M0) != 0 and
            c.ds4_gpu_tensor_copy(graph.window[layer], offset, key, 0, @as(u64, width) * 4) != 0;
    }

    fn trace(self: *const Attention, tensor: ?*Tensor, width: u32, layer: u32, name: [*:0]const u8) bool {
        // Existing diagnostic I/O only: default builds return true. The graph
        // controller and every GPU/math decision above remain native.
        return c.ds4_native_graph_trace_row(tensor, width, self.graph.pos, 1, layer, name);
    }
};
