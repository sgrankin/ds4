//! V4.1 support operations over an explicit model/config. Command lifetime and
//! device tensor ownership belong to the graph caller; temporary row views are
//! released here. No process-global shape or precision-mode accessors.
const std = @import("std");
const c = @import("gguf_c");
const ModelConfig = @import("../config.zig").ModelConfig;
const Tensor = c.ds4_gpu_tensor;
const Weight = c.ds4_tensor;
const Graph = c.ds41_gpu_graph;
const Row = c.ds41_prefill_row;
const Layer = c.ds4_layer_weights;

fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn dimension(value: u64) u32 {
    return @truncate(value); // Existing C narrowing; validated model dimensions.
}
pub fn bf16(tensor: ?*Tensor, width: u32) bool {
    return c.ds4_gpu_dsv41_quantize(tensor, width, 1, c.DS4_V41_BF16) != 0;
}
fn denseQuant(kind: u32) bool {
    return kind == c.DS4_TENSOR_Q8_0 or kind == c.DS4_TENSOR_Q4_K or kind == c.DS4_TENSOR_Q4_0;
}

pub const Ops = struct {
    config: *const ModelConfig,
    model: *const c.ds4_model,
    // Scoped to the inference call. Diagnostic/short-prefill precision must
    // not change another session by mutating process-wide environment flags.
    exact_prefill_rows: bool = false,

    pub fn exactDenseRows(self: *const Ops) bool {
        return self.exact_prefill_rows or present("DS4_METAL_V41_EXACT_DENSE_PREFILL");
    }

    pub fn plainMatmul(self: *const Ops, out: ?*Tensor, weight: *const Weight, width: u64, outputs: u64, input: ?*const Tensor, rows: u64) bool {
        const model = self.model;
        return switch (weight.type) {
            c.DS4_TENSOR_F16 => c.ds4_gpu_matmul_f16_tensor(out, model.map, model.size, weight.abs_offset, width, outputs, input, rows) != 0,
            c.DS4_TENSOR_F32 => c.ds4_gpu_matmul_f32_tensor(out, model.map, model.size, weight.abs_offset, width, outputs, input, rows) != 0,
            c.DS4_TENSOR_Q8_0 => c.ds4_gpu_matmul_q8_0_tensor(out, model.map, model.size, weight.abs_offset, width, outputs, input, rows) != 0,
            c.DS4_TENSOR_Q4_K, c.DS4_TENSOR_Q4_0 => c.ds4_gpu_matmul_quant_tensor(out, model.map, model.size, weight.abs_offset, weight.type, width, outputs, input, rows) != 0,
            else => unsupported: {
                const info = c.ds4_gguf_tensor_type(weight.type);
                const name: [*c]const u8 = if (info == null) "unknown" else info.*.name;
                _ = c.fprintf(c.__stderrp, "ds4: Metal plain matmul does not support %s\n", name);
                break :unsupported false;
            },
        };
    }

    pub fn quantKslice(self: *const Ops, out: ?*Tensor, weight: ?*const Weight, full_width: u64, first: u64, count: u64, outputs: u64, input: ?*const Tensor, input_element_offset: u64) bool {
        const projection = weight orelse return false;
        if (!denseQuant(projection.type)) return false;
        return c.ds4_gpu_matmul_quant_kslice_tensor(out, self.model.map, self.model.size, projection.abs_offset, projection.type, full_width, first, count, outputs, input, input_element_offset) != 0;
    }

    pub fn matmul(self: *const Ops, out: ?*Tensor, weight: *const Weight, input: ?*const Tensor, round: bool) bool {
        if (round and weight.type == c.DS4_TENSOR_Q8_0 and present("DS4_METAL_V41_FUSED_Q8_BF16"))
            return c.ds4_gpu_dsv41_matmul_q8_0_bf16_rows(out, self.model.map, self.model.size, weight.abs_offset, weight.dim[0], weight.dim[1], input, 1) != 0;
        return self.plainMatmul(out, weight, weight.dim[0], weight.dim[1], input, 1) and
            (!round or bf16(out, dimension(weight.dim[1])));
    }

    pub fn matmulBatch(self: *const Ops, out: ?*Tensor, weight: *const Weight, input: ?*const Tensor, count: u32, round: bool) bool {
        const width = dimension(weight.dim[0]);
        const outputs = dimension(weight.dim[1]);
        const exact_rows = count <= c.DS4_TP_BATCH_MAX_ROWS or self.exactDenseRows();
        const model = self.model;
        if (round and exact_rows and outputs != self.config.shape.n_vocab and
            weight.type == c.DS4_TENSOR_Q8_0 and present("DS4_METAL_V41_FUSED_Q8_BF16"))
            return c.ds4_gpu_dsv41_matmul_q8_0_bf16_rows(out, model.map, model.size, weight.abs_offset, width, outputs, input, count) != 0;

        // Small decode batches retain scalar reductions before BF16/sparse
        // routing boundaries. Metal's vocabulary-head dispatch stays separate.
        const ok = if (count >= 2 and exact_rows and outputs != self.config.shape.n_vocab and weight.type == c.DS4_TENSOR_Q8_0)
            c.ds4_gpu_matmul_q8_0_decode_rows_exact_tensor(out, model.map, model.size, weight.abs_offset, width, outputs, input, count) != 0
        else if (count >= 2 and exact_rows and weight.type == c.DS4_TENSOR_F16)
            c.ds4_gpu_dsv41_projection_rows(out, model.map, model.size, weight.abs_offset, width, outputs, count, input) != 0
        else if (count >= 2 and exact_rows and weight.type == c.DS4_TENSOR_F32)
            self.f32Rows(out, weight, input, width, outputs, count)
        else
            self.plainMatmul(out, weight, width, outputs, input, count);
        return ok and (!round or c.ds4_gpu_dsv41_quantize(out, outputs, count, c.DS4_V41_BF16) != 0);
    }

    fn f32Rows(self: *const Ops, out: ?*Tensor, weight: *const Weight, input: ?*const Tensor, width: u32, outputs: u32, count: u32) bool {
        for (0..count) |index| {
            const x = c.ds4_gpu_tensor_view(input, @as(u64, index) * width * 4, @as(u64, width) * 4);
            const y = c.ds4_gpu_tensor_view(out, @as(u64, index) * outputs * 4, @as(u64, outputs) * 4);
            const ok = x != null and y != null and self.matmul(y, weight, x, false);
            c.ds4_gpu_tensor_free(x);
            c.ds4_gpu_tensor_free(y);
            if (!ok) return false;
        }
        return true;
    }

    pub fn matmulRowsBatch(self: *const Ops, out: ?*Tensor, weight: *const Weight, input: ?*const Tensor, first: u32, count: u32, rows: u32) bool {
        var row_bytes: u64 = undefined;
        if (first > weight.dim[1] or count > weight.dim[1] - first or
            !c.ds4_gguf_tensor_nbytes(weight.type, weight.dim[0], &row_bytes)) return false;
        var slice = weight.*;
        slice.abs_offset +%= @as(u64, first) *% row_bytes;
        slice.dim[1] = count;
        return self.matmulBatch(out, &slice, input, rows, true);
    }

    pub fn matmulRows(self: *const Ops, out: ?*Tensor, weight: *const Weight, input: ?*const Tensor, first: u32, count: u32) bool {
        return self.matmulRowsBatch(out, weight, input, first, count, 1);
    }

    pub fn outputProjection(self: *const Ops, graph: *const Graph, out: ?*Tensor, weights: *const c.ds4_weights, input: ?*const Tensor, rows: u32) bool {
        var head = weights.output.*;
        if (graph.tp_logits_half != null) {
            var row_bytes: u64 = undefined;
            if (!c.ds4_gguf_tensor_nbytes(head.type, head.dim[0], &row_bytes)) return false;
            head.dim[1] /= 2;
            head.abs_offset +%= @as(u64, graph.tp_rank) *% head.dim[1] *% row_bytes;
        }
        // Head rows are independent. Keep their reduction and do not add the
        // BF16 rounding used by attention's row-slice helper.
        return self.matmulBatch(out, &head, input, rows, false);
    }

    pub fn norm(self: *const Ops, out: ?*Tensor, input: ?*const Tensor, weight: *const Weight) bool {
        const width = dimension(weight.dim[0]);
        if (present("DS4_METAL_V41_FUSED_NORM"))
            return c.ds4_gpu_dsv41_rms_norm_weight_bf16_rows(out, input, self.model.map, self.model.size, weight.abs_offset, width, 1, self.config.shape.rms_eps) != 0;
        return c.ds4_gpu_rms_norm_weight_tensor(out, input, self.model.map, self.model.size, weight.abs_offset, width, self.config.shape.rms_eps) != 0 and bf16(out, width);
    }

    pub fn normBatch(self: *const Ops, out: ?*Tensor, input: ?*const Tensor, weight: *const Weight, count: u32) bool {
        const width = dimension(weight.dim[0]);
        if (present("DS4_METAL_V41_FUSED_NORM"))
            return c.ds4_gpu_dsv41_rms_norm_weight_bf16_rows(out, input, self.model.map, self.model.size, weight.abs_offset, width, count, self.config.shape.rms_eps) != 0;
        return c.ds4_gpu_rms_norm_weight_rows_tensor(out, input, self.model.map, self.model.size, weight.abs_offset, width, count, self.config.shape.rms_eps) != 0 and
            c.ds4_gpu_dsv41_quantize(out, width, count, c.DS4_V41_BF16) != 0;
    }

    pub fn hcSum(self: *const Ops, out: ?*Tensor, input: ?*const Tensor, weights: ?*const Tensor, split: bool) bool {
        const shape = &self.config.shape;
        const ok = if (split)
            c.ds4_gpu_hc_weighted_sum_split_tensor(out, input, weights, shape.n_embd, shape.n_hc) != 0
        else
            c.ds4_gpu_hc_weighted_sum_tensor(out, input, weights, shape.n_embd, shape.n_hc) != 0;
        return ok and bf16(out, shape.n_embd);
    }

    pub fn hcExpand(self: *const Ops, out: ?*Tensor, block: ?*const Tensor, residual: ?*const Tensor, split: ?*const Tensor) bool {
        const shape = &self.config.shape;
        return c.ds4_gpu_hc_expand_split_tensor(out, block, residual, split, shape.n_embd, shape.n_hc) != 0 and
            bf16(out, shape.n_embd *% shape.n_hc);
    }

    pub fn hcMix(self: *const Ops, graph: *Graph, layer: *const Layer, ffn: bool) bool {
        const shape = &self.config.shape;
        const residual = if (ffn) graph.after_attn else graph.residual;
        const projection: *const Weight = if (ffn) layer.hc_ffn_fn else layer.hc_attn_fn;
        const scale: *const Weight = if (ffn) layer.hc_ffn_scale else layer.hc_attn_scale;
        const base: *const Weight = if (ffn) layer.hc_ffn_base else layer.hc_attn_base;
        return c.ds4_gpu_rms_norm_plain_tensor(graph.flat_norm, residual, shape.n_hc *% shape.n_embd, shape.rms_eps) != 0 and
            self.matmul(graph.mix, projection, graph.flat_norm, false) and
            c.ds4_gpu_hc_split_sinkhorn_tensor(if (ffn) graph.ffn_split else graph.attn_split, graph.mix, self.model.map, self.model.size, scale.abs_offset, base.abs_offset, shape.n_hc, shape.n_hc_sinkhorn_iter, shape.hc_eps) != 0;
    }

    pub fn hcMixBatch(self: *const Ops, row: *Row, layer: *const Layer, ffn: bool, count: u32) bool {
        const shape = &self.config.shape;
        const projection: *const Weight = if (ffn) layer.hc_ffn_fn else layer.hc_attn_fn;
        const scale: *const Weight = if (ffn) layer.hc_ffn_scale else layer.hc_attn_scale;
        const base: *const Weight = if (ffn) layer.hc_ffn_base else layer.hc_attn_base;
        const input = if (ffn) row.after_attn else row.residual;
        const projected = if (projection.type == c.DS4_TENSOR_F16 and count > c.DS4_TP_BATCH_MAX_ROWS and !self.exactDenseRows())
            c.ds4_gpu_hc_rms_scale_project_f16_tensor(row.mix, row.flat_norm, self.model.map, self.model.size, projection.abs_offset, shape.n_hc *% shape.n_embd, 24, input, count, shape.rms_eps) != 0
        else
            c.ds4_gpu_rms_norm_plain_rows_tensor(row.flat_norm, input, shape.n_hc *% shape.n_embd, count, shape.rms_eps) != 0 and
                self.matmulBatch(row.mix, projection, row.flat_norm, count, false);
        return projected and c.ds4_gpu_hc_split_sinkhorn_tensor(if (ffn) row.ffn_split else row.attn_split, row.mix, self.model.map, self.model.size, scale.abs_offset, base.abs_offset, shape.n_hc, shape.n_hc_sinkhorn_iter, shape.hc_eps) != 0;
    }

    pub fn rope(self: *const Ops, tensor: ?*Tensor, heads: u32, width: u32, layer: u32, position: u32, inverse: bool) bool {
        if (layer >= self.config.shape.n_layer)
            c.ds4_engine_model_die("DeepSeek4 layer index is outside the loaded model layout");
        return c.ds4_gpu_dsv41_rope(tensor, width, heads, 1, position, self.config.compress_ratios[layer] != 0, inverse) != 0;
    }

    pub fn sumPartial(self: *const Ops, graph: *Graph, tensor: ?*Tensor, layer: u32, gate: u32) bool {
        if (graph.tp_world != 2) return true;
        const slot = layer *% c.DS4_TP_GATES_PER_LAYER +% gate;
        if (c.ds4_gpu_tensor_copy(graph.tp_out[slot], 0, tensor, 0, @as(u64, self.config.shape.n_embd) * 4) == 0 or
            c.ds4_gpu_tp_gate_encode(layer, gate) == 0) return false;
        const first = if (graph.tp_rank != 0) graph.tp_in[slot] else graph.tp_out[slot];
        const second = if (graph.tp_rank != 0) graph.tp_out[slot] else graph.tp_in[slot];
        return c.ds4_gpu_add_tensor(tensor, first, second, self.config.shape.n_embd) != 0;
    }

    pub fn sumPartialBatch(self: *const Ops, graph: *Graph, tensor: ?*Tensor, layer: u32, gate: u32, count: u32) bool {
        if (graph.tp_world != 2) return true;
        const width = self.config.shape.n_embd;
        const bytes = @as(u64, count) * width * 4;
        if (graph.verify_prefix) {
            const slot = layer *% c.DS4_TP_BATCH_GATES_PER_LAYER +% gate;
            if (graph.tp_batch_out == null or graph.tp_batch_in == null or count > c.DS4_TP_BATCH_MAX_ROWS or
                c.ds4_gpu_tensor_copy(graph.tp_batch_out[slot], 0, tensor, 0, bytes) == 0 or
                c.ds4_gpu_tp_batch_gate_encode(slot, count) == 0) return false;
            return c.ds4_gpu_add_tensor(tensor, if (graph.tp_rank != 0) graph.tp_batch_in[slot] else graph.tp_batch_out[slot], if (graph.tp_rank != 0) graph.tp_batch_out[slot] else graph.tp_batch_in[slot], count *% width) != 0;
        }
        // Q is dead after attention; its expert-output alias is dead after
        // routed reduction. Borrow it for the peer without another buffer.
        const peer = graph.batch.q;
        if (c.ds4_gpu_tp_big_gate_encode(layer, count, tensor, peer, bytes) == 0) return false;
        return c.ds4_gpu_add_tensor(tensor, if (graph.tp_rank != 0) peer else tensor, if (graph.tp_rank != 0) tensor else peer, count *% width) != 0;
    }

    pub fn imageAt(_: *const Ops, graph: *const Graph, position: u32) ?*const c.ds4_vision_span {
        if (graph.image_count == 0) return null;
        const spans: [*]const c.ds4_vision_span = graph.images;
        for (spans[0..graph.image_count]) |*span| {
            if (position < span.token_start) break;
            if (@as(u64, position) - span.token_start < span.embedding.token_count) return span;
        }
        return null;
    }

    pub fn textMask(_: *const Ops, graph: *const Graph, position: u32, count: u32, mask: [*c]u8) void {
        if (count != 0) @memset(mask[0..count], 1);
        if (graph.image_count == 0) return;
        const spans: [*]const c.ds4_vision_span = graph.images;
        for (spans[0..graph.image_count]) |span| {
            const end = @as(u64, span.token_start) + span.embedding.token_count;
            const first = @max(span.token_start, position);
            const last = @min(end, @as(u64, position) + count);
            if (first < last) @memset(mask[first - position .. @intCast(last - position)], 0);
        }
    }

    pub fn hashTokens(self: *const Ops, graph: *Graph, history: *c.ds4_engram_history, tokens: [*c]const c_int, count: u32, ids: [*c]u32) bool {
        var memory: ?*anyopaque = null;
        defer c.free(memory);
        var mask: [*c]u8 = null;
        if (graph.image_count != 0) {
            memory = c.malloc(count) orelse return false;
            mask = @ptrCast(memory.?);
            self.textMask(graph, graph.pos, count, mask);
        }
        return c.ds4_engram_hash(&graph.engram, history, tokens, mask, count, ids);
    }

    pub fn embed(self: *const Ops, graph: *const Graph, weights: *const c.ds4_weights, out: ?*Tensor, scratch: ?*Tensor, token: c_int, position: u32) bool {
        const shape = &self.config.shape;
        const image = self.imageAt(graph, position) orelse
            return c.ds4_gpu_embed_token_hc_tensor(out, self.model.map, self.model.size, weights.token_embd.*.abs_offset, shape.n_vocab, @as(u32, @bitCast(token)), shape.n_embd, shape.n_hc) != 0;
        const offset = @as(u64, position - image.token_start) * shape.n_embd;
        const data = image.embedding.data + @as(usize, @intCast(offset));
        return c.ds4_gpu_tensor_write(scratch, 0, data, @as(u64, shape.n_embd) * 4) != 0 and
            c.ds4_gpu_repeat_hc_tensor(out, scratch, shape.n_embd, shape.n_hc) != 0;
    }
};
