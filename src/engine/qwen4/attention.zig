//! GDN recurrent layers and indexed gated attention. Sparse selection begins
//! only after the dense block budget; exact three-row verifies retain2/1-row
//! kernel paths and each sub-batch's own causal block universe.
const c = @import("gguf_c");
const ops = @import("ops.zig");
const dimensions = @import("config.zig");
const Graph = c.ds4_qwen4_gpu_graph;
const Model = c.ds4_model;
const Layer = c.ds4_layer_weights;
const Tensor = c.ds4_gpu_tensor;

pub fn linear(graph: *Graph, model: *const Model, layer: *const Layer, index: u32, tokens: u32) bool {
    const shape = &graph.config.shape;
    var paired = false;
    if (tokens == 1 and graph.mtp_R == null and c.ds4_gpu_qwen4_decode_fusions_enabled() != 0 and
        layer.lin_qkv.*.type == c.DS4_TENSOR_Q8_0 and layer.lin_gate.*.type == c.DS4_TENSOR_Q8_0)
    {
        paired = c.ds4_gpu_qwen4_q8_pair_tensor(graph.qkv, graph.z, model.map, model.size, layer.lin_qkv.*.abs_offset, layer.lin_gate.*.abs_offset, shape.n_embd, @intCast(layer.lin_qkv.*.dim[1]), @intCast(layer.lin_gate.*.dim[1]), graph.mixed, tokens) != 0;
    }
    if (!paired and !(ops.dense(graph.qkv, model, layer.lin_qkv, graph.mixed, tokens) and
        ops.dense(graph.z, model, layer.lin_gate, graph.mixed, tokens))) return false;
    const first_history = if (graph.snap_after_first) graph.snap_lin_hist[index] else null;
    const second_history = if (graph.snap_after_second) graph.snap2_lin_hist[index] else null;
    const prepared = if (ops.fused(graph, tokens))
        c.ds4_gpu_qwen4_gdn_front_tensor(graph.qkv, graph.layer_lin_hist[index], graph.mixed, graph.ga, graph.gb, model.map, model.size, layer.lin_conv.*.abs_offset, layer.lin_alpha.*.abs_offset, layer.lin_beta.*.abs_offset, layer.lin_a.*.abs_offset, layer.lin_dt_bias.*.abs_offset, layer.lin_alpha.*.type, tokens, shape.n_lin_k_head, shape.n_lin_v_head, shape.n_lin_head_dim, shape.n_lin_conv, shape.n_embd, first_history, 0, second_history, 1) != 0
    else
        ops.dense(graph.ga, model, layer.lin_alpha, graph.mixed, tokens) and
            ops.dense(graph.gb, model, layer.lin_beta, graph.mixed, tokens) and
            c.ds4_gpu_qwen4_conv_stream_tensor(graph.qkv, graph.layer_lin_hist[index], model.map, model.size, layer.lin_conv.*.abs_offset, tokens, @intCast(dimensions.convWidth(shape)), shape.n_lin_conv, true) != 0 and
            c.ds4_gpu_qwen4_gdn_prep_tensor(graph.qkv, graph.ga, graph.gb, model.map, model.size, layer.lin_a.*.abs_offset, layer.lin_dt_bias.*.abs_offset, tokens, shape.n_lin_k_head, shape.n_lin_v_head, shape.n_lin_head_dim) != 0;
    if (!prepared) return false;
    return c.ds4_gpu_qwen4_gdn_scan_tensor(graph.lin_o, graph.layer_lin_state[index], graph.qkv, graph.ga, graph.gb, tokens, shape.n_lin_k_head, shape.n_lin_v_head, shape.n_lin_head_dim, if (graph.snap_after_first) graph.snap_lin_state[index] else null, 0, if (graph.snap_after_second) graph.snap2_lin_state[index] else null, 1) != 0 and
        c.ds4_gpu_qwen4_gdn_out_tensor(graph.lin_o, graph.z, model.map, model.size, layer.lin_norm.*.abs_offset, tokens, shape.n_lin_v_head, shape.n_lin_head_dim, shape.rms_eps) != 0 and
        ops.dense(graph.blk, model, layer.lin_out, graph.lin_o, tokens);
}

pub fn core(graph: *Graph, index: u32, queries: ?*Tensor, gates: ?*Tensor, index_queries: ?*Tensor, output: ?*Tensor, blocks_after: u32, position: u32, tokens: u32) bool {
    const shape = &graph.config.shape;
    const query_width = @as(u64, shape.n_head) * shape.n_head_dim;
    const index_width = @as(u64, shape.n_indexer_head) * shape.n_indexer_head_dim;
    const scale: f32 = 1 / @sqrt(@as(f32, @floatFromInt(shape.n_head_dim)));
    const sparse_start = (graph.k_blocks + 1) * 4 - 1;
    const last = position + tokens - 1;
    const dense: u32 = if (last < sparse_start) tokens else sparse_start -| position;
    if (dense > 0 and c.ds4_gpu_qwen4_attn_decode_tensor(output, queries, gates, graph.layer_k_cache[index], graph.layer_v_cache[index], graph.sel_tokens, graph.n_sel, if (tokens <= 2) graph.attn_part else null, dense, shape.n_head, shape.n_head_kv, shape.n_head_dim, position, false, graph.sel_stride, scale) == 0) return false;
    if (dense >= tokens) return true;
    const sparse = tokens - dense;
    const first = position + dense;
    const q = ops.rowsView(queries, dense, sparse, query_width) orelse return false;
    defer c.ds4_gpu_tensor_free(q);
    const gate = ops.rowsView(gates, dense, sparse, query_width) orelse return false;
    defer c.ds4_gpu_tensor_free(gate);
    const out = ops.rowsView(output, dense, sparse, query_width) orelse return false;
    defer c.ds4_gpu_tensor_free(out);
    const iq = ops.rowsView(index_queries, dense, sparse, index_width) orelse return false;
    defer c.ds4_gpu_tensor_free(iq);
    const maxima = if (sparse <= 2) graph.tile_max else null;
    return c.ds4_gpu_qwen4_idx_score_tensor(graph.score, maxima, iq, graph.layer_block_key[index], sparse, blocks_after, shape.n_indexer_head, shape.n_indexer_head_dim, first, 4) != 0 and
        ops.selectBlocks(graph.sel_blocks, graph.score, maxima, blocks_after, sparse, graph.k_blocks) and
        c.ds4_gpu_qwen4_idx_expand_tensor(graph.sel_tokens, graph.n_sel, graph.sel_blocks, sparse, graph.k_blocks, 4, first, graph.sel_stride) != 0 and
        c.ds4_gpu_qwen4_attn_decode_tensor(out, q, gate, graph.layer_k_cache[index], graph.layer_v_cache[index], graph.sel_tokens, graph.n_sel, if (tokens <= 2) graph.attn_part else null, sparse, shape.n_head, shape.n_head_kv, shape.n_head_dim, first, true, graph.sel_stride, scale) != 0;
}

pub fn tail(graph: *Graph, model: *const Model, layer: *const Layer, index: u32, position: u32, tokens: u32) bool {
    const shape = &graph.config.shape;
    if (c.ds4_gpu_qwen4_attn_prep_tensor(graph.q, graph.gate, graph.layer_k_cache[index], graph.layer_v_cache[index], graph.iqn, graph.layer_ik_cache[index], graph.qg, graph.kp, graph.vp, graph.iq, graph.ik, graph.pos3, model.map, model.size, layer.attn_q_norm.*.abs_offset, layer.attn_k_norm.*.abs_offset, layer.indexer_q_norm.*.abs_offset, tokens, shape.n_head, shape.n_head_kv, shape.n_head_dim, shape.n_rot, shape.n_indexer_head, shape.n_indexer_head_dim, position, graph.ctx_cap, shape.rope_freq_base, shape.rms_eps) == 0) return false;
    const first_block = position / 4;
    const blocks_after = (position + tokens) / 4;
    if (blocks_after > first_block and c.ds4_gpu_qwen4_idx_block_key_tensor(graph.layer_block_key[index], graph.layer_ik_cache[index], graph.pos3, model.map, model.size, layer.indexer_k_norm.*.abs_offset, first_block, blocks_after - first_block, 4, shape.n_indexer_head_dim, shape.n_rot, shape.rope_freq_base, shape.rms_eps) == 0) return false;
    if (tokens != 3 or !graph.verify_rows_exact)
        return core(graph, index, graph.q, graph.gate, graph.iqn, graph.attn_o, blocks_after, position, tokens);
    const query_width = @as(u64, shape.n_head) * shape.n_head_dim;
    const index_width = @as(u64, shape.n_indexer_head) * shape.n_indexer_head_dim;
    var first: u32 = 0;
    while (first < tokens) : (first += 2) {
        const count = @min(2, tokens - first);
        const q = if (first == 0) graph.q else ops.rowsView(graph.q, first, count, query_width);
        const gate = if (first == 0) graph.gate else ops.rowsView(graph.gate, first, count, query_width);
        const out = if (first == 0) graph.attn_o else ops.rowsView(graph.attn_o, first, count, query_width);
        const iq = if (first == 0) graph.iqn else ops.rowsView(graph.iqn, first, count, index_width);
        defer if (first != 0) {
            c.ds4_gpu_tensor_free(iq);
            c.ds4_gpu_tensor_free(out);
            c.ds4_gpu_tensor_free(gate);
            c.ds4_gpu_tensor_free(q);
        };
        // A true two-row pass cannot select the future block universe of the
        // entire three-row verify, even though its pooled keys already exist.
        const sub_blocks = (position + first + count) / 4;
        if (q == null or gate == null or out == null or iq == null or !core(graph, index, q, gate, iq, out, sub_blocks, position + first, count)) return false;
    }
    return true;
}

pub fn run(graph: *Graph, model: *const Model, layer: *const Layer, index: u32, position: u32, tokens: u32) bool {
    if (!ops.dense(graph.qg, model, layer.attn_q, graph.mixed, tokens)) return false;
    if (ops.fused(graph, tokens)) {
        var outputs = [_]?*Tensor{ graph.kp, graph.vp, graph.iq, graph.ik };
        const projections = .{ layer.attn_k, layer.attn_v, layer.indexer_q_proj, layer.indexer_k_proj };
        var offsets: [4]u64 = undefined;
        var kinds: [4]u32 = undefined;
        var rows: [4]u32 = undefined;
        inline for (projections, 0..) |weight, part| {
            offsets[part] = weight.*.abs_offset;
            kinds[part] = weight.*.type;
            rows[part] = @intCast(weight.*.dim[1]);
        }
        if (c.ds4_gpu_qwen4_multi_gemv_tensor(graph.mixed, tokens, graph.config.shape.n_embd, 4, &outputs, model.map, model.size, &offsets, &kinds, &rows) == 0) return false;
    } else if (!(ops.dense(graph.kp, model, layer.attn_k, graph.mixed, tokens) and
        ops.dense(graph.vp, model, layer.attn_v, graph.mixed, tokens) and
        ops.dense(graph.iq, model, layer.indexer_q_proj, graph.mixed, tokens) and
        ops.dense(graph.ik, model, layer.indexer_k_proj, graph.mixed, tokens))) return false;
    return tail(graph, model, layer, index, position, tokens) and ops.dense(graph.blk, model, layer.attn_output, graph.attn_o, tokens);
}
