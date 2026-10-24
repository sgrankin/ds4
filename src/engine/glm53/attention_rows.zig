//! Layer-major absorbed MLA prefill, including chunks that cross from causal
//! dense attention to pooled sparse selection. Kernels and flush boundaries
//! match the existing Metal graph; transient views retain their allocations.
const c = @import("gguf_c");
const Ops = @import("ops.zig");
const Ablate = @import("attention.zig").Ablate;
const Graph = c.ds4_glm_gpu_graph;
const Model = c.ds4_model;
const Layer = c.ds4_layer_weights;
const Tensor = c.ds4_gpu_tensor;

fn view(tensor: ?*const Tensor, start: u32, rows: u32, width: u64) ?*Tensor {
    return c.ds4_gpu_tensor_view(tensor, @as(u64, start) * width * @sizeOf(f32), @as(u64, rows) * width * @sizeOf(f32));
}

fn projections(graph: *Graph, model: *const Model, weights: *const Layer, layer: u32, position: u32, rows: u32, ablate: u32) bool {
    const shape = &graph.config.shape;
    if (ablate & Ablate.query == 0 and !Ops.matmul(graph.batch_q_rank, model, weights.attn_q_a, shape.n_embd, shape.n_lora_q, graph.batch_attn_norm, rows)) return false;
    if (c.ds4_gpu_rms_norm_weight_rows_tensor(graph.batch_q_rank_norm, graph.batch_q_rank, model.map, model.size, weights.attn_q_a_norm.*.abs_offset, shape.n_lora_q, rows, shape.rms_eps) == 0 or
        !Ops.matmul(graph.batch_q, model, weights.attn_q_b, shape.n_lora_q, @intCast(graph.q_dim), graph.batch_q_rank_norm, rows)) return false;
    if (!Ops.matmul(graph.batch_indexer_k, model, weights.indexer_attn_k, shape.n_embd, shape.n_indexer_head_dim, graph.batch_attn_norm, rows) or
        !Ops.matmul(graph.batch_indexer_gate, model, weights.indexer_compressor_gate, shape.n_embd, shape.n_indexer_head_dim, graph.batch_attn_norm, rows)) return false;
    if (c.ds4_gpu_glm53_indexer_pool_update_tensor(
        graph.layer_indexer_key_cache[layer],
        graph.layer_indexer_tail_k[layer],
        graph.layer_indexer_tail_gate[layer],
        graph.batch_indexer_k,
        graph.batch_indexer_gate,
        model.map,
        model.size,
        weights.indexer_k_norm.*.abs_offset,
        weights.indexer_k_norm_b.*.abs_offset,
        weights.indexer_compressor_ape.*.abs_offset,
        position,
        rows,
        graph.compact_cache_cap,
        shape.n_indexer_head_dim,
        graph.config.index_pool_size,
        1.0e-6,
        true,
    ) == 0) return false;
    const kv_width: u32 = @intCast(weights.attn_kv_a_mqa.*.dim[1]);
    return Ops.matmul(graph.batch_kv_raw, model, weights.attn_kv_a_mqa, shape.n_embd, kv_width, graph.batch_attn_norm, rows) and
        c.ds4_gpu_glm_kv_lora_rms_norm_tensor(graph.batch_kv_norm, graph.batch_kv_raw, model.map, model.size, weights.attn_kv_a_norm.*.abs_offset, rows, kv_width, shape.n_kv_lora, shape.rms_eps) != 0 and
        c.ds4_gpu_glm_store_compact_kv_tensor(graph.layer_kv_lora_cache[layer], graph.layer_k_rope_cache[layer], graph.batch_kv_norm, graph.batch_kv_raw, position, rows, graph.compact_cache_cap, kv_width, shape.n_kv_lora, shape.n_rot, true) != 0;
}

fn select(graph: *Graph, model: *const Model, weights: *const Layer, layer: u32, position: u32, rows: u32, selected_width: u32) bool {
    const shape = &graph.config.shape;
    const query_width = shape.n_indexer_head * shape.n_indexer_head_dim;
    if (!Ops.matmul(graph.batch_indexer_q, model, weights.indexer_attn_q_b, shape.n_lora_q, query_width, graph.batch_q_rank_norm, rows) or
        !Ops.matmul(graph.batch_indexer_weights, model, weights.indexer_proj, shape.n_embd, shape.n_indexer_head, graph.batch_attn_norm, rows)) return false;
    const score_rows = (position + rows) / graph.config.index_pool_size;
    const selected_pools = shape.n_indexer_top_k / graph.config.index_pool_size;
    const score_cap = if (graph.indexed_prefill_score_cap != 0) graph.indexed_prefill_score_cap else graph.indexed_prefill_cap;
    const scale = 1 / @sqrt(@as(f32, @floatFromInt(query_width)));
    var start: u32 = 0;
    while (start < rows) {
        const count = @min(rows - start, score_cap);
        if (count == 0) return false;
        const query = view(graph.batch_indexer_q, start, count, query_width);
        defer c.ds4_gpu_tensor_free(query);
        const routing = view(graph.batch_indexer_weights, start, count, shape.n_indexer_head);
        defer c.ds4_gpu_tensor_free(routing);
        const pools = view(graph.batch_indexer_pool_selected, start, count, selected_pools);
        defer c.ds4_gpu_tensor_free(pools);
        if (query == null or routing == null or pools == null) return false;
        if (c.ds4_gpu_glm53_indexer_scores_batch_tensor(
            graph.batch_indexer_scores,
            query,
            routing,
            graph.layer_indexer_key_cache[layer],
            score_rows,
            count,
            position + start,
            graph.config.index_pool_size,
            shape.n_indexer_head,
            shape.n_indexer_head_dim,
            scale,
            true,
        ) == 0 or c.ds4_gpu_indexer_topk_tensor(pools, graph.batch_indexer_scores, score_rows, count, selected_pools) == 0) return false;
        start += count;
    }
    return c.ds4_gpu_glm53_expand_pool_selection_tensor(graph.batch_indexer_selected, graph.batch_indexer_pool_selected, rows, position, selected_pools, shape.n_indexer_top_k, graph.config.index_pool_size, selected_width) != 0;
}

fn attend(graph: *Graph, model: *const Model, weights: *const Layer, layer: u32, position: u32, rows: u32, selected_width: u32) bool {
    const shape = &graph.config.shape;
    const dense_limit = shape.n_indexer_top_k + graph.config.index_pool_size - 1;
    const end = position + rows;
    const latent_width = @as(u64, shape.n_head) * shape.n_kv_lora;
    var start: u32 = 0;
    while (start < rows) {
        const slice_position = position + start;
        const causal = slice_position < dense_limit;
        var count = @min(rows - start, 2048);
        if (causal) count = @min(count, dense_limit - slice_position);
        const selected_count = if (causal) @min(end, dense_limit) else selected_width;
        const query = view(graph.batch_q, start, count, graph.q_dim);
        defer c.ds4_gpu_tensor_free(query);
        const lowrank = view(graph.batch_qk_low, start, count, latent_width);
        defer c.ds4_gpu_tensor_free(lowrank);
        const heads = view(graph.batch_heads, start, count, graph.heads_dim);
        defer c.ds4_gpu_tensor_free(heads);
        const latent = view(graph.batch_attn_lora, start, count, latent_width);
        defer c.ds4_gpu_tensor_free(latent);
        const selected = if (causal) null else view(graph.batch_indexer_selected, start, count, selected_width);
        defer c.ds4_gpu_tensor_free(selected);
        if (query == null or lowrank == null or heads == null or latent == null or (!causal and selected == null)) return false;

        const attention_ok = if (causal and graph.config.flash_prefill and count >= 24)
            c.ds4_gpu_glm_attention_dense_compact_lora_causal_tensor(
                latent,
                lowrank,
                graph.layer_kv_lora_cache[layer],
                slice_position,
                count,
                selected_count,
                graph.compact_cache_cap,
                true,
                shape.n_head,
                shape.n_kv_lora,
                @intCast(graph.q_nope),
            ) != 0
        else if (causal)
            c.ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
                latent,
                query,
                lowrank,
                graph.layer_kv_lora_cache[layer],
                graph.layer_k_rope_cache[layer],
                count,
                slice_position,
                selected_count,
                graph.compact_cache_cap,
                true,
                shape.n_head,
                shape.n_kv_lora,
                @intCast(graph.q_nope),
                shape.n_rot,
                0,
                shape.rope_freq_base,
                shape.rope_scale_factor,
                0,
                1,
                shape.rope_yarn_beta_fast,
                shape.rope_yarn_beta_slow,
            ) != 0
        else
            // Pooled expansion pads invalid IDs in its incomplete pool. Keep
            // the checked kernel; the all-valid variant would read those IDs.
            c.ds4_gpu_glm_attention_indexed_batch_lora_tensor(
                latent,
                query,
                lowrank,
                graph.layer_kv_lora_cache[layer],
                graph.layer_k_rope_cache[layer],
                selected,
                count,
                selected_width,
                graph.compact_cache_cap,
                true,
                shape.n_head,
                shape.n_kv_lora,
                @intCast(graph.q_nope),
                shape.n_rot,
                0,
                shape.rope_freq_base,
                shape.rope_scale_factor,
                0,
                1,
                shape.rope_yarn_beta_fast,
                shape.rope_yarn_beta_slow,
            ) != 0;
        if (!attention_ok or c.ds4_gpu_glm_value_project_typed_batch_heads_tensor(
            heads,
            latent,
            model.map,
            model.size,
            weights.attn_v_b.*.abs_offset,
            weights.attn_v_b.*.type,
            count,
            shape.n_head,
            shape.n_kv_lora,
            shape.n_value_mla,
        ) == 0) return false;
        start += count;
    }
    return true;
}

pub fn encode(graph: *Graph, model: *const Model, weights: *const Layer, layer: u32, position: u32, rows: u32, ablate_mask: u32) bool {
    if (!graph.glm53 or graph.tp_world > 1 or graph.config.shape.n_rot != 0 or rows == 0 or graph.batch_attn_lora == null) return false;
    const shape = &graph.config.shape;
    const ablate = if (rows <= 8) ablate_mask else 0;
    const dense_limit = shape.n_indexer_top_k + graph.config.index_pool_size - 1;
    const end = position + rows;
    const selected_width = if (end <= dense_limit) end else dense_limit;
    if (!projections(graph, model, weights, layer, position, rows, ablate)) return false;
    if (end > dense_limit and !select(graph, model, weights, layer, position, rows, selected_width)) return false;
    c.ds4_native_graph_debug_tensor("glm_indexed_q", graph.batch_q, @as(u64, rows) * graph.q_dim, layer, position);
    if (ablate & Ablate.core == 0 and c.ds4_gpu_glm_qk_lowrank_typed_batch_tensor(
        graph.batch_qk_low,
        graph.batch_q,
        model.map,
        model.size,
        weights.attn_k_b.*.abs_offset,
        weights.attn_k_b.*.type,
        rows,
        shape.n_head,
        shape.n_kv_lora,
        @intCast(graph.q_nope),
        shape.n_key_mla,
    ) == 0) return false;
    c.ds4_native_graph_debug_tensor("glm_indexed_qk_low", graph.batch_qk_low, @as(u64, rows) * shape.n_head * shape.n_kv_lora, layer, position);
    if (c.ds4_gpu_flush_encoder() == 0) return false;
    if (ablate & Ablate.core == 0 and !attend(graph, model, weights, layer, position, rows, selected_width)) return false;
    if (c.ds4_gpu_flush_encoder() == 0) return false;
    c.ds4_native_graph_debug_tensor("glm_indexed_heads", graph.batch_heads, @as(u64, rows) * graph.heads_dim, layer, position);
    return ablate & Ablate.output != 0 or Ops.matmul(graph.batch_attn_out, model, weights.attn_output, @intCast(graph.heads_dim), shape.n_embd, graph.batch_heads, rows);
}
