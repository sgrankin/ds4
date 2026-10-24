//! GLM-5.3's absorbed MLA decode. The shared latent KV cache serves both the
//! dense prefix and pooled sparse attention; it never expands into per-head
//! keys/values on the normal Metal path.
const c = @import("gguf_c");
const Config = @import("config.zig");
const Ops = @import("ops.zig");
const Graph = c.ds4_glm_gpu_graph;
const Model = c.ds4_model;
const Layer = c.ds4_layer_weights;
const Selection = c.ds4_glm_decode_selection;

/// Timing-only skips retain the C diagnostic's bit meanings. Their outputs
/// deliberately contain stale data, so they are not inference approximations.
pub const Ablate = struct {
    pub const output: u32 = 1 << 0;
    pub const core: u32 = 1 << 1;
    pub const query: u32 = 1 << 2;
    pub const indexer: u32 = 1 << 3;
    pub const lowrank: u32 = 1 << 6;
};

fn projectAndStore(graph: *Graph, model: *const Model, weights: *const Layer, layer: u32, position: u32, indexed: bool, ablate: u32) bool {
    const shape = &graph.config.shape;
    const kv_width: u32 = @intCast(weights.attn_kv_a_mqa.*.dim[1]);
    if (ablate & Ablate.query == 0 and
        !Ops.matmul(graph.q_rank, model, weights.attn_q_a, shape.n_embd, shape.n_lora_q, graph.attn_norm, 1)) return false;
    if (!Ops.matmul(graph.kv_raw, model, weights.attn_kv_a_mqa, shape.n_embd, kv_width, graph.attn_norm, 1)) return false;

    // Preserve the existing fused norm/store dispatch on absorbed attention.
    // A full-KV graph needs the normalized latent as a separate tensor instead.
    if (indexed and graph.compact_cache_cap != 0) {
        if (c.ds4_gpu_glm_qkv_norm_store_compact_kv_tensor(
            graph.q_rank_norm,
            graph.q_rank,
            model.map,
            model.size,
            weights.attn_q_a_norm.*.abs_offset,
            shape.n_lora_q,
            graph.layer_kv_lora_cache[layer],
            graph.layer_k_rope_cache[layer],
            graph.kv_raw,
            weights.attn_kv_a_norm.*.abs_offset,
            position,
            1,
            graph.compact_cache_cap,
            kv_width,
            shape.n_kv_lora,
            shape.n_rot,
            true,
            shape.rms_eps,
        ) == 0) return false;
    } else {
        if (c.ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
            graph.q_rank_norm,
            graph.q_rank,
            model.map,
            model.size,
            weights.attn_q_a_norm.*.abs_offset,
            shape.n_lora_q,
            graph.kv_norm,
            graph.kv_raw,
            weights.attn_kv_a_norm.*.abs_offset,
            shape.n_kv_lora,
            1,
            shape.rms_eps,
        ) == 0) return false;
    }
    if (ablate & Ablate.query == 0 and
        !Ops.matmul(graph.q, model, weights.attn_q_b, shape.n_lora_q, @intCast(graph.q_dim), graph.q_rank_norm, 1)) return false;
    c.ds4_native_graph_debug_tensor("glm_decode_q", graph.q, @as(u64, shape.n_head) * shape.n_key_mla, layer, position);

    if (graph.compact_cache_cap != 0 and Config.hasIndexer(&graph.config, layer) and ablate & Ablate.indexer == 0) {
        if (!Ops.matmul(graph.indexer_k, model, weights.indexer_attn_k, shape.n_embd, shape.n_indexer_head_dim, graph.attn_norm, 1) or
            !Ops.matmul(graph.indexer_gate, model, weights.indexer_compressor_gate, shape.n_embd, shape.n_indexer_head_dim, graph.attn_norm, 1)) return false;
        if (c.ds4_gpu_glm53_indexer_pool_update_tensor(
            graph.layer_indexer_key_cache[layer],
            graph.layer_indexer_tail_k[layer],
            graph.layer_indexer_tail_gate[layer],
            graph.indexer_k,
            graph.indexer_gate,
            model.map,
            model.size,
            weights.indexer_k_norm.*.abs_offset,
            weights.indexer_k_norm_b.*.abs_offset,
            weights.indexer_compressor_ape.*.abs_offset,
            position,
            1,
            graph.compact_cache_cap,
            shape.n_indexer_head_dim,
            graph.config.index_pool_size,
            1.0e-6,
            true,
        ) == 0) return false;
    }
    if (!indexed and graph.compact_cache_cap != 0) {
        if (c.ds4_gpu_glm_store_compact_kv_tensor(
            graph.layer_kv_lora_cache[layer],
            graph.layer_k_rope_cache[layer],
            graph.kv_norm,
            graph.kv_raw,
            position,
            1,
            graph.compact_cache_cap,
            kv_width,
            shape.n_kv_lora,
            shape.n_rot,
            true,
        ) == 0) return false;
    }
    return true;
}

fn select(graph: *Graph, model: *const Model, weights: *const Layer, layer: u32, position: u32, selection: *Selection, ablate: u32) bool {
    if (!Config.hasIndexer(&graph.config, layer)) return selection.selected != null and selection.count != 0;
    const shape = &graph.config.shape;
    const visible = position + 1;
    const top_k = shape.n_indexer_top_k;
    const width = top_k + graph.config.index_pool_size - 1;
    if (visible <= width) {
        // The same contiguous IDs remain valid across DSA layers. Later pools
        // are layer-specific, so only this dense fill can be reused.
        if ((selection.selected != graph.indexer_selected or selection.count != visible) and
            c.ds4_gpu_glm_fill_selected_range_tensor(graph.indexer_selected, visible) == 0) return false;
        selection.count = visible;
        selection.guaranteed_prefix = 0;
    } else if (ablate & Ablate.indexer != 0) {
        if (c.ds4_gpu_glm_fill_selected_range_tensor(graph.indexer_selected, top_k) == 0) return false;
        selection.count = top_k;
        selection.guaranteed_prefix = 0;
    } else {
        if (!Ops.matmul(graph.indexer_q, model, weights.indexer_attn_q_b, shape.n_lora_q, shape.n_indexer_head * shape.n_indexer_head_dim, graph.q_rank_norm, 1) or
            !Ops.matmul(graph.indexer_weights, model, weights.indexer_proj, shape.n_embd, shape.n_indexer_head, graph.attn_norm, 1)) return false;
        const score_rows = visible / graph.config.index_pool_size;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(shape.n_indexer_head_dim * shape.n_indexer_head)));
        if (c.ds4_gpu_glm_indexer_score_one_tensor(
            graph.indexer_scores,
            graph.indexer_q,
            graph.indexer_weights,
            graph.layer_indexer_key_cache[layer],
            score_rows,
            shape.n_indexer_head,
            shape.n_indexer_head_dim,
            scale,
            true,
        ) == 0) return false;
        const selected_pools = top_k / graph.config.index_pool_size;
        if (c.ds4_gpu_indexer_topk_tensor(graph.indexer_pool_selected, graph.indexer_scores, score_rows, 1, selected_pools) == 0 or
            c.ds4_gpu_glm53_expand_pool_selection_tensor(graph.indexer_selected, graph.indexer_pool_selected, 1, position, selected_pools, top_k, graph.config.index_pool_size, width) == 0) return false;
        selection.count = width;
        // The incomplete pool is always included. Slots beyond top_k may be
        // 0xffffffff padding, so do not advertise every selected row as valid.
        selection.guaranteed_prefix = top_k;
    }
    selection.selected = graph.indexer_selected;
    return true;
}

fn indexedAttention(graph: *Graph, model: *const Model, weights: *const Layer, layer: u32, position: u32, selection: *Selection, ablate: u32) bool {
    const shape = &graph.config.shape;
    if (!select(graph, model, weights, layer, position, selection, ablate)) return false;
    if (ablate & (Ablate.core | Ablate.lowrank) == 0) {
        if (c.ds4_gpu_glm_qk_lowrank_typed_tensor(
            graph.qk_low,
            graph.q,
            model.map,
            model.size,
            weights.attn_k_b.*.abs_offset,
            weights.attn_k_b.*.type,
            shape.n_head,
            shape.n_kv_lora,
            @intCast(graph.q_nope),
            shape.n_key_mla,
        ) == 0) return false;
        c.ds4_native_graph_debug_tensor("glm_decode_qk_low", graph.qk_low, @as(u64, shape.n_head) * shape.n_kv_lora, layer, position);
    }
    if (ablate & Ablate.core != 0)
        return c.ds4_gpu_tensor_fill_f32(graph.heads, 0, graph.heads_dim) != 0;

    // GLM-5.3 has no rotary tail. The old split-group8 specialization requires
    // 64 rotary dimensions, so the typed absorbed kernel is its Metal path.
    return c.ds4_gpu_glm_attention_indexed_decode_typed_tensor(
        graph.heads,
        graph.q,
        graph.qk_low,
        graph.layer_kv_lora_cache[layer],
        graph.layer_k_rope_cache[layer],
        model.map,
        model.size,
        weights.attn_v_b.*.abs_offset,
        weights.attn_v_b.*.type,
        selection.selected,
        selection.count,
        graph.compact_cache_cap,
        true,
        shape.n_head,
        shape.n_kv_lora,
        @intCast(graph.q_nope),
        shape.n_rot,
        shape.n_value_mla,
        0,
        shape.rope_freq_base,
        shape.rope_scale_factor,
        0,
        1,
        shape.rope_yarn_beta_fast,
        shape.rope_yarn_beta_slow,
    ) != 0;
}

fn fullAttention(graph: *Graph, model: *const Model, weights: *const Layer, layer: u32, position: u32) bool {
    const shape = &graph.config.shape;
    if (c.ds4_gpu_glm_k_b_project_typed_tensor(graph.k_nope, graph.kv_norm, model.map, model.size, weights.attn_k_b.*.abs_offset, weights.attn_k_b.*.type, 1, shape.n_kv_lora, @intCast(graph.q_nope), shape.n_head) == 0 or
        !Ops.matmul(graph.value, model, weights.attn_v_b, shape.n_kv_lora, @intCast(graph.heads_dim), graph.kv_norm, 1)) return false;
    if (c.ds4_gpu_glm_build_kv_cache_tensor(
        graph.layer_key_cache[layer],
        graph.layer_value_cache[layer],
        graph.kv_raw,
        graph.k_nope,
        graph.value,
        position,
        1,
        graph.ctx_cap,
        shape.n_head,
        @intCast(weights.attn_kv_a_mqa.*.dim[1]),
        shape.n_kv_lora,
        @intCast(graph.q_nope),
        shape.n_rot,
        shape.n_value_mla,
        0,
        shape.rope_freq_base,
        shape.rope_scale_factor,
        0,
        1,
        shape.rope_yarn_beta_fast,
        shape.rope_yarn_beta_slow,
        true,
    ) == 0) return false;
    return c.ds4_gpu_glm_attention_full_tensor(graph.heads, graph.q, graph.layer_key_cache[layer], graph.layer_value_cache[layer], position, 1, position + 1, graph.ctx_cap, shape.n_head, shape.n_key_mla, shape.n_value_mla, true) != 0;
}

/// Called inside the existing layer command batch, after mHC preparation.
/// Mapping/admission, cancellation and GPU submission remain with its caller.
pub fn decode(graph: *Graph, model: *const Model, weights: *const Layer, layer: u32, position: u32, indexed: bool, selection: *Selection, ablate: u32) bool {
    if (!graph.glm53 or graph.config.shape.n_rot != 0 or graph.tp_world > 1 or layer >= c.DS4_MAX_LAYER) return false;
    if (!projectAndStore(graph, model, weights, layer, position, indexed, ablate)) return false;
    const ok = if (indexed)
        indexedAttention(graph, model, weights, layer, position, selection, ablate)
    else
        fullAttention(graph, model, weights, layer, position);
    if (!ok) return false;
    c.ds4_native_graph_debug_tensor("glm_decode_heads", graph.heads, graph.heads_dim, layer, position);
    return ablate & Ablate.output != 0 or Ops.matmul(graph.attn_out, model, weights.attn_output, @intCast(graph.heads_dim), graph.config.shape.n_embd, graph.heads, 1);
}
