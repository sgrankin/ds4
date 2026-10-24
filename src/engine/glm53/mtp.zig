//! The checkpoint's separate nextn layer drafts one token from the current
//! target hidden state and its next input token. Its compact window starts at
//! min_position, so uninitialized prompt slots are never selected.
const c = @import("gguf_c");
const std = @import("std");
const Ops = @import("ops.zig");
const Moe = @import("moe.zig");
const Speculation = @import("speculation.zig");
const Graph = c.ds4_glm_gpu_graph;
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const allocator = std.heap.c_allocator;

fn begin() bool {
    return c.ds4_gpu_commands_active() != 0 or c.ds4_gpu_begin_commands() != 0;
}

fn end() bool {
    return c.ds4_gpu_commands_active() == 0 or c.ds4_gpu_end_commands() != 0;
}

fn encode(graph: *Graph, model: *const Model, weights: *const Weights, next_token: i32, position: u32, capacity: u32, selected_count: u32, stage: *[*:0]const u8) bool {
    const shape = &graph.config.shape;
    const layer = shape.n_layer - shape.n_nextn_predict;
    const draft = &weights.layer[layer];
    const width = @as(u64, shape.n_embd) * @sizeOf(f32);
    const enorm = c.ds4_gpu_tensor_view(graph.mtp_concat, 0, width);
    defer c.ds4_gpu_tensor_free(enorm);
    const hnorm = c.ds4_gpu_tensor_view(graph.mtp_concat, width, width);
    defer c.ds4_gpu_tensor_free(hnorm);
    if (enorm == null or hnorm == null) return false;
    // Ordinary bounded decode finishes with only the language head mapped.
    // The draft is a separate layer, so submit each stage before replacing
    // its borrowed model views. A complete static decode map needs no changes.
    const bounded_map = graph.ssd_streaming and !graph.streaming_static_decode_map_current;
    if (bounded_map) {
        stage.* = "token embedding map";
        if (!end() or !c.ds4_native_glm_map_token(graph, model, weights)) return false;
    }
    if (!begin()) return false;

    stage.* = "next-token embedding";
    if (next_token < 0 or next_token >= shape.n_vocab) return false;
    const embedding = weights.token_embd;
    if (embedding.*.type == c.DS4_TENSOR_BF16) {
        if (c.ds4_gpu_tensor_write(graph.prefill_tokens, 0, &next_token, @sizeOf(i32)) == 0 or
            c.ds4_gpu_glm53_embedding_bf16(graph.next, model.map, model.size, embedding.*.abs_offset, graph.prefill_tokens, 1, shape.n_embd, shape.n_vocab) == 0) return false;
    } else if (c.ds4_gpu_embed_token_quant_tensor(graph.next, model.map, model.size, embedding.*.abs_offset, embedding.*.type, shape.n_vocab, @intCast(next_token), shape.n_embd) == 0) return false;
    if (bounded_map) {
        stage.* = "draft layer map";
        if (!end() or !c.ds4_native_glm_map_decode_layer(graph, model, weights, layer) or !begin()) return false;
    }
    stage.* = "input norms and projection";
    if (c.ds4_gpu_rms_norm_weight_tensor(enorm, graph.next, model.map, model.size, draft.nextn_enorm.*.abs_offset, shape.n_embd, shape.rms_eps) == 0 or
        c.ds4_gpu_hc_weighted_sum_tensor(graph.hc_output, graph.hc_cur, graph.hc_mean_weights, shape.n_embd, shape.n_hc) == 0 or
        c.ds4_gpu_rms_norm_weight_tensor(hnorm, graph.hc_output, model.map, model.size, draft.nextn_hnorm.*.abs_offset, shape.n_embd, shape.rms_eps) == 0 or
        !Ops.matmul(graph.cur, model, draft.nextn_eh_proj, 2 * shape.n_embd, shape.n_embd, graph.mtp_concat, 1)) return false;

    stage.* = "attention projections";
    const kv_width: u32 = @intCast(draft.attn_kv_a_mqa.*.dim[1]);
    if (c.ds4_gpu_rms_norm_weight_tensor(graph.attn_norm, graph.cur, model.map, model.size, draft.attn_norm.*.abs_offset, shape.n_embd, shape.rms_eps) == 0 or
        !Ops.matmul(graph.q_rank, model, draft.attn_q_a, shape.n_embd, shape.n_lora_q, graph.attn_norm, 1) or
        c.ds4_gpu_rms_norm_weight_tensor(graph.q_rank_norm, graph.q_rank, model.map, model.size, draft.attn_q_a_norm.*.abs_offset, shape.n_lora_q, shape.rms_eps) == 0 or
        !Ops.matmul(graph.q, model, draft.attn_q_b, shape.n_lora_q, @intCast(graph.q_dim), graph.q_rank_norm, 1) or
        !Ops.matmul(graph.kv_raw, model, draft.attn_kv_a_mqa, shape.n_embd, kv_width, graph.attn_norm, 1) or
        c.ds4_gpu_glm_kv_lora_rms_norm_tensor(graph.kv_norm, graph.kv_raw, model.map, model.size, draft.attn_kv_a_norm.*.abs_offset, 1, kv_width, shape.n_kv_lora, shape.rms_eps) == 0) return false;
    stage.* = "draft cache and attention";
    if (c.ds4_gpu_glm_store_compact_kv_tensor(graph.mtp_kv_lora_cache, graph.mtp_k_rope_cache, graph.kv_norm, graph.kv_raw, position, 1, capacity, kv_width, shape.n_kv_lora, shape.n_rot, true) == 0 or
        c.ds4_gpu_glm_qk_lowrank_typed_tensor(graph.qk_low, graph.q, model.map, model.size, draft.attn_k_b.*.abs_offset, draft.attn_k_b.*.type, shape.n_head, shape.n_kv_lora, @intCast(graph.q_nope), shape.n_key_mla) == 0 or
        c.ds4_gpu_glm_attention_indexed_decode_typed_tensor(
            graph.heads,
            graph.q,
            graph.qk_low,
            graph.mtp_kv_lora_cache,
            graph.mtp_k_rope_cache,
            model.map,
            model.size,
            draft.attn_v_b.*.abs_offset,
            draft.attn_v_b.*.type,
            graph.mtp_selected,
            selected_count,
            capacity,
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
        ) == 0 or !Ops.matmul(graph.attn_out, model, draft.attn_output, @intCast(graph.heads_dim), shape.n_embd, graph.heads, 1)) return false;

    stage.* = "FFN norm and router";
    if (c.ds4_gpu_add_rms_norm_weight_tensor(graph.ffn_norm, graph.after_attn, graph.cur, graph.attn_out, model.map, model.size, draft.ffn_norm.*.abs_offset, shape.n_embd, shape.rms_eps) == 0 or
        c.ds4_gpu_matmul_f32_tensor(graph.router_logits, model.map, model.size, draft.ffn_gate_inp.*.abs_offset, shape.n_embd, shape.n_expert, graph.ffn_norm, 1) == 0 or
        c.ds4_gpu_glm_router_select_tensor(graph.router_selected, graph.router_weights, graph.router_probs, model.map, model.size, draft.ffn_exp_probs_b.*.abs_offset, graph.router_logits, shape.n_expert, shape.n_expert_used, shape.expert_weight_scale) == 0) return false;
    stage.* = "routed and shared experts";
    const cached = c.ds4_native_glm_mtp_stream_cache(graph, draft, layer);
    if (!Moe.routedOne(graph, model, draft, layer, graph.ffn_out, graph.ffn_mid, graph.router_selected, graph.router_weights, graph.ffn_norm, graph.ssd_streaming and !cached) or
        !Moe.shared(graph, model, draft) or
        c.ds4_gpu_add3_tensor(graph.next, graph.after_attn, graph.ffn_out, graph.ffn_sum, shape.n_embd) == 0) return false;

    // The nextn head norm differs from the ordinary language-head norm.
    stage.* = "draft head";
    if (c.ds4_gpu_rms_norm_weight_tensor(graph.output_norm, graph.next, model.map, model.size, draft.nextn_shared_head_norm.*.abs_offset, shape.n_embd, shape.rms_eps) == 0) return false;
    if (bounded_map) {
        stage.* = "draft output map";
        if (!end() or !c.ds4_native_glm_map_output(graph, model, weights) or !begin()) return false;
    }
    return Ops.matmul(graph.logits, model, weights.output, shape.n_embd, shape.n_vocab, graph.output_norm, 1) and
        end() and
        c.ds4_gpu_tensor_read(graph.logits, 0, graph.mtp_logits_host, @as(u64, shape.n_vocab) * @sizeOf(f32)) != 0;
}

pub fn step(graph: *Graph, model: *const Model, weights: *const Weights, next_token: i32, position: u32, min_position: u32, output: *i32) bool {
    const shape = &graph.config.shape;
    if (!graph.glm53 or graph.placement != null or graph.tp_world > 1 or shape.n_nextn_predict == 0 or shape.n_rot != 0) return false;
    const capacity = if (graph.compact_cache_cap != 0) graph.compact_cache_cap else graph.ctx_size;
    if (position >= capacity or min_position > position) return false;
    const draft = &weights.layer[shape.n_layer - shape.n_nextn_predict];
    if (draft.nextn_eh_proj == null or draft.nextn_enorm == null or draft.nextn_hnorm == null or draft.nextn_shared_head_norm == null or draft.ffn_gate_exps == null) return false;
    if (!Speculation.ensure(graph)) return false;
    const count = position - min_position + 1;
    const selected = allocator.alloc(i32, count) catch return false;
    defer allocator.free(selected);
    for (selected, 0..) |*id, index| id.* = @intCast(min_position + index);
    if (c.ds4_gpu_tensor_write(graph.mtp_selected, 0, selected.ptr, @as(u64, count) * @sizeOf(i32)) == 0) return false;
    var stage: [*:0]const u8 = "concat views";
    if (!encode(graph, model, weights, next_token, position, capacity, count, &stage)) {
        _ = c.ds4_gpu_synchronize();
        _ = c.fprintf(c.__stderrp, "ds4: GLM-5.3 MTP step failed at stage '%s' (pos %u)\n", stage, position);
        return false;
    }
    var best: u32 = 0;
    for (1..shape.n_vocab) |index| {
        if (graph.mtp_logits_host[index] > graph.mtp_logits_host[best]) best = @intCast(index);
    }
    output.* = @intCast(best);
    return true;
}
