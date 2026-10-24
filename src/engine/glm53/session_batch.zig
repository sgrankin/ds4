//! Resident decoding of independent sessions. Projections share row kernels;
//! recurrence and KV publication still use each row's own graph and position.
const c = @import("gguf_c");
const std = @import("std");
const Config = @import("config.zig");
const Ops = @import("ops.zig");
const Kda = @import("kda.zig");
const Moe = @import("moe.zig");
const Resources = @import("resources.zig");
const Graph = c.ds4_glm_gpu_graph;
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const Layer = c.ds4_layer_weights;
const Tensor = c.ds4_gpu_tensor;
const Row = c.ds4_glm_session_row;
const allocator = std.heap.c_allocator;

pub fn sameSteering(a: *const Graph, b: *const Graph) bool {
    return a.directional_steering_mode == b.directional_steering_mode and a.directional_steering_layers == b.directional_steering_layers and
        a.directional_steering_attn_scale == b.directional_steering_attn_scale and
        a.directional_steering_ffn_scale == b.directional_steering_ffn_scale and
        ((a.directional_steering_attn_scale == 0 and a.directional_steering_ffn_scale == 0) or
            (a.directional_steering_dirs_by_tier[0] != null and b.directional_steering_dirs_by_tier[0] != null));
}

fn view(tensor: ?*const Tensor, row: usize, width: u64) ?*Tensor {
    return c.ds4_gpu_tensor_view(tensor, row * width * @sizeOf(f32), width * @sizeOf(f32));
}

fn release(views: anytype) void {
    inline for (@typeInfo(@TypeOf(views)).@"struct".field_names) |name| c.ds4_gpu_tensor_free(@field(views, name));
}

fn ready(views: anytype) bool {
    inline for (@typeInfo(@TypeOf(views)).@"struct".field_names) |name| if (@field(views, name) == null) return false;
    return true;
}

fn kda(batch: *Graph, model: *const Model, weights: *const Layer, layer: u32, sessions: []const Row) bool {
    const shape = &batch.config.shape;
    const projection = Config.projection(&batch.config);
    if (!Kda.project(true, batch, model, weights, layer, 0, @intCast(sessions.len))) return false;
    for (sessions, 0..) |session, row| {
        const graph: *Graph = session.graph;
        const work = .{
            .q = view(batch.batch_kda_q, row, projection),
            .k = view(batch.batch_kda_k, row, projection),
            .v = view(batch.batch_kda_v, row, projection),
            .gate = view(batch.batch_kda_raw_gate, row, projection),
            .beta = view(batch.batch_kda_raw_beta, row, shape.n_kda_head),
            .output_gate = view(batch.batch_kda_output_gate, row, projection),
            .output = view(batch.batch_kda_out, row, projection),
        };
        defer release(work);
        if (!ready(work) or c.ds4_gpu_glm53_kda_decode(
            work.output,
            graph.layer_kda_conv_state[layer],
            graph.layer_kda_recurrent_state[layer],
            work.q,
            work.k,
            work.v,
            work.gate,
            work.beta,
            work.output_gate,
            model.map,
            model.size,
            weights.kda_q_conv.*.abs_offset,
            weights.kda_k_conv.*.abs_offset,
            weights.kda_v_conv.*.abs_offset,
            weights.kda_a_log.*.abs_offset,
            weights.kda_dt_bias.*.abs_offset,
            weights.kda_o_norm.*.abs_offset,
            shape.n_kda_head,
            1,
            shape.kda_gate_lower_bound,
            shape.rms_eps,
        ) == 0) return false;
    }
    return Ops.matmul(batch.batch_attn_out, model, weights.kda_output, projection, shape.n_embd, batch.batch_kda_out, @intCast(sessions.len));
}

fn dsa(batch: *Graph, model: *const Model, weights: *const Layer, layer: u32, sessions: []const Row) bool {
    const shape = &batch.config.shape;
    const rows: u32 = @intCast(sessions.len);
    const kv_width: u32 = @intCast(weights.attn_kv_a_mqa.*.dim[1]);
    if (!Ops.matmul(batch.batch_q_rank, model, weights.attn_q_a, shape.n_embd, shape.n_lora_q, batch.batch_attn_norm, rows) or
        !Ops.matmul(batch.batch_kv_raw, model, weights.attn_kv_a_mqa, shape.n_embd, kv_width, batch.batch_attn_norm, rows) or
        c.ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(batch.batch_q_rank_norm, batch.batch_q_rank, model.map, model.size, weights.attn_q_a_norm.*.abs_offset, shape.n_lora_q, batch.batch_kv_norm, batch.batch_kv_raw, weights.attn_kv_a_norm.*.abs_offset, shape.n_kv_lora, rows, shape.rms_eps) == 0 or
        !Ops.matmul(batch.batch_q, model, weights.attn_q_b, shape.n_lora_q, @intCast(batch.q_dim), batch.batch_q_rank_norm, rows) or
        !Ops.matmul(batch.batch_indexer_k, model, weights.indexer_attn_k, shape.n_embd, shape.n_indexer_head_dim, batch.batch_attn_norm, rows) or
        !Ops.matmul(batch.batch_indexer_gate, model, weights.indexer_compressor_gate, shape.n_embd, shape.n_indexer_head_dim, batch.batch_attn_norm, rows)) return false;
    for (sessions, 0..) |session, row| {
        const graph: *Graph = session.graph;
        const position = session.position;
        const work = .{
            .q = view(batch.batch_q, row, batch.q_dim),
            .kv_raw = c.ds4_gpu_tensor_view(batch.batch_kv_raw, @as(u64, row) * batch.kv_raw_dim * @sizeOf(f32), @as(u64, kv_width) * @sizeOf(f32)),
            .kv_norm = view(batch.batch_kv_norm, row, shape.n_kv_lora),
            .qk = view(batch.batch_qk_low, row, @as(u64, shape.n_head) * shape.n_kv_lora),
            .indexer_k = view(batch.batch_indexer_k, row, shape.n_indexer_head_dim),
            .indexer_gate = view(batch.batch_indexer_gate, row, shape.n_indexer_head_dim),
            .heads = view(batch.batch_heads, row, batch.heads_dim),
        };
        defer release(work);
        if (!ready(work) or c.ds4_gpu_glm53_indexer_pool_update_tensor(
            graph.layer_indexer_key_cache[layer],
            graph.layer_indexer_tail_k[layer],
            graph.layer_indexer_tail_gate[layer],
            work.indexer_k,
            work.indexer_gate,
            model.map,
            model.size,
            weights.indexer_k_norm.*.abs_offset,
            weights.indexer_k_norm_b.*.abs_offset,
            weights.indexer_compressor_ape.*.abs_offset,
            position,
            1,
            graph.compact_cache_cap,
            shape.n_indexer_head_dim,
            batch.config.index_pool_size,
            1.0e-6,
            true,
        ) == 0 or c.ds4_gpu_glm_store_compact_kv_tensor(graph.layer_kv_lora_cache[layer], graph.layer_k_rope_cache[layer], work.kv_norm, work.kv_raw, position, 1, graph.compact_cache_cap, kv_width, shape.n_kv_lora, shape.n_rot, true) == 0 or
            c.ds4_gpu_glm_fill_selected_range_tensor(graph.indexer_selected, position + 1) == 0 or
            c.ds4_gpu_glm_qk_lowrank_typed_tensor(work.qk, work.q, model.map, model.size, weights.attn_k_b.*.abs_offset, weights.attn_k_b.*.type, shape.n_head, shape.n_kv_lora, @intCast(graph.q_nope), shape.n_key_mla) == 0 or
            c.ds4_gpu_glm_attention_indexed_decode_typed_tensor(
                work.heads,
                work.q,
                work.qk,
                graph.layer_kv_lora_cache[layer],
                graph.layer_k_rope_cache[layer],
                model.map,
                model.size,
                weights.attn_v_b.*.abs_offset,
                weights.attn_v_b.*.type,
                graph.indexer_selected,
                position + 1,
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
            ) == 0) return false;
    }
    return Ops.matmul(batch.batch_attn_out, model, weights.attn_output, @intCast(batch.heads_dim), shape.n_embd, batch.batch_heads, rows);
}

pub fn encode(model: *const Model, weights: *const Weights, sessions: []const Row, ablate: u32) bool {
    if (sessions.len < 2) return false;
    const graph: *Graph = sessions[0].graph;
    for (sessions) |session| if (!sameSteering(graph, session.graph)) return false;
    const shape = &graph.config.shape;
    const rows: u32 = @intCast(sessions.len);
    if (!graph.glm53 or graph.ssd_streaming or graph.placement != null or graph.tp_world > 1 or graph.full_kv_cache or rows > graph.ctx_cap) return false;
    if (!Resources.ensurePrefill(graph, rows) or !Resources.ensureBatchLogits(graph, rows)) return false;
    const plain = @as(u64, rows) * shape.n_embd;
    const hc = plain * shape.n_hc;
    const work = .{
        .cur = c.ds4_gpu_tensor_view(graph.batch_cur, 0, plain * @sizeOf(f32)),
        .next = c.ds4_gpu_tensor_view(graph.batch_next, 0, plain * @sizeOf(f32)),
        .after = c.ds4_gpu_tensor_view(graph.batch_after_attn, 0, plain * @sizeOf(f32)),
        .hc_cur = c.ds4_gpu_tensor_view(graph.batch_hc_cur, 0, hc * @sizeOf(f32)),
        .hc_next = c.ds4_gpu_tensor_view(graph.batch_hc_next, 0, hc * @sizeOf(f32)),
        .hc_after = c.ds4_gpu_tensor_view(graph.batch_hc_after_attn, 0, hc * @sizeOf(f32)),
    };
    defer release(work);
    if (!ready(work)) return false;
    const tokens = allocator.alloc(i32, rows) catch return false;
    defer allocator.free(tokens);
    for (sessions, tokens) |session, *token| token.* = session.token;
    if (c.ds4_gpu_tensor_write(graph.prefill_tokens, 0, tokens.ptr, @as(u64, rows) * @sizeOf(i32)) == 0) return false;
    const embedding = weights.token_embd;
    const embedded = if (embedding.*.type == c.DS4_TENSOR_BF16)
        c.ds4_gpu_glm53_embedding_bf16(work.cur, model.map, model.size, embedding.*.abs_offset, graph.prefill_tokens, rows, shape.n_embd, shape.n_vocab) != 0
    else
        c.ds4_gpu_embed_tokens_quant_tensor(work.cur, graph.prefill_tokens, model.map, model.size, embedding.*.abs_offset, embedding.*.type, shape.n_vocab, rows, shape.n_embd) != 0;
    if (!embedded or c.ds4_gpu_repeat_hc_rows_tensor(work.hc_cur, work.cur, rows, shape.n_embd, shape.n_hc) == 0) return false;
    var current = work.hc_cur;
    var next = work.hc_next;
    for (graph.layer_start..graph.layer_end + 1) |index| {
        const layer: u32 = @intCast(index);
        const layer_weights = &weights.layer[layer];
        if (!Ops.hcPre(true, graph, model, layer_weights.hc_attn_fn, layer_weights.hc_attn_scale, layer_weights.hc_attn_base, layer_weights.attn_norm, current, work.cur, graph.batch_attn_norm, graph.batch_hc_flat, graph.batch_hc_mix, graph.batch_hc_split, rows)) return false;
        const attention_ok = if (Config.isKda(&graph.config, layer)) kda(graph, model, layer_weights, layer, sessions) else dsa(graph, model, layer_weights, layer, sessions);
        if (!attention_ok or !Ops.hcExpand(graph, work.hc_after, graph.batch_attn_out, current, graph.batch_hc_split, layer, graph.directional_steering_attn_scale) or
            !Ops.hcPre(true, graph, model, layer_weights.hc_ffn_fn, layer_weights.hc_ffn_scale, layer_weights.hc_ffn_base, layer_weights.ffn_norm, work.hc_after, work.after, graph.batch_ffn_norm, graph.batch_hc_flat, graph.batch_hc_mix, graph.batch_hc_split, rows) or
            !Moe.encodeRows(graph, model, layer_weights, layer, 0, rows, work.next, false, ablate) or
            !Ops.hcExpand(graph, next, work.next, work.hc_after, graph.batch_hc_split, layer, graph.directional_steering_ffn_scale)) return false;
        std.mem.swap(?*Tensor, &current, &next);
    }
    for (0..rows) |row| {
        const hidden = view(current, row, @as(u64, shape.n_hc) * shape.n_embd);
        defer c.ds4_gpu_tensor_free(hidden);
        const collapsed = view(work.cur, row, shape.n_embd);
        defer c.ds4_gpu_tensor_free(collapsed);
        if (hidden == null or collapsed == null or c.ds4_gpu_hc_weighted_sum_tensor(collapsed, hidden, graph.hc_mean_weights, shape.n_embd, shape.n_hc) == 0) return false;
    }
    if (c.ds4_gpu_rms_norm_weight_rows_tensor(graph.batch_ffn_norm, work.cur, model.map, model.size, weights.output_norm.*.abs_offset, shape.n_embd, rows, shape.rms_eps) == 0 or
        !Ops.matmul(graph.session_batch_logits, model, weights.output, shape.n_embd, shape.n_vocab, graph.batch_ffn_norm, rows)) return false;
    for (sessions, 0..) |session, row| {
        const target: *Graph = session.graph;
        if (c.ds4_gpu_tensor_copy(target.logits, 0, graph.session_batch_logits, @as(u64, row) * shape.n_vocab * @sizeOf(f32), @as(u64, shape.n_vocab) * @sizeOf(f32)) == 0) return false;
    }
    return true;
}
