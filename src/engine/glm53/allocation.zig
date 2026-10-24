//! Allocate GLM-5.3's Metal graph after context admission and weight validation.
//! Cache dimensions are already planned in the actual record. Keeping tensor
//! ownership here makes partial failure unwind through the same native owner.
const c = @import("gguf_c");
const Config = @import("config.zig");
const Resources = @import("resources.zig");
const Graph = c.ds4_glm_gpu_graph;
const Failure = error{ AllocationFailed, TransferFailed };

fn fields(graph: *Graph, bytes: anytype) Failure!void {
    inline for (@typeInfo(@TypeOf(bytes)).@"struct".field_names) |name| {
        @field(graph, name) = c.ds4_gpu_tensor_alloc(@field(bytes, name));
        if (@field(graph, name) == null) {
            _ = c.fprintf(c.__stderrp, "ds4: GLM graph could not allocate %s\n", @as([*:0]const u8, name ++ "\x00"));
            return error.AllocationFailed;
        }
    }
}

fn arrayTensor(graph: *Graph, comptime name: []const u8, layer: usize, bytes: u64) Failure!void {
    // Metal's legacy tier-zero shim is exactly this single-device allocator.
    @field(graph, name)[layer] = c.ds4_gpu_tensor_alloc(bytes);
    if (@field(graph, name)[layer] == null) {
        _ = c.fprintf(c.__stderrp, "ds4: GLM graph could not allocate %s on tier 0\n", @as([*:0]const u8, name ++ "\x00"));
        return error.AllocationFailed;
    }
}

fn fill(tensor: ?*c.ds4_gpu_tensor, value: f32, elements: u64) Failure!void {
    if (c.ds4_gpu_tensor_fill_f32(tensor, value, elements) == 0) return error.TransferFailed;
}

fn allocate(graph: *Graph) Failure!void {
    const config = &graph.config;
    const shape = &config.shape;
    const embedding = @as(u64, shape.n_embd) * 4;
    const hc = @as(u64, Config.hcWidth(config)) * 4;
    const mix = @as(u64, Config.hcMix(config)) * 4;
    const projection = @as(u64, Config.projection(config)) * 4;
    const rank = @as(u64, shape.n_lora_q) * 4;
    const q = graph.q_dim * 4;
    const kv_raw = graph.kv_raw_dim * 4;
    const latent = @as(u64, shape.n_kv_lora) * 4;
    const key = @as(u64, shape.n_head) * graph.q_nope * 4;
    const heads = graph.heads_dim * 4;
    const index_key = @as(u64, shape.n_indexer_head_dim) * 4;
    const index_query = @as(u64, shape.n_indexer_head) * index_key;
    const index_weights = @as(u64, shape.n_indexer_head) * 4;
    const index_rows: u64 = if (graph.compact_cache_cap != 0) Config.poolCapacity(config, graph.compact_cache_cap) else graph.ctx_cap;
    const selected = shape.n_indexer_top_k + config.index_pool_size - 1;
    const selected_pools: u64 = shape.n_indexer_top_k / config.index_pool_size;
    const dense_selected: u64 = if (graph.full_kv_cache) selected else graph.ctx_cap;
    const qk = @as(u64, shape.n_head) * latent;
    const split_blocks: u64 = (selected + 31) / 32;
    const routed_mid = @as(u64, shape.n_expert_used) * shape.n_ff_exp * 4;
    const routed_output = @as(u64, shape.n_expert_used) * embedding;

    var batch = @min(graph.ctx_cap, config.prefill_chunk);
    if (graph.indexed_prefill_cap != 0) batch = @min(batch, graph.indexed_prefill_cap);
    batch = @max(batch, 1);
    graph.batch_cap = batch;
    const rows: u64 = batch;
    const indexed: u64 = graph.indexed_prefill_cap;
    const scores: u64 = graph.indexed_prefill_score_cap;

    const full_key = if (graph.full_kv_cache) @as(u64, graph.ctx_cap) * graph.q_dim * 2 else 0;
    const full_value = if (graph.full_kv_cache) @as(u64, graph.ctx_cap) * graph.heads_dim * 2 else 0;
    // Both the current Metal compact cache and expanded K/V use F16 storage.
    // The indexer's incomplete pool remains F32 and has two retained views.
    const compact_latent = @as(u64, graph.compact_cache_cap) * shape.n_kv_lora * 2;
    const compact_rope = @as(u64, graph.compact_cache_cap) * shape.n_rot * 2;
    const compact_index = @as(u64, Config.poolCapacity(config, graph.compact_cache_cap)) * shape.n_indexer_head_dim * 2;
    const tail = @as(u64, config.index_pool_size) * index_key;
    const attention_layers = graph.indexer_full_layers;
    if (graph.full_kv_cache) {
        _ = c.fprintf(c.__stderrp, "ds4: GLM graph allocating full-attention KV cache: work_ctx=%u layers=%u:%u (%u) f16 %.2f GiB\n", graph.ctx_cap, graph.layer_start, graph.layer_end, graph.layer_count, @as(f64, @floatFromInt(@as(u64, attention_layers) * (full_key + full_value))) / (1024 * 1024 * 1024));
    } else {
        _ = c.fputs("ds4: GLM graph using compact DSA KV only; expanded full-attention KV cache is skipped\n", c.__stderrp);
    }
    if (graph.compact_cache_cap != 0) {
        const total = @as(u64, attention_layers) * (compact_latent + compact_rope) + @as(u64, graph.indexer_full_layers) * compact_index;
        _ = c.fprintf(c.__stderrp, "ds4: GLM graph allocating compact DSA cache: rows=%u logical_ctx=%u kv_layers=%u indexer_layers=%u f16 %.2f GiB\n", graph.compact_cache_cap, graph.ctx_size, graph.layer_count, graph.indexer_full_layers, @as(f64, @floatFromInt(total)) / (1024 * 1024 * 1024));
        _ = c.fprintf(c.__stderrp, "ds4: GLM compact indexed prefill chunk=%u score_rows=%u score_scratch=%.2f MiB\n", graph.indexed_prefill_cap, graph.indexed_prefill_score_cap, @as(f64, @floatFromInt(scores * index_rows * 4)) / (1024 * 1024));
    }

    try fields(graph, .{ .cur = embedding, .next = embedding });
    try fields(graph, .{ .hc_cur = hc, .hc_next = hc, .hc_flat = hc, .hc_mix = mix, .hc_split = mix });
    graph.hc_post = c.ds4_gpu_tensor_view(graph.hc_split, @as(u64, shape.n_hc) * 4, @as(u64, shape.n_hc) * 4) orelse return error.AllocationFailed;
    graph.hc_comb = c.ds4_gpu_tensor_view(graph.hc_split, @as(u64, 2) * shape.n_hc * 4, @as(u64, shape.n_hc) * shape.n_hc * 4) orelse return error.AllocationFailed;
    try fields(graph, .{ .hc_after_attn = hc, .hc_output = embedding, .hc_mean_weights = @as(u64, shape.n_hc) * 4 });
    try fields(graph, .{
        .kda_q = projection,
        .kda_k = projection,
        .kda_v = projection,
        .kda_lowrank = @as(u64, shape.n_kda_head_dim) * 4,
        .kda_raw_gate = projection,
        .kda_raw_beta = @as(u64, shape.n_kda_head) * 4,
        .kda_output_gate = projection,
        .kda_out = projection,
    });
    try fields(graph, .{
        .attn_norm = embedding,
        .q_rank = rank,
        .q_rank_norm = rank,
        .q = q,
        .indexer_k = index_key,
        .indexer_gate = index_key,
        .indexer_q = index_query,
        .indexer_weights = index_weights,
        .indexer_scores = index_rows * 4,
        .indexer_pool_selected = selected_pools * 4,
        .indexer_selected = dense_selected * 4,
        .qk_low = qk,
        .attn_partial_lora = split_blocks * qk,
        .attn_partial_ms = split_blocks * shape.n_head * 2 * 4,
    });
    try fields(graph, .{
        .kv_raw = kv_raw,
        .kv_norm = latent,
        .k_nope = key,
        .value = heads,
        .heads = heads,
        .attn_out = embedding,
        .after_attn = embedding,
        .ffn_norm = embedding,
        .ffn_gate = graph.dense_hidden_max * 4,
        .ffn_up = graph.dense_hidden_max * 4,
        .ffn_mid = graph.ffn_mid_elems * 4,
    });
    if (graph.generic_routed_moe)
        try fields(graph, .{ .routed_gate = routed_mid, .routed_up = routed_mid, .routed_down = routed_output });
    try fields(graph, .{
        .ffn_out = embedding,
        .ffn_sum = embedding,
        .router_logits = @as(u64, shape.n_expert) * 4,
        .router_probs = @as(u64, shape.n_expert) * 4,
        .router_selected = @as(u64, shape.n_expert_used) * 4,
        .router_weights = @as(u64, shape.n_expert_used) * 4,
        .output_norm = embedding,
        .logits = @as(u64, shape.n_vocab) * 4,
    });
    try fields(graph, .{
        .batch_router_logits = rows * shape.n_expert * 4,
        .batch_router_probs = rows * shape.n_expert * 4,
        .batch_router_selected = rows * shape.n_expert_used * 4,
        .batch_router_weights = rows * shape.n_expert_used * 4,
        .prefill_seed_router_selected = @as(u64, shape.n_layer) * 64 * shape.n_expert_used * 4,
    });
    try fields(graph, .{
        .prefill_tokens = rows * 4,
        .batch_cur = rows * embedding,
        .batch_next = rows * embedding,
        .batch_attn_norm = rows * embedding,
        .batch_q_rank = rows * rank,
        .batch_q_rank_norm = rows * rank,
        .batch_q = rows * q,
        .batch_indexer_k = rows * index_key,
        .batch_indexer_gate = rows * index_key,
    });
    if (graph.compact_cache_cap != 0 and graph.indexed_prefill_cap != 0)
        try fields(graph, .{
            .batch_indexer_q = indexed * index_query,
            .batch_indexer_weights = indexed * index_weights,
            .batch_indexer_scores = scores * index_rows * 4,
            .batch_indexer_pool_selected = indexed * selected_pools * 4,
            .batch_indexer_selected = indexed * selected * 4,
            .batch_qk_low = indexed * qk,
            .batch_attn_lora = indexed * qk,
        });
    try fields(graph, .{
        .batch_kv_raw = rows * kv_raw,
        .batch_kv_norm = rows * latent,
        .batch_k_nope = rows * key,
        .batch_value = rows * heads,
        .batch_heads = rows * heads,
        .batch_attn_out = rows * embedding,
        .batch_after_attn = rows * embedding,
        .batch_ffn_norm = rows * embedding,
        .batch_ffn_gate = rows * graph.dense_hidden_max * 4,
        .batch_ffn_up = rows * graph.dense_hidden_max * 4,
        .batch_shared_mid = rows * shape.n_ff_exp * 4,
        .batch_ffn_mid = rows * graph.ffn_mid_elems * 4,
    });
    if (graph.generic_routed_moe)
        try fields(graph, .{ .batch_routed_gate = rows * routed_mid, .batch_routed_up = rows * routed_mid, .batch_routed_down = rows * routed_output });
    try fields(graph, .{ .batch_ffn_out = rows * embedding });

    for (graph.layer_start..graph.layer_end + 1) |layer| {
        if (Config.isKda(config, @intCast(layer))) {
            try arrayTensor(graph, "layer_kda_conv_state", layer, Config.convStateBytes(config));
            try arrayTensor(graph, "layer_kda_recurrent_state", layer, Config.recurrentStateBytes(config));
        } else {
            if (graph.full_kv_cache) {
                try arrayTensor(graph, "layer_key_cache", layer, full_key);
                try arrayTensor(graph, "layer_value_cache", layer, full_value);
            }
            if (graph.compact_cache_cap != 0) {
                try arrayTensor(graph, "layer_kv_lora_cache", layer, compact_latent);
                // A zero-width RoPE tail still has a valid placeholder handle.
                try arrayTensor(graph, "layer_k_rope_cache", layer, @max(compact_rope, 1));
                try arrayTensor(graph, "layer_indexer_key_cache", layer, compact_index);
                try arrayTensor(graph, "layer_indexer_tail_k", layer, 2 * tail);
                graph.layer_indexer_tail_gate[layer] = c.ds4_gpu_tensor_view(graph.layer_indexer_tail_k[layer], tail, tail) orelse return error.AllocationFailed;
            }
        }
    }
    try fill(graph.hc_mean_weights, 1 / @as(f32, @floatFromInt(shape.n_hc)), shape.n_hc);
    for (graph.layer_start..graph.layer_end + 1) |layer| {
        if (Config.isKda(config, @intCast(layer))) {
            try fill(graph.layer_kda_conv_state[layer], 0, Config.convStateBytes(config) / 4);
            try fill(graph.layer_kda_recurrent_state[layer], 0, Config.recurrentStateBytes(config) / 4);
        } else if (graph.layer_indexer_tail_k[layer] != null and graph.layer_indexer_tail_gate[layer] != null) {
            try fill(graph.layer_indexer_tail_k[layer], 0, tail / 4);
            try fill(graph.layer_indexer_tail_gate[layer], 0, tail / 4);
        }
    }
    if (!Resources.ensurePrefill(graph, graph.batch_cap)) return error.AllocationFailed;
    graph.ws_ready = 0;
    graph.ws_tier = -1;
}

pub fn init(graph: *Graph) bool {
    if (!graph.glm53 or graph.placement != null) return false;
    allocate(graph) catch {
        _ = Resources.deinit(graph);
        return false;
    };
    return true;
}
