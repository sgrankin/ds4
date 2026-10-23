//! V4.1's three-stage drafter and bounded verification frontier. The target
//! cache stays shared: rollback saves only overwritten raw/features/pooling
//! carries, never a context-sized second compressed cache.
const std = @import("std");
const c = @import("gguf_c");
const Config = @import("../config.zig").ModelConfig;
const Ops = @import("ops.zig").Ops;
const Layout = @import("layout.zig").Layout;
const RowField = @import("layout.zig").RowField;
const SessionBatch = @import("session_batch.zig").SessionBatch;
const Executor = @import("executor.zig").Executor;
const Policy = @import("prefill_policy.zig").Policy;
const experts = @import("expert_cache.zig");
const Record = c.ds41_dspark;
const Graph = c.ds41_gpu_graph;
const Row = c.ds41_prefill_row;
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const DraftWeights = c.ds4_dspark_weights;
const Layer = c.ds4_layer_weights;
const Tensor = c.ds4_gpu_tensor;
const window = c.DS41_DRAFT_WINDOW;
const draft_rows = c.DS41_DRAFT_ROWS;
const verify_rows = c.DS41_VERIFY_ROWS;

fn draftWeights(state: *const Record) *const DraftWeights {
    // Allocation accepts only a validated, borrowed weight table. Narrow its
    // generated C field once before taking ordinary native field pointers.
    return @ptrCast(state.weights);
}

pub fn reset(record: ?*Record) void {
    const state = record orelse return;
    for (&state.positions) |*positions| @memset(positions, std.math.maxInt(u32));
    state.kv_valid = false;
    state.kv_end = 0;
    state.verify_count = 0;
    state.rewind_end = 0;
}

pub fn free(record: ?*Record) void {
    const state = record orelse return;
    inline for (comptime std.meta.fieldNames(Row)) |name| c.ds4_gpu_tensor_free(@field(state.batch, name));
    for (state.target, state.window) |target, keys| {
        c.ds4_gpu_tensor_free(target);
        c.ds4_gpu_tensor_free(keys);
    }
    inline for (.{ "mean_weights", "mean", "input", "projected", "main_x", "main_kv", "logits", "markov_embed", "tokens", "markov_logits", "confidence_input", "confidence" }) |name|
        c.ds4_gpu_tensor_free(@field(state, name));
    for (&state.verified) |*graph| {
        // These target-graph copies borrow every cache and workspace. Only
        // these stable view handles were allocated for verifier snapshots.
        c.ds4_gpu_tensor_free(graph.residual);
        c.ds4_gpu_tensor_free(graph.pre);
        c.ds4_gpu_tensor_free(graph.logits);
        c.ds4_gpu_tensor_free(graph.tp_logits_half);
        for (graph.verify_previous_kv, graph.verify_previous_score) |keys, scores| {
            c.ds4_gpu_tensor_free(keys);
            c.ds4_gpu_tensor_free(scores);
        }
    }
    inline for (.{ "verify_residual", "verify_pre", "verify_logits", "verify_pool", "old_raw", "old_features" }) |name|
        c.ds4_gpu_tensor_free(@field(state, name));
    c.free(state.host_verified);
    c.free(state.host_logits);
    c.free(state.draft_probs);
    c.free(state);
}

pub fn allocate(config: *const Config, weights: ?*const DraftWeights) ?*Record {
    const draft = weights orelse return null;
    if (!draft.v41 or draft.n_stages != 3 or draft.block_size != draft_rows or
        draft.missing_tensors != 0 or draft.invalid_tensors != 0 or draft.metadata_errors != 0) return null;
    const memory = c.calloc(1, @sizeOf(Record)) orelse return null;
    const state: *Record = @ptrCast(@alignCast(memory));
    var complete = false;
    defer if (!complete) free(state);
    state.weights = draft;
    const shape = &config.shape;
    var graph = std.mem.zeroes(Graph);
    graph.ctx = window;
    const plan = Layout.fromRecord(config, &graph);
    inline for (comptime std.meta.fieldNames(Row)) |name| {
        @field(state.batch, name) = c.ds4_gpu_tensor_alloc(plan.rowWidth(@field(RowField, name)) * draft_rows * 4) orelse return null;
    }
    for (&state.target, &state.window) |*target, *keys| {
        target.* = c.ds4_gpu_tensor_alloc(@as(u64, window) * shape.n_embd * 4) orelse return null;
        keys.* = c.ds4_gpu_tensor_alloc(@as(u64, window + draft_rows) * shape.n_head_dim * 4) orelse return null;
    }
    const names = .{ "mean_weights", "mean", "input", "projected", "main_x", "main_kv", "logits", "tokens", "markov_embed", "markov_logits", "confidence_input", "confidence", "verify_residual", "verify_pre", "verify_logits", "verify_pool", "old_raw", "old_features" };
    const counts = [_]u64{
        @as(u64, window) * shape.n_hc,                         @as(u64, window) * shape.n_embd,
        @as(u64, window) * 3 * shape.n_embd,                   @as(u64, window) * shape.n_embd,
        @as(u64, window) * shape.n_embd,                       @as(u64, window) * shape.n_head_dim,
        @as(u64, draft_rows) * shape.n_vocab,                  draft_rows,
        draft.markov_rank,                                     shape.n_vocab,
        shape.n_embd + draft.markov_rank,                      1,
        @as(u64, verify_rows + 1) * shape.n_hc * shape.n_embd, @as(u64, verify_rows + 1) * shape.n_hc,
        @as(u64, verify_rows + 1) * shape.n_vocab,             @as(u64, verify_rows + 1) * 8 * shape.n_head_dim,
        @as(u64, 40) * verify_rows * shape.n_head_dim,         @as(u64, 3) * verify_rows * shape.n_embd,
    };
    inline for (names, 0..) |name, index| {
        @field(state, name) = c.ds4_gpu_tensor_alloc(counts[index] * 4) orelse return null;
    }
    state.host_logits = @ptrCast(@alignCast(c.malloc(@as(u64, draft_rows) * shape.n_vocab * 4) orelse return null));
    state.draft_probs = @ptrCast(@alignCast(c.malloc(@as(u64, draft_rows) * shape.n_vocab * 4) orelse return null));
    state.host_verified = @ptrCast(@alignCast(c.malloc(@as(u64, verify_rows + 1) * shape.n_vocab * 4) orelse return null));
    if (c.ds4_gpu_tensor_fill_f32(state.mean_weights, 1.0 / @as(f32, @floatFromInt(shape.n_hc)), window * shape.n_hc) == 0 or
        c.ds4_gpu_tensor_fill_f32(state.tokens, 0, draft_rows) == 0) return null;
    reset(state);
    complete = true;
    return state;
}

pub fn capture(config: *const Config, record: ?*Record, layer: u32, position: u32, residual: ?*const Tensor, rows: u32) bool {
    const state = record orelse return true;
    const weights = draftWeights(state);
    state.rewind_end = 0;
    var stage: usize = 0;
    while (stage < 3 and weights.target_layers[stage] != layer) stage += 1;
    if (stage == 3) return true;
    if (rows == 0 or rows > std.math.maxInt(u32) - position) return false;
    const count: u32 = @min(rows, window);
    const first = rows - count;
    const start = position + first;
    const shape = &config.shape;
    const bytes = @as(u64, shape.n_embd) * 4;
    const source = c.ds4_gpu_tensor_view(@constCast(residual), @as(u64, first) * shape.n_hc * bytes, @as(u64, count) * shape.n_hc * bytes);
    defer c.ds4_gpu_tensor_free(source);
    const out = c.ds4_gpu_tensor_view(state.mean, 0, @as(u64, count) * bytes);
    defer c.ds4_gpu_tensor_free(out);
    const mix = c.ds4_gpu_tensor_view(state.mean_weights, 0, @as(u64, count) * shape.n_hc * 4);
    defer c.ds4_gpu_tensor_free(mix);
    if (source == null or out == null or mix == null or
        c.ds4_gpu_hc_weighted_sum_tensor(out, source, mix, shape.n_embd, shape.n_hc) == 0 or
        c.ds4_gpu_dsv41_quantize(out, shape.n_embd, count, c.DS4_V41_BF16) == 0) return false;
    const slot = start % window;
    const part: u32 = @min(count, window - slot);
    if (c.ds4_gpu_tensor_copy(state.target[stage], @as(u64, slot) * bytes, out, 0, @as(u64, part) * bytes) == 0 or
        (part != count and c.ds4_gpu_tensor_copy(state.target[stage], 0, out, @as(u64, part) * bytes, @as(u64, count - part) * bytes) == 0)) return false;
    for (0..count) |index| state.positions[stage][(start + @as(u32, @intCast(index))) % window] = start + @as(u32, @intCast(index));
    return true;
}

pub const Drafter = struct {
    ops: Ops, // Draft-model math; the validated target shape is shared.
    state: *Record,

    // Only accepted target features enter these rings. A fresh prefill seeds
    // its last window once; later calls project only newly accepted rows.
    pub fn seed(self: *const Drafter, position: u32) bool {
        if (position == 0) return false;
        const state = self.state;
        const weights = draftWeights(state);
        const shape = &self.ops.config.shape;
        const oldest = position -| window;
        const start = if (state.kv_valid and state.kv_end >= oldest and state.kv_end <= position) state.kv_end else oldest;
        const count = position - start;
        if (count == 0) return true;
        const row_bytes = @as(u64, shape.n_embd) * 4;
        for (0..count) |index| {
            const slot = (start + @as(u32, @intCast(index))) % window;
            for (0..3) |stage_index| {
                if (state.positions[stage_index][slot] != start + index or
                    c.ds4_gpu_tensor_copy(state.input, (index * 3 + stage_index) * row_bytes, state.target[stage_index], @as(u64, slot) * row_bytes, row_bytes) == 0) return false;
            }
        }
        const first = &weights.stage[0];
        if (!self.ops.matmulBatch(state.projected, first.main_proj, state.input, count, true) or
            !self.ops.normBatch(state.main_x, state.projected, first.main_norm, count)) return false;
        const kv_bytes = @as(u64, shape.n_head_dim) * 4;
        for (0..3) |stage_index| {
            const layer = &weights.stage[stage_index].block;
            if (!self.ops.matmulBatch(state.main_kv, layer.attn_kv, state.main_x, count, true) or
                !self.ops.normBatch(state.main_kv, state.main_kv, layer.attn_kv_a_norm, count) or
                c.ds4_gpu_dsv41_rope(state.main_kv, shape.n_head_dim, 1, count, start, false, false) == 0 or
                c.ds4_gpu_dsv41_quantize(state.main_kv, shape.n_head_dim, count, c.DS4_V41_FP8_E8M0) == 0) return false;
            const slot = start % window;
            const part: u32 = @min(count, window - slot);
            if (c.ds4_gpu_tensor_copy(state.window[stage_index], @as(u64, slot) * kv_bytes, state.main_kv, 0, @as(u64, part) * kv_bytes) == 0 or
                (part < count and c.ds4_gpu_tensor_copy(state.window[stage_index], 0, state.main_kv, @as(u64, part) * kv_bytes, @as(u64, count - part) * kv_bytes) == 0)) return false;
        }
        state.kv_end = position;
        state.kv_valid = true;
        return true;
    }

    fn attentionOutput(self: *const Drafter, layer: *const Layer) bool {
        const batch = &self.state.batch;
        const shape = &self.ops.config.shape;
        const groups = shape.n_out_group;
        const width = @as(u64, shape.n_head) * shape.n_head_dim / groups;
        var weight_row: u64 = undefined;
        if (!c.ds4_gguf_tensor_nbytes(layer.attn_output_a.*.type, width, &weight_row)) return false;
        for (0..draft_rows) |row| {
            for (0..groups) |group| {
                var slice = layer.attn_output_a.*;
                slice.abs_offset +%= @as(u64, group) *% shape.n_lora_o *% weight_row;
                slice.dim[1] = shape.n_lora_o;
                const input = c.ds4_gpu_tensor_view(batch.heads, (row * groups + group) * width * 4, width * 4);
                defer c.ds4_gpu_tensor_free(input);
                const out = c.ds4_gpu_tensor_view(batch.low, (row * groups + group) * shape.n_lora_o * 4, @as(u64, shape.n_lora_o) * 4);
                defer c.ds4_gpu_tensor_free(out);
                if (input == null or out == null or !self.ops.matmul(out, &slice, input, true)) return false;
            }
        }
        return self.ops.matmulBatch(batch.block, layer.attn_output_b, batch.low, draft_rows, true);
    }

    pub fn stage(self: *const Drafter, index: u32, position: u32) bool {
        const state = self.state;
        const model = self.ops.model;
        const shape = &self.ops.config.shape;
        const batch = &state.batch;
        const weights = draftWeights(state);
        const layer = &weights.stage[index].block;
        const past: u32 = @min(position, window);
        const mixed = self.ops.hcMixBatch(batch, layer, false, draft_rows) and (if (index != 0)
            c.ds4_gpu_hc_weighted_sum_split_tensor(batch.x, batch.residual, batch.ffn_split, shape.n_embd, shape.n_hc)
        else
            c.ds4_gpu_hc_weighted_sum_tensor(batch.x, batch.residual, batch.pre, shape.n_embd, shape.n_hc)) != 0;
        if (!mixed or
            c.ds4_gpu_dsv41_quantize(batch.x, shape.n_embd, draft_rows, c.DS4_V41_BF16) == 0 or
            !self.ops.normBatch(batch.norm, batch.x, layer.attn_norm, draft_rows) or
            !self.ops.matmulBatch(batch.qr, layer.attn_q_a, batch.norm, draft_rows, true) or
            !self.ops.normBatch(batch.qr, batch.qr, layer.attn_q_a_norm, draft_rows) or
            !self.ops.matmulBatch(batch.q, layer.attn_q_b, batch.qr, draft_rows, true) or
            c.ds4_gpu_dsv41_rope(batch.q, shape.n_head_dim, shape.n_head, draft_rows, position, false, false) == 0 or
            !self.ops.matmulBatch(batch.kv, layer.attn_kv, batch.norm, draft_rows, true) or
            !self.ops.normBatch(batch.kv, batch.kv, layer.attn_kv_a_norm, draft_rows) or
            c.ds4_gpu_dsv41_rope(batch.kv, shape.n_head_dim, 1, draft_rows, position, false, false) == 0 or
            c.ds4_gpu_dsv41_quantize(batch.kv, shape.n_head_dim, draft_rows, c.DS4_V41_FP8_E8M0) == 0 or
            c.ds4_gpu_tensor_copy(state.window[index], @as(u64, past) * shape.n_head_dim * 4, batch.kv, 0, @as(u64, draft_rows) * shape.n_head_dim * 4) == 0 or
            c.ds4_gpu_attention_noncausal_raw_batch_heads_tensor(batch.heads, model.map, model.size, layer.attn_sinks.*.abs_offset, batch.q, state.window[index], draft_rows, past + draft_rows, window + draft_rows, 0, shape.n_head, shape.n_head_dim) == 0 or
            c.ds4_gpu_dsv41_quantize(batch.heads, shape.n_head * shape.n_head_dim, draft_rows, c.DS4_V41_BF16) == 0 or
            c.ds4_gpu_dsv41_rope(batch.heads, shape.n_head_dim, shape.n_head, draft_rows, position, false, true) == 0 or
            !self.attentionOutput(layer) or !self.afterAttention(layer))
        {
            _ = c.fprintf(c.__stderrp, "ds4: V4.1 draft attention failed\n");
            return false;
        }
        const count = weights.n_experts;
        const active = weights.n_experts_used;
        const gate_row = experts.rowBytes(layer.ffn_gate_exps);
        const down_row = experts.rowBytes(layer.ffn_down_exps);
        if (!self.ops.matmulBatch(batch.route_logits, layer.ffn_gate_inp, batch.norm, draft_rows, false) or
            c.ds4_gpu_router_select_batch_tensor(batch.selected, batch.route_weights, batch.route_probs, model.map, model.size, layer.ffn_exp_probs_b.*.abs_offset, 0, 0, 0, 0, true, false, batch.route_logits, state.tokens, count, active, shape.expert_weight_scale, draft_rows) == 0)
        {
            _ = c.fprintf(c.__stderrp, "ds4: V4.1 draft routing failed\n");
            return false;
        }
        if (!self.ops.matmulBatch(batch.shared_gate, layer.ffn_gate_shexp, batch.norm, draft_rows, true) or
            !self.ops.matmulBatch(batch.shared_up, layer.ffn_up_shexp, batch.norm, draft_rows, true) or
            c.ds4_gpu_swiglu_tensor(batch.shared_mid, batch.shared_gate, batch.shared_up, draft_rows * shape.n_ff_exp, shape.swiglu_clamp_exp, 1.0) == 0 or
            c.ds4_gpu_dsv41_quantize(batch.shared_mid, shape.n_ff_exp, draft_rows, c.DS4_V41_BF16) == 0 or
            !self.ops.matmulBatch(batch.shared, layer.ffn_down_shexp, batch.shared_mid, draft_rows, true))
        {
            _ = c.fprintf(c.__stderrp, "ds4: V4.1 draft shared expert failed\n");
            return false;
        }
        var mid_f16 = false;
        if (c.ds4_gpu_routed_moe_batch_tensor(batch.routed, batch.gate, batch.up, batch.mid, batch.experts, model.map, model.size, layer.ffn_gate_exps.*.abs_offset, layer.ffn_up_exps.*.abs_offset, layer.ffn_down_exps.*.abs_offset, layer.ffn_gate_exps.*.type, layer.ffn_down_exps.*.type, gate_row *% shape.n_ff_exp, gate_row, down_row *% shape.n_embd, down_row, shape.n_embd, shape.n_ff_exp, shape.n_embd, batch.selected, batch.route_weights, count, active, shape.swiglu_clamp_exp, batch.norm, index, draft_rows, &mid_f16, true) == 0) {
            _ = c.fprintf(c.__stderrp, "ds4: V4.1 draft routed experts failed\n");
            return false;
        }
        return c.ds4_gpu_add_tensor(batch.block, batch.routed, batch.shared, draft_rows * shape.n_embd) != 0 and
            c.ds4_gpu_dsv41_quantize(batch.block, shape.n_embd, draft_rows, c.DS4_V41_BF16) != 0 and
            c.ds4_gpu_hc_expand_split_tensor(batch.residual, batch.block, batch.after_attn, batch.ffn_split, shape.n_embd, shape.n_hc) != 0 and
            c.ds4_gpu_dsv41_quantize(batch.residual, shape.n_hc * shape.n_embd, draft_rows, c.DS4_V41_BF16) != 0;
    }

    fn afterAttention(self: *const Drafter, layer: *const Layer) bool {
        // The drafter retains the target's HC/FFN normalization boundaries;
        // no target executor services or graph ownership are involved here.
        const batch = &self.state.batch;
        const shape = &self.ops.config.shape;
        return c.ds4_gpu_hc_expand_split_tensor(batch.after_attn, batch.block, batch.residual, batch.attn_split, shape.n_embd, shape.n_hc) != 0 and
            c.ds4_gpu_dsv41_quantize(batch.after_attn, shape.n_embd * shape.n_hc, draft_rows, c.DS4_V41_BF16) != 0 and
            self.ops.hcMixBatch(batch, layer, true, draft_rows) and
            c.ds4_gpu_hc_weighted_sum_split_tensor(batch.x, batch.after_attn, batch.attn_split, shape.n_embd, shape.n_hc) != 0 and
            c.ds4_gpu_dsv41_quantize(batch.x, shape.n_embd, draft_rows, c.DS4_V41_BF16) != 0 and
            self.ops.normBatch(batch.norm, batch.x, layer.ffn_norm, draft_rows);
    }

    // Leave five base-logit rows and collapsed features on the GPU. Markov
    // sampling is sequential and follows the whole three-stage chain.
    pub fn forward(self: *const Drafter, target: *const Model, weights: *const Weights, position: u32, token: c_int, target_graph: ?*Graph) bool {
        const state = self.state;
        const shape = &self.ops.config.shape;
        const draft_weights = draftWeights(state);
        if (position == 0 or token < 0 or @as(u32, @bitCast(token)) >= shape.n_vocab) return false;
        const local_experts = if (target_graph) |graph| graph.tp_world == 2 else false;
        if (local_experts) c.ds4_gpu_tp_suspend_expert_sharding(1);
        defer if (local_experts) c.ds4_gpu_tp_suspend_expert_sharding(0);
        var ok = c.ds4_gpu_begin_commands() != 0 and self.seed(position);
        if (!ok) _ = c.fprintf(c.__stderrp, "ds4: V4.1 DSpark seed failed\n");
        const pre = [_][4]f32{ .{ 1, 0, 0, 0 }, .{ 1, 0, 0, 0 }, .{ 1, 0, 0, 0 }, .{ 1, 0, 0, 0 }, .{ 1, 0, 0, 0 } };
        if (ok) ok = c.ds4_gpu_tensor_write(state.batch.pre, 0, &pre, @sizeOf(@TypeOf(pre))) != 0;
        var row: u32 = 0;
        while (ok and row < draft_rows) : (row += 1) {
            const bytes = @as(u64, shape.n_embd) * shape.n_hc * 4;
            const out = c.ds4_gpu_tensor_view(state.batch.residual, @as(u64, row) * bytes, bytes);
            ok = out != null and c.ds4_gpu_embed_token_hc_tensor(out, target.map, target.size, weights.token_embd.*.abs_offset, shape.n_vocab, if (row != 0) draft_weights.noise_token_id else @as(u32, @bitCast(token)), shape.n_embd, shape.n_hc) != 0;
            c.ds4_gpu_tensor_free(out);
        }
        var index: u32 = 0;
        while (ok and index < 3) : (index += 1) {
            ok = self.stage(index, position);
            if (!ok) _ = c.fprintf(c.__stderrp, "ds4: V4.1 DSpark stage %u failed\n", index);
        }
        if (ok) ok = c.ds4_gpu_hc_weighted_sum_split_tensor(state.batch.x, state.batch.residual, state.batch.ffn_split, shape.n_embd, shape.n_hc) != 0 and
            c.ds4_gpu_dsv41_quantize(state.batch.x, shape.n_embd, draft_rows, c.DS4_V41_BF16) != 0 and
            self.ops.normBatch(state.batch.norm, state.batch.x, draft_weights.stage[2].norm, draft_rows);
        const target_ops = Ops{ .config = self.ops.config, .model = target, .exact_prefill_rows = self.ops.exact_prefill_rows };
        if (ok) ok = if (target_graph) |graph|
            target_ops.outputProjection(graph, state.logits, weights, state.batch.norm, draft_rows)
        else
            target_ops.matmulBatch(state.logits, weights.output, state.batch.norm, draft_rows, false);
        if (c.ds4_gpu_commands_active() != 0 and c.ds4_gpu_end_commands() == 0) ok = false;
        if (!ok) state.kv_valid = false;
        return ok;
    }

    pub fn markov(self: *const Drafter, row: u32, previous: c_int, confidence: *f32) bool {
        const shape = &self.ops.config.shape;
        if (row >= draft_rows or previous < 0 or @as(u32, @bitCast(previous)) >= shape.n_vocab) return false;
        const state = self.state;
        const weights = draftWeights(state);
        const last = &weights.stage[2];
        const rank = weights.markov_rank;
        var embedding: [256]f32 = undefined;
        if (rank != embedding.len or !self.readEmbedding(&embedding, last.markov_w1, @bitCast(previous))) return false;
        const bytes = @as(u64, shape.n_vocab) * 4;
        const logits = c.ds4_gpu_tensor_view(state.logits, @as(u64, row) * bytes, bytes);
        defer c.ds4_gpu_tensor_free(logits);
        var ok = logits != null and c.ds4_gpu_begin_commands() != 0 and
            c.ds4_gpu_tensor_write(state.markov_embed, 0, &embedding, @sizeOf(@TypeOf(embedding))) != 0 and
            self.ops.matmul(state.markov_logits, last.markov_w2, state.markov_embed, false) and
            c.ds4_gpu_add_tensor(logits, logits, state.markov_logits, shape.n_vocab) != 0 and
            c.ds4_gpu_tensor_copy(state.confidence_input, 0, state.batch.x, @as(u64, row) * shape.n_embd * 4, @as(u64, shape.n_embd) * 4) != 0 and
            c.ds4_gpu_tensor_copy(state.confidence_input, @as(u64, shape.n_embd) * 4, state.markov_embed, 0, @as(u64, rank) * 4) != 0 and
            self.ops.matmul(state.confidence, last.confidence_proj, state.confidence_input, false);
        if (c.ds4_gpu_commands_active() != 0 and c.ds4_gpu_end_commands() == 0) ok = false;
        return ok and c.ds4_gpu_tensor_read(logits, 0, state.host_logits + @as(u64, row) * shape.n_vocab, bytes) != 0 and
            c.ds4_gpu_tensor_read(state.confidence, 0, confidence, @sizeOf(f32)) != 0;
    }

    fn readEmbedding(self: *const Drafter, output: []f32, pointer: [*c]const c.ds4_tensor, row: u32) bool {
        if (pointer == null or pointer.*.ndim != 2 or row >= pointer.*.dim[1]) return false;
        const weight: *const c.ds4_tensor = @ptrCast(pointer);
        const model = self.ops.model;
        if (weight.abs_offset > model.size or weight.bytes > model.size - weight.abs_offset)
            c.ds4_engine_model_die("tensor is outside the mapped weights; Engram requires disk row reads");
        const mapped: [*]const u8 = @ptrCast(model.map);
        const data = mapped + @as(usize, @intCast(weight.abs_offset));
        const width = weight.dim[0];
        switch (weight.type) {
            c.DS4_TENSOR_F32 => {
                const start = @as(u64, row) * width * 4;
                for (output[0..@intCast(width)], 0..) |*value, index|
                    value.* = std.mem.bytesToValue(f32, data[start + index * 4 ..][0..4]);
            },
            c.DS4_TENSOR_F16 => {
                const start = @as(u64, row) * width * 2;
                for (output[0..@intCast(width)], 0..) |*value, index| {
                    const bits = std.mem.bytesToValue(u16, data[start + index * 2 ..][0..2]);
                    value.* = @floatCast(@as(f16, @bitCast(bits)));
                }
            },
            c.DS4_TENSOR_Q8_0 => {
                const blocks = (width +% 31) / 32;
                const start = @as(u64, row) * blocks * 34;
                for (0..@as(usize, @intCast(blocks))) |block| {
                    const bytes = data + start + block * 34;
                    const bits = std.mem.bytesToValue(u16, bytes[0..2]);
                    const scale: f32 = @floatCast(@as(f16, @bitCast(bits)));
                    const first = block * 32;
                    const count: usize = @intCast(@min(width - first, 32));
                    for (output[first..][0..count], bytes[2..][0..count]) |*value, quantized|
                        value.* = scale * @as(f32, @floatFromInt(@as(i8, @bitCast(quantized))));
                }
            },
            else => return false,
        }
        return true;
    }
};

// Appends beyond an accepted frontier are invisible. Save only the six
// overwritten raw-ring/feature slots and pooling carries for rollback.
pub fn verify(state: *Record, executor: Executor, transport: ?*c.ds4_tp, tokens: []const c_int) bool {
    const graph = executor.graph;
    const config = executor.ops.config;
    const shape = &config.shape;
    const policy = Policy{ .config = config, .graph = graph, .weights = executor.weights };
    if (!graph.valid or tokens.len < 2 or tokens.len > verify_rows or graph.pos > graph.ctx or
        tokens.len > graph.ctx - graph.pos or !policy.tpBatchEnabled()) return false;
    const count: u32 = @intCast(tokens.len);
    const tp_block = graph.tp_world == 2;
    var gates: u32 = 2;
    // Prepost the complete small-row exchange window. Per-layer bulk
    // prefill handshakes are unnecessarily expensive for verification.
    if (tp_block and (graph.tp_batch_out == null or graph.tp_batch_in == null or transport == null or
        c.ds4_tp_batch_block_begin_gates(transport, count, shape.n_layer, &gates) == 0 or gates != 2)) return false;
    const kv = @as(u64, shape.n_head_dim) * 4;
    const feature = @as(u64, shape.n_embd) * 4;
    var graphs: [verify_rows]*Graph = undefined;
    var ok = c.ds4_gpu_begin_commands() != 0;
    state.verify_start = graph.pos;
    state.verify_count = 0;
    state.rewind_end = 0;
    var index: u32 = 0;
    while (ok and index <= verify_rows) : (index += 1) {
        if (index >= count and index != verify_rows) continue;
        const slot = (graph.pos + index) % window;
        var layer: u32 = 0;
        while (ok and index < count and layer < 40) : (layer += 1)
            ok = c.ds4_gpu_tensor_copy(state.old_raw, (@as(u64, layer) * verify_rows + index) * kv, graph.window[layer], @as(u64, slot) * kv, kv) != 0;
        var stage_index: u32 = 0;
        while (ok and index < count and stage_index < 3) : (stage_index += 1) {
            state.old_positions[stage_index][index] = state.positions[stage_index][slot];
            ok = c.ds4_gpu_tensor_copy(state.old_features, (@as(u64, stage_index) * verify_rows + index) * feature, state.target[stage_index], @as(u64, slot) * feature, feature) != 0;
        }
        const row = &state.verified[index];
        const residual = row.residual;
        const pre = row.pre;
        const logits = row.logits;
        const half = row.tp_logits_half;
        const pool_keys = row.verify_previous_kv;
        const pool_scores = row.verify_previous_score;
        row.* = graph.*;
        row.residual = if (residual != null) residual else c.ds4_gpu_tensor_view(state.verify_residual, @as(u64, index) * shape.n_hc * feature, @as(u64, shape.n_hc) * feature);
        row.pre = if (pre != null) pre else c.ds4_gpu_tensor_view(state.verify_pre, @as(u64, index) * shape.n_hc * 4, @as(u64, shape.n_hc) * 4);
        row.logits = if (logits != null) logits else c.ds4_gpu_tensor_view(state.verify_logits, @as(u64, index) * shape.n_vocab * 4, @as(u64, shape.n_vocab) * 4);
        row.tp_logits_half = null;
        if (graph.tp_logits_half != null) {
            const bytes = @as(u64, shape.n_vocab) * 2;
            row.tp_logits_half = if (half != null) half else c.ds4_gpu_tensor_view(row.logits, @as(u64, graph.tp_rank) * bytes, bytes);
            ok = ok and row.tp_logits_half != null;
        }
        row.verify_prefix = true;
        for (0..4) |owner| {
            const offset = (@as(u64, index) * 8 + owner * 2) * kv;
            row.verify_previous_kv[owner] = if (pool_keys[owner] != null) pool_keys[owner] else c.ds4_gpu_tensor_view(state.verify_pool, offset, kv);
            row.verify_previous_score[owner] = if (pool_scores[owner] != null) pool_scores[owner] else c.ds4_gpu_tensor_view(state.verify_pool, offset + kv, kv);
            ok = ok and row.verify_previous_kv[owner] != null and row.verify_previous_score[owner] != null;
        }
        ok = ok and row.residual != null and row.pre != null and row.logits != null;
        if (index < count) {
            graphs[index] = row;
        } else {
            // Save the block start too: exact sampling can switch modes
            // before consuming even the first verified token.
            ok = ok and c.ds4_gpu_tensor_copy(row.residual, 0, graph.residual, 0, @as(u64, shape.n_hc) * feature) != 0 and
                c.ds4_gpu_tensor_copy(row.pre, 0, graph.pre, 0, @as(u64, shape.n_hc) * 4) != 0 and
                c.ds4_gpu_tensor_copy(row.logits, 0, graph.logits, 0, @as(u64, shape.n_vocab) * 4) != 0;
            var owner: usize = 0;
            while (ok and owner < 4) : (owner += 1)
                ok = c.ds4_gpu_tensor_copy(row.verify_previous_kv[owner], 0, graph.previous_kv[owner], 0, kv) != 0 and
                    c.ds4_gpu_tensor_copy(row.verify_previous_score[owner], 0, graph.previous_score[owner], 0, kv) != 0;
        }
    }
    const session_batch = SessionBatch{ .executor = executor };
    if (ok) ok = session_batch.run(graphs[0..count], tokens, count);
    if (c.ds4_gpu_commands_active() != 0 and c.ds4_gpu_end_commands() == 0) ok = false;
    if (tp_block and (!ok or c.ds4_tp_batch_block_end(transport) == 0)) {
        c.ds4_tp_mark_failed(transport);
        ok = false;
    }
    if (ok) ok = c.ds4_gpu_tensor_read(state.verify_logits, 0, state.host_verified, @as(u64, count) * shape.n_vocab * 4) != 0 and
        c.ds4_gpu_tensor_read(state.verify_logits, @as(u64, verify_rows) * shape.n_vocab * 4, state.host_verified + @as(u64, verify_rows) * shape.n_vocab, @as(u64, shape.n_vocab) * 4) != 0;
    if (ok) state.verify_count = count else graph.valid = false;
    return ok;
}

pub fn commit(config: *const Config, state: *Record, graph: *Graph, keep: u32, output: [*]f32) bool {
    const shape = &config.shape;
    if (state.verify_count == 0 or keep > state.verify_count or
        (graph.pos != state.verify_start and graph.pos != state.rewind_end) or
        (state.rewind_end != 0 and state.verify_start + keep > state.rewind_end)) return false;
    const kv = @as(u64, shape.n_head_dim) * 4;
    const feature = @as(u64, shape.n_embd) * 4;
    const row = if (keep != 0) keep - 1 else verify_rows;
    const last = &state.verified[row];
    var ok = c.ds4_gpu_begin_commands() != 0;
    var index = keep;
    while (ok and index < state.verify_count) : (index += 1) {
        const slot = (state.verify_start + index) % window;
        var layer: usize = 0;
        while (ok and layer < 40) : (layer += 1)
            ok = c.ds4_gpu_tensor_copy(graph.window[layer], @as(u64, slot) * kv, state.old_raw, (layer * verify_rows + index) * kv, kv) != 0;
        var stage_index: usize = 0;
        while (ok and stage_index < 3) : (stage_index += 1) {
            ok = c.ds4_gpu_tensor_copy(state.target[stage_index], @as(u64, slot) * feature, state.old_features, (stage_index * verify_rows + index) * feature, feature) != 0;
            state.positions[stage_index][slot] = state.old_positions[stage_index][index];
        }
    }
    var owner: usize = 0;
    while (ok and owner < 4) : (owner += 1)
        ok = c.ds4_gpu_tensor_copy(graph.previous_kv[owner], 0, last.verify_previous_kv[owner], 0, kv) != 0 and
            c.ds4_gpu_tensor_copy(graph.previous_score[owner], 0, last.verify_previous_score[owner], 0, kv) != 0;
    if (ok) ok = c.ds4_gpu_tensor_copy(graph.residual, 0, last.residual, 0, @as(u64, shape.n_hc) * feature) != 0 and
        c.ds4_gpu_tensor_copy(graph.pre, 0, last.pre, 0, @as(u64, shape.n_hc) * 4) != 0 and
        c.ds4_gpu_tensor_copy(graph.logits, 0, last.logits, 0, @as(u64, shape.n_vocab) * 4) != 0;
    if (c.ds4_gpu_commands_active() != 0 and c.ds4_gpu_end_commands() == 0) ok = false;
    if (ok) {
        graph.pos = last.pos;
        graph.history = last.history;
        const source = state.host_verified + @as(u64, row) * shape.n_vocab;
        @memcpy(output[0..shape.n_vocab], source[0..shape.n_vocab]);
    } else graph.valid = false;
    state.rewind_end = if (ok) graph.pos else 0;
    return ok;
}
