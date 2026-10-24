//! Small multi-session V4.1 batches and same-session short-prefill reuse.
//! Attention/history stay private to each graph; only the largest graph lends
//! its transient batch workspace. No borrowed graph/view becomes an owner.
const std = @import("std");
const c = @import("gguf_c");
const Executor = @import("executor.zig").Executor;
const Prefill = @import("prefill.zig").Prefill;
const Attention = @import("attention.zig").Attention;
const BatchAttention = @import("batch_attention.zig").BatchAttention;
const buffers = @import("prefill_buffers.zig");
const Graph = c.ds41_gpu_graph;
const Tensor = c.ds4_gpu_tensor;
const max_rows = c.DS4_TP_BATCH_MAX_ROWS;
const DiskRows = [2][c.DS4_ENGRAM_COLS * c.DS4_ENGRAM_DIM]f32;

pub const SessionBatch = struct {
    executor: Executor,

    pub fn workspace(graphs: []const *Graph) ?*Graph {
        if (graphs.len < 2 or graphs.len > max_rows) return null;
        var largest = graphs[0];
        for (graphs[1..]) |graph| {
            if (graph.steering_mode != graphs[0].steering_mode or graph.steering_layers != graphs[0].steering_layers or graph.steering_attn_scale != graphs[0].steering_attn_scale or
                graph.steering_ffn_scale != graphs[0].steering_ffn_scale) return null;
            if (graph.ctx > largest.ctx or (graph.ctx == largest.ctx and graph.prefill_cap > largest.prefill_cap)) largest = graph;
        }
        return if (largest.prefill_cap >= graphs.len) largest else null;
    }

    pub fn run(self: *const SessionBatch, graphs: []const *Graph, tokens: []const c_int, prefill_rows: u32) bool {
        if (graphs.len != tokens.len or graphs.len < 2 or graphs.len > max_rows or prefill_rows > graphs.len) return false;
        const graph = workspace(graphs) orelse return false;
        const rows: u32 = @intCast(graphs.len);
        var executor = self.executor;
        executor.graph = graph;
        const model = executor.ops.model;
        const weights = executor.weights;
        const shape = &executor.ops.config.shape;
        const prefill_only = prefill_rows == rows and !graph.verify_prefix;
        const shared_owner = graph.tp_world == 2 and c.getenv("DS4_METAL_DISABLE_V41_TP_SHARED_OWNER") == null;
        var history: [max_rows]c.ds4_engram_history = undefined;
        var positions: [max_rows]u32 = undefined;
        // Only leading prefill rows share a session. Keep their sparse reads
        // and hash histories distinct until the final row is published.
        const allocator = std.heap.c_allocator;
        const engram: ?[]DiskRows = if (prefill_rows > 1) allocator.alloc(DiskRows, prefill_rows) catch null else null;
        defer if (engram) |memory| allocator.free(memory);
        var active = buffers.Views{};
        defer active.deinit();
        var queries: [max_rows]?*Tensor = @splat(null);
        var heads: [max_rows]?*Tensor = @splat(null);
        defer for (queries, heads) |query, head| {
            c.ds4_gpu_tensor_free(query);
            c.ds4_gpu_tensor_free(head);
        };
        var ok = rows <= graph.prefill_cap and (prefill_rows <= 1 or engram != null);
        if (ok) {
            if (buffers.Views.init(executor.ops.config, graph, rows)) |views| active = views else ok = false;
        }
        const head_bytes = @as(u64, shape.n_head / graph.tp_world) * shape.n_head_dim * 4;
        for (0..rows) |index| {
            if (!ok) break;
            queries[index] = c.ds4_gpu_tensor_view(graph.batch.q, @as(u64, index) * head_bytes, head_bytes);
            heads[index] = c.ds4_gpu_tensor_view(graph.batch.heads, @as(u64, index) * head_bytes, head_bytes);
            ok = queries[index] != null and heads[index] != null;
        }
        const initial_pre = [_]f32{ 1, 0, 0, 0 };
        for (graphs, tokens, 0..) |session, token, index| {
            if (!ok) break;
            var ids: [2][c.DS4_ENGRAM_COLS]u32 = undefined;
            const continued = index > 0 and index < prefill_rows;
            positions[index] = session.pos + if (continued) @as(u32, @intCast(index)) else 0;
            history[index] = if (continued) history[index - 1] else session.history;
            ok = c.ds4_engram_hash(&session.engram, &history[index], &token, null, 1, @ptrCast(&ids));
            const disk_rows: *DiskRows = if (engram != null and index < prefill_rows) &engram.?[index] else &session.rows;
            for (0..2) |table| {
                if (!ok) break;
                ok = c.ds4_engram_read(&session.table[table], &ids[table], c.DS4_ENGRAM_COLS, &disk_rows[table]);
            }
            if (ok) ok = c.ds4_gpu_tensor_write(graph.rows_view[index].pre, 0, &initial_pre, @sizeOf(@TypeOf(initial_pre))) != 0 and
                c.ds4_gpu_embed_token_hc_tensor(graph.rows_view[index].residual, model.map, model.size, weights.token_embd.*.abs_offset, shape.n_vocab, @bitCast(token), shape.n_embd, shape.n_hc) != 0;
        }
        if (ok) ok = c.ds4_gpu_tensor_write(graph.prefill_tokens, 0, tokens.ptr, @as(u64, rows) * @sizeOf(c_int)) != 0;
        var layer_index: u32 = 0;
        while (ok and layer_index < shape.n_layer) : (layer_index += 1) {
            const layer = &weights.layer[layer_index];
            if (executor.ops.config.layerPolicy(layer_index).?.uses_engram) {
                const table: usize = if (layer_index == 1) 0 else 1;
                for (graphs, 0..) |session, index| {
                    if (!ok) break;
                    const source = if (engram != null and index < prefill_rows) &engram.?[index][table] else &session.rows[table];
                    ok = c.ds4_gpu_tensor_write(graph.rows_view[index].engram_rows, 0, source, @sizeOf(@TypeOf(session.rows[table]))) != 0;
                }
            }
            for (graphs, 0..) |session, index| {
                if (!ok) break;
                var local = executor;
                local.graph = session;
                ok = local.capture(layer_index, positions[index], graph.rows_view[index].residual, 1);
            }
            const batch = BatchAttention{ .ops = executor.ops, .graph = graph };
            if (ok) ok = Prefill.beforeBatch(executor, &active.row, layer, layer_index, rows) and batch.project(layer, rows);
            const batch_output = layer.attn_output_a.*.type == c.DS4_TENSOR_Q8_0 and
                layer.attn_output_b.*.type == c.DS4_TENSOR_Q8_0 and c.getenv("DS4_DISABLE_V41_BATCH_OUTPUT") == null;
            for (graphs, 0..) |session, index| {
                if (!ok) break;
                var row = session.*;
                row.pos = positions[index];
                buffers.selectRow(&row, @ptrCast(&graph.rows_view[index]));
                row.q = queries[index];
                row.heads = heads[index];
                const attention = Attention{ .config = executor.ops.config, .model = model, .graph = &row };
                ok = attention.run(layer, layer_index, true) and (batch_output or attention.output(layer));
            }
            if (ok and batch_output) ok = (if (graph.tp_world == 2)
                c.ds4_gpu_dsv41_attention_output_tp_batch(active.row.block, active.row.low, model.map, model.size, layer.attn_output_a.*.abs_offset, layer.attn_output_b.*.abs_offset, graph.batch.heads, rows, graph.tp_rank)
            else
                c.ds4_gpu_dsv41_attention_output_batch(active.row.block, active.row.low, model.map, model.size, layer.attn_output_a.*.abs_offset, layer.attn_output_b.*.abs_offset, graph.batch.heads, rows)) != 0;
            if (ok) ok = executor.ops.sumPartialBatch(graph, active.row.block, layer_index, c.DS4_TP_GATE_ATTN, rows);
            const moe = executor.moe();
            if (ok) ok = c.ds4_gpu_dsv41_quantize(active.row.block, shape.n_embd, rows, c.DS4_V41_BF16) != 0 and
                Prefill.afterBatch(executor, &active.row, layer, layer_index, rows) and moe.batch(layer, layer_index, rows, shared_owner, true) and
                (if (shared_owner)
                    c.ds4_gpu_tensor_copy(active.row.block, 0, active.row.routed, 0, @as(u64, rows) * shape.n_embd * 4)
                else
                    c.ds4_gpu_add_tensor(active.row.block, active.row.routed, active.row.shared, rows * shape.n_embd)) != 0 and
                c.ds4_gpu_dsv41_quantize(active.row.block, shape.n_embd, rows, c.DS4_V41_BF16) != 0 and
                executor.expand(active.row.residual, active.row.block, active.row.after_attn, active.row.ffn_split, layer_index, rows, true);
            // The second Engram upload reuses the first one's input storage.
            if (ok and layer_index == 13) ok = c.ds4_gpu_end_commands() != 0 and c.ds4_gpu_begin_commands() != 0;
        }
        const logits_bytes = @as(u64, shape.n_vocab) * 4 / @as(u64, if (graph.tp_logits_half != null) 2 else 1);
        var batch_logits: ?*Tensor = null;
        defer c.ds4_gpu_tensor_free(batch_logits);
        // Final FFN intermediates are dead; lend them to the output head
        // instead of allocating vocabulary-sized scratch for every session.
        if (ok and !prefill_only and c.getenv("DS4_METAL_DISABLE_V41_BATCH_HEAD") == null and
            c.ds4_gpu_tensor_bytes(graph.batch.mid) >= @as(u64, rows) * logits_bytes)
            batch_logits = c.ds4_gpu_tensor_view(graph.batch.mid, 0, @as(u64, rows) * logits_bytes);
        // Pure prefill publishes its final row only. Its caller computes the
        // output head once at the complete prompt frontier.
        const first_output: u32 = if (prefill_only) rows - 1 else 0;
        var index = first_output;
        while (ok and index < rows) : (index += 1) {
            const session = graphs[index];
            ok = c.ds4_gpu_tensor_copy(session.residual, 0, graph.rows_view[index].residual, 0, @as(u64, shape.n_hc) * shape.n_embd * 4) != 0 and
                c.ds4_gpu_tensor_copy(session.pre, 0, graph.rows_view[index].ffn_split, 0, @as(u64, shape.n_hc) * 4) != 0;
            if (!ok or prefill_only) continue;
            ok = executor.ops.hcSum(session.x, session.residual, session.pre, false) and
                executor.ops.norm(session.norm, session.x, weights.output_norm);
            if (ok) ok = if (batch_logits != null)
                c.ds4_gpu_tensor_copy(graph.rows_view[index].norm, 0, session.norm, 0, @as(u64, shape.n_embd) * 4) != 0
            else
                executor.ops.outputProjection(session, if (session.tp_logits_half != null) session.tp_logits_half else session.logits, weights, session.norm, 1);
        }
        if (ok and batch_logits != null) ok = executor.ops.outputProjection(graph, batch_logits, weights, active.row.norm, rows);
        index = first_output;
        while (ok and index < rows) : (index += 1) {
            const session = graphs[index];
            if (batch_logits != null) ok = c.ds4_gpu_tensor_copy(if (session.tp_logits_half != null) session.tp_logits_half else session.logits, 0, batch_logits, @as(u64, index) * logits_bytes, logits_bytes) != 0;
            if (ok) {
                session.history = history[index];
                session.pos = positions[index] + 1;
            }
        }
        if (!ok) _ = c.fprintf(c.__stderrp, "ds4: V4.1 session batch failed at layer %u (%d rows)\n", layer_index, @as(c_int, @intCast(rows)));
        return ok;
    }

    pub fn shortPrefill(self: *const SessionBatch, tokens: []const c_int) bool {
        const graph = self.executor.graph;
        if (!graph.valid or tokens.len < 2 or tokens.len > max_rows or graph.pos > graph.ctx or tokens.len > graph.ctx - graph.pos) return false;
        var graphs: [max_rows]*Graph = undefined;
        for (graphs[0..tokens.len]) |*entry| entry.* = graph;
        var ok = c.ds4_gpu_begin_commands() != 0 and self.run(graphs[0..tokens.len], tokens, @intCast(tokens.len));
        if (ok) ok = c.ds4_gpu_end_commands() != 0 else _ = c.ds4_gpu_synchronize();
        if (!ok) graph.valid = false;
        return ok;
    }
};
