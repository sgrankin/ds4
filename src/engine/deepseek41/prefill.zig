//! Layer-major V4.1 prefill. Config, numerical operations, cache publication,
//! expert I/O and execution are explicit owners. A partial failure leaves the
//! graph invalid: its layers have different frontiers and cannot be exported.
const c = @import("gguf_c");
const Routes = @import("route_services.zig");
const Executor = @import("executor.zig").Executor;
const Attention = @import("attention.zig").Attention;
const BatchAttention = @import("batch_attention.zig").BatchAttention;
const Policy = @import("prefill_policy.zig").Policy;
const replaySelected = @import("prefill_policy.zig").replaySelected;
const Prefetch = @import("engram_prefetch.zig").Prefetch;
const Streaming = @import("streaming.zig").Streaming;
const buffers = @import("prefill_buffers.zig");
const cache = @import("expert_cache.zig");
const support = @import("ops.zig");
const Graph = c.ds41_gpu_graph;
const Layer = c.ds4_layer_weights;
const Row = c.ds41_prefill_row;
const Model = c.ds4_model;
const Weights = c.ds4_weights;

pub const Callbacks = struct {
    progress: c.ds4_session_progress_fn = null,
    progress_userdata: ?*anyopaque = null,
    total: c_int = 0,
    cancel: c.ds4_session_cancel_fn = null,
    cancel_userdata: ?*anyopaque = null,

    fn cancelled(self: Callbacks) bool {
        return if (self.cancel) |check| check(self.cancel_userdata) else false;
    }
};
fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn nowSeconds() f64 {
    var time: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &time);
    const seconds: f64 = @floatFromInt(time.tv_sec);
    const nanoseconds: f64 = @floatFromInt(time.tv_nsec);
    return seconds + nanoseconds * 1.0e-9;
}
const Stage = struct {
    enabled: bool,
    started: f64,
    layer: u32,
    position: u32,
    rows: u32,

    fn mark(self: *Stage, ok: *bool, name: [*:0]const u8) void {
        if (!ok.* or !self.enabled) return;
        ok.* = c.ds4_gpu_end_commands() != 0;
        const now = nowSeconds();
        _ = c.fprintf(c.__stderrp, "ds4: V4.1 stage layer=%u pos=%u rows=%u %s=%.3f ms\n", self.layer, self.position, self.rows, name, (now - self.started) * 1000);
        self.started = now;
        if (ok.*) ok.* = c.ds4_gpu_begin_commands() != 0;
    }
};

pub const Prefill = struct {
    executor: Executor,

    fn attention(executor: Executor, graph: *Graph) Attention {
        return .{ .config = executor.ops.config, .model = executor.ops.model, .graph = graph };
    }
    fn rowExecutor(executor: Executor, graph: *Graph) Executor {
        var result = executor;
        result.graph = graph;
        return result;
    }

    pub fn beforeBatch(executor: Executor, row: *Row, layer: *const Layer, index: u32, count: u32) bool {
        const graph = executor.graph;
        const shape = &executor.ops.config.shape;
        if (executor.ops.config.layerPolicy(index).?.uses_engram) {
            const table: usize = if (index == 1) 0 else 1;
            if (!executor.ops.matmulBatch(row.engram_kv, layer.engram_kv, row.engram_rows, count, true) or
                c.ds4_gpu_dsv41_engram_add(row.residual, row.engram_kv, graph.engram_q_norm[table], graph.engram_k_norm[table], if (graph.image_count != 0) graph.image_text_mask else null, shape.n_embd, count, shape.rms_eps) == 0) return false;
        }
        if (!executor.ops.hcMixBatch(row, layer, false, count)) return false;
        // V4.1 uses the preceding sublayer's mixer, not the newly computed one.
        const mixed = if (index != 0)
            c.ds4_gpu_hc_weighted_sum_split_tensor(row.x, row.residual, row.ffn_split, shape.n_embd, shape.n_hc)
        else
            c.ds4_gpu_hc_weighted_sum_tensor(row.x, row.residual, row.pre, shape.n_embd, shape.n_hc);
        return mixed != 0 and c.ds4_gpu_dsv41_quantize(row.x, shape.n_embd, count, c.DS4_V41_BF16) != 0 and
            executor.ops.normBatch(row.norm, row.x, layer.attn_norm, count);
    }

    pub fn afterBatch(executor: Executor, row: *Row, layer: *const Layer, count: u32) bool {
        const shape = &executor.ops.config.shape;
        return c.ds4_gpu_hc_expand_split_tensor(row.after_attn, row.block, row.residual, row.attn_split, shape.n_embd, shape.n_hc) != 0 and
            c.ds4_gpu_dsv41_quantize(row.after_attn, shape.n_embd * shape.n_hc, count, c.DS4_V41_BF16) != 0 and
            executor.ops.hcMixBatch(row, layer, true, count) and
            c.ds4_gpu_hc_weighted_sum_split_tensor(row.x, row.after_attn, row.attn_split, shape.n_embd, shape.n_hc) != 0 and
            c.ds4_gpu_dsv41_quantize(row.x, shape.n_embd, count, c.DS4_V41_BF16) != 0 and
            executor.ops.normBatch(row.norm, row.x, layer.ffn_norm, count);
    }

    // Publish all encoder compressed keys, but warm only the decoder's local
    // raw dependency suffix. No preceding decoder outputs are needed.
    fn decoderPrepare(executor: Executor, layer: *const Layer, index: u32, initial_start: u32, offset: u32, rows: u32, publish: bool, split_pre: bool, batch_attention: bool, callbacks: Callbacks) bool {
        const graph = executor.graph;
        const shape = &executor.ops.config.shape;
        var row = graph.*;
        const original_attention = attention(executor, graph);
        var ok = true;
        const batch_publish = publish and batch_attention and !present("DS4_METAL_DISABLE_V41_BATCH_PUBLISH");
        const chunk: u32 = @min(graph.prefill_cap, 2048);
        var first = offset;
        while (ok and first < offset + rows) : (first += chunk) {
            if (callbacks.cancelled()) return false;
            const count: u32 = @min(offset + rows - first, chunk);
            var active = buffers.Views{};
            if (buffers.Views.init(executor.ops.config, graph, count)) |views| active = views else ok = false;
            defer active.deinit();
            if (ok) ok = c.ds4_gpu_begin_commands() != 0;
            if (ok) ok = buffers.carryCopy(executor.ops.config, graph, first, count, false);
            if (ok and batch_attention) ok = (if (split_pre)
                c.ds4_gpu_hc_weighted_sum_split_tensor(active.row.x, active.row.residual, active.row.ffn_split, shape.n_embd, shape.n_hc)
            else
                c.ds4_gpu_hc_weighted_sum_tensor(active.row.x, active.row.residual, active.row.pre, shape.n_embd, shape.n_hc)) != 0;
            if (ok and batch_attention) ok = c.ds4_gpu_dsv41_quantize(active.row.x, shape.n_embd, count, c.DS4_V41_BF16) != 0 and
                executor.ops.normBatch(active.row.norm, active.row.x, layer.attn_norm, count);
            const batch = BatchAttention{ .ops = executor.ops, .graph = graph };
            if (ok and batch_publish) ok = batch.publish(&active.row, layer, index, initial_start + first, count);
            if (ok and !publish and batch_attention) ok =
                executor.ops.matmulBatch(active.row.kv, layer.attn_kv, active.row.norm, count, true) and
                executor.ops.normBatch(active.row.kv, active.row.kv, layer.attn_kv_a_norm, count) and
                original_attention.archiveRaw(index, active.row.kv, initial_start + first, count) and
                c.ds4_gpu_dsv41_rope(active.row.kv, shape.n_head_dim, 1, count, initial_start + first, true, false) != 0 and
                c.ds4_gpu_dsv41_quantize(active.row.kv, shape.n_head_dim, count, c.DS4_V41_FP8_E8M0) != 0;
            var token: u32 = 0;
            while (ok and !batch_publish and token < count) : (token += 1) {
                row.pos = initial_start + first + token;
                row.norm = graph.rows_view[token].norm;
                row.kv = graph.rows_view[token].kv;
                const row_executor = rowExecutor(executor, &row);
                if (!batch_attention) {
                    row.residual = graph.rows_view[token].residual;
                    row.pre = graph.rows_view[token].pre;
                    ok = row_executor.beforeAttention(layer, index);
                }
                const local_attention = attention(executor, &row);
                if (ok and publish) {
                    ok = local_attention.publish(layer, index);
                } else if (ok) {
                    if (!batch_attention) ok = executor.ops.matmul(row.kv, layer.attn_kv, row.norm, true) and
                        executor.ops.norm(row.kv, row.kv, layer.attn_kv_a_norm) and
                        original_attention.archiveRaw(index, row.kv, row.pos, 1) and
                        executor.ops.rope(row.kv, 1, shape.n_head_dim, index, row.pos, false) and
                        c.ds4_gpu_dsv41_quantize(row.kv, shape.n_head_dim, 1, c.DS4_V41_FP8_E8M0) != 0;
                    if (ok) ok = c.ds4_gpu_tensor_copy(graph.window[index], @as(u64, row.pos % 128) * shape.n_head_dim * 4, row.kv, 0, @as(u64, shape.n_head_dim) * 4) != 0;
                }
            }
            if (c.ds4_gpu_commands_active() != 0 and c.ds4_gpu_end_commands() == 0) ok = false;
        }
        return ok;
    }

    pub fn run(self: *const Prefill, tokens: []const c_int, callbacks: Callbacks) bool {
        return self.sweep(tokens, callbacks, false, false);
    }

    pub fn sweep(self: *const Prefill, tokens: []const c_int, callbacks: Callbacks, encoder_only: bool, resume_encoder: bool) bool {
        const graph = self.executor.graph;
        const weights = self.executor.weights;
        const model = self.executor.ops.model;
        const config = self.executor.ops.config;
        const shape = &config.shape;
        const total_count: u32 = @intCast(tokens.len);
        const policy = Policy{ .config = config, .graph = graph, .weights = weights };
        const encoder_chunk = policy.encoderChunk(total_count);
        const wide = total_count > encoder_chunk;
        if ((!graph.valid and !resume_encoder) or total_count == 0 or (wide and total_count > graph.carry_cap) or
            total_count > graph.ctx - graph.pos or graph.imatrix != null or !policy.tpBatchEnabled()) return false;
        const profile = present("DS4_METAL_GRAPH_PREFILL_PROFILE");
        const stage_profile = present("DS4_METAL_V41_STAGE_PROFILE");
        const selected_small = policy.selectedSmall(total_count);
        if (!(Routes.Transaction{ .ops = self.executor.ops, .graph = graph }).beginPrefill(weights, selected_small)) return false;
        const exact_short = selected_small or policy.exactShort(total_count);
        const batch_moe = (!exact_short or (selected_small and !present("DS4_METAL_DISABLE_V41_SELECTED_SMALL_MOE"))) and
            !present("DS4_METAL_DISABLE_V41_BATCH_MOE");
        const batch_attention = !present("DS4_METAL_DISABLE_V41_BATCH_ATTN");
        const exact_moe = exact_short and !selected_small and batch_attention and present("DS4_METAL_V41_EXACT_MOE_TILES");
        const batch_core = !exact_short and batch_attention and !present("DS4_METAL_DISABLE_V41_BATCH_CORE");
        const batch_hc = batch_attention and (batch_moe or (exact_short and
            (selected_small or present("DS4_METAL_V41_EXACT_BATCH_HC")))) and !present("DS4_METAL_DISABLE_V41_BATCH_HC");
        const decoder_suffix = wide and total_count >= 8192 and !present("DS4_METAL_DISABLE_V41_DECODER_SUFFIX");
        const replay_rows = policy.replayRows(total_count);
        const replay_selected_requested = replay_rows != 0 and replaySelected();
        var replay_selected: [c.DS4_MAX_LAYER]bool = @splat(false);
        var replay_selected_layers: u32 = 0;
        if (replay_selected_requested and graph.streaming and batch_moe) {
            for (20..shape.n_layer) |index| {
                replay_selected[index] = c.ds4_gpu_stream_expert_batch_supported(replay_rows, shape.n_expert, shape.n_expert_used, weights.layer[index].ffn_gate_exps.*.type, weights.layer[index].ffn_down_exps.*.type) != 0;
                replay_selected_layers += @intFromBool(replay_selected[index]);
            }
        }
        if ((encoder_only or resume_encoder) and !decoder_suffix) return false;
        var next_history = graph.history;
        if (!self.executor.ops.hashTokens(graph, &next_history, tokens.ptr, total_count, @ptrCast(graph.prefill_ids))) return false;
        var executor = self.executor;
        // Precision scope belongs to this inference call; never mutate a
        // process-wide flag while another session may be running.
        executor.ops.exact_prefill_rows = executor.ops.exact_prefill_rows or exact_short;
        const initial_start = graph.pos;
        const replay_floor = if (replay_rows != 0) initial_start + total_count - replay_rows else 0;
        graph.valid = false;
        var row = graph.*;
        row.streaming = selected_small;
        var prefetch = Prefetch{};
        const overlap_engram = (total_count >= 1024 or (selected_small and present("DS4_METAL_V41_SMALL_ENGRAM_PREFETCH"))) and
            !present("DS4_METAL_DISABLE_V41_ENGRAM_PREFETCH") and !present("DS4_METAL_DISABLE_V41_BATCH_ENGRAM");
        const pipeline_engram = overlap_engram and !present("DS4_METAL_DISABLE_V41_ENGRAM_PIPELINE");
        var prefetched = overlap_engram and prefetch.start(graph, 0, total_count);
        const prepare_count: u32 = if (present("DS4_METAL_V41_PREFETCH_TWO_LAYERS")) 2 else 1;
        var streaming = Streaming.init(model, weights, config, prepare_count);
        var ok = selected_small or !graph.streaming or streaming.mapToken();
        var replay_static_map = false;
        for (0..shape.n_layer) |layer_offset| {
            if (!ok) break;
            const index: u32 = @intCast(layer_offset);
            const layer: *const Layer = &weights.layer[index];
            if (callbacks.cancelled()) {
                ok = false;
                break;
            }
            const layer_selected = selected_small or replay_selected[index];
            row.streaming = layer_selected;
            graph.decoder_replay_floor = if (replay_rows != 0 and index >= 20) replay_floor else 0;
            row.decoder_replay_floor = graph.decoder_replay_floor;
            if (encoder_only and index == 20) {
                // The last sweep must rebuild the exact dependency suffix;
                // publishing encoder keys alone does not validate the decoder.
                if (graph.streaming) ok = streaming.mapLayerDecode(index);
                if (ok) ok = decoderPrepare(executor, layer, index, initial_start, 0, total_count, true, batch_hc, batch_attention, callbacks);
                break;
            }
            // Layer1 no longer borrows prefix output. Fill it for layer14
            // while the intervening encoder layers execute.
            if (index == 2) prefetched = overlap_engram and prefetch.start(graph, 1, total_count);
            const map_started = if (profile) nowSeconds() else 0;
            const first_count: u32 = @min(total_count, encoder_chunk);
            if (graph.streaming and replay_selected[index] and !replay_static_map) {
                ok = streaming.mapStatic();
                replay_static_map = ok;
            }
            if (graph.streaming and !layer_selected) {
                replay_static_map = false;
                if (!graph.encoder_resident or index >= 20) ok = streaming.prepareJoin(index, first_count);
                if (ok) ok = streaming.mapLayer(index);
                var ahead: u32 = 1;
                while (ok and ahead <= prepare_count and index + ahead < (if (encoder_only) @as(u32, 20) else shape.n_layer) and
                    (replay_selected_layers == 0 or index >= 20 or index + ahead < 20)) : (ahead += 1)
                {
                    if (replay_selected[index + ahead]) continue;
                    if (!graph.encoder_resident or index + ahead >= 20) ok = streaming.prepareStart(index + ahead, first_count);
                }
            }
            const mapped_at = if (profile) nowSeconds() else 0;
            var first: u32 = 0;
            if (ok and decoder_suffix and index >= 20) {
                if (index == 20) ok = decoderPrepare(executor, layer, index, initial_start, 0, total_count, true, batch_hc, batch_attention, callbacks);
                if (replay_rows != 0) {
                    first = total_count - replay_rows;
                } else {
                    const needed = (if (graph.dspark != null) @as(u32, 128) else 1) + (shape.n_layer - 1 - index) * 127;
                    first = total_count - needed;
                    if (ok) ok = decoderPrepare(executor, layer, index, initial_start, first - 127, 127, false, batch_hc, batch_attention, callbacks);
                }
            }
            // A shrinking decoder suffix keeps its established partitions,
            // even when the encoder uses larger aligned matrix tiles.
            const chunk: u32 = if (wide and index >= 20 and encoder_chunk > 2048) 2048 else encoder_chunk;
            var offset = first;
            while (ok and offset < total_count) : (offset += chunk) {
                if (callbacks.cancelled()) {
                    ok = false;
                    break;
                }
                const count: u32 = @min(total_count - offset, chunk);
                const start = initial_start + offset;
                graph.pos = start;
                var active = buffers.Views{};
                if (batch_hc) {
                    // HC kernels infer row count from actual view size,
                    // including the final short chunk.
                    if (buffers.Views.init(config, graph, count)) |views| active = views else ok = false;
                }
                defer active.deinit();
                if (ok) ok = c.ds4_gpu_tensor_write(graph.prefill_tokens, 0, tokens.ptr + offset, @as(u64, count) * @sizeOf(i32)) != 0;
                if (ok and graph.image_count != 0) {
                    if (c.ds4_gpu_tensor_contents(graph.image_text_mask)) |mask|
                        executor.ops.textMask(graph, start, count, @ptrCast(mask))
                    else
                        ok = false;
                }
                if (ok and index == 0 and batch_core) {
                    if (c.ds4_gpu_tensor_contents(graph.batch.selected_comp)) |memory| {
                        const selection: [*]u8 = @ptrCast(memory);
                        @memset(selection[0 .. @as(usize, count) * shape.n_indexer_top_k * @sizeOf(i32)], 0xff);
                    } else ok = false;
                }
                if (ok) ok = c.ds4_gpu_begin_commands() != 0;
                if (index == 0) {
                    const initial_pre = [_]f32{ 1, 0, 0, 0 };
                    var token: u32 = 0;
                    while (ok and token < count) : (token += 1) {
                        ok = c.ds4_gpu_tensor_write(graph.rows_view[token].pre, 0, &initial_pre, @sizeOf(@TypeOf(initial_pre))) != 0 and
                            executor.ops.embed(graph, weights, graph.rows_view[token].residual, graph.rows_view[token].x, tokens[offset + token], start + token);
                    }
                } else if (wide) {
                    if (ok) ok = buffers.carryCopy(config, graph, offset, count, false);
                }
                if (c.ds4_gpu_commands_active() != 0 and c.ds4_gpu_end_commands() == 0) ok = false;
                const ready_at = if (profile) nowSeconds() else 0;
                const uses_engram = config.layerPolicy(index).?.uses_engram;
                if (ok and uses_engram) {
                    const table: usize = if (index == 1) 0 else 1;
                    if (prefetched) {
                        ok = if (pipeline_engram) prefetch.wait(offset + count, callbacks.cancel, callbacks.cancel_userdata) else prefetch.join(false);
                    } else if (present("DS4_METAL_DISABLE_V41_BATCH_ENGRAM")) {
                        var token: u32 = 0;
                        while (ok and token < count) : (token += 1) {
                            ok = c.ds4_engram_read(&graph.table[table], &graph.prefill_ids[offset + token][table][0], c.DS4_ENGRAM_COLS, @ptrCast(@alignCast(c.ds4_gpu_tensor_contents(graph.rows_view[token].engram_rows))));
                        }
                    } else {
                        ok = c.ds4_engram_read_batch(&graph.table[table], &graph.prefill_ids[offset][table][0], count, 2 * c.DS4_ENGRAM_COLS, @ptrCast(@alignCast(c.ds4_gpu_tensor_contents(graph.batch.engram_rows))));
                    }
                }
                const engram_at = if (profile) nowSeconds() else 0;
                var stage = Stage{ .enabled = stage_profile, .started = if (stage_profile) nowSeconds() else 0, .layer = index, .position = start, .rows = count };
                if (ok) ok = c.ds4_gpu_begin_commands() != 0;
                if (ok and prefetched and uses_engram) {
                    const bytes = @as(u64, c.DS4_ENGRAM_COLS) * c.DS4_ENGRAM_DIM * 4;
                    ok = c.ds4_gpu_tensor_copy(graph.batch.engram_rows, 0, graph.engram_prefetch, @as(u64, offset) * bytes, @as(u64, count) * bytes) != 0;
                }
                if (ok and batch_hc) ok = executor.capture(index, start, active.row.residual, count) and
                    beforeBatch(executor, &active.row, layer, index, count);
                if (ok and batch_hc) ok = executor.trace(active.row.norm, shape.n_embd, start, count, index, "1-norm") and
                    executor.trace(active.row.attn_split, 24, start, count, index, "1-split");
                stage.mark(&ok, "hc/engram");
                var token: u32 = 0;
                while (ok and !batch_hc and token < count) : (token += 1) {
                    row.pos = start + token;
                    buffers.selectRow(&row, @ptrCast(&graph.rows_view[token]));
                    const local = rowExecutor(executor, &row);
                    ok = if (batch_attention) local.beforeAttention(layer, index) else if (batch_moe)
                        local.beforeMoe(layer, index)
                    else
                        local.runLayer(layer, index, @bitCast(tokens[offset + token]));
                }
                if (ok and batch_attention) {
                    const batch = BatchAttention{ .ops = executor.ops, .graph = graph };
                    ok = batch.project(layer, count);
                    if (ok) ok = executor.trace(graph.batch.qr, shape.n_lora_q, start, count, index, "1a-qr") and
                        executor.trace(graph.batch.q, shape.n_head * shape.n_head_dim / graph.tp_world, start, count, index, "1b-q") and
                        executor.trace(graph.batch.kv, shape.n_head_dim, start, count, index, "1c-kv");
                    stage.mark(&ok, "attention projections");
                    if (ok and batch_core) ok = batch.run(layer, index, count);
                    token = 0;
                    while (ok and !batch_core and token < count) : (token += 1) {
                        row.pos = start + token;
                        buffers.selectRow(&row, @ptrCast(&graph.rows_view[token]));
                        const local_attention = attention(executor, &row);
                        ok = local_attention.run(layer, index, true);
                    }
                    if (ok) ok = executor.trace(graph.batch.heads, shape.n_head * shape.n_head_dim / graph.tp_world, start, count, index, "1d-heads");
                    stage.mark(&ok, "attention core/index");
                    if (ok and graph.tp_world == 2) {
                        ok = c.ds4_gpu_dsv41_attention_output_tp_batch(graph.batch.block, graph.batch.low, model.map, model.size, layer.attn_output_a.*.abs_offset, layer.attn_output_b.*.abs_offset, graph.batch.heads, count, graph.tp_rank) != 0 and
                            executor.ops.sumPartialBatch(graph, graph.batch.block, index, c.DS4_TP_GATE_ATTN, count) and
                            c.ds4_gpu_dsv41_quantize(graph.batch.block, shape.n_embd, count, c.DS4_V41_BF16) != 0;
                    } else if (ok and layer.attn_output_b.*.type == c.DS4_TENSOR_Q8_0 and !executor.ops.exactDenseRows()) {
                        ok = c.ds4_gpu_dsv41_attention_output_batch(graph.batch.block, graph.batch.low, model.map, model.size, layer.attn_output_a.*.abs_offset, layer.attn_output_b.*.abs_offset, graph.batch.heads, count) != 0 and
                            c.ds4_gpu_dsv41_quantize(graph.batch.block, shape.n_embd, count, c.DS4_V41_BF16) != 0;
                    } else if (ok) {
                        token = 0;
                        while (ok and token < count) : (token += 1) {
                            row.heads = graph.rows_view[token].heads;
                            row.low = graph.rows_view[token].low;
                            const local_attention = attention(executor, &row);
                            ok = local_attention.low(layer);
                        }
                        if (ok) ok = executor.ops.matmulBatch(graph.batch.block, layer.attn_output_b, graph.batch.low, count, true);
                    }
                    if (ok) ok = executor.trace(graph.batch.block, shape.n_embd, start, count, index, "2-attn");
                    stage.mark(&ok, "attention output");
                    if (ok and batch_hc) ok = afterBatch(executor, &active.row, layer, count);
                    if (ok and batch_hc) ok = executor.trace(active.row.norm, shape.n_embd, start, count, index, "3-norm") and
                        executor.trace(active.row.ffn_split, 24, start, count, index, "3-split");
                    stage.mark(&ok, "hc/ffn norm");
                    token = 0;
                    while (ok and (!batch_hc or (!batch_moe and !exact_moe)) and token < count) : (token += 1) {
                        row.pos = start + token;
                        buffers.selectRow(&row, @ptrCast(&graph.rows_view[token]));
                        const local = rowExecutor(executor, &row);
                        if (!batch_hc) ok = local.afterAttention(layer);
                        if (ok and !batch_moe and !exact_moe) {
                            const services = (Routes.Transaction{ .ops = local.ops, .graph = &row }).snapshot(index);
                            const moe = local.moe();
                            ok = moe.partial(layer, index, @bitCast(tokens[offset + token]), &services) and moe.finish(index);
                            if (ok and !batch_hc) ok = local.afterMoe();
                        }
                    }
                    if (ok and batch_hc and !batch_moe and !exact_moe) ok =
                        c.ds4_gpu_hc_expand_split_tensor(active.row.residual, active.row.block, active.row.after_attn, active.row.ffn_split, shape.n_embd, shape.n_hc) != 0 and
                        c.ds4_gpu_dsv41_quantize(active.row.residual, shape.n_embd * shape.n_hc, count, c.DS4_V41_BF16) != 0;
                }
                if (ok and (batch_moe or exact_moe)) {
                    const moe = executor.moe();
                    ok = if (exact_moe) moe.exactTiles(layer, index, count) else moe.batch(layer, index, count, false, !layer_selected);
                    if (ok) ok = executor.trace(graph.batch.selected, shape.n_expert_used, start, count, index, "3a-selected") and
                        executor.trace(graph.batch.route_weights, shape.n_expert_used, start, count, index, "3b-weights") and
                        executor.trace(graph.batch.shared, shape.n_embd, start, count, index, "3c-shared") and
                        executor.trace(graph.batch.routed, shape.n_embd, start, count, index, "3d-routed");
                    stage.mark(&ok, "shared/routed ffn");
                    if (ok and batch_hc) ok = c.ds4_gpu_add_tensor(active.row.block, active.row.routed, active.row.shared, count * shape.n_embd) != 0 and
                        c.ds4_gpu_dsv41_quantize(active.row.block, shape.n_embd, count, c.DS4_V41_BF16) != 0 and
                        c.ds4_gpu_hc_expand_split_tensor(active.row.residual, active.row.block, active.row.after_attn, active.row.ffn_split, shape.n_embd, shape.n_hc) != 0 and
                        c.ds4_gpu_dsv41_quantize(active.row.residual, shape.n_embd * shape.n_hc, count, c.DS4_V41_BF16) != 0;
                    token = 0;
                    while (ok and !batch_hc and token < count) : (token += 1) {
                        buffers.selectRow(&row, @ptrCast(&graph.rows_view[token]));
                        const local = rowExecutor(executor, &row);
                        ok = c.ds4_gpu_add_tensor(row.block, row.routed, row.shared, shape.n_embd) != 0 and
                            support.bf16(row.block, shape.n_embd) and local.afterMoe();
                    }
                }
                stage.mark(&ok, "hc expand");
                if (ok and wide and index + 1 < shape.n_layer) ok = buffers.carryCopy(config, graph, offset, count, true);
                if (ok) ok = executor.trace(graph.batch.residual, shape.n_hc * shape.n_embd, start, count, index, "4-residual");
                const encoded_at = if (profile) nowSeconds() else 0;
                if (c.ds4_gpu_commands_active() != 0 and c.ds4_gpu_end_commands() == 0) ok = false;
                if (graph.tp_world == 2 and c.ds4_gpu_tp_failed() != 0) ok = false;
                const done_at = if (profile) nowSeconds() else 0;
                if (ok and !layer_selected and !encoder_only and offset + count == total_count) ok = cache.seed(config, graph, model, layer, index, count);
                if (prefetched and uses_engram and offset + count == total_count and !prefetch.join(!ok)) ok = false;
                if (profile) _ = c.fprintf(c.__stderrp, "ds4: V4.1 prefill layer=%u pos=%u rows=%u encoder_resident=%u map=%.3f engram=%.3f encode=%.3f drain=%.3f seed=%.3f ms\n", index, start, count, @as(c_uint, @intFromBool(graph.encoder_resident)), (if (offset == first) mapped_at - map_started else 0) * 1000, (engram_at - ready_at) * 1000, (encoded_at - engram_at) * 1000, (done_at - encoded_at) * 1000, (nowSeconds() - done_at) * 1000);
                if (!ok) _ = c.fprintf(c.__stderrp, "ds4: V4.1 layer-major prefill failed at layer %u\n", index);
                if (ok) if (callbacks.progress) |progress| {
                    const after = (total_count * index + offset + count) / shape.n_layer;
                    progress(callbacks.progress_userdata, "prefill_display", @intCast(initial_start + after), callbacks.total);
                };
                if (batch_hc and index + 1 == shape.n_layer and offset + count == total_count) {
                    row.residual = graph.rows_view[count - 1].residual;
                    row.pre = graph.rows_view[count - 1].ffn_split;
                }
            }
        }
        if (!prefetch.join(!ok)) ok = false;
        if (!streaming.joinAll()) ok = false;
        if (graph.streaming and !selected_small and !streaming.mapStatic()) ok = false;
        if (ok and !encoder_only) {
            ok = c.ds4_gpu_begin_commands() != 0 and
                c.ds4_gpu_tensor_copy(graph.residual, 0, row.residual, 0, @as(u64, shape.n_hc) * shape.n_embd * 4) != 0 and
                c.ds4_gpu_tensor_copy(graph.pre, 0, row.pre, 0, @as(u64, shape.n_hc) * 4) != 0;
            if (c.ds4_gpu_commands_active() != 0 and c.ds4_gpu_end_commands() == 0) ok = false;
        }
        graph.pos = initial_start;
        graph.decoder_replay_floor = 0;
        if (ok) {
            graph.pos += total_count;
            graph.history = next_history;
            graph.valid = !encoder_only;
        }
        if (ok and replay_rows != 0 and !encoder_only)
            _ = c.fprintf(c.__stderrp, "ds4: V4.1 decoder replay rows=%u floor=%u end=%u approximate=1\n", replay_rows, replay_floor, initial_start + total_count);
        if (ok and replay_selected_requested and !encoder_only)
            _ = c.fprintf(c.__stderrp, "ds4: V4.1 decoder replay selected_layers=%u full_layers=%u\n", replay_selected_layers, shape.n_layer - 20 - replay_selected_layers);
        if (graph.resident_prefill_active) {
            graph.resident_prefill_active = false;
            graph.resident_route_gap = 0;
        }
        return ok;
    }
};
