//! V4.1 layer execution. Ops owns numerical dispatch, Attention owns cache
//! publication/selection, and Moe owns expert scheduling. The caller owns the
//! command scope and graph; borrowed execution-row copies never free resources.
const c = @import("gguf_c");
const Routes = @import("route_services.zig");
const support = @import("ops.zig");
const Ops = support.Ops;
const Attention = @import("attention.zig").Attention;
const routed = @import("moe.zig");
const Moe = routed.Moe;
const Streaming = @import("streaming.zig").Streaming;
const draft_owner = @import("dspark.zig");
const narrow = @import("decode_weights.zig");
const Graph = c.ds41_gpu_graph;
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const Layer = c.ds4_layer_weights;
const Tensor = c.ds4_gpu_tensor;

// Private diagnostics and feature owners keep their state during migration.
// These are complete actions on the real records, not per-field accessors.
pub const Services = struct {
    trace: *const fn (?*Tensor, u32, u32, u32, u32, [*:0]const u8) callconv(.c) bool,
    observe_imatrix: *const fn (*Graph, u32) callconv(.c) bool,
};

pub const Executor = struct {
    ops: Ops,
    graph: *Graph,
    weights: *const Weights,
    services: Services,
    direct_pre: bool = false,

    fn routing(self: *const Executor) Routes.Transaction {
        return .{ .ops = self.ops, .graph = self.graph };
    }

    pub fn moe(self: *const Executor) Moe {
        return .{ .ops = self.ops, .graph = self.graph };
    }

    pub fn capture(self: *const Executor, layer: u32, start: u32, tensor: ?*const Tensor, count: u32) bool {
        return draft_owner.capture(self.ops.config, self.graph.dspark, layer, start, tensor, count);
    }

    pub fn trace(self: *const Executor, tensor: ?*Tensor, width: u32, start: u32, count: u32, layer: u32, name: [*:0]const u8) bool {
        return self.services.trace(tensor, width, start, count, layer, name);
    }

    pub fn beforeAttention(self: *const Executor, layer: *const Layer, index: u32) bool {
        const graph = self.graph;
        const shape = &self.ops.config.shape;
        const policy = self.ops.config.layerPolicy(index) orelse return false;
        if (policy.uses_engram and self.ops.imageAt(graph, graph.pos) == null) {
            const owner: usize = if (index == 1) 0 else 1;
            const rows = if (owner == 0) graph.engram_rows else graph.engram_rows_b;
            const copies = narrow.Copies.fromOpaque(graph.decode_weights);
            const copied = if (copies) |weights| !graph.quality and narrow.project(weights.engram[owner], graph.engram_kv, rows, layer.engram_kv) else false;
            if ((!copied and !self.ops.matmul(graph.engram_kv, layer.engram_kv, rows, true)) or
                c.ds4_gpu_dsv41_engram_add(graph.residual, graph.engram_kv, graph.engram_q_norm[owner], graph.engram_k_norm[owner], null, shape.n_embd, 1, shape.rms_eps) == 0) return false;
        }
        const split_pre = self.direct_pre and index != 0;
        return self.capture(index, graph.pos, graph.residual, 1) and
            self.ops.hcInput(graph, layer, false, if (split_pre) graph.ffn_split else graph.pre, split_pre);
    }

    /// Publish the residual first; steering acts on every complete HC stream.
    /// Off keeps the established expansion and BF16 kernels unchanged.
    pub fn expand(self: *const Executor, out: ?*Tensor, block: ?*Tensor, residual: ?*const Tensor, split: ?*const Tensor, index: u32, count: u32, ffn: bool) bool {
        const graph = self.graph;
        const shape = &self.ops.config.shape;
        const raw = if (ffn) graph.steering_ffn_scale else graph.steering_attn_scale;
        const scale = @import("../steering.zig").effectiveScale(raw, graph.steering_mode, graph.steering_layers);
        const ok = if (scale != 0)
            c.ds4_gpu_hc_expand_split_steering_tensor(out, block, residual, split, graph.steering_dirs, index, shape.n_embd, shape.n_hc, scale, true, graph.steering_mode) != 0
        else if (count == 1)
            self.ops.hcExpand(out, block, residual, split)
        else
            c.ds4_gpu_hc_expand_split_tensor(out, block, residual, split, shape.n_embd, shape.n_hc) != 0 and
                c.ds4_gpu_dsv41_quantize(out, shape.n_embd * shape.n_hc, count, c.DS4_V41_BF16) != 0;
        return ok and @import("steering_capture.zig").capture(graph, out, index, count, shape.n_embd, shape.n_hc, ffn);
    }

    pub fn afterAttention(self: *const Executor, layer: *const Layer, index: u32) bool {
        const graph = self.graph;
        if (graph.tp_gate_fused) graph.attn_expanded = self.ops.expandPartials(graph, graph.after_attn, graph.residual, graph.attn_split, index, c.DS4_TP_GATE_ATTN);
        if (graph.tp_gate_fused and !graph.attn_expanded) return false;
        const expanded = if (graph.attn_expanded)
            @import("steering_capture.zig").capture(graph, graph.after_attn, index, 1, self.ops.config.shape.n_embd, self.ops.config.shape.n_hc, false)
        else
            self.expand(graph.after_attn, graph.block, graph.residual, graph.attn_split, index, 1, false);
        graph.attn_expanded = false;
        return expanded and self.ops.hcInput(graph, layer, true, graph.attn_split, true);
    }

    pub fn beforeMoe(self: *const Executor, layer: *const Layer, index: u32) bool {
        const attention = Attention{ .config = self.ops.config, .model = self.ops.model, .graph = self.graph };
        return self.beforeAttention(layer, index) and attention.run(layer, index, false) and self.afterAttention(layer, index);
    }

    pub fn afterMoe(self: *const Executor, index: u32) bool {
        const graph = self.graph;
        const layer = &self.weights.layer[index];
        const shape = &self.ops.config.shape;
        var expanded = false;
        if (graph.tp_gate_fused) {
            if (!self.ops.expandPartials(graph, graph.residual, graph.after_attn, graph.ffn_split, index, c.DS4_TP_GATE_FFN)) return false;
            expanded = true;
        } else if (graph.tp_world == 1) {
            if (!c.ds4_native_graph_trace_active() and graph.steering_ffn_scale == 0 and layer.ffn_down_shexp.*.type == c.DS4_TENSOR_Q8_0 and c.getenv("DS4_METAL_DISABLE_V41_EXPAND_FUSION") == null) {
                if (!graph.shared_done) {
                    expanded = c.ds4_gpu_dsv41_matmul_expand_rows(graph.residual, self.ops.model.map, self.ops.model.size, layer.ffn_down_shexp.*.abs_offset, @intCast(layer.ffn_down_shexp.*.dim[0]), @intCast(layer.ffn_down_shexp.*.dim[1]), graph.shared_mid, graph.routed, graph.after_attn, graph.ffn_split, shape.n_hc, 1) != 0;
                } else if (graph.streaming and graph.scalar_decode) {
                    // In scalar streaming, shared_done identifies the opt-in
                    // early down projection. Preserve matmul-expand's shared
                    // first addition and BF16 boundaries at the sum and HC output.
                    expanded = c.ds4_gpu_hc_expand_split_add_bf16_tensor(graph.residual, graph.shared, graph.routed, graph.after_attn, graph.ffn_split, shape.n_embd, shape.n_hc) != 0;
                }
            }
            if (!expanded) {
                if (!graph.shared_done and !self.sharedDown(graph.shared, graph.shared_mid, layer, 1)) return false;
                if (c.ds4_gpu_add_tensor(graph.block, graph.routed, graph.shared, shape.n_embd) == 0 or
                    c.ds4_gpu_dsv41_quantize(graph.block, shape.n_embd, 1, c.DS4_V41_BF16) == 0) return false;
            }
        }
        graph.shared_done = false;
        const ok = if (expanded)
            @import("steering_capture.zig").capture(graph, graph.residual, index, 1, shape.n_embd, shape.n_hc, true)
        else
            self.expand(graph.residual, graph.block, graph.after_attn, graph.ffn_split, index, 1, true);
        // The next attention HC reads this FFN split before its own FFN HC
        // overwrites it. Publish the final PRE for logits and saved state;
        // the persistent graph.pre handle and first-layer initialization stay.
        return ok and ((self.direct_pre and index + 1 < shape.n_layer) or
            c.ds4_gpu_tensor_copy(graph.pre, 0, graph.ffn_split, 0, @as(u64, shape.n_hc) * 4) != 0);
    }

    fn sharedDown(self: *const Executor, out: ?*Tensor, mid: ?*Tensor, layer: *const Layer, count: u32) bool {
        const weight = layer.ffn_down_shexp;
        if (weight.*.type == c.DS4_TENSOR_Q8_0)
            return c.ds4_gpu_matmul_q8_0_bf16io_tensor(out, self.ops.model.map, self.ops.model.size, weight.*.abs_offset, weight.*.dim[0], weight.*.dim[1], mid, count) != 0;
        return c.ds4_gpu_dsv41_quantize(mid, self.ops.config.shape.n_ff_exp, count, c.DS4_V41_BF16) != 0 and
            self.ops.matmulBatch(out, weight, mid, count, true);
    }

    /// Finish a batch whose shared down projection is deliberately pending.
    pub fn finishMoeRows(self: *const Executor, row: *c.ds41_prefill_row, layer: *const Layer, index: u32, count: u32) bool {
        const graph = self.graph;
        const shape = &self.ops.config.shape;
        if (graph.steering_ffn_scale == 0 and layer.ffn_down_shexp.*.type == c.DS4_TENSOR_Q8_0 and c.getenv("DS4_METAL_DISABLE_V41_EXPAND_FUSION") == null and
            c.ds4_gpu_dsv41_matmul_expand_rows(row.residual, self.ops.model.map, self.ops.model.size, layer.ffn_down_shexp.*.abs_offset, @intCast(layer.ffn_down_shexp.*.dim[0]), @intCast(layer.ffn_down_shexp.*.dim[1]), row.shared_mid, row.routed, row.after_attn, row.ffn_split, shape.n_hc, count) != 0)
            return @import("steering_capture.zig").capture(graph, row.residual, index, count, shape.n_embd, shape.n_hc, true);
        return self.sharedDown(row.shared, row.shared_mid, layer, count) and
            c.ds4_gpu_add_tensor(row.block, row.routed, row.shared, count * shape.n_embd) != 0 and
            c.ds4_gpu_dsv41_quantize(row.block, shape.n_embd, count, c.DS4_V41_BF16) != 0 and
            self.expand(row.residual, row.block, row.after_attn, row.ffn_split, index, count, true);
    }

    pub fn runLayer(self: *const Executor, weights: *const Layer, index: u32, token: u32) bool {
        const graph = self.graph;
        const shape = &self.ops.config.shape;
        const position = graph.pos;
        const attention = Attention{ .config = self.ops.config, .model = self.ops.model, .graph = graph };
        if (!self.beforeAttention(weights, index) or
            !self.routing().probe(weights, token) or
            !self.routing().predict(weights, index, token) or
            !self.trace(graph.norm, shape.n_embd, position, 1, index, "1-norm") or
            !self.trace(graph.attn_split, 24, position, 1, index, "1-split") or
            !attention.run(weights, index, false) or
            !self.trace(graph.block, shape.n_embd, position, 1, index, "2-attn") or
            !self.afterAttention(weights, index) or
            !self.trace(graph.norm, shape.n_embd, position, 1, index, "3-norm") or
            !self.trace(graph.ffn_split, 24, position, 1, index, "3-split")) return false;
        const expert = self.moe();
        const services = self.routing().snapshot(index);
        return expert.partial(weights, index, token, &services) and expert.finish(index) and
            self.trace(graph.selected, shape.n_expert_used, position, 1, index, "3a-selected") and
            self.trace(graph.route_weights, shape.n_expert_used, position, 1, index, "3b-weights") and
            self.trace(graph.routed, shape.n_embd, position, 1, index, "3d-routed") and
            self.afterMoe(index) and self.trace(graph.shared, shape.n_embd, position, 1, index, "3c-shared") and self.trace(graph.residual, shape.n_hc * shape.n_embd, position, 1, index, "4-residual");
    }

    fn encodeLogits(self: *const Executor) bool {
        const graph = self.graph;
        const device_output = if (graph.tp_logits_half != null) graph.tp_logits_half else graph.logits;
        return self.ops.hcNorm(graph.x, graph.norm, graph.residual, graph.pre, false, self.weights.output_norm, 1) and
            self.ops.outputProjection(graph, device_output, self.weights, graph.norm, 1);
    }

    pub fn logits(self: *const Executor, output: [*c]f32) bool {
        const graph = self.graph;
        if (!graph.valid or output == null or c.ds4_gpu_begin_commands() == 0) return false;
        var ok = self.encodeLogits();
        if (c.ds4_gpu_end_commands() == 0) ok = false;
        return ok and c.ds4_gpu_tensor_read(graph.logits, 0, output, @as(u64, self.ops.config.shape.n_vocab) * 4) != 0;
    }

    pub fn step(self: *const Executor, token: c_int, output: [*c]f32) bool {
        const graph = self.graph;
        const model = self.ops.model;
        const shape = &self.ops.config.shape;
        if (!graph.valid or graph.pos >= graph.ctx or token < 0 or token >= shape.n_vocab) return false;
        const oracle_active = self.routing().beginToken(@intCast(token), output != null);
        var ids: [2][c.DS4_ENGRAM_COLS]u32 = undefined;
        var history = graph.history;
        if (!self.ops.hashTokens(graph, &history, &token, 1, @ptrCast(&ids))) return false;
        const image = self.ops.imageAt(graph, graph.pos) != null;
        const overlap = graph.engram_reader != null and !image and c.getenv("DS4_DISABLE_V41_ENGRAM_OVERLAP") == null;
        if (overlap and !c.ds4_engram_reader_start(graph.engram_reader, &graph.table, &ids, @ptrCast(&graph.rows))) return false;
        var reader_live = overlap;
        // The reader borrows stack IDs and host output. Every exit after start
        // must join it before this call or a borrowed row can retire.
        defer if (reader_live) {
            _ = c.ds4_engram_reader_wait(graph.engram_reader, 1);
        };
        var reads: [2]?*c.ds4_engram_prefetch = @splat(null);
        // Each job borrows IDs/output until joined. Separate GPU input tensors
        // let table 14 upload without draining work that still reads table 1.
        defer for (reads) |read| {
            if (read) |job| _ = c.ds4_engram_read_batch_finish(job);
        };
        if (!overlap and !image) {
            for (0..2) |owner| {
                reads[owner] = c.ds4_engram_read_batch_start(&graph.table[owner], &ids[owner], 1, c.DS4_ENGRAM_COLS, &graph.rows[owner]) orelse return false;
            }
        }
        const initial = [_]f32{ 1, 0, 0, 0 };
        var ok = c.ds4_gpu_tensor_write(graph.pre, 0, &initial, @sizeOf(@TypeOf(initial))) != 0 and c.ds4_gpu_begin_commands() != 0;
        graph.predict_active = ok and c.getenv("DS4_METAL_V41_GATE_PREFETCH") != null;
        if (graph.predict_active) {
            if (!graph.streaming or graph.quality or graph.imatrix != null or graph.image_count != 0 or graph.tp_world != 1 or
                c.getenv("DS4_METAL_DISABLE_V41_SHORT_OPTIMIZATIONS") != null or
                c.getenv("DS4_METAL_V41_ASYNC_EXPERT_LOAD") != null or oracle_active)
                c.ds4_engine_model_die("gate prefetch requires ordinary scalar text SSD execution without oracle modes");
            if (graph.predict_selected == null) {
                graph.predict_selected = c.ds4_gpu_tensor_alloc(@as(u64, shape.n_expert_used) * @sizeOf(i32)) orelse
                    c.ds4_engine_model_die("cannot allocate predicted selected IDs");
            }
        }
        if (ok) ok = self.ops.embed(graph, self.weights, graph.residual, graph.x, token, graph.pos);
        // Unfused quality kernels read whole experts. Map one admitted layer
        // at a time, then restore static decode views before returning.
        const layer_resident = graph.streaming and graph.quality;
        var streaming: Streaming = undefined;
        if (layer_resident) {
            streaming = Streaming.init(model, self.weights, self.ops.config, 1);
            if (c.ds4_gpu_end_commands() == 0) ok = false;
        }
        const queue = graph.imatrix == null and !layer_resident and
            c.getenv(if (graph.tp_world == 2) "DS4_METAL_DISABLE_V41_TP_DECODE_QUEUE" else "DS4_METAL_DISABLE_V41_DECODE_QUEUE") == null and
            c.getenv("DS4_METAL_DISABLE_V41_SCALAR_QUEUE") == null;
        const queued_head = queue and output != null and graph.tp_world == 1 and
            c.getenv("DS4_METAL_DISABLE_V41_QUEUED_HEAD") == null;
        const flush_layers: u32 = if (c.getenv("DS4_METAL_V41_DECODE_FLUSH_LAYERS")) |value|
            @intCast(@max(c.atoi(value), 0))
        else
            2;
        graph.scalar_decode = true;
        graph.shared_done = false;
        graph.tp_gate_fused = !c.ds4_native_graph_trace_active() and graph.tp_world == 2 and graph.steering_attn_scale == 0 and graph.steering_ffn_scale == 0 and
            c.getenv("DS4_METAL_DISABLE_V41_TP_GATE_EXPAND") == null and c.getenv("DS4_METAL_DISABLE_V41_TP_SHARED_OWNER") == null;
        defer {
            graph.scalar_decode = false;
            graph.tp_gate_fused = false;
        }
        graph.rope_tab_pos = 0;
        defer graph.rope_tab_pos = 0;
        // The graph-owned table survives streaming command boundaries. Opt in
        // to transforming each attention head once before its low projection;
        // retain the existing rounding, shape fallbacks and per-token validity.
        const streaming_rope_setting = c.getenv("DS4_METAL_V41_STREAMING_ROPE_TABLE");
        const streaming_rope = streaming_rope_setting != null and streaming_rope_setting[0] != 0 and c.strcmp(streaming_rope_setting, "0") != 0;
        if (output != null and graph.rope_tab != null and graph.tp_world == 1 and (!graph.streaming or streaming_rope) and !graph.quality and
            (c.ds4_gpu_v41_off() & c.DS4_GPU_V41_ROPE_TABLE) == 0 and c.ds4_gpu_commands_active() != 0 and
            c.ds4_gpu_dsv41_rope_coeff_table(graph.rope_tab, graph.pos) != 0) graph.rope_tab_pos = graph.pos + 1;
        // Keep one decision for both copy suppression and its next consumer.
        // Benchmark logits/hash logging and Metal profiling do not change
        // this admission; graph trace capture and alternate executors do.
        const direct_pre_setting = c.getenv("DS4_METAL_V41_DIRECT_PRE");
        var scalar = self.*;
        scalar.direct_pre = direct_pre_setting != null and c.strcmp(direct_pre_setting, "1") == 0 and
            queue and graph.streaming and graph.tp_world == 1 and !graph.quality and graph.imatrix == null and
            graph.image_count == 0 and graph.dspark == null and !graph.verify_prefix and graph.decoder_replay_floor == 0 and
            graph.steering_attn_scale == 0 and graph.steering_ffn_scale == 0 and !graph.steering_capture_enabled and shape.n_hc == 4 and
            !graph.predict_active and !oracle_active and !c.ds4_native_graph_trace_active();
        for (self.weights.layer[0..shape.n_layer], 0..) |*weights, index| {
            if (!ok) break;
            const layer_index: u32 = @intCast(index);
            if (layer_resident) ok = streaming.mapLayer(layer_index) and c.ds4_gpu_begin_commands() != 0;
            if (ok and self.ops.config.layerPolicy(layer_index).?.uses_engram) {
                const owner: usize = if (index == 1) 0 else 1;
                var loaded = true;
                if (overlap) loaded = c.ds4_engram_reader_wait(graph.engram_reader, @intCast(owner));
                if (reads[owner]) |job| {
                    loaded = c.ds4_engram_read_batch_finish(job);
                    reads[owner] = null;
                }
                const input = if (owner == 0) graph.engram_rows else graph.engram_rows_b;
                ok = loaded and c.ds4_gpu_tensor_write(input, 0, &graph.rows[owner], @sizeOf(@TypeOf(graph.rows[owner]))) != 0;
            }
            if (ok) ok = self.routing().oraclePrefetch(weights, layer_index);
            if (ok) ok = scalar.runLayer(weights, layer_index, @intCast(token));
            const drain = !queue or (graph.tp_world == 2 and index == 13) or index + 1 == shape.n_layer;
            if (ok and queued_head and index + 1 == shape.n_layer) ok = self.encodeLogits();
            if (drain and c.ds4_gpu_end_commands() == 0) ok = false;
            if (graph.tp_world == 2 and c.ds4_gpu_tp_failed() != 0) ok = false;
            if (ok and graph.imatrix != null) ok = self.services.observe_imatrix(graph, layer_index);
            if (!ok) _ = c.fprintf(c.__stderrp, "ds4: V4.1 layer %u failed at position %u\n", layer_index, graph.pos);
            if (ok and drain and !layer_resident and index + 1 < shape.n_layer) ok = c.ds4_gpu_begin_commands() != 0;
            // Submit a bounded prefix while the CPU encodes the next layers.
            // Inline TP gates also need submissions so the service can exchange
            // each published partial while the CPU encodes later layers.
            if (ok and queue and !drain and (graph.tp_world != 2 or c.ds4_gpu_tp_decode_inline_gates() != 0) and flush_layers != 0 and
                (index == 0 or (index + 1) % flush_layers == 0)) ok = c.ds4_gpu_flush_commands() != 0;
        }
        if (graph.predict_load.active) {
            _ = c.ds4_gpu_flush_commands();
            _ = c.ds4_native_moe_selected_finish(&graph.predict_load);
            _ = c.ds4_gpu_routed_moe_set_selected_override(null, 0);
        }
        graph.predict_active = false;
        if (c.ds4_gpu_commands_active() != 0 and c.ds4_gpu_end_commands() == 0) ok = false;
        if (layer_resident and !streaming.mapStatic()) ok = false;
        if (graph.tp_world == 2 and c.ds4_gpu_tp_failed() != 0) ok = false;
        if (ok and output != null) ok = if (queued_head)
            c.ds4_gpu_tensor_read(graph.logits, 0, output, @as(u64, shape.n_vocab) * 4) != 0
        else
            self.logits(output);
        if (overlap) {
            if (!c.ds4_engram_reader_wait(graph.engram_reader, 1)) ok = false;
            reader_live = false;
        }
        ok = self.routing().endToken(ok);
        if (!ok) {
            graph.valid = false;
            return false;
        }
        graph.history = history;
        graph.pos += 1;
        return true;
    }
};
