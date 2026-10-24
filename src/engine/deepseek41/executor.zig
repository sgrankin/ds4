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
            if (!self.ops.matmul(graph.engram_kv, layer.engram_kv, graph.engram_rows, true) or
                c.ds4_gpu_dsv41_engram_add(graph.residual, graph.engram_kv, graph.engram_q_norm[owner], graph.engram_k_norm[owner], null, shape.n_embd, 1, shape.rms_eps) == 0) return false;
        }
        return self.capture(index, graph.pos, graph.residual, 1) and
            self.ops.hcMix(graph, layer, false) and
            self.ops.hcSum(graph.x, graph.residual, graph.pre, false) and
            self.ops.norm(graph.norm, graph.x, layer.attn_norm);
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
        return self.expand(graph.after_attn, graph.block, graph.residual, graph.attn_split, index, 1, false) and
            self.ops.hcMix(graph, layer, true) and
            self.ops.hcSum(graph.x, graph.after_attn, graph.attn_split, true) and
            self.ops.norm(graph.norm, graph.x, layer.ffn_norm);
    }

    pub fn beforeMoe(self: *const Executor, layer: *const Layer, index: u32) bool {
        const attention = Attention{ .config = self.ops.config, .model = self.ops.model, .graph = self.graph };
        return self.beforeAttention(layer, index) and attention.run(layer, index, false) and self.afterAttention(layer, index);
    }

    pub fn afterMoe(self: *const Executor, index: u32) bool {
        const graph = self.graph;
        return self.expand(graph.residual, graph.block, graph.after_attn, graph.ffn_split, index, 1, true) and
            c.ds4_gpu_tensor_copy(graph.pre, 0, graph.ffn_split, 0, @as(u64, self.ops.config.shape.n_hc) * 4) != 0;
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
            self.trace(graph.shared, shape.n_embd, position, 1, index, "3c-shared") and
            self.trace(graph.routed, shape.n_embd, position, 1, index, "3d-routed") and
            self.afterMoe(index) and self.trace(graph.residual, shape.n_hc * shape.n_embd, position, 1, index, "4-residual");
    }

    pub fn logits(self: *const Executor, output: [*c]f32) bool {
        const graph = self.graph;
        if (!graph.valid or output == null or c.ds4_gpu_begin_commands() == 0) return false;
        const device_output = if (graph.tp_logits_half != null) graph.tp_logits_half else graph.logits;
        var ok = self.ops.hcSum(graph.x, graph.residual, graph.pre, false) and
            self.ops.norm(graph.norm, graph.x, self.weights.output_norm) and
            self.ops.outputProjection(graph, device_output, self.weights, graph.norm, 1);
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
        if (!overlap and !image) {
            for (0..2) |owner| {
                if (!c.ds4_engram_read_batch(&graph.table[owner], &ids[owner], 1, c.DS4_ENGRAM_COLS, &graph.rows[owner])) return false;
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
        const queue = graph.imatrix == null and
            ((graph.tp_world == 2 and c.getenv("DS4_METAL_DISABLE_V41_TP_DECODE_QUEUE") == null) or
                (graph.tp_world == 1 and graph.streaming and !graph.quality and c.getenv("DS4_METAL_DISABLE_V41_SCALAR_QUEUE") == null));
        for (self.weights.layer[0..shape.n_layer], 0..) |*weights, index| {
            if (!ok) break;
            const layer_index: u32 = @intCast(index);
            if (layer_resident) ok = streaming.mapLayer(layer_index) and c.ds4_gpu_begin_commands() != 0;
            if (ok and self.ops.config.layerPolicy(layer_index).?.uses_engram) {
                const owner: usize = if (index == 1) 0 else 1;
                ok = (!overlap or c.ds4_engram_reader_wait(graph.engram_reader, @intCast(owner))) and
                    c.ds4_gpu_tensor_write(graph.engram_rows, 0, &graph.rows[owner], @sizeOf(@TypeOf(graph.rows[owner]))) != 0;
            }
            if (ok) ok = self.routing().oraclePrefetch(weights, layer_index);
            if (ok) ok = self.runLayer(weights, layer_index, @intCast(token));
            // Gates/cache checks already order their command buffers. Drain
            // before Engram input reuse at layer14 and before CPU publication.
            const drain = !queue or index == 13 or index + 1 == shape.n_layer;
            if (drain and c.ds4_gpu_end_commands() == 0) ok = false;
            if (graph.tp_world == 2 and c.ds4_gpu_tp_failed() != 0) ok = false;
            if (ok and graph.imatrix != null) ok = self.services.observe_imatrix(graph, layer_index);
            if (!ok) _ = c.fprintf(c.__stderrp, "ds4: V4.1 layer %u failed at position %u\n", layer_index, graph.pos);
            if (ok and drain and !layer_resident and index + 1 < shape.n_layer) ok = c.ds4_gpu_begin_commands() != 0;
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
        if (ok and output != null) ok = self.logits(output);
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
