//! V4.1 shared/routed MoE orchestration. Tensor storage belongs to Graph; the
//! native controller owns scheduling and the lifetime of its temporary views.
const std = @import("std");
const c = @import("gguf_c");
const Routes = @import("route_services.zig");
const support = @import("ops.zig");
const Ops = support.Ops;
const layout = @import("layout.zig");
const Layout = layout.Layout;
const Graph = c.ds41_gpu_graph;
const Layer = c.ds4_layer_weights;
const Tensor = c.ds4_gpu_tensor;
const Narrow = @import("decode_weights.zig").Copies;

fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}

fn nowMilliseconds() f64 {
    var time: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &time);
    const seconds: f64 = @floatFromInt(time.tv_sec);
    const nanoseconds: f64 = @floatFromInt(time.tv_nsec);
    return (seconds + nanoseconds * 1.0e-9) * 1000.0;
}

pub const Moe = struct {
    ops: Ops,
    graph: *Graph,

    // Preserve both projection BF16 boundaries before SwiGLU, followed by its
    // own BF16 boundary. The optional fused kernel keeps those reductions.
    pub fn sharedGateUp(self: *const Moe, layer: *const Layer, gate: ?*Tensor, up: ?*Tensor, mid: ?*Tensor, input: ?*const Tensor, count: u32) bool {
        const shape = &self.ops.config.shape;
        const model = self.ops.model;
        if (!self.graph.quality and self.graph.tp_world == 1 and count <= 128 and
            layer.ffn_gate_shexp.*.type == c.DS4_TENSOR_Q8_0 and layer.ffn_up_shexp.*.type == c.DS4_TENSOR_Q8_0 and
            present("DS4_METAL_V41_FUSED_SHARED_BF16"))
            return c.ds4_gpu_dsv41_shared_gate_up_bf16_rows(gate, up, mid, model.map, model.size, layer.ffn_gate_shexp.*.abs_offset, layer.ffn_up_shexp.*.abs_offset, shape.n_embd, shape.n_ff_exp, input, count, shape.swiglu_clamp_exp) != 0;

        if (count <= c.DS4_TP_BATCH_MAX_ROWS and layer.ffn_gate_shexp.*.type == c.DS4_TENSOR_Q8_0 and
            layer.ffn_up_shexp.*.type == c.DS4_TENSOR_Q8_0 and
            c.ds4_gpu_dsv41_shared_swiglu(mid, gate, up, model.map, model.size, layer.ffn_gate_shexp.*.abs_offset, layer.ffn_up_shexp.*.abs_offset, shape.n_embd, shape.n_ff_exp, input, shape.swiglu_clamp_exp, count) != 0) return true;
        const gate_ok = if (count == 1)
            self.ops.matmul(gate, layer.ffn_gate_shexp, input, true)
        else
            self.ops.matmulBatch(gate, layer.ffn_gate_shexp, input, count, true);
        if (!gate_ok) return false;
        const up_ok = if (count == 1)
            self.ops.matmul(up, layer.ffn_up_shexp, input, true)
        else
            self.ops.matmulBatch(up, layer.ffn_up_shexp, input, count, true);
        return up_ok and c.ds4_gpu_swiglu_tensor(mid, gate, up, count *% shape.n_ff_exp, shape.swiglu_clamp_exp, 1.0) != 0 and
            c.ds4_gpu_dsv41_quantize(mid, shape.n_ff_exp, count, c.DS4_V41_BF16) != 0;
    }

    pub fn routeBatch(self: *const Moe, layer: *const Layer, count: u32) bool {
        const graph = self.graph;
        const row = &graph.batch;
        const model = self.ops.model;
        const shape = &self.ops.config.shape;
        if (graph.image_count == 0)
            return c.ds4_gpu_router_select_batch_tensor(row.selected, row.route_weights, row.route_probs, model.map, model.size, layer.ffn_exp_probs_b.*.abs_offset, 0, 0, 0, 0, true, false, row.route_logits, graph.prefill_tokens, shape.n_expert, shape.n_expert_used, shape.expert_weight_scale, count) != 0;

        // The ordinary selector already has the trained scoring/normalization.
        // Split only where text versus visual selection bias changes.
        var first: u32 = 0;
        while (first < count) {
            const image = self.ops.imageAt(graph, graph.pos +% first) != null;
            var end = first + 1;
            while (end < count and (self.ops.imageAt(graph, graph.pos +% end) != null) == image) end += 1;
            const bias = if (image) layer.ffn_exp_probs_vl else layer.ffn_exp_probs_b;
            if (bias == null or !self.routeRun(first, end - first, bias)) return false;
            first = end;
        }
        return true;
    }

    fn routeRun(self: *const Moe, first: u32, count: u32, bias: *const c.ds4_tensor) bool {
        const row = &self.graph.batch;
        const model = self.ops.model;
        const shape = &self.ops.config.shape;
        const buffers = [_]?*Tensor{ row.selected, row.route_weights, row.route_probs, row.route_logits, self.graph.prefill_tokens };
        const widths = [_]u32{ shape.n_expert_used, shape.n_expert_used, shape.n_expert, shape.n_expert, 1 };
        var views: [5]?*Tensor = @splat(null);
        defer for (views) |view| c.ds4_gpu_tensor_free(view);
        for (&views, buffers, widths) |*view, buffer, width| {
            view.* = c.ds4_gpu_tensor_view(buffer, @as(u64, first) * width * 4, @as(u64, count) * width * 4) orelse return false;
        }
        return c.ds4_gpu_router_select_batch_tensor(views[0], views[1], views[2], model.map, model.size, bias.abs_offset, 0, 0, 0, 0, true, false, views[3], views[4], shape.n_expert, shape.n_expert_used, shape.expert_weight_scale, count) != 0;
    }

    fn expertRowBytes(weight: *const c.ds4_tensor) u64 {
        const info = c.ds4_gguf_tensor_type(weight.type);
        if (info == null or info.*.block_elems == 0)
            c.ds4_engine_model_die("unsupported routed expert tensor type");
        if (weight.dim[0] % info.*.block_elems != 0)
            c.ds4_engine_model_die("routed expert row is not quant block aligned");
        switch (weight.type) {
            c.DS4_TENSOR_Q8_0, c.DS4_TENSOR_IQ2_XXS, c.DS4_TENSOR_Q2_K, c.DS4_TENSOR_Q4_K, c.DS4_TENSOR_Q5_K, c.DS4_TENSOR_Q6_K, c.DS4_TENSOR_MXFP4 => {},
            else => c.ds4_engine_model_die("unsupported routed expert tensor type"),
        }
        return (weight.dim[0] / info.*.block_elems) *% info.*.block_bytes;
    }

    fn streamTable(self: *const Moe, layer: *const Layer, index: u32, gate_bytes: u64, down_bytes: u64) c.ds4_gpu_stream_expert_table {
        return .{
            .model_map = self.ops.model.map,
            .model_size = self.ops.model.size,
            .layer = index,
            .n_total_expert = self.ops.config.shape.n_expert,
            .gate_offset = layer.ffn_gate_exps.*.abs_offset,
            .up_offset = layer.ffn_up_exps.*.abs_offset,
            .down_offset = layer.ffn_down_exps.*.abs_offset,
            .gate_expert_bytes = gate_bytes,
            .down_expert_bytes = down_bytes,
        };
    }

    fn routing(self: *const Moe) Routes.Transaction {
        return .{ .ops = self.ops, .graph = self.graph };
    }

    fn router(self: *const Moe, layer: *const Layer, index: u32) bool {
        const graph = self.graph;
        if (!graph.quality) {
            if (Narrow.fromOpaque(graph.decode_weights)) |copies| {
                if (index < copies.router.len) {
                    if (copies.router[index]) |weights| {
                        if (c.ds4_gpu_matmul_bf16_weights_tensor(graph.route_logits, weights, layer.ffn_gate_inp.*.dim[0], layer.ffn_gate_inp.*.dim[1], graph.norm) != 0) return true;
                    }
                }
            }
        }
        return self.ops.matmul(graph.route_logits, layer.ffn_gate_inp, graph.norm, false);
    }

    fn routerBatch(self: *const Moe, layer: *const Layer, count: u32) bool {
        const graph = self.graph;
        const model = self.ops.model;
        const shape = &self.ops.config.shape;
        if (graph.plan_rows > count and count > c.DS4_TP_BATCH_MAX_ROWS and layer.ffn_gate_inp.*.type == c.DS4_TENSOR_F32)
            return c.ds4_gpu_matmul_f32_plan_tensor(graph.batch.route_logits, model.map, model.size, layer.ffn_gate_inp.*.abs_offset, shape.n_embd, shape.n_expert, graph.batch.norm, count, graph.plan_rows) != 0;
        return self.ops.matmulBatch(graph.batch.route_logits, layer.ffn_gate_inp, graph.batch.norm, count, false);
    }

    pub fn partial(self: *const Moe, layer: *const Layer, index: u32, token: u32, services: *const c.ds4_native_moe_services) bool {
        const graph = self.graph;
        const model = self.ops.model;
        const shape = &self.ops.config.shape;
        var gate_row: u64 = 0;
        var down_row: u64 = 0;
        if (!c.ds4_gguf_tensor_nbytes(layer.ffn_gate_exps.*.type, shape.n_embd, &gate_row) or
            !c.ds4_gguf_tensor_nbytes(layer.ffn_down_exps.*.type, shape.n_ff_exp, &down_row)) return false;
        const shared_owner = graph.tp_world == 2 and !present("DS4_METAL_DISABLE_V41_TP_SHARED_OWNER");
        const shared_split = shared_owner and !present("DS4_METAL_DISABLE_V41_TP_SHARED_SPLIT") and
            layer.ffn_gate_shexp.*.type == c.DS4_TENSOR_Q8_0 and layer.ffn_up_shexp.*.type == c.DS4_TENSOR_Q8_0 and
            layer.ffn_down_shexp.*.type == c.DS4_TENSOR_Q8_0 and shape.n_ff_exp % 64 == 0;
        const shared_here = !shared_owner or shared_split or graph.tp_rank == (index & 1);
        const shared_dim = if (shared_split) shape.n_ff_exp / 2 else shape.n_ff_exp;
        const shared_lane: u64 = if (shared_split) @as(u64, graph.tp_rank) * shared_dim * (shape.n_embd / 32 * 34) else 0;
        const sum = if (shared_owner) graph.block else graph.routed;
        const routed = if (graph.tp_world == 2 and !c.ds4_native_graph_trace_active() and !present("DS4_METAL_DISABLE_V41_TP_SLAB_WRITE")) graph.tp_out[index * c.DS4_TP_GATES_PER_LAYER + c.DS4_TP_GATE_FFN] else sum;
        const balanced = graph.tp_world == 2 and graph.tp_balanced_route;
        if (balanced) {
            const bytes = @as(u64, shape.n_expert_used / 2) * 4;
            if (graph.selected_half == null) graph.selected_half = c.ds4_gpu_tensor_view(graph.selected, graph.tp_rank * bytes, bytes);
            if (graph.route_weights_half == null) graph.route_weights_half = c.ds4_gpu_tensor_view(graph.route_weights, graph.tp_rank * bytes, bytes);
            if (graph.selected_half == null or graph.route_weights_half == null) return false;
        }
        graph.shared_done = false;
        const bias = if (self.ops.imageAt(graph, graph.pos) != null) layer.ffn_exp_probs_vl else layer.ffn_exp_probs_b;
        if (bias == null or !self.router(layer, index)) return false;
        const routed_one = !present("DS4_METAL_DISABLE_V41_ROUTER_ONE") and
            c.ds4_gpu_dsv41_router_one(graph.selected, graph.route_weights, graph.route_probs, graph.route_logits, model.map, model.size, bias.*.abs_offset, shape.n_expert, shape.n_expert_used, shape.expert_weight_scale) != 0;
        if (!routed_one and c.ds4_gpu_router_select_tensor(graph.selected, graph.route_weights, graph.route_probs, model.map, model.size, bias.*.abs_offset, 0, 0, token, shape.n_expert, shape.n_expert_used, shape.expert_weight_scale, 0, 0, true, false, graph.route_logits) == 0) return false;

        // Baseline: resolve SSD reads before shared work to overlap their I/O.
        // The optional event path instead overlaps shared work with the host
        // routing decision. Both overrides avoid a second ID readback.
        const eligible = graph.streaming and !graph.quality and graph.imatrix == null and
            graph.image_count == 0 and graph.tp_world == 1 and
            !present("DS4_METAL_DISABLE_V41_SHORT_OPTIMIZATIONS");
        const asynchronous = eligible and present("DS4_METAL_V41_ASYNC_EXPERT_LOAD");
        const early = eligible and !asynchronous;
        const overlap_setting = c.getenv("DS4_METAL_V41_ROUTE_SHARED_OVERLAP");
        const overlap_shared = early and graph.scalar_decode and graph.resident_route_gap > 0 and
            !graph.hc_ffn_deferred and !graph.predict_load.active and !services.oracle_deferred and
            overlap_setting != null and std.mem.eql(u8, std.mem.span(overlap_setting), "1");
        const table = self.streamTable(layer, index, gate_row *% shape.n_ff_exp, down_row *% shape.n_embd);
        var load = std.mem.zeroes(c.metal_graph_selected_async_load);
        var selected_ids: [c.DS4_MAX_EXPERT_USED]i32 = undefined;
        // Unlike the old C early exit after a failed predicted-load join,
        // every exit joins a successfully started local job before its model
        // and tensor borrows may retire. Normal success ordering is unchanged.
        defer if (load.active) {
            _ = c.ds4_native_moe_selected_finish(&load);
            _ = c.ds4_gpu_routed_moe_set_selected_override(null, 0);
        };
        if (asynchronous) {
            var event: u64 = 0;
            // Commit the event before handing it to the worker. Later encoder
            // failure must never leave that worker waiting on unsubmitted work.
            if (c.ds4_gpu_signal_selected_readback_ready(&event) == 0 or
                c.ds4_gpu_flush_commands() == 0 or
                !c.ds4_native_moe_selected_start(&load, graph.selected, model, layer, index, event, table.gate_expert_bytes, table.down_expert_bytes)) return false;
        }
        if (graph.predict_load.active) {
            const flushed = c.ds4_gpu_flush_commands() != 0;
            // Join speculative cache mutations before exact demand loading.
            // Failure is harmless: demand loading retries its own selected set.
            _ = c.ds4_native_moe_selected_finish(&graph.predict_load);
            const cleared = c.ds4_gpu_routed_moe_set_selected_override(null, 0) != 0;
            if (!flushed or !cleared) return false;
        }
        if (early) {
            const profile: ?*c.ds4_resident_route_layer_metrics = if (services.profile == null) null else @ptrCast(services.profile);
            var started = if (profile != null) nowMilliseconds() else 0.0;
            if (services.oracle_deferred) {
                // Oracle-only bound: preserve the native selections for later
                // validation while loading exact recorded addresses now.
                if (!self.routing().captureDeferred(index, selected_ids[0..shape.n_expert_used])) return false;
            } else {
                if (overlap_shared) {
                    // Only routing must finish before the host reads IDs and
                    // affinities. Shared gate/up reads norm, not routing state,
                    // so it can run while the host applies resident policy.
                    // Pending commands retain their static/shared resources.
                    var event: u64 = 0;
                    if (c.ds4_gpu_signal_selected_readback_ready(&event) == 0 or
                        !self.sharedGateUp(layer, graph.shared_gate, graph.shared_up, graph.shared_mid, graph.norm, 1) or
                        c.ds4_gpu_commit_and_wait_selected_readback(event, "resident route/shared overlap") == 0) return false;
                    // Earlier routed outputs feed this router, so every
                    // submitted expert-cache read precedes its event. Retire
                    // those reads only: otherwise conservative in-flight pins
                    // change eviction choices under a small cache budget.
                    // The new empty batch and pending static views stay live.
                    c.ds4_gpu_retire_selected_expert_reads();
                } else if (c.ds4_gpu_end_commands() == 0) return false;
                if (profile) |metrics| {
                    metrics.router_wait_ms += nowMilliseconds() - started;
                    metrics.calls +%= 1;
                    started = nowMilliseconds();
                }
                if (c.ds4_gpu_tensor_read(graph.selected, 0, &selected_ids, @as(u64, shape.n_expert_used) * @sizeOf(i32)) == 0) return false;
                if (profile) |metrics| metrics.selected_read_ms += nowMilliseconds() - started;
                self.routing().observe(&table, index, token, selected_ids[0..shape.n_expert_used]);
                if (!self.routing().apply(layer, index, &table, selected_ids[0..shape.n_expert_used])) return false;
                started = if (profile != null) nowMilliseconds() else 0.0;
                if (c.ds4_gpu_stream_expert_cache_begin_selected_load(&table, &selected_ids, shape.n_expert_used) == 0) return false;
                if (profile) |metrics| metrics.begin_load_ms += nowMilliseconds() - started;
                // The event wait already opened the next command buffer.
                if (!overlap_shared and c.ds4_gpu_begin_commands() == 0) return false;
            }
        }
        const concurrent_eligible = shared_here and !graph.streaming and !graph.quality and
            !present("DS4_METAL_DISABLE_V41_CONCURRENT_FFN") and
            layer.ffn_gate_exps.*.type == c.DS4_TENSOR_MXFP4 and layer.ffn_down_exps.*.type == c.DS4_TENSOR_MXFP4 and
            layer.ffn_gate_shexp.*.type == c.DS4_TENSOR_Q8_0 and layer.ffn_up_shexp.*.type == c.DS4_TENSOR_Q8_0 and
            layer.ffn_down_shexp.*.type == c.DS4_TENSOR_Q8_0;
        const concurrent = concurrent_eligible and graph.tp_world != 2 and
            c.ds4_gpu_parallel_ffn_start_bf16(graph.shared_gate, graph.shared_up, graph.shared_mid, graph.shared, model.map, model.size, layer.ffn_gate_shexp.*.abs_offset, layer.ffn_up_shexp.*.abs_offset, layer.ffn_down_shexp.*.abs_offset, shape.n_embd, shape.n_ff_exp, graph.norm, shape.swiglu_clamp_exp) != 0;
        const concurrent_tp = concurrent_eligible and shared_split and balanced and
            !present("DS4_METAL_DISABLE_V41_TP_CONCURRENT_FFN") and !present("DS4_METAL_DISABLE_V41_TP_FIXED_ROUTE") and
            c.ds4_gpu_parallel_ffn_start_bf16_kslice(graph.shared_gate, graph.shared_up, graph.shared_mid, graph.shared, model.map, model.size, layer.ffn_gate_shexp.*.abs_offset, layer.ffn_up_shexp.*.abs_offset, layer.ffn_down_shexp.*.abs_offset, shape.n_embd, shape.n_ff_exp, graph.tp_rank * shared_dim, shared_dim, graph.norm, shape.swiglu_clamp_exp) != 0;
        var shared_queued = concurrent or concurrent_tp;
        defer if (shared_queued) c.ds4_gpu_parallel_ffn_abort();
        if (graph.hc_ffn_deferred) {
            if (c.ds4_gpu_dsv41_hc_block_input_rows(graph.mix, graph.x, graph.flat_norm, graph.ffn_split, graph.after_attn, graph.attn_split, model.map, model.size, layer.hc_ffn_fn.*.abs_offset, layer.hc_ffn_scale.*.abs_offset, layer.hc_ffn_base.*.abs_offset, layer.ffn_norm.*.abs_offset, shape.n_hc * shape.n_embd, 24, shape.n_embd, shape.n_hc, shape.n_hc_sinkhorn_iter, shape.hc_eps, shape.rms_eps, 1, shape.n_hc) == 0) return false;
            graph.hc_ffn_deferred = false;
        }
        if (shared_here and !shared_queued and !overlap_shared and !self.sharedSerial(layer, shared_split, shared_dim, shared_lane, graph.tp_world == 2 or !graph.scalar_decode)) return false;
        if (!shared_queued and !graph.scalar_decode) graph.shared_done = true;
        if (asynchronous) {
            const flushed = c.ds4_gpu_flush_commands() != 0;
            var loaded = c.ds4_native_moe_selected_finish(&load);
            // A worker cannot wait for still-live cache entries. Retry only
            // after joining it, using the exact IDs read by that worker.
            if (!loaded and load.ids_ok)
                loaded = c.ds4_gpu_stream_expert_cache_begin_selected_load(&table, &load.selected_ids, shape.n_expert_used) != 0 and
                    c.ds4_gpu_routed_moe_set_selected_override(&load.selected_ids, shape.n_expert_used) != 0;
            if (!flushed or !loaded) {
                _ = c.ds4_gpu_routed_moe_set_selected_override(null, 0);
                return false;
            }
        }
        if (early and ((!overlap_shared and c.ds4_gpu_flush_commands() == 0) or
            c.ds4_gpu_routed_moe_set_selected_override(&selected_ids, shape.n_expert_used) == 0)) return false;
        if (balanced) c.ds4_gpu_tp_full_expert_bind(1);
        const routed_ok = c.ds4_gpu_routed_moe_one_tensor(routed, graph.gate, graph.up, graph.mid, graph.experts, model.map, model.size, layer.ffn_gate_exps.*.abs_offset, layer.ffn_up_exps.*.abs_offset, layer.ffn_down_exps.*.abs_offset, layer.ffn_gate_exps.*.type, layer.ffn_down_exps.*.type, table.gate_expert_bytes, gate_row, table.down_expert_bytes, down_row, shape.n_embd, shape.n_ff_exp, shape.n_embd, if (balanced) graph.selected_half else graph.selected, if (balanced) graph.route_weights_half else graph.route_weights, shape.n_expert, if (balanced) shape.n_expert_used / 2 else shape.n_expert_used, shape.swiglu_clamp_exp, graph.norm, null, index, !graph.streaming) != 0;
        if (balanced) c.ds4_gpu_tp_full_expert_bind(0);
        if (!routed_ok) return false;
        if (shared_queued) {
            const complete = c.ds4_gpu_parallel_ffn_finish() != 0;
            shared_queued = false;
            if (!complete and !self.sharedSerial(layer, shared_split, shared_dim, shared_lane, true)) return false;
            graph.shared_done = true;
        }
        // Half-width shared down projections remain F32 until the rank sum;
        // whole shared projections preserve their BF16 output boundary.
        return !shared_owner or !shared_here or c.ds4_gpu_add_tensor_tp_flag(routed, routed, graph.shared, shape.n_embd, index, c.DS4_TP_GATE_FFN) != 0;
    }

    fn sharedDown(self: *const Moe, layer: *const Layer, out: ?*Tensor, mid: ?*Tensor, count: u32) bool {
        const shape = &self.ops.config.shape;
        const model = self.ops.model;
        if (count <= c.DS4_TP_BATCH_MAX_ROWS and layer.ffn_down_shexp.*.type == c.DS4_TENSOR_Q8_0)
            return c.ds4_gpu_matmul_q8_0_bf16io_tensor(out, model.map, model.size, layer.ffn_down_shexp.*.abs_offset, shape.n_ff_exp, shape.n_embd, mid, count) != 0;
        return c.ds4_gpu_dsv41_quantize(mid, shape.n_ff_exp, count, c.DS4_V41_BF16) != 0 and
            self.ops.matmulBatch(out, layer.ffn_down_shexp, mid, count, true);
    }

    fn sharedSerial(self: *const Moe, layer: *const Layer, split: bool, count: u32, lane: u64, down: bool) bool {
        const g = self.graph;
        const shape = &self.ops.config.shape;
        const m = self.ops.model;
        if (split) {
            if (c.ds4_gpu_dsv41_shared_swiglu(g.shared_mid, g.shared_gate, g.shared_up, m.map, m.size, layer.ffn_gate_shexp.*.abs_offset + lane, layer.ffn_up_shexp.*.abs_offset + lane, shape.n_embd, count, g.norm, shape.swiglu_clamp_exp, 1) == 0) return false;
        } else if (!self.sharedGateUp(layer, g.shared_gate, g.shared_up, g.shared_mid, g.norm, 1)) return false;
        if (!down) return true;
        return if (split) support.bf16(g.shared_mid, count) and
            c.ds4_gpu_matmul_q8_0_kslice_tensor(g.shared, m.map, m.size, layer.ffn_down_shexp.*.abs_offset, shape.n_ff_exp, @as(u64, g.tp_rank) * count, count, shape.n_embd, g.shared_mid, 0) != 0 else self.sharedDown(layer, g.shared, g.shared_mid, 1);
    }

    pub fn batch(self: *const Moe, layer: *const Layer, index: u32, count: u32, shared_owner: bool, force_resident: bool) bool {
        const graph = self.graph;
        const row = &graph.batch;
        const model = self.ops.model;
        const shape = &self.ops.config.shape;
        const gate_row = expertRowBytes(layer.ffn_gate_exps);
        const down_row = expertRowBytes(layer.ffn_down_exps);
        var mid_f16 = false;
        var ok = self.routerBatch(layer, count) and self.routeBatch(layer, count);
        if (ok) ok = self.routing().prefill(layer, index, count);
        const overlap_shared = !force_resident and graph.streaming and present("DS4_METAL_V41_BATCH_SHARED_OVERLAP") and
            c.ds4_gpu_stream_expert_batch_supported(count, shape.n_expert, shape.n_expert_used, layer.ffn_gate_exps.*.type, layer.ffn_down_exps.*.type) != 0;
        // Even a failed router/shared/routed operation must clear the scoped
        // loader hint. The following call must not inherit an old selected set.
        defer if (overlap_shared) c.ds4_gpu_stream_expert_batch_set_ready(null);
        if (ok and overlap_shared) {
            ok = c.ds4_gpu_end_commands() != 0 and c.ds4_gpu_begin_commands() != 0;
            if (ok) c.ds4_gpu_stream_expert_batch_set_ready(row.selected);
        }
        // Metal's shared owner is static layer parity. Spark-only balancing
        // remains outside this backend.
        const other_owner = shared_owner and graph.tp_rank != (index & 1);
        graph.shared_done = !graph.defer_shared_down;
        ok = ok and (other_owner or (self.sharedGateUp(layer, row.shared_gate, row.shared_up, row.shared_mid, row.norm, count) and
            (graph.defer_shared_down or self.sharedDown(layer, row.shared, row.shared_mid, count)))) and
            c.ds4_gpu_routed_moe_batch_tensor(row.routed, row.gate, row.up, row.mid, row.experts, model.map, model.size, layer.ffn_gate_exps.*.abs_offset, layer.ffn_up_exps.*.abs_offset, layer.ffn_down_exps.*.abs_offset, layer.ffn_gate_exps.*.type, layer.ffn_down_exps.*.type, gate_row *% shape.n_ff_exp, gate_row, down_row *% shape.n_embd, down_row, shape.n_embd, shape.n_ff_exp, shape.n_embd, row.selected, row.route_weights, shape.n_expert, shape.n_expert_used, shape.swiglu_clamp_exp, row.norm, index, count, &mid_f16, force_resident and !graph.verify_prefix) != 0 and
            (!shared_owner or other_owner or c.ds4_gpu_add_tensor(row.routed, row.routed, row.shared, count *% shape.n_embd) != 0) and
            self.ops.sumPartialBatch(graph, row.routed, index, c.DS4_TP_GATE_FFN, count);
        return ok;
    }

    // Route/shared work through <=8-row decode kernels. F32 intermediates and
    // reductions keep the scalar BF16/router boundaries; all other graph state
    // is borrowed. Never pass these shallow tile records to Graph.deinit.
    pub fn exactTiles(self: *const Moe, layer: *const Layer, index: u32, count: u32) bool {
        var first: u32 = 0;
        while (first < count) : (first += 8) {
            if (!self.exactTile(layer, index, first, @min(count - first, 8))) return false;
        }
        return true;
    }

    fn exactTile(self: *const Moe, layer: *const Layer, index: u32, first: u32, count: u32) bool {
        var tile = self.graph.*;
        tile.pos +%= first;
        tile.batch = std.mem.zeroes(c.ds41_prefill_row);
        tile.prefill_tokens = c.ds4_gpu_tensor_view(self.graph.prefill_tokens, @as(u64, first) * 4, @as(u64, count) * 4);
        defer {
            inline for (comptime blk: {
                @setEvalBranchQuota(10000);
                break :blk std.meta.fieldNames(c.ds41_prefill_row);
            }) |name| c.ds4_gpu_tensor_free(@field(tile.batch, name));
            c.ds4_gpu_tensor_free(tile.prefill_tokens);
        }
        if (tile.prefill_tokens == null) return false;
        const plan = Layout.fromRecord(self.ops.config, self.graph);
        inline for (comptime std.meta.fieldNames(c.ds41_prefill_row)) |name| {
            const width = plan.rowWidth(@field(layout.RowField, name));
            @field(tile.batch, name) = c.ds4_gpu_tensor_view(@field(self.graph.batch, name), @as(u64, first) * width * 4, @as(u64, count) * width * 4) orelse return false;
        }
        const controller = Moe{ .ops = self.ops, .graph = &tile };
        return controller.batch(layer, index, count, false, true);
    }

    pub fn finish(self: *const Moe, index: u32) bool {
        const graph = self.graph;
        const width = self.ops.config.shape.n_embd;
        if (graph.tp_world != 2) return graph.scalar_decode or
            (c.ds4_gpu_add_tensor(graph.block, graph.routed, graph.shared, width) != 0 and support.bf16(graph.block, width));
        const shared_owner = !present("DS4_METAL_DISABLE_V41_TP_SHARED_OWNER");
        const sum = if (shared_owner) graph.block else graph.routed;
        const routed = if (!c.ds4_native_graph_trace_active() and !present("DS4_METAL_DISABLE_V41_TP_SLAB_WRITE")) graph.tp_out[index * c.DS4_TP_GATES_PER_LAYER + c.DS4_TP_GATE_FFN] else sum;
        if (graph.tp_gate_fused) return self.ops.gatePartial(graph, routed, index, c.DS4_TP_GATE_FFN);
        return self.ops.sumPartialInto(graph, routed, sum, index, c.DS4_TP_GATE_FFN) and
            (shared_owner or c.ds4_gpu_add_tensor(graph.block, sum, graph.shared, width) != 0) and support.bf16(graph.block, width);
    }
};
