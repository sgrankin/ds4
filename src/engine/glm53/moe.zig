//! GLM's sigmoid router and dense/shared/routed expert dispatch. Storage-cache
//! services borrow the same selected-ID tape; only this owner orders numerical
//! work and overlaps a selected load with the shared expert.
const c = @import("gguf_c");
const std = @import("std");
const Ops = @import("ops.zig");
const Seed = @import("cache_seed.zig");
const Graph = c.ds4_glm_gpu_graph;
const Model = c.ds4_model;
const Layer = c.ds4_layer_weights;
const Tensor = c.ds4_gpu_tensor;
const Weight = c.ds4_tensor;

const routed_skip: u32 = 1 << 4;
const shared_skip: u32 = 1 << 5;

const Expert = struct {
    row_bytes: u64,
    bytes: u64,

    fn layout(weight: *const Weight) ?Expert {
        var row: u64 = 0;
        if (!c.ds4_gguf_tensor_nbytes(weight.type, weight.dim[0], &row)) return null;
        return .{ .row_bytes = row, .bytes = row * weight.dim[1] };
    }
};

fn generic(weights: *const Layer) bool {
    return weights.ffn_gate_exps.*.type == c.DS4_TENSOR_IQ2_XXS;
}

pub fn routedOne(graph: *const Graph, model: *const Model, weights: *const Layer, layer: u32, output: ?*Tensor, mid: ?*Tensor, selected: ?*const Tensor, routing_weights: ?*const Tensor, input: ?*const Tensor, force_resident: bool) bool {
    if (graph.tp_world > 1) return false;
    const shape = &graph.config.shape;
    const gate = weights.ffn_gate_exps;
    const up = weights.ffn_up_exps;
    const down = weights.ffn_down_exps;
    const gate_size = Expert.layout(gate) orelse return false;
    const up_size = Expert.layout(up) orelse return false;
    const down_size = Expert.layout(down) orelse return false;
    if (generic(weights)) {
        if (graph.routed_gate == null or graph.routed_up == null or graph.routed_down == null or gate.*.type != up.*.type) return false;
        return c.ds4_gpu_routed_moe_one_tensor(
            output,
            graph.routed_gate,
            graph.routed_up,
            mid,
            graph.routed_down,
            model.map,
            model.size,
            gate.*.abs_offset,
            up.*.abs_offset,
            down.*.abs_offset,
            gate.*.type,
            down.*.type,
            gate_size.bytes,
            gate_size.row_bytes,
            down_size.bytes,
            down_size.row_bytes,
            shape.n_embd,
            shape.n_ff_exp,
            shape.n_embd,
            selected,
            routing_weights,
            shape.n_expert,
            shape.n_expert_used,
            shape.swiglu_clamp_exp,
            input,
            null,
            layer,
            force_resident,
        ) != 0;
    }
    return c.ds4_gpu_glm_routed_moe_one_tensor(
        output,
        mid,
        model.map,
        model.size,
        gate.*.abs_offset,
        up.*.abs_offset,
        down.*.abs_offset,
        gate.*.type,
        up.*.type,
        down.*.type,
        gate_size.bytes,
        gate_size.row_bytes,
        up_size.bytes,
        up_size.row_bytes,
        down_size.bytes,
        down_size.row_bytes,
        shape.n_embd,
        shape.n_ff_exp,
        shape.n_embd,
        selected,
        routing_weights,
        shape.n_expert,
        shape.n_expert_used,
        shape.swiglu_clamp_exp,
        layer,
        input,
        force_resident,
    ) != 0;
}

pub fn routedRows(graph: *Graph, model: *const Model, weights: *const Layer, layer: u32, output: ?*Tensor, mid: ?*Tensor, selected: ?*const Tensor, routing_weights: ?*const Tensor, input: ?*const Tensor, rows: u32, mid_stride: u32, force_resident: bool, direct_scalar_q4: bool) bool {
    if (graph.tp_world > 1) return false;
    graph.batch_routed_mid_is_f16 = false;
    const shape = &graph.config.shape;
    const gate = weights.ffn_gate_exps;
    const up = weights.ffn_up_exps;
    const down = weights.ffn_down_exps;
    const gate_size = Expert.layout(gate) orelse return false;
    const up_size = Expert.layout(up) orelse return false;
    const down_size = Expert.layout(down) orelse return false;
    if (generic(weights)) {
        if (graph.batch_routed_gate == null or graph.batch_routed_up == null or graph.batch_routed_down == null or gate.*.type != up.*.type) return false;
        return c.ds4_gpu_routed_moe_batch_tensor(
            output,
            graph.batch_routed_gate,
            graph.batch_routed_up,
            mid,
            graph.batch_routed_down,
            model.map,
            model.size,
            gate.*.abs_offset,
            up.*.abs_offset,
            down.*.abs_offset,
            gate.*.type,
            down.*.type,
            gate_size.bytes,
            gate_size.row_bytes,
            down_size.bytes,
            down_size.row_bytes,
            shape.n_embd,
            shape.n_ff_exp,
            shape.n_embd,
            selected,
            routing_weights,
            shape.n_expert,
            shape.n_expert_used,
            shape.swiglu_clamp_exp,
            input,
            layer,
            rows,
            &graph.batch_routed_mid_is_f16,
            force_resident,
        ) != 0;
    }
    if (direct_scalar_q4) {
        return c.ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor(
            output,
            mid,
            model.map,
            model.size,
            gate.*.abs_offset,
            up.*.abs_offset,
            down.*.abs_offset,
            gate.*.type,
            up.*.type,
            down.*.type,
            gate_size.bytes,
            gate_size.row_bytes,
            up_size.bytes,
            up_size.row_bytes,
            down_size.bytes,
            down_size.row_bytes,
            shape.n_embd,
            shape.n_ff_exp,
            shape.n_embd,
            selected,
            routing_weights,
            shape.n_expert,
            shape.n_expert_used,
            shape.swiglu_clamp_exp,
            layer,
            input,
            rows,
            mid_stride,
        ) != 0;
    }
    return c.ds4_gpu_glm_routed_moe_batch_tensor(
        output,
        mid,
        model.map,
        model.size,
        gate.*.abs_offset,
        up.*.abs_offset,
        down.*.abs_offset,
        gate.*.type,
        up.*.type,
        down.*.type,
        gate_size.bytes,
        gate_size.row_bytes,
        up_size.bytes,
        up_size.row_bytes,
        down_size.bytes,
        down_size.row_bytes,
        shape.n_embd,
        shape.n_ff_exp,
        shape.n_embd,
        selected,
        routing_weights,
        shape.n_expert,
        shape.n_expert_used,
        shape.swiglu_clamp_exp,
        layer,
        input,
        rows,
        mid_stride,
        force_resident,
    ) != 0;
}

pub fn shared(graph: *Graph, model: *const Model, weights: *const Layer) bool {
    const shape = &graph.config.shape;
    const q8 = weights.ffn_gate_shexp.*.type == c.DS4_TENSOR_Q8_0 and weights.ffn_up_shexp.*.type == c.DS4_TENSOR_Q8_0;
    if (q8) {
        // This shared-only fusion is distinct from the dense layer's three
        // output fusion below. Preserve the original dispatch and reduction.
        if (c.ds4_gpu_shared_mid_swiglu_q8_0_tensor(
            graph.ffn_mid,
            model.map,
            model.size,
            weights.ffn_gate_shexp.*.abs_offset,
            weights.ffn_up_shexp.*.abs_offset,
            shape.n_embd,
            shape.n_ff_exp,
            graph.ffn_norm,
            shape.swiglu_clamp_exp,
        ) == 0) return false;
    } else {
        if (!Ops.matmul(graph.ffn_gate, model, weights.ffn_gate_shexp, shape.n_embd, shape.n_ff_exp, graph.ffn_norm, 1) or
            !Ops.matmul(graph.ffn_up, model, weights.ffn_up_shexp, shape.n_embd, shape.n_ff_exp, graph.ffn_norm, 1) or
            c.ds4_gpu_swiglu_tensor(graph.ffn_mid, graph.ffn_gate, graph.ffn_up, shape.n_ff_exp, shape.swiglu_clamp_exp, 1) == 0) return false;
    }
    return Ops.matmul(graph.ffn_sum, model, weights.ffn_down_shexp, shape.n_ff_exp, shape.n_embd, graph.ffn_mid, 1);
}

fn dense(graph: *Graph, model: *const Model, weights: *const Layer) bool {
    const shape = &graph.config.shape;
    const hidden: u32 = @intCast(weights.ffn_gate.*.dim[1]);
    const fused = weights.ffn_gate.*.type == c.DS4_TENSOR_Q8_0 and weights.ffn_up.*.type == c.DS4_TENSOR_Q8_0 and
        c.ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
            graph.ffn_gate,
            graph.ffn_up,
            graph.ffn_mid,
            model.map,
            model.size,
            weights.ffn_gate.*.abs_offset,
            weights.ffn_up.*.abs_offset,
            shape.n_embd,
            hidden,
            graph.ffn_norm,
            shape.swiglu_clamp_exp,
        ) != 0;
    if (!fused and
        (!Ops.matmul(graph.ffn_gate, model, weights.ffn_gate, shape.n_embd, hidden, graph.ffn_norm, 1) or
            !Ops.matmul(graph.ffn_up, model, weights.ffn_up, shape.n_embd, hidden, graph.ffn_norm, 1) or
            c.ds4_gpu_swiglu_tensor(graph.ffn_mid, graph.ffn_gate, graph.ffn_up, hidden, shape.swiglu_clamp_exp, 1) == 0)) return false;
    return Ops.matmul(graph.ffn_out, model, weights.ffn_down, hidden, shape.n_embd, graph.ffn_mid, 1) and
        c.ds4_gpu_tensor_copy(graph.next, 0, graph.ffn_out, 0, @as(u64, shape.n_embd) * @sizeOf(f32)) != 0;
}

fn sparse(graph: *Graph, model: *const Model, weights: *const Layer, layer: u32, ablate: u32) bool {
    const shape = &graph.config.shape;
    if (c.ds4_gpu_matmul_f32_tensor(graph.router_logits, model.map, model.size, weights.ffn_gate_inp.*.abs_offset, shape.n_embd, shape.n_expert, graph.ffn_norm, 1) == 0 or
        c.ds4_gpu_glm_router_select_tensor(graph.router_selected, graph.router_weights, graph.router_probs, model.map, model.size, weights.ffn_exp_probs_b.*.abs_offset, graph.router_logits, shape.n_expert, shape.n_expert_used, shape.expert_weight_scale) == 0) return false;

    const gate_type = weights.ffn_gate_exps.*.type;
    const uniform = gate_type == weights.ffn_up_exps.*.type and gate_type == weights.ffn_down_exps.*.type and
        (gate_type == c.DS4_TENSOR_Q2_K or gate_type == c.DS4_TENSOR_Q4_K);
    const selected_cache = c.ds4_native_glm_stream_selected_cache(graph, weights, layer) and (generic(weights) or uniform);
    var load = std.mem.zeroes(c.metal_graph_selected_async_load);
    var ok = true;
    if (selected_cache) {
        const gate_size = Expert.layout(weights.ffn_gate_exps) orelse return false;
        const down_size = Expert.layout(weights.ffn_down_exps) orelse return false;
        const table = c.ds4_gpu_stream_expert_table{
            .model_map = model.map,
            .model_size = model.size,
            .layer = layer,
            .n_total_expert = shape.n_expert,
            .gate_offset = weights.ffn_gate_exps.*.abs_offset,
            .up_offset = weights.ffn_up_exps.*.abs_offset,
            .down_offset = weights.ffn_down_exps.*.abs_offset,
            .gate_expert_bytes = gate_size.bytes,
            .down_expert_bytes = down_size.bytes,
        };
        var event: u64 = 0;
        const async_selected = generic(weights) and graph.config.stream_selected_async;
        if (async_selected and c.ds4_gpu_signal_selected_readback_ready(&event) != 0) {
            _ = c.ds4_native_moe_selected_start(&load, graph.router_selected, model, weights, layer, event, gate_size.bytes, down_size.bytes);
            if (load.active) ok = c.ds4_gpu_flush_commands() != 0;
        }
        if (!load.active) {
            if (async_selected and event != 0) ok = c.ds4_gpu_wait_selected_readback_ready(event, "selected-id sync expert load fallback") != 0;
            if (ok) ok = c.ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(&table, graph.router_selected, shape.n_expert_used) != 0;
        }
    }
    // Once a worker starts it must be joined even when shared encoding fails.
    // Keep that join explicit rather than allowing an early return to leak it.
    if (ok and selected_cache) ok = shared(graph, model, weights);
    if (load.active) {
        const flushed = c.ds4_gpu_flush_commands() != 0;
        const joined = c.ds4_native_moe_selected_finish(&load);
        ok = ok and flushed and joined;
    }
    if (ok and ablate & routed_skip == 0)
        ok = routedOne(graph, model, weights, layer, graph.ffn_out, graph.ffn_mid, graph.router_selected, graph.router_weights, graph.ffn_norm, graph.ssd_streaming and !selected_cache);
    if (ok and !selected_cache and ablate & shared_skip == 0) ok = shared(graph, model, weights);
    return ok and c.ds4_gpu_add_tensor(graph.next, graph.ffn_out, graph.ffn_sum, shape.n_embd) != 0;
}

pub fn decode(graph: *Graph, model: *const Model, weights: *const Layer, layer: u32, position: u32, ablate: u32) bool {
    if (graph.tp_world > 1 or graph.placement != null or graph.imatrix != null) return false;
    const shape = &graph.config.shape;
    const ok = if (layer < shape.n_leading_dense) dense(graph, model, weights) else sparse(graph, model, weights, layer, ablate);
    if (!ok) return false;
    c.ds4_native_graph_debug_tensor("ffn_out", graph.next, shape.n_embd, layer, position);
    if (graph.directional_steering_ffn_scale != 0) {
        if (graph.directional_steering_dirs_by_tier[0] == null or
            c.ds4_gpu_directional_steering_project_tensor(graph.next, graph.directional_steering_dirs_by_tier[0], layer, shape.n_embd, 1, graph.directional_steering_ffn_scale) == 0) return false;
    }
    return c.ds4_gpu_hc_expand_tensor(graph.hc_next, graph.next, graph.hc_after_attn, graph.hc_post, graph.hc_comb, shape.n_embd, shape.n_hc) != 0;
}

fn gateUpRows(graph: *Graph, model: *const Model, gate: *const Weight, up: *const Weight, mid: ?*Tensor, hidden: u32, count: u32) bool {
    const shape = &graph.config.shape;
    const fused = gate.type == c.DS4_TENSOR_Q8_0 and up.type == c.DS4_TENSOR_Q8_0 and
        (if (count == 1)
            c.ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(graph.batch_ffn_gate, graph.batch_ffn_up, mid, model.map, model.size, gate.abs_offset, up.abs_offset, shape.n_embd, hidden, graph.batch_ffn_norm, shape.swiglu_clamp_exp) != 0
        else
            c.ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(graph.batch_ffn_gate, graph.batch_ffn_up, mid, model.map, model.size, gate.abs_offset, up.abs_offset, shape.n_embd, hidden, graph.batch_ffn_norm, count, shape.swiglu_clamp_exp) != 0);
    return fused or (Ops.matmul(graph.batch_ffn_gate, model, gate, shape.n_embd, hidden, graph.batch_ffn_norm, count) and
        Ops.matmul(graph.batch_ffn_up, model, up, shape.n_embd, hidden, graph.batch_ffn_norm, count) and
        c.ds4_gpu_swiglu_tensor(mid, graph.batch_ffn_gate, graph.batch_ffn_up, count * hidden, shape.swiglu_clamp_exp, 1) != 0);
}

/// mHC has already prepared batch_ffn_norm; this returns only the FFN block,
/// leaving the caller to apply steering and expand its hyperconnections.
pub fn encodeRows(graph: *Graph, model: *const Model, weights: *const Layer, layer: u32, position: u32, count: u32, output: ?*Tensor, full_layer_prefill: bool, ablate: u32) bool {
    if (graph.tp_world > 1 or graph.placement != null or count == 0 or count > graph.config.prefill_chunk or output == null) return false;
    const shape = &graph.config.shape;
    const elements = count * shape.n_embd;
    c.ds4_native_graph_debug_tensor("glm_ffn_norm", graph.batch_ffn_norm, elements, layer, position);
    if (layer < shape.n_leading_dense) {
        const hidden: u32 = @intCast(weights.ffn_gate.*.dim[1]);
        if (!gateUpRows(graph, model, weights.ffn_gate, weights.ffn_up, graph.batch_ffn_mid, hidden, count)) return false;
        c.ds4_native_graph_debug_tensor("glm53_dense_ffn_mid", graph.batch_ffn_mid, @as(u64, count) * hidden, layer, position);
        if (!Ops.matmul(graph.batch_ffn_out, model, weights.ffn_down, hidden, shape.n_embd, graph.batch_ffn_mid, count)) return false;
        c.ds4_native_graph_debug_tensor("glm53_dense_ffn_out", graph.batch_ffn_out, elements, layer, position);
        return c.ds4_gpu_tensor_copy(output, 0, graph.batch_ffn_out, 0, @as(u64, elements) * @sizeOf(f32)) != 0;
    }
    if (c.ds4_gpu_matmul_f32_tensor(graph.batch_router_logits, model.map, model.size, weights.ffn_gate_inp.*.abs_offset, shape.n_embd, shape.n_expert, graph.batch_ffn_norm, count) == 0 or
        c.ds4_gpu_glm_router_select_batch_tensor(graph.batch_router_selected, graph.batch_router_weights, graph.batch_router_probs, model.map, model.size, weights.ffn_exp_probs_b.*.abs_offset, graph.batch_router_logits, shape.n_expert, shape.n_expert_used, shape.expert_weight_scale, count) == 0) return false;
    c.ds4_native_graph_debug_tensor("glm_ffn_router_logits", graph.batch_router_logits, @as(u64, count) * shape.n_expert, layer, position);
    c.ds4_native_graph_debug_tensor("glm_ffn_router_probs", graph.batch_router_probs, @as(u64, count) * shape.n_expert, layer, position);
    c.ds4_native_graph_debug_i32_tensor("glm_ffn_router_selected", graph.batch_router_selected, @as(u64, count) * shape.n_expert_used, layer, position);
    c.ds4_native_graph_debug_tensor("glm_ffn_router_weights", graph.batch_router_weights, @as(u64, count) * shape.n_expert_used, layer, position);
    const gate_size = Expert.layout(weights.ffn_gate_exps) orelse return false;
    const down_size = Expert.layout(weights.ffn_down_exps) orelse return false;
    if (!Seed.seedLayer(graph, model, graph.weights, weights, layer, count, gate_size.bytes, down_size.bytes, full_layer_prefill)) return false;
    if ((count > 8 or ablate & routed_skip == 0) and !routedRows(graph, model, weights, layer, graph.batch_ffn_out, graph.batch_ffn_mid, graph.batch_router_selected, graph.batch_router_weights, graph.batch_ffn_norm, count, @intCast(graph.ffn_mid_elems), full_layer_prefill, false)) return false;
    c.ds4_native_graph_debug_tensor("glm_ffn_routed_out", graph.batch_ffn_out, elements, layer, position);
    if (!gateUpRows(graph, model, weights.ffn_gate_shexp, weights.ffn_up_shexp, graph.batch_shared_mid, shape.n_ff_exp, count) or
        !Ops.matmul(graph.batch_attn_out, model, weights.ffn_down_shexp, shape.n_ff_exp, shape.n_embd, graph.batch_shared_mid, count)) return false;
    c.ds4_native_graph_debug_tensor("glm_ffn_shared_out", graph.batch_attn_out, elements, layer, position);
    if (c.ds4_gpu_add_tensor(output, graph.batch_ffn_out, graph.batch_attn_out, elements) == 0) return false;
    c.ds4_native_graph_debug_tensor("glm_ffn_next", output, elements, layer, position);
    return true;
}
