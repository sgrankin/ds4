//! GLM-5.3's projected KDA attention. The Metal primitive owns the causal
//! convolution and recurrent update; this owner keeps projection/command order.
const c = @import("gguf_c");
const Config = @import("config.zig");
const Ops = @import("ops.zig");
const Graph = c.ds4_glm_gpu_graph;
const Model = c.ds4_model;
const Layer = c.ds4_layer_weights;
const Tensor = c.ds4_gpu_tensor;
const Weight = c.ds4_tensor;

const Buffers = struct {
    q: ?*Tensor,
    k: ?*Tensor,
    v: ?*Tensor,
    lowrank: ?*Tensor,
    gate: ?*Tensor,
    beta: ?*Tensor,
    output_gate: ?*Tensor,
    output: ?*Tensor,
    input: ?*const Tensor,
};

fn buffers(comptime batch: bool, graph: *Graph) Buffers {
    return if (batch) .{
        .q = graph.batch_kda_q,
        .k = graph.batch_kda_k,
        .v = graph.batch_kda_v,
        .lowrank = graph.batch_kda_lowrank,
        .gate = graph.batch_kda_raw_gate,
        .beta = graph.batch_kda_raw_beta,
        .output_gate = graph.batch_kda_output_gate,
        .output = graph.batch_kda_out,
        .input = graph.batch_attn_norm,
    } else .{
        .q = graph.kda_q,
        .k = graph.kda_k,
        .v = graph.kda_v,
        .lowrank = graph.kda_lowrank,
        .gate = graph.kda_raw_gate,
        .beta = graph.kda_raw_beta,
        .output_gate = graph.kda_output_gate,
        .output = graph.kda_out,
        .input = graph.attn_norm,
    };
}

const Projection = struct {
    stage: [*:0]const u8,
    trace: ?[*:0]const u8,
    output: ?*Tensor,
    weight: ?*const Weight,
    inputs: u32,
    outputs: u32,
    input: ?*const Tensor,
};

fn failed(layer: u32, stage: [*:0]const u8, weight: ?*const Weight, position: u32, rows: u32) bool {
    if (weight) |tensor| {
        _ = c.fprintf(c.__stderrp, "ds4: GLM-5.3 KDA failed at layer %u stage '%s' on tensor %.*s (type %u; pos %u, rows %u)\n", layer, stage, @as(c_int, @intCast(tensor.name.len)), tensor.name.ptr, tensor.type, position, rows);
    } else {
        _ = c.fprintf(c.__stderrp, "ds4: GLM-5.3 KDA failed at layer %u stage '%s' (pos %u, rows %u)\n", layer, stage, position, rows);
    }
    return false;
}

pub fn project(comptime batch: bool, graph: *Graph, model: *const Model, weights: *const Layer, layer: u32, position: u32, rows: u32) bool {
    if (layer >= c.DS4_MAX_LAYER or rows == 0) return false;
    const shape = &graph.config.shape;
    const projection = Config.projection(&graph.config);
    const work = buffers(batch, graph);

    // Retain Metal's fused BF16 Q/K/V decode path. Prefill uses its established
    // row matmuls, and a failed fused dispatch falls back to those projections.
    const paired = !batch and graph.config.fused_bf16_qkv and
        weights.kda_q.*.type == c.DS4_TENSOR_BF16 and
        weights.kda_k.*.type == c.DS4_TENSOR_BF16 and
        weights.kda_v.*.type == c.DS4_TENSOR_BF16 and
        c.ds4_gpu_glm53_matmul_bf16_qkv(work.q, work.k, work.v, model.map, model.size, weights.kda_q.*.abs_offset, weights.kda_k.*.abs_offset, weights.kda_v.*.abs_offset, shape.n_embd, projection, work.input) != 0;

    const projections = [_]Projection{
        .{ .stage = "Q projection", .trace = "glm53_kda_q_ready", .output = work.q, .weight = weights.kda_q, .inputs = shape.n_embd, .outputs = projection, .input = work.input },
        .{ .stage = "K projection", .trace = "glm53_kda_k_ready", .output = work.k, .weight = weights.kda_k, .inputs = shape.n_embd, .outputs = projection, .input = work.input },
        .{ .stage = "V projection", .trace = "glm53_kda_v_ready", .output = work.v, .weight = weights.kda_v, .inputs = shape.n_embd, .outputs = projection, .input = work.input },
        .{ .stage = "decay low-rank projection", .trace = null, .output = work.lowrank, .weight = weights.kda_f_a, .inputs = shape.n_embd, .outputs = shape.n_kda_head_dim, .input = work.input },
        .{ .stage = "decay gate projection", .trace = "glm53_kda_raw_gate_ready", .output = work.gate, .weight = weights.kda_f_b, .inputs = shape.n_kda_head_dim, .outputs = projection, .input = work.lowrank },
        .{ .stage = "beta projection", .trace = "glm53_kda_raw_beta_ready", .output = work.beta, .weight = weights.kda_beta, .inputs = shape.n_embd, .outputs = shape.n_kda_head, .input = work.input },
        .{ .stage = "output-gate low-rank projection", .trace = null, .output = work.lowrank, .weight = weights.kda_g_a, .inputs = shape.n_embd, .outputs = shape.n_kda_head_dim, .input = work.input },
        .{ .stage = "output-gate projection", .trace = "glm53_kda_output_gate_ready", .output = work.output_gate, .weight = weights.kda_g_b, .inputs = shape.n_kda_head_dim, .outputs = projection, .input = work.lowrank },
    };
    // Unroll this fixed sequence so the decode path does not rebuild descriptors
    // or loop over runtime dimensions while encoding every layer.
    inline for (projections, 0..) |step, index| {
        if (!paired or index >= 3) {
            if (!Ops.matmul(step.output, model, step.weight, step.inputs, step.outputs, step.input, rows))
                return failed(layer, step.stage, step.weight, position, rows);
            if (batch) if (step.trace) |label|
                c.ds4_native_graph_debug_tensor(label, step.output, @as(u64, rows) * step.outputs, layer, position);
        }
    }

    return true;
}

pub fn attention(comptime batch: bool, graph: *Graph, model: *const Model, weights: *const Layer, layer: u32, position: u32, rows: u32, output: ?*Tensor) bool {
    if (layer >= c.DS4_MAX_LAYER or rows == 0 or output == null or
        graph.layer_kda_conv_state[layer] == null or graph.layer_kda_recurrent_state[layer] == null) return false;
    if (!project(batch, graph, model, weights, layer, position, rows)) return false;
    const shape = &graph.config.shape;
    const projection = Config.projection(&graph.config);
    const work = buffers(batch, graph);
    const recurrent_ok = if (batch)
        c.ds4_gpu_glm53_kda_prefill(work.output, graph.layer_kda_conv_state[layer], graph.layer_kda_recurrent_state[layer], work.q, work.k, work.v, work.gate, work.beta, work.output_gate, model.map, model.size, weights.kda_q_conv.*.abs_offset, weights.kda_k_conv.*.abs_offset, weights.kda_v_conv.*.abs_offset, weights.kda_a_log.*.abs_offset, weights.kda_dt_bias.*.abs_offset, weights.kda_o_norm.*.abs_offset, shape.n_kda_head, rows, shape.kda_gate_lower_bound, shape.rms_eps) != 0
    else
        c.ds4_gpu_glm53_kda_decode(work.output, graph.layer_kda_conv_state[layer], graph.layer_kda_recurrent_state[layer], work.q, work.k, work.v, work.gate, work.beta, work.output_gate, model.map, model.size, weights.kda_q_conv.*.abs_offset, weights.kda_k_conv.*.abs_offset, weights.kda_v_conv.*.abs_offset, weights.kda_a_log.*.abs_offset, weights.kda_dt_bias.*.abs_offset, weights.kda_o_norm.*.abs_offset, shape.n_kda_head, 1, shape.kda_gate_lower_bound, shape.rms_eps) != 0;
    if (!recurrent_ok) return failed(layer, "KDA recurrence", null, position, rows);
    if (batch) c.ds4_native_graph_debug_tensor("glm53_kda_out_ready", work.output, @as(u64, rows) * projection, layer, position);
    if (!Ops.matmul(output, model, weights.kda_output, projection, shape.n_embd, work.output, rows))
        return failed(layer, "output projection", weights.kda_output, position, rows);
    if (batch) c.ds4_native_graph_debug_tensor("glm53_kda_attn_out_ready", output, @as(u64, rows) * shape.n_embd, layer, position);
    return true;
}
