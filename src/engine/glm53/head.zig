//! The shared language head consumes the mean of the four mHC streams.
const c = @import("gguf_c");
const Ops = @import("ops.zig");
const Graph = c.ds4_glm_gpu_graph;
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const Tensor = c.ds4_gpu_tensor;

pub fn from(graph: *Graph, model: *const Model, weights: *const Weights, hidden: ?*const Tensor) bool {
    const shape = &graph.config.shape;
    return c.ds4_gpu_rms_norm_weight_tensor(graph.output_norm, hidden, model.map, model.size, weights.output_norm.*.abs_offset, shape.n_embd, shape.rms_eps) != 0 and
        Ops.matmul(graph.logits, model, weights.output, shape.n_embd, shape.n_vocab, graph.output_norm, 1);
}

pub fn encode(graph: *Graph, model: *const Model, weights: *const Weights) bool {
    const shape = &graph.config.shape;
    if (c.ds4_gpu_hc_weighted_sum_tensor(graph.hc_output, graph.hc_cur, graph.hc_mean_weights, shape.n_embd, shape.n_hc) == 0) return false;
    return from(graph, model, weights, graph.hc_output);
}

pub fn forward(graph: *Graph, model: *const Model, weights: *const Weights, hidden: ?*const Tensor, logits: [*]f32) bool {
    if (hidden == null) return false;
    const shape = &graph.config.shape;
    var plain = hidden;
    var ok = c.ds4_gpu_begin_commands() != 0;
    if (ok and (hidden == graph.hc_cur or hidden == graph.hc_next)) {
        ok = c.ds4_gpu_hc_weighted_sum_tensor(graph.hc_output, hidden, graph.hc_mean_weights, shape.n_embd, shape.n_hc) != 0;
        plain = graph.hc_output;
    }
    if (ok) ok = from(graph, model, weights, plain);
    if (ok) {
        ok = c.ds4_gpu_end_commands() != 0;
    } else {
        _ = c.ds4_gpu_synchronize();
    }
    return ok and c.ds4_gpu_tensor_read(graph.logits, 0, logits, @as(u64, shape.n_vocab) * @sizeOf(f32)) != 0;
}
