//! Prompt residual captures use the same HC mean as the other native families.
const c = @import("gguf_c");
const capture_owner = @import("../steering_capture.zig");
const Graph = c.ds4_qwen4_gpu_graph;

pub fn lastFfn(graph: *Graph, layer: u32, rows: u32) bool {
    return capture_owner.capture(graph.R, layer, rows, graph.config.shape.n_embd, graph.config.shape.n_hc, "residual", 0);
}

pub fn lastAttention(graph: *Graph, layer: u32, rows: u32) bool {
    return capture_owner.capture(graph.R, layer, rows, graph.config.shape.n_embd, graph.config.shape.n_hc, "attn_residual", 0);
}
