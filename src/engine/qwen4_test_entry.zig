//! Test-only C access to the actual native owners. Linked by qualification
//! fixtures, never production: no copied engine or alternate implementation.
const c = @import("gguf_c");
const ops = @import("qwen4/ops.zig");
const attention = @import("qwen4/attention.zig");
const mtp = @import("qwen4/mtp.zig");
const Graph = c.ds4_qwen4_gpu_graph;
const Tensor = c.ds4_gpu_tensor;
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const Weight = c.ds4_tensor;
const Layer = c.ds4_layer_weights;
export fn ds4_test_qwen4_dense_rows(out: ?*Tensor, model: *const Model, weight: *const Weight, x: ?*const Tensor, tokens: u32, rows: u64) bool {
    return ops.denseRows(out, model, weight, x, tokens, rows);
}
export fn ds4_test_qwen4_hc_mix(graph: *Graph, model: *const Model, norm: [*c]const Weight, down: [*c]const Weight, up: [*c]const Weight, injection: [*c]const Weight, tokens: u32) bool {
    return ops.hcMix(graph, model, norm, down, up, injection, tokens);
}
export fn ds4_test_qwen4_attention(graph: *Graph, model: *const Model, layer: *const Layer, index: u32, position: u32, tokens: u32) bool {
    return attention.run(graph, model, layer, index, position, tokens);
}
export fn ds4_test_qwen4_dispatch(graph: *Graph, index: u32, out: ?*Tensor, q: ?*Tensor, gate: ?*Tensor, rows: u32, position: u32, selected: bool, parent_rows: u32) bool {
    return attention.dispatch(graph, index, out, q, gate, rows, position, selected, parent_rows);
}
export fn ds4_test_qwen4_moe(graph: *Graph, model: *const Model, layer: *const Layer, _: u32, tokens: u32) bool {
    return @import("qwen4/moe.zig").run(graph, model, layer, tokens);
}
export fn ds4_test_qwen4_cache_steps(graph: *Graph, model: *const Model, weights: *const Weights, row: u32, tokens: [*]const c_int, count: u32, position: u32) bool {
    return mtp.cacheSteps(graph, model, weights, row, tokens[0..count], position);
}
export fn ds4_test_qwen4_chain(graph: *Graph, model: *const Model, weights: *const Weights, token: c_int, position: u32, draft: *c_int) bool {
    return mtp.chain(graph, model, weights, token, position, draft);
}
export fn ds4_test_qwen4_fused(graph: *const Graph, tokens: u32) bool {
    return ops.fused(graph, tokens);
}
export fn ds4_test_qwen4_draft_rows(graph: *const Graph) u32 {
    return mtp.draftRows(graph);
}
export fn ds4_test_qwen4_draft_load(graph: *Graph, model: *const Model, weight: *const Weight) bool {
    return mtp.loadDraftHead(graph, model, weight);
}
