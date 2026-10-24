//! Transitional C ABI only. Model work and resource ownership live in the
//! handwritten modules beside this file, using the one actual graph record.
const c = @import("gguf_c");
const Config = @import("config.zig");
const Resources = @import("resources.zig");
const Ops = @import("ops.zig");
const Kda = @import("kda.zig");
const Layout = @import("layout.zig");
const Allocation = @import("allocation.zig");
const AllocationPlan = @import("allocation_plan.zig");
const Attention = @import("attention.zig");
const AttentionRows = @import("attention_rows.zig");
const Head = @import("head.zig");
const Speculation = @import("speculation.zig");
const Moe = @import("moe.zig");
const Payload = @import("payload.zig");
const Executor = @import("executor.zig");
const Prefill = @import("prefill.zig");
const Mtp = @import("mtp.zig");
const SessionBatch = @import("session_batch.zig");
const Chunk = @import("chunk.zig");
const Graph = c.ds4_glm_gpu_graph;
const Model = c.ds4_model;
const Tensor = c.ds4_gpu_tensor;
const Weight = c.ds4_tensor;

export fn ds4_native_glm_graph_configure(graph: *Graph, shape: *const c.ds4_shape) void {
    Config.capture(graph, shape);
}
export fn ds4_native_glm_graph_free(graph: ?*Graph) bool {
    return Resources.deinit(graph);
}
export fn ds4_native_glm_allocate_tensors(graph: *Graph) bool {
    return Allocation.init(graph);
}
export fn ds4_native_glm_plan_allocation(graph: *Graph, model: *const c.ds4_model, weights: *const c.ds4_weights, first: u32, last: u32, require_embedding: bool, require_output: bool) bool {
    return AllocationPlan.initialize(graph, model, weights, first, last, require_embedding, require_output);
}
export fn ds4_native_glm_reset_kda(graph: ?*Graph) bool {
    return Resources.resetKda(graph);
}
export fn ds4_native_glm_prefill_workspace_free(maybe_graph: ?*Graph) void {
    if (maybe_graph) |graph| Resources.freePrefill(graph);
}
export fn ds4_native_glm_prefill_workspace_ensure(graph: *Graph, rows: u32) bool {
    return Resources.ensurePrefill(graph, rows);
}
export fn ds4_native_glm_session_batch_logits_ensure(graph: *Graph, rows: u32) bool {
    return Resources.ensureBatchLogits(graph, rows);
}
export fn ds4_native_glm_matmul(output: ?*Tensor, model: *const Model, weight: ?*const Weight, inputs: u32, outputs: u32, input: ?*const Tensor, rows: u32) bool {
    return Ops.matmul(output, model, weight, inputs, outputs, input, rows);
}
export fn ds4_native_glm_hc_pre(graph: *Graph, model: *const Model, mixer: ?*const Weight, scale: ?*const Weight, base: ?*const Weight, norm: ?*const Weight, residual: ?*const Tensor, collapsed: ?*Tensor, normalized: ?*Tensor) bool {
    return Ops.hcPre(false, graph, model, mixer, scale, base, norm, residual, collapsed, normalized, graph.hc_flat, graph.hc_mix, graph.hc_split, 1);
}
export fn ds4_native_glm_hc_pre_rows(graph: *Graph, model: *const Model, mixer: ?*const Weight, scale: ?*const Weight, base: ?*const Weight, norm: ?*const Weight, residual: ?*const Tensor, collapsed: ?*Tensor, normalized: ?*Tensor, flat: ?*Tensor, mix: ?*Tensor, split: ?*Tensor, rows: u32) bool {
    return Ops.hcPre(true, graph, model, mixer, scale, base, norm, residual, collapsed, normalized, flat, mix, split, rows);
}
export fn ds4_native_glm_kda(graph: *Graph, model: *const Model, weights: *const c.ds4_layer_weights, layer: u32) bool {
    return Kda.attention(false, graph, model, weights, layer, 0, 1, graph.attn_out);
}
export fn ds4_native_glm_kda_rows(graph: *Graph, model: *const Model, weights: *const c.ds4_layer_weights, layer: u32, position: u32, rows: u32, output: ?*Tensor) bool {
    return Kda.attention(true, graph, model, weights, layer, position, rows, output);
}
export fn ds4_native_glm_attention_decode(graph: *Graph, model: *const Model, weights: *const c.ds4_layer_weights, layer: u32, position: u32, indexed: bool, selection: *c.ds4_glm_decode_selection, ablate_mask: u32) bool {
    return Attention.decode(graph, model, weights, layer, position, indexed, selection, ablate_mask);
}
export fn ds4_native_glm_attention_rows(graph: *Graph, model: *const Model, weights: *const c.ds4_layer_weights, layer: u32, position: u32, rows: u32, ablate_mask: u32) bool {
    return AttentionRows.encode(graph, model, weights, layer, position, rows, ablate_mask);
}
export fn ds4_native_glm_output_head_from(graph: *Graph, model: *const Model, weights: *const c.ds4_weights, hidden: ?*const Tensor) bool {
    return Head.from(graph, model, weights, hidden);
}
export fn ds4_native_glm_output_head(graph: *Graph, model: *const Model, weights: *const c.ds4_weights) bool {
    return Head.encode(graph, model, weights);
}
export fn ds4_native_glm_output_head_forward(graph: *Graph, model: *const Model, weights: *const c.ds4_weights, hidden: ?*const Tensor, logits: [*]f32) bool {
    return Head.forward(graph, model, weights, hidden, logits);
}
export fn ds4_native_glm_spec_state_bytes(graph: *const Graph) u64 {
    return Speculation.bytes(graph);
}
export fn ds4_native_glm_spec_state_copy(graph: *Graph, save: bool) bool {
    return Speculation.copy(graph, save);
}
export fn ds4_native_glm_mtp_ensure(graph: *Graph) bool {
    return Speculation.ensure(graph);
}
export fn ds4_native_glm_mtp_step(graph: *Graph, model: *const Model, weights: *const c.ds4_weights, next_token: c_int, position: u32, min_position: u32, draft: *c_int) bool {
    return Mtp.step(graph, model, weights, next_token, position, min_position, draft);
}
export fn ds4_native_glm_ffn_decode(graph: *Graph, model: *const Model, weights: *const c.ds4_layer_weights, layer: u32, position: u32, ablate_mask: u32) bool {
    return Moe.decode(graph, model, weights, layer, position, ablate_mask);
}
export fn ds4_native_glm_ffn_rows(graph: *Graph, model: *const Model, weights: *const c.ds4_layer_weights, layer: u32, position: u32, rows: u32, output: ?*Tensor, full_layer_prefill: bool, ablate_mask: u32) bool {
    return Moe.encodeRows(graph, model, weights, layer, position, rows, output, full_layer_prefill, ablate_mask);
}
export fn ds4_native_glm_routed_one(graph: *const Graph, model: *const Model, weights: *const c.ds4_layer_weights, layer: u32, output: ?*Tensor, mid: ?*Tensor, selected: ?*const Tensor, routing_weights: ?*const Tensor, input: ?*const Tensor, force_resident: bool) bool {
    return Moe.routedOne(graph, model, weights, layer, output, mid, selected, routing_weights, input, force_resident);
}
export fn ds4_native_glm_routed_rows(graph: *Graph, model: *const Model, weights: *const c.ds4_layer_weights, layer: u32, output: ?*Tensor, mid: ?*Tensor, selected: ?*const Tensor, routing_weights: ?*const Tensor, input: ?*const Tensor, rows: u32, mid_stride: u32, force_resident: bool, direct_scalar_q4: bool) bool {
    return Moe.routedRows(graph, model, weights, layer, output, mid, selected, routing_weights, input, rows, mid_stride, force_resident, direct_scalar_q4);
}
export fn ds4_native_glm_payload_bytes(graph: *const Graph, frontier: *const c.ds4_session_frontier, dense_cache_len: u32) u64 {
    return Payload.bytes(graph, frontier, dense_cache_len);
}
export fn ds4_native_glm_payload_save(graph: *Graph, frontier: *const c.ds4_session_frontier, dense_cache_len: u32, stream: *c.FILE, error_text: [*c]u8, error_size: usize) c_int {
    Payload.save(graph, frontier, dense_cache_len, stream, error_text[0..error_size]) catch return 1;
    return 0;
}
export fn ds4_native_glm_payload_load(graph: *Graph, frontier: *c.ds4_session_frontier, dense_cache_len: *u32, stream: *c.FILE, header: *const Payload.Header, remaining: u64, error_text: [*c]u8, error_size: usize) c_int {
    Payload.load(graph, frontier, dense_cache_len, stream, header, remaining, error_text[0..error_size]) catch return 1;
    return 0;
}
export fn ds4_native_glm_forward_token(graph: *Graph, model: *const Model, weights: *const c.ds4_weights, token: c_int, input_hc: ?[*]const f32, position: u32, output_hc: ?[*]f32, logits: ?[*]f32, defer_completion: bool, ablate_mask: u32) bool {
    return Executor.forward(graph, model, weights, token, input_hc, position, output_hc, logits, defer_completion, ablate_mask);
}
export fn ds4_native_glm_prefill_layers(graph: *Graph, model: *const Model, weights: *const c.ds4_weights, pass: *c.ds4_glm_prefill_pass, ablate_mask: u32) bool {
    return Prefill.encode(graph, model, weights, pass, ablate_mask);
}
export fn ds4_native_glm_chunk_forward(graph: *Graph, model: *const Model, weights: *const c.ds4_weights, tokens: [*]const c_int, input_hc: ?[*]const f32, images: [*c]const c.ds4_vision_span, image_count: usize, position: u32, rows: u32, output_hc: ?[*]f32, logits: ?[*]f32, progress: c.ds4_session_progress_fn, progress_user: ?*anyopaque, absolute_base: u32, done_base: u32, work_total: u32, ablate_mask: u32) bool {
    return Chunk.forward(graph, model, weights, tokens[0..rows], input_hc, images[0..image_count], position, output_hc, logits, .{ .callback = progress, .user = progress_user, .absolute_base = absolute_base, .done_base = done_base, .total = work_total }, ablate_mask);
}
export fn ds4_native_glm_session_batch(model: *const Model, weights: *const c.ds4_weights, rows: [*]const c.ds4_glm_session_row, count: u32, ablate_mask: u32) bool {
    return SessionBatch.encode(model, weights, rows[0..count], ablate_mask);
}
export fn ds4_native_glm_payload_live_bytes(graph: *const Graph, full_live: u32, compact_live: u32) u64 {
    return Layout.payloadLiveBytes(graph, full_live, compact_live);
}
