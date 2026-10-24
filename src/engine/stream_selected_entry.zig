//! C compatibility exports; worker policy and execution live in native code.
const c = @import("stream_selected_c");
const selected = @import("stream_selected.zig");

export fn ds4_native_stream_selected_start(job: ?*c.metal_graph_selected_async_load, tensor: ?*c.ds4_gpu_tensor, model: [*c]const c.ds4_model, layer: [*c]const c.ds4_layer_weights, index: u32, event: u64, gate_bytes: u64, down_bytes: u64, expert_count: u32, selected_count: u32) bool {
    return selected.start(job, tensor, model, layer, index, event, gate_bytes, down_bytes, expert_count, selected_count);
}
export fn ds4_native_stream_selected_finish(job: ?*c.metal_graph_selected_async_load) bool {
    return selected.finish(job);
}
