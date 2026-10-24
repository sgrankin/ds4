//! Prompt-only residual capture; repeated chunks replace the last-row dump.
const c = @import("gguf_c");
const capture_owner = @import("../steering_capture.zig");

pub fn enabled() bool {
    const prefix = c.getenv("DS4_METAL_GRAPH_DUMP_PREFIX");
    return prefix != null and prefix[0] != 0;
}

pub fn capture(graph: *const c.ds41_gpu_graph, residual: ?*const c.ds4_gpu_tensor, layer: u32, count: u32, width: u32, streams: u32, ffn: bool) bool {
    if (!graph.steering_capture_enabled or !graph.steering_capture_prompt) return true;
    return capture_owner.capture(residual, layer, count, width, streams, if (ffn) "residual" else "attn_residual", 0);
}
