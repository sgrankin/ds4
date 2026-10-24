//! State that speculative verification may mutate without committing tokens.
//! KDA needs both recurrence buffers; pooled DSA needs its unfinished pool.
//! Published compact rows are append-only and already use absolute positions.
const c = @import("gguf_c");
const std = @import("std");
const Config = @import("config.zig");
const Graph = c.ds4_glm_gpu_graph;
const Tensor = c.ds4_gpu_tensor;

pub fn states(graph: *const Graph, layer: u32) [2]?*Tensor {
    if (Config.isKda(&graph.config, layer))
        return .{ graph.layer_kda_conv_state[layer], graph.layer_kda_recurrent_state[layer] };
    // tail_gate is a view into tail_k's allocation, so copying tail_k saves
    // both tensors once and preserves the original backup layout.
    return .{ graph.layer_indexer_tail_k[layer], null };
}

pub fn bytes(graph: *const Graph) u64 {
    if (!graph.glm53) return 0;
    var total: u64 = 0;
    for (graph.layer_start..graph.layer_end + 1) |layer| {
        for (states(graph, @intCast(layer))) |maybe_tensor| {
            if (maybe_tensor) |tensor|
                total = std.math.add(u64, total, c.ds4_gpu_tensor_bytes(tensor)) catch return 0;
        }
    }
    return total;
}

pub fn copy(graph: *Graph, save: bool) bool {
    const backup = graph.mtp_state_backup orelse return false;
    const expected = bytes(graph);
    if (expected == 0 or c.ds4_gpu_tensor_bytes(backup) < expected) return false;
    var ok = c.ds4_gpu_commands_active() != 0 or c.ds4_gpu_begin_commands() != 0;
    var offset: u64 = 0;
    for (graph.layer_start..graph.layer_end + 1) |layer| {
        for (states(graph, @intCast(layer))) |maybe_tensor| {
            const tensor = maybe_tensor orelse continue;
            if (!ok) break;
            const length = c.ds4_gpu_tensor_bytes(tensor);
            ok = if (save)
                c.ds4_gpu_tensor_copy(backup, offset, tensor, 0, length) != 0
            else
                c.ds4_gpu_tensor_copy(tensor, 0, backup, offset, length) != 0;
            offset += length;
        }
        if (!ok) break;
    }
    if (ok) {
        ok = c.ds4_gpu_end_commands() != 0;
    } else {
        _ = c.ds4_gpu_synchronize();
    }
    return ok and offset == expected;
}

fn free(graph: *Graph) void {
    inline for (.{ "mtp_kv_lora_cache", "mtp_k_rope_cache", "mtp_concat", "mtp_selected", "mtp_state_backup" }) |name| {
        c.ds4_gpu_tensor_free(@field(graph, name));
        @field(graph, name) = null;
    }
    c.free(graph.mtp_logits_host);
    graph.mtp_logits_host = null;
}

/// The draft layer has its own compact cache and language-head output. It must
/// never overwrite trunk recurrence or incomplete pools on a rejected draft.
pub fn ensure(graph: *Graph) bool {
    if (graph.mtp_ready != 0) return true;
    const capacity = if (graph.compact_cache_cap != 0) graph.compact_cache_cap else graph.ctx_size;
    if (capacity == 0) return false;
    const shape = &graph.config.shape;
    graph.mtp_kv_lora_cache = c.ds4_gpu_tensor_alloc(@as(u64, capacity) * shape.n_kv_lora * @sizeOf(u16));
    // GLM53 has a zero-width rotary tail; retain the API's valid companion
    // handle without attempting an invalid zero-byte allocation.
    graph.mtp_k_rope_cache = c.ds4_gpu_tensor_alloc(@max(@as(u64, capacity) * shape.n_rot * @sizeOf(u16), @sizeOf(f32)));
    graph.mtp_concat = c.ds4_gpu_tensor_alloc(2 * @as(u64, shape.n_embd) * @sizeOf(f32));
    graph.mtp_selected = c.ds4_gpu_tensor_alloc(@as(u64, capacity) * @sizeOf(i32));
    const state_bytes = bytes(graph);
    if (state_bytes != 0) graph.mtp_state_backup = c.ds4_gpu_tensor_alloc(state_bytes);
    graph.mtp_logits_host = @ptrCast(@alignCast(c.malloc(@as(usize, shape.n_vocab) * @sizeOf(f32))));
    if (graph.mtp_kv_lora_cache == null or graph.mtp_k_rope_cache == null or graph.mtp_concat == null or
        graph.mtp_selected == null or graph.mtp_state_backup == null or graph.mtp_logits_host == null)
    {
        _ = c.fprintf(c.__stderrp, "ds4: GLM-5.3 speculative state allocation failed\n");
        free(graph);
        return false;
    }
    graph.mtp_ready = 1;
    return true;
}
