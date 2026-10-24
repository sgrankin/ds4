//! Layer-major Metal traversal. A chunk lends its bounded views and page-in
//! slot; numerical execution, layer order and progress boundaries live here.
//! Token/image assembly and final output publication remain outside this loop.
const c = @import("gguf_c");
const std = @import("std");
const Config = @import("config.zig");
const Ops = @import("ops.zig");
const Kda = @import("kda.zig");
const Attention = @import("attention_rows.zig");
const Moe = @import("moe.zig");
const Graph = c.ds4_glm_gpu_graph;
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const Pass = c.ds4_glm_prefill_pass;
const Tensor = c.ds4_gpu_tensor;

pub fn report(pass: *const Pass, total_layers: u32, completed: u32) void {
    const callback = pass.progress orelse return;
    if (pass.work_total == 0) return;
    const chunk_done = if (total_layers == 0 or completed >= total_layers) pass.rows else @as(u64, pass.rows) * completed / total_layers;
    var done = @min(@as(u64, pass.work_done_base) + chunk_done, pass.work_total);
    if (!pass.allow_complete and done >= pass.work_total) done = pass.work_total - 1;
    callback(pass.progress_user, "prefill_display", @intCast(pass.absolute_base + done), @intCast(@as(u64, pass.absolute_base) + pass.work_total));
}

fn layers(graph: *Graph, model: *const Model, weights: *const Weights, pass: *Pass, ablate: u32) bool {
    if (!graph.glm53 or graph.placement != null or graph.tp_world > 1 or graph.imatrix != null or pass.rows == 0) return false;
    const shape = &graph.config.shape;
    const plain_elements = @as(u64, pass.rows) * shape.n_embd;
    const progress_requested = pass.progress != null and pass.work_total > 0;
    for (graph.layer_start..graph.layer_end + 1) |index| {
        const layer: u32 = @intCast(index);
        const completed = layer - graph.layer_start + 1;
        const layer_weights = &weights.layer[layer];
        const slot: *c.metal_graph_stream_prepare_slot = @ptrCast(@alignCast(pass.pagein_slot));
        if (pass.layer_prepare and !c.ds4_native_stream_prepare_join_layer(model, weights, layer, pass.rows, c.DS4_STREAM_PREPARE_PREAD, slot, 1)) return false;
        if (graph.ssd_streaming) {
            if (!c.ds4_native_glm_map_prefill_layer(graph, model, weights, layer, pass.rows, pass.full_layer_prefill)) return false;
            if (pass.layer_prepare and layer < graph.layer_end and !c.ds4_native_stream_prepare_start(model, weights, layer + 1, pass.rows, c.DS4_STREAM_PREPARE_PREAD, slot, 1)) return false;
            if (c.ds4_gpu_begin_commands() == 0) return false;
        }
        if (!Ops.hcPre(true, graph, model, layer_weights.hc_attn_fn, layer_weights.hc_attn_scale, layer_weights.hc_attn_base, layer_weights.attn_norm, pass.hc_current, pass.current, graph.batch_attn_norm, graph.batch_hc_flat, graph.batch_hc_mix, graph.batch_hc_split, pass.rows)) return false;
        const attention_ok = if (Config.isKda(&graph.config, layer))
            Kda.attention(true, graph, model, layer_weights, layer, pass.position, pass.rows, graph.batch_attn_out)
        else
            Attention.encode(graph, model, layer_weights, layer, pass.position, pass.rows, ablate);
        if (!attention_ok) return false;
        c.ds4_native_graph_debug_tensor("attn_out", graph.batch_attn_out, plain_elements, layer, pass.position);
        if (!Ops.hcExpand(graph, pass.hc_after_attention, graph.batch_attn_out, pass.hc_current, graph.batch_hc_split, layer, graph.directional_steering_attn_scale) or
            !@import("../steering_capture.zig").capture(pass.hc_after_attention, layer, pass.rows, shape.n_embd, shape.n_hc, "attn_residual", pass.position) or
            !Ops.hcPre(true, graph, model, layer_weights.hc_ffn_fn, layer_weights.hc_ffn_scale, layer_weights.hc_ffn_base, layer_weights.ffn_norm, pass.hc_after_attention, pass.after_attention, graph.batch_ffn_norm, graph.batch_hc_flat, graph.batch_hc_mix, graph.batch_hc_split, pass.rows)) return false;
        c.ds4_native_graph_debug_tensor("glm_indexed_after_attn", graph.batch_after_attn, plain_elements, layer, pass.position);
        c.ds4_native_graph_debug_tensor("glm_indexed_attn_out", graph.batch_attn_out, plain_elements, layer, pass.position);
        if (!Moe.encodeRows(graph, model, layer_weights, layer, pass.position, pass.rows, pass.next, pass.full_layer_prefill, ablate)) return false;
        c.ds4_native_graph_debug_tensor("ffn_out", pass.next, plain_elements, layer, pass.position);
        if (!Ops.hcExpand(graph, pass.hc_next, pass.next, pass.hc_after_attention, graph.batch_hc_split, layer, graph.directional_steering_ffn_scale) or
            !@import("../steering_capture.zig").capture(pass.hc_next, layer, pass.rows, shape.n_embd, shape.n_hc, "residual", pass.position)) return false;
        std.mem.swap(?*Tensor, &pass.hc_current, &pass.hc_next);

        if (!graph.ssd_streaming and pass.flush_interval != 0 and
            (layer < graph.layer_end or progress_requested) and completed % pass.flush_interval == 0)
        {
            const drain = pass.drain_interval != 0 and layer < graph.layer_end and completed % pass.drain_interval == 0;
            if (drain) {
                if (c.ds4_gpu_end_commands() == 0 or c.ds4_gpu_begin_commands() == 0) return false;
            } else if (c.ds4_gpu_flush_commands() == 0) return false;
            if (pass.drain_interval == 0 or drain) report(pass, graph.layer_count, completed);
        }
        if (graph.ssd_streaming) {
            if (pass.streaming_sync_each_layer and c.ds4_gpu_end_commands() == 0) return false;
            report(pass, graph.layer_count, completed);
        }
    }
    return true;
}

pub fn encode(graph: *Graph, model: *const Model, weights: *const Weights, pass: *Pass, ablate: u32) bool {
    const ok = layers(graph, model, weights, pass, ablate);
    // The chunk owner joins pending page-in work next. Drain a failed encoding
    // first, matching the old Metal loop's per-layer failure boundary.
    if (!ok) _ = c.ds4_gpu_synchronize();
    return ok;
}
