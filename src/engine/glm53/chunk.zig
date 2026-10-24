//! Complete Flash prefill chunk: bounded views, token/image input, page-in
//! lifecycle, numerical layers, expert-cache seeding and output publication.
//! The storage leaves own only their actual jobs and model-range mappings.
const c = @import("gguf_c");
const std = @import("std");
const Planning = @import("planning.zig");
const Resources = @import("resources.zig");
const Overlay = @import("overlay.zig");
const Seed = @import("cache_seed.zig");
const Prefill = @import("prefill.zig");
const Head = @import("head.zig");
const Graph = c.ds4_glm_gpu_graph;
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const Span = c.ds4_vision_span;
const Tensor = c.ds4_gpu_tensor;

pub const Progress = struct {
    callback: c.ds4_session_progress_fn = null,
    user: ?*anyopaque = null,
    absolute_base: u32 = 0,
    done_base: u32 = 0,
    total: u32 = 0,
};

fn input(graph: *Graph, model: *const Model, weights: *const Weights, tokens: []const c_int, input_hc: ?[*]const f32, overlay: *const c.glm_vision_overlay, pass: *const c.ds4_glm_prefill_pass) bool {
    const shape = &graph.config.shape;
    if (input_hc) |hidden| {
        if (c.ds4_gpu_tensor_write(pass.hc_current, 0, hidden, @as(u64, pass.rows) * shape.n_embd * shape.n_hc * @sizeOf(f32)) == 0) return false;
    } else if (!c.ds4_native_glm_map_token(graph, model, weights)) return false;
    if (c.ds4_gpu_begin_commands() == 0) return false;
    if (input_hc != null) return true;
    const embedding = weights.token_embd;
    const embedded = if (embedding.*.type == c.DS4_TENSOR_BF16)
        c.ds4_gpu_glm53_embedding_bf16(pass.current, model.map, model.size, embedding.*.abs_offset, graph.prefill_tokens, @intCast(tokens.len), shape.n_embd, shape.n_vocab) != 0
    else
        c.ds4_gpu_embed_tokens_quant_tensor(pass.current, graph.prefill_tokens, model.map, model.size, embedding.*.abs_offset, embedding.*.type, shape.n_vocab, @intCast(tokens.len), shape.n_embd) != 0;
    if (!embedded or c.ds4_gpu_repeat_hc_rows_tensor(pass.hc_current, pass.current, pass.rows, shape.n_embd, shape.n_hc) == 0) return false;
    for (overlay.segments[0..overlay.count]) |segment| {
        if (c.ds4_gpu_glm53_scatter_image_hc(pass.hc_current, overlay.tensor, segment.dst_row, segment.src_row, segment.rows, pass.rows, shape.n_embd, shape.n_hc) == 0) return false;
    }
    return true;
}

pub fn forward(graph: *Graph, model: *const Model, weights: *const Weights, tokens: []const c_int, input_hc: ?[*]const f32, images: []const Span, position: u32, output_hc: ?[*]f32, logits: ?[*]f32, progress: Progress, ablate: u32) bool {
    const rows: u32 = @intCast(tokens.len);
    const shape = &graph.config.shape;
    if (graph.glm53 and graph.indexed_prefill_cap == 0 and !graph.full_kv_cache and
        Planning.fitsContext(graph, position, rows) and Planning.fitsFull(graph, position, rows))
    {
        // DISABLE_INDEXED_PREFILL has always selected the full-cache path,
        // while production Flash allocates only absorbed KV. Preserve that
        // explicit rejection instead of silently inventing another algorithm.
        _ = c.fputs("ds4: GLM full-attention prefill was requested without an expanded KV cache\n", c.__stderrp);
        return false;
    }
    if (!graph.glm53 or graph.placement != null or graph.tp_world > 1 or graph.imatrix != null or
        !Planning.indexedAvailable(graph) or graph.layer_count == 0 or rows > graph.indexed_prefill_cap or
        !Planning.fitsContext(graph, position, rows) or position + rows > graph.compact_cache_cap) return false;
    if (!Resources.ensurePrefill(graph, rows)) return false;
    for (tokens, 0..) |token, index| {
        if (token < 0 or (token >= shape.n_vocab and !Overlay.syntheticCovered(token, index, images, position, shape.n_vocab))) return false;
    }
    if ((input_hc == null and !graph.has_token_embd) or (input_hc != null and images.len != 0) or (logits != null and !graph.has_output_head)) return false;
    Seed.reset(graph);
    var overlay = std.mem.zeroes(c.glm_vision_overlay);
    if (!Overlay.prepare(&overlay, images, position, rows)) return false;
    defer Overlay.deinit(&overlay);
    if (c.ds4_gpu_tensor_write(graph.prefill_tokens, 0, tokens.ptr, tokens.len * @sizeOf(c_int)) == 0) return false;

    const plain_bytes = @as(u64, rows) * shape.n_embd * @sizeOf(f32);
    const hc_bytes = plain_bytes * shape.n_hc;
    const work = .{
        .current = c.ds4_gpu_tensor_view(graph.batch_cur, 0, plain_bytes),
        .next = c.ds4_gpu_tensor_view(graph.batch_next, 0, plain_bytes),
        .after = c.ds4_gpu_tensor_view(graph.batch_after_attn, 0, plain_bytes),
        .hc_current = c.ds4_gpu_tensor_view(graph.batch_hc_cur, 0, hc_bytes),
        .hc_next = c.ds4_gpu_tensor_view(graph.batch_hc_next, 0, hc_bytes),
        .hc_after = c.ds4_gpu_tensor_view(graph.batch_hc_after_attn, 0, hc_bytes),
    };
    defer inline for (@typeInfo(@TypeOf(work)).@"struct".field_names) |name| c.ds4_gpu_tensor_free(@field(work, name));
    inline for (@typeInfo(@TypeOf(work)).@"struct".field_names) |name| if (@field(work, name) == null) return false;
    var slot = std.mem.zeroes(c.metal_graph_stream_prepare_slot);
    const full_layer = Planning.fullLayerPrefill(graph, position, rows);
    const flush = Planning.flushInterval(rows, rows, logits != null);
    var pass = c.ds4_glm_prefill_pass{
        .position = position,
        .rows = rows,
        .full_layer_prefill = full_layer,
        .layer_prepare = graph.ssd_streaming and full_layer and c.getenv("DS4_METAL_DISABLE_GLM_STREAMING_PREFILL_FULL_LAYER_PREPARE") == null,
        .streaming_sync_each_layer = true,
        .allow_complete = logits == null and output_hc == null,
        .flush_interval = flush,
        .drain_interval = if (flush != 0) 16 else 0,
        .progress = progress.callback,
        .progress_user = progress.user,
        .absolute_base = progress.absolute_base,
        .work_done_base = progress.done_base,
        .work_total = progress.total,
        .current = work.current,
        .next = work.next,
        .after_attention = work.after,
        .hc_after_attention = work.hc_after,
        .hc_current = work.hc_current,
        .hc_next = work.hc_next,
        .pagein_slot = &slot,
    };
    Prefill.report(&pass, graph.layer_count, 0);
    c.ds4_gpu_set_glm_streaming_prefill_full_layer(full_layer);
    defer c.ds4_gpu_set_glm_streaming_prefill_full_layer(false);
    var ok = input(graph, model, weights, tokens, input_hc, &overlay, &pass);
    if (ok and graph.ssd_streaming) ok = c.ds4_gpu_end_commands() != 0;
    c.ds4_gpu_tp_set_attn_head_split(0);
    if (ok) ok = Prefill.encode(graph, model, weights, &pass, ablate);
    c.ds4_gpu_tp_set_attn_head_split(0);
    const joined = c.ds4_native_stream_prepare_join_all(&slot, 1);
    ok = ok and joined;
    if (ok and !graph.ssd_streaming) ok = c.ds4_gpu_end_commands() != 0;
    if (!ok) {
        _ = c.ds4_gpu_synchronize();
        return false;
    }
    Prefill.report(&pass, graph.layer_count, graph.layer_count);
    if (output_hc) |hidden| {
        if (c.ds4_gpu_tensor_read(pass.hc_current, 0, hidden, hc_bytes) == 0) return false;
    }
    if (logits) |values| {
        if (!Seed.finish(graph, model, weights)) return false;
        const last = c.ds4_gpu_tensor_view(pass.hc_current, @as(u64, rows - 1) * shape.n_hc * shape.n_embd * @sizeOf(f32), @as(u64, shape.n_hc) * shape.n_embd * @sizeOf(f32));
        defer c.ds4_gpu_tensor_free(last);
        if (last == null or c.ds4_gpu_hc_weighted_sum_tensor(graph.hc_output, last, graph.hc_mean_weights, shape.n_embd, shape.n_hc) == 0 or
            (graph.ssd_streaming and !c.ds4_native_glm_map_output(graph, model, weights)) or !Head.forward(graph, model, weights, graph.hc_output, values)) return false;
        pass.allow_complete = true;
        Prefill.report(&pass, graph.layer_count, graph.layer_count);
    }
    return true;
}
