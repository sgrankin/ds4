//! Embedded next-token predictor. Draft heads may use a vocabulary subset;
//! target verification and host-logit consumers always retain the full head.
//! Predictor recurrence is proposal-only and intentionally receives no steering.
const std = @import("std");
const c = @import("gguf_c");
const ops = @import("ops.zig");
const attention = @import("attention.zig");
const moe = @import("moe.zig");
const embedding = @import("embedding.zig");
const Graph = c.ds4_qwen4_gpu_graph;
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const Layer = c.ds4_layer_weights;
const Weight = c.ds4_tensor;
const Tensor = c.ds4_gpu_tensor;

/// Token IDs follow merge order, so a leading vocabulary prefix favors common
/// tokens. This knob changes proposal coverage; target verification still
/// scores the full vocabulary and decides every committed token.
pub fn draftRows(graph: *const Graph) u32 {
    const raw = c.getenv("DS4_QWEN4_MTP_DRAFT_ROWS");
    const requested = if (raw != null and raw[0] != 0) c.strtoll(raw, null, 10) else 0;
    return if (requested > 0 and requested < graph.config.shape.n_vocab) @intCast(requested) else graph.config.shape.n_vocab;
}

/// Optional gathered vocabulary (the FR-Spec idea used by MTPLX): token IDs
/// from a file become contiguous head rows, then argmax maps back to the IDs.
/// A missing/invalid file leaves the full head; the gathered copy is graph-owned.
pub fn loadDraftHead(graph: *Graph, model: *const Model, weight: *const Weight) bool {
    if (graph.draft_head_tried) return graph.draft_head != null;
    graph.draft_head_tried = true;
    const path = c.getenv("DS4_QWEN4_MTP_DRAFT_VOCAB");
    if (path == null or path[0] == 0 or weight.type != c.DS4_TENSOR_Q8_0 or weight.ndim < 2 or weight.dim[0] % 32 != 0) return false;
    const stream = c.fopen(path, "r") orelse {
        _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 MTP draft vocabulary %s: %s\n", path, c.strerror(c.__error().*));
        return false;
    };
    const vocab: usize = @intCast(weight.dim[1]);
    const allocation = c.malloc(vocab * @sizeOf(i32));
    const ids: [*c]i32 = @ptrCast(@alignCast(allocation));
    const seen_allocation = c.calloc(vocab, 1);
    const seen: [*c]u8 = @ptrCast(seen_allocation);
    var transferred = false;
    defer if (!transferred) c.free(allocation);
    var count: u32 = 0;
    var token: c_longlong = undefined;
    while (ids != null and seen != null and count < vocab and c.fscanf(stream, "%lld", &token) == 1) {
        if (token < 0 or token >= vocab or seen[@intCast(token)] != 0) continue;
        seen[@intCast(token)] = 1;
        ids[count] = @intCast(token);
        count += 1;
    }
    _ = c.fclose(stream);
    c.free(seen_allocation);
    if (ids == null or count == 0 or count >= vocab) {
        _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 MTP draft vocabulary %s: need1..%llu distinct ids (got %u)\n", path, @as(c_ulonglong, vocab - 1), count);
        return false;
    }
    const row_bytes: usize = @intCast(weight.dim[0] / 32 * 34);
    const bytes = @as(usize, count) * row_bytes;
    const staging = c.malloc(bytes);
    defer c.free(staging);
    const head = if (staging != null) c.ds4_gpu_tensor_alloc(bytes) else null;
    if (staging != null and head != null) {
        const source: [*]const u8 = @ptrCast(model.map + weight.abs_offset);
        const destination: [*]u8 = @ptrCast(staging.?);
        for (ids[0..count], 0..) |id, row| @memcpy(destination[row * row_bytes ..][0..row_bytes], source[@as(usize, @intCast(id)) * row_bytes ..][0..row_bytes]);
    }
    if (staging == null or head == null or c.ds4_gpu_tensor_write(head, 0, staging, bytes) == 0) {
        _ = c.fputs("ds4: Qwen3.8 MTP draft head upload failed\n", c.__stderrp);
        c.ds4_gpu_tensor_free(head);
        return false;
    }
    graph.draft_head = head;
    graph.draft_ids = ids;
    graph.draft_rows = count;
    transferred = true;
    _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 MTP draft head: %u of %llu vocabulary rows from %s (%.0f MiB)\n", count, @as(c_ulonglong, vocab), path, @as(f64, @floatFromInt(bytes)) / (1024 * 1024));
    return true;
}

fn stageEmbedding(graph: *Graph, model: *const Model, weights: *const Weights, tokens: []const c_int) bool {
    const width: usize = graph.config.shape.n_embd;
    for (tokens, 0..) |token, index| {
        if (token < 0 or token >= graph.config.shape.n_vocab or
            !embedding.row(model, weights.token_embd, @intCast(token), graph.host_row[index * width ..][0..width])) return false;
    }
    return c.ds4_gpu_tensor_write(graph.mtp_e, 0, graph.host_row, tokens.len * width * @sizeOf(f32)) != 0 and ops.beginCommands();
}

fn stageResidual(graph: *Graph, model: *const Model, layer: *const Layer, source: ?*Tensor, first: u32, count: u32) bool {
    const shape = &graph.config.shape;
    const embd: u64 = shape.n_embd;
    const streams: u64 = shape.n_hc;
    for (0..count) |row| {
        const e = ops.rowsView(graph.mtp_e, @intCast(row), 1, embd) orelse return false;
        defer c.ds4_gpu_tensor_free(e);
        const residual = ops.rowsView(source, first + @as(u32, @intCast(row)), 1, streams * embd) orelse return false;
        defer c.ds4_gpu_tensor_free(residual);
        const cat = ops.rowsView(graph.mtp_cat, @intCast(row), 1, (streams + 1) * 2 * embd) orelse return false;
        defer c.ds4_gpu_tensor_free(cat);
        if (c.ds4_gpu_qwen4_mtp_stage_tensor(cat, e, residual, model.map, model.size, layer.nextn_enorm.*.abs_offset, layer.nextn_hnorm.*.abs_offset, shape.n_embd, shape.n_hc, shape.rms_eps) == 0) return false;
    }
    if (!ops.dense(graph.mtp_proj, model, layer.nextn_eh_proj, graph.mtp_cat, count * (shape.n_hc + 1))) return false;
    for (0..count) |row| {
        const projection = ops.rowsView(graph.mtp_proj, @intCast(row), 1, (streams + 1) * embd) orelse return false;
        defer c.ds4_gpu_tensor_free(projection);
        const residual = ops.rowsView(graph.mtp_R, @intCast(row), 1, streams * embd) orelse return false;
        defer c.ds4_gpu_tensor_free(residual);
        if (c.ds4_gpu_qwen4_mtp_combine_tensor(residual, projection, shape.n_embd, shape.n_hc) == 0) return false;
    }
    return true;
}

fn predictor(graph: *Graph, model: *const Model, layer: *const Layer, position: u32, count: u32) bool {
    const shape = &graph.config.shape;
    return ops.hcMix(graph, model, layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject, count) and
        attention.run(graph, model, layer, shape.n_layer - 1, position, count) and
        c.ds4_gpu_qwen4_hc_combine_tensor(graph.R, graph.blk, graph.inj, count, shape.n_embd, shape.n_hc) != 0 and
        ops.hcMix(graph, model, layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject, count) and
        moe.run(graph, model, layer, count);
}

pub fn steps(graph: *Graph, model: *const Model, weights: *const Weights, row: u32, tokens: []const c_int, position: u32, want_logits: bool, output: [*c]f32, draft: [*c]c_int) bool {
    const count: u32 = @intCast(tokens.len);
    if (graph.mtp_R == null or count == 0 or count > 3 or position > graph.ctx_cap or count > graph.ctx_cap - position or
        row > graph.cap_tokens or count > graph.cap_tokens - row) return false;
    if (!stageEmbedding(graph, model, weights, tokens)) return false;
    const layer = &weights.layer[graph.config.shape.n_layer - 1];
    const residual = graph.R;
    var ok = stageResidual(graph, model, layer, residual, row, count);
    graph.R = graph.mtp_R;
    ok = ok and predictor(graph, model, layer, position, count);
    const raw_argmax = c.getenv("DS4_QWEN4_MTP_GPU_ARGMAX");
    const gpu_argmax = want_logits and draft != null and output == null and (raw_argmax == null or c.strcmp(raw_argmax, "0") != 0);
    const gathered = gpu_argmax and loadDraftHead(graph, model, weights.output);
    const head_rows = if (gathered) graph.draft_rows else if (gpu_argmax) draftRows(graph) else graph.config.shape.n_vocab;
    var last: ?*Tensor = null;
    if (ok and want_logits) {
        last = ops.rowsView(graph.mtp_R, count - 1, 1, @as(u64, graph.config.shape.n_embd) * graph.config.shape.n_hc);
        graph.R = last;
        ok = last != null and ops.hcMix(graph, model, layer.nextn_hc_head_norm, layer.nextn_hc_head_down, layer.nextn_hc_head_up, null, 1);
        if (ok) ok = if (gathered)
            c.ds4_gpu_qwen4_matmul_q8_0_weights_tensor(graph.logits, graph.draft_head, @intCast(weights.output.*.dim[0]), head_rows, graph.mixed) != 0
        else
            ops.denseRows(graph.logits, model, weights.output, graph.mixed, 1, head_rows);
    }
    graph.R = residual;
    if (ok and gpu_argmax) ok = c.ds4_gpu_qwen4_argmax_tensor(graph.mtp_argmax, graph.mtp_argmax_tmp, graph.logits, head_rows) != 0;
    // Batched predictor callers keep the command batch open and finish every
    // session's head together; only an explicit logits request waits/readbacks.
    if (want_logits and c.ds4_gpu_end_commands() == 0) ok = false;
    c.ds4_gpu_tensor_free(last);
    if (ok and gpu_argmax) {
        var token: i32 = -1;
        ok = c.ds4_gpu_tensor_read(graph.mtp_argmax, 0, &token, @sizeOf(i32)) != 0;
        if (ok and gathered) {
            if (token < 0 or token >= graph.draft_rows) ok = false else token = graph.draft_ids[@intCast(token)];
        }
        if (ok and (token < 0 or token >= graph.config.shape.n_vocab)) ok = false;
        if (ok) draft[0] = token;
    } else if (ok and want_logits) {
        const destination = if (output != null) output else graph.host_logits;
        ok = c.ds4_gpu_tensor_read(graph.logits, 0, destination, @as(u64, graph.config.shape.n_vocab) * @sizeOf(f32)) != 0;
        if (ok and draft != null) draft[0] = c.ds4_sampling_argmax(destination, graph.config.shape.n_vocab);
    }
    if (ok) {
        graph.mtp_pos = position + count;
        graph.mtp_last_rows = count;
    }
    return ok;
}

/// Recursive draft: use the predictor's own last stream before overwriting it
/// with this proposal. Target verification remains the only commit authority.
pub fn chain(graph: *Graph, model: *const Model, weights: *const Weights, token: c_int, position: u32, draft: *c_int) bool {
    if (graph.mtp_R == null or graph.mtp_last_rows == 0 or graph.mtp_last_rows > 3 or position >= graph.ctx_cap or
        token < 0 or token >= graph.config.shape.n_vocab) return false;
    if (!stageEmbedding(graph, model, weights, &.{token})) return false;
    const layer = &weights.layer[graph.config.shape.n_layer - 1];
    const residual = graph.R;
    var ok = stageResidual(graph, model, layer, graph.mtp_R, graph.mtp_last_rows - 1, 1);
    graph.R = graph.mtp_R;
    ok = ok and predictor(graph, model, layer, position, 1);
    const head_rows = draftRows(graph);
    var last: ?*Tensor = null;
    if (ok) {
        last = ops.rowsView(graph.mtp_R, 0, 1, @as(u64, graph.config.shape.n_embd) * graph.config.shape.n_hc);
        graph.R = last;
        ok = last != null and ops.hcMix(graph, model, layer.nextn_hc_head_norm, layer.nextn_hc_head_down, layer.nextn_hc_head_up, null, 1) and
            ops.denseRows(graph.logits, model, weights.output, graph.mixed, 1, head_rows);
    }
    graph.R = residual;
    ok = ok and c.ds4_gpu_qwen4_argmax_tensor(graph.mtp_argmax, graph.mtp_argmax_tmp, graph.logits, head_rows) != 0;
    if (c.ds4_gpu_end_commands() == 0) ok = false;
    c.ds4_gpu_tensor_free(last);
    if (ok) {
        var selected: i32 = -1;
        ok = c.ds4_gpu_tensor_read(graph.mtp_argmax, 0, &selected, @sizeOf(i32)) != 0;
        if (ok and (selected < 0 or selected >= head_rows)) ok = false;
        if (ok) draft.* = selected;
    }
    if (ok) {
        graph.mtp_pos = position + 1;
        graph.mtp_last_rows = 1;
    }
    return ok;
}
