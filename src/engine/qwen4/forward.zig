//! One causal Qwen forward over1..capacity rows. Preserve the early Metal
//! submission, existing fusions and exact verify partitions; position is
//! committed only after commands finish and requested logits reach the host.
const std = @import("std");
const c = @import("gguf_c");
const dimensions = @import("config.zig");
const resources = @import("resources.zig");
const ops = @import("ops.zig");
const input = @import("input.zig");
const attention = @import("attention.zig");
const moe = @import("moe.zig");
const dumps = @import("diagnostics.zig");
const Graph = c.ds4_qwen4_gpu_graph;
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const Tensor = c.ds4_gpu_tensor;

const Profile = struct {
    enabled: bool,
    last: f64,
    elapsed: [6]f64 = @splat(0),

    fn mark(self: *Profile, stage: usize) void {
        if (!self.enabled) return;
        _ = c.ds4_gpu_end_commands();
        const time = ops.now();
        self.elapsed[stage] += time - self.last;
        self.last = time;
        _ = ops.beginCommands();
    }
};

var timing_mode: ?u32 = null;
var timing_accumulated: [4]f64 = @splat(0);
var timing_calls: u32 = 0;

pub fn run(graph: *Graph, model: *const Model, weights: *const Weights, tokens: []const c_int, output: [*c]f32, all_rows: bool) bool {
    const count: u32 = @intCast(tokens.len);
    if (count == 0 or count > graph.cap_tokens or graph.pos + count > graph.ctx_cap or
        (all_rows and count > graph.n_logit_rows)) return false;
    const shape = &graph.config.shape;
    for (tokens) |token| {
        if (token < 0 or token >= shape.n_vocab) {
            _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 token id %d is outside the vocabulary\n", token);
            return false;
        }
    }
    const position = graph.pos;
    const trunk = dimensions.trunkLayers(&graph.config);
    const hc_width = @as(u64, shape.n_embd) * shape.n_hc;
    if (timing_mode == null) {
        const raw = c.getenv("DS4_QWEN4_TIMING");
        timing_mode = if (raw == null) 0 else if (c.atoi(raw) >= 2) 2 else 1;
    }
    const timing = timing_mode.?;
    // Starting the GPU after layer2 overlaps host encoding of the remaining
    // trunk. Queue order is retained; timing2 deliberately synchronizes stages.
    var flush_layer: u32 = if (timing != 2 and trunk > 2) 2 else 0;
    const raw_flush = c.getenv("DS4_QWEN4_FLUSH_LAYER");
    if (timing != 2 and raw_flush != null and raw_flush[0] != 0) {
        var end: [*c]u8 = null;
        const layer = c.strtoul(raw_flush, &end, 10);
        if (end != raw_flush and end[0] == 0 and layer < trunk) flush_layer = @intCast(layer);
    }
    const started = if (timing != 0) ops.now() else 0;
    if (!input.stage(graph, model, weights, tokens)) return false;
    const staged = if (timing != 0) ops.now() else 0;
    if (!ops.beginCommands()) return false;
    var profile = Profile{ .enabled = timing == 2 and count > 1, .last = if (timing == 2 and count > 1) ops.now() else 0 };
    var ok = true;
    for (0..trunk) |offset| {
        if (!ok) break;
        const index: u32 = @intCast(offset);
        const layer = &weights.layer[index];
        if (dimensions.isPle(&graph.config, index)) {
            ok = ops.dense(graph.ple_key, model, layer.ple_key, graph.ple_emb, count) and
                ops.dense(graph.ple_val, model, layer.ple_value, graph.ple_emb, count) and
                c.ds4_gpu_qwen4_ple_gate_tensor(graph.ple_gated, graph.ple_normed, graph.R, graph.ple_key, graph.ple_val, model.map, model.size, layer.ple_norm_key.*.abs_offset, layer.ple_norm_query.*.abs_offset, layer.ple_norm_conv.*.abs_offset, count, shape.n_embd, shape.n_hc, shape.rms_eps) != 0 and
                c.ds4_gpu_qwen4_ple_conv_tensor(graph.R, graph.ple_gated, graph.ple_normed, graph.ple_hist, model.map, model.size, layer.ple_conv.*.abs_offset, layer.ple_conv.*.type, count, @intCast(hc_width), shape.n_ple_conv, shape.n_ple_ngram, if (graph.snap_after_first) graph.snap_ple_hist else null, 0, if (graph.snap_after_second) graph.snap2_ple_hist else null, 1) != 0;
        }
        profile.mark(0);
        ok = ok and ops.hcMix(graph, model, layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject, count);
        profile.mark(1);
        const linear = dimensions.isLinear(&graph.config, index);
        if (ok) ok = if (linear) attention.linear(graph, model, layer, index, count) else attention.run(graph, model, layer, index, position, count);
        profile.mark(if (linear) 2 else 3);
        if (graph.steer_attn_scale == 0 and count == 1 and graph.mtp_R == null and shape.n_hc == 4 and c.ds4_gpu_qwen4_decode_fusions_enabled() != 0 and layer.hc_ffn_inject.*.type == c.DS4_TENSOR_F16) {
            ok = ok and c.ds4_gpu_qwen4_hc_combine_norm_tensor(graph.hc_u, graph.blk, graph.inj, graph.xn, graph.inj_alt, graph.R, model.map, model.size, layer.hc_ffn_norm.*.abs_offset, layer.hc_ffn_inject.*.abs_offset, layer.hc_ffn_inject.*.type, count, shape.n_embd, shape.n_hc, shape.n_hc, shape.rms_eps) != 0;
            if (ok) {
                // Both residual buffers retain full prefill capacity. hc_u is
                // spare in this one-token fused mixer, so exchange ownership.
                std.mem.swap(?*Tensor, &graph.R, &graph.hc_u);
                std.mem.swap(?*Tensor, &graph.inj, &graph.inj_alt);
                ok = ops.dense(graph.lo, model, layer.hc_ffn_down, graph.xn, count) and
                    c.ds4_gpu_qwen4_hc_gate_mix_tensor(graph.mixed, graph.xn, graph.lo, model.map, model.size, layer.hc_ffn_up.*.abs_offset, layer.hc_ffn_up.*.type, count, shape.n_embd, shape.n_hc, shape.n_hc_lowrank) != 0;
            }
        } else {
            ok = ok and resources.combineAttention(graph, index, count) and
                ops.hcMix(graph, model, layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject, count);
        }
        if (ok and graph.dump_prompt_rows) ok = dumps.lastAttention(graph, index, count);
        profile.mark(4);
        ok = ok and moe.runSteered(graph, model, layer, index, count);
        if (ok and graph.dump_prompt_rows) {
            c.ds4_native_graph_debug_tensor("qwen_router", graph.router, @as(u64, count) * shape.n_expert, index, position);
            ok = dumps.lastFfn(graph, index, count);
        }
        profile.mark(5);
        if (ok and index + 1 == flush_layer) ok = c.ds4_gpu_flush_commands() != 0;
    }
    if (profile.enabled) _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 prefill stage ms/chunk (pos=%u T=%u ok=%d): ple %.1f hc_attn %.1f gdn %.1f attn %.1f hc_ffn %.1f moe %.1f\n", position, count, @as(c_int, @intFromBool(ok)), profile.elapsed[0] * 1e3, profile.elapsed[1] * 1e3, profile.elapsed[2] * 1e3, profile.elapsed[3] * 1e3, profile.elapsed[4] * 1e3, profile.elapsed[5] * 1e3);
    if (ok and output != null) ok = projectLogits(graph, model, weights, count, all_rows);
    const encoded = if (timing != 0) ops.now() else 0;
    if (c.ds4_gpu_end_commands() == 0) ok = false;
    const finished = if (timing != 0) ops.now() else 0;
    if (ok and output != null) {
        const rows: u64 = if (all_rows) count else 1;
        ok = c.ds4_gpu_tensor_read(graph.logits, 0, output, rows * shape.n_vocab * @sizeOf(f32)) != 0;
    }
    if (timing != 0) {
        timing_accumulated[0] += staged - started;
        timing_accumulated[1] += encoded - staged;
        timing_accumulated[2] += finished - encoded;
        timing_accumulated[3] += ops.now() - finished;
        timing_calls += 1;
        if (timing_calls % 50 == 0) {
            _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 forward(T=%u) avg ms: stage %.3f encode %.3f gpu %.3f read %.3f\n", count, 1e3 * timing_accumulated[0] / 50, 1e3 * timing_accumulated[1] / 50, 1e3 * timing_accumulated[2] / 50, 1e3 * timing_accumulated[3] / 50);
            timing_accumulated = @splat(0);
        }
    }
    if (ok) graph.pos += count;
    if (graph.snap_after_first) {
        graph.snap_valid = ok;
        graph.snap_after_first = false;
    }
    if (graph.snap_after_second) {
        graph.snap2_valid = ok;
        graph.snap_after_second = false;
    }
    return ok;
}

fn projectLogits(graph: *Graph, model: *const Model, weights: *const Weights, rows: u32, all_rows: bool) bool {
    if (all_rows) return ops.hcMix(graph, model, weights.output_hc_norm, weights.output_hc_down, weights.output_hc_up, null, rows) and
        ops.dense(graph.logits, model, weights.output, graph.mixed, rows);
    const width = @as(u64, graph.config.shape.n_embd) * graph.config.shape.n_hc;
    const last = ops.rowsView(graph.R, rows - 1, 1, width) orelse return false;
    defer c.ds4_gpu_tensor_free(last);
    const residual = graph.R;
    graph.R = last;
    const mixed = ops.hcMix(graph, model, weights.output_hc_norm, weights.output_hc_down, weights.output_hc_up, null, 1);
    graph.R = residual;
    return mixed and ops.dense(graph.logits, model, weights.output, graph.mixed, 1);
}
