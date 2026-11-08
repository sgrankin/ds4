//! Host-only checks of the production policy and accounting owners.
const std = @import("std");
const c = @import("gguf_c");
const memory = @import("qwen4/memory.zig");
const admission = @import("qwen4/admission.zig");
const attention = @import("qwen4/attention.zig");
const mtp = @import("qwen4/mtp.zig");
const payload = @import("qwen4/payload.zig");
const metadata = @import("qwen4/metadata.zig");
const lifecycle = @import("engine_lifecycle.zig");
const expect = std.testing.expect;
const equal = std.testing.expectEqual;

test "Qwen SSD startup maps trunk and predictor statics but no routed or ngram payload" {
    var shape = metadata.profile(false);
    shape.n_layer = 2;
    var weights = std.mem.zeroes(c.ds4_weights);
    var tensors = std.mem.zeroes([6]c.ds4_tensor);
    for (&tensors, 0..) |*tensor, index| {
        tensor.type = c.DS4_TENSOR_F32;
        tensor.abs_offset = 100 + @as(u64, @intCast(index)) * 100;
        tensor.bytes = 16;
    }
    weights.token_embd = &tensors[0];
    weights.layer[0].hc_attn_norm = &tensors[1];
    weights.layer[1].nextn_eh_proj = &tensors[2];
    weights.output = &tensors[3];
    weights.layer[0].ffn_gate_exps = &tensors[4];
    weights.layer[1].ffn_down_exps = &tensors[4];
    weights.ple_embd = &tensors[5];
    var spans = std.mem.zeroes(c.ds4_model_map_span_vec);
    try expect(lifecycle.initialStreamingSpans(&weights, &shape, &spans));
    {
        defer c.free(spans.v);
        try equal(@as(u32, 4), spans.len);
        for (spans.v[0..spans.len], 0..) |span, index| {
            try equal(tensors[index].abs_offset, span.off);
            try equal(tensors[index].abs_offset + tensors[index].bytes, span.end);
        }
    }
    // The adjacent paged GLM executor still starts from token-only views and
    // owns its later layer transitions; Qwen must not change that policy.
    shape.family = c.DS4_MODEL_FAMILY_GLM_DSA;
    try expect(lifecycle.initialStreamingSpans(&weights, &shape, &spans));
    defer c.free(spans.v);
    try equal(@as(u32, 1), spans.len);
    try equal(tensors[0].abs_offset, spans.v[0].off);
}

test "explicit chunks clamp and override environment; graph bytes are monotone" {
    const saved = c.getenv("DS4_QWEN4_PREFILL_CHUNK");
    const keep = if (saved == null) null else c.strdup(saved);
    defer {
        if (keep) |text| {
            _ = c.setenv("DS4_QWEN4_PREFILL_CHUNK", text, 1);
            c.free(text);
        } else {
            _ = c.unsetenv("DS4_QWEN4_PREFILL_CHUNK");
        }
    }
    _ = c.setenv("DS4_QWEN4_PREFILL_CHUNK", "32", 1);
    var config = std.mem.zeroes(c.ds4_qwen4_config);
    config.shape = metadata.profile(false);
    for ([_]u32{ 1, 128, 511, 512, 513, std.math.maxInt(u32) }, [_]u32{ 1, 128, 511, 512, 512, 512 }) |request, expected| {
        try equal(expected, memory.resolveChunk(512, request));
        try equal(expected, memory.estimate(&config, 512, request).prefill_cap);
    }
    try equal(@as(u32, 32), memory.estimate(&config, 512, 0).prefill_cap);
    var previous: u64 = 0;
    for ([_]u32{ 1, 2, 8, 128, 1024, 4096, 8192 }) |rows| {
        const value = memory.estimate(&config, 8192, rows);
        try expect(value.total_bytes > previous);
        try equal(value.total_bytes, value.raw_bytes + value.compressed_bytes + value.scratch_bytes);
        previous = value.total_bytes;
    }
    const large = memory.estimate(&config, std.math.maxInt(i32), std.math.maxInt(u32));
    try expect(large.total_bytes > previous);
}

test "host budget does not discount an already smaller Metal recommendation" {
    const gib: u64 = 1024 * 1024 * 1024;
    try equal(24 * gib, admission.budget(32 * gib, 24 * gib));
    try equal(28 * gib, admission.budget(32 * gib, 32 * gib));
    try equal(@as(u64, 0), admission.budget(0, 24 * gib));
    const high = admission.budget(std.math.maxInt(u64), std.math.maxInt(u64));
    try expect(high > std.math.maxInt(u64) / 2 and high < std.math.maxInt(u64));
}

test "dense query cutoff follows each row's causal block universe" {
    for ([_]u32{ 2, 512 }) |blocks| {
        var position: u32 = 0;
        while (position <= blocks * 4 + 132) : (position += 1) {
            for ([_]u32{ 0, 1, 2, 3, 4, 8, 63, 128 }) |tokens| {
                var dense: u32 = 0;
                for (0..tokens) |row| {
                    if ((@as(u64, position) + row + 1) / 4 <= blocks) dense += 1;
                }
                try equal(dense, attention.denseRows(position, tokens, blocks));
                try equal(dense < tokens, attention.needsIndexQuery(position, tokens, blocks, false, false));
                try equal(tokens != 0, attention.needsIndexQuery(position, tokens, blocks, true, false));
                try expect(!attention.needsIndexQuery(position, tokens, blocks, true, true));
            }
        }
    }
}

test "history setting affects proposals only and disabled MTP ignores it" {
    for ([_]?[]const u8{ null, "", "auto" }) |value| {
        try equal(@as(?bool, false), mtp.historyPolicy(true, false, value));
        try equal(@as(?bool, true), mtp.historyPolicy(true, true, value));
    }
    for ([_][]const u8{ "on", "1" }) |value| try equal(@as(?bool, true), mtp.historyPolicy(true, false, value));
    for ([_][]const u8{ "off", "0" }) |value| try equal(@as(?bool, false), mtp.historyPolicy(true, true, value));
    for ([_][]const u8{ "2", "invalid", "ON", "AUTO", " on", "off ", "auto-extra" }) |value| {
        try equal(@as(?bool, null), mtp.historyPolicy(true, true, value));
        try equal(@as(?bool, false), mtp.historyPolicy(false, true, value));
    }
}

test "v3 canonical payload counts only initialized prefix and pending trunk row" {
    var config = std.mem.zeroes(c.ds4_qwen4_config);
    config.shape = metadata.profile(false);
    const bytes = @as(u64, config.shape.n_embd) * config.shape.n_hc * 4;
    try equal(payload.tensorBytes(&config, 13, 12) + bytes, payload.tensorBytesWithTail(&config, 13, 12, true));
    var graph = std.mem.zeroes(c.ds4_qwen4_gpu_graph);
    graph.mtp_R = @ptrFromInt(16);
    graph.mtp_tail_valid = true;
    graph.mtp_tail_pos = 12;
    graph.mtp_pos = 12;
    try expect(payload.hasTail(&graph, 13));
    graph.mtp_pos = 11;
    try expect(!payload.hasTail(&graph, 13));
}

test "streaming windows include padded down rows and off-slab predictor fallback" {
    var shape = metadata.profile(false);
    var gate = std.mem.zeroes(c.ds4_tensor);
    gate.type = c.DS4_TENSOR_IQ2_XXS;
    gate.ndim = 3;
    gate.dim = .{ 2560, 640, 512, 0, 0, 0, 0, 0 };
    var down = gate;
    down.type = c.DS4_TENSOR_Q2_K;
    down.dim = .{ 768, 2560, 512, 0, 0, 0, 0, 0 };
    var predictor_gate = gate;
    predictor_gate.type = c.DS4_TENSOR_Q4_K;
    var predictor_down = down;
    predictor_down.type = c.DS4_TENSOR_MXFP4;
    predictor_down.dim[0] = 640;
    var weights = std.mem.zeroes(c.ds4_weights);
    for (weights.layer[0..shape.n_layer], 0..) |*layer, i| {
        layer.ffn_gate_exps = if (i + 1 == shape.n_layer) &predictor_gate else &gate;
        layer.ffn_up_exps = layer.ffn_gate_exps;
        layer.ffn_down_exps = if (i + 1 == shape.n_layer) &predictor_down else &down;
    }
    const trunk: u64 = 1489920 * 512;
    try equal(@as(?u64, 2 * trunk), admission.staging(&shape, &weights, true));
    try equal(@as(?u64, 2 * trunk), admission.staging(&shape, &weights, false));
    predictor_gate.dim[1] *= 2;
    predictor_down.dim[1] *= 2;
    try equal(@as(?u64, 2 * @as(u64, 2713600) * 512), admission.staging(&shape, &weights, true));
    predictor_gate.dim[1] /= 2;
    predictor_down.dim[1] /= 2;
    shape.n_expert = 16;
    try equal(@as(?u64, 2 * (2713600 * 10 + 3 * 16 * 8)), admission.staging(&shape, &weights, true));
    shape.n_expert_used = 17;
    try equal(@as(?u64, null), admission.staging(&shape, &weights, true));
}
