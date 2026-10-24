//! Synthetic descriptor fixtures exercise CPU policy without model/device work.
//! The only stand-in is the small GGUF format table used by byte accounting.
const std = @import("std");
const c = @import("gguf_c");
const model = @import("model_spans");

const q2: c.gguf_type_info = .{ .name = "q2_k", .block_elems = 256, .block_bytes = 84 };
const q4: c.gguf_type_info = .{ .name = "q4_k", .block_elems = 256, .block_bytes = 144 };
const iq2: c.gguf_type_info = .{ .name = "iq2_xxs", .block_elems = 256, .block_bytes = 66 };
export fn ds4_gguf_tensor_type(kind: u32) [*c]const c.gguf_type_info {
    return switch (kind) {
        c.DS4_TENSOR_Q2_K => &q2,
        c.DS4_TENSOR_Q4_K => &q4,
        c.DS4_TENSOR_IQ2_XXS => &iq2,
        else => null,
    };
}
export fn ds4_engine_model_die(message: [*c]const u8) noreturn {
    std.debug.panic("unexpected model diagnostic: {s}", .{std.mem.span(message)});
}
fn weight(offset: u64, bytes: u64) c.ds4_tensor {
    var value = std.mem.zeroes(c.ds4_tensor);
    value.type = c.DS4_TENSOR_F32;
    value.abs_offset = offset;
    value.bytes = bytes;
    return value;
}
fn expert(kind: u32, offset: u64) c.ds4_tensor {
    var value = weight(offset, @as(u64, ds4_gguf_tensor_type(kind).*.block_bytes) * 2 * 128);
    value.type = kind;
    value.ndim = 3;
    value.dim[0] = 256;
    value.dim[1] = 2;
    value.dim[2] = 128;
    return value;
}
fn clear(vector: *model.Vec) void {
    c.free(vector.v);
    vector.* = std.mem.zeroes(model.Vec);
}

test "model spans separate guard/runtime policy and retain outlier/resident weights" {
    const expect = std.testing.expect;
    const equal = std.testing.expectEqual;
    for ([_][*:0]const u8{
        "DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE",
        "DS4_METAL_MOE_WRITE_CLAMPED_ACT",
        "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
        "DS4_METAL_DISABLE_IQ2_SELECTED_EXPERT_VIEWS",
        "DS4_METAL_DISABLE_IQ2_STREAM_ADDR_TABLE",
    }) |name| _ = c.unsetenv(name);
    var shape = std.mem.zeroes(c.ds4_shape);
    shape.family = c.DS4_MODEL_FAMILY_DEEPSEEK41;
    shape.n_layer = 3;
    shape.n_expert = 128;
    shape.n_expert_used = 6;
    var policy = std.mem.zeroes(c.ds4_streaming_policy);
    var weights = std.mem.zeroes(c.ds4_weights);
    var token = weight(100, 40);
    var output = weight(900000, 128);
    var ple = weight(1000000, 1024);
    weights.token_embd = &token;
    weights.output = &output;
    weights.ple_embd = &ple;
    var norms = [_]c.ds4_tensor{ weight(200, 20), weight(400, 30), weight(600, 40) };
    var experts: [3][3]c.ds4_tensor = undefined;
    for (&experts, 0..) |*layer, index| {
        const kind: u32 = if (index == 1) c.DS4_TENSOR_Q4_K else c.DS4_TENSOR_Q2_K;
        for (layer, 0..) |*tensor, projection| tensor.* = expert(kind, 1000 + index * 300000 + projection * 80000);
        weights.layer[index].attn_norm = &norms[index];
        weights.layer[index].ffn_gate_exps = &layer[0];
        weights.layer[index].ffn_up_exps = &layer[1];
        weights.layer[index].ffn_down_exps = &layer[2];
    }
    try expect(model.uniform(&weights, &shape, 0));
    try expect(!model.uniform(&weights, &shape, 1));
    try expect(!model.staticSupported(&weights, &shape, &policy));
    var vector = std.mem.zeroes(model.Vec);
    defer clear(&vector);
    const last = std.math.maxInt(u32);
    try expect(model.build(&weights, &shape, &policy, 0, last, true, true, .static, &vector));
    try equal(@as(u64, 258), model.total(&vector));
    try equal(@as(u32, 5), vector.len);
    clear(&vector);
    try expect(model.build(&weights, &shape, &policy, 0, last, true, true, .runtime, &vector));
    try equal(@as(u64, 258 + 3 * 144 * 2 * 128), model.total(&vector));
    clear(&vector);
    try expect(model.build(&weights, &shape, &policy, 1, last, false, false, .static, &vector));
    try equal(@as(u64, 70), model.total(&vector));
    clear(&vector);
    try expect(model.tokenOnly(&weights, &vector));
    try equal(@as(u64, 40), model.total(&vector));
    clear(&vector);
    try expect(model.outputOnly(&weights, &vector));
    try equal(@as(u64, 128), model.total(&vector)); // Disk-only PLE is excluded.
    clear(&vector);
    var guarded: u64 = 0;
    try expect(model.staticBytes(&weights, &shape, true, true, &guarded));
    try equal(@as(u64, 258), guarded);
    vector.len = 123;
    try expect(!model.build(&weights, &shape, &policy, 3, last, false, false, .full, &vector));
    try equal(@as(u32, 123), vector.len);
    vector.len = 0;

    shape.family = c.DS4_MODEL_FAMILY_GLM_DSA;
    try expect(model.glmAddrLayout(&weights, &shape, &weights.layer[0], 0));
    try expect(!model.glmAddrLayout(&weights, &shape, &weights.layer[1], 1));
    policy.resident_start = 2;
    policy.resident_layers = 1;
    try expect(model.residentLayer(&shape, &policy, &weights.layer[2], 2));
    try expect(!model.residentLayer(&shape, &policy, &weights.layer[0], 0));
    try expect(model.build(&weights, &shape, &policy, 0, last, true, true, .runtime, &vector));
    try equal(@as(u64, 258 + 3 * 144 * 2 * 128 + 3 * 84 * 2 * 128), model.total(&vector));
    clear(&vector);
    try expect(!model.staticSupported(&weights, &shape, &policy));
    policy.static_decode_map = true;
    try expect(model.staticSupported(&weights, &shape, &policy));
    experts[0][0].type = c.DS4_TENSOR_IQ2_XXS;
    experts[0][1].type = c.DS4_TENSOR_IQ2_XXS;
    try expect(model.glmSelectedLayout(&shape, &weights.layer[0], 0));
    shape.n_expert_used = 8;
    try expect(!model.glmSelectedLayout(&shape, &weights.layer[0], 0));
    experts[0][2].type = c.DS4_TENSOR_IQ2_XXS;
    try expect(model.glmSelectedLayout(&shape, &weights.layer[0], 0));
    _ = c.setenv("DS4_METAL_DISABLE_IQ2_STREAM_ADDR_TABLE", "1", 1);
    defer _ = c.unsetenv("DS4_METAL_DISABLE_IQ2_STREAM_ADDR_TABLE");
    try expect(!model.glmSelectedLayout(&shape, &weights.layer[0], 0));
}

test "PRO group views stay isolated and totals saturate" {
    const expect = std.testing.expect;
    const equal = std.testing.expectEqual;
    _ = c.setenv("DS4_METAL_Q4_PRO_MAP_GROUPS", "3", 1);
    defer _ = c.unsetenv("DS4_METAL_Q4_PRO_MAP_GROUPS");
    const gib = 1024 * 1024 * 1024;
    var tensor = weight(2 * gib, 3 * gib);
    tensor.type = c.DS4_TENSOR_Q4_K;
    tensor.ndim = 3;
    tensor.dim[2] = 384;
    var spans = model.Spans{};
    defer spans.deinit();
    spans.include(&tensor);
    try expect(spans.finish());
    try equal(@as(usize, 3), spans.items.items.len);
    try equal(@as(u64, gib), spans.max_tensor_bytes);
    for (spans.items.items, 0..) |span, index| {
        try expect(span.isolate);
        try equal(@as(u64, (2 + index) * gib), span.off);
        try equal(@as(u64, gib), span.end - span.off);
    }
    var huge = [_]model.Span{
        .{ .off = 0, .end = std.math.maxInt(u64), .isolate = true },
        .{ .off = 0, .end = 1, .isolate = true },
    };
    const vector: model.Vec = .{ .v = &huge, .len = 2, .cap = 2, .max_tensor_bytes = std.math.maxInt(u64) };
    try equal(std.math.maxInt(u64), model.total(&vector));
}
