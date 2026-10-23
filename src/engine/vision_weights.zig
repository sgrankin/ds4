//! Strict DeepSeek vision-sidecar metadata and tensor directory binding.
//! Offsets borrow the Model mapping. V4.1 visual router biases live in the
//! language GGUF; the pinned V4 Vision-Exp sidecar also owns those offsets.
//! This is the binding owner, separate from the native 32-block encoder.
const std = @import("std");
const c = @import("gguf_c");
const Model = c.ds4_model;
const Tensor = c.ds4_tensor;
const Weights = c.ds4_deepseek4_vision_weights;

fn bytes(value: c.ds4_str) []const u8 {
    return if (value.len == 0) &.{} else value.ptr[0..@intCast(value.len)];
}
fn string(model: *const Model, key: [*:0]const u8, expected: []const u8) bool {
    var value: c.ds4_str = undefined;
    return c.ds4_gguf_get_string(model, key, &value) and std.mem.eql(u8, bytes(value), expected);
}
fn typeName(tag: u32) [*c]const u8 {
    const info = c.ds4_gguf_tensor_type(tag);
    return if (info == null) "unknown" else info.*.name;
}
fn find(model: *const Model, name: []const u8) ?*Tensor {
    if (model.n_tensors == 0) return null;
    const tensors: [*]Tensor = @ptrCast(model.tensors);
    for (tensors[0..@intCast(model.n_tensors)]) |*tensor|
        if (std.mem.eql(u8, bytes(tensor.name), name)) return tensor;
    return null; // The first duplicate directory entry is authoritative.
}
fn requiredOffset(model: *const Model, name: [:0]const u8, kind: u32, dimensions: []const u64) u64 {
    const tensor = find(model, name) orelse {
        _ = c.fprintf(c.__stderrp, "ds4: required tensor is missing: %s\n", name.ptr);
        c.exit(1);
    };
    if (tensor.type != kind or tensor.ndim != dimensions.len) {
        _ = c.fprintf(c.__stderrp, "ds4: DeepSeek vision tensor %s has type %s/rank %u, expected %s/rank %u\n", name.ptr, typeName(tensor.type), tensor.ndim, typeName(kind), @as(c_uint, @intCast(dimensions.len)));
        c.exit(1);
    }
    for (dimensions, 0..) |dimension, index| if (tensor.dim[index] != dimension) {
        _ = c.fprintf(c.__stderrp, "ds4: DeepSeek vision tensor %s has dim[%u]=%llu, expected %llu\n", name.ptr, @as(c_uint, @intCast(index)), @as(c_ulonglong, tensor.dim[index]), @as(c_ulonglong, dimension));
        c.exit(1);
    };
    return tensor.abs_offset;
}
fn expectU32(model: *const Model, key: [*:0]const u8, label: [*:0]const u8, expected: u32, shape: *const c.ds4_shape) void {
    var value: u32 = undefined;
    if (!c.ds4_gguf_get_u32(model, key, &value)) {
        _ = c.fprintf(c.__stderrp, "ds4: required metadata key is missing: %s\n", key);
        c.exit(1);
    }
    if (value != expected) {
        _ = c.fprintf(c.__stderrp, "ds4: expected %s=%u for %s, got %u\n", label, expected, shape.name, value);
        c.exit(1);
    }
}
fn expectEpsilon(model: *const Model, shape: *const c.ds4_shape) void {
    @setFloatMode(.optimized); // Same F32 tolerance arithmetic as the C TU.
    const key = "deepseek4-vision.attention.layer_norm_rms_epsilon";
    var value: f32 = undefined;
    if (!c.ds4_gguf_get_f32_compat(model, key, &value)) {
        _ = c.fprintf(c.__stderrp, "ds4: required metadata key is missing: %s\n", @as([*:0]const u8, key));
        c.exit(1);
    }
    const expected: f32 = 1.0e-6;
    const tolerance: f32 = @max(@abs(expected) * 1.0e-5, 1.0e-30);
    if (@abs(value - expected) <= tolerance) return;
    _ = c.fprintf(c.__stderrp, "ds4: expected %s=%.9g for %s, got %.9g\n", @as([*:0]const u8, "DeepSeek vision RMS epsilon"), @as(f64, expected), shape.name, @as(f64, value));
    c.exit(1);
}

pub fn bind(weights: *Weights, model: *const Model, target: *const c.ds4_shape) void {
    const v41 = target.family == c.DS4_MODEL_FAMILY_DEEPSEEK41;
    const projection: u32 = if (v41) 5120 else 4096;
    if (!string(model, "general.architecture", "deepseek4-vision"))
        c.ds4_engine_model_die("--vision file is not a DeepSeek V4 vision encoder GGUF");
    // V4.1 has several released checkpoints. Engine open pairs source revision
    // against the language GGUF; this binder checks presence and sidecar kind.
    var revision: c.ds4_str = undefined;
    if (!c.ds4_gguf_get_string(model, "general.source.revision", &revision) or
        (!v41 and !std.mem.eql(u8, bytes(revision), "e46e16bf6035c6f317eb2ac7458eb0362926d402")) or
        !string(model, "deepseek4-vision.checkpoint_variant", if (v41) "v4.1-flash" else "vision-exp"))
        c.ds4_engine_model_die("DeepSeek vision sidecar does not match the language checkpoint");
    const expected_tensors: u32 = if (v41) 266 else 316;
    if (model.n_tensors != expected_tensors) {
        _ = c.fprintf(c.__stderrp, "ds4: DeepSeek vision GGUF has %llu tensors, expected %u\n", @as(c_ulonglong, model.n_tensors), expected_tensors);
        c.exit(1);
    }
    expectU32(model, "deepseek4-vision.block_count", "DeepSeek vision block_count", 32, target);
    expectU32(model, "deepseek4-vision.embedding_length", "DeepSeek vision embedding_length", 1024, target);
    expectU32(model, "deepseek4-vision.feed_forward_length", "DeepSeek vision feed_forward_length", 2816, target);
    expectU32(model, "deepseek4-vision.attention.head_count", "DeepSeek vision head_count", 16, target);
    expectU32(model, "deepseek4-vision.projection_length", "DeepSeek vision projection_length", projection, target);
    expectU32(model, "deepseek4-vision.patch_size", "DeepSeek vision patch_size", 14, target);
    expectU32(model, "deepseek4-vision.downsample_ratio", "DeepSeek vision downsample_ratio", 3, target);
    expectU32(model, "deepseek4-vision.image.max_tokens", "DeepSeek vision max_tokens", if (v41) 1024 else 384, target);
    expectU32(model, "deepseek4-vision.image.min_pixels", "DeepSeek vision min_pixels", if (v41) 295936 else 147456, target);
    expectU32(model, "deepseek4-vision.image.max_width_height_ratio", "DeepSeek vision max_width_height_ratio", if (v41) 0 else 8, target);
    if (v41) expectU32(model, "deepseek4-vision.image.token_id", "DeepSeek V4.1 image token", 129264, target);
    expectU32(model, "deepseek4-vision.language.block_count", "DeepSeek vision language layers", if (v41) 40 else 43, target);
    expectU32(model, "deepseek4-vision.language.expert_count", "DeepSeek vision language experts", if (v41) 384 else 256, target);
    expectEpsilon(model, target);

    // Preserve failure publication: metadata validation precedes zeroing the
    // output record; subsequent required lookups populate it in source order.
    weights.* = std.mem.zeroes(Weights);
    weights.projection_dim = projection;
    weights.patch_weight = requiredOffset(model, "vision.patch_embed.proj.weight", c.DS4_TENSOR_BF16, &.{ 588, 1024 });
    weights.patch_bias = requiredOffset(model, "vision.patch_embed.proj.bias", c.DS4_TENSOR_BF16, &.{1024});
    weights.post_norm = requiredOffset(model, "vision.norm.weight", c.DS4_TENSOR_BF16, &.{1024});
    weights.aligner_w1 = requiredOffset(model, "aligner.w1.weight", c.DS4_TENSOR_BF16, &.{ 9216, projection });
    weights.aligner_w1_bias = requiredOffset(model, "aligner.w1.bias", c.DS4_TENSOR_BF16, &.{projection});
    weights.aligner_w2 = requiredOffset(model, "aligner.w2.weight", c.DS4_TENSOR_BF16, &.{ projection, projection });
    weights.aligner_w2_bias = requiredOffset(model, "aligner.w2.bias", c.DS4_TENSOR_BF16, &.{projection});
    weights.image_start = requiredOffset(model, "image_start", c.DS4_TENSOR_BF16, &.{projection});
    if (!v41) weights.image_pad = requiredOffset(model, "image_pad", c.DS4_TENSOR_BF16, &.{projection});
    weights.image_newline = requiredOffset(model, "image_newline", c.DS4_TENSOR_BF16, &.{projection});
    weights.image_end = requiredOffset(model, "image_end", c.DS4_TENSOR_BF16, &.{projection});

    for (&weights.layer, 0..) |*layer, index| {
        layer.norm1 = layerOffset(model, index, "norm1.weight", &.{1024});
        layer.qkv_weight = layerOffset(model, index, "attn.wqkv.weight", &.{ 1024, 3072 });
        layer.qkv_bias = layerOffset(model, index, "attn.wqkv.bias", &.{3072});
        layer.attn_proj_weight = layerOffset(model, index, "attn.wo.weight", &.{ 1024, 1024 });
        layer.attn_proj_bias = layerOffset(model, index, "attn.wo.bias", &.{1024});
        layer.norm2 = layerOffset(model, index, "norm2.weight", &.{1024});
        layer.mlp_w1 = layerOffset(model, index, "mlp.w1.weight", &.{ 1024, 5632 });
        layer.mlp_w2 = layerOffset(model, index, "mlp.w2.weight", &.{ 2816, 1024 });
    }
    // V4.1 visual routing biases are bound from the language model, not here.
    if (v41) return;
    for (&weights.visual_router_bias, 0..) |*offset, index| {
        var storage: [128]u8 = undefined;
        const name = std.mem.printSentinel(&storage, "layers.{d}.ffn.gate.bias_vl", .{index}, 0) catch c.ds4_engine_model_die("DeepSeek visual router tensor name overflow");
        offset.* = requiredOffset(model, name, c.DS4_TENSOR_F32, &.{256});
    }
    for (&weights.mtp_visual_router_bias, 0..) |*offset, index| {
        var storage: [128]u8 = undefined;
        const name = std.mem.printSentinel(&storage, "mtp.{d}.ffn.gate.bias_vl", .{index}, 0) catch c.ds4_engine_model_die("DeepSeek MTP visual router tensor name overflow");
        offset.* = requiredOffset(model, name, c.DS4_TENSOR_F32, &.{256});
    }
    for (&weights.hash_router_bias, 0..) |*offset, index| {
        var storage: [128]u8 = undefined;
        const name = std.mem.printSentinel(&storage, "layers.{d}.ffn.gate.bias", .{index}, 0) catch c.ds4_engine_model_die("DeepSeek hash router bias tensor name overflow");
        offset.* = requiredOffset(model, name, c.DS4_TENSOR_F32, &.{256});
    }
}
fn layerOffset(model: *const Model, layer: usize, suffix: []const u8, dimensions: []const u64) u64 {
    var storage: [128]u8 = undefined;
    const name = std.mem.printSentinel(&storage, "vision.blocks.{d}.{s}", .{ layer, suffix }, 0) catch c.ds4_engine_model_die("DeepSeek vision tensor name overflow");
    return requiredOffset(model, name, c.DS4_TENSOR_BF16, dimensions);
}
