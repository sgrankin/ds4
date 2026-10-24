//! Support-model admission and borrowed weight tables. Directory/metadata
//! strings and every bound tensor pointer live only as long as Model's mmap.
//! DSpark is optional: missing/invalid tensors accumulate diagnostics and the
//! enclosing Engine decides whether runtime use is admissible. Legacy MTP
//! retains its immediate fatal missing/layout policy.
const std = @import("std");
const c = @import("gguf_c");
const Model = c.ds4_model;
const Tensor = c.ds4_tensor;
const Summary = c.ds4_dspark_summary;
const Shape = c.ds4_shape;
const Draft = c.ds4_dspark_weights;

fn bytes(value: c.ds4_str) []const u8 {
    return if (value.len == 0) &.{} else value.ptr[0..@intCast(value.len)];
}
fn tensors(model: *const Model) []Tensor {
    return if (model.n_tensors == 0) &.{} else @as([*]Tensor, @ptrCast(model.tensors))[0..@intCast(model.n_tensors)];
}
fn find(model: *const Model, name: []const u8) ?*Tensor {
    for (tensors(model)) |*tensor| if (std.mem.eql(u8, bytes(tensor.name), name)) return tensor;
    return null; // Directory order/first duplicate wins, as in the target binder.
}
fn string(model: *const Model, key: [*:0]const u8) ?[]const u8 {
    var value: c.ds4_str = undefined;
    return if (c.ds4_gguf_get_string(model, key, &value)) bytes(value) else null;
}
fn isString(model: *const Model, key: [*:0]const u8, expected: []const u8) bool {
    return if (string(model, key)) |value| std.mem.eql(u8, value, expected) else false;
}
fn u32Any(model: *const Model, keys: []const [*:0]const u8, out: *u32) bool {
    for (keys) |key| if (c.ds4_gguf_get_u32(model, key, out)) return true;
    return false;
}
fn arrayAny(model: *const Model, keys: []const [*:0]const u8, out: []u32, count: *u32) bool {
    for (keys) |key| {
        var array: c.ds4_array_ref = undefined;
        if (!c.ds4_gguf_get_array(model, key, &array)) continue;
        if (array.type != c.GGUF_VALUE_UINT32 and array.type != c.GGUF_VALUE_INT32) continue;
        var cursor = c.ds4_cursor{ .base = model.map, .size = model.size, .pos = array.data_pos, .@"error" = @splat(0) };
        const length: usize = @intCast(@min(array.len, @as(u64, out.len)));
        for (out[0..length]) |*value| {
            if (array.type == c.GGUF_VALUE_UINT32) {
                if (!c.ds4_gguf_u32(&cursor, value)) return false;
            } else {
                var signed: i32 = undefined;
                if (!c.ds4_gguf_read(&cursor, &signed, @sizeOf(i32)) or signed < 0) return false;
                value.* = @intCast(signed);
            }
        }
        count.* = @intCast(length);
        return true;
    }
    return false;
}
fn mtpStage(name: []const u8) ?u32 {
    if (!std.mem.startsWith(u8, name, "mtp.")) return null;
    var position: usize = 4;
    if (position == name.len or !std.ascii.isDigit(name[position])) return null;
    var stage: u32 = 0;
    while (position < name.len and std.ascii.isDigit(name[position])) : (position += 1) {
        const digit: u32 = name[position] - '0';
        if (stage > (std.math.maxInt(u32) - digit) / 10) return null;
        stage = stage * 10 + digit;
    }
    return if (position < name.len and name[position] == '.') stage else null;
}

pub fn summarize(model: *const Model) Summary {
    var result = std.mem.zeroes(Summary);
    const block_keys = [_][*:0]const u8{ "deepseek4.dspark.block_size", "deepseek4.dspark_block_size", "dspark.block_size" };
    const rank_keys = [_][*:0]const u8{ "deepseek4.dspark.markov_rank", "deepseek4.dspark_markov_rank", "dspark.markov_rank" };
    const noise_keys = [_][*:0]const u8{ "deepseek4.dspark.noise_token_id", "deepseek4.dspark_noise_token_id", "dspark.noise_token_id" };
    const layer_keys = [_][*:0]const u8{ "deepseek4.dspark.target_layer_ids", "deepseek4.dspark_target_layer_ids", "dspark.target_layer_ids" };
    result.has_block_size = u32Any(model, &block_keys, &result.block_size);
    result.has_markov_rank = u32Any(model, &rank_keys, &result.markov_rank);
    result.has_noise_token_id = u32Any(model, &noise_keys, &result.noise_token_id);
    result.has_target_layers = arrayAny(model, &layer_keys, &result.target_layers, &result.target_layer_count);
    result.has_metadata = result.has_block_size or result.has_markov_rank or result.has_noise_token_id or result.has_target_layers;
    var maximum: ?u32 = null;
    for (tensors(model)) |tensor| {
        const name = bytes(tensor.name);
        const stage = mtpStage(name) orelse continue;
        maximum = if (maximum) |old| @max(old, stage) else stage;
        result.has_main_proj = result.has_main_proj or std.mem.find(u8, name, ".main_proj.") != null;
        result.has_main_norm = result.has_main_norm or std.mem.find(u8, name, ".main_norm.") != null;
        result.has_markov_head = result.has_markov_head or std.mem.find(u8, name, ".markov_head.") != null;
        result.has_confidence_head = result.has_confidence_head or std.mem.find(u8, name, ".confidence_head.") != null;
        result.has_final_head = result.has_final_head or std.mem.find(u8, name, ".hc_head_") != null or std.mem.find(u8, name, ".norm.weight") != null;
    }
    if (maximum) |stage| result.stages = stage +% 1;
    return result;
}
pub fn detect(model: ?*const Model, stages: ?*u32, output: ?*Summary) c.ds4_support_kind {
    if (stages) |value| value.* = 0;
    if (output) |value| value.* = std.mem.zeroes(Summary);
    const source = model orelse return c.DS4_SUPPORT_NONE;
    const summary = summarize(source);
    if (output) |value| value.* = summary;
    if (stages) |value| value.* = summary.stages;
    if (summary.stages >= 3 and summary.has_main_proj and summary.has_markov_head and summary.has_confidence_head) return c.DS4_SUPPORT_DSPARK;
    if (find(source, "mtp.0.e_proj.weight") != null and find(source, "mtp.0.h_proj.weight") != null and find(source, "mtp.0.hc_head_base.weight") != null) {
        if (stages) |value| value.* = if (summary.stages != 0) summary.stages else 1;
        return c.DS4_SUPPORT_MTP_LEGACY;
    }
    return c.DS4_SUPPORT_NONE;
}
pub fn compatible(model: *const Model, target_family: c.ds4_model_family) bool {
    const family = string(model, "dspark.target_architecture");
    const arch = string(model, "general.architecture");
    // Released V4.1 files can omit the redundant family key. The distinct
    // architecture still identifies them; an explicit family must agree.
    const v41_arch = if (arch) |value| std.mem.eql(u8, value, "deepseek41-dspark") else false;
    const v41 = v41_arch and (family == null or std.mem.eql(u8, family.?, "deepseek41"));
    if ((target_family == c.DS4_MODEL_FAMILY_DEEPSEEK41) != v41 or
        (family != null and !v41 and !std.mem.eql(u8, family.?, "deepseek4")) or (v41_arch and !v41))
    {
        _ = c.fputs("ds4: DSpark support model and target checkpoint families differ\n", c.__stderrp);
        return false;
    }
    const variant = string(model, "deepseek4.checkpoint_variant");
    if (variant != null and std.mem.eql(u8, variant.?, "vision-exp")) {
        _ = c.fputs("ds4: retired V4 Flash Vision-Exp support model\n", c.__stderrp);
        return false;
    }
    return true;
}

const Kind = enum {
    f32,
    plain,
    dense,
    routed,
    fn name(self: Kind) [*:0]const u8 {
        return switch (self) {
            .f32 => "F32",
            .plain => "F16 or F32",
            .dense => "F16, F32, or Q8_0",
            .routed => "routed expert quant",
        };
    }
    fn accepts(self: Kind, tag: u32) bool {
        return switch (self) {
            .f32 => tag == c.DS4_TENSOR_F32,
            .plain => tag == c.DS4_TENSOR_F16 or tag == c.DS4_TENSOR_F32,
            .dense => tag == c.DS4_TENSOR_F16 or tag == c.DS4_TENSOR_F32 or tag == c.DS4_TENSOR_Q8_0,
            .routed => switch (tag) {
                c.DS4_TENSOR_Q8_0, c.DS4_TENSOR_IQ2_XXS, c.DS4_TENSOR_Q2_K, c.DS4_TENSOR_Q4_K, c.DS4_TENSOR_Q5_K, c.DS4_TENSOR_Q6_K, c.DS4_TENSOR_MXFP4 => true,
                else => false,
            },
        };
    }
};
fn typeName(tag: u32) [*c]const u8 {
    const info = c.ds4_gguf_tensor_type(tag);
    return if (info == null) "unknown" else info.*.name;
}
fn metadataError(draft: *Draft, message: [*:0]const u8) void {
    _ = c.fprintf(c.__stderrp, "ds4: DSpark metadata error: %s\n", message);
    draft.metadata_errors +%= 1;
}
fn validateTensor(draft: *Draft, pointer: [*c]const Tensor, role: [*:0]const u8, kind: Kind, shape: []const u64) void {
    if (pointer == null) return;
    const tensor: *const Tensor = @ptrCast(pointer);
    var valid = true;
    if (!kind.accepts(tensor.type)) {
        _ = c.fprintf(c.__stderrp, "ds4: DSpark tensor %.*s (%s) has type %s, expected %s\n", @as(c_int, @intCast(tensor.name.len)), tensor.name.ptr, role, typeName(tensor.type), kind.name());
        valid = false;
    }
    if (tensor.ndim != shape.len) {
        _ = c.fprintf(c.__stderrp, "ds4: DSpark tensor %.*s (%s) has %u dimensions, expected %u\n", @as(c_int, @intCast(tensor.name.len)), tensor.name.ptr, role, tensor.ndim, @as(c_uint, @intCast(shape.len)));
        valid = false;
    }
    for (shape[0..@min(@as(usize, tensor.ndim), shape.len)], 0..) |dimension, index| {
        if (tensor.dim[index] == dimension) continue;
        _ = c.fprintf(c.__stderrp, "ds4: DSpark tensor %.*s (%s) has dim[%u]=%llu, expected %llu\n", @as(c_int, @intCast(tensor.name.len)), tensor.name.ptr, role, @as(c_uint, @intCast(index)), @as(c_ulonglong, tensor.dim[index]), @as(c_ulonglong, dimension));
        valid = false;
    }
    // Report every bad property but count this tensor only once.
    if (!valid) draft.invalid_tensors +%= 1;
}
fn validateMetadata(draft: *Draft, shape: *const Shape) void {
    if (!draft.has_block_size or draft.block_size == 0) metadataError(draft, "missing or zero block size") else if (draft.block_size > c.DS4_DSPARK_MAX_BLOCK_SIZE) metadataError(draft, "block size exceeds runtime limit");
    if (!draft.has_markov_rank or draft.markov_rank == 0) metadataError(draft, "missing or zero Markov rank");
    if (!draft.has_noise_token_id or draft.noise_token_id >= shape.n_vocab) metadataError(draft, "missing or out-of-range noise token");
    if (!draft.has_target_layers or draft.target_layer_count == 0) {
        metadataError(draft, "missing target layer list");
        return;
    }
    var previous: u32 = std.math.maxInt(u32);
    for (draft.target_layers[0..draft.target_layer_count], 0..) |layer, index| {
        if (layer >= shape.n_layer) metadataError(draft, "target layer is outside the target model");
        if (index != 0 and layer <= previous) metadataError(draft, "target layers are not strictly increasing");
        previous = layer;
    }
}
fn validateBlock(draft: *Draft, layer: *const c.ds4_layer_weights, shape: *const Shape) void {
    const embedding: u64 = shape.n_embd;
    const hidden = embedding * shape.n_hc;
    const mix: u64 = 2 * @as(u64, shape.n_hc) + @as(u64, shape.n_hc) * shape.n_hc;
    const query = @as(u64, shape.n_head) * shape.n_head_dim;
    const output_low = @as(u64, shape.n_out_group) * shape.n_lora_o;
    const experts: u64 = if (draft.v41) draft.n_experts else shape.n_expert;
    validateTensor(draft, layer.hc_attn_fn, "hc_attn_fn", .plain, &.{ hidden, mix });
    validateTensor(draft, layer.hc_attn_scale, "hc_attn_scale", .f32, &.{3});
    validateTensor(draft, layer.hc_attn_base, "hc_attn_base", .f32, &.{mix});
    validateTensor(draft, layer.attn_norm, "attn_norm", .f32, &.{embedding});
    validateTensor(draft, layer.attn_q_a, "attn_q_a", .dense, &.{ embedding, shape.n_lora_q });
    validateTensor(draft, layer.attn_q_a_norm, "attn_q_a_norm", .f32, &.{shape.n_lora_q});
    validateTensor(draft, layer.attn_q_b, "attn_q_b", .dense, &.{ shape.n_lora_q, query });
    validateTensor(draft, layer.attn_kv, "attn_kv", .dense, &.{ embedding, shape.n_head_dim });
    validateTensor(draft, layer.attn_kv_a_norm, "attn_kv_a_norm", .f32, &.{shape.n_head_dim});
    validateTensor(draft, layer.attn_sinks, "attn_sinks", .f32, &.{shape.n_head});
    validateTensor(draft, layer.attn_output_a, "attn_output_a", .dense, &.{ @as(u64, shape.n_head_dim) * (shape.n_head / shape.n_out_group), output_low });
    validateTensor(draft, layer.attn_output_b, "attn_output_b", .dense, &.{ output_low, embedding });
    validateTensor(draft, layer.hc_ffn_fn, "hc_ffn_fn", .plain, &.{ hidden, mix });
    validateTensor(draft, layer.hc_ffn_scale, "hc_ffn_scale", .f32, &.{3});
    validateTensor(draft, layer.hc_ffn_base, "hc_ffn_base", .f32, &.{mix});
    validateTensor(draft, layer.ffn_norm, "ffn_norm", .f32, &.{embedding});
    validateTensor(draft, layer.ffn_gate_inp, "ffn_gate_inp", .dense, &.{ embedding, experts });
    validateTensor(draft, layer.ffn_exp_probs_b, "exp_probs_b", .f32, &.{experts});
    validateTensor(draft, layer.ffn_gate_exps, "ffn_gate_exps", .routed, &.{ embedding, shape.n_ff_exp, experts });
    validateTensor(draft, layer.ffn_up_exps, "ffn_up_exps", .routed, &.{ embedding, shape.n_ff_exp, experts });
    validateTensor(draft, layer.ffn_down_exps, "ffn_down_exps", .routed, &.{ shape.n_ff_exp, embedding, experts });
    if (layer.ffn_gate_exps != null and layer.ffn_up_exps != null and layer.ffn_gate_exps.*.type != layer.ffn_up_exps.*.type) {
        _ = c.fputs("ds4: DSpark routed gate/up experts use different quant types\n", c.__stderrp);
        draft.invalid_tensors +%= 1;
    }
    validateTensor(draft, layer.ffn_gate_shexp, "ffn_gate_shexp", .dense, &.{ embedding, shape.n_ff_exp });
    validateTensor(draft, layer.ffn_up_shexp, "ffn_up_shexp", .dense, &.{ embedding, shape.n_ff_exp });
    validateTensor(draft, layer.ffn_down_shexp, "ffn_down_shexp", .dense, &.{ shape.n_ff_exp, embedding });
}
pub fn validate(draft: ?*Draft, shape: *const Shape) void {
    const weights = draft orelse return;
    validateMetadata(weights, shape);
    for (weights.stage[0..weights.n_stages], 0..) |*stage, index| {
        validateBlock(weights, &stage.block, shape);
        if (index == 0) {
            validateTensor(weights, stage.main_proj, "main_proj", .dense, &.{ @as(u64, weights.target_layer_count) * shape.n_embd, shape.n_embd });
            validateTensor(weights, stage.main_norm, "main_norm", .f32, &.{shape.n_embd});
        }
    }
    if (weights.n_stages == 0) return;
    const final = &weights.stage[weights.n_stages - 1];
    validateTensor(weights, final.norm, "norm", .f32, &.{shape.n_embd});
    validateTensor(weights, final.hc_head_base, "hc_head_base", .f32, &.{shape.n_hc});
    validateTensor(weights, final.hc_head_fn, "hc_head_fn", .plain, &.{ @as(u64, shape.n_embd) * shape.n_hc, shape.n_hc });
    validateTensor(weights, final.hc_head_scale, "hc_head_scale", .f32, &.{1});
    validateTensor(weights, final.markov_w1, "markov_w1", .dense, &.{ weights.markov_rank, shape.n_vocab });
    validateTensor(weights, final.markov_w2, "markov_w2", .dense, &.{ weights.markov_rank, shape.n_vocab });
    validateTensor(weights, final.confidence_proj, "confidence_proj", .dense, &.{ @as(u64, shape.n_embd) + weights.markov_rank, 1 });
}
fn stageTensor(model: *const Model, stage: u32, suffix: []const u8) ?*Tensor {
    var storage: [160]u8 = undefined;
    const name = std.mem.printSentinel(&storage, "mtp.{d}.{s}", .{ stage, suffix }, 0) catch c.ds4_engine_model_die("tensor name is too long");
    return find(model, name);
}
fn bindTensor(draft: *Draft, model: *const Model, stage: u32, suffix: []const u8, must_exist: bool) [*c]Tensor {
    var tensor = stageTensor(model, stage, suffix);
    // Some V4.1 publishers retain upstream Markov spellings. Prefer canonical
    // converter names if both exist; aliases retain every strict layout check.
    if (tensor == null and draft.v41) {
        if (std.mem.eql(u8, suffix, "markov_head.markov_w1.weight")) tensor = stageTensor(model, stage, "markov_head.embed.weight");
        if (std.mem.eql(u8, suffix, "markov_head.markov_w2.weight")) tensor = stageTensor(model, stage, "markov_head.head.weight");
    }
    if (tensor != null) draft.present_tensors +%= 1 else if (must_exist) draft.missing_tensors +%= 1;
    return tensor;
}
// Shared bind order; bias is the only suffix that is not <field>.weight.
const block_fields = .{
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm",
    "attn_q_a",
    "attn_q_a_norm",
    "attn_q_b",
    "attn_kv",
    "attn_kv_a_norm",
    "attn_sinks",
    "attn_output_a",
    "attn_output_b",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "ffn_norm",
    "ffn_gate_inp",
    "ffn_exp_probs_b",
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
    "ffn_gate_shexp",
    "ffn_up_shexp",
    "ffn_down_shexp",
};
pub fn bind(draft: *Draft, model: ?*const Model, summary: ?*const Summary, shape: *const Shape) void {
    draft.* = std.mem.zeroes(Draft);
    const source = model orelse return;
    const metadata = summary orelse return;
    draft.n_stages = @min(metadata.stages, @as(u32, c.DS4_DSPARK_MAX_STAGES));
    inline for (.{ "block_size", "markov_rank", "noise_token_id", "target_layer_count", "has_block_size", "has_markov_rank", "has_noise_token_id", "has_target_layers" }) |field| @field(draft, field) = @field(metadata, field);
    draft.v41 = isString(source, "general.architecture", "deepseek41-dspark");
    draft.n_experts = shape.n_expert;
    draft.n_experts_used = shape.n_expert_used;
    if (draft.v41) {
        if (!c.ds4_gguf_get_u32(source, "dspark.n_routed_experts", &draft.n_experts) or
            !c.ds4_gguf_get_u32(source, "dspark.num_experts_per_tok", &draft.n_experts_used) or
            draft.n_experts != 128 or draft.n_experts_used != 3 or draft.n_stages != 3 or
            draft.block_size != 5 or draft.markov_rank != 256 or draft.target_layer_count != 3)
            metadataError(draft, "unsupported V4.1 drafter dimensions");
    }
    @memcpy(draft.target_layers[0..draft.target_layer_count], metadata.target_layers[0..draft.target_layer_count]);
    if (draft.v41 and (draft.target_layers[0] != 37 or draft.target_layers[1] != 38 or draft.target_layers[2] != 39)) metadataError(draft, "unsupported V4.1 target layers");
    if (metadata.stages > c.DS4_DSPARK_MAX_STAGES) draft.missing_tensors +%= 1;
    for (draft.stage[0..draft.n_stages], 0..) |*stage, index| {
        inline for (block_fields) |field| {
            const suffix = if (comptime std.mem.eql(u8, field, "ffn_exp_probs_b")) "exp_probs_b.bias" else field ++ ".weight";
            @field(stage.block, field) = bindTensor(draft, source, @intCast(index), suffix, true);
        }
        if (index == 0) {
            stage.main_proj = bindTensor(draft, source, 0, "main_proj.weight", true);
            stage.main_norm = bindTensor(draft, source, 0, "main_norm.weight", true);
        }
    }
    if (draft.n_stages != 0) {
        const index = draft.n_stages - 1;
        const final = &draft.stage[index];
        final.norm = bindTensor(draft, source, index, "norm.weight", true);
        final.hc_head_base = bindTensor(draft, source, index, "hc_head_base.weight", !draft.v41);
        final.hc_head_fn = bindTensor(draft, source, index, "hc_head_fn.weight", !draft.v41);
        final.hc_head_scale = bindTensor(draft, source, index, "hc_head_scale.weight", !draft.v41);
        final.markov_w1 = bindTensor(draft, source, index, "markov_head.markov_w1.weight", true);
        final.markov_w2 = bindTensor(draft, source, index, "markov_head.markov_w2.weight", true);
        final.confidence_proj = bindTensor(draft, source, index, "confidence_head.proj.weight", true);
    }
    validate(draft, shape);
}

fn required(model: *const Model, suffix: []const u8) *Tensor {
    if (stageTensor(model, 0, suffix)) |tensor| return tensor;
    _ = c.fprintf(c.__stderrp, "ds4: required tensor is missing: mtp.0.%.*s\n", @as(c_int, @intCast(suffix.len)), suffix.ptr);
    c.exit(1);
}
fn strict(pointer: [*c]const Tensor, kind: Kind, exact: ?u32, shape: []const u64) void {
    if (pointer == null) c.ds4_engine_model_die(if (kind == .routed) "internal error: missing routed expert tensor while validating layout" else "internal error: missing tensor while validating layout");
    const tensor: *const Tensor = @ptrCast(pointer);
    if (if (exact) |tag| tensor.type != tag else !kind.accepts(tensor.type)) {
        if (kind == .routed) _ = c.fprintf(c.__stderrp, "ds4: tensor %.*s has type %u (%s), expected a routed expert quant type\n", @as(c_int, @intCast(tensor.name.len)), tensor.name.ptr, tensor.type, typeName(tensor.type)) else _ = c.fprintf(c.__stderrp, "ds4: tensor %.*s has type %s, expected %s\n", @as(c_int, @intCast(tensor.name.len)), tensor.name.ptr, typeName(tensor.type), if (exact) |tag| typeName(tag) else kind.name());
        c.exit(1);
    }
    if (tensor.ndim != shape.len) {
        _ = c.fprintf(c.__stderrp, "ds4: tensor %.*s has %u dimensions, expected %u\n", @as(c_int, @intCast(tensor.name.len)), tensor.name.ptr, tensor.ndim, @as(c_uint, @intCast(shape.len)));
        c.exit(1);
    }
    for (shape, 0..) |dimension, index| if (tensor.dim[index] != dimension) {
        _ = c.fprintf(c.__stderrp, "ds4: tensor %.*s has dim[%u]=%llu, expected %llu\n", @as(c_int, @intCast(tensor.name.len)), tensor.name.ptr, @as(c_uint, @intCast(index)), @as(c_ulonglong, tensor.dim[index]), @as(c_ulonglong, dimension));
        c.exit(1);
    };
}
pub fn bindLegacy(weights: *c.ds4_mtp_weights, model: *const Model, shape: *const Shape) void {
    weights.* = std.mem.zeroes(c.ds4_mtp_weights);
    // Binding order is observable on missing tensors and remains independent
    // of validation order. No Model bytes or tensor descriptors are allocated.
    inline for (.{ "hc_head_base", "hc_head_fn", "hc_head_scale", "e_proj", "h_proj", "enorm", "hnorm", "norm" }) |field| @field(weights, field) = required(model, field ++ ".weight");
    inline for (block_fields) |field| {
        const suffix = if (comptime std.mem.eql(u8, field, "ffn_exp_probs_b")) "exp_probs_b.bias" else field ++ ".weight";
        @field(weights.block, field) = required(model, suffix);
    }
    const embedding: u64 = shape.n_embd;
    const hidden = embedding * shape.n_hc;
    const mix: u64 = 2 * @as(u64, shape.n_hc) + @as(u64, shape.n_hc) * shape.n_hc;
    strict(weights.hc_head_base, .f32, c.DS4_TENSOR_F32, &.{shape.n_hc});
    strict(weights.hc_head_fn, .plain, null, &.{ hidden, shape.n_hc });
    strict(weights.hc_head_scale, .f32, c.DS4_TENSOR_F32, &.{1});
    inline for (.{ "e_proj", "h_proj" }) |field| strict(@field(weights, field), .dense, c.DS4_TENSOR_Q8_0, &.{ embedding, embedding });
    inline for (.{ "enorm", "hnorm", "norm" }) |field| strict(@field(weights, field), .f32, c.DS4_TENSOR_F32, &.{embedding});
    const layer = &weights.block;
    strict(layer.hc_attn_fn, .plain, null, &.{ hidden, mix });
    strict(layer.hc_attn_scale, .f32, c.DS4_TENSOR_F32, &.{3});
    strict(layer.hc_attn_base, .f32, c.DS4_TENSOR_F32, &.{mix});
    strict(layer.attn_norm, .f32, c.DS4_TENSOR_F32, &.{embedding});
    strict(layer.attn_q_a, .dense, c.DS4_TENSOR_Q8_0, &.{ embedding, shape.n_lora_q });
    strict(layer.attn_q_a_norm, .f32, c.DS4_TENSOR_F32, &.{shape.n_lora_q});
    strict(layer.attn_q_b, .dense, c.DS4_TENSOR_Q8_0, &.{ shape.n_lora_q, @as(u64, shape.n_head) * shape.n_head_dim });
    strict(layer.attn_kv, .dense, c.DS4_TENSOR_Q8_0, &.{ embedding, shape.n_head_dim });
    strict(layer.attn_kv_a_norm, .f32, c.DS4_TENSOR_F32, &.{shape.n_head_dim});
    strict(layer.attn_sinks, .f32, c.DS4_TENSOR_F32, &.{shape.n_head});
    const output_low = @as(u64, shape.n_out_group) * shape.n_lora_o;
    strict(layer.attn_output_a, .dense, c.DS4_TENSOR_Q8_0, &.{ @as(u64, shape.n_head_dim) * (shape.n_head / shape.n_out_group), output_low });
    strict(layer.attn_output_b, .dense, c.DS4_TENSOR_Q8_0, &.{ output_low, embedding });
    strict(layer.hc_ffn_fn, .plain, null, &.{ hidden, mix });
    strict(layer.hc_ffn_scale, .f32, c.DS4_TENSOR_F32, &.{3});
    strict(layer.hc_ffn_base, .f32, c.DS4_TENSOR_F32, &.{mix});
    strict(layer.ffn_norm, .f32, c.DS4_TENSOR_F32, &.{embedding});
    strict(layer.ffn_gate_inp, .plain, null, &.{ embedding, shape.n_expert });
    strict(layer.ffn_exp_probs_b, .f32, c.DS4_TENSOR_F32, &.{shape.n_expert});
    strict(layer.ffn_gate_exps, .routed, null, &.{ embedding, shape.n_ff_exp, shape.n_expert });
    strict(layer.ffn_up_exps, .routed, null, &.{ embedding, shape.n_ff_exp, shape.n_expert });
    strict(layer.ffn_down_exps, .routed, null, &.{ shape.n_ff_exp, embedding, shape.n_expert });
    if (layer.ffn_gate_exps.*.type != layer.ffn_up_exps.*.type) c.ds4_engine_model_die("MTP routed gate/up experts use different quant types");
    strict(layer.ffn_gate_shexp, .dense, c.DS4_TENSOR_Q8_0, &.{ embedding, shape.n_ff_exp });
    strict(layer.ffn_up_shexp, .dense, c.DS4_TENSOR_Q8_0, &.{ embedding, shape.n_ff_exp });
    strict(layer.ffn_down_shexp, .dense, c.DS4_TENSOR_Q8_0, &.{ shape.n_ff_exp, embedding });
}
