//! GLM 5.3 Flash's trained KDA/DSA profile and strict layer-type validation.
//! The actual graph configuration travels by value; GGUF bytes remain borrowed.
const std = @import("std");
const c = @import("gguf_c");
const policy = @import("config.zig");
const Config = c.ds4_glm_model_config;
const InvalidModel = error{InvalidModel};

fn profile() c.ds4_shape {
    var s = std.mem.zeroes(c.ds4_shape);
    s.name = "GLM 5.3 Flash";
    s.family = c.DS4_MODEL_FAMILY_GLM_DSA;
    s.variant = c.DS4_VARIANT_GLM53;
    s.n_layer = 46;
    s.n_embd = 4096;
    s.n_vocab = 154880;
    s.n_head = 64;
    s.n_head_kv = 1;
    // A raw DSA row has 512 latent values and no rotary tail.
    s.n_head_dim = 512;
    s.n_value_dim = 256;
    s.n_rot = 0;
    s.n_lora_q = 1536;
    s.n_expert = 288;
    s.n_expert_used = 8;
    s.n_expert_shared = 1;
    s.n_ff_exp = 2048;
    s.n_ff_dense = 12288;
    s.n_indexer_head = 32;
    s.n_indexer_head_dim = 128;
    s.n_indexer_top_k = 2048;
    s.n_hc = 4;
    s.n_hc_sinkhorn_iter = 20;
    s.n_nextn_predict = 1;
    s.n_leading_dense = 3;
    s.n_kv_lora = 512;
    s.n_key_mla = 256;
    s.n_value_mla = 256;
    s.n_kda_head = 64;
    s.n_kda_head_dim = 128;
    s.n_kda_conv = 4;
    s.rms_eps = 1.0e-5;
    s.hc_eps = 1.0e-6;
    s.expert_weight_scale = 2.5;
    s.swiglu_clamp_exp = 10.0;
    s.rope_orig_ctx = 1048576;
    s.kda_gate_lower_bound = -5.0;
    return s;
}

const Reader = struct {
    model: *const c.ds4_model,
    errors: []u8,

    fn fail(self: *const Reader, comptime format: []const u8, args: anytype) InvalidModel {
        if (self.errors.len != 0) {
            _ = std.mem.printSentinel(self.errors, format, args, 0) catch {
                self.errors[self.errors.len - 1] = 0;
                return error.InvalidModel;
            };
        }
        return error.InvalidModel;
    }

    fn integer(self: *const Reader, comptime suffix: [:0]const u8, expected: u32) InvalidModel!void {
        const key = "glm5-next." ++ suffix;
        var got: u32 = undefined;
        if (!c.ds4_gguf_get_u32(self.model, key, &got)) return self.fail("required metadata key is missing: {s}", .{key});
        if (got != expected) return self.fail("expected {s}={d} for GLM 5.3 Flash, got {d}", .{ suffix, expected, got });
    }

    fn float(self: *const Reader, comptime suffix: [:0]const u8, expected: f32) InvalidModel!void {
        @setFloatMode(.strict);
        const key = "glm5-next." ++ suffix;
        var got: f32 = undefined;
        if (!c.ds4_gguf_get_f32_compat(self.model, key, &got)) return self.fail("required metadata key is missing: {s}", .{key});
        const tolerance = @max(@abs(expected), 1.0) * @as(f32, 1.0e-6);
        if (!(@abs(got - expected) <= tolerance)) return self.fail("expected {s}={d} for GLM 5.3 Flash, got {d}", .{ suffix, expected, got });
    }

    fn layerTypes(self: *const Reader, config: *const Config) InvalidModel!void {
        var array: c.ds4_array_ref = undefined;
        if (!c.ds4_gguf_get_array(self.model, "glm5-next.layer_types", &array) or
            (array.type != c.GGUF_VALUE_UINT32 and array.type != c.GGUF_VALUE_INT32) or array.len != config.shape.n_layer)
            return self.fail("glm5-next.layer_types must be an int32/uint32 array with one entry per layer", .{});
        var cursor = c.ds4_cursor{ .base = self.model.map, .size = self.model.size, .pos = array.data_pos, .@"error" = @splat(0) };
        for (0..config.shape.n_layer) |layer| {
            var got: u32 = undefined;
            if (array.type == c.GGUF_VALUE_UINT32) {
                if (!c.ds4_gguf_u32(&cursor, &got)) return self.fail("{s}", .{std.mem.sliceTo(&cursor.@"error", 0)});
            } else {
                var signed: i32 = undefined;
                if (!c.ds4_gguf_read(&cursor, &signed, @sizeOf(i32))) return self.fail("{s}", .{std.mem.sliceTo(&cursor.@"error", 0)});
                if (signed < 0) return self.fail("glm5-next.layer_types contains a negative value", .{});
                got = @intCast(signed);
            }
            const expected: u32 = if (policy.isKda(config, @intCast(layer))) 0 else 1;
            if (got != expected) return self.fail("unexpected GLM 5.3 attention type at layer {d}: got {d}, expected {d}", .{ layer, got, expected });
        }
    }
};

pub fn load(model: *const c.ds4_model, errors: []u8) InvalidModel!Config {
    var result: Config = .{ .shape = profile(), .prefill_chunk = 2048, .index_pool_size = 4, .reference_hc = false, .fused_bf16_qkv = true, .stream_selected_async = false, .flash_prefill = true, .decode_flush_interval = -1 };
    const reader = Reader{ .model = model, .errors = errors };
    const s = &result.shape;
    inline for (.{ .{ "block_count", s.n_layer }, .{ "trunk_block_count", s.n_layer - s.n_nextn_predict }, .{ "nextn_predict_layers", s.n_nextn_predict } }) |field| try reader.integer(field[0], field[1]);
    var context: u64 = undefined;
    if (!c.ds4_gguf_get_u64_compat(model, "glm5-next.context_length", &context)) return reader.fail("required metadata key is missing: glm5-next.context_length", .{});
    if (context != s.rope_orig_ctx) return reader.fail("expected context_length={d} for GLM 5.3 Flash, got {d}", .{ s.rope_orig_ctx, context });
    inline for (.{ .{ "embedding_length", s.n_embd }, .{ "vocab_size", s.n_vocab }, .{ "feed_forward_length", s.n_ff_dense }, .{ "expert_feed_forward_length", s.n_ff_exp }, .{ "expert_count", s.n_expert }, .{ "expert_used_count", s.n_expert_used }, .{ "expert_shared_count", s.n_expert_shared }, .{ "leading_dense_block_count", s.n_leading_dense } }) |field| try reader.integer(field[0], field[1]);
    try reader.float("expert_weights_scale", s.expert_weight_scale);
    var normalized: bool = undefined;
    if (!c.ds4_gguf_get_bool(model, "glm5-next.expert_weights_norm", &normalized)) return reader.fail("required metadata key is missing: glm5-next.expert_weights_norm", .{});
    if (!normalized) return reader.fail("expected expert_weights_norm=true for GLM 5.3 Flash, got false", .{});
    try reader.float("swiglu_limit", s.swiglu_clamp_exp);
    try reader.float("attention.layer_norm_rms_epsilon", s.rms_eps);
    inline for (.{
        .{ "attention.head_count", s.n_head },                 .{ "attention.key_length", s.n_key_mla },                  .{ "attention.value_length", s.n_value_mla },
        .{ "attention.q_lora_rank", s.n_lora_q },              .{ "attention.kv_lora_rank", s.n_kv_lora },                .{ "attention.rope_dimension_count", s.n_rot },
        .{ "attention.indexer.head_count", s.n_indexer_head }, .{ "attention.indexer.key_length", s.n_indexer_head_dim }, .{ "attention.indexer.top_k", s.n_indexer_top_k },
        .{ "attention.indexer.pool_size", @as(u32, 4) },       .{ "linear_attention.head_count", s.n_kda_head },          .{ "linear_attention.head_dimension", s.n_kda_head_dim },
        .{ "linear_attention.conv_kernel", s.n_kda_conv },
    }) |field| try reader.integer(field[0], field[1]);
    try reader.float("linear_attention.gate_lower_bound", s.kda_gate_lower_bound);
    try reader.integer("hyper_connection.count", s.n_hc);
    try reader.integer("hyper_connection.sinkhorn_iterations", s.n_hc_sinkhorn_iter);
    try reader.float("hyper_connection.epsilon", s.hc_eps);
    try reader.layerTypes(&result);

    // Preserve the current per-graph feature switches while capturing them once
    // with the validated model, rather than making hot paths read globals.
    const reference = c.getenv("DS4_METAL_DISABLE_HC_FUSION");
    result.reference_hc = reference != null and reference[0] != 0 and !std.mem.eql(u8, std.mem.span(reference), "0");
    result.fused_bf16_qkv = c.getenv("DS4_METAL_DISABLE_M3_ULTRA_GLM53_DECODE") == null and c.getenv("DS4_METAL_DISABLE_GLM53_BF16_QKV") == null;
    result.stream_selected_async = c.getenv("DS4_METAL_ENABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD") != null and
        c.getenv("DS4_METAL_DISABLE_STREAMING_SELECTED_ASYNC_LOAD") == null and
        c.getenv("DS4_METAL_DISABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD") == null;
    result.flash_prefill = c.getenv("DS4_GLM_DISABLE_FLASH_PREFILL") == null;
    const flush = c.getenv("DS4_GLM_DECODE_FLUSH_INTERVAL");
    if (flush != null and flush[0] != 0) result.decode_flush_interval = @max(0, c.atoi(flush));
    return result;
}
