//! Qwen3.8 Flash Next metadata, including the synthetic mini profile and optional
//! MTP block. The returned value owns dimensions, PLE hashes and rotary values;
//! nothing here consults or publishes the C engine's active-model globals.
const std = @import("std");
const c = @import("gguf_c");
const policy = @import("config.zig");
const Config = c.ds4_qwen4_config;
const Shape = c.ds4_shape;
const InvalidModel = error{InvalidModel};

pub fn profile(comptime mini: bool) Shape {
    var s = std.mem.zeroes(Shape);
    s.name = if (mini) "Qwen3.8 Flash Next mini" else "Qwen3.8 Flash Next";
    s.family = c.DS4_MODEL_FAMILY_QWEN4_EXP;
    s.variant = if (mini) c.DS4_VARIANT_QWEN4_MINI else c.DS4_VARIANT_QWEN4_EXP;
    s.n_layer = if (mini) 9 else 49;
    s.n_embd = if (mini) 64 else 2560;
    s.n_vocab = 248320;
    s.n_head = if (mini) 4 else 24;
    s.n_head_kv = 2;
    s.n_head_dim = if (mini) 32 else 256;
    s.n_value_dim = s.n_head_dim;
    s.n_rot = if (mini) 8 else 64;
    s.n_expert = if (mini) 32 else 512;
    s.n_expert_used = 10;
    s.n_expert_shared = 1;
    s.n_ff_exp = if (mini) 32 else 640;
    s.n_indexer_head = 4;
    s.n_indexer_head_dim = if (mini) 32 else 128;
    s.n_indexer_top_k = if (mini) 8 else 2048;
    s.n_hc = 4;
    s.n_hc_lowrank = if (mini) 8 else 320;
    s.n_nextn_predict = 1;
    s.n_lin_k_head = if (mini) 2 else 16;
    s.n_lin_v_head = if (mini) 6 else 48;
    s.n_lin_head_dim = if (mini) 32 else 128;
    s.n_lin_conv = 4;
    s.n_full_attn_interval = 4;
    s.n_ple_layer = 1;
    s.n_ple_ngram = 3;
    s.n_ple_heads_per_ngram = 8;
    s.n_ple_conv = 4;
    s.n_ple_head_dim = if (mini) 4 else 160;
    s.ple_eos_id = 248044;
    s.rms_eps = 1.0e-6;
    s.rope_freq_base = 10000000.0;
    s.rope_orig_ctx = 262144;
    return s;
}

const Reader = struct {
    model: *const c.ds4_model,
    errors: []u8,
    name: []const u8 = "Qwen3.8 Flash Next",

    fn fail(self: *const Reader, comptime format: []const u8, args: anytype) InvalidModel {
        if (self.errors.len != 0) {
            _ = std.mem.printSentinel(self.errors, format, args, 0) catch {
                self.errors[self.errors.len - 1] = 0;
                return error.InvalidModel;
            };
        }
        return error.InvalidModel;
    }

    fn required(self: *const Reader, comptime suffix: [:0]const u8) InvalidModel!u32 {
        const key = "qwen4exp." ++ suffix;
        var value: u32 = undefined;
        if (!c.ds4_gguf_get_u32(self.model, key, &value))
            return self.fail("required metadata key is missing: {s}", .{key});
        return value;
    }

    fn integer(self: *const Reader, comptime suffix: [:0]const u8, expected: u32) InvalidModel!void {
        const got = try self.required(suffix);
        if (got != expected)
            return self.fail("expected {s}={d} for {s}, got {d}", .{ suffix, expected, self.name, got });
    }

    fn float(self: *const Reader, comptime suffix: [:0]const u8, expected: f32, epsilon: bool) InvalidModel!void {
        @setFloatMode(.strict);
        const key = "qwen4exp." ++ suffix;
        var got: f32 = undefined;
        if (!c.ds4_gguf_get_f32_compat(self.model, key, &got))
            return self.fail("required metadata key is missing: {s}", .{key});
        const tolerance = if (epsilon) @max(@abs(expected) * @as(f32, 1.0e-5), @as(f32, 1.0e-30)) else @max(@abs(expected), 1.0) * @as(f32, 1.0e-6);
        if (!(@abs(got - expected) <= tolerance))
            return self.fail("expected {s}={d} for {s}, got {d}", .{ suffix, expected, self.name, got });
    }

    fn array(self: *const Reader, comptime suffix: [:0]const u8, output: []u64) InvalidModel![]u64 {
        const key = "qwen4exp." ++ suffix;
        var ref: c.ds4_array_ref = undefined;
        if (!c.ds4_gguf_get_array(self.model, key, &ref))
            return self.fail("required array metadata key is missing: {s}", .{key});
        if (ref.len > output.len)
            return self.fail("metadata array {s} has {d} entries, at most {d} supported", .{ key, ref.len, output.len });
        var cursor = c.ds4_cursor{ .base = self.model.map, .size = self.model.size, .pos = ref.data_pos, .@"error" = @splat(0) };
        for (output[0..@intCast(ref.len)]) |*value| {
            switch (ref.type) {
                c.GGUF_VALUE_UINT64, c.GGUF_VALUE_INT64 => {
                    if (!c.ds4_gguf_u64(&cursor, value)) return self.fail("{s}", .{std.mem.sliceTo(&cursor.@"error", 0)});
                },
                c.GGUF_VALUE_UINT32, c.GGUF_VALUE_INT32 => {
                    var narrow: u32 = undefined;
                    if (!c.ds4_gguf_u32(&cursor, &narrow)) return self.fail("{s}", .{std.mem.sliceTo(&cursor.@"error", 0)});
                    value.* = narrow;
                },
                else => return self.fail("metadata array {s} has an unsupported element type {d}", .{ key, ref.type }),
            }
        }
        return output[0..@intCast(ref.len)];
    }
};

/// Preserve libc's double-precision YaRN construction and float publication.
/// A non-default factor changes model arithmetic and stays explicitly opt-in.
fn rotary(config: *Config, native_context: u32) void {
    @setFloatMode(.strict);
    const dimensions: f64 = @floatFromInt(config.shape.n_rot);
    const half = config.shape.n_rot / 2;
    const base: f64 = config.shape.rope_freq_base;
    const environment = c.getenv("DS4_QWEN4_YARN_FACTOR");
    const factor = if (environment == null) 0 else c.atof(environment);
    const yarn = factor > 1 and native_context > 0;
    var low: f64 = 0;
    var high: f64 = 0;
    if (yarn) {
        const context: f64 = @floatFromInt(native_context);
        low = c.fmax(0, c.floor(dimensions * c.log(context / (32.0 * 2.0 * std.math.pi)) / (2.0 * c.log(base))));
        high = c.fmin(dimensions - 1, c.ceil(dimensions * c.log(context / (2.0 * std.math.pi)) / (2.0 * c.log(base))));
        if (high == low) high += 0.001;
    }
    for (&config.rope_freq, 0..) |*frequency, i| {
        const index: f64 = @floatFromInt(i);
        var f = if (i < half) c.pow(base, -2.0 * index / dimensions) else 0;
        if (yarn and i < half) {
            const extrapolation = 1.0 - c.fmin(1.0, c.fmax(0.0, (index - low) / (high - low)));
            f = (f / factor) * (1.0 - extrapolation) + f * extrapolation;
        }
        frequency.* = @floatCast(f);
    }
    config.rope_mscale = if (yarn) @floatCast(0.1 * c.log(factor) + 1.0) else 1;
    config.native_ctx = native_context;
    config.rope_yarn = yarn;
    if (yarn) _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 YaRN factor %g over %u native tokens (pairs %g..%g blended, mscale %.4f)\n", factor, native_context, low, high, @as(f64, config.rope_mscale));
}

pub fn load(model: *const c.ds4_model, errors: []u8) InvalidModel!Config {
    var reader = Reader{ .model = model, .errors = errors };
    const width = try reader.required("embedding_length");
    var result = std.mem.zeroes(Config);
    result.shape = switch (width) {
        2560 => profile(false),
        64 => profile(true),
        else => return reader.fail("unsupported qwen4exp embedding_length {d}", .{width}),
    };
    reader.name = std.mem.span(result.shape.name);
    const shape = &result.shape;
    // Upstream exports without MTP contain the trunk only. Missing metadata
    // therefore means zero, rather than the compiled profile's one draft block.
    var nextn: u32 = 0;
    _ = c.ds4_gguf_get_u32(model, "qwen4exp.nextn_predict_layers", &nextn);
    if (nextn > shape.n_nextn_predict)
        return reader.fail("qwen4exp.nextn_predict_layers {d} exceeds {d}", .{ nextn, shape.n_nextn_predict });
    shape.n_layer -= shape.n_nextn_predict - nextn;
    shape.n_nextn_predict = nextn;

    inline for (.{
        .{ "block_count", shape.n_layer },             .{ "attention.head_count", shape.n_head },        .{ "attention.head_count_kv", shape.n_head_kv },
        .{ "attention.key_length", shape.n_head_dim }, .{ "attention.value_length", shape.n_value_dim }, .{ "rope.dimension_count", shape.n_rot },
    }) |field| try reader.integer(field[0], field[1]);
    try reader.float("rope.freq_base", shape.rope_freq_base, false);
    rotary(&result, try reader.required("context_length"));
    try reader.float("attention.layer_norm_rms_epsilon", shape.rms_eps, true);
    inline for (.{
        .{ "expert_count", shape.n_expert },                              .{ "expert_used_count", shape.n_expert_used },
        .{ "expert_feed_forward_length", shape.n_ff_exp },                .{ "expert_shared_feed_forward_length", shape.n_ff_exp },
        .{ "ssm.conv_kernel", shape.n_lin_conv },                         .{ "ssm.state_size", shape.n_lin_head_dim },
        .{ "ssm.group_count", shape.n_lin_k_head },                       .{ "ssm.time_step_rank", shape.n_lin_v_head },
        .{ "ssm.inner_size", shape.n_lin_v_head * shape.n_lin_head_dim }, .{ "full_attention_interval", shape.n_full_attn_interval },
        .{ "hyper_connection.count", shape.n_hc },                        .{ "hyper_connection.low_rank", shape.n_hc_lowrank },
        .{ "attention.indexer.head_count", shape.n_indexer_head },        .{ "attention.indexer.key_length", shape.n_indexer_head_dim },
        .{ "attention.indexer.top_k", shape.n_indexer_top_k },
    }) |field| try reader.integer(field[0], field[1]);

    var ratios: [c.DS4_MAX_LAYER]u64 = undefined;
    const read_ratios = try reader.array("attention.compress_ratios", &ratios);
    if (read_ratios.len < shape.n_layer) return reader.fail("qwen4exp.attention.compress_ratios is shorter than the layer count", .{});
    for (read_ratios[0..shape.n_layer], 0..) |ratio, layer| {
        const expected: u32 = if (policy.isLinear(&result, @intCast(layer))) 0 else 4;
        if (ratio != expected)
            return reader.fail("unexpected qwen4exp compress ratio at layer {d}: got {d}, expected {d}", .{ layer, ratio, expected });
        result.compress_ratios[layer] = @intCast(ratio);
    }

    var values: [c.DS4_MAX_PLE_HEADS]u64 = undefined;
    const layers = try reader.array("ple.layers", &values);
    if (layers.len != 1 or layers[0] != shape.n_ple_layer) return reader.fail("qwen4exp.ple.layers must be [{d}]", .{shape.n_ple_layer});
    inline for (.{ .{ "ple.ngram_size", shape.n_ple_ngram }, .{ "ple.heads_per_ngram", shape.n_ple_heads_per_ngram }, .{ "ple.conv_kernel", shape.n_ple_conv }, .{ "ple.eos_token_id", @as(u32, @intCast(shape.ple_eos_id)) }, .{ "embedding_length_per_layer_input", shape.n_ple_head_dim } }) |field|
        try reader.integer(field[0], field[1]);
    const heads = (shape.n_ple_ngram - 1) * shape.n_ple_heads_per_ngram;
    if (heads > c.DS4_MAX_PLE_HEADS or shape.n_ple_ngram > c.DS4_MAX_PLE_NGRAM)
        return reader.fail("PLE head or n-gram count exceeds the compiled limit", .{});
    const multipliers = try reader.array("ple.layer_multipliers", &values);
    if (multipliers.len != shape.n_ple_ngram) return reader.fail("qwen4exp.ple.layer_multipliers has the wrong length", .{});
    @memcpy(result.ple.multiplier[0..multipliers.len], multipliers);
    const offsets = try reader.array("ple.head_offsets", &values);
    if (offsets.len != heads) return reader.fail("qwen4exp.ple.head_offsets has the wrong length", .{});
    for (offsets, 0..) |offset, i| {
        if (offset > std.math.maxInt(u32)) return reader.fail("PLE head offset does not fit 32 bits", .{});
        result.ple.head_offset[i] = @intCast(offset);
    }
    const vocabularies = try reader.array("ple.head_vocab_sizes", &values);
    if (vocabularies.len != heads) return reader.fail("qwen4exp.ple.head_vocab_sizes has the wrong length", .{});
    for (vocabularies, 0..) |vocabulary, i| {
        if (vocabulary == 0 or vocabulary > std.math.maxInt(u32)) return reader.fail("PLE head vocab size is out of range", .{});
        result.ple.head_vocab[i] = @intCast(vocabulary);
        result.ple.n_rows = @max(result.ple.n_rows, @as(u64, result.ple.head_offset[i]) + vocabulary);
    }
    result.ple.n_heads = heads;
    return result;
}
