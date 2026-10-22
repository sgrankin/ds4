//! Validated, per-model configuration. This first constructor owns DeepSeek
//! V4.1 metadata/profile selection; the other available families retain their
//! C validators while the engine moves to explicit configuration values.
//! No model mapping, engine/session allocation or process globals live here.
const std = @import("std");
const c = @import("gguf_c");
const Shape = c.ds4_shape;
const Model = c.ds4_model;
const Tensor = c.ds4_tensor;
const max_layers = c.DS4_MAX_LAYER;
const InvalidModel = error{InvalidModel};

// Actual trained profile, not workspace maxima. The 1e-20 RMS epsilon is
// intentional: replacing tiny literals with zero changes model semantics.
const deepseek41_shape = profile: {
    var shape = std.mem.zeroes(Shape);
    shape.name = "DeepSeek V4.1 Flash";
    shape.family = c.DS4_MODEL_FAMILY_DEEPSEEK41;
    shape.variant = c.DS4_VARIANT_FLASH41;
    shape.n_layer = 40;
    shape.n_embd = 5120;
    shape.n_vocab = 129280;
    shape.n_head = 64;
    shape.n_head_kv = 1;
    shape.n_head_dim = 512;
    shape.n_value_dim = 512;
    shape.n_rot = 64;
    shape.n_out_group = 8;
    shape.n_lora_q = 1280;
    shape.n_lora_o = 1024;
    shape.n_expert = 384;
    shape.n_expert_used = 6;
    shape.n_expert_shared = 1;
    shape.n_ff_exp = 2304;
    shape.n_swa = 128;
    shape.n_indexer_head = 32;
    shape.n_indexer_head_dim = 128;
    shape.n_indexer_top_k = 512;
    shape.n_hc = 4;
    shape.n_hc_sinkhorn_iter = 20;
    shape.rms_eps = 1.0e-20;
    shape.hc_eps = 1.0e-6;
    shape.expert_weight_scale = 1.5;
    shape.swiglu_clamp_exp = 10.0;
    shape.rope_freq_base = 10000.0;
    shape.rope_scale_factor = 16.0;
    shape.rope_yarn_beta_fast = 32.0;
    shape.rope_yarn_beta_slow = 1.0;
    shape.compress_rope_freq_base = 160000.0;
    shape.rope_orig_ctx = 65536;
    break :profile shape;
};

// Source layers publish shared cache rows; other layers consume those rows.
// Engram is a separate disk-row lookup at the two trained layer positions.
const kv_sources = [_]u32{ 2, 8, 14, 20 };
const index_sources = [_]u32{ 2, 8, 14, 20, 24, 28, 32, 36 };
const engram_layers = [_]u32{ 1, 14 };
const engram_rows = [_]u32{ 384006168, 384016682 };

pub const LayerPolicy = struct {
    compression_ratio: u32,
    publishes_kv: bool,
    publishes_index: bool,
    uses_engram: bool,
};

// Owned by value and immutable when held as const by an engine. Shape names
// are static text; neither this value nor its layer policies borrow GGUF bytes.
pub const ModelConfig = struct {
    shape: Shape,
    compress_ratios: [max_layers]u32,

    pub fn layerPolicy(self: *const ModelConfig, layer: u32) ?LayerPolicy {
        if (layer >= self.shape.n_layer) return null;
        return .{
            .compression_ratio = self.compress_ratios[layer],
            .publishes_kv = std.mem.indexOfScalar(u32, &kv_sources, layer) != null,
            .publishes_index = std.mem.indexOfScalar(u32, &index_sources, layer) != null,
            .uses_engram = std.mem.indexOfScalar(u32, &engram_layers, layer) != null,
        };
    }

    pub fn deepseek41(model: *const Model, error_text: []u8) InvalidModel!ModelConfig {
        var config = ModelConfig{
            .shape = deepseek41_shape,
            .compress_ratios = @splat(0),
        };
        // Two raw-only layers, paired compression through the encoder, then
        // one-row publication in the decoder. Unused maximum-capacity slots
        // remain zero; the profile's layer count is the live bound.
        for (config.compress_ratios[0..config.shape.n_layer], 0..) |*ratio, layer| {
            ratio.* = if (layer < 2) 0 else if (layer < 20) 2 else 1;
        }
        var validation = Validation{ .model = model, .error_text = error_text };
        try validation.metadata(&config);
        try validation.engramTensors();
        return config;
    }

    fn compatibilityValue(self: *const ModelConfig) c.ds4_model_config {
        return .{ .shape = self.shape, .compress_ratios = self.compress_ratios };
    }
};

fn stringBytes(text: c.ds4_str) []const u8 {
    return if (text.len == 0) "" else text.ptr[0..@as(usize, @intCast(text.len))];
}

const Validation = struct {
    model: *const Model,
    error_text: []u8,

    fn fail(self: *Validation, comptime format: []const u8, args: anytype) InvalidModel {
        if (self.error_text.len != 0) {
            _ = std.mem.printSentinel(self.error_text, format, args, 0) catch {
                self.error_text[self.error_text.len - 1] = 0;
                return error.InvalidModel;
            };
        }
        return error.InvalidModel;
    }

    fn requiredU32(self: *Validation, key: [:0]const u8) InvalidModel!u32 {
        var value: u32 = undefined;
        if (!c.ds4_gguf_get_u32(self.model, key.ptr, &value))
            return self.fail("required metadata key is missing: {s}", .{key});
        return value;
    }

    fn requiredFloat(self: *Validation, key: [:0]const u8) InvalidModel!f32 {
        var value: f32 = undefined;
        if (!c.ds4_gguf_get_f32_compat(self.model, key.ptr, &value))
            return self.fail("required metadata key is missing: {s}", .{key});
        return value;
    }

    fn expectU32(self: *Validation, key: [:0]const u8, expected: u32) InvalidModel!void {
        const got = try self.requiredU32(key);
        if (got != expected)
            return self.fail("expected {s}={d} for DeepSeek V4.1 Flash, got {d}", .{ key, expected, got });
    }

    fn expectFloat(self: *Validation, key: [:0]const u8, expected: f32) InvalidModel!void {
        @setFloatMode(.strict);
        const got = try self.requiredFloat(key);
        // Match the existing epsilon check, including its tiny absolute floor.
        // A relative tolerance scaled by max(1,expected) would accept zero for
        // the trained 1e-20 RMS epsilon and is not equivalent.
        const tolerance = @max(@abs(expected) * @as(f32, 1.0e-5), @as(f32, 1.0e-30));
        if (@abs(got - expected) <= tolerance) return;
        return self.fail("expected {s}={d} for DeepSeek V4.1 Flash, got {d}", .{ key, expected, got });
    }

    fn expectString(self: *Validation, key: [:0]const u8, expected: []const u8) InvalidModel!void {
        var got: c.ds4_str = undefined;
        if (!c.ds4_gguf_get_string(self.model, key.ptr, &got) or !std.mem.eql(u8, stringBytes(got), expected))
            return self.fail("unsupported {s}", .{key});
    }

    fn expectArray(self: *Validation, key: [:0]const u8, expected: []const u32) InvalidModel!void {
        var array: c.ds4_array_ref = undefined;
        if (!c.ds4_gguf_get_array(self.model, key.ptr, &array) or
            array.type != c.GGUF_VALUE_UINT32 or array.len != expected.len)
            return self.fail("{s} must be a uint32 array of length {d}", .{ key, expected.len });
        var cursor = c.ds4_cursor{
            .base = self.model.map,
            .size = self.model.size,
            .pos = array.data_pos,
            .@"error" = @splat(0),
        };
        for (expected) |wanted| {
            var got: u32 = undefined;
            if (!c.ds4_gguf_u32(&cursor, &got))
                return self.fail("{s}", .{std.mem.sliceTo(&cursor.@"error", 0)});
            if (got != wanted)
                return self.fail("expected {s}={d} for DeepSeek V4.1 Flash, got {d}", .{ key, wanted, got });
        }
    }

    fn metadata(self: *Validation, config: *const ModelConfig) InvalidModel!void {
        const shape = &config.shape;
        const integers = [_]struct { key: [:0]const u8, expected: u32 }{
            .{ .key = "deepseek41.vocab_size", .expected = shape.n_vocab },
            .{ .key = "deepseek41.hidden_size", .expected = shape.n_embd },
            .{ .key = "deepseek41.moe_intermediate_size", .expected = shape.n_ff_exp },
            .{ .key = "deepseek41.num_hidden_layers", .expected = shape.n_layer },
            .{ .key = "deepseek41.num_attention_heads", .expected = shape.n_head },
            .{ .key = "deepseek41.num_key_value_heads", .expected = shape.n_head_kv },
            .{ .key = "deepseek41.head_dim", .expected = shape.n_head_dim },
            .{ .key = "deepseek41.qk_rope_head_dim", .expected = shape.n_rot },
            .{ .key = "deepseek41.q_lora_rank", .expected = shape.n_lora_q },
            .{ .key = "deepseek41.o_lora_rank", .expected = shape.n_lora_o },
            .{ .key = "deepseek41.o_groups", .expected = shape.n_out_group },
            .{ .key = "deepseek41.n_routed_experts", .expected = shape.n_expert },
            .{ .key = "deepseek41.n_shared_experts", .expected = shape.n_expert_shared },
            .{ .key = "deepseek41.num_experts_per_tok", .expected = shape.n_expert_used },
            .{ .key = "deepseek41.max_position_embeddings", .expected = 1048576 },
            .{ .key = "deepseek41.sliding_window", .expected = shape.n_swa },
            .{ .key = "deepseek41.index_n_heads", .expected = shape.n_indexer_head },
            .{ .key = "deepseek41.index_head_dim", .expected = shape.n_indexer_head_dim },
            .{ .key = "deepseek41.index_topk", .expected = shape.n_indexer_top_k },
            .{ .key = "deepseek41.candidate_source_layer_id", .expected = 20 },
            .{ .key = "deepseek41.candidate_topk_blocks", .expected = 2048 },
            .{ .key = "deepseek41.candidate_block_size", .expected = 8 },
            .{ .key = "deepseek41.hc_mult", .expected = shape.n_hc },
            .{ .key = "deepseek41.hc_sinkhorn_iters", .expected = shape.n_hc_sinkhorn_iter },
            .{ .key = "deepseek41.rope_theta", .expected = 10000 },
            .{ .key = "deepseek41.compress_rope_theta", .expected = 160000 },
            .{ .key = "deepseek41.engram.compressed_vocab_size", .expected = 99092 },
        };
        for (integers) |field| try self.expectU32(field.key, field.expected);
        const floats = [_]struct { key: [:0]const u8, expected: f32 }{
            .{ .key = "deepseek41.rms_norm_eps", .expected = shape.rms_eps },
            .{ .key = "deepseek41.hc_eps", .expected = shape.hc_eps },
            .{ .key = "deepseek41.swiglu_limit", .expected = shape.swiglu_clamp_exp },
            .{ .key = "deepseek41.routed_scaling_factor", .expected = shape.expert_weight_scale },
            .{ .key = "deepseek41.rope_scaling.factor", .expected = shape.rope_scale_factor },
            .{ .key = "deepseek41.rope_scaling.beta_fast", .expected = shape.rope_yarn_beta_fast },
            .{ .key = "deepseek41.rope_scaling.beta_slow", .expected = shape.rope_yarn_beta_slow },
            .{ .key = "deepseek41.rope_scaling.original_max_position_embeddings", .expected = @floatFromInt(shape.rope_orig_ctx) },
        };
        for (floats) |field| try self.expectFloat(field.key, field.expected);
        try self.expectString("deepseek41.scoring_func", "sqrtsoftplus");
        try self.expectString("deepseek41.hidden_act", "silu");
        try self.expectString("deepseek41.topk_method", "noaux_tc");
        try self.expectString("deepseek41.engram.encoding", "e4m3_e8m0_32_row264");
        var normalized: bool = undefined;
        if (!c.ds4_gguf_get_bool(self.model, "deepseek41.norm_topk_prob", &normalized))
            return self.fail("required metadata key is missing: deepseek41.norm_topk_prob", .{});
        if (!normalized)
            return self.fail("expected deepseek41.norm_topk_prob=true for DeepSeek V4.1 Flash, got false", .{});
        try self.expectArray("deepseek41.kv_source_layer_ids", &kv_sources);
        try self.expectArray("deepseek41.index_source_layer_ids", &index_sources);
        try self.expectArray("deepseek41.engram.layer_ids", &engram_layers);
        try self.expectArray("deepseek41.engram.rows", &engram_rows);
        try self.expectArray("deepseek41.compress_ratios", config.compress_ratios[0..shape.n_layer]);
    }

    fn engramTensors(self: *Validation) InvalidModel!void {
        const names = [_][]const u8{ "blk.1.engram_embd.weight", "blk.14.engram_embd.weight" };
        for (names, engram_rows) |name, rows| {
            const tensor = self.findTensor(name) orelse
                return self.fail("required tensor is missing: {s}", .{name});
            // GGUF i8 format 24: each disk row is 256 values plus eight scales.
            // Do not pull the table into the main mapped expert address range.
            if (tensor.type != 24)
                return self.fail("tensor {s} must have GGUF i8 type, got {d}", .{ name, tensor.type });
            if (tensor.ndim != 2)
                return self.fail("tensor {s} has {d} dimensions, expected 2", .{ name, tensor.ndim });
            const dimensions = [_]u64{ 264, rows };
            for (dimensions, 0..) |dimension, index| {
                if (tensor.dim[index] != dimension)
                    return self.fail("tensor {s} has dim[{d}]={d}, expected {d}", .{ name, index, tensor.dim[index], dimension });
            }
            if (tensor.abs_offset < self.model.size)
                return self.fail("Engram table is still mapped", .{});
        }
    }

    fn findTensor(self: *const Validation, name: []const u8) ?*const Tensor {
        if (self.model.n_tensors == 0) return null;
        const tensors: [*]const Tensor = @ptrCast(self.model.tensors);
        for (tensors[0..@as(usize, @intCast(self.model.n_tensors))]) |*tensor| {
            if (std.mem.eql(u8, stringBytes(tensor.name), name)) return tensor;
        }
        return null;
    }
};

pub fn readRecord(model: *const Model, output: *c.ds4_model_config, error_pointer: [*c]u8, error_capacity: usize) bool {
    const error_text: []u8 = if (error_capacity == 0) &.{} else error_pointer[0..error_capacity];
    if (error_text.len != 0) error_text[0] = 0;
    const config = ModelConfig.deepseek41(model, error_text) catch return false;
    output.* = config.compatibilityValue();
    return true;
}
