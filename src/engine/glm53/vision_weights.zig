//! GLM 5.3's BF16 sidecar directory. Its fixed architecture is checked before
//! publishing offsets; the mapping remains owned by the engine's model record.
const std = @import("std");
const c = @import("gguf_c");
const Weights = c.ds4_glm53_vision_weights;
const InvalidVision = error{InvalidVision};

fn bytes(value: c.ds4_str) []const u8 {
    return if (value.len == 0) &.{} else value.ptr[0..@intCast(value.len)];
}
const Reader = struct {
    model: *const c.ds4_model,
    errors: []u8,

    fn fail(self: Reader, comptime format: []const u8, args: anytype) InvalidVision {
        if (self.errors.len != 0) {
            _ = std.mem.printSentinel(self.errors, format, args, 0) catch {
                self.errors[self.errors.len - 1] = 0;
                return error.InvalidVision;
            };
        }
        return error.InvalidVision;
    }
    fn integer(self: Reader, comptime suffix: [:0]const u8, expected: u32) InvalidVision!void {
        const key = "glm5-next-vision." ++ suffix;
        var value: u32 = undefined;
        if (!c.ds4_gguf_get_u32(self.model, key, &value)) return self.fail("required metadata key is missing: {s}", .{key});
        if (value != expected) return self.fail("expected vision {s}={d}, got {d}", .{ suffix, expected, value });
    }
    fn offset(self: Reader, name: []const u8, dimensions: []const u64) InvalidVision!u64 {
        if (self.model.n_tensors != 0) for (self.model.tensors[0..@intCast(self.model.n_tensors)]) |*value| {
            if (!std.mem.eql(u8, bytes(value.name), name)) continue;
            if (value.type != c.DS4_TENSOR_BF16 or value.ndim != dimensions.len)
                return self.fail("vision tensor {s} must be BF16 with rank {d}", .{ name, dimensions.len });
            for (dimensions, 0..) |expected, axis| if (value.dim[axis] != expected)
                return self.fail("vision tensor {s} has dim[{d}]={d}, expected {d}", .{ name, axis, value.dim[axis], expected });
            return value.abs_offset;
        };
        return self.fail("required tensor is missing: {s}", .{name});
    }
};

pub fn bind(weights: *Weights, model: *const c.ds4_model, errors: []u8) InvalidVision!void {
    const reader = Reader{ .model = model, .errors = errors };
    var architecture: c.ds4_str = undefined;
    if (!c.ds4_gguf_get_string(model, "general.architecture", &architecture) or !std.mem.eql(u8, bytes(architecture), "glm5-next-vision"))
        return reader.fail("--vision file is not a GLM-5.3 vision encoder GGUF", .{});
    if (model.n_tensors != 347) return reader.fail("vision GGUF has {d} tensors, expected 347", .{model.n_tensors});
    inline for (.{
        .{ "block_count", 24 },           .{ "embedding_length", 1024 },
        .{ "feed_forward_length", 4096 }, .{ "attention.head_count", 16 },
        .{ "projection_length", 4096 },   .{ "projection.feed_forward_length", 10240 },
        .{ "patch_size", 14 },            .{ "temporal_patch_size", 2 },
        .{ "spatial_merge_size", 2 },
    }) |field| try reader.integer(field[0], field[1]);
    weights.* = std.mem.zeroes(Weights);
    weights.patch_weight = try reader.offset("model.visual.patch_embed.proj.weight", &.{ 14, 14, 2, 3, 1024 });
    weights.patch_bias = try reader.offset("model.visual.patch_embed.proj.bias", &.{1024});
    weights.post_norm = try reader.offset("model.visual.post_layernorm.weight", &.{1024});
    weights.downsample_weight = try reader.offset("model.visual.downsample.weight", &.{ 2, 2, 1024, 4096 });
    weights.downsample_bias = try reader.offset("model.visual.downsample.bias", &.{4096});
    weights.merger_proj = try reader.offset("model.visual.merger.proj.weight", &.{ 4096, 4096 });
    weights.merger_norm = try reader.offset("model.visual.merger.post_projection_norm.weight", &.{4096});
    weights.merger_norm_bias = try reader.offset("model.visual.merger.post_projection_norm.bias", &.{4096});
    weights.merger_gate = try reader.offset("model.visual.merger.gate_proj.weight", &.{ 4096, 10240 });
    weights.merger_up = try reader.offset("model.visual.merger.up_proj.weight", &.{ 4096, 10240 });
    weights.merger_down = try reader.offset("model.visual.merger.down_proj.weight", &.{ 10240, 4096 });
    for (&weights.layer, 0..) |*layer, index| {
        inline for (.{
            .{ "norm1", "norm1.weight", &.{1024} },
            .{ "qkv_weight", "attn.qkv.weight", &.{ 1024, 3072 } },
            .{ "qkv_bias", "attn.qkv.bias", &.{3072} },
            .{ "q_norm", "attn.q_norm.weight", &.{64} },
            .{ "k_norm", "attn.k_norm.weight", &.{64} },
            .{ "attn_proj_weight", "attn.proj.weight", &.{ 1024, 1024 } },
            .{ "attn_proj_bias", "attn.proj.bias", &.{1024} },
            .{ "norm2", "norm2.weight", &.{1024} },
            .{ "gate_weight", "mlp.gate_proj.weight", &.{ 1024, 4096 } },
            .{ "gate_bias", "mlp.gate_proj.bias", &.{4096} },
            .{ "up_weight", "mlp.up_proj.weight", &.{ 1024, 4096 } },
            .{ "up_bias", "mlp.up_proj.bias", &.{4096} },
            .{ "down_weight", "mlp.down_proj.weight", &.{ 4096, 1024 } },
            .{ "down_bias", "mlp.down_proj.bias", &.{1024} },
        }) |field| {
            var storage: [160]u8 = undefined;
            const name = std.mem.printSentinel(&storage, "model.visual.blocks.{d}.{s}", .{ index, field[1] }, 0) catch unreachable;
            @field(layer, field[0]) = try reader.offset(name, field[2]);
        }
    }
}
