//! Qwen3-VL mmproj binding for the Qwen3.8 language model. Tensor offsets
//! borrow the original GGUF mapping; projection matrices keep their real type.
const std = @import("std");
const c = @import("gguf_c");
const Weights = c.ds4_qwen4_vision_weights;
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
    fn string(self: Reader, key: [:0]const u8, expected: []const u8) bool {
        var value: c.ds4_str = undefined;
        return c.ds4_gguf_get_string(self.model, key.ptr, &value) and std.mem.eql(u8, bytes(value), expected);
    }
    fn integer(self: Reader, comptime suffix: [:0]const u8) InvalidVision!u32 {
        const key = "clip.vision." ++ suffix;
        var value: u32 = undefined;
        if (!c.ds4_gguf_get_u32(self.model, key, &value)) return self.fail("required metadata key is missing: {s}", .{key});
        return value;
    }
    fn expect(self: Reader, comptime label: []const u8, value: u32, expected: u32) InvalidVision!void {
        if (value != expected) return self.fail("expected vision {s}={d}, got {d}", .{ label, expected, value });
    }
    fn tensor(self: Reader, name: []const u8) InvalidVision!*const c.ds4_tensor {
        if (self.model.n_tensors != 0) for (self.model.tensors[0..@intCast(self.model.n_tensors)]) |*value| {
            if (std.mem.eql(u8, bytes(value.name), name)) return value;
        };
        return self.fail("required tensor is missing: {s}", .{name});
    }
    fn offset(self: Reader, name: []const u8, dimensions: []const u64) InvalidVision!u64 {
        const value = try self.tensor(name);
        if (value.type != c.DS4_TENSOR_F32 or value.ndim != dimensions.len)
            return self.fail("vision tensor {s} must be F32 with rank {d}", .{ name, dimensions.len });
        for (dimensions, 0..) |expected, axis| if (value.dim[axis] != expected)
            return self.fail("vision tensor {s} has dim[{d}]={d}, expected {d}", .{ name, axis, value.dim[axis], expected });
        return value.abs_offset;
    }
    fn dense(self: Reader, name: []const u8, input: u64, output: u64, kind: *u32) InvalidVision!u64 {
        const value = try self.tensor(name);
        var width: u64 = 1;
        var axis: u32 = 0;
        while (axis + 1 < value.ndim) : (axis += 1) width *%= value.dim[axis];
        if ((value.type != c.DS4_TENSOR_F32 and value.type != c.DS4_TENSOR_F16 and value.type != c.DS4_TENSOR_Q8_0) or
            value.ndim < 2 or width != input or value.dim[value.ndim - 1] != output)
            return self.fail("vision tensor {s} must be F32/F16/Q8_0 [{d} x {d}]", .{ name, input, output });
        kind.* = value.type;
        return value.abs_offset;
    }
};

pub fn bind(weights: *Weights, model: *const c.ds4_model, errors: []u8) InvalidVision!void {
    const reader = Reader{ .model = model, .errors = errors };
    if (!reader.string("general.architecture", "clip") or !reader.string("clip.projector_type", "qwen3vl_merger"))
        return reader.fail("--vision file is not a Qwen3-VL mmproj GGUF (clip / qwen3vl_merger)", .{});
    weights.* = std.mem.zeroes(Weights);
    weights.n_embd = try reader.integer("embedding_length");
    weights.n_ff = try reader.integer("feed_forward_length");
    weights.n_head = try reader.integer("attention.head_count");
    weights.n_patch = try reader.integer("patch_size");
    weights.n_merge = try reader.integer("spatial_merge_size");
    weights.n_out = try reader.integer("projection_dim");
    try reader.expect("block_count", try reader.integer("block_count"), c.DS4_QWEN4_VISION_LAYERS);
    try reader.expect("embedding_length", weights.n_embd, 1152);
    try reader.expect("head_count", weights.n_head, 16);
    try reader.expect("patch_size", weights.n_patch, 16);
    try reader.expect("spatial_merge_size", weights.n_merge, 2);
    weights.eps = 1.0e-6;
    _ = c.ds4_gguf_get_f32_compat(model, "clip.vision.attention.layer_norm_epsilon", &weights.eps);
    const width: u64 = weights.n_embd;
    const hidden: u64 = weights.n_ff;
    const merged: u64 = width * weights.n_merge * weights.n_merge;
    const output: u64 = weights.n_out;
    const position = try reader.tensor("v.position_embd.weight");
    if (position.type != c.DS4_TENSOR_F32 or position.ndim != 2 or position.dim[0] != width)
        return reader.fail("vision tensor v.position_embd.weight must be F32 [embd, positions]", .{});
    weights.n_pos_side = @intFromFloat(@sqrt(@as(f64, @floatFromInt(position.dim[1]))));
    if (@as(u64, weights.n_pos_side) * weights.n_pos_side != position.dim[1] or weights.n_pos_side < 2)
        return reader.fail("vision position table is not square", .{});
    weights.pos_embd = position.abs_offset;
    var second_patch_type: u32 = 0;
    const patch_width = 3 * @as(u64, weights.n_patch) * weights.n_patch;
    weights.patch_w0 = try reader.dense("v.patch_embd.weight", patch_width, width, &weights.patch_type);
    weights.patch_w1 = try reader.dense("v.patch_embd.weight.1", patch_width, width, &second_patch_type);
    if (second_patch_type != weights.patch_type or weights.patch_type == c.DS4_TENSOR_Q8_0)
        return reader.fail("vision patch weights must both be F32 or F16", .{});
    weights.patch_b = try reader.offset("v.patch_embd.bias", &.{width});
    weights.post_ln_w = try reader.offset("v.post_ln.weight", &.{width});
    weights.post_ln_b = try reader.offset("v.post_ln.bias", &.{width});
    weights.mm0_w = try reader.dense("mm.0.weight", merged, merged, &weights.mm0_type);
    weights.mm0_b = try reader.offset("mm.0.bias", &.{merged});
    weights.mm2_w = try reader.dense("mm.2.weight", merged, output, &weights.mm2_type);
    weights.mm2_b = try reader.offset("mm.2.bias", &.{output});
    for (&weights.layer, 0..) |*layer, index| {
        var storage: [96]u8 = undefined;
        const prefix = std.mem.printSentinel(&storage, "v.blk.{d}.", .{index}, 0) catch unreachable;
        inline for (.{
            .{ "ln1_w", "ln1.weight", width },        .{ "ln1_b", "ln1.bias", width },
            .{ "qkv_b", "attn_qkv.bias", 3 * width }, .{ "out_b", "attn_out.bias", width },
            .{ "ln2_w", "ln2.weight", width },        .{ "ln2_b", "ln2.bias", width },
            .{ "up_b", "ffn_up.bias", hidden },       .{ "down_b", "ffn_down.bias", width },
        }) |field| {
            var name: [96]u8 = undefined;
            const full = std.mem.printSentinel(&name, "{s}{s}", .{ prefix, field[1] }, 0) catch unreachable;
            @field(layer, field[0]) = try reader.offset(full, &.{field[2]});
        }
        inline for (.{
            .{ "qkv_w", "attn_qkv.weight", "qkv_type", width, 3 * width },
            .{ "out_w", "attn_out.weight", "out_type", width, width },
            .{ "up_w", "ffn_up.weight", "up_type", width, hidden },
            .{ "down_w", "ffn_down.weight", "down_type", hidden, width },
        }) |field| {
            var name: [96]u8 = undefined;
            const full = std.mem.printSentinel(&name, "{s}{s}", .{ prefix, field[1] }, 0) catch unreachable;
            @field(layer, field[0]) = try reader.dense(full, field[3], field[4], &@field(layer, field[2]));
        }
    }
}
