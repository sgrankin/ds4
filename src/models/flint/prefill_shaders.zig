//! Assemble the small, vendored MLX Metal dependency closure into one source.
//! This uses shader source only; the serving executable does not link MLX.
const std = @import("std");

const Entry = struct { name: []const u8, text: []const u8 };
const entries = [_]Entry{
    .{ .name = "mlx/backend/metal/kernels/bf16.h", .text = @embedFile("../../metal/flint/prefill/vendor/mlx/backend/metal/kernels/bf16.h") },
    .{ .name = "mlx/backend/metal/kernels/bf16_math.h", .text = @embedFile("../../metal/flint/prefill/vendor/mlx/backend/metal/kernels/bf16_math.h") },
    .{ .name = "mlx/backend/metal/kernels/complex.h", .text = @embedFile("../../metal/flint/prefill/vendor/mlx/backend/metal/kernels/complex.h") },
    .{ .name = "mlx/backend/metal/kernels/defines.h", .text = @embedFile("../../metal/flint/prefill/vendor/mlx/backend/metal/kernels/defines.h") },
    .{ .name = "mlx/backend/metal/kernels/logging.h", .text = @embedFile("../../metal/flint/prefill/vendor/mlx/backend/metal/kernels/logging.h") },
    .{ .name = "mlx/backend/metal/kernels/utils.h", .text = @embedFile("../../metal/flint/prefill/vendor/mlx/backend/metal/kernels/utils.h") },
    .{ .name = "mlx/backend/metal/kernels/steel/defines.h", .text = @embedFile("../../metal/flint/prefill/vendor/mlx/backend/metal/kernels/steel/defines.h") },
    .{ .name = "mlx/backend/metal/kernels/steel/utils.h", .text = @embedFile("../../metal/flint/prefill/vendor/mlx/backend/metal/kernels/steel/utils.h") },
    .{ .name = "mlx/backend/metal/kernels/steel/utils/type_traits.h", .text = @embedFile("../../metal/flint/prefill/vendor/mlx/backend/metal/kernels/steel/utils/type_traits.h") },
    .{ .name = "mlx/backend/metal/kernels/steel/utils/integral_constant.h", .text = @embedFile("../../metal/flint/prefill/vendor/mlx/backend/metal/kernels/steel/utils/integral_constant.h") },
    .{ .name = "mlx/backend/metal/kernels/steel/gemm/nax.h", .text = @embedFile("../../metal/flint/prefill/vendor/mlx/backend/metal/kernels/steel/gemm/nax.h") },
    .{ .name = "mlx/backend/metal/kernels/steel/attn/params.h", .text = @embedFile("../../metal/flint/prefill/vendor/mlx/backend/metal/kernels/steel/attn/params.h") },
    .{ .name = "mlx/backend/metal/kernels/steel/attn/transforms.h", .text = @embedFile("../../metal/flint/prefill/vendor/mlx/backend/metal/kernels/steel/attn/transforms.h") },
    .{ .name = "mlx/backend/metal/kernels/quantized_nax.h", .text = @embedFile("../../metal/flint/prefill/vendor/mlx/backend/metal/kernels/quantized_nax.h") },
    .{ .name = "ops.metal", .text = @embedFile("../../metal/flint/prefill/ops.metal") },
    .{ .name = "matmul.metal", .text = @embedFile("../../metal/flint/prefill/matmul.metal") },
    .{ .name = "attention.metal", .text = @embedFile("../../metal/flint/prefill/attention.metal") },
};

const Builder = struct {
    allocator: std.mem.Allocator,
    out: std.ArrayList(u8) = .empty,
    included: [entries.len]bool = @splat(false),

    fn include(self: *Builder, name: []const u8) !void {
        for (entries, 0..) |entry, i| {
            if (!std.mem.eql(u8, name, entry.name)) continue;
            if (self.included[i]) return;
            self.included[i] = true;
            try self.out.appendSlice(self.allocator, "\n// Source: ");
            try self.out.appendSlice(self.allocator, name);
            try self.out.append(self.allocator, '\n');
            var lines = std.mem.splitScalar(u8, entry.text, '\n');
            while (lines.next()) |line| {
                if (std.mem.startsWith(u8, line, "#include \"")) {
                    const close = std.mem.lastIndexOfScalar(u8, line, '"') orelse return error.InvalidShaderInclude;
                    try self.include(line[10..close]);
                } else if (std.mem.eql(u8, line, "#pragma once") or
                    (std.mem.eql(u8, entry.name, "mlx/backend/metal/kernels/quantized_nax.h") and
                        std.mem.startsWith(u8, line, "#include <")))
                {
                    continue;
                } else if (std.mem.eql(u8, entry.name, "mlx/backend/metal/kernels/quantized_nax.h") and
                    std.mem.startsWith(u8, line, "constant bool align_"))
                {
                    // All Flint widths are multiples of 64. Token rows retain
                    // tail checks. Fixed entries need no PSO specializations.
                    const replacement = if (std.mem.indexOf(u8, line, "align_M") != null)
                        "constant bool align_M = false;"
                    else if (std.mem.indexOf(u8, line, "align_N") != null)
                        "constant bool align_N = true;"
                    else
                        "constant bool align_K = true;";
                    try self.out.appendSlice(self.allocator, replacement);
                    try self.out.append(self.allocator, '\n');
                } else {
                    try self.out.appendSlice(self.allocator, line);
                    try self.out.append(self.allocator, '\n');
                }
            }
            return;
        }
        return error.UnknownShaderInclude;
    }
};

pub fn source(allocator: std.mem.Allocator) ![]u8 {
    var builder = Builder{ .allocator = allocator };
    errdefer builder.out.deinit(allocator);
    try builder.include("mlx/backend/metal/kernels/utils.h");
    try builder.include("mlx/backend/metal/kernels/steel/utils.h");
    try builder.include("mlx/backend/metal/kernels/steel/gemm/nax.h");
    try builder.include("mlx/backend/metal/kernels/steel/attn/params.h");
    try builder.include("mlx/backend/metal/kernels/steel/attn/transforms.h");
    try builder.include("ops.metal");
    try builder.include("matmul.metal");
    try builder.include("attention.metal");
    return builder.out.toOwnedSlice(allocator);
}

test "prefill source resolves its vendored headers and fixed entries" {
    const text = try source(std.testing.allocator);
    defer std.testing.allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "#include \"") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "[[function_constant(") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "flint_prefill_qmm4") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "flint_prefill_attention") != null);
}
