//! GLM 5.3 vision tower: per-image workspace, 24 transformer blocks, and the
//! 2x2 downsampler/merger. Metal leaves retain shader bindings and dispatch.
const std = @import("std");
const c = @import("gguf_c");
const Tensor = c.ds4_gpu_tensor;
const Weights = c.ds4_glm53_vision_weights;

const Workspace = struct {
    patch: ?*Tensor = null,
    a: ?*Tensor = null,
    b: ?*Tensor = null,
    qkv: ?*Tensor = null,
    q: ?*Tensor = null,
    k: ?*Tensor = null,
    v: ?*Tensor = null,
    attention: ?*Tensor = null,
    gate: ?*Tensor = null,
    up: ?*Tensor = null,
    middle: ?*Tensor = null,

    const fields = .{ "patch", "a", "b", "qkv", "q", "k", "v", "attention", "gate", "up", "middle" };

    fn init(rows: u32) ?Workspace {
        var result = Workspace{};
        var complete = false;
        defer if (!complete) result.deinit();
        // a/b become merged_rows x 4096 after the tower. The 4096-wide MLP
        // buffers also cover merged_rows x 10240 in the final merger.
        const widths = [_]u32{ 1176, 1024, 1024, 3072, 1024, 1024, 1024, 1024, 4096, 4096, 4096 };
        inline for (fields, 0..) |name, index|
            @field(result, name) = c.ds4_gpu_tensor_alloc(@as(u64, rows) * widths[index] * @sizeOf(f32)) orelse return null;
        complete = true;
        return result;
    }
    fn deinit(self: *Workspace) void {
        inline for (0..fields.len) |index| c.ds4_gpu_tensor_free(@field(self, fields[fields.len - 1 - index]));
    }
};

const Encoder = struct {
    map: *const anyopaque,
    size: u64,

    fn matmul(self: Encoder, out: ?*Tensor, offset: u64, input_width: u32, output_width: u32, input: ?*const Tensor, rows: u32) bool {
        return c.ds4_gpu_glm53_matmul_bf16(out, self.map, self.size, offset, input_width, output_width, input, rows) != 0;
    }
    fn norm(self: Encoder, out: ?*Tensor, input: ?*const Tensor, offset: u64, rows: u32, label: [*:0]const u8) bool {
        return c.ds4_native_vision_rms(out, input, self.map, self.size, offset, 1024, rows, 1.0e-5, label) != 0;
    }
    fn bias(self: Encoder, out: ?*Tensor, residual: ?*const Tensor, offset: u64, width: u32, rows: u32, label: [*:0]const u8) bool {
        const kernel: [*:0]const u8 = if (residual != null) "kernel_glm53_vision_bias_residual" else "kernel_glm53_vision_add_bias";
        return c.ds4_native_vision_bias(kernel, out, residual, self.map, self.size, offset, width, rows, label) != 0;
    }
};

pub fn encode(out: [*c]f32, patches: [*c]const f32, grid_height: u32, grid_width: u32, model_map: ?*const anyopaque, model_size: u64, pointer: [*c]const Weights) c_int {
    if (out == null or patches == null or model_map == null or pointer == null or grid_height == 0 or grid_width == 0 or
        (grid_height & 1) != 0 or (grid_width & 1) != 0 or grid_height > std.math.maxInt(u32) / grid_width or c.ds4_gpu_commands_active() != 0)
    {
        _ = c.fprintf(c.__stderrp, "ds4: invalid GLM-5.3 vision encoder input\n");
        return 0;
    }
    if (c.ds4_gpu_init() == 0) return 0;
    const weights: *const Weights = @ptrCast(pointer);
    const rows = grid_height * grid_width;
    const merged_rows = rows / 4;
    if (@as(u64, rows) * 4096 > std.math.maxInt(usize) / @sizeOf(f32) or
        @as(u64, merged_rows) * 10240 > std.math.maxInt(usize) / @sizeOf(f32)) return 0;
    var workspace = Workspace.init(rows) orelse return 0;
    defer workspace.deinit();
    const encoder = Encoder{ .map = model_map.?, .size = model_size };
    if (c.ds4_gpu_tensor_write(workspace.patch, 0, patches, @as(u64, rows) * 1176 * @sizeOf(f32)) == 0 or
        c.ds4_gpu_begin_commands() == 0) return 0;
    var ok = encoder.matmul(workspace.a, weights.patch_weight, 1176, 1024, workspace.patch, rows) and
        encoder.bias(workspace.a, null, weights.patch_bias, 1024, rows, "vision patch bias");
    var current = workspace.a;
    var temporary = workspace.b;
    for (&weights.layer) |*layer| {
        if (!ok) break;
        ok = encoder.norm(temporary, current, layer.norm1, rows, "vision norm1") and
            encoder.matmul(workspace.qkv, layer.qkv_weight, 1024, 3072, temporary, rows) and
            c.ds4_native_glm53_vision_qkv(workspace.q, workspace.k, workspace.v, workspace.qkv, model_map, model_size, layer, rows, grid_height, grid_width) != 0 and
            c.ds4_native_vision_attention(workspace.attention, workspace.q, workspace.k, workspace.v, rows) != 0 and
            encoder.matmul(temporary, layer.attn_proj_weight, 1024, 1024, workspace.attention, rows) and
            encoder.bias(temporary, current, layer.attn_proj_bias, 1024, rows, "vision attention bias");
        std.mem.swap(?*Tensor, &current, &temporary);
        if (ok) ok = encoder.norm(temporary, current, layer.norm2, rows, "vision norm2") and
            encoder.matmul(workspace.gate, layer.gate_weight, 1024, 4096, temporary, rows) and
            encoder.matmul(workspace.up, layer.up_weight, 1024, 4096, temporary, rows) and
            c.ds4_native_glm53_vision_swiglu_bias(workspace.middle, workspace.gate, workspace.up, model_map, model_size, layer.gate_bias, layer.up_bias, 4096, rows) != 0 and
            encoder.matmul(temporary, layer.down_weight, 4096, 1024, workspace.middle, rows) and
            encoder.bias(temporary, current, layer.down_bias, 1024, rows, "vision down bias");
        std.mem.swap(?*Tensor, &current, &temporary);
    }
    if (ok) ok = encoder.norm(temporary, current, weights.post_norm, rows, "vision post norm") and
        c.ds4_native_glm53_vision_reorder(current, temporary, merged_rows) != 0 and
        encoder.matmul(temporary, weights.downsample_weight, 4096, 4096, current, merged_rows) and
        encoder.bias(temporary, null, weights.downsample_bias, 4096, merged_rows, "vision downsample bias") and
        encoder.matmul(current, weights.merger_proj, 4096, 4096, temporary, merged_rows) and
        c.ds4_native_glm53_vision_merger_norm(temporary, current, model_map, model_size, weights.merger_norm, weights.merger_norm_bias, merged_rows) != 0 and
        encoder.matmul(workspace.gate, weights.merger_gate, 4096, 10240, temporary, merged_rows) and
        encoder.matmul(workspace.up, weights.merger_up, 4096, 10240, temporary, merged_rows) and
        c.ds4_gpu_swiglu_tensor(workspace.middle, workspace.gate, workspace.up, merged_rows * 10240, 10.0, 1.0) != 0 and
        encoder.matmul(current, weights.merger_down, 10240, 4096, workspace.middle, merged_rows);
    // Close the command group even if a leaf fails, before releasing buffers.
    const ended = c.ds4_gpu_end_commands() != 0;
    ok = ok and ended;
    if (ok) ok = c.ds4_gpu_tensor_read(current, 0, out, @as(u64, merged_rows) * 4096 * @sizeOf(f32)) != 0;
    return @intFromBool(ok);
}
