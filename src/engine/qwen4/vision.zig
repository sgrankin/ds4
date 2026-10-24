//! Qwen3-VL vision tower. The native owner controls its 27-block workspace,
//! temporal patch sum and two-layer merger; Objective-C encodes shader leaves.
const std = @import("std");
const c = @import("gguf_c");
const Tensor = c.ds4_gpu_tensor;
const Weights = c.ds4_qwen4_vision_weights;

const Workspace = struct {
    patch: ?*Tensor = null,
    first: ?*Tensor = null,
    second: ?*Tensor = null,
    position: ?*Tensor = null,
    residual: ?*Tensor = null,
    temporary: ?*Tensor = null,
    qkv: ?*Tensor = null,
    q: ?*Tensor = null,
    k: ?*Tensor = null,
    v: ?*Tensor = null,
    attention: ?*Tensor = null,
    ffn: ?*Tensor = null,
    merger: ?*Tensor = null,
    output: ?*Tensor = null,

    const fields = .{ "patch", "first", "second", "position", "residual", "temporary", "qkv", "q", "k", "v", "attention", "ffn", "merger", "output" };

    fn init(rows: u32, merged_rows: u32, patch_width: u32, weights: *const Weights) ?Workspace {
        var result = Workspace{};
        var complete = false;
        defer if (!complete) result.deinit();
        const n: u64 = rows;
        const width: u64 = weights.n_embd;
        const values = [_]u64{
            n * patch_width, n * width,                             n * width, n * width, n * width, n * width,
            n * 3 * width,   n * width,                             n * width, n * width, n * width, n * weights.n_ff,
            n * width,       @as(u64, merged_rows) * weights.n_out,
        };
        inline for (fields, 0..) |name, index|
            @field(result, name) = c.ds4_gpu_tensor_alloc(values[index] * @sizeOf(f32)) orelse return null;
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

    fn matmul(self: Encoder, out: ?*Tensor, input: ?*const Tensor, offset: u64, kind: u32, rows: u32, input_width: u32, output_width: u32) bool {
        return c.ds4_gpu_qwen4_dense_mm_tensor(out, input, self.map, self.size, offset, kind, rows, input_width, output_width) != 0;
    }
    fn norm(self: Encoder, out: ?*Tensor, input: ?*const Tensor, weight: u64, bias: u64, rows: u32, width: u32, epsilon: f32) bool {
        return c.ds4_native_qwen4_vision_norm(out, input, self.map, self.size, weight, bias, rows, width, epsilon) != 0;
    }
    fn residual(self: Encoder, out: ?*Tensor, input: ?*const Tensor, bias: u64, rows: u32, width: u32) bool {
        return c.ds4_native_qwen4_vision_residual(out, input, self.map, self.size, bias, rows, width) != 0;
    }
    fn activate(self: Encoder, tensor: ?*Tensor, bias: u64, rows: u32, width: u32, mode: u32) bool {
        return c.ds4_native_qwen4_vision_bias_activation(tensor, self.map, self.size, bias, rows, width, mode) != 0;
    }
};

pub fn encode(out: [*c]f32, patches: [*c]const f32, position: [*c]const f32, rows: u32, grid_width: u32, model_map: ?*const anyopaque, model_size: u64, pointer: [*c]const Weights) c_int {
    @setFloatMode(.optimized); // Match the original C controller's F32 scale.
    if (out == null or patches == null or position == null or model_map == null or pointer == null) return 0;
    const weights: *const Weights = @ptrCast(pointer);
    const width = weights.n_embd;
    const hidden = weights.n_ff;
    const heads = weights.n_head;
    const head_width = if (heads != 0) width / heads else 0;
    const merge = weights.n_merge;
    const merge_area = merge *% merge;
    const merged_width = width *% merge_area;
    const patch_width = 3 *% weights.n_patch *% weights.n_patch;
    const output_width = weights.n_out;
    if (rows == 0 or merge_area == 0 or rows % merge_area != 0 or grid_width == 0 or grid_width % merge != 0 or
        rows % grid_width != 0 or (rows / grid_width) % merge != 0 or (head_width != 64 and head_width != 72) or
        heads *% head_width != width or hidden == 0 or output_width == 0 or patch_width % 32 != 0 or width % 32 != 0 or
        rows > 65535 or c.ds4_gpu_commands_active() != 0)
    {
        _ = c.fprintf(c.__stderrp, "ds4: invalid Qwen3.8 vision encoder input\n");
        return 0;
    }
    if (c.ds4_gpu_init() == 0) return 0;
    const merged_rows = rows / merge_area;
    var workspace = Workspace.init(rows, merged_rows, patch_width, weights) orelse return 0;
    defer workspace.deinit();
    const encoder = Encoder{ .map = model_map.?, .size = model_size };
    if (c.ds4_gpu_tensor_write(workspace.patch, 0, patches, @as(u64, rows) * patch_width * @sizeOf(f32)) == 0 or
        c.ds4_gpu_tensor_write(workspace.position, 0, position, @as(u64, rows) * width * @sizeOf(f32)) == 0 or
        c.ds4_gpu_begin_commands() == 0) return 0;
    var ok = encoder.matmul(workspace.first, workspace.patch, weights.patch_w0, weights.patch_type, rows, patch_width, width) and
        encoder.matmul(workspace.second, workspace.patch, weights.patch_w1, weights.patch_type, rows, patch_width, width) and
        c.ds4_native_qwen4_vision_patch(workspace.residual, workspace.first, workspace.second, workspace.position, model_map, model_size, weights.patch_b, rows, width) != 0;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_width)));
    for (&weights.layer) |*layer| {
        if (!ok) break;
        ok = encoder.norm(workspace.temporary, workspace.residual, layer.ln1_w, layer.ln1_b, rows, width, weights.eps) and
            encoder.matmul(workspace.qkv, workspace.temporary, layer.qkv_w, layer.qkv_type, rows, width, 3 * width) and
            c.ds4_native_qwen4_vision_qkv(workspace.q, workspace.k, workspace.v, workspace.qkv, model_map, model_size, layer.qkv_b, rows, width, grid_width, heads, head_width) != 0 and
            c.ds4_native_qwen4_vision_attention(workspace.attention, workspace.q, workspace.k, workspace.v, rows, width, heads, head_width, scale) != 0 and
            encoder.matmul(workspace.temporary, workspace.attention, layer.out_w, layer.out_type, rows, width, width) and
            encoder.residual(workspace.residual, workspace.temporary, layer.out_b, rows, width) and
            encoder.norm(workspace.temporary, workspace.residual, layer.ln2_w, layer.ln2_b, rows, width, weights.eps) and
            encoder.matmul(workspace.ffn, workspace.temporary, layer.up_w, layer.up_type, rows, width, hidden) and
            encoder.activate(workspace.ffn, layer.up_b, rows, hidden, 0) and // tanh GELU
            encoder.matmul(workspace.temporary, workspace.ffn, layer.down_w, layer.down_type, rows, hidden, width) and
            encoder.residual(workspace.residual, workspace.temporary, layer.down_b, rows, width);
    }
    // Patch rows are already grouped into 2x2 windows. Reinterpret them as
    // merged_rows x merged_width; a second permutation here would corrupt it.
    if (ok) ok = encoder.norm(workspace.temporary, workspace.residual, weights.post_ln_w, weights.post_ln_b, rows, width, weights.eps) and
        encoder.matmul(workspace.merger, workspace.temporary, weights.mm0_w, weights.mm0_type, merged_rows, merged_width, merged_width) and
        encoder.activate(workspace.merger, weights.mm0_b, merged_rows, merged_width, 1) and // erf GELU
        encoder.matmul(workspace.output, workspace.merger, weights.mm2_w, weights.mm2_type, merged_rows, merged_width, output_width) and
        encoder.activate(workspace.output, weights.mm2_b, merged_rows, output_width, 2); // bias only
    const ended = c.ds4_gpu_end_commands() != 0;
    ok = ok and ended;
    if (ok) ok = c.ds4_gpu_tensor_read(workspace.output, 0, out, @as(u64, merged_rows) * output_width * @sizeOf(f32)) != 0;
    return @intFromBool(ok);
}

/// Bilinear align_corners interpolation of the learned square position table.
/// Emit rows in the same 2x2 window order as preprocessing and the merger.
pub fn positionEmbeddings(table: [*c]const f32, side: u32, width: u32, grid_height: u32, grid_width: u32, out: [*c]f32) void {
    @setFloatMode(.optimized); // Original C TU uses -ffast-math.
    var row: usize = 0;
    for (0..grid_height / 2) |block_y| for (0..grid_width / 2) |block_x| {
        for (0..2) |dy| for (0..2) |dx| {
            const y = block_y * 2 + dy;
            const x = block_x * 2 + dx;
            const source_y = if (grid_height > 1) @as(f64, @floatFromInt(y)) * @as(f64, @floatFromInt(side - 1)) / @as(f64, @floatFromInt(grid_height - 1)) else 0;
            const source_x = if (grid_width > 1) @as(f64, @floatFromInt(x)) * @as(f64, @floatFromInt(side - 1)) / @as(f64, @floatFromInt(grid_width - 1)) else 0;
            const y0: u32 = @intFromFloat(source_y);
            const x0: u32 = @intFromFloat(source_x);
            const y1 = if (y0 + 1 < side) y0 + 1 else y0;
            const x1 = if (x0 + 1 < side) x0 + 1 else x0;
            const wy: f32 = @floatCast(source_y - @as(f64, @floatFromInt(y0)));
            const wx: f32 = @floatCast(source_x - @as(f64, @floatFromInt(x0)));
            const top_left = table[(@as(usize, y0) * side + x0) * width ..][0..width];
            const top_right = table[(@as(usize, y0) * side + x1) * width ..][0..width];
            const bottom_left = table[(@as(usize, y1) * side + x0) * width ..][0..width];
            const bottom_right = table[(@as(usize, y1) * side + x1) * width ..][0..width];
            const destination = out[row * width ..][0..width];
            for (destination, 0..) |*value, axis|
                value.* = (1.0 - wy) * ((1.0 - wx) * top_left[axis] + wx * top_right[axis]) +
                    wy * ((1.0 - wx) * bottom_left[axis] + wx * bottom_right[axis]);
            row += 1;
        };
    };
}
