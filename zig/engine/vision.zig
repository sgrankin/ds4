//! DeepSeek V4/V4.1 vision inference. The sidecar binder owns format validation
//! and its mapping. This controller owns a per-image workspace and the exact
//! 32-block/BF16/3x3-aligner flow; Metal leaves own ARC and shader encoding.
const std = @import("std");
const c = @import("gguf_c");
const Tensor = c.ds4_gpu_tensor;
const Weights = c.ds4_deepseek4_vision_weights;

const Workspace = struct {
    patch: ?*Tensor = null,
    a: ?*Tensor = null,
    b: ?*Tensor = null,
    qkv: ?*Tensor = null,
    q: ?*Tensor = null,
    k: ?*Tensor = null,
    v: ?*Tensor = null,
    attention: ?*Tensor = null,
    mlp_w1: ?*Tensor = null,
    mlp_mid: ?*Tensor = null,
    align_in: ?*Tensor = null,
    align_a: ?*Tensor = null,
    align_b: ?*Tensor = null,

    const fields = .{ "patch", "a", "b", "qkv", "q", "k", "v", "attention", "mlp_w1", "mlp_mid", "align_in", "align_a", "align_b" };

    fn init(rows: u32, aligned_rows: u32, projection: u32) ?Workspace {
        var result = Workspace{};
        var complete = false;
        defer if (!complete) result.deinit();
        const values = [_]u64{
            @as(u64, rows) * 588,                @as(u64, rows) * 1024,         @as(u64, rows) * 1024,
            @as(u64, rows) * 3072,               @as(u64, rows) * 1024,         @as(u64, rows) * 1024,
            @as(u64, rows) * 1024,               @as(u64, rows) * 1024,         @as(u64, rows) * 5632,
            @as(u64, rows) * 2816,               @as(u64, aligned_rows) * 9216, @as(u64, aligned_rows) * projection,
            @as(u64, aligned_rows) * projection,
        };
        inline for (fields, 0..) |name, index|
            @field(result, name) = c.ds4_gpu_tensor_alloc(values[index] * 4) orelse return null;
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
        return c.ds4_native_vision_rms(out, input, self.map, self.size, offset, 1024, rows, 1.0e-6, label) != 0;
    }
    fn bias(self: Encoder, kernel: [*:0]const u8, tensor: ?*Tensor, residual: ?*const Tensor, offset: u64, width: u32, rows: u32, label: [*:0]const u8) bool {
        return c.ds4_native_vision_bias(kernel, tensor, residual, self.map, self.size, offset, width, rows, label) != 0;
    }
};

fn round(tensor: ?*Tensor, width: u32, rows: u32) bool {
    return c.ds4_native_vision_round(tensor, width, rows) != 0;
}
fn dump(stage: [*:0]const u8, tensor: ?*const Tensor, values: u64) bool {
    const prefix = c.getenv("DS4_DEEPSEEK4_VISION_DEBUG_PREFIX");
    if (prefix == null or prefix[0] == 0) return true;
    if (tensor == null or values > std.math.maxInt(usize) / 4) return false;
    const resume_commands = c.ds4_gpu_commands_active() != 0;
    if (resume_commands and c.ds4_gpu_end_commands() == 0) return false;
    const data = c.malloc(@intCast(values * 4)) orelse return false;
    var ok = c.ds4_gpu_tensor_read(tensor, 0, data, values * 4) != 0;
    var path: [c.PATH_MAX]u8 = undefined;
    const length = c.snprintf(&path, path.len, "%s.%s.f32", prefix, stage);
    if (ok and (length < 0 or @as(usize, @intCast(length)) >= path.len)) ok = false;
    const file = if (ok) c.fopen(&path, "wb") else null;
    if (ok and (file == null or c.fwrite(data, 4, @intCast(values), file) != values)) ok = false;
    if (file != null and c.fclose(file) != 0) ok = false;
    c.free(data);
    if (resume_commands and c.ds4_gpu_begin_commands() == 0) ok = false;
    return ok;
}

// Preserve the public GPU entry's nullable-pointer validation and both sidecar
// variants. The output is a natural row-major 3x3-aligned grid; prompt assembly
// applies sentinels/N-layout only after the image token position is known.
pub fn encode(out: [*c]f32, patches: [*c]const f32, grid_h: u32, grid_w: u32, model_map: ?*const anyopaque, model_size: u64, pointer: [*c]const Weights) c_int {
    if (out == null or patches == null or model_map == null or pointer == null or grid_h == 0 or grid_w == 0 or
        grid_h > std.math.maxInt(u32) / grid_w or c.ds4_gpu_commands_active() != 0)
    {
        _ = c.fprintf(c.__stderrp, "ds4: invalid DeepSeek vision encoder input\n");
        return 0;
    }
    if (c.ds4_gpu_init() == 0) return 0;
    const weights: *const Weights = @ptrCast(pointer);
    const projection = if (weights.projection_dim != 0) weights.projection_dim else 4096;
    if (projection != 4096 and projection != 5120) return 0;
    const rows = grid_h * grid_w;
    // The public C geometry uses unsigned32 arithmetic, including padding.
    const aligned_rows = ((grid_h +% 2) / 3) *% ((grid_w +% 2) / 3);
    const row1024 = @as(u64, rows) * 1024;
    const row2816 = @as(u64, rows) * 2816;
    const row3072 = @as(u64, rows) * 3072;
    const row5632 = @as(u64, rows) * 5632;
    const aligned_projection = @as(u64, aligned_rows) * projection;
    if (row5632 > std.math.maxInt(usize) / 4 or @as(u64, aligned_rows) * 9216 > std.math.maxInt(usize) / 4) return 0;
    var workspace = Workspace.init(rows, aligned_rows, projection) orelse return 0;
    defer workspace.deinit();
    const encoder = Encoder{ .map = model_map.?, .size = model_size };
    if (c.ds4_gpu_tensor_write(workspace.patch, 0, patches, @as(u64, rows) * 588 * 4) == 0 or
        c.ds4_gpu_begin_commands() == 0) return 0;
    var ok = (projection != 5120 or round(workspace.patch, 588, rows)) and
        encoder.matmul(workspace.a, weights.patch_weight, 588, 1024, workspace.patch, rows) and
        encoder.bias("kernel_glm53_vision_add_bias", workspace.a, null, weights.patch_bias, 1024, rows, "DeepSeek vision patch bias") and
        round(workspace.a, 1024, rows) and dump("patch_embed", workspace.a, row1024);

    var current = workspace.a;
    var temporary = workspace.b;
    for (&weights.layer, 0..) |*layer, index| {
        if (!ok) break;
        ok = encoder.norm(temporary, current, layer.norm1, rows, "DeepSeek vision norm1") and round(temporary, 1024, rows);
        if (ok and index == 0) ok = dump("block0_norm1", temporary, row1024);
        if (ok) ok = encoder.matmul(workspace.qkv, layer.qkv_weight, 1024, 3072, temporary, rows);
        if (ok and index == 0) ok = dump("block0_qkv", workspace.qkv, row3072);
        if (ok) ok = c.ds4_native_vision_qkv(workspace.q, workspace.k, workspace.v, workspace.qkv, model_map, model_size, layer.qkv_bias, rows, grid_w) != 0;
        if (ok and index == 0) ok = dump("block0_q", workspace.q, row1024) and dump("block0_k", workspace.k, row1024) and dump("block0_v", workspace.v, row1024);
        if (ok) ok = c.ds4_native_vision_attention(workspace.attention, workspace.q, workspace.k, workspace.v, rows) != 0 and round(workspace.attention, 1024, rows);
        if (ok and index == 0) ok = dump("block0_attention", workspace.attention, row1024);
        if (ok) ok = encoder.matmul(temporary, layer.attn_proj_weight, 1024, 1024, workspace.attention, rows) and
            encoder.bias(if (projection == 5120) "kernel_deepseek41_vision_bias_residual" else "kernel_glm53_vision_bias_residual", temporary, current, layer.attn_proj_bias, 1024, rows, "DeepSeek vision attention bias") and round(temporary, 1024, rows);
        if (ok and index == 0) ok = dump("block0_attn_residual", temporary, row1024);
        std.mem.swap(?*Tensor, &current, &temporary);
        if (ok) ok = encoder.norm(temporary, current, layer.norm2, rows, "DeepSeek vision norm2") and round(temporary, 1024, rows);
        if (ok and index == 0) ok = dump("block0_norm2", temporary, row1024);
        if (ok) ok = encoder.matmul(workspace.mlp_w1, layer.mlp_w1, 1024, 5632, temporary, rows) and round(workspace.mlp_w1, 5632, rows);
        if (ok and index == 0) ok = dump("block0_mlp_w1", workspace.mlp_w1, row5632);
        if (ok) ok = c.ds4_native_vision_two_rows(if (projection == 5120) "kernel_deepseek41_vision_swiglu_split" else "kernel_deepseek4_vision_swiglu_split", workspace.mlp_w1, workspace.mlp_mid, 2816, rows, "DeepSeek vision SwiGLU") != 0;
        if (ok and index == 0) ok = dump("block0_mlp_mid", workspace.mlp_mid, row2816);
        if (ok) ok = encoder.matmul(temporary, layer.mlp_w2, 2816, 1024, workspace.mlp_mid, rows) and round(temporary, 1024, rows) and
            c.ds4_native_vision_two_rows("kernel_deepseek4_vision_add_residual", temporary, current, 1024, rows, "DeepSeek vision MLP residual") != 0;
        std.mem.swap(?*Tensor, &current, &temporary);
        if (ok and (index == 0 or index == 15 or index == 31)) {
            var stage: [24]u8 = undefined;
            _ = c.snprintf(&stage, stage.len, "block%u", @as(c_uint, @intCast(index)));
            ok = dump(@ptrCast(&stage), current, row1024);
        }
    }
    if (ok) ok = encoder.norm(temporary, current, weights.post_norm, rows, "DeepSeek vision final norm") and
        round(temporary, 1024, rows) and dump("vision_norm", temporary, row1024) and
        c.ds4_native_vision_reorder(workspace.align_in, temporary, grid_h, grid_w, aligned_rows) != 0 and
        encoder.matmul(workspace.align_a, weights.aligner_w1, 9216, projection, workspace.align_in, aligned_rows) and
        c.ds4_native_vision_gelu_bias(workspace.align_b, workspace.align_a, model_map, model_size, weights.aligner_w1_bias, aligned_rows, projection) != 0 and
        encoder.matmul(workspace.align_a, weights.aligner_w2, projection, projection, workspace.align_b, aligned_rows) and
        encoder.bias("kernel_glm53_vision_add_bias", workspace.align_a, null, weights.aligner_w2_bias, projection, aligned_rows, "DeepSeek vision aligner output bias") and
        round(workspace.align_a, projection, aligned_rows) and dump("aligned", workspace.align_a, aligned_projection);
    const ended = c.ds4_gpu_end_commands() != 0;
    ok = ok and ended;
    if (ok) ok = c.ds4_gpu_tensor_read(workspace.align_a, 0, out, aligned_projection * 4) != 0;
    return @intFromBool(ok);
}
