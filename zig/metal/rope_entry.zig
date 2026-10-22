//! Sole existing GPU ABI facade; the pure RoPE owner has no exported symbols.
const c = @import("metal_c");
const owner = @import("rope.zig");
const Tensor = c.ds4_gpu_tensor;
fn tuning(context: u32, inverse: bool, base: f32, scale: f32, ext: f32, attention: f32, fast: f32, slow: f32) owner.Tune {
    return .{ .context = context, .inverse = inverse, .base = base, .scale = scale, .ext = ext, .attention = attention, .fast = fast, .slow = slow };
}
export fn ds4_native_rope_apply(
    command: ?*anyopaque,
    input: owner.View,
    args: *const owner.TailArgs,
    tokens: u32,
    heads: u32,
    width: u32,
    position: u32,
    step: u32,
) c_int {
    // C owns this borrowed command and view for the whole synchronous call.
    const borrowed = owner.Command{ .token = command, .owned = 0 };
    return @intFromBool(owner.apply(&borrowed, input, args, tokens, heads, width, position, step));
}
export fn ds4_gpu_head_rms_norm_tensor(x: ?*Tensor, tokens: u32, heads: u32, width: u32, eps: f32) c_int {
    return @intFromBool(owner.headNorm(x, tokens, heads, width, eps));
}
export fn ds4_gpu_rope_tail_tensor(
    x: ?*Tensor,
    tokens: u32,
    heads: u32,
    width: u32,
    rotated: u32,
    position: u32,
    context: u32,
    inverse: bool,
    base: f32,
    scale: f32,
    ext: f32,
    attention: f32,
    fast: f32,
    slow: f32,
) c_int {
    return @intFromBool(owner.plain(x, tokens, heads, width, rotated, position, tuning(context, inverse, base, scale, ext, attention, fast, slow)));
}
export fn ds4_gpu_head_rms_norm_rope_tail_tensor(
    x: ?*Tensor,
    tokens: u32,
    heads: u32,
    width: u32,
    rotated: u32,
    position: u32,
    context: u32,
    inverse: bool,
    base: f32,
    scale: f32,
    ext: f32,
    attention: f32,
    fast: f32,
    slow: f32,
    eps: f32,
) c_int {
    return @intFromBool(owner.fused(x, tokens, heads, width, rotated, position, tuning(context, inverse, base, scale, ext, attention, fast, slow), eps));
}
export fn ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
    out: ?*Tensor,
    half: ?*Tensor,
    model: ?*const anyopaque,
    size: u64,
    offset: u64,
    input_width: u64,
    outputs: u64,
    x: ?*const Tensor,
    tokens: u32,
    heads: u32,
    width: u32,
    rotated: u32,
    position: u32,
    context: u32,
    inverse: bool,
    base: f32,
    scale: f32,
    ext: f32,
    attention: f32,
    fast: f32,
    slow: f32,
    eps: f32,
) c_int {
    // Preserve the existing unsupported public stub; no new fused algorithm.
    _ = out;
    _ = half;
    _ = model;
    _ = size;
    _ = offset;
    _ = input_width;
    _ = outputs;
    _ = x;
    _ = tokens;
    _ = heads;
    _ = width;
    _ = rotated;
    _ = position;
    _ = context;
    _ = inverse;
    _ = base;
    _ = scale;
    _ = ext;
    _ = attention;
    _ = fast;
    _ = slow;
    _ = eps;
    return 0;
}
export fn ds4_gpu_glm_rope_tail_tensor(
    x: ?*Tensor,
    tokens: u32,
    heads: u32,
    width: u32,
    rotated: u32,
    position: u32,
    context: u32,
    base: f32,
    scale: f32,
    ext: f32,
    attention: f32,
    fast: f32,
    slow: f32,
) c_int {
    if (rotated > width) return 0;
    return @intFromBool(owner.glm(x, tokens, heads, width, rotated, width - rotated, position, tuning(context, false, base, scale, ext, attention, fast, slow), "GLM RoPE"));
}
export fn ds4_gpu_glm_indexer_rope_tail_tensor(
    x: ?*Tensor,
    tokens: u32,
    heads: u32,
    width: u32,
    rotated: u32,
    position: u32,
    context: u32,
    base: f32,
    scale: f32,
    ext: f32,
    attention: f32,
    fast: f32,
    slow: f32,
) c_int {
    return @intFromBool(owner.glm(x, tokens, heads, width, rotated, 0, position, tuning(context, false, base, scale, ext, attention, fast, slow), "GLM indexer RoPE"));
}
export fn ds4_gpu_dsv41_rope_stride(
    x: ?*Tensor,
    width: u32,
    heads: u32,
    rows: u32,
    start: u32,
    stride: u32,
    compressed: bool,
    inverse: bool,
) c_int {
    return @intFromBool(owner.v41(x, width, heads, rows, start, stride, compressed, inverse));
}
export fn ds4_gpu_dsv41_rope(x: ?*Tensor, width: u32, heads: u32, rows: u32, start: u32, compressed: bool, inverse: bool) c_int {
    return @intFromBool(owner.v41(x, width, heads, rows, start, 1, compressed, inverse));
}
