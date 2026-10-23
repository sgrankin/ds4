//! Sole existing primitive ABI facade plus borrowed-copy compatibility actions.
const c = @import("metal_c");
const owner = @import("primitives.zig");
const Copy = @import("copy.zig");
const Tensor = c.ds4_gpu_tensor;
const View = c.ds4_metal_attention_view;
export fn ds4_gpu_dsv4_fp8_kv_quantize_tensor(x: ?*Tensor, rows: u32, width: u32, rotated: u32) c_int {
    return @intFromBool(owner.fp8(x, rows, width, rotated, false));
}
export fn ds4_gpu_dsv4_indexer_qat_tensor(x: ?*Tensor, rows: u32, width: u32) c_int {
    return @intFromBool(owner.fp8(x, rows, width, 0, true));
}
export fn ds4_gpu_store_raw_kv_tensor(
    cache: ?*Tensor,
    x: ?*const Tensor,
    capacity: u32,
    row: u32,
    width: u32,
) c_int {
    return @intFromBool(owner.raw(cache, x, capacity, row, 1, width, false));
}
export fn ds4_gpu_store_raw_kv_batch_tensor(
    cache: ?*Tensor,
    x: ?*const Tensor,
    capacity: u32,
    position: u32,
    rows: u32,
    width: u32,
) c_int {
    return @intFromBool(owner.raw(cache, x, capacity, position, rows, width, true));
}
export fn ds4_gpu_kv_rope_fp8_fuse_available() c_int {
    return @intFromBool(owner.fuseAvailable());
}
export fn ds4_gpu_kv_fp8_store_raw_tensor(
    x: ?*Tensor,
    cache: ?*Tensor,
    capacity: u32,
    row: u32,
    width: u32,
    rotated: u32,
) c_int {
    return @intFromBool((owner.Store{ .input = x, .cache = cache, .capacity = capacity, .row = row, .width = width, .rotated = rotated }).fused(null));
}
export fn ds4_gpu_kv_rope_fp8_store_raw_tensor(
    x: ?*Tensor,
    cache: ?*Tensor,
    capacity: u32,
    row: u32,
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
    const row_bytes = @as(u64, width) * 4;
    const rope = c.ds4_metal_attention_rope_args{
        .row_bytes = row_bytes,
        .token_bytes = row_bytes,
        .head_dim = @bitCast(width),
        .n_dims = @bitCast(rotated),
        .n_ctx_orig = @bitCast(context),
        .inverse = @intFromBool(inverse),
        .pos0 = position,
        .pos_step = 1,
        .freq_base = base,
        .freq_scale = scale,
        .ext_factor = ext,
        .attn_factor = attention,
        .beta_fast = fast,
        .beta_slow = slow,
    };
    return @intFromBool((owner.Store{ .input = x, .cache = cache, .capacity = capacity, .row = row, .width = width, .rotated = rotated }).fused(rope));
}
export fn ds4_gpu_swiglu_tensor(
    out: ?*Tensor,
    gate: ?*const Tensor,
    up: ?*const Tensor,
    count: u32,
    clamp: f32,
    weight: f32,
) c_int {
    return @intFromBool(owner.flat(out, gate, up, null, count, c.DS4_PRIM_SWIGLU, clamp, weight));
}
export fn ds4_gpu_add_tensor(out: ?*Tensor, a: ?*const Tensor, b: ?*const Tensor, count: u32) c_int {
    return @intFromBool(owner.flat(out, a, b, null, count, c.DS4_PRIM_ADD, 0, 0));
}
export fn ds4_gpu_add3_tensor(
    out: ?*Tensor,
    a: ?*const Tensor,
    b: ?*const Tensor,
    third: ?*const Tensor,
    count: u32,
) c_int {
    return @intFromBool(owner.flat(out, a, b, third, count, c.DS4_PRIM_ADD3, 0, 0));
}
export fn ds4_native_metal_copy_1d(
    command: ?*anyopaque,
    source: View,
    out: View,
    count: u32,
    kind: u32,
    zero_ok: bool,
) c_int {
    return @intFromBool(Copy.one(command, source, out, count, kind, zero_ok));
}
export fn ds4_native_metal_copy_3d(
    command: ?*anyopaque,
    source: View,
    out: View,
    columns: u32,
    rows: u32,
    planes: u32,
    source_col: u64,
    source_row: u64,
    source_plane: u64,
    out_row: u64,
    out_plane: u64,
    kind: u32,
) c_int {
    return @intFromBool(Copy.three(command, source, out, .{
        .columns = columns,
        .rows = rows,
        .planes = planes,
        .source_col = source_col,
        .source_row = source_row,
        .source_plane = source_plane,
        .target_row = out_row,
        .target_plane = out_plane,
        .kind = kind,
    }));
}

export fn ds4_native_metal_set_rows(command: ?*anyopaque, source: View, out: View, rows: [*c]const i32, count: u32, out_rows: u32, width: u32) c_int {
    if (command == null or rows == null or count == 0) return 0;
    const K = @import("compressor_ops.zig");
    var borrowed = K.Command{ .token = command, .owned = 0 };
    var ops = K.Ops{ .command = &borrowed };
    defer ops.deinit();
    return @intFromBool(ops.setRows(source, out, rows[0..count], out_rows, width));
}
export fn ds4_native_metal_add_1d(command: ?*anyopaque, a: View, b: View, out: View, count: u32) c_int {
    if (command == null) return 0;
    const K = @import("compressor_ops.zig");
    var borrowed = K.Command{ .token = command, .owned = 0 };
    var ops = K.Ops{ .command = &borrowed };
    defer ops.deinit();
    return @intFromBool(ops.add(a, b, out, count));
}
