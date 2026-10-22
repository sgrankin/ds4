//! Ordinary FP8/QAT, raw-ring publication and flat add/SwiGLU transactions.
//! Row scatter composes the complete Compressor operation; packets are shared
//! actual C records. Driver leaves own ARC handles, not host policy.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const K = @import("compressor_ops.zig");
const Buffer = K.Buffer;
const View = K.View;
const Copy = @import("copy.zig");
const max32 = std.math.maxInt(u32);
fn report(line: [*:0]const u8) bool {
    return K.errorLine(line);
}
fn plan(kind: u32) c.ds4_metal_primitive_launch {
    var launch = std.mem.zeroes(c.ds4_metal_primitive_launch);
    launch.kind = kind;
    launch.pipeline = c.ds4_metal_primitive_pipeline(kind);
    launch.groups = 1;
    launch.threads = 64;
    launch.shared = 256;
    return launch;
}
fn encode(command: *const K.Command, launch: *const c.ds4_metal_primitive_launch) bool {
    return c.ds4_metal_primitive_encode(command.token, launch) != 0;
}
pub fn fp8(input: ?*api.Tensor, rows: u32, width: u32, rotated: u32, qat: bool) bool {
    if (api.init() == 0 or input == null or rows == 0 or width == 0 or (if (qat) width != 128 else rotated > width)) return false;
    if (!qat and rotated == width) return true;
    const x = Buffer.tensor(input);
    defer x.deinit();
    if (!x.fits(@as(u64, rows) *% width *% 4)) return report(if (qat) "ds4: Metal DSV4 indexer QAT received undersized activation buffer" else "ds4: Metal DSV4 FP8 KV quantize received undersized activation buffer");
    var launch = plan(if (qat) c.DS4_PRIM_QAT else c.DS4_PRIM_FP8);
    launch.input = x.view;
    launch.groups = rows;
    if (qat) {
        launch.args.qat = .{ .n_rows = rows, .head_dim = width, .row_stride = @as(u64, width) * 4 };
        launch.threads = 128;
        launch.shared = 1024;
    } else {
        const row = @as(u64, width) * 4;
        const bytes = @as(u64, rows) *% row;
        launch.args.fp8 = .{
            .ne00 = width,
            .ne01 = rows,
            .ne02 = 1,
            .ne03 = 1,
            .nb00 = 4,
            .nb01 = row,
            .nb02 = bytes,
            .nb03 = bytes,
            .nb0 = 4,
            .nb1 = row,
            .nb2 = bytes,
            .nb3 = bytes,
            .n_rot = K.narrow(rotated),
        };
    }
    var command: K.Command = .{};
    defer command.drop();
    return command.acquire() and encode(&command, &launch) and command.finish(if (qat) "DSV4 indexer Hadamard+FP4" else "DSV4 FP8 KV quantize");
}
pub fn fuseAvailable() bool {
    if (api.init() == 0 or K.present("DS4_METAL_DISABLE_INPLACE_ROPE_PAIR") or K.present("DS4_METAL_DISABLE_AFFINE_ROPE_PAIR")) return false;
    const facts = c.ds4_metal_rope_facts();
    return c.ds4_metal_primitive_pipeline(c.DS4_PRIM_ROPE_STORE).handle != null and facts & c.DS4_ROPE_AFFINE_AVAILABLE != 0 and facts & (c.DS4_ROPE_M3 | c.DS4_ROPE_M5) != 0;
}
pub const Store = struct {
    input: ?*api.Tensor,
    cache: ?*api.Tensor,
    capacity: u32,
    row: u32,
    width: u32,
    rotated: u32,
    pub fn fused(self: Store, rope: ?c.ds4_metal_attention_rope_args) bool {
        if (api.init() == 0 or self.input == null or self.cache == null or self.capacity == 0 or self.row >= self.capacity or self.width == 0 or self.rotated > self.width or self.capacity > std.math.maxInt(i32)) return false;
        var launch = plan(if (rope != null) c.DS4_PRIM_ROPE_STORE else c.DS4_PRIM_FP8_STORE);
        if (rope != null and (self.rotated & 1 != 0 or launch.pipeline.handle == null)) return false;
        const x = Buffer.tensor(self.input);
        defer x.deinit();
        const out = Buffer.tensor(self.cache);
        defer out.deinit();
        if (!x.fits(@as(u64, self.width) * 4) or !out.fits(@as(u64, self.capacity) *% self.width *% 4)) return report(if (rope != null) "ds4: Metal fused KV RoPE/FP8 store received undersized buffers" else "ds4: Metal fused KV FP8/raw-store received undersized buffers");
        launch.args.store = .{ .head_dim = K.narrow(self.width), .n_rot = K.narrow(self.rotated), .raw_row = K.narrow(self.row) };
        launch.rope = rope orelse std.mem.zeroes(c.ds4_metal_attention_rope_args);
        launch.input = x.view;
        launch.out = out.view;
        var command: K.Command = .{};
        defer command.drop();
        return command.acquire() and encode(&command, &launch) and command.finish(if (rope != null) "KV RoPE/FP8 store fused" else "KV FP8/raw-store fused");
    }
};
fn round(command: *K.Command, source: View, count: u32, half: View, rounded: View) bool {
    if (source.buffer == null or source.bytes < @as(u64, count) * 4) return report("ds4: Metal raw KV store received undersized source buffer");
    return Copy.one(command.token, source, half, count, c.DS4_PRIM_COPY_F32_F16, false) and Copy.one(command.token, half, rounded, count, c.DS4_PRIM_COPY_F16_F32, false);
}
pub fn raw(
    cache: ?*api.Tensor,
    input: ?*const api.Tensor,
    capacity: u32,
    position: u32,
    count: u32,
    width: u32,
    batch: bool,
) bool {
    if (api.init() == 0 or cache == null or input == null or capacity == 0 or count == 0 or width == 0 or capacity > std.math.maxInt(i32) or (!batch and position >= capacity)) return false;
    const out = Buffer.tensor(cache);
    defer out.deinit();
    if (out.view.bytes < @as(u64, capacity) *% width *% 4) return report(if (batch) "ds4: Metal raw KV batch store received undersized destination buffer" else "ds4: Metal raw KV store received undersized destination buffer");
    var stack: [512]i32 = undefined;
    const indices = if (count <= stack.len) stack[0..count] else std.heap.c_allocator.alloc(i32, count) catch return report("ds4: failed to allocate raw KV set_rows index list");
    defer if (count > stack.len) std.heap.c_allocator.free(indices);
    for (indices, 0..) |*row, index| row.* = @intCast((position +% @as(u32, @intCast(index))) % capacity);
    var command: K.Command = .{};
    defer command.drop();
    if (!command.acquire()) return false;
    // Preserve original order: batch count overflow is checked after acquire;
    // source validation and scratch allocation follow that admission.
    const elements = @as(u64, count) * width;
    if (elements > max32) return false;
    const source = Buffer.tensor(input);
    defer source.deinit();
    if (!source.fits(elements * 4)) return report("ds4: Metal raw KV store received undersized source buffer");
    const half = Buffer{ .view = c.ds4_metal_primitive_round_scratch(true, elements * 2) };
    defer half.deinit();
    if (!half.valid()) return false;
    const rounded = Buffer{ .view = c.ds4_metal_primitive_round_scratch(false, elements * 4) };
    defer rounded.deinit();
    if (!rounded.valid()) return false;
    var ops = K.Ops{ .command = &command };
    defer ops.deinit();
    return round(&command, source.view, @intCast(elements), half.view, rounded.view) and ops.setRows(rounded.view, out.view, indices, capacity, width) and command.finish(if (batch) "raw KV batch DS4 set_rows store" else "raw KV DS4 set_rows store");
}
pub fn flat(
    out_tensor: ?*api.Tensor,
    a_tensor: ?*const api.Tensor,
    b_tensor: ?*const api.Tensor,
    third_tensor: ?*const api.Tensor,
    count: u32,
    kind: u32,
    clamp: f32,
    weight: f32,
) bool {
    @setFloatMode(.strict);
    if (api.init() == 0 or out_tensor == null or a_tensor == null or b_tensor == null or count == 0 or (kind == c.DS4_PRIM_ADD3 and third_tensor == null)) return false;
    if (kind == c.DS4_PRIM_SWIGLU and (!std.math.isFinite(clamp) or clamp < 0 or !std.math.isFinite(weight))) return false;
    const a = Buffer.tensor(a_tensor);
    defer a.deinit();
    const b = Buffer.tensor(b_tensor);
    defer b.deinit();
    const third = Buffer.tensor(third_tensor);
    defer third.deinit();
    const out = Buffer.tensor(out_tensor);
    defer out.deinit();
    const bytes = @as(u64, count) * 4;
    if (!a.fits(bytes) or !b.fits(bytes) or !out.fits(bytes) or (kind == c.DS4_PRIM_ADD3 and !third.fits(bytes))) return report(if (kind == c.DS4_PRIM_SWIGLU) "ds4: Metal SwiGLU received undersized buffers" else if (kind == c.DS4_PRIM_ADD3) "ds4: Metal tensor add3 received undersized buffers" else "ds4: Metal tensor add received undersized buffers");
    var launch = plan(kind);
    launch.input = a.view;
    launch.other = b.view;
    launch.third = third.view;
    launch.out = out.view;
    launch.shared = 0;
    launch.threads = @max(1, @min(count, 256, launch.pipeline.threads));
    launch.groups = (@as(u64, count) + launch.threads - 1) / launch.threads;
    if (kind == c.DS4_PRIM_SWIGLU) launch.args.glu = .{
        .ne00 = K.narrow(count),
        .nb01 = bytes,
        .ne10 = K.narrow(count),
        .nb11 = bytes,
        .ne0 = K.narrow(count),
        .nb1 = bytes,
        .i00 = 0,
        .i10 = 0,
        .alpha = weight,
        .limit = clamp,
    } else launch.args.add = .{ .n = count };
    var command: K.Command = .{};
    defer command.drop();
    return command.acquire() and encode(&command, &launch) and command.finish(if (kind == c.DS4_PRIM_SWIGLU) "SwiGLU" else if (kind == c.DS4_PRIM_ADD3) "tensor add3" else "tensor add");
}
