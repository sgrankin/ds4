//! Decode-only q/kv RMS norm + KV RoPE tail + FP8/raw publication. Shader
//! arithmetic, order and rounding match the separate dispatches. The q task
//! can run now and the KV task later under q_b; unconsumed work flushes alone.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const commands = @import("command_schedule.zig");
const Buffer = @import("attention_decode.zig").Buffer;
const View = c.ds4_metal_attention_view;
const Launch = c.ds4_metal_kv_launch;
const Request = c.ds4_metal_fused_kv_request;
fn bits(value: u32) i32 {
    return @bitCast(value);
}
fn rmsThreads(width: u32) u64 {
    const vectors = width / 4;
    var threads: u64 = 32;
    while (threads < vectors and threads < 1024) threads *= 2;
    return @max(@min(threads, vectors), 1);
}
fn pipeline() ?*const c.ds4_metal_hc_pipeline {
    return c.ds4_metal_routed_pipeline(0, "kernel_dsv4_qkv_rms_norm_kv_rope_fp8_store_f32", 0);
}
fn weight(map: ?*const anyopaque, size: u64, offset: u64, bytes: u64) Buffer {
    var inner: u64 = 0;
    const retained = api.retainModelRange(map, size, offset, bytes, &inner);
    return .{ .view = .{ .buffer = retained, .offset = inner, .bytes = bytes } };
}
fn at(buffer: ?*anyopaque, offset: u64) View {
    return .{ .buffer = @ptrCast(buffer), .offset = offset, .bytes = 0 };
}
fn deferred() Launch {
    const task = &commands.state().kv;
    return .{
        .norm = task.nargs,
        .rope = task.rope,
        .store = task.store,
        .buffers = .{
            at(task.q, task.q_off),     at(task.q_w, task.q_w_off),   at(task.q_out, task.q_out_off),
            at(task.kv, task.kv_off),   at(task.kv_w, task.kv_w_off), at(task.kv_out, task.kv_out_off),
            at(task.raw, task.raw_off),
        },
        .threads = task.norm_threads,
        .tasks = task.nargs.tasks,
        .groups_y = 2,
    };
}
pub fn deferNext() void {
    commands.state().defer_kv = 1;
}

/// The full kernel's q task recomputes identical values if no staging consumed
/// the KV task. Clear pending before acquisition, including failure paths.
pub fn flush() c_int {
    if (commands.state().kv.pending == 0) return 1;
    commands.state().kv.pending = 0;
    var owned: c_int = 0;
    const command = commands.acquire(&owned) orelse return 0;
    var success: c_int = 0;
    defer {
        if (success == 0) _ = commands.release(command, owned, 0, "deferred kv norm task", 0);
    }
    const pso = pipeline() orelse return 0;
    const enc = commands.encoder(command, 0);
    defer commands.drop(enc);
    const launch = deferred();
    c.ds4_metal_schedule_kv_encode(enc, pso, &launch);
    commands.endEncoder(command, enc);
    success = 1;
    return commands.release(command, owned, 1, "deferred kv norm task", 0);
}

/// Leave the concurrent encoder open for q_b. On encoder creation failure,
/// restore pending so the ordinary staging/flush safety path can consume it.
pub fn beginConcurrent() c_int {
    const s = commands.state();
    if (s.kv.pending == 0 or s.commands.batch == null or s.commands.concurrent) return 0;
    const pso = pipeline() orelse return 0;
    s.kv.pending = 0;
    commands.closeEncoder();
    s.commands.concurrent = true;
    const enc = commands.encoder(s.commands.batch, 0);
    defer commands.drop(enc);
    if (enc == null or !c.ds4_metal_schedule_is_concurrent(enc)) {
        commands.closeEncoder();
        s.commands.concurrent = false;
        s.kv.pending = 1;
        return 0;
    }
    const launch = deferred();
    c.ds4_metal_schedule_kv_encode(enc, pso, &launch);
    commands.endEncoder(s.commands.batch, enc);
    return 1;
}
pub fn endConcurrent() void {
    if (!commands.state().commands.concurrent) return;
    commands.closeEncoder();
    commands.state().commands.concurrent = false;
}
pub fn fused(r: *const Request) c_int {
    if (api.init() == 0) return 0;
    if (r.q_out == null or r.q == null or r.kv_out == null or r.kv == null or r.raw == null or
        r.q_n == 0 or r.kv_n == 0 or (r.q_n | r.kv_n) & 3 != 0 or
        r.n_rot > r.kv_n or r.n_rot & 1 != 0) return 0;
    const q = Buffer.tensor(r.q);
    defer q.deinit();
    const q_out = Buffer.tensor(r.q_out);
    defer q_out.deinit();
    const kv = Buffer.tensor(r.kv);
    defer kv.deinit();
    const kv_out = Buffer.tensor(r.kv_out);
    defer kv_out.deinit();
    const raw = Buffer.tensor(r.raw);
    defer raw.deinit();
    const q_bytes = @as(u64, r.q_n) * 4;
    const kv_bytes = @as(u64, r.kv_n) * 4;
    // C's raw-capacity byte product is unsigned modular arithmetic.
    if (!q.fits(q_bytes) or !q_out.fits(q_bytes) or !kv.fits(kv_bytes) or
        !kv_out.fits(kv_bytes) or !raw.fits(r.raw_cap *% kv_bytes) or r.raw_row >= r.raw_cap)
        return api.reportError("ds4: Metal fused q/kv norm RoPE store received undersized buffers");
    if (r.q_weight_offset > r.size or q_bytes > r.size - r.q_weight_offset or
        r.kv_weight_offset > r.size or kv_bytes > r.size - r.kv_weight_offset)
        return api.reportError("ds4: Metal fused q/kv norm RoPE store weight range is outside the mapped model");
    const q_w = weight(r.map, r.size, r.q_weight_offset, q_bytes);
    defer q_w.deinit();
    if (!q_w.valid()) return 0;
    const kv_w = weight(r.map, r.size, r.kv_weight_offset, kv_bytes);
    defer kv_w.deinit();
    if (!kv_w.valid()) return 0;
    const pso = pipeline() orelse return 0;
    var launch = std.mem.zeroes(Launch);
    launch.norm = .{
        .q_n = bits(r.q_n),
        .q_n4 = bits(r.q_n / 4),
        .kv_n = bits(r.kv_n),
        .kv_n4 = bits(r.kv_n / 4),
        .q_row_stride = q_bytes,
        .kv_row_stride = kv_bytes,
        .eps = r.eps,
        .tasks = 0,
    };
    launch.rope = .{
        .row_bytes = kv_bytes,
        .token_bytes = kv_bytes,
        .head_dim = bits(r.kv_n),
        .n_dims = bits(r.n_rot),
        .n_ctx_orig = bits(r.n_ctx_orig),
        .inverse = 0,
        .pos0 = r.pos0,
        .pos_step = 1,
        .freq_base = r.freq_base,
        .freq_scale = r.freq_scale,
        .ext_factor = r.ext_factor,
        .attn_factor = r.attn_factor,
        .beta_fast = r.beta_fast,
        .beta_slow = r.beta_slow,
    };
    launch.store = .{ .head_dim = bits(r.kv_n), .n_rot = bits(r.n_rot), .raw_row = bits(r.raw_row) };
    launch.buffers = .{ q.view, q_w.view, q_out.view, kv.view, kv_w.view, kv_out.view, raw.view };
    launch.threads = rmsThreads(r.q_n);
    var owned: c_int = 0;
    const command = commands.acquire(&owned) orelse return 0;
    const enc = commands.encoder(command, 0);
    defer commands.drop(enc);
    const s = commands.state();
    const defer_kv = s.defer_kv != 0 and s.kv.pending == 0;
    s.defer_kv = 0;
    launch.tasks = if (defer_kv) 1 else 0;
    launch.groups_y = if (defer_kv) 1 else 2;
    c.ds4_metal_schedule_kv_encode(enc, pso, &launch);
    if (defer_kv) {
        const task = &s.kv;
        task.pending = 1;
        c.ds4_metal_schedule_kv_install(&launch.buffers);
        task.q_off = q.view.offset;
        task.q_w_off = q_w.view.offset;
        task.q_out_off = q_out.view.offset;
        task.kv_off = kv.view.offset;
        task.kv_w_off = kv_w.view.offset;
        task.kv_out_off = kv_out.view.offset;
        task.raw_off = raw.view.offset;
        task.nargs = launch.norm;
        task.nargs.tasks = 2; // KV task only
        task.rope = launch.rope;
        task.store = launch.store;
        task.norm_threads = launch.threads;
    }
    commands.endEncoder(command, enc);
    return commands.release(command, owned, 1, "fused q/kv norm RoPE store", 0);
}
