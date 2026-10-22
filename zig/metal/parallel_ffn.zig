//! Shared Q8 gate/up and routed pair-SwiGLU overlap in one concurrent encoder.
//! The two explicit levels retain the original ordering and complementary TP
//! split. Reset closes encoded work and clears all references, even on failure.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const commands = @import("command_schedule.zig");
const Buffer = @import("attention_decode.zig").Buffer;
const View = c.ds4_metal_attention_view;
const Pipeline = c.ds4_metal_hc_pipeline;
const Request = c.ds4_metal_parallel_request;
fn bits(value: u32) i32 {
    return @bitCast(value);
}
fn unsigned(value: i32) u64 {
    return @bitCast(@as(i64, value));
}
fn weights(map: ?*const anyopaque, size: u64, offset: u64, bytes: u64) Buffer {
    var inner: u64 = 0;
    const value = api.retainModelRange(map, size, offset, bytes, &inner);
    return .{ .view = .{ .buffer = value, .offset = inner, .bytes = bytes } };
}
fn pipeline(name: [*:0]const u8, nsg: i16) ?*const Pipeline {
    return c.ds4_metal_routed_pipeline(1, name, nsg);
}
fn matrix(input: u32, output: u32) c.ds4_metal_mv_args {
    const row = (@as(u64, input) / 32) * 34;
    return .{
        .ne00 = bits(input),
        .ne01 = bits(output),
        .ne02 = 1,
        .nb00 = 34,
        .nb01 = row,
        .nb02 = row *% output,
        .nb03 = row *% output,
        .ne10 = bits(input),
        .ne11 = 1,
        .ne12 = 1,
        .nb10 = 4,
        .nb11 = @as(u64, input) * 4,
        .nb12 = @as(u64, input) * 4,
        .nb13 = @as(u64, input) * 4,
        .ne0 = bits(output),
        .ne1 = 1,
        .nr0 = 2,
        .r2 = 1,
        .r3 = 1,
    };
}
fn range(size: u64, offset: u64, skip: u64, bytes: u64) bool {
    return offset <= size and skip <= size - offset and bytes <= size - offset - skip;
}
pub fn start(r: *const Request) c_int {
    if (api.init() == 0) return 0;
    const s = commands.state();
    if (s.commands.batch == null or s.parallel.pending or s.commands.concurrent or
        r.gate == null or r.up == null or r.mid == null or r.out == null or
        r.x == null or r.map == null or r.model_dim == 0 or r.shared_dim == 0 or
        r.model_dim & 31 != 0 or !std.math.isFinite(r.clamp) or r.clamp < 0) return 0;
    if (r.split) {
        if (r.selected == null or r.tp_world != 2 or r.tp_rank >= r.tp_world or
            r.experts == 0 or r.experts % r.tp_world != 0 or r.used == 0 or r.used > 64 or
            r.shift_q16 > 65536 or r.shared_dim & 63 != 0) return 0;
    } else {
        if (r.lane_count == 0 or r.lane_offset > r.shared_dim or
            r.lane_count > r.shared_dim - r.lane_offset or
            (r.shared_dim | r.lane_offset | r.lane_count) & 31 != 0) return 0;
    }
    const count = if (r.split) r.shared_dim else r.lane_count;
    const lane = if (r.split) 0 else r.lane_offset;
    const x = Buffer.tensor(r.x);
    defer x.deinit();
    const gate = Buffer.tensor(r.gate);
    defer gate.deinit();
    const up = Buffer.tensor(r.up);
    defer up.deinit();
    const mid = Buffer.tensor(r.mid);
    defer mid.deinit();
    const out = Buffer.tensor(r.out);
    defer out.deinit();
    const ids = if (r.split) Buffer.tensor(r.selected) else Buffer{ .view = std.mem.zeroes(View) };
    defer ids.deinit();
    if (!x.fits(@as(u64, r.model_dim) * 4) or !out.fits(@as(u64, r.model_dim) * 4) or
        !gate.fits(@as(u64, count) * 4) or !up.fits(@as(u64, count) * 4) or
        !mid.fits(@as(u64, count) * 4) or (r.split and !ids.fits(@as(u64, r.used) * 4))) return 0;
    const gate_row = (@as(u64, r.model_dim) / 32) * 34;
    const gate_bytes = @as(u64, count) *% gate_row;
    const skip = @as(u64, lane) *% gate_row;
    const down_bytes = @as(u64, r.model_dim) *% ((@as(u64, r.shared_dim) / 32) * 34);
    if (!range(r.size, r.gate_offset, skip, gate_bytes) or
        !range(r.size, r.up_offset, skip, gate_bytes) or
        !range(r.size, r.down_offset, 0, down_bytes)) return 0;
    // All three wraps happen before testing any result, as in the C path.
    const gate_w = weights(r.map, r.size, r.gate_offset + skip, gate_bytes);
    defer gate_w.deinit();
    const up_w = weights(r.map, r.size, r.up_offset + skip, gate_bytes);
    defer up_w.deinit();
    const down_w = weights(r.map, r.size, r.down_offset, down_bytes);
    defer down_w.deinit();
    if (!gate_w.valid() or !up_w.valid() or !down_w.valid()) return 0;
    const fallback: u64 = if (c.ds4_metal_dense_facts(false) & c.DS4_DENSE_TP_TWO != 0) 2 else 4;
    const nsg = c.ds4_zig_metal_env_u64("DS4_METAL_Q8_MV_NSG", fallback, 1, 8);
    const gate_pso = pipeline(if (r.split) "kernel_dsv4_shared_gate_up_swiglu_q8_0_split" else "kernel_dsv4_shared_gate_up_swiglu_q8_0", @intCast(nsg));
    const down_pso = pipeline(if (r.split) "kernel_dsv4_shared_down_q8_0_split" else "kernel_mul_mv_q8_0_f32", 4);
    if (gate_pso == null or down_pso == null or c.ds4_metal_schedule_pipeline_threads(down_pso) < 128) return 0;

    commands.closeEncoder();
    s.commands.concurrent = true;
    // Publish retained resources before retiring any local view. The fixed
    // install leaf changes only strong fields, not policy or shader geometry.
    const views = [_]View{ gate_w.view, up_w.view, x.view, gate.view, up.view, mid.view, down_w.view, out.view, ids.view };
    c.ds4_metal_schedule_parallel_install(&views, gate_pso, down_pso);
    const p = &s.parallel;
    p.gate_weight_offset = gate_w.view.offset;
    p.up_weight_offset = up_w.view.offset;
    p.gate_x_offset = x.view.offset;
    p.gate_out_offset = gate.view.offset;
    p.up_out_offset = up.view.offset;
    p.mid_out_offset = mid.view.offset;
    p.down_weight_offset = down_w.view.offset +% ((@as(u64, lane) / 32) * 34);
    p.down_x_offset = mid.view.offset;
    p.down_out_offset = out.view.offset;
    p.ids_offset = ids.view.offset;
    p.gate_args = matrix(r.model_dim, count);
    p.down_args = matrix(r.shared_dim, r.model_dim);
    if (!r.split) {
        p.down_args.ne00 = bits(count);
        p.down_args.ne10 = bits(count);
    }
    p.clamp = r.clamp;
    p.nsg = nsg;
    p.nr0 = 2;
    p.shared_bytes = 256;
    p.split = r.split;
    if (r.split) p.split_args = .{
        .tp_rank = bits(r.tp_rank),
        .tp_world = bits(r.tp_world),
        .n_expert = bits(r.experts),
        .n_expert_used = bits(r.used),
        .shared_dim = bits(r.shared_dim),
        .lane_granule = 64,
        .shift_q16 = bits(r.shift_q16),
    };
    p.mode = 2;
    p.stage = 0;
    p.pending = true;
    p.encoded = false;
    return 1;
}
fn admits(command: ?*anyopaque, stage: c_int) bool {
    const s = commands.state();
    return s.parallel.pending and s.parallel.mode == 2 and s.parallel.stage == stage and
        s.commands.concurrent and s.commands.batch != null and command == s.commands.batch;
}
pub fn first(command: ?*anyopaque, routed_mid: ?*anyopaque) c_int {
    if (!admits(command, 0)) return 0;
    const enc = commands.encoder(command, 0) orelse return 0;
    defer commands.drop(enc);
    if (!c.ds4_metal_schedule_is_concurrent(enc)) return 0;
    const p = &commands.state().parallel;
    const groups = (unsigned(p.gate_args.ne0) +% p.nr0 -% 1) / p.nr0;
    c.ds4_metal_schedule_parallel_gate(enc, groups);
    p.stage = 1;
    // Both barrier forms complete all earlier concurrent dispatches; the
    // resource list narrows visibility, not execution. This level break waits
    // for routed pair-SwiGLU and shared gate/up before either down consumer.
    c.ds4_metal_schedule_parallel_barrier(enc, routed_mid);
    return 1;
}
pub fn second(command: ?*anyopaque) c_int {
    if (!admits(command, 1)) return 0;
    const enc = commands.encoder(command, 0) orelse return 0;
    defer commands.drop(enc);
    if (!c.ds4_metal_schedule_is_concurrent(enc)) return 0;
    const p = &commands.state().parallel;
    c.ds4_metal_schedule_parallel_down(enc, (unsigned(p.down_args.ne0) +% 1) / 2);
    // finish closes this concurrent encoder before serial HC. Ordered command
    // encoders supply the final dependency; no extra memory barrier is needed.
    p.stage = 2;
    p.encoded = true;
    return 1;
}
pub fn finish() c_int {
    const s = commands.state();
    const complete = s.parallel.pending and s.parallel.encoded and s.parallel.stage == 2 and s.commands.concurrent;
    commands.resetParallel(true); // also reset an incomplete join
    return @intFromBool(complete);
}
