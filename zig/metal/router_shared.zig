//! Shared-expert Q8 gate/up/SwiGLU, row variants and optional router fusions.
//! Native owns admission, packets and dispatch geometry; shaders are unchanged.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const Dense = @import("dense_matmul.zig");
const Buffer = @import("attention_decode.zig").Buffer;
const Command = @import("attention_decode.zig").Command;
const Tensor = api.Tensor;
const Launch = c.ds4_metal_router_shared_launch;
const View = c.ds4_metal_attention_view;
const max32 = std.math.maxInt(u32);
const max64 = std.math.maxInt(u64);
pub const Mode = enum { store, mid_only, bf16, ext, scalar_rows };
pub const Shared = struct {
    gate: ?*Tensor,
    up: ?*Tensor,
    mid: ?*Tensor,
    map: ?*const anyopaque,
    size: u64,
    gate_offset: u64,
    up_offset: u64,
    width: u64,
    outputs: u64,
    x: ?*const Tensor,
    rows: u64 = 1,
    clamp: f32,
    mode: Mode = .store,
};
fn inRange(size: u64, offset: u64, bytes: u64) bool {
    return offset <= size and bytes <= size - offset;
}
fn weight(map: ?*const anyopaque, size: u64, offset: u64, bytes: u64) Buffer {
    var inner: u64 = 0;
    const value = api.retainModelRange(map, size, offset, bytes, &inner);
    return .{ .view = .{ .buffer = value, .offset = inner, .bytes = bytes } };
}
fn errorLine(line: [*:0]const u8) c_int {
    return api.reportError(line);
}
fn vectorGroups() u64 {
    const fallback: u64 = if (c.ds4_metal_dense_facts(false) & c.DS4_DENSE_TP_TWO != 0) 2 else 4;
    return c.ds4_zig_metal_env_u64("DS4_METAL_Q8_MV_NSG", fallback, 1, 8);
}
fn dispatch(plan: *const Launch, label: [*:0]const u8, unavailable: c_int) c_int {
    var command = Command.init() orelse return unavailable;
    defer command.deinit();
    if (c.ds4_metal_router_shared_encode(command.token.?, plan) == 0) return unavailable;
    return if (command.finish(label) != 0) 1 else -1;
}
fn extName(rows: i16) ?[*:0]const u8 {
    return switch (rows) {
        2 => "kernel_mul_mv_ext_q8_0_pair_swiglu_f32_r1_2",
        3 => "kernel_mul_mv_ext_q8_0_pair_swiglu_f32_r1_3",
        4 => "kernel_mul_mv_ext_q8_0_pair_swiglu_f32_r1_4",
        5 => "kernel_mul_mv_ext_q8_0_pair_swiglu_f32_r1_5",
        else => null,
    };
}
pub fn shared(request: Shared) c_int {
    var r = request;
    // Both row APIs deliberately delegate one row to the ordinary path before
    // their own restrictions (including EXT's width divisibility) are applied.
    if ((r.mode == .ext or r.mode == .scalar_rows) and r.rows == 1) r.mode = .store;
    if (api.init() == 0) return 0;
    const store = r.mode != .mid_only;
    if (r.mid == null or r.x == null or r.map == null or r.rows == 0 or
        (store and (r.gate == null or r.up == null)) or r.width & 31 != 0 or
        r.width > max32 or r.outputs > max32 or !std.math.isFinite(r.clamp) or r.clamp < 0) return 0;
    switch (r.mode) {
        .store, .mid_only, .bf16 => if (r.rows > 128 or
            (r.mode == .bf16 and (r.outputs == 0 or r.outputs % 2 != 0 or r.outputs > 65536))) return 0,
        .ext => {
            if (r.width % 128 != 0 or r.rows > max32) return 0;
            const limit = c.ds4_zig_metal_env_u64("DS4_METAL_Q8_MV_EXT_MAX_TOKENS", 16, 2, 128);
            if (r.rows > limit) return 0;
        },
        .scalar_rows => if (r.rows > max32) return 0,
    }
    const input_row = r.width * 4;
    const output_row = r.outputs * 4;
    if (r.mode == .scalar_rows and ((input_row != 0 and r.rows > max64 / input_row) or
        (output_row != 0 and r.rows > max64 / output_row))) return 0;
    const input_bytes = r.rows *% input_row;
    const output_bytes = r.rows *% output_row;
    const input = Buffer.tensor(r.x);
    defer input.deinit();
    const mid = Buffer.tensor(r.mid);
    defer mid.deinit();
    // Mid-only kernels bind mid into the unused gate/up slots. These borrowed
    // aliases have no second free authority; store variants own their views.
    const gate = if (store) Buffer.tensor(r.gate) else Buffer{ .view = std.mem.zeroes(View) };
    defer gate.deinit();
    const up = if (store) Buffer.tensor(r.up) else Buffer{ .view = std.mem.zeroes(View) };
    defer up.deinit();
    const activation_error: [*:0]const u8 = switch (r.mode) {
        .ext => "ds4: Metal fused Q8_0 gate/up rows received undersized activation buffers",
        .scalar_rows => "ds4: Metal shared expert scalar-row fused gate/up received undersized activation buffers",
        else => "ds4: Metal shared expert fused gate/up received undersized activation buffers",
    };
    if (!input.fits(input_bytes) or !mid.fits(output_bytes) or
        (store and (!gate.fits(output_bytes) or !up.fits(output_bytes)))) return errorLine(activation_error);
    const row = (r.width / 32) * 34;
    // The scalar-row C API checks this multiplication. The zero row is also
    // rejected here; C's otherwise-unchecked division by zero has no result.
    if (r.mode == .scalar_rows and (row == 0 or r.outputs > max64 / row)) return 0;
    const bytes = r.outputs *% row;
    const range_error: [*:0]const u8 = switch (r.mode) {
        .ext => "ds4: Metal fused Q8_0 gate/up rows range is outside the mapped model",
        .scalar_rows => "ds4: Metal shared expert scalar-row fused gate/up range is outside the mapped model",
        else => "ds4: Metal shared expert fused gate/up range is outside the mapped model",
    };
    if (!inRange(r.size, r.gate_offset, bytes) or !inRange(r.size, r.up_offset, bytes)) return errorLine(range_error);
    const gate_w = weight(r.map, r.size, r.gate_offset, bytes);
    defer gate_w.deinit();
    const up_w = weight(r.map, r.size, r.up_offset, bytes);
    defer up_w.deinit();
    if (!gate_w.valid() or !up_w.valid()) return 0;
    var plan = std.mem.zeroes(Launch);
    plan.gate_weight = gate_w.view;
    plan.up_weight = up_w.view;
    plan.input = input.view;
    plan.gate = if (store) gate.view else mid.view;
    plan.up = if (store) up.view else mid.view;
    plan.mid = mid.view;
    plan.clamp = r.clamp;
    plan.kind = c.DS4_SHARED_VECTOR;
    plan.shared = Dense.mvArgs(r.width, r.outputs, r.rows, true, 2);
    plan.groups_x = (r.outputs + 1) / 2;
    plan.groups_y = r.rows;
    plan.threads_x = 32;
    plan.threads_y = if (r.mode == .ext) 2 else vectorGroups();
    plan.shared_bytes = 512;
    if (r.mode == .ext) {
        const facts = c.ds4_metal_dense_facts(false);
        const nxpsg: i16 = if (r.width % 256 == 0 and (r.rows < 3 or (r.rows == 3 and facts & c.DS4_DENSE_VERIFY_EXACT != 0))) 16 else 8;
        const r1 = Dense.extRows(r.rows);
        const name = extName(r1) orelse return 0;
        plan.kind = c.DS4_SHARED_EXT;
        plan.ext = Dense.extArgs(.{ .out = r.mid, .map = r.map, .size = r.size, .offset = r.gate_offset, .width = r.width, .outputs = r.outputs, .x = r.x, .rows = r.rows }, true);
        plan.pipeline = c.ds4_metal_dense_pipeline(c.DS4_DENSE_EXT, name, 2, nxpsg, false, false);
        const r0: u64 = @intCast(@divTrunc(32, nxpsg) * 2);
        const token_rows: u64 = @intCast(r1);
        plan.groups_x = (r.outputs + r0 - 1) / r0;
        plan.groups_y = (r.rows + token_rows - 1) / token_rows;
        plan.threads_y = 2;
        plan.shared_bytes = 0;
    } else {
        const name: [*:0]const u8 = if (r.mode == .bf16) "kernel_dsv41_shared_gate_up_bf16_q8_0" else if (store) "kernel_dsv4_shared_gate_up_swiglu_q8_0" else "kernel_dsv4_shared_mid_swiglu_q8_0";
        plan.pipeline = c.ds4_metal_routed_pipeline(1, name, @intCast(plan.threads_y));
    }
    if (plan.pipeline == null) return 0;
    const label: [*:0]const u8 = switch (r.mode) {
        .ext => "fused Q8_0 gate/up rows",
        .scalar_rows => "shared expert scalar-row fused gate/up",
        else => if (store) "shared expert fused gate/up" else "shared expert fused mid",
    };
    const result = dispatch(&plan, label, 0);
    return if (result < 0) 0 else result;
}
pub const RouterShared = struct { shared: Shared, logits: ?*Tensor, router_offset: u64, router_outputs: u64, router_only: bool };
/// Decode-only router-logit matvec plus shared SwiGLU on the same normalized
/// input. Return0 for unsupported shape and-1 for an admitted path's error.
pub fn routerShared(r: RouterShared) c_int {
    if (api.init() == 0) return -1;
    const s = r.shared;
    if (r.logits == null or s.gate == null or s.up == null or s.mid == null or s.x == null or s.map == null or
        s.width & 31 != 0 or s.width > max32 or r.router_outputs > max32 or s.outputs > max32 or
        r.router_outputs & 1 != 0 or s.outputs & 3 != 0 or !std.math.isFinite(s.clamp) or s.clamp < 0) return -1;
    if (s.width != 4096) return 0;
    const router_bytes = s.width *% 2 *% r.router_outputs;
    const shared_bytes = (s.width / 32) * 34 *% s.outputs;
    if (!inRange(s.size, r.router_offset, router_bytes) or !inRange(s.size, s.gate_offset, shared_bytes) or !inRange(s.size, s.up_offset, shared_bytes)) return -1;
    const input = Buffer.tensor(s.x);
    defer input.deinit();
    const logits = Buffer.tensor(r.logits);
    defer logits.deinit();
    const gate = Buffer.tensor(s.gate);
    defer gate.deinit();
    const up = Buffer.tensor(s.up);
    defer up.deinit();
    const mid = Buffer.tensor(s.mid);
    defer mid.deinit();
    if (!input.fits(s.width * 4) or !logits.fits(r.router_outputs * 4) or
        !gate.fits(s.outputs * 4) or !up.fits(s.outputs * 4) or !mid.fits(s.outputs * 4)) return -1;
    const router_w = weight(s.map, s.size, r.router_offset, router_bytes);
    defer router_w.deinit();
    const gate_w = weight(s.map, s.size, s.gate_offset, shared_bytes);
    defer gate_w.deinit();
    const up_w = weight(s.map, s.size, s.up_offset, shared_bytes);
    defer up_w.deinit();
    if (!router_w.valid() or !gate_w.valid() or !up_w.valid()) return -1;
    var plan = std.mem.zeroes(Launch);
    plan.kind = c.DS4_ROUTER_SHARED;
    plan.pipeline = c.ds4_metal_routed_pipeline(0, "kernel_dsv4_router_shared_gate_up_q8_0", 0);
    if (plan.pipeline == null) return -1;
    plan.router = Dense.mvArgs(s.width, r.router_outputs, 1, false, 2);
    plan.shared = Dense.mvArgs(s.width, s.outputs, 1, true, 2);
    plan.router_weight = router_w.view;
    plan.gate_weight = gate_w.view;
    plan.up_weight = up_w.view;
    plan.input = input.view;
    plan.logits = logits.view;
    plan.gate = gate.view;
    plan.up = up.view;
    plan.mid = mid.view;
    plan.clamp = s.clamp;
    plan.groups_x = (r.router_outputs + 1) / 2 + (if (r.router_only) 0 else (s.outputs / 2 + 1) / 2);
    plan.groups_y = 1;
    plan.threads_x = 32;
    plan.threads_y = 8;
    plan.shared_bytes = 1024;
    return dispatch(&plan, "router + shared gate/up fused", -1);
}
pub const Project = struct {
    logits: ?*Tensor,
    probs: ?*Tensor,
    selected: ?*Tensor,
    weights: ?*Tensor,
    map: ?*const anyopaque,
    size: u64,
    router_offset: u64,
    bias_offset: u64,
    has_bias: bool,
    x: ?*const Tensor,
};
pub fn project(r: Project) c_int {
    if (api.init() == 0) return 0;
    if (r.logits == null or r.probs == null or r.selected == null or r.weights == null or r.map == null or r.x == null) return 0;
    if (!inRange(r.size, r.router_offset, 4096 * 2 * 256) or (r.has_bias and !inRange(r.size, r.bias_offset, 256 * 4))) return 0;
    const input = Buffer.tensor(r.x);
    defer input.deinit();
    const logits = Buffer.tensor(r.logits);
    defer logits.deinit();
    const probs = Buffer.tensor(r.probs);
    defer probs.deinit();
    const selected = Buffer.tensor(r.selected);
    defer selected.deinit();
    const weights = Buffer.tensor(r.weights);
    defer weights.deinit();
    if (!input.fits(4096 * 4) or !logits.fits(256 * 4) or !probs.fits(256 * 4) or !selected.fits(6 * 4) or !weights.fits(6 * 4)) return 0;
    const router_w = weight(r.map, r.size, r.router_offset, 4096 * 2 * 256);
    defer router_w.deinit();
    const bias = if (r.has_bias) weight(r.map, r.size, r.bias_offset, 256 * 4) else Buffer{ .view = std.mem.zeroes(View) };
    defer bias.deinit();
    if (!router_w.valid() or (r.has_bias and !bias.valid())) return 0;
    const pso = c.ds4_metal_routed_pipeline(0, "kernel_dsv4_router_project_select_fused", 0);
    if (pso == null or c.ds4_metal_routed_max_threads(pso) < 256) return 0;
    const completion = Buffer{ .view = c.ds4_metal_router_completion(logits.view) };
    defer completion.deinit();
    if (!completion.valid()) return 0;
    var plan = std.mem.zeroes(Launch);
    plan.kind = c.DS4_ROUTER_PROJECT;
    plan.pipeline = pso;
    plan.router = Dense.mvArgs(4096, 256, 1, false, 2);
    plan.select.has_bias = @intFromBool(r.has_bias);
    plan.router_weight = router_w.view;
    plan.bias = bias.view;
    plan.has_bias = r.has_bias;
    plan.completion = completion.view;
    plan.input = input.view;
    plan.logits = logits.view;
    plan.probs = probs.view;
    plan.selected = selected.view;
    plan.weights = weights.view;
    plan.groups_x = 128;
    plan.groups_y = 1;
    plan.threads_x = 256;
    plan.threads_y = 1;
    plan.shared_bytes = 4096;
    return dispatch(&plan, "router project/select fused", 0);
}
