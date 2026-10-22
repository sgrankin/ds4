//! TP gate/fold and selected-readback control. The service queue/transport,
//! actual events and ARC command storage remain shared libraries; policy and
//! publication order are native. This has no second-host qualification claim.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const Buffer = @import("attention_decode.zig").Buffer;
const Command = @import("routed_batch.zig").Command;
const Schedule = @import("command_schedule.zig");
const Prefetch = @import("tp_prefetch.zig");
const Policy = c.ds4_metal_tp_policy;
const Facts = c.ds4_metal_tp_facts;
const View = c.ds4_metal_attention_view;
const Launch = c.ds4_metal_tp_launch;
const max_slots = c.DS4_TP_POLL_MAX_SLOTS;
const gate_count: u32 = c.DS4_GPU_TP_GATES_PER_LAYER;
var request_debug: i32 = -1;
var take_debug: i32 = -1;
fn state() *Policy {
    return @ptrCast(c.ds4_metal_tp_policy_state());
}
fn facts() Facts {
    return c.ds4_metal_tp_get_facts();
}
fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn at(view: View, offset: u64) View {
    var result = view;
    result.offset +%= offset;
    return result;
}
fn errorLine(line: [*:0]const u8) bool {
    _ = api.reportError(line);
    return false;
}
fn pipeline(name: [*:0]const u8) ?*const c.ds4_metal_hc_pipeline {
    return c.ds4_metal_routed_pipeline(0, name, 0);
}
fn slotFor(layer: u32, gate: u32) u32 {
    return layer *% gate_count +% gate;
}
fn foldOk(s: *const Policy, f: Facts, slot: u32, bytes: u64) bool {
    return f.initialized and f.batch and f.service_running and s.session_batch_mode == 0 and s.flag_gates and s.poll_gates and !f.ssd and f.checks and slot < max_slots and s.vector_bytes == bytes and !present("DS4_TP_DISABLE_FLAG_FOLD");
}
fn payloadOffset(s: *const Policy, slot: u32) u64 {
    return s.slab_offset +% s.out_offset +% @as(u64, slot) *% s.vector_bytes;
}
pub fn requestFold(layer: u32, gate: u32) void {
    const s = state();
    const f = facts();
    s.fold_requested = 0;
    const slot = slotFor(layer, gate);
    if (request_debug < 0) request_debug = if (present("DS4_TP_FOLD_DEBUG")) 8 else 0;
    if (request_debug > 0) {
        request_debug -= 1;
        _ = c.fprintf(c.__stderrp, "ds4: fold request slot %u ok=%d init=%d cb=%d thr=%d batch=%d flag=%d poll=%d ssd=%d chk=%d vec=%llu\n", slot, @as(c_int, @intFromBool(foldOk(s, f, slot, s.vector_bytes))), @as(c_int, @intFromBool(f.initialized)), @as(c_int, @intFromBool(f.batch)), @as(c_int, @intFromBool(f.service_running)), s.session_batch_mode, @as(c_int, @intFromBool(s.flag_gates)), @as(c_int, @intFromBool(s.poll_gates)), @as(c_int, @intFromBool(f.ssd)), @as(c_int, @intFromBool(f.checks)), @as(c_ulonglong, s.vector_bytes));
    }
    if (!foldOk(s, f, slot, s.vector_bytes) or s.vector_bytes == 0) return;
    s.fold_requested = 1;
    s.fold_layer = layer;
    s.fold_gate = gate;
}
/// Returns the requested slot only for its exact partial-output view.
pub fn takeFold(out: ?*const api.Tensor, bytes: u64, slot_out: *u32, value_out: *u32) bool {
    const s = state();
    if (s.fold_requested == 0 or out == null) return false;
    const f = facts();
    const slot = slotFor(s.fold_layer, s.fold_gate);
    const output = Buffer.tensor(out);
    defer output.deinit();
    if (take_debug < 0) take_debug = if (present("DS4_TP_FOLD_DEBUG")) 8 else 0;
    if (take_debug > 0) {
        take_debug -= 1;
        _ = c.fprintf(c.__stderrp, "ds4: fold take slot %u ok=%d buf_eq=%d off=%llu want=%llu bytes=%llu vec=%llu\n", slot, @as(c_int, @intFromBool(foldOk(s, f, slot, bytes))), @as(c_int, @intFromBool(output.view.buffer == f.slab.buffer)), @as(c_ulonglong, output.view.offset), @as(c_ulonglong, payloadOffset(s, slot)), @as(c_ulonglong, bytes), @as(c_ulonglong, s.vector_bytes));
    }
    if (!foldOk(s, f, slot, bytes) or output.view.buffer != f.slab.buffer or output.view.offset != payloadOffset(s, slot) or c.ds4_metal_tp_fold_control().buffer == null) return false;
    s.fold_requested = 0;
    slot_out.* = slot;
    value_out.* = @truncate(s.row_sequence +% 1);
    return true;
}
pub fn addFold(out: ?*api.Tensor, a: ?*const api.Tensor, b: ?*const api.Tensor, count: u32, layer: u32, gate: u32) bool {
    const s = state();
    const f = facts();
    const slot = slotFor(layer, gate);
    const output = Buffer.tensor(out);
    defer output.deinit();
    if (!foldOk(s, f, slot, @as(u64, count) * 4) or out == null or a == null or b == null or count == 0 or count % 256 != 0 or output.view.buffer != f.slab.buffer or output.view.offset != payloadOffset(s, slot)) return c.ds4_gpu_add_tensor(out, a, b, count) != 0;
    const control = c.ds4_metal_tp_fold_control();
    if (control.buffer == null) return c.ds4_gpu_add_tensor(out, a, b, count) != 0;
    const pso = pipeline("kernel_dsv4_add2_f32_tp_flag_checked");
    if (pso == null) return c.ds4_gpu_add_tensor(out, a, b, count) != 0;
    const lhs = Buffer.tensor(a);
    defer lhs.deinit();
    const rhs = Buffer.tensor(b);
    defer rhs.deinit();
    if (!lhs.valid() or !rhs.valid()) return c.ds4_gpu_add_tensor(out, a, b, count) != 0;
    var command = Command{};
    defer command.drop();
    if (!command.acquire()) return false;
    var launch = std.mem.zeroes(Launch);
    launch.kind = c.DS4_TP_FOLD_ADD;
    launch.pipeline = pso;
    launch.count = count;
    launch.value = @truncate(s.row_sequence +% 1);
    launch.groups = count / 256;
    launch.threads = 256;
    launch.shared_bytes = 32;
    launch.end_encoder = false;
    launch.a = lhs.view;
    launch.b = rhs.view;
    launch.out = output.view;
    launch.flags = at(f.slab, s.slab_offset +% s.flags_offset +% @as(u64, slot) * 4);
    launch.checks = at(f.checks_buffer, @as(u64, slot) * 4);
    launch.control = at(control, @as(u64, slot) * 8);
    if (c.ds4_metal_tp_encode(command.token.?, &launch) == 0) return false;
    s.prepublished_sequence = s.row_sequence +% 1;
    return true;
}
fn enqueue(layer: u32, gate: u32, rows: u32, sequence: u64, poll: bool, event: bool, out: ?*const anyopaque, input: ?*anyopaque, bytes: u64) bool {
    const request = c.ds4_gpu_tp_request{ .layer = layer, .gate = gate, .rows = rows, .event_arrival = @intFromBool(event), .seq = sequence, .big_out = out, .big_in = input, .big_bytes = bytes, .poll = @intFromBool(poll) };
    return c.ds4_metal_tp_enqueue(&request);
}
fn flag(command: *const Command, s: *const Policy, f: Facts, slot: u32, sequence: u64, checked: bool) bool {
    const pso = pipeline(if (checked) "kernel_dsv4_tp_flag_set_checked" else "kernel_dsv4_tp_flag_set");
    if (pso == null) return false;
    var launch = std.mem.zeroes(Launch);
    launch.kind = if (checked) c.DS4_TP_CHECKED_FLAG else c.DS4_TP_FLAG;
    launch.pipeline = pso;
    launch.value = @truncate(sequence);
    launch.words = @truncate(s.vector_bytes / 4);
    launch.groups = 1;
    launch.threads = if (checked) 256 else 1;
    launch.shared_bytes = if (checked) 32 else 0;
    launch.end_encoder = true;
    launch.flags = at(f.slab, s.slab_offset +% s.flags_offset +% @as(u64, slot) * 4);
    launch.checks = at(f.checks_buffer, @as(u64, slot) * 4);
    launch.payload = at(f.slab, payloadOffset(s, slot));
    if (c.ds4_metal_tp_encode(command.token.?, &launch) == 0) return false;
    Schedule.closeEncoder();
    return true;
}
pub fn rowGate(layer: u32, which: u32) bool {
    const s = state();
    const f = facts();
    if (!f.batch) {
        _ = c.fprintf(c.__stderrp, "ds4: TP gate encode without an open command batch (layer %u gate %u)\n", layer, which);
        return false;
    }
    if (!f.service_running) {
        _ = c.fprintf(c.__stderrp, "ds4: TP gate encode without the gate service (layer %u)\n", layer);
        return false;
    }
    s.fold_requested = 0;
    s.row_sequence +%= 1;
    const sequence = s.row_sequence;
    const event = s.session_batch_mode != 0 or !s.flag_gates;
    const slot = slotFor(layer, which);
    if (!event) {
        // The payload hazard orders flag publication after its output stores.
        var command = Command{};
        defer command.drop();
        if (!command.acquire() or command.owned != 0) return false;
        const checked = s.poll_gates and !f.ssd and f.checks and slot < max_slots and s.vector_bytes != 0;
        const published = checked and s.prepublished_sequence == sequence;
        s.prepublished_sequence = 0;
        if (published) Schedule.closeEncoder() else if (!flag(&command, s, f, slot, sequence, checked)) return false;
    } else {
        Schedule.closeEncoder();
        c.ds4_metal_tp_signal(c.DS4_TP_EVENT_ROW, sequence);
    }
    const poll = !event and s.poll_gates and !f.ssd and slot < max_slots;
    if (poll) {
        // Flush publishes CPU-visible payload; the next segment starts with the
        // poll. Keep only a bounded runway to avoid firmware scheduling delay.
        if (c.ds4_gpu_flush_commands() == 0) return false;
        const next = facts();
        if (next.pending_count > c.DS4_TP_POLL_MAX_INFLIGHT) c.ds4_metal_tp_wait_pending_index(next.pending_count - 1 - c.DS4_TP_POLL_MAX_INFLIGHT);
        _ = Prefetch.encode(which);
        const pso = pipeline("kernel_dsv4_tp_poll_release");
        if (pso == null) return false;
        var command = Command{};
        defer command.drop();
        if (!command.acquire()) return false;
        var launch = std.mem.zeroes(Launch);
        launch.kind = c.DS4_TP_POLL;
        launch.pipeline = pso;
        launch.value = @truncate(sequence);
        launch.words = c.DS4_TP_POLL_LINES;
        launch.groups = 1;
        launch.threads = 32;
        launch.end_encoder = true;
        launch.flags = at(next.poll, (sequence % c.DS4_TP_POLL_RING) * c.DS4_TP_POLL_LINES * c.DS4_TP_POLL_LINE_BYTES);
        launch.checks = at(next.poll, @as(u64, c.DS4_TP_POLL_LINES) * c.DS4_TP_POLL_LINE_BYTES * c.DS4_TP_POLL_RING + @as(u64, slot) * 8);
        if (c.ds4_metal_tp_encode(command.token.?, &launch) == 0) return false;
    } else c.ds4_metal_tp_wait(c.DS4_TP_EVENT_ROW, sequence);
    return enqueue(layer, which, 0, sequence, poll, event, null, null, 0);
}
// Batch gates have a separate sequence/release event: RDMA row receives need
// consecutive values, and a stale row flag must not satisfy a batch arrival.
pub fn batch(layer: u32, rows: u32) bool {
    const s = state();
    const f = facts();
    if (!f.batch or !f.service_running or rows == 0) return false;
    s.batch_sequence +%= 1;
    const sequence = s.batch_sequence;
    const event = s.session_batch_mode != 0 or !s.flag_gates;
    if (!event) {
        var command = Command{};
        defer command.drop();
        if (!command.acquire() or command.owned != 0) return false;
        if (!flag(&command, s, f, slotFor(layer, 1), @as(u64, c.DS4_TP_BATCH_FLAG_TAG) | @as(u32, @truncate(sequence)), false)) return false;
    } else {
        Schedule.closeEncoder();
        c.ds4_metal_tp_signal(c.DS4_TP_EVENT_BATCH, sequence);
    }
    c.ds4_metal_tp_wait(c.DS4_TP_EVENT_BATCH, sequence);
    return enqueue(layer, 1, rows, sequence, false, event, null, null, 0);
}
pub fn bigKick(layer: u32, rows: u32, out: ?*const api.Tensor, input: ?*api.Tensor, bytes: u64) u64 {
    const s = state();
    const f = facts();
    if (!f.batch or !f.service_running or rows == 0 or bytes == 0) return 0;
    const source = c.ds4_gpu_tensor_contents(@constCast(out));
    const target = c.ds4_gpu_tensor_contents(input);
    if (source == null or target == null) {
        _ = errorLine("ds4: TP big gate needs CPU-visible bounce buffers");
        return 0;
    }
    s.batch_sequence +%= 1;
    const sequence = s.batch_sequence;
    Schedule.closeEncoder();
    c.ds4_metal_tp_signal(c.DS4_TP_EVENT_BATCH, sequence);
    return if (enqueue(layer, 1, rows, sequence, false, true, source, target, bytes)) sequence else 0;
}
pub fn bigWait(sequence: u64) bool {
    if (!facts().batch or sequence == 0) return false;
    Schedule.closeEncoder();
    c.ds4_metal_tp_wait(c.DS4_TP_EVENT_BATCH, sequence);
    return true;
}
fn eventSupport(f: Facts) bool {
    return if (f.shared_events) true else errorLine("ds4: selected-id overlap requires MTLSharedEvent support");
}
pub fn signalSelected(output: *u64) bool {
    output.* = 0;
    if (api.init() == 0) return false;
    Schedule.resetParallel(true);
    const f = facts();
    if (!f.batch or !eventSupport(f)) return false;
    if (!c.ds4_metal_tp_selected_ready()) return errorLine("ds4: failed to create Metal shared event for selected-id overlap");
    Schedule.closeEncoder();
    const s = state();
    s.selected_sequence +%= 1;
    c.ds4_metal_tp_signal(c.DS4_TP_EVENT_SELECTED, s.selected_sequence);
    output.* = s.selected_sequence;
    return true;
}
fn retireReadback(command: ?*anyopaque, sequence: u64, label: [*:0]const u8) bool {
    c.ds4_metal_tp_commit(command);
    const signaled = c.ds4_metal_tp_wait_selected(sequence);
    c.ds4_metal_tp_append_pending(command);
    if (!signaled) {
        _ = c.fprintf(c.__stderrp, "ds4: timeout waiting for Metal shared event in %s\n", label);
        _ = Schedule.waitPending(label);
        c.ds4_metal_tp_clear_transients();
        return false;
    }
    if (c.ds4_metal_tp_command_error(command, label)) {
        _ = Schedule.waitPending(label);
        c.ds4_metal_tp_clear_transients();
        return false;
    }
    if (!c.ds4_metal_tp_new_batch()) {
        _ = Schedule.waitPending(label);
        c.ds4_metal_tp_clear_transients();
        return false;
    }
    return true;
}
pub fn commitSelected(sequence: u64, label: ?[*:0]const u8) bool {
    if (api.init() == 0) return false;
    Schedule.resetParallel(true);
    const f = facts();
    if (!f.batch or sequence == 0 or !eventSupport(f) or !f.selected_event) return false;
    Schedule.closeEncoder();
    const command = c.ds4_metal_tp_take_batch(0);
    defer c.ds4_metal_tp_drop(command);
    return retireReadback(command, sequence, label orelse "selected-id overlap");
}
pub fn waitSelected(sequence: u64, label: ?[*:0]const u8) bool {
    if (api.init() == 0 or sequence == 0) return false;
    const f = facts();
    if (!eventSupport(f) or !f.selected_event) return false;
    if (!c.ds4_metal_tp_wait_selected(sequence)) {
        _ = c.fprintf(c.__stderrp, "ds4: timeout waiting for Metal shared event in %s\n", label orelse "selected-id readback");
        return false;
    }
    return true;
}
pub fn signalBatchWait(label: ?[*:0]const u8) bool {
    if (api.init() == 0) return false;
    Schedule.resetParallel(true);
    const f = facts();
    if (!f.batch) return false;
    const what = label orelse "selected-id readback";
    if (!f.shared_events) {
        Schedule.closeEncoder();
        const command = c.ds4_metal_tp_take_batch(0);
        defer c.ds4_metal_tp_drop(command);
        if (Schedule.finish(command, 1, what, 0) == 0) return false;
        return c.ds4_gpu_begin_commands() != 0;
    }
    if (!c.ds4_metal_tp_selected_ready()) {
        _ = c.fprintf(c.__stderrp, "ds4: failed to create Metal shared event for %s\n", what);
        return false;
    }
    Schedule.closeEncoder();
    const command = c.ds4_metal_tp_take_batch(2); // preserve has_work until signal
    defer c.ds4_metal_tp_drop(command);
    const s = state();
    s.selected_sequence +%= 1;
    c.ds4_metal_tp_signal_detached(command, s.selected_sequence);
    return retireReadback(command, s.selected_sequence, what);
}

pub fn splitFlushSafe() bool {
    const f = facts();
    const s = state();
    return f.service_running and s.flag_gates and s.session_batch_mode == 0;
}

pub fn setBatchMode(enabled: c_int) void {
    state().session_batch_mode = if (enabled != 0) 1 else 0;
}
