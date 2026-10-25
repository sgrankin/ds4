//! One command transaction owner for ordinary dispatch, batched work and
//! concurrent FFN/KV sections. ARC leaves retain storage; native code decides
//! when an encoder is reused, submitted, waited or retired.
const c = @import("metal_c");
const std = @import("std");
const api = @import("bridge.zig");
pub fn state() *c.ds4_metal_scheduling {
    return @ptrCast(c.ds4_metal_scheduling_state());
}
fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
var use_unretained = false;
var policy_once: c.pthread_once_t = c.ds4_metal_policy_once_init;
fn initializePolicy() callconv(.c) void {
    use_unretained = present("DS4_METAL_UNRETAINED_COMMAND_BUFFERS");
}
pub fn newCommand() ?*anyopaque {
    _ = c.pthread_once(&policy_once, initializePolicy);
    return c.ds4_metal_schedule_new_command(use_unretained);
}
pub fn drop(object: ?*anyopaque) void {
    if (object) |value| c.ds4_metal_schedule_drop(value);
}

/// The returned token owns one retain in both paths. A batch keeps its own
/// independent retain, and its caller remains responsible for submission.
pub fn acquire(owned: *c_int) ?*anyopaque {
    if (state().commands.batch) |batch| {
        owned.* = 0;
        return c.ds4_metal_schedule_retain(batch);
    }
    owned.* = 1;
    const command = newCommand() orelse return null;
    c.ds4_metal_schedule_cache_event(c.DS4_SCHEDULE_OWNED_CREATED);
    return command;
}
pub fn closeEncoder() void {
    const current_encoder = state().commands.encoder orelse return;
    c.ds4_metal_schedule_end_encoder(current_encoder);
    c.ds4_metal_schedule_set_encoder(null);
}

/// Diagnostic timeline passes are separate except inside one concurrent
/// section. A failed timeline encoder falls through to the ordinary encoder.
/// Returned encoders are retained so a caller's local autorelease pool may end.
pub fn encoder(command: ?*anyopaque, caller: usize) ?*anyopaque {
    const site = if (caller == 0) @returnAddress() else caller;
    const s = &state().commands;
    const batch = s.batch != null and command == s.batch;
    if (!batch) return c.ds4_metal_schedule_new_encoder(command, false, false, site);
    s.has_work = true;
    if (c.ds4_metal_schedule_timeline_active()) {
        if (s.encoder != null and !s.concurrent) closeEncoder();
        if (s.encoder == null) {
            const timed = c.ds4_metal_schedule_new_encoder(command, s.concurrent, true, site);
            c.ds4_metal_schedule_set_encoder(timed);
            drop(timed);
        }
        if (s.encoder) |value| return c.ds4_metal_schedule_retain(value);
    }
    if (s.encoder == null) {
        const created = c.ds4_metal_schedule_new_encoder(command, s.concurrent, false, site);
        c.ds4_metal_schedule_set_encoder(created);
        drop(created);
    }
    return if (s.encoder) |value| c.ds4_metal_schedule_retain(value) else null;
}
pub fn endEncoder(command: ?*anyopaque, current: ?*anyopaque) void {
    const s = &state().commands;
    if (current == null) return;
    if (s.batch != null and command == s.batch and current == s.encoder) return;
    c.ds4_metal_schedule_end_encoder(current);
}
pub fn waitPending(label: [*c]const u8) c_int {
    var ok: c_int = 1;
    const count = c.ds4_metal_schedule_pending_count();
    for (0..@intCast(count)) |index| {
        const pending = c.ds4_metal_schedule_pending_at(index);
        defer drop(pending);
        if (c.ds4_metal_schedule_wait(pending, label) == 0) ok = 0;
    }
    c.ds4_metal_schedule_pending_clear();
    c.ds4_metal_schedule_cache_event(c.DS4_SCHEDULE_PENDING_DONE);
    if (ok == 0) c.ds4_metal_schedule_invalidate_maps();
    return ok;
}

/// Release consumes the token even on failure. Failed owned sequences are
/// dropped uncommitted; failed borrowed sequences leave the caller batch live.
pub fn release(command: ?*anyopaque, owned: c_int, success: c_int, label: [*c]const u8, caller: usize) c_int {
    defer drop(command);
    if (command == null or success == 0) return 0;
    return finish(command, owned, label, caller);
}
pub fn finish(command: ?*anyopaque, owned: c_int, label: [*c]const u8, caller: usize) c_int {
    if (owned == 0) return 1;
    const commit_ms = c.ds4_metal_dense_now_ms();
    c.ds4_metal_schedule_commit(command);
    var ok = waitPending(label);
    if (c.ds4_metal_schedule_wait(command, label) == 0) {
        ok = 0;
        c.ds4_metal_schedule_invalidate_maps();
    }
    const profile = present("DS4_METAL_CB_TIMES");
    if (profile) c.ds4_metal_schedule_finish_trace(command, label, commit_ms, state().commands.created_ms, if (caller == 0) @returnAddress() else caller);
    c.ds4_metal_schedule_cache_event(c.DS4_SCHEDULE_OWNED_DONE);
    c.ds4_metal_schedule_clear_transients(profile);
    c.ds4_metal_schedule_evict(label);
    return ok;
}

/// Closing preserves already encoded work. Retiring every admission/reference
/// field prevents a failed FFN from making a later ordinary dispatch concurrent.
pub fn resetParallel(close: bool) void {
    const s = state();
    const p = &s.parallel;
    if (close and (s.commands.concurrent or p.pending or p.encoded or p.mode != 0 or p.stage != 0)) closeEncoder();
    s.commands.concurrent = false;
    p.pending = false;
    p.encoded = false;
    p.mode = 0;
    p.stage = 0;
    c.ds4_metal_schedule_parallel_retire();
    p.down_weight_offset = 0;
    p.down_x_offset = 0;
    p.down_out_offset = 0;
    p.gate_weight_offset = 0;
    p.up_weight_offset = 0;
    p.gate_x_offset = 0;
    p.gate_out_offset = 0;
    p.up_out_offset = 0;
    p.mid_out_offset = 0;
    p.ids_offset = 0;
    p.down_args = std.mem.zeroes(c.ds4_metal_mv_args);
    p.gate_args = std.mem.zeroes(c.ds4_metal_mv_args);
    p.clamp = 0;
    p.nsg = 0;
    p.nr0 = 0;
    p.shared_bytes = 0;
    p.split = false;
    p.split_args = std.mem.zeroes(c.ds4_metal_shared_split_args);
}
fn installBatch() bool {
    const created = newCommand();
    defer drop(created);
    c.ds4_metal_schedule_set_batch(created);
    state().commands.has_work = false;
    if (created != null) c.ds4_metal_schedule_cache_event(c.DS4_SCHEDULE_BATCH_CREATED);
    c.ds4_metal_schedule_timeline_attach(created);
    return created != null;
}
pub fn begin() c_int {
    if (api.init() == 0) return 0;
    // A failed concurrent FFN must never affect the next command batch.
    resetParallel(true);
    if (state().commands.batch != null) return 0;
    // Refresh once per batch: same-engine A/B runs can change this environment.
    const facts = c.ds4_metal_schedule_facts();
    state().commands.static_head_norm = facts & c.DS4_SCHEDULE_STREAMING == 0 and
        facts & c.DS4_SCHEDULE_PRE_M5 != 0 and
        !present("DS4_METAL_DISABLE_PRE_M5_HEAD_RMS_ROPE_PIPELINE_STATIC");
    state().commands.created_ms = c.ds4_metal_dense_now_ms();
    return @intFromBool(installBatch());
}
pub fn flushEncoder() c_int {
    if (api.init() == 0) return 0;
    resetParallel(true);
    if (state().commands.batch == null) return 0;
    closeEncoder();
    return 1;
}
pub fn flush() c_int {
    if (api.init() == 0) return 0;
    resetParallel(true);
    const batch = state().commands.batch orelse return 0;
    const held = c.ds4_metal_schedule_retain(batch);
    defer drop(held);
    closeEncoder();
    c.ds4_metal_schedule_set_batch(null);
    state().commands.has_work = false;
    c.ds4_metal_schedule_commit(held);
    c.ds4_metal_schedule_pending_append(held);
    c.ds4_metal_schedule_cache_event(c.DS4_SCHEDULE_BATCH_COMMITTED);
    if (!installBatch()) {
        _ = waitPending("command batch");
        c.ds4_metal_schedule_clear_transients(false);
        return 0;
    }
    return 1;
}
pub fn end() c_int {
    if (state().commands.batch == null) {
        resetParallel(true);
        return 0;
    }
    resetParallel(true);
    closeEncoder();
    const held = c.ds4_metal_schedule_retain(state().commands.batch);
    c.ds4_metal_schedule_set_batch(null);
    state().commands.has_work = false;
    c.ds4_metal_schedule_cache_event(c.DS4_SCHEDULE_BATCH_TO_OWNED);
    return release(held, 1, 1, "command batch", 0);
}
pub fn synchronize() c_int {
    if (api.init() == 0) return 0;
    if (state().commands.batch != null) return end();
    resetParallel(true);
    if (c.ds4_metal_schedule_pending_count() != 0) {
        const ok = waitPending("synchronize");
        c.ds4_metal_schedule_clear_transients(false);
        c.ds4_metal_schedule_evict("synchronize");
        return ok;
    }
    const created = newCommand() orelse return 0;
    // synchronize historically creates no owned-cache epoch of its own.
    return release(created, 1, 1, "synchronize", 0);
}
