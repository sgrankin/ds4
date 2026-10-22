//! Sole C facade and the original one process-wide prefill-label sequence.
const c = @import("agent_runtime_c");
const owner = @import("communications.zig");
const Communications = owner.Communications;
var next_label: c_uint = 0;
fn comms(worker: *c.agent_worker) Communications {
    return .{ .worker = worker };
}
fn bytes(pointer: [*c]const u8, length: usize) []const u8 {
    return if (length == 0) &.{} else pointer[0..length];
}
fn errors(pointer: [*c]u8, length: usize) []u8 {
    return if (pointer == null or length == 0) &.{} else pointer[0..length];
}
export fn ds4_agent_comms_next_label() c_uint {
    const label = next_label;
    next_label +%= 1;
    return label;
}
export fn ds4_agent_comms_wake_locked(worker: *c.agent_worker) void {
    comms(worker).wakeLocked();
}
export fn ds4_agent_comms_publish(worker: *c.agent_worker, value: [*c]const u8, length: usize) void {
    comms(worker).publish(bytes(value, length));
}
export fn ds4_agent_comms_status(worker: *c.agent_worker, state: c.agent_worker_state) void {
    comms(worker).setStatus(state);
}
export fn ds4_agent_comms_error(worker: *c.agent_worker, value: [*c]const u8) void {
    comms(worker).setError(value);
}
export fn ds4_agent_comms_progress(context: ?*anyopaque, event: [*c]const u8, current: c_int, total: c_int) void {
    _ = total;
    if (context) |pointer| comms(@ptrCast(@alignCast(pointer))).progress(event, current);
}
export fn ds4_agent_comms_interrupted(worker: *c.agent_worker) bool {
    return comms(worker).interrupted();
}
export fn ds4_agent_comms_clear_interrupt(worker: *c.agent_worker) void {
    comms(worker).clearInterrupt();
}
export fn ds4_agent_comms_cancel(context: ?*anyopaque) bool {
    return comms(@ptrCast(@alignCast(context.?))).interrupted();
}
export fn ds4_agent_comms_web_confirm(context: ?*anyopaque, prompt: [*c]const u8, error_text: [*c]u8, capacity: usize) c_int {
    const output = errors(error_text, capacity);
    const worker: ?*c.agent_worker = if (context) |pointer| @ptrCast(@alignCast(pointer)) else null;
    if (worker == null or worker.?.cfg.*.non_interactive) {
        if (output.len != 0) _ = c.snprintf(output.ptr, output.len, "visible Chrome browser startup requires interactive approval");
        return 0;
    }
    return @intFromBool(comms(worker.?).confirmWeb(prompt, output));
}
export fn ds4_agent_comms_take_web(worker: *c.agent_worker, prompt: [*c]u8, capacity: usize) bool {
    return comms(worker).takeWeb(errors(prompt, capacity));
}
export fn ds4_agent_comms_answer_web(worker: *c.agent_worker, accepted: bool, reason: [*c]const u8) void {
    comms(worker).answerWeb(accepted, reason);
}
export fn ds4_agent_comms_request_drain(worker: *c.agent_worker) [*c]u8 {
    return comms(worker).requestDrain();
}
export fn ds4_agent_comms_take_drain(worker: *c.agent_worker) bool {
    return comms(worker).takeDrain();
}
export fn ds4_agent_comms_answer_drain(worker: *c.agent_worker, owned: [*c]u8) void {
    comms(worker).answerDrain(owned);
}
export fn ds4_agent_comms_request_save(worker: *c.agent_worker) void {
    comms(worker).requestSave();
}
export fn ds4_agent_comms_request_compact(worker: *c.agent_worker) void {
    comms(worker).requestCompact();
}
export fn ds4_agent_comms_request_power(worker: *c.agent_worker, percent: c_int) void {
    comms(worker).requestPower(percent);
}
export fn ds4_agent_comms_take_save(worker: *c.agent_worker) bool {
    return comms(worker).takeSave();
}
export fn ds4_agent_comms_take_compact(worker: *c.agent_worker) bool {
    return comms(worker).takeCompact();
}

export fn ds4_agent_comms_apply_power(worker: *c.agent_worker) void {
    comms(worker).applyPendingPower();
}
export fn ds4_agent_comms_take_raw_restore(worker: *c.agent_worker) bool {
    return comms(worker).takeRawRestore();
}
export fn ds4_agent_comms_drain_fd(descriptor: c_int) void {
    owner.drainFileDescriptor(descriptor);
}
export fn ds4_agent_comms_submit(worker: *c.agent_worker, value: [*c]const u8) bool {
    return comms(worker).submit(value);
}

export fn ds4_agent_comms_interrupt(worker: *c.agent_worker) void {
    comms(worker).interrupt();
}
export fn ds4_agent_comms_stop(worker: *c.agent_worker) void {
    comms(worker).stop();
}
export fn ds4_agent_comms_consume(worker: *c.agent_worker, output: [*c][*c]u8, length: [*c]usize, status: [*c]c.agent_status) void {
    comms(worker).consume(output, length, status);
}
export fn ds4_agent_comms_get_status(worker: *c.agent_worker, status: *c.agent_status) void {
    comms(worker).getStatus(status);
}
export fn ds4_agent_comms_idle(worker: *c.agent_worker) bool {
    return comms(worker).idle();
}
export fn ds4_agent_comms_initialized(worker: *c.agent_worker, status: [*c]c.agent_status) bool {
    return comms(worker).initialized(status);
}
