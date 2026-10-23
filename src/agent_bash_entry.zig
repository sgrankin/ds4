//! Sole C facade for the private shell-job owner.
const owner = @import("agent_bash.zig");
const c = @import("agent_bash_c");

fn job(pointer: *c.ds4_agent_bash) *owner.Job {
    return @ptrCast(@alignCast(pointer));
}
export fn ds4_agent_bash_start(command: [*c]const u8, timeout: c_int, next_id: *c_int, terminal: c.ds4_agent_bash_terminal_fn, userdata: ?*anyopaque, err: [*c]u8, errlen: usize) ?*c.ds4_agent_bash {
    const process = owner.start(command, timeout, next_id, terminal, userdata, if (errlen == 0) &.{} else err[0..errlen]) orelse return null;
    return @ptrCast(process);
}
export fn ds4_agent_bash_free(pointer: ?*c.ds4_agent_bash) void {
    if (pointer) |process| job(process).deinit();
}
export fn ds4_agent_bash_running(pointer: *c.ds4_agent_bash) bool {
    return job(pointer).running();
}
export fn ds4_agent_bash_signal(pointer: *c.ds4_agent_bash, number: c_int) void {
    job(pointer).signal(number);
}
export fn ds4_agent_bash_get_snapshot(pointer: *c.ds4_agent_bash, observed: bool, output: *c.ds4_agent_bash_snapshot) void {
    output.* = job(pointer).snapshot(observed);
}
export fn ds4_agent_bash_observation(pointer: *c.ds4_agent_bash, observed: bool, done: [*c]bool) [*c]u8 {
    return job(pointer).observation(observed, if (done == null) null else @ptrCast(done));
}
