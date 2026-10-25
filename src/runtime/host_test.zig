//! Actual host contexts and independent CPU callback pools; no model or GPU.
const std = @import("std");
const c = @import("runtime_c");
const Job = struct {
    owner: *c.ds4_host_context,
    width: u32,
    rows: [1024]u32 = @splat(0),
    wrong_owner: std.atomic.Value(bool) = .init(false),

    fn rowsFn(raw: ?*anyopaque, first: u64, end: u64) callconv(.c) void {
        const self: *Job = @ptrCast(@alignCast(raw.?));
        if (c.ds4_host_context_selected() != self.owner or c.ds4_host_context_shape(self.owner).*.n_embd != self.width)
            self.wrong_owner.store(true, .monotonic);
        for (first..end) |i| self.rows[i] = self.width + @as(u32, @intCast(i));
    }
    fn run(raw: ?*anyopaque) callconv(.c) ?*anyopaque {
        const self: *Job = @ptrCast(@alignCast(raw.?));
        c.ds4_host_context_parallel(self.owner, self.rows.len, rowsFn, self, 1);
        return null;
    }
    fn check(self: *const Job) !void {
        if (self.wrong_owner.load(.monotonic)) return error.WrongOwner;
        for (self.rows, 0..) |value, i| if (value != self.width + i) return error.WrongRows;
    }
};
fn configure(owner: *c.ds4_host_context, width: u32) void {
    var configuration = std.mem.zeroes(c.ds4_engine_configuration);
    configuration.shape.n_embd = width;
    configuration.shape.family = c.DS4_MODEL_FAMILY_DEEPSEEK41;
    const previous = c.ds4_host_context_select(owner);
    c.ds4_core_publish_configuration(&configuration);
    _ = c.ds4_host_context_select(previous);
}
fn run() !void {
    const first = c.ds4_host_context_create(2) orelse return error.CreateOwner;
    const second = c.ds4_host_context_create(3) orelse return error.CreateOwner;
    configure(first, 5120);
    configure(second, 4096);
    var a = Job{ .owner = first, .width = 5120 };
    var b = Job{ .owner = second, .width = 4096 };
    var threads: [2]c.pthread_t = undefined;
    if (c.pthread_create(&threads[0], null, Job.run, &a) != 0) return error.CreateThread;
    if (c.pthread_create(&threads[1], null, Job.run, &b) != 0) return error.CreateThread;
    for (threads) |thread| if (c.pthread_join(thread, null) != 0) return error.JoinThread;
    try a.check();
    try b.check();
    _ = c.ds4_host_context_select(second);
    c.ds4_host_context_destroy(first);
    if (c.ds4_host_context_selected() != second) return error.SurvivorSelection;
    @memset(&b.rows, 0);
    _ = Job.run(&b);
    try b.check();
    c.ds4_host_context_destroy(second);
    if (c.ds4_host_context_selected() != null) return error.DanglingSelection;
    std.debug.print("Host contexts: independent shapes/pools, concurrent callbacks, worker owner capture and survivor after close pass.\n", .{});
}
export fn main() c_int {
    run() catch |err| {
        std.debug.print("Host contexts: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}
