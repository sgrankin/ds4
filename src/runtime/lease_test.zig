//! Cross-owner exclusion must allow multiple in-flight commands from one engine.
const std = @import("std");
const c = @import("runtime_c");
var owner_a: u8 = 0;
var owner_b: u8 = 0;
var started = std.atomic.Value(bool).init(false);
var entered = std.atomic.Value(bool).init(false);
extern "c" fn usleep(microseconds: c_uint) c_int;

fn worker(_: ?*anyopaque) callconv(.c) ?*anyopaque {
    started.store(true, .release);
    c.ds4_runtime_gpu_begin(&owner_b);
    entered.store(true, .release);
    c.ds4_runtime_gpu_end(&owner_b);
    return null;
}
fn run() !void {
    // Resident and streamed work from A must both be admissible before A waits.
    c.ds4_runtime_gpu_begin(&owner_a);
    c.ds4_runtime_gpu_begin(&owner_a);
    var thread: c.pthread_t = undefined;
    if (c.pthread_create(&thread, null, worker, null) != 0) return error.CreateThread;
    while (!started.load(.acquire)) _ = usleep(100);
    _ = usleep(10000);
    const before = entered.load(.acquire);
    c.ds4_runtime_gpu_end(&owner_a);
    _ = usleep(10000);
    const halfway = entered.load(.acquire);
    c.ds4_runtime_gpu_end(&owner_a);
    // A fresh A group must yield to B, while A's earlier dependent commands
    // above must remain admissible. Otherwise chunking alone can starve B.
    c.ds4_runtime_gpu_begin(&owner_a);
    const handed_over = entered.load(.acquire);
    c.ds4_runtime_gpu_end(&owner_a);
    _ = c.pthread_join(thread, null);
    if (before or halfway or !handed_over) return error.CrossOwnerOverlap;
    std.debug.print("Runtime lease: same-owner dependencies overlap; waiting owners get the next drained group.\n", .{});
}
export fn main() c_int {
    run() catch |err| {
        std.debug.print("Runtime lease: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}
