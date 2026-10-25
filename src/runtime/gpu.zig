//! Cross-engine GPU exclusion, preserving an engine's overlapping command
//! buffers. Completion can retire a group from a Metal callback thread.
const c = @import("runtime_c");
var mutex: c.pthread_mutex_t = c.ds4_runtime_mutex_init;
var ready: c.pthread_cond_t = c.ds4_runtime_cond_init;
var active_owner: ?*const anyopaque = null;
var in_flight: usize = 0;
const Waiter = struct { next: ?*Waiter = null };
var first: ?*Waiter = null;
var last: ?*Waiter = null;

export fn ds4_runtime_gpu_begin(owner: *const anyopaque) void {
    _ = c.pthread_mutex_lock(&mutex);
    // Dependent submissions from the current owner must remain admissible.
    // Once that group drains, a newly submitted group joins behind owners
    // already waiting, rather than repeatedly winning the wake-up race.
    if (active_owner != owner and (active_owner != null or first != null)) {
        var waiter: Waiter = .{};
        if (last) |tail| tail.next = &waiter else first = &waiter;
        last = &waiter;
        while (active_owner != owner and (active_owner != null or first != &waiter))
            _ = c.pthread_cond_wait(&ready, &mutex);

        // A dependent same-owner waiter can enter while its group is active,
        // even if another owner's next group is ahead of it in the queue.
        var previous: ?*Waiter = null;
        var cursor = first;
        while (cursor != &waiter) {
            previous = cursor;
            cursor = cursor.?.next;
        }
        if (previous) |before| before.next = waiter.next else first = waiter.next;
        if (last == &waiter) last = previous;
    }
    active_owner = owner;
    in_flight += 1;
    _ = c.pthread_mutex_unlock(&mutex);
}
export fn ds4_runtime_gpu_end(owner: *const anyopaque) void {
    _ = c.pthread_mutex_lock(&mutex);
    if (active_owner != owner or in_flight == 0) c.abort();
    in_flight -= 1;
    if (in_flight == 0) {
        active_owner = null;
        _ = c.pthread_cond_broadcast(&ready);
    }
    _ = c.pthread_mutex_unlock(&mutex);
}
