//! Actual persistent pread-pool scheduling and synchronous fallback. Task and
//! mutex records are shared with C; workers never touch Metal or flush commands.
const std = @import("std");
const c = @import("metal_c");
const Cache = @import("expert_cache.zig");
const Task = c.ds4_gpu_stream_expert_pread_task;
const Worker = c.ds4_gpu_stream_expert_pread_worker_args;
const Pool = c.ds4_metal_expert_pread_pool;
fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
pub fn now() f64 {
    return c.ds4_metal_expert_now_ms();
}
pub fn threadLimit() u32 {
    var count: u32 = 9;
    const raw = c.getenv("DS4_METAL_STREAMING_EXPERT_PREAD_THREADS");
    if (raw != null and raw[0] != 0) {
        var end: [*c]u8 = null;
        const value = c.strtoul(raw, &end, 10);
        if (end != raw and end[0] == 0) count = @intCast(@min(value, std.math.maxInt(u32)));
    }
    return std.math.clamp(count, 1, 18);
}
pub fn threadCount(tasks: u32) u32 {
    return if (tasks <= 1) tasks else @min(threadLimit(), tasks);
}
pub fn poolEnabled() bool {
    const raw = c.getenv("DS4_METAL_STREAMING_EXPERT_PREAD_POOL");
    return !(raw != null and c.strcmp(raw, "0") == 0);
}
pub fn readInto(offset: u64, length: u64, destination: [*c]u8, read_bytes: [*c]u64, elapsed: [*c]f64) bool {
    if (read_bytes != null) read_bytes[0] = 0;
    if (elapsed != null) elapsed[0] = 0;
    const view = c.ds4_metal_expert_load_driver();
    const state: *Cache.State = @ptrCast(view.cache);
    if (view.model_fd < 0 or
        destination == null or
        length == 0 or
        offset > std.math.maxInt(c.off_t) or
        length > @as(u64, std.math.maxInt(c.off_t)) - offset) return false;
    const start = now();
    var position: u64 = 0;
    var ok = true;
    while (position < length) {
        const wanted: usize = @intCast(@min(length - position, std.math.maxInt(isize)));
        var received: isize = undefined;
        while (true) {
            if (state.io_metrics_enabled) {
                _ = @atomicRmw(u64, &state.io_pread_calls, .Add, 1, .monotonic);
                _ = @atomicRmw(u64, &state.io_requested_bytes, .Add, wanted, .monotonic);
            }
            received = c.pread(view.model_fd, destination + @as(usize, @intCast(position)), wanted, @intCast(offset + position));
            if (received >= 0 or c.__error().* != c.EINTR) break;
        }
        if (received <= 0) {
            ok = false;
            break;
        }
        if (state.io_metrics_enabled) _ = @atomicRmw(u64, &state.io_returned_bytes, .Add, @intCast(received), .monotonic);
        position += @intCast(received);
    }
    const duration = now() - start;
    if (read_bytes != null) read_bytes[0] = position;
    if (elapsed != null) elapsed[0] = duration;
    if (!ok or position != length) {
        _ = c.fprintf(c.__stderrp, "ds4: Metal streaming expert explicit pread failed offset=%.2f GiB len=%.2f MiB read=%.2f MiB\n", @as(f64, @floatFromInt(offset)) / 1073741824, @as(f64, @floatFromInt(length)) / 1048576, @as(f64, @floatFromInt(position)) / 1048576);
        return false;
    }
    return true;
}
fn worker(raw: ?*anyopaque) callconv(.c) ?*anyopaque {
    const args: *Worker = @ptrCast(@alignCast(raw.?));
    var index = args.worker_index;
    while (index < args.n_tasks) : (index +%= args.n_workers) {
        const task: *Task = @ptrCast(&args.tasks[index]);
        task.ok = @intFromBool(readInto(task.offset, task.len, task.dst, &task.read_bytes, &task.ms));
    }
    return null;
}
fn poolWorker(raw: ?*anyopaque) callconv(.c) ?*anyopaque {
    const index: u32 = @truncate(@intFromPtr(raw));
    const view = c.ds4_metal_expert_load_driver();
    const pool: *Pool = @ptrCast(&view.load.*.pool);
    var generation: u64 = 0;
    while (true) {
        _ = c.pthread_mutex_lock(&pool.mutex);
        while (pool.stopping == 0 and pool.generation == generation) _ = c.pthread_cond_wait(&pool.start_cond, &pool.mutex);
        if (pool.stopping != 0) {
            _ = c.pthread_mutex_unlock(&pool.mutex);
            break;
        }
        generation = pool.generation;
        if (index >= pool.active_workers) {
            _ = c.pthread_mutex_unlock(&pool.mutex);
            continue;
        }
        while (true) {
            const task_index = pool.next_task;
            pool.next_task +%= 1;
            if (task_index >= pool.n_tasks) break;
            const task: *Task = @ptrCast(&pool.tasks[task_index]);
            _ = c.pthread_mutex_unlock(&pool.mutex);
            task.ok = @intFromBool(readInto(task.offset, task.len, task.dst, &task.read_bytes, &task.ms));
            _ = c.pthread_mutex_lock(&pool.mutex);
            task.done = 1;
            if (task.gate_stage != 0) _ = c.pthread_cond_broadcast(&pool.done_cond);
        }
        if (pool.remaining_workers > 0) {
            pool.remaining_workers -= 1;
            if (pool.remaining_workers == 0) {
                pool.tasks = null;
                pool.n_tasks = 0;
                pool.active_workers = 0;
                _ = c.pthread_cond_signal(&pool.done_cond);
            }
        }
        _ = c.pthread_mutex_unlock(&pool.mutex);
    }
    return null;
}
pub fn init(pool: *Pool, requested: u32) bool {
    if (pool.initialized != 0) return true;
    if (!poolEnabled() or requested <= 1) return false;
    const count = @min(requested, 18);
    _ = c.pthread_mutex_lock(&pool.mutex);
    pool.thread_count = count;
    pool.stopping = 0;
    pool.generation = 0;
    pool.tasks = null;
    pool.n_tasks = 0;
    pool.next_task = 0;
    pool.active_workers = 0;
    pool.remaining_workers = 0;
    _ = c.pthread_mutex_unlock(&pool.mutex);
    var started: u32 = 0;
    while (started < count) : (started += 1) {
        const result = c.pthread_create(&pool.threads[started], null, poolWorker, @ptrFromInt(started));
        if (result == 0) continue;
        _ = c.fprintf(c.__stderrp, "ds4: Metal streaming expert pread pool thread creation failed: %s\n", c.strerror(result));
        _ = c.pthread_mutex_lock(&pool.mutex);
        pool.stopping = 1;
        pool.generation +%= 1;
        _ = c.pthread_cond_broadcast(&pool.start_cond);
        _ = c.pthread_mutex_unlock(&pool.mutex);
        for (pool.threads[0..started]) |thread| _ = c.pthread_join(thread, null);
        _ = c.pthread_mutex_lock(&pool.mutex);
        pool.thread_count = 0;
        pool.stopping = 0;
        _ = c.pthread_mutex_unlock(&pool.mutex);
        return false;
    }
    pool.initialized = 1;
    return true;
}
pub fn begin(pool: *Pool, tasks: [*c]Task, count: u32, requested: u32) bool {
    if (requested <= 1 or !init(pool, threadLimit())) return false;
    _ = c.pthread_mutex_lock(&pool.mutex);
    defer _ = c.pthread_mutex_unlock(&pool.mutex);
    if (pool.initialized == 0 or pool.stopping != 0 or pool.thread_count == 0) return false;
    const workers = @min(requested, pool.thread_count);
    if (workers == 0 or pool.remaining_workers != 0 or pool.tasks != null) return false;
    pool.tasks = tasks;
    pool.n_tasks = count;
    pool.next_task = 0;
    pool.active_workers = workers;
    pool.remaining_workers = workers;
    pool.generation +%= 1;
    _ = c.pthread_cond_broadcast(&pool.start_cond);
    return true;
}
pub fn wait(pool: *Pool) bool {
    if (pool.initialized == 0) return false;
    _ = c.pthread_mutex_lock(&pool.mutex);
    defer _ = c.pthread_mutex_unlock(&pool.mutex);
    while (pool.remaining_workers != 0) _ = c.pthread_cond_wait(&pool.done_cond, &pool.mutex);
    return true;
}
/// Gate/up can submit while the SAME pool is still reading down weights.
pub fn gateWait(view: c.ds4_metal_expert_load_view) c_int {
    const state: *Cache.State = @ptrCast(view.cache);
    const pool: *Pool = @ptrCast(&view.load.*.pool);
    const pending = &view.load.*.pending;
    state.gate_probes +%= 1;
    _ = c.pthread_mutex_lock(&pool.mutex);
    defer _ = c.pthread_mutex_unlock(&pool.mutex);
    while (true) {
        var ready = true;
        var ok = true;
        for (pending.tasks[0..pending.n_tasks]) |task| {
            if (task.gate_stage == 0) continue;
            if (task.done == 0) ready = false else if (task.ok == 0) ok = false;
        }
        if (ready or pool.remaining_workers == 0) {
            var down_pending = false;
            for (pending.tasks[0..pending.n_tasks]) |task| if (task.gate_stage == 0 and task.done == 0) {
                down_pending = true;
            };
            return if (ready and ok) (if (down_pending) @as(c_int, 2) else 1) else 0;
        }
        _ = c.pthread_cond_wait(&pool.done_cond, &pool.mutex);
    }
}
pub fn shutdown(pool: *Pool) void {
    if (pool.initialized == 0) return;
    _ = c.pthread_mutex_lock(&pool.mutex);
    pool.stopping = 1;
    pool.generation +%= 1;
    _ = c.pthread_cond_broadcast(&pool.start_cond);
    _ = c.pthread_mutex_unlock(&pool.mutex);
    for (pool.threads[0..pool.thread_count]) |thread| _ = c.pthread_join(thread, null);
    _ = c.pthread_mutex_lock(&pool.mutex);
    defer _ = c.pthread_mutex_unlock(&pool.mutex);
    pool.thread_count = 0;
    pool.active_workers = 0;
    pool.remaining_workers = 0;
    pool.n_tasks = 0;
    pool.next_task = 0;
    pool.tasks = null;
    pool.initialized = 0;
    pool.stopping = 0;
}
pub fn readTasks(pool: *Pool, input: [*c]Task, count: u32, total: [*c]u64, elapsed: [*c]f64) bool {
    if (total != null) total[0] = 0;
    if (elapsed != null) elapsed[0] = 0;
    if (input == null or count == 0) return true;
    const workers = threadCount(count);
    const start = now();
    var ok = true;
    if (workers <= 1) {
        var args = Worker{ .tasks = input, .n_tasks = count, .worker_index = 0, .n_workers = 1 };
        _ = worker(&args);
    } else if (!(begin(pool, input, count, workers) and wait(pool))) {
        var threads: [18]c.pthread_t = undefined;
        var args: [18]Worker = undefined;
        var started: u32 = 0;
        while (started < workers) : (started += 1) {
            args[started] = .{ .tasks = input, .n_tasks = count, .worker_index = started, .n_workers = workers };
            const result = c.pthread_create(&threads[started], null, worker, &args[started]);
            if (result == 0) continue;
            _ = c.fprintf(c.__stderrp, "ds4: Metal streaming expert pread thread creation failed: %s\n", c.strerror(result));
            ok = false;
            break;
        }
        for (threads[0..started]) |thread| if (c.pthread_join(thread, null) != 0) {
            ok = false;
        };
    }
    const duration = now() - start;
    var bytes: u64 = 0;
    for (input[0..count]) |task| {
        if (task.ok == 0) ok = false;
        bytes +|= task.read_bytes;
    }
    if (total != null) total[0] = bytes;
    if (elapsed != null) elapsed[0] = duration;
    return ok;
}
pub fn noteRead(state: *Cache.State, layer: u32, bytes: u64, ms: f64) void {
    state.cache_pread_bytes +|= bytes;
    state.cache_pread_ms += ms;
    if (layer < c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER) {
        state.cache_layer_pread_bytes[layer] +|= bytes;
        state.cache_layer_pread_ms[layer] += ms;
    }
}
pub fn readahead(view: c.ds4_metal_expert_load_view, offset: u64, length: u64) void {
    const state: *Cache.State = @ptrCast(view.cache);
    if (!view.streaming or present("DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD") or view.model_fd < 0 or length == 0) return;
    if (comptime !@hasDecl(c, "F_RDADVISE")) return;
    const timing = Cache.timingSummary();
    const start = if (timing) now() else 0;
    var position = offset;
    var remaining = length;
    while (remaining > 0) {
        const count = @min(remaining, std.math.maxInt(c_int));
        if (position > std.math.maxInt(c.off_t)) break;
        var advice = c.struct_radvisory{ .ra_offset = @intCast(position), .ra_count = @intCast(count) };
        _ = c.fcntl(view.model_fd, c.F_RDADVISE, &advice);
        position +%= count;
        remaining -= count;
    }
    if (timing) {
        state.timing_readahead_calls +%= 1;
        state.timing_readahead_bytes +%= length;
        state.timing_readahead_ms += now() - start;
    }
}
