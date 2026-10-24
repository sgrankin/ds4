//! Generic layer/selected-range page-in jobs. Model-specific span selection is
//! borrowed from the current planner; ranges, workers, joins and slot policy
//! belong here. Join before retiring the borrowed model or a slot's storage.
const std = @import("std");
const c = @import("pagein_c");
pub const IO = @import("pagein_io.zig");
const allocator = std.heap.c_allocator;
pub const Job = c.metal_graph_stream_pagein_job;
pub const Range = c.metal_graph_stream_pagein_range;
pub const Slot = c.metal_graph_stream_prepare_slot;
const Worker = c.metal_graph_stream_pagein_worker;
const Model = c.ds4_model;
const Weights = c.ds4_weights;
const Layer = c.ds4_layer_weights;
const maximum = std.math.maxInt(u64);

fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn alloc(comptime T: type, count: usize) []T {
    return allocator.alloc(T, count) catch IO.outOfMemory();
}
fn zeroed(comptime T: type, count: usize) []T {
    const values = alloc(T, count);
    @memset(values, std.mem.zeroes(T));
    return values;
}
fn countSetting(primary: [*:0]const u8, fallback: [*:0]const u8, default: u32, cap: u32) u32 {
    var value = c.getenv(primary);
    if (value == null or value[0] == 0) value = c.getenv(fallback);
    if (value == null or value[0] == 0) return default;
    var end: [*c]u8 = null;
    const number = c.strtoul(value, &end, 10);
    if (end == value or end[0] != 0 or number == 0) return 1;
    return @intCast(@min(number, cap));
}
fn layerThreads() u32 {
    return countSetting("DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_THREADS", "DS4_METAL_STREAMING_PREFILL_LAYER_PAGEIN_THREADS", 8, 16);
}
fn selectedThreads(madvise: bool) u32 {
    if (!madvise) return 1;
    return countSetting("DS4_METAL_STREAMING_PREFILL_SELECTED_PREPARE_THREADS", "DS4_METAL_STREAMING_PREFILL_SELECTED_MADVISE_THREADS", layerThreads(), 16);
}
fn selectedGap() u32 {
    const value = c.getenv("DS4_METAL_STREAMING_PREFILL_SELECTED_PREPARE_GAP");
    if (value == null or value[0] == 0) return 0;
    var end: [*c]u8 = null;
    const number = c.strtoul(value, &end, 10);
    if (end == value or end[0] != 0) return 0;
    return @intCast(@min(number, 8));
}
fn profileLayer() bool {
    return present("DS4_METAL_STREAMING_PREFILL_LAYER_PAGEIN_PROFILE") or
        present("DS4_METAL_STREAMING_PREFILL_LAYER_PREAD_PROFILE") or
        present("DS4_METAL_STREAMING_PREFILL_LAYER_MADVISE_PROFILE") or
        present("DS4_METAL_STREAMING_PREFILL_LAYER_READAHEAD_PROFILE");
}
fn enabled(job: *const Job, have_graph: bool, streaming: bool) bool {
    if (have_graph and !streaming) return false;
    if (job.pread_only) return !present("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREAD") and !present("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE");
    if (job.readahead_only) return present("DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_READAHEAD") and
        !present("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_READAHEAD") and !present("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE");
    if (job.madvise_only) return !present("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_MADVISE") and !present("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE");
    return present("DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_PAGEIN") and !present("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN");
}

pub fn install(model: [*c]const Model, spans: [*c]const c.ds4_model_map_span_vec, label: [*c]const u8) bool {
    if (model == null or spans == null or spans.*.len == 0) return false;
    const offsets = alloc(u64, spans.*.len);
    defer allocator.free(offsets);
    const sizes = alloc(u64, spans.*.len);
    defer allocator.free(sizes);
    for (spans.*.v[0..spans.*.len], offsets, sizes) |span, *offset, *bytes| {
        offset.* = span.off;
        bytes.* = span.end - span.off;
    }
    const ok = c.ds4_gpu_set_model_map_spans(model.*.map, model.*.size, offsets.ptr, sizes.ptr, spans.*.len, spans.*.max_tensor_bytes) != 0;
    if (!ok) _ = c.fprintf(c.__stderrp, "ds4: Metal SSD streaming failed to map %s model spans\n", if (label == null) @as([*c]const u8, "requested") else label);
    return ok;
}

fn prepareRange(job: *const Job, range: Range, touched: *u64, sink: *u8) bool {
    if (job.pread_only) return IO.pread(job.model, range.off, range.size, touched, sink);
    if (job.readahead_only) {
        IO.readAhead(job.model, range.off, range.size, true);
        touched.* +|= range.size;
        return true;
    }
    if (job.madvise_only) return IO.advise(job.model, range.off, range.size, true, touched);
    return IO.touch(job.model, range.off, range.size, touched, sink);
}
fn singleMain(context: ?*anyopaque) callconv(.c) ?*anyopaque {
    const job: *Job = @ptrCast(@alignCast(context.?));
    const start = if (job.profile) IO.now() else 0;
    job.ok = true;
    for (job.ranges[0..job.n_ranges]) |range| {
        if (!prepareRange(job, range, &job.touched, &job.sink)) {
            job.ok = false;
            break;
        }
    }
    if (job.profile) job.thread_ms = (IO.now() - start) * 1000;
    return null;
}
fn workerMain(context: ?*anyopaque) callconv(.c) ?*anyopaque {
    const worker: *Worker = @ptrCast(@alignCast(context.?));
    const job = worker.job;
    const start = if (job != null and job.*.profile) IO.now() else 0;
    worker.ok = true;
    if (job == null or worker.stride == 0) {
        worker.ok = false;
        return null;
    }
    var index = worker.first;
    while (index < job.*.n_ranges) : (index += worker.stride) {
        if (!prepareRange(job, job.*.ranges[index], &worker.touched, &worker.sink)) {
            worker.ok = false;
            break;
        }
    }
    if (job.*.profile) worker.thread_ms = (IO.now() - start) * 1000;
    return null;
}
fn release(job: *Job) void {
    c.free(job.workers);
    c.free(@ptrCast(job.threads));
    c.free(job.ranges);
    job.* = std.mem.zeroes(Job);
}
fn spawn(job: *Job, selected: bool) bool {
    if (job.n_threads <= 1) {
        const result = c.pthread_create(&job.thread, null, singleMain, job);
        if (result != 0) {
            const message: [*:0]const u8 = if (selected) "ds4: Metal streaming prefill selected page-in thread failed: %s\n" else "ds4: Metal streaming prefill layer page-in thread failed: %s\n";
            _ = c.fprintf(c.__stderrp, message, c.strerror(result));
            release(job);
            return false;
        }
    } else {
        job.threads = zeroed(c.pthread_t, job.n_threads).ptr;
        job.workers = zeroed(Worker, job.n_threads).ptr;
        for (0..job.n_threads) |index| {
            const worker = &job.workers[index];
            worker.* = std.mem.zeroes(Worker);
            worker.job = job;
            worker.first = @intCast(index);
            worker.stride = job.n_threads;
            const result = c.pthread_create(&job.threads[index], null, workerMain, worker);
            if (result != 0) {
                const message: [*:0]const u8 = if (selected) "ds4: Metal streaming prefill selected page-in worker failed: %s\n" else "ds4: Metal streaming prefill layer page-in worker failed: %s\n";
                _ = c.fprintf(c.__stderrp, message, c.strerror(result));
                for (job.threads[0..index]) |thread| _ = c.pthread_join(thread, null);
                release(job);
                return false;
            }
        }
    }
    job.started = true;
    return true;
}

pub fn layerStart(model: [*c]const Model, weights: [*c]const Weights, index: u32, rows: u32, flags: u32, have_graph: bool, streaming: bool, layer_count: u32, maybe_job: ?*Job) bool {
    const job = maybe_job orelse return false;
    job.* = std.mem.zeroes(Job);
    job.ok = true;
    job.madvise_only = flags & c.DS4_STREAM_PREPARE_MADVISE != 0;
    job.pread_only = flags & c.DS4_STREAM_PREPARE_PREAD != 0;
    job.readahead_only = flags & c.DS4_STREAM_PREPARE_READAHEAD != 0;
    job.profile = profileLayer();
    job.layer = index;
    job.n_tokens = rows;
    if (!enabled(job, have_graph, streaming)) return true;
    if (model == null or weights == null or index >= layer_count) return false;
    const threads = layerThreads();
    var spans = std.mem.zeroes(c.ds4_model_map_span_vec);
    defer c.free(spans.v);
    if (!c.ds4_native_pagein_layer_spans(weights, index, flags & c.DS4_STREAM_PREPARE_DECODE != 0, &spans)) return false;
    const ranges = alloc(Range, @as(usize, spans.len) * threads);
    var count: u32 = 0;
    const page: u64 = @intCast(c.getpagesize());
    for (spans.v[0..spans.len]) |span| {
        const bytes = span.end - span.off;
        var consumed: u64 = 0;
        var chunk = bytes / threads;
        if (chunk > page) chunk = chunk / page * page;
        if (chunk == 0) chunk = bytes;
        for (0..threads) |thread| {
            if (consumed >= bytes) break;
            const take = if (thread + 1 == threads or bytes - consumed <= chunk) bytes - consumed else chunk;
            ranges[count] = .{ .off = span.off + consumed, .size = take };
            count += 1;
            consumed += take;
        }
        job.bytes +|= bytes;
    }
    job.unique = spans.len;
    job.model = model;
    job.ranges = ranges.ptr;
    job.n_ranges = count;
    job.n_threads = threads;
    return spawn(job, false);
}

pub fn selectedStart(model: [*c]const Model, layer: [*c]const Layer, selected: ?*c.ds4_gpu_tensor, index: u32, rows: u32, gate_bytes: u64, down_bytes: u64, have_graph: bool, streaming: bool, expert_count: u32, selected_count: u32, maybe_job: ?*Job) bool {
    const job = maybe_job orelse return false;
    job.* = std.mem.zeroes(Job);
    job.ok = true;
    job.profile = present("DS4_METAL_STREAMING_PREFILL_SELECTED_PAGEIN_PROFILE") or present("DS4_METAL_STREAMING_PREFILL_SELECTED_MADVISE_PROFILE");
    job.layer = index;
    job.n_tokens = rows;
    const madvise = have_graph and streaming and present("DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_MADVISE") and !present("DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_MADVISE");
    job.madvise_only = madvise;
    const pagein = have_graph and streaming and present("DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_PAGEIN") and !present("DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_PAGEIN");
    if (!pagein and !madvise) return true;
    if (model == null or layer == null or selected == null or rows == 0 or expert_count > c.DS4_MAX_EXPERT or gate_bytes == 0 or down_bytes == 0) return false;
    const ids_count = @as(u64, rows) * selected_count;
    if (ids_count > std.math.maxInt(usize) / @sizeOf(i32)) return false;
    var seen: [c.DS4_MAX_EXPERT]bool = @splat(false);
    {
        const ids = alloc(i32, @intCast(ids_count));
        defer allocator.free(ids);
        const read_start = if (job.profile) IO.now() else 0;
        const read_ok = c.ds4_gpu_tensor_read(selected, 0, ids.ptr, ids_count * @sizeOf(i32)) != 0;
        if (job.profile) job.read_ms = (IO.now() - read_start) * 1000;
        if (!read_ok) return false;
        for (ids) |id| {
            if (id < 0 or @as(u32, @intCast(id)) >= expert_count) {
                _ = c.fprintf(c.__stderrp, "ds4: Metal streaming prefill selected page-in expert id %d is outside 0..%u at layer %u\n", id, expert_count, index);
                return false;
            }
            if (!seen[@intCast(id)]) {
                seen[@intCast(id)] = true;
                job.unique += 1;
            }
        }
    }
    if (job.unique == 0) return true;
    const ranges = alloc(Range, @as(usize, expert_count) * 3);
    var count: u32 = 0;
    var ok = true;
    const gap = if (madvise) selectedGap() else 0;
    var expert: u32 = 0;
    while (expert < expert_count) {
        while (expert < expert_count and !seen[expert]) expert += 1;
        if (expert >= expert_count) break;
        const first = expert;
        var last = expert;
        var skipped: u32 = 0;
        expert += 1;
        while (expert < expert_count) : (expert += 1) {
            if (seen[expert]) {
                last = expert;
                skipped = 0;
            } else if (skipped < gap) skipped += 1 else break;
        }
        const number: u64 = @as(u64, last) - first + 1;
        if (first > maximum / gate_bytes or first > maximum / down_bytes or number > maximum / gate_bytes or number > maximum / down_bytes) {
            ok = false;
            break;
        }
        const gate_relative = first * gate_bytes;
        const down_relative = first * down_bytes;
        if (gate_relative > maximum - layer.*.ffn_gate_exps.*.abs_offset or gate_relative > maximum - layer.*.ffn_up_exps.*.abs_offset or down_relative > maximum - layer.*.ffn_down_exps.*.abs_offset) {
            ok = false;
            break;
        }
        const gate_size = number * gate_bytes;
        const down_size = number * down_bytes;
        ranges[count] = .{ .off = layer.*.ffn_gate_exps.*.abs_offset + gate_relative, .size = gate_size };
        ranges[count + 1] = .{ .off = layer.*.ffn_up_exps.*.abs_offset + gate_relative, .size = gate_size };
        ranges[count + 2] = .{ .off = layer.*.ffn_down_exps.*.abs_offset + down_relative, .size = down_size };
        count += 3;
        job.bytes +|= gate_size *| 2 +| down_size;
    }
    if (!ok or count == 0) {
        allocator.free(ranges);
        if (!ok) _ = c.fprintf(c.__stderrp, "ds4: Metal streaming prefill selected page-in offset overflow\n");
        return ok;
    }
    job.model = model;
    job.ranges = ranges.ptr;
    job.n_ranges = count;
    job.n_threads = selectedThreads(madvise);
    return spawn(job, true);
}

pub fn join(maybe_job: ?*Job, selected: bool) bool {
    const job = maybe_job orelse return true;
    if (!job.started) return true;
    const start = if (job.profile) IO.now() else 0;
    var first_error: c_int = 0;
    var ok = true;
    if (job.n_threads <= 1) {
        first_error = c.pthread_join(job.thread, null);
        ok = first_error == 0 and job.ok;
    } else {
        job.touched = 0;
        job.thread_ms = 0;
        job.sink = 0;
        for (0..job.n_threads) |index| {
            const result = c.pthread_join(job.threads[index], null);
            if (result != 0 and first_error == 0) first_error = result;
            if (result != 0 or !job.workers[index].ok) ok = false;
            job.touched +|= job.workers[index].touched;
            job.thread_ms = @max(job.thread_ms, job.workers[index].thread_ms);
            job.sink ^= job.workers[index].sink;
        }
    }
    if (job.profile) {
        const wait = (IO.now() - start) * 1000;
        const kind: [*:0]const u8 = if (job.pread_only) "pread" else if (job.readahead_only) "readahead" else if (job.madvise_only) "madvise" else "page-in";
        const label: [*:0]const u8 = if (job.pread_only) "read" else if (job.readahead_only) "requested" else if (job.madvise_only) "advised" else "touched";
        const bytes = @as(f64, @floatFromInt(job.bytes)) / 1073741824.0;
        const touched = @as(f64, @floatFromInt(job.touched)) / 1073741824.0;
        if (selected) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal streaming prefill selected %s layer=%u tokens=%u unique=%u ranges=%u bytes=%.2f GiB read=%.3f ms wait=%.3f ms thread=%.3f ms %s=%.2f GiB ok=%d\n", kind, job.layer, job.n_tokens, job.unique, job.n_ranges, bytes, job.read_ms, wait, job.thread_ms, label, touched, @as(c_int, @intFromBool(ok)));
        } else {
            _ = c.fprintf(c.__stderrp, "ds4: Metal streaming prefill layer %s layer=%u tokens=%u threads=%u ranges=%u bytes=%.2f GiB wait=%.3f ms thread=%.3f ms %s=%.2f GiB ok=%d\n", kind, job.layer, job.n_tokens, @max(job.n_threads, 1), job.n_ranges, bytes, wait, job.thread_ms, label, touched, @as(c_int, @intFromBool(ok)));
        }
    }
    if (first_error != 0) {
        const message: [*:0]const u8 = if (selected) "ds4: Metal streaming prefill selected page-in join failed: %s\n" else "ds4: Metal streaming prefill layer page-in join failed: %s\n";
        _ = c.fprintf(c.__stderrp, message, c.strerror(first_error));
    }
    release(job);
    return ok;
}

fn find(slots: []Slot, index: u32) ?*Slot {
    for (slots) |*slot| if (slot.active and slot.layer == index) return slot;
    return null;
}
pub fn prepareStart(model: [*c]const Model, weights: [*c]const Weights, index: u32, rows: u32, flags: u32, have_graph: bool, streaming: bool, layer_count: u32, slots: []Slot) bool {
    if (index >= layer_count or find(slots, index) != null) return true;
    for (slots) |*slot| {
        if (slot.active) continue;
        slot.* = std.mem.zeroes(Slot);
        slot.layer = index;
        if (!layerStart(model, weights, index, rows, flags, have_graph, streaming, layer_count, &slot.job)) {
            slot.* = std.mem.zeroes(Slot);
            return false;
        }
        slot.active = slot.job.started;
        return true;
    }
    _ = c.fprintf(c.__stderrp, "ds4: Metal streaming prefill prepare queue is full before layer %u\n", index);
    return false;
}
pub fn prepareJoin(model: [*c]const Model, weights: [*c]const Weights, index: u32, rows: u32, flags: u32, have_graph: bool, streaming: bool, layer_count: u32, slots: []Slot) bool {
    if (find(slots, index)) |slot| {
        const ok = join(&slot.job, false);
        slot.* = std.mem.zeroes(Slot);
        return ok;
    }
    var job = std.mem.zeroes(Job);
    return layerStart(model, weights, index, rows, flags, have_graph, streaming, layer_count, &job) and join(&job, false);
}
pub fn prepareJoinAll(slots: []Slot) bool {
    var ok = true;
    for (slots) |*slot| {
        if (!slot.active) continue;
        const joined = join(&slot.job, false);
        ok = joined and ok;
        slot.* = std.mem.zeroes(Slot);
    }
    return ok;
}
