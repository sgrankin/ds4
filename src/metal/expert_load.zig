//! Complete expert-load transactions, composing the actual cache and storage
//! owners. Pending state is shared with C; no cache/model/session mirror exists.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const Cache = @import("expert_cache.zig");
const Storage = @import("expert_storage.zig").Storage;
const IO = @import("expert_io.zig");
pub const Table = c.ds4_gpu_stream_expert_table;
pub const Entry = Cache.Entry;
pub const Reuse = Cache.Reuse;
pub const Pending = c.ds4_gpu_stream_expert_pending_load;
pub const Task = c.ds4_gpu_stream_expert_pread_task;
pub const selected_max = c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED;
pub const experts_max = c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT;
const max = std.math.maxInt(u64);
const invalid = std.math.maxInt(u32);
pub fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
pub fn slice(pointer: [*c]const i32, count: u32) []const i32 {
    return if (pointer == null or count == 0) &.{} else pointer[0..count];
}
pub fn ready(group: *const Reuse) bool {
    return group.gate_buffer != null and group.up_buffer != null and group.down_buffer != null;
}
fn bit(slot: usize) u32 {
    return @as(u32, 1) << @as(u5, @intCast(slot));
}
pub const Offsets = struct { gate: u64, up: u64, down: u64 };
pub fn offsets(table: *const Table, expert: u32) ?Offsets {
    if (expert > max / table.gate_expert_bytes or expert > max / table.down_expert_bytes) return null;
    const gate = @as(u64, expert) * table.gate_expert_bytes;
    const down = @as(u64, expert) * table.down_expert_bytes;
    if (gate > max - table.gate_offset or gate > max - table.up_offset or down > max - table.down_offset) return null;
    return .{ .gate = table.gate_offset + gate, .up = table.up_offset + gate, .down = table.down_offset + down };
}
pub fn identity(table: *const Table, at: Offsets) Cache.Identity {
    return .{ .model = table.model_map, .model_bytes = table.model_size, .gate = at.gate, .up = at.up, .down = at.down, .gate_bytes = table.gate_expert_bytes, .down_bytes = table.down_expert_bytes };
}
pub fn batchReuse(gate: u64, down: u64) bool {
    return gate <= (max - down) / 2 and gate * 2 + down <= 16 * 1024 * 1024;
}
fn destination(buffer: Cache.Buffer, inner: usize) [*c]u8 {
    const memory = c.ds4_metal_expert_buffer_storage(buffer);
    // Preserve the original contents+inner expression; the shared buffer owner
    // holds this borrowed CPU pointer throughout all reads.
    return @ptrFromInt(@intFromPtr(memory.data) +% inner);
}
pub fn destinations(buffers: *const Reuse) ?[3][*c]u8 {
    const result = [3][*c]u8{ destination(buffers.gate_buffer, buffers.gate_inner), destination(buffers.up_buffer, buffers.up_inner), destination(buffers.down_buffer, buffers.down_inner) };
    return if (result[0] == null or result[1] == null or result[2] == null) null else result;
}
pub fn appendTasks(tasks: []Task, count: *u32, targets: [3][*c]u8, at: Offsets, gate: u64, down: u64) void {
    tasks[count.*] = std.mem.zeroes(Task);
    tasks[count.*].offset = at.gate;
    tasks[count.*].len = gate;
    tasks[count.*].dst = targets[0];
    count.* += 1;
    tasks[count.*] = std.mem.zeroes(Task);
    tasks[count.*].offset = at.up;
    tasks[count.*].len = gate;
    tasks[count.*].dst = targets[1];
    count.* += 1;
    tasks[count.*] = std.mem.zeroes(Task);
    tasks[count.*].offset = at.down;
    tasks[count.*].len = down;
    tasks[count.*].dst = targets[2];
    count.* += 1;
}
pub fn moveGroup(out: *Reuse, source: *const Reuse) void {
    c.ds4_metal_expert_reuse_assign(out, source.gate_buffer, source.up_buffer, source.down_buffer, source.gate_inner, source.up_inner, source.down_inner);
}
pub fn wrap(table: *const Table, at: Offsets, gate_bytes: u64, down_bytes: u64, out: *Reuse) bool {
    var gate_inner: u64 = 0;
    var up_inner: u64 = 0;
    var down_inner: u64 = 0;
    const gate = api.retainModelRange(table.model_map, table.model_size, at.gate, gate_bytes, &gate_inner);
    defer if (gate) |buffer| api.releaseModelBuffer(buffer);
    const up = api.retainModelRange(table.model_map, table.model_size, at.up, gate_bytes, &up_inner);
    defer if (up) |buffer| api.releaseModelBuffer(buffer);
    const down = api.retainModelRange(table.model_map, table.model_size, at.down, down_bytes, &down_inner);
    defer if (down) |buffer| api.releaseModelBuffer(buffer);
    if (gate == null or up == null or down == null) return false;
    c.ds4_metal_expert_reuse_assign(out, @ptrCast(gate.?), @ptrCast(up.?), @ptrCast(down.?), @intCast(gate_inner), @intCast(up_inner), @intCast(down_inner));
    return true;
}
pub const Context = struct {
    view: c.ds4_metal_expert_load_view,
    cache: Cache.Cache,
    load: *c.ds4_metal_expert_load_state,
    pub fn current() Context {
        const view = c.ds4_metal_expert_load_driver();
        return .{ .view = view, .cache = .{ .state = @ptrCast(view.cache), .streaming = view.streaming }, .load = @ptrCast(view.load) };
    }
    pub fn storage(self: Context) Storage {
        return .{ .cache = self.cache, .device = self.view.device_ready };
    }
    pub fn initialize(self: *Context) bool {
        if (api.init() == 0) return false;
        self.view = c.ds4_metal_expert_load_driver();
        return true;
    }
    pub fn timingLoad(self: Context, prepare: f64, read: f64, install: f64) void {
        if (!Cache.timingSummary()) return;
        const state = self.cache.state;
        state.timing_load_calls +%= 1;
        state.timing_load_prepare_ms += prepare;
        state.timing_load_pread_ms += read;
        state.timing_load_install_ms += install;
    }
    pub fn timingBuffer(self: Context, duration: f64) void {
        if (Cache.timingSummary()) {
            self.cache.state.timing_prepare_buffer_calls +%= 1;
            self.cache.state.timing_prepare_buffer_ms += duration;
        }
    }
    pub fn timingTasks(self: Context, count: u32, duration: f64) void {
        if (Cache.timingSummary()) {
            self.cache.state.timing_prepare_task_experts +%= count;
            self.cache.state.timing_prepare_task_ms += duration;
        }
    }
    fn timingReuse(self: Context, duration: f64) void {
        if (Cache.timingSummary()) {
            self.cache.state.timing_prepare_batch_reuse_calls +%= 1;
            self.cache.state.timing_prepare_batch_reuse_ms += duration;
        }
    }
    pub fn matches(self: Context, table: *const Table, ids: []const i32) bool {
        const pending = &self.load.pending;
        return pending.active != 0 and pending.model_map == table.model_map and pending.model_size == table.model_size and pending.layer == table.layer and pending.n_total_expert == table.n_total_expert and pending.n_selected == ids.len and pending.gate_expert_bytes == table.gate_expert_bytes and pending.down_expert_bytes == table.down_expert_bytes and std.mem.eql(i32, pending.selected_ids[0..ids.len], ids);
    }
    fn installPending(self: Context, pending: *Pending, entries: [*c][*c]Entry, elapsed: f64) bool {
        if (pending.n_loads == 0) return true;
        var bytes: u64 = 0;
        var ok = true;
        for (pending.tasks[0..pending.n_tasks]) |task| {
            if (task.ok == 0) ok = false;
            bytes +|= task.read_bytes;
        }
        if (!ok) return false;
        IO.noteRead(self.cache.state, pending.layer, bytes, elapsed);
        const timing = Cache.timingSummary();
        const start = if (timing) IO.now() else 0;
        var loaded: [selected_max]?*Entry = @splat(null);
        for (0..pending.n_loads) |index| {
            const slot = pending.load_slots[index];
            const key = Cache.Identity{ .model = pending.model_map, .model_bytes = pending.model_size, .gate = pending.gate_abs_offsets[slot], .up = pending.up_abs_offsets[slot], .down = pending.down_abs_offsets[slot], .gate_bytes = pending.gate_expert_bytes, .down_bytes = pending.down_expert_bytes };
            const entry = self.cache.install(key, pending.layer, @intCast(pending.selected_ids[slot]), pending.gate_bufs[index], pending.up_bufs[index], pending.down_bufs[index], pending.gate_inners[index], pending.up_inners[index], pending.down_inners[index]) orelse return false;
            loaded[slot] = entry;
            if (entries != null) entries[slot] = entry;
        }
        for (0..pending.n_selected) |slot| {
            if (pending.missing_mask & bit(slot) == 0 or (entries != null and entries[slot] != null)) continue;
            const source = pending.source_slots[slot];
            if (source >= pending.n_selected) return false;
            const entry: *Entry = if (entries != null and entries[source] != null) @ptrCast(entries[source]) else loaded[source] orelse return false;
            entry.use_count +%= 1;
            if (entries != null) entries[slot] = entry;
        }
        if (timing) self.timingLoad(pending.prepare_ms, elapsed, IO.now() - start);
        if (present("DS4_METAL_STREAMING_EXPERT_EARLY_LOAD_PROFILE")) _ = c.fprintf(c.__stderrp, "ds4: Metal streaming expert early-load finish layer=%u experts=%u tensors=%u bytes=%.2f GiB wall=%.3f ms\n", pending.layer, pending.n_loads, pending.n_tasks, @as(f64, @floatFromInt(bytes)) / 1073741824, elapsed);
        return true;
    }
    pub fn finish(self: Context, entries: [*c][*c]Entry) bool {
        const pending = &self.load.pending;
        if (pending.active == 0) return true;
        const start = pending.start_ms;
        const timing = Cache.timingSummary();
        const join = if (timing) IO.now() else 0;
        const ok = IO.wait(&self.load.pool);
        if (timing) {
            self.cache.state.pending_wait_ms += IO.now() - join;
            self.cache.state.pending_joins +%= 1;
        }
        if (!ok) {
            c.ds4_metal_expert_pending_retire(pending);
            pending.active = 0;
            return false;
        }
        const elapsed = IO.now() - start;
        pending.active = 0;
        const installed = self.installPending(pending, entries, elapsed);
        c.ds4_metal_expert_pending_retire(pending);
        pending.n_tasks = 0;
        pending.n_loads = 0;
        pending.prepare_ms = 0;
        return installed;
    }
    pub fn clear(self: Context) void {
        if (self.load.pending.active == 0) c.ds4_metal_expert_pending_retire(&self.load.pending) else _ = self.finish(null);
    }
    pub fn begin(self: *Context, maybe_table: ?*const Table, ids_pointer: [*c]const i32, count: u32) bool {
        if (!self.cache.streaming or present("DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD")) return true;
        const table = maybe_table orelse return false;
        if (table.model_map == null or
            ids_pointer == null or
            table.layer >= c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER or
            count == 0 or
            count > selected_max or
            table.n_total_expert == 0 or
            table.n_total_expert > experts_max or
            !self.cache.noteSize(table.gate_expert_bytes, table.down_expert_bytes) or
            self.cache.effectiveCap(table.n_total_expert, count) == 0) return true;
        if (!self.initialize()) return false;
        const ids = ids_pointer[0..count];
        if (self.matches(table, ids)) return true;
        self.clear();
        const pending = &self.load.pending;
        pending.active = 0;
        pending.model_map = table.model_map;
        pending.model_size = table.model_size;
        pending.layer = table.layer;
        pending.n_total_expert = table.n_total_expert;
        pending.n_selected = count;
        pending.missing_mask = 0;
        pending.n_loads = 0;
        pending.n_tasks = 0;
        pending.gate_expert_bytes = table.gate_expert_bytes;
        pending.down_expert_bytes = table.down_expert_bytes;
        pending.prepare_ms = 0;
        const timing = Cache.timingSummary();
        const start = if (timing) IO.now() else 0;
        @memset(&pending.selected_ids, -1);
        @memset(&pending.load_slots, 0);
        @memset(&pending.source_slots, invalid);
        @memset(&pending.gate_abs_offsets, 0);
        @memset(&pending.up_abs_offsets, 0);
        @memset(&pending.down_abs_offsets, 0);
        c.ds4_metal_expert_pending_retire(pending);
        @memset(std.mem.asBytes(&pending.tasks), 0);
        for (ids, 0..) |expert, slot| {
            if (expert < 0 or @as(u32, @intCast(expert)) >= table.n_total_expert) {
                _ = c.fprintf(c.__stderrp, "ds4: Metal streaming early-load expert id %d is outside 0..%u\n", expert, table.n_total_expert);
                return false;
            }
            pending.selected_ids[slot] = expert;
            const at = offsets(table, @intCast(expert)) orelse {
                _ = c.fprintf(c.__stderrp, "ds4: Metal streaming early-load offset overflow\n");
                return false;
            };
            pending.gate_abs_offsets[slot] = at.gate;
            pending.up_abs_offsets[slot] = at.up;
            pending.down_abs_offsets[slot] = at.down;
            if (Cache.matches(&self.cache.state.entries[table.layer][@intCast(expert)], identity(table, at))) continue;
            var source: u32 = invalid;
            for (ids[0..slot], 0..) |previous, before| {
                if (previous == expert and pending.gate_abs_offsets[before] == at.gate and pending.up_abs_offsets[before] == at.up and pending.down_abs_offsets[before] == at.down and pending.missing_mask & bit(before) != 0) {
                    source = if (pending.source_slots[before] != invalid) pending.source_slots[before] else @intCast(before);
                    break;
                }
            }
            pending.source_slots[slot] = if (source != invalid) source else @intCast(slot);
            pending.missing_mask |= bit(slot);
            if (source != invalid) continue;
            pending.load_slots[pending.n_loads] = @intCast(slot);
            pending.n_loads += 1;
        }
        if (pending.n_loads == 0) return true;
        const budget = self.cache.budget();
        var reserved = self.cache.state.cache_entry_count;
        var reuse: [selected_max]Reuse = std.mem.zeroes([selected_max]Reuse);
        defer for (&reuse) |*group| c.ds4_metal_expert_reuse_reset(group);
        var reused: u32 = 0;
        if (budget != 0 and reserved >= budget and pending.n_loads > 1 and batchReuse(table.gate_expert_bytes, table.down_expert_bytes)) {
            const before = if (timing) IO.now() else 0;
            reused = self.cache.takeBatch(pending.n_loads, table.layer, ids, table.gate_expert_bytes, table.down_expert_bytes, &reuse);
            if (timing) self.timingReuse(IO.now() - before);
        }
        for (0..pending.n_loads) |index| {
            const slot = pending.load_slots[index];
            const expert: u32 = @intCast(pending.selected_ids[slot]);
            const force = budget != 0 and reserved >= budget;
            const at = Offsets{ .gate = pending.gate_abs_offsets[slot], .up = pending.up_abs_offsets[slot], .down = pending.down_abs_offsets[slot] };
            IO.readahead(self.view, at.gate, table.gate_expert_bytes);
            IO.readahead(self.view, at.up, table.gate_expert_bytes);
            IO.readahead(self.view, at.down, table.down_expert_bytes);
            var buffers = std.mem.zeroes(Reuse);
            defer c.ds4_metal_expert_reuse_reset(&buffers);
            if (index < reused and ready(&reuse[index])) moveGroup(&buffers, &reuse[index]) else {
                const before = if (timing) IO.now() else 0;
                const prepared = self.storage().prepare(table.layer, expert, table.layer, ids, table.gate_expert_bytes, table.down_expert_bytes, force, &buffers);
                if (timing) self.timingBuffer(IO.now() - before);
                if (!prepared) {
                    c.ds4_metal_expert_pending_retire(pending);
                    return false;
                }
            }
            c.ds4_metal_expert_pending_assign(pending, @intCast(index), &buffers);
            if (!force) reserved +|= 1;
            const targets = destinations(&buffers) orelse {
                c.ds4_metal_expert_pending_retire(pending);
                return false;
            };
            const before = if (timing) IO.now() else 0;
            appendTasks(&pending.tasks, &pending.n_tasks, targets, at, table.gate_expert_bytes, table.down_expert_bytes);
            if (timing) self.timingTasks(1, IO.now() - before);
        }
        if (present("DS4_METAL_V41_STAGED_EXPERT_LOAD")) {
            var ordered: [selected_max * 3]Task = undefined;
            var at: usize = 0;
            for (pending.tasks[0..pending.n_tasks], 0..) |task, index| if (index % 3 != 2) {
                ordered[at] = task;
                ordered[at].gate_stage = 1;
                at += 1;
            };
            var index: usize = 2;
            while (index < pending.n_tasks) : (index += 3) {
                ordered[at] = pending.tasks[index];
                at += 1;
            }
            @memcpy(pending.tasks[0..pending.n_tasks], ordered[0..pending.n_tasks]);
        }
        const workers = IO.threadCount(pending.n_tasks);
        pending.start_ms = IO.now();
        if (timing) pending.prepare_ms = pending.start_ms - start;
        if (IO.begin(&self.load.pool, &pending.tasks, pending.n_tasks, workers)) {
            pending.active = 1;
            if (present("DS4_METAL_STREAMING_EXPERT_EARLY_LOAD_PROFILE")) _ = c.fprintf(c.__stderrp, "ds4: Metal streaming expert early-load begin layer=%u experts=%u tensors=%u threads=%u\n", table.layer, pending.n_loads, pending.n_tasks, workers);
            return true;
        }
        var bytes: u64 = 0;
        var elapsed: f64 = 0;
        if (!IO.readTasks(&self.load.pool, &pending.tasks, pending.n_tasks, &bytes, &elapsed) or !self.installPending(pending, null, elapsed)) {
            c.ds4_metal_expert_pending_retire(pending);
            return false;
        }
        c.ds4_metal_expert_pending_retire(pending);
        pending.n_tasks = 0;
        pending.n_loads = 0;
        pending.prepare_ms = 0;
        return true;
    }
};
