//! Synchronous selected misses, hotlist seeding and GLM early-load integration.
//! Buffer ownership uses the real reusable/pending records and ARC actions.
const std = @import("std");
const c = @import("metal_c");
const api = @import("bridge.zig");
const Core = @import("expert_load.zig");
const Cache = @import("expert_cache.zig");
const IO = @import("expert_io.zig");
const Context = Core.Context;
const Table = Core.Table;
const Entry = Core.Entry;
const Reuse = Core.Reuse;
const maximum = Core.selected_max;
const invalid = std.math.maxInt(u32);
const max = std.math.maxInt(u64);
const present = Core.present;
fn bit(index: usize) u32 {
    return @as(u32, 1) << @as(u5, @intCast(index));
}
fn ids(pointer: [*c]const i32, count: u32) []const i32 {
    return Core.slice(pointer, count);
}
fn at(gate: [*c]const u64, up: [*c]const u64, down: [*c]const u64, index: usize) Core.Offsets {
    return .{ .gate = gate[index], .up = up[index], .down = down[index] };
}
fn same(a: Core.Offsets, b: Core.Offsets) bool {
    return a.gate == b.gate and a.up == b.up and a.down == b.down;
}
fn bind(entries: [*c][*c]Entry, selected: u32, mask: u32, sources: *const [maximum]u32) bool {
    for (0..selected) |slot| {
        if (mask & bit(slot) == 0 or entries[slot] != null) continue;
        const source = sources[slot];
        if (source >= selected or entries[source] == null) return false;
        entries[slot] = entries[source];
        entries[slot].*.use_count +%= 1;
    }
    return true;
}
pub fn missing(
    ctx: *Context,
    table: *const Table,
    selected: [*c]const i32,
    count: u32,
    gate: [*c]const u64,
    up: [*c]const u64,
    down: [*c]const u64,
    wanted: u32,
    gpu_source: bool,
    entries: [*c][*c]Entry,
) bool {
    if (!ctx.cache.streaming or
        table.model_map == null or
        selected == null or
        gate == null or
        up == null or
        down == null or
        entries == null or
        table.layer >= c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER or
        count == 0 or
        count > maximum or
        table.n_total_expert == 0 or
        table.n_total_expert > Core.experts_max or
        !ctx.cache.noteSize(table.gate_expert_bytes, table.down_expert_bytes) or
        ctx.cache.effectiveCap(table.n_total_expert, count) == 0) return false;
    var mask = wanted & (bit(count) - 1);
    if (mask == 0) return true;
    if (ctx.load.pending.active != 0) {
        if (ctx.matches(table, selected[0..count])) {
            if (!ctx.finish(entries)) return false;
            for (0..count) |slot| if (mask & bit(slot) != 0 and entries[slot] != null) {
                mask &= ~bit(slot);
            };
            if (mask == 0) return true;
        } else ctx.clear();
    }
    const timing = Cache.timingSummary();
    var started = if (timing) IO.now() else 0;
    var prepare_ms: f64 = 0;
    var loads: [maximum]u32 = @splat(0);
    var sources: [maximum]u32 = @splat(invalid);
    var load_count: u32 = 0;
    for (0..count) |slot| {
        if (mask & bit(slot) == 0) continue;
        if (entries[slot] != null) {
            sources[slot] = @intCast(slot);
            continue;
        }
        if (selected[slot] < 0 or @as(u32, @intCast(selected[slot])) >= table.n_total_expert) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal streaming selected missing expert id %d is outside 0..%u\n", selected[slot], table.n_total_expert);
            return false;
        }
        var source: u32 = invalid;
        for (0..slot) |previous| {
            if (selected[previous] == selected[slot] and same(at(gate, up, down, previous), at(gate, up, down, slot)) and (entries[previous] != null or
                mask & bit(previous) != 0))
            {
                source = if (sources[previous] != invalid) sources[previous] else @intCast(previous);
                break;
            }
        }
        sources[slot] = if (source != invalid) source else @intCast(slot);
        if (source == invalid) {
            loads[load_count] = @intCast(slot);
            load_count += 1;
        }
    }
    if (load_count == 0) return bind(entries, count, mask, &sources);
    var buffers = std.mem.zeroes([maximum]Reuse);
    defer for (&buffers) |*group| c.ds4_metal_expert_reuse_reset(group);
    var reuse = std.mem.zeroes([maximum]Reuse);
    defer for (&reuse) |*group| c.ds4_metal_expert_reuse_reset(group);
    var tasks = std.mem.zeroes([maximum * 3]Core.Task);
    var task_count: u32 = 0;
    const budget = ctx.cache.budget();
    var reserved = ctx.cache.state.cache_entry_count;
    var reused: u32 = 0;
    if (budget != 0 and reserved >= budget and load_count > 1 and Core.batchReuse(table.gate_expert_bytes, table.down_expert_bytes)) {
        const before = if (timing) IO.now() else 0;
        reused = ctx.cache.takeBatch(load_count, table.layer, selected[0..count], table.gate_expert_bytes, table.down_expert_bytes, &reuse);
        if (timing) {
            ctx.cache.state.timing_prepare_batch_reuse_calls +%= 1;
            ctx.cache.state.timing_prepare_batch_reuse_ms += IO.now() - before;
        }
    }
    for (0..load_count) |index| {
        const slot = loads[index];
        const point = at(gate, up, down, slot);
        const force = budget != 0 and reserved >= budget;
        if (!gpu_source) {
            IO.readahead(ctx.view, point.gate, table.gate_expert_bytes);
            IO.readahead(ctx.view, point.up, table.gate_expert_bytes);
            IO.readahead(ctx.view, point.down, table.down_expert_bytes);
        }
        if (index < reused and Core.ready(&reuse[index])) Core.moveGroup(&buffers[index], &reuse[index]) else {
            const before = if (timing) IO.now() else 0;
            const prepared = ctx.storage().prepare(table.layer, @intCast(selected[slot]), table.layer, selected[0..count], table.gate_expert_bytes, table.down_expert_bytes, force, &buffers[index]);
            if (timing) ctx.timingBuffer(IO.now() - before);
            if (!prepared) return false;
        }
        if (!force) reserved +|= 1;
        if (!Core.ready(&buffers[index])) return false;
        const targets = Core.destinations(&buffers[index]) orelse return false;
        if (gpu_source and (point.gate > table.model_size or
            table.gate_expert_bytes > table.model_size - point.gate or
            point.up > table.model_size or
            table.gate_expert_bytes > table.model_size - point.up or
            point.down > table.model_size or
            table.down_expert_bytes > table.model_size - point.down))
        {
            _ = c.fprintf(c.__stderrp, "ds4: Metal streaming mapped expert source is outside the model mapping\n");
            return false;
        }
        const before = if (timing) IO.now() else 0;
        Core.appendTasks(&tasks, &task_count, targets, point, table.gate_expert_bytes, table.down_expert_bytes);
        if (timing) ctx.timingTasks(1, IO.now() - before);
    }
    if (timing) {
        const next = IO.now();
        prepare_ms = next - started;
        started = next;
    }
    var bytes: u64 = 0;
    var read_ms: f64 = 0;
    if (gpu_source) {
        if (!c.ds4_metal_expert_load_driver().batch_active or
            table.gate_expert_bytes > std.math.maxInt(usize) or
            table.down_expert_bytes > std.math.maxInt(usize)) return false;
        var source_buffers = std.mem.zeroes([maximum]c.ds4_metal_expert_copy_sources);
        defer for (&source_buffers) |*source| {
            if (source.gate != null) api.releaseModelBuffer(source.gate);
            if (source.up != null) api.releaseModelBuffer(source.up);
            if (source.down != null) api.releaseModelBuffer(source.down);
        };
        for (0..load_count) |index| {
            const point = at(gate, up, down, loads[index]);
            var gate_inner: u64 = 0;
            var up_inner: u64 = 0;
            var down_inner: u64 = 0;
            source_buffers[index].gate = api.retainModelRange(table.model_map, table.model_size, point.gate, table.gate_expert_bytes, &gate_inner);
            source_buffers[index].up = api.retainModelRange(table.model_map, table.model_size, point.up, table.gate_expert_bytes, &up_inner);
            source_buffers[index].down = api.retainModelRange(table.model_map, table.model_size, point.down, table.down_expert_bytes, &down_inner);
            if (source_buffers[index].gate == null or
                source_buffers[index].up == null or
                source_buffers[index].down == null or
                gate_inner > std.math.maxInt(usize) or
                up_inner > std.math.maxInt(usize) or
                down_inner > std.math.maxInt(usize)) return false;
            source_buffers[index].gate_inner = @intCast(gate_inner);
            source_buffers[index].up_inner = @intCast(up_inner);
            source_buffers[index].down_inner = @intCast(down_inner);
        }
        if (!c.ds4_metal_expert_seed_blit(&source_buffers, &buffers, load_count, table.gate_expert_bytes, table.down_expert_bytes)) return false;
        const per_expert = if (table.gate_expert_bytes > (max - table.down_expert_bytes) / 2) max else table.gate_expert_bytes * 2 + table.down_expert_bytes;
        for (0..load_count) |_| bytes +|= per_expert;
    } else if (!IO.readTasks(&ctx.load.pool, &tasks, task_count, &bytes, &read_ms)) return false;
    if (timing) started = IO.now();
    if (!gpu_source) IO.noteRead(ctx.cache.state, table.layer, bytes, read_ms);
    if (gpu_source and present("DS4_METAL_STREAMING_PREFILL_CACHE_SEED_PROFILE")) _ = c.fprintf(c.__stderrp, "ds4: Metal streaming expert GPU cache copy layer=%u experts=%u bytes=%.2f GiB\n", table.layer, load_count, @as(f64, @floatFromInt(bytes)) / 1073741824) else if (!gpu_source and present("DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE")) _ = c.fprintf(c.__stderrp, "ds4: Metal streaming expert parallel pread layer=%u experts=%u tensors=%u threads=%u bytes=%.2f GiB wall=%.3f ms\n", table.layer, load_count, task_count, IO.threadCount(task_count), @as(f64, @floatFromInt(bytes)) / 1073741824, read_ms);
    for (0..load_count) |index| {
        const slot = loads[index];
        const group = &buffers[index];
        entries[slot] = ctx.cache.install(Core.identity(table, at(gate, up, down, slot)), table.layer, @intCast(selected[slot]), group.gate_buffer, group.up_buffer, group.down_buffer, group.gate_inner, group.up_inner, group.down_inner) orelse return false;
    }
    if (!bind(entries, count, mask, &sources)) return false;
    for (0..count) |slot| if (mask & bit(slot) != 0 and entries[slot] == null) return false;
    if (timing and !gpu_source) ctx.timingLoad(prepare_ms, read_ms, IO.now() - started);
    return true;
}
pub fn get(ctx: *Context, table: *const Table, expert: u32, selected: u32, point: Core.Offsets, protect_input: []const i32) ?*Entry {
    if (!ctx.cache.streaming or
        table.layer >= c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER or
        expert >= Core.experts_max or
        expert >= table.n_total_expert or
        !ctx.cache.noteSize(table.gate_expert_bytes, table.down_expert_bytes) or
        ctx.cache.effectiveCap(table.n_total_expert, selected) == 0) return null;
    const old = &ctx.cache.state.entries[table.layer][expert];
    if (Cache.matches(old, Core.identity(table, point))) {
        ctx.cache.state.cache_clock +%= 1;
        old.last_used = ctx.cache.state.cache_clock;
        old.use_count +%= 1;
        ctx.cache.state.cache_hits +%= 1;
        ctx.cache.state.cache_layer_hits[table.layer] +%= 1;
        return old;
    }
    IO.readahead(ctx.view, point.gate, table.gate_expert_bytes);
    IO.readahead(ctx.view, point.up, table.gate_expert_bytes);
    IO.readahead(ctx.view, point.down, table.down_expert_bytes);
    var buffers = std.mem.zeroes(Reuse);
    defer c.ds4_metal_expert_reuse_reset(&buffers);
    const one = [_]i32{@bitCast(expert)};
    const protect = if (protect_input.len == 0) &one else protect_input;
    const budget = ctx.cache.budget();
    if (!ctx.storage().prepare(table.layer, expert, table.layer, protect, table.gate_expert_bytes, table.down_expert_bytes, budget != 0 and ctx.cache.state.cache_entry_count >= budget, &buffers) or
        !Core.ready(&buffers)) return null;
    var tasks = std.mem.zeroes([3]Core.Task);
    var count: u32 = 0;
    const targets = Core.destinations(&buffers) orelse return null;
    Core.appendTasks(&tasks, &count, targets, point, table.gate_expert_bytes, table.down_expert_bytes);
    var bytes: u64 = 0;
    var elapsed: f64 = 0;
    if (!IO.readTasks(&ctx.load.pool, &tasks, 3, &bytes, &elapsed)) return null;
    IO.noteRead(ctx.cache.state, table.layer, bytes, elapsed);
    if (present("DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE")) _ = c.fprintf(c.__stderrp, "ds4: Metal streaming expert parallel pread layer=%u experts=1 tensors=3 threads=%u bytes=%.2f GiB wall=%.3f ms\n", table.layer, IO.threadCount(3), @as(f64, @floatFromInt(bytes)) / 1073741824, elapsed);
    return ctx.cache.install(Core.identity(table, point), table.layer, expert, buffers.gate_buffer, buffers.up_buffer, buffers.down_buffer, buffers.gate_inner, buffers.up_inner, buffers.down_inner);
}
pub fn prefetchSet(ctx: *Context, maybe_table: ?*const Table, selected: [*c]const i32, count: u32) void {
    const pending = &ctx.load.prefetch;
    pending.active = 0;
    const table = maybe_table orelse return;
    if (selected == null or count == 0 or count > maximum) return;
    pending.model_map = table.model_map;
    pending.model_size = table.model_size;
    pending.layer = table.layer;
    pending.n_total_expert = table.n_total_expert;
    pending.n_selected = count;
    pending.gate_offset = table.gate_offset;
    pending.up_offset = table.up_offset;
    pending.down_offset = table.down_offset;
    pending.gate_expert_bytes = table.gate_expert_bytes;
    pending.down_expert_bytes = table.down_expert_bytes;
    @memcpy(pending.selected_ids[0..count], selected[0..count]);
    pending.active = 1;
}
pub fn prefetchTake(ctx: *Context, table: *const Table, count: u32, selected: [*c]i32) bool {
    const pending = &ctx.load.prefetch;
    if (selected == null or
        pending.active == 0 or
        pending.model_map != table.model_map or
        pending.model_size != table.model_size or
        pending.layer != table.layer or
        pending.n_total_expert != table.n_total_expert or
        pending.n_selected != count or
        pending.gate_offset != table.gate_offset or
        pending.up_offset != table.up_offset or
        pending.down_offset != table.down_offset or
        pending.gate_expert_bytes != table.gate_expert_bytes or
        pending.down_expert_bytes != table.down_expert_bytes) return false;
    @memcpy(selected[0..count], pending.selected_ids[0..count]);
    pending.active = 0;
    return true;
}
pub fn beginTensor(ctx: *Context, table: ?*const Table, selected: ?*const c.ds4_gpu_tensor, count: u32) bool {
    ctx.load.prefetch.active = 0;
    if (!ctx.cache.streaming or present("DS4_METAL_DISABLE_GLM_STREAMING_EXPERT_EARLY_LOAD")) return true;
    if (table == null or selected == null or count == 0 or count > maximum) return true;
    const timing = Cache.timingSummary();
    var start = if (timing) IO.now() else 0;
    var sync: f64 = 0;
    const batch = c.ds4_gpu_commands_active() != 0;
    if (batch) {
        if (c.ds4_gpu_end_commands() == 0) return false;
        if (timing) {
            sync = IO.now() - start;
            start = IO.now();
        }
    }
    var selected_ids: [maximum]i32 = @splat(-1);
    var ok = c.ds4_gpu_tensor_read(selected, 0, &selected_ids, @as(u64, count) * 4) != 0;
    const copy = if (timing) IO.now() - start else 0;
    if (timing) start = IO.now();
    if (ok) ok = ctx.begin(table, &selected_ids, count);
    if (timing) {
        ctx.cache.state.timing_selected_calls +%= 1;
        ctx.cache.state.timing_selected_sync_ms += sync;
        ctx.cache.state.timing_selected_copy_ms += copy;
        ctx.cache.state.timing_selected_read_ms += sync + copy;
        ctx.cache.state.timing_selected_bind_ms += IO.now() - start;
    }
    if (ok) prefetchSet(ctx, table, &selected_ids, count);
    if (batch and c.ds4_gpu_begin_commands() == 0) ok = false;
    return ok;
}
pub fn seedSelected(ctx: *Context, maybe_table: ?*const Table, selected: [*c]const i32, count: u32) bool {
    if (!ctx.cache.streaming) return true;
    const table = maybe_table orelse return false;
    if (table.model_map == null or
        selected == null or
        table.layer >= c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER or
        count == 0 or
        count > 6 or
        table.n_total_expert == 0 or
        table.n_total_expert > Core.experts_max or
        !ctx.cache.noteSize(table.gate_expert_bytes, table.down_expert_bytes) or
        ctx.cache.effectiveCap(table.n_total_expert, count) == 0) return true;
    if (!ctx.initialize()) return false;
    ctx.cache.noteSelected(table.layer, selected, count);
    for (selected[0..count]) |expert| {
        if (expert < 0 or @as(u32, @intCast(expert)) >= table.n_total_expert) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal prefill expert-cache seed selected expert id %d is outside 0..%u\n", expert, table.n_total_expert);
            return false;
        }
        const point = Core.offsets(table, @intCast(expert)) orelse {
            _ = c.fprintf(c.__stderrp, "ds4: Metal prefill expert-cache seed offset overflow\n");
            return false;
        };
        if (get(ctx, table, @intCast(expert), count, point, &.{}) == null) return false;
    }
    ctx.cache.pruneLayer(table.layer, table.n_total_expert, count, selected[0..count]);
    ctx.cache.pruneGlobal(table.layer, selected[0..count]);
    return true;
}
pub fn seedExperts(ctx: *Context, maybe_table: ?*const Table, input: [*c]const i32, priorities: [*c]const u32, count: u32, gpu_source: bool) bool {
    if (!ctx.cache.streaming) return true;
    const table = maybe_table orelse return false;
    if (table.model_map == null or
        input == null or
        table.layer >= c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER or
        count == 0 or
        table.n_total_expert == 0 or
        table.n_total_expert > Core.experts_max or
        !ctx.cache.noteSize(table.gate_expert_bytes, table.down_expert_bytes) or
        ctx.cache.effectiveCap(table.n_total_expert, 1) == 0) return true;
    if (!ctx.initialize()) return false;
    ctx.cache.maybeDecay();
    var remaining = count;
    while (remaining != 0) {
        const batch = @min(remaining, 6);
        remaining -= batch;
        var selected: [6]i32 = @splat(-1);
        var gate: [6]u64 = @splat(0);
        var up: [6]u64 = @splat(0);
        var down: [6]u64 = @splat(0);
        var entries: [6][*c]Entry = @splat(null);
        var mask: u32 = 0;
        for (0..batch) |index| {
            const expert = input[remaining + index];
            const priority: u32 = if (priorities != null) priorities[remaining + index] else 0;
            if (expert < 0 or @as(u32, @intCast(expert)) >= table.n_total_expert) {
                _ = c.fprintf(c.__stderrp, "ds4: Metal streaming hotlist seed expert id %d is outside 0..%u\n", expert, table.n_total_expert);
                return false;
            }
            const point = Core.offsets(table, @intCast(expert)) orelse {
                _ = c.fprintf(c.__stderrp, "ds4: Metal streaming hotlist seed offset overflow\n");
                return false;
            };
            ctx.cache.addHotness(table.layer, @intCast(expert), if (priority != 0) priority else 1);
            selected[index] = expert;
            gate[index] = point.gate;
            up[index] = point.up;
            down[index] = point.down;
            entries[index] = ctx.cache.peek(Core.identity(table, point), table.layer, @intCast(expert), table.n_total_expert, batch);
            if (entries[index] == null) mask |= bit(index) else if (priority != 0 and entries[index].*.use_count < priority) entries[index].*.use_count = priority;
        }
        if (mask != 0 and !missing(ctx, table, &selected, batch, &gate, &up, &down, mask, gpu_source, &entries)) return false;
        if (priorities != null) for (0..batch) |index| {
            const priority = priorities[remaining + index];
            if (entries[index] != null and priority != 0 and entries[index].*.use_count < priority) entries[index].*.use_count = priority;
        };
    }
    const protect = input[0..@min(count, 6)];
    ctx.cache.pruneLayer(table.layer, table.n_total_expert, 1, protect);
    ctx.cache.pruneGlobal(table.layer, protect);
    return true;
}
