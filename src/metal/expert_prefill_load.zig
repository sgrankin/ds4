//! Complete unique-expert prefill preparation. The real address-table owner
//! publishes handles; this transaction controls IDs, I/O and cache/view order.
const std = @import("std");
const c = @import("metal_c");
const Core = @import("expert_load.zig");
const Cache = @import("expert_cache.zig");
const IO = @import("expert_io.zig");
const Context = Core.Context;
const Table = Core.Table;
const Reuse = Core.Reuse;
const Entry = Core.Entry;
const maximum = Core.experts_max;
const max = std.math.maxInt(u64);
const present = Core.present;
pub fn prepare(
    ctx: *Context,
    table: *const Table,
    selected: ?*const c.ds4_gpu_tensor,
    tokens: u32,
    used: u32,
    addresses: *Reuse,
    resources: [*c][*c]Entry,
    resource_count: *u32,
    unique: *u32,
    overflow: *Reuse,
) bool {
    c.ds4_metal_expert_reuse_reset(overflow);
    if (!ctx.cache.streaming or
        table.model_map == null or
        selected == null or
        resources == null or
        table.layer >= c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER or
        tokens == 0 or
        used == 0 or
        used > Core.selected_max or
        table.n_total_expert == 0 or
        table.n_total_expert > maximum or
        table.gate_expert_bytes == 0 or
        table.down_expert_bytes == 0 or
        tokens > std.math.maxInt(u32) / used) return false;
    const count = @as(u64, tokens) * used;
    if (count > std.math.maxInt(usize) / 4) return false;
    var ids_storage = c.malloc(@intCast(count * 4));
    defer if (ids_storage) |pointer| c.free(pointer);
    const allocation = ids_storage orelse return false;
    const ids: [*]i32 = @ptrCast(@alignCast(allocation));
    const profile = present("DS4_METAL_STREAMING_PREFILL_BATCH_SELECTED_ADDR_PROFILE");
    const start = if (profile) IO.now() else 0;
    var ok = c.ds4_gpu_tensor_read(selected, 0, allocation, count * 4) != 0;
    const read = if (profile) IO.now() else 0;
    var seen: [maximum]bool = @splat(false);
    var frequency: [maximum]u32 = @splat(0);
    var selected_ids: [maximum]i32 = undefined;
    var selected_count: u32 = 0;
    resource_count.* = 0;
    unique.* = 0;
    if (ok) ok = c.ds4_metal_expert_load_addresses(table.layer, null);
    if (ok) for (ids[0..@as(usize, @intCast(count))]) |id| {
        if (id < 0 or @as(u32, @intCast(id)) >= table.n_total_expert) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal streaming batch selected expert id %d is outside 0..%u at layer %u\n", id, table.n_total_expert, table.layer);
            ok = false;
            break;
        }
        const expert: u32 = @intCast(id);
        frequency[expert] +%= 1;
        if (!seen[expert]) {
            seen[expert] = true;
            selected_ids[selected_count] = id;
            selected_count += 1;
        }
    };
    if (ok) ctx.cache.noteFrequency(table.layer, &frequency, table.n_total_expert);
    // Overflow views retain the identical expert bytes; they must not switch
    // this request to a different routed kernel/accumulation order.
    var view_served: u32 = 0;
    var points: [maximum]Core.Offsets = @splat(.{ .gate = 0, .up = 0, .down = 0 });
    var entries: [maximum]?*Entry = @splat(null);
    var load_unique: [maximum]u32 = @splat(0);
    var buffers = std.mem.zeroes([maximum]Reuse);
    defer for (&buffers) |*group| c.ds4_metal_expert_reuse_reset(group);
    var task_storage: ?*anyopaque = null;
    defer if (task_storage) |pointer| c.free(pointer);
    var tasks: [*c]Core.Task = null;
    var loads: u32 = 0;
    var task_count: u32 = 0;
    var prepare_ms: f64 = 0;
    var timed = if (Cache.timingSummary()) IO.now() else 0;
    const timing = timed != 0;
    if (ok and selected_count != 0) {
        task_storage = c.calloc(@as(usize, selected_count) * 3, @sizeOf(Core.Task));
        if (task_storage) |pointer| tasks = @ptrCast(@alignCast(pointer)) else ok = false;
    }
    if (ok) {
        const budget = ctx.cache.budget();
        var reserved = ctx.cache.state.cache_entry_count;
        for (0..selected_count) |index| {
            const expert: u32 = @intCast(selected_ids[index]);
            points[index] = Core.offsets(table, expert) orelse {
                _ = c.fprintf(c.__stderrp, "ds4: Metal streaming batch selected expert offset overflow\n");
                ok = false;
                break;
            };
            entries[index] = ctx.cache.peek(Core.identity(table, points[index]), table.layer, expert, table.n_total_expert, used);
            if (entries[index] != null) continue;
            const force = budget != 0 and reserved >= budget;
            // A serial read-ahead here would wait for the same pages before
            // the worker pool gets to issue the complete parallel batch.
            const before = if (timing) IO.now() else 0;
            const prepared = ctx.storage().prepare(table.layer, expert, table.layer, selected_ids[0..selected_count], table.gate_expert_bytes, table.down_expert_bytes, force, &buffers[loads]);
            if (timing) ctx.timingBuffer(IO.now() - before);
            if (!prepared) {
                ok = false;
                break;
            }
            if (!force) reserved +|= 1;
            if (!Core.ready(&buffers[loads])) {
                ok = false;
                break;
            }
            const targets = Core.destinations(&buffers[loads]) orelse {
                ok = false;
                break;
            };
            load_unique[loads] = @intCast(index);
            const before_tasks = if (timing) IO.now() else 0;
            Core.appendTasks(tasks[0 .. @as(usize, selected_count) * 3], &task_count, targets, points[index], table.gate_expert_bytes, table.down_expert_bytes);
            if (timing) ctx.timingTasks(1, IO.now() - before_tasks);
            loads += 1;
        }
    }
    if (ok and loads != 0) {
        if (timed != 0) {
            const next = IO.now();
            prepare_ms = next - timed;
            timed = next;
        }
        var bytes: u64 = 0;
        var elapsed: f64 = 0;
        ok = IO.readTasks(&ctx.load.pool, tasks, task_count, &bytes, &elapsed);
        if (ok) IO.noteRead(ctx.cache.state, table.layer, bytes, elapsed);
        if (timed != 0) timed = IO.now();
        if (ok and present("DS4_METAL_STREAMING_EXPERT_PREAD_PROFILE")) _ = c.fprintf(c.__stderrp, "ds4: Metal streaming batch expert parallel pread layer=%u experts=%u tensors=%u threads=%u bytes=%.2f GiB wall=%.3f ms\n", table.layer, loads, task_count, IO.threadCount(task_count), @as(f64, @floatFromInt(bytes)) / 1073741824, elapsed);
        if (ok) for (0..loads) |index| {
            const slot = load_unique[index];
            const group = &buffers[index];
            entries[slot] = ctx.cache.install(Core.identity(table, points[slot]), table.layer, @intCast(selected_ids[slot]), group.gate_buffer, group.up_buffer, group.down_buffer, group.gate_inner, group.up_inner, group.down_inner);
            if (entries[slot] == null) {
                ok = false;
                break;
            }
        };
        if (timed != 0) ctx.timingLoad(prepare_ms, elapsed, IO.now() - timed);
    }
    // The original task allocation retires before publishing address resources.
    if (task_storage) |pointer| c.free(pointer);
    task_storage = null;
    if (ok) for (0..selected_count) |index| {
        const expert: u32 = @intCast(selected_ids[index]);
        if (entries[index] == null) {
            const gate_rel = points[index].gate -% table.gate_offset;
            const down_rel = points[index].down -% table.down_offset;
            if (overflow.gate_buffer == null) {
                const base = Core.Offsets{ .gate = table.gate_offset, .up = table.up_offset, .down = table.down_offset };
                if (!Core.wrap(table, base, @as(u64, table.n_total_expert) *% table.gate_expert_bytes, @as(u64, table.n_total_expert) *% table.down_expert_bytes, overflow)) {
                    _ = c.fprintf(c.__stderrp, "ds4: Metal streaming prefill batch selected addr failed to map overflow expert views at layer %u\n", table.layer);
                    ok = false;
                    break;
                }
            }
            // Only this short-lived borrowed projection adds per-expert inner
            // offsets. The real overflow record remains the whole view owner.
            var address = std.mem.zeroes(Reuse);
            defer c.ds4_metal_expert_reuse_reset(&address);
            c.ds4_metal_expert_reuse_assign(&address, overflow.gate_buffer, overflow.up_buffer, overflow.down_buffer, @truncate(@as(u64, overflow.gate_inner) +% gate_rel), @truncate(@as(u64, overflow.up_inner) +% gate_rel), @truncate(@as(u64, overflow.down_inner) +% down_rel));
            if (!c.ds4_metal_expert_load_address_install(table.layer, expert, &address)) {
                ok = false;
                break;
            }
            view_served += 1;
            continue;
        }
        const entry = entries[index].?;
        const extra = if (frequency[expert] != 0) frequency[expert] - 1 else 0;
        if (extra != 0) {
            entry.use_count +|= extra;
            ctx.cache.state.cache_hits +|= extra;
            ctx.cache.state.cache_layer_hits[table.layer] +|= extra;
        }
        resources[resource_count.*] = entry;
        resource_count.* += 1;
    };
    // IDs retire before final resource admission, as in the original action.
    c.free(allocation);
    ids_storage = null;
    if (!ok or (resource_count.* == 0 and view_served == 0)) {
        ctx.cache.clearLayer(table.layer);
        return false;
    }
    if (!c.ds4_metal_expert_load_addresses(table.layer, addresses)) {
        ctx.cache.clearLayer(table.layer);
        return false;
    }
    unique.* = resource_count.* + view_served;
    if (view_served != 0 and Cache.timingSummary()) _ = c.fprintf(c.__stderrp, "ds4: Metal streaming prefill batch selected addr layer=%u served %u/%u unique experts via mapped views (cache budget %u)\n", table.layer, view_served, selected_count, ctx.cache.budget());
    if (profile) {
        var logical: u64 = max;
        if (table.gate_expert_bytes <= (max - table.down_expert_bytes) / 2) {
            const per_expert = table.gate_expert_bytes * 2 + table.down_expert_bytes;
            if (per_expert == 0 or resource_count.* <= max / per_expert) logical = @as(u64, resource_count.*) * per_expert;
        }
        _ = c.fprintf(c.__stderrp, "ds4: Metal streaming prefill batch selected addr layer=%u tokens=%u unique=%u read=%.3f ms wrap=%.3f ms bytes=%.2f GiB\n", table.layer, tokens, resource_count.*, read - start, IO.now() - read, @as(f64, @floatFromInt(logical)) / 1073741824);
    }
    return true;
}
