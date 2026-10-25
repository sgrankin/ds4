//! Persistent Metal tensor ownership, regions and live-handle accounting.
//! Handles remain retained Objective-C objects. Only the narrow ARC/driver
//! leaves create/release buffers or enqueue a blit; policy and CPU copies live
//! here. Ordinary reads borrow a region directly, without a registry lookup.
const std = @import("std");
const c = @import("resource_c");
pub const Tensor = c.ds4_gpu_tensor;
pub const Stats = c.ds4_metal_tensor_stats;
const allocator = std.heap.c_allocator;
const max_bytes = std.math.maxInt(usize);

// A second bridge_transfer of a freed opaque handle is a use-after-free. The
// registry validates membership BEFORE any Objective-C dereference on free.
// One mutex also serializes live/peak accounting; views are live handles but
// do not count their shared backing allocation twice. Other operations still
// require the caller to retain its handle, just as the public C API does.
const State = struct {
    mutex: c.pthread_mutex_t = c.ds4_metal_resource_mutex_init,
    handles: std.AutoHashMapUnmanaged(usize, void) = .empty,
    allocation_stats: Stats = .{ .live_bytes = 0, .peak_bytes = 0 },
    trace_allocations: ?bool = null,
};
fn peek() ?*State {
    const value = c.ds4_metal_resource_state_slot().* orelse return null;
    return @ptrCast(@alignCast(value));
}
fn state() *State {
    const slot = c.ds4_metal_resource_state_slot();
    if (slot.* == null) {
        const value = allocator.create(State) catch {
            c.perror("ds4: resource state");
            c.exit(1);
        };
        value.* = .{};
        slot.* = value;
    }
    return @ptrCast(@alignCast(slot.*.?));
}

fn lock() void {
    const self = state();
    _ = c.pthread_mutex_lock(&self.mutex);
}
fn unlock() void {
    const self = state();
    _ = c.pthread_mutex_unlock(&self.mutex);
}
fn traceEnabledLocked() bool {
    const self = state();
    if (self.trace_allocations == null)
        self.trace_allocations = c.getenv("DS4_METAL_TRACE_ALLOCS") != null;
    return self.trace_allocations.?;
}
fn logAllocation(action: [*:0]const u8, bytes: u64, snapshot: Stats) void {
    @setFloatMode(.optimized);
    const mib: f64 = 1024 * 1024;
    _ = c.fprintf(c.__stderrp, "ds4: Metal tensor %s %.3f MiB live %.3f MiB peak %.3f MiB\n", action, @as(f64, @floatFromInt(bytes)) / mib, @as(f64, @floatFromInt(snapshot.live_bytes)) / mib, @as(f64, @floatFromInt(snapshot.peak_bytes)) / mib);
}

const Tracked = struct { snapshot: Stats, trace: bool };
fn track(tensor: *Tensor, owned_bytes: u64) ?Tracked {
    const self = state();
    const key = @intFromPtr(tensor);
    if (key == std.math.maxInt(usize)) return null;
    lock();
    defer unlock();
    const entry = self.handles.getOrPut(allocator, key) catch return null;
    if (entry.found_existing) return null;
    // Allocation sizes are already admitted by the Metal device. Preserve the
    // original unsigned diagnostic counter rather than adding a new budget.
    self.allocation_stats.live_bytes +%= owned_bytes;
    self.allocation_stats.peak_bytes = @max(self.allocation_stats.peak_bytes, self.allocation_stats.live_bytes);
    return .{ .snapshot = self.allocation_stats, .trace = owned_bytes != 0 and traceEnabledLocked() };
}

pub fn stats() Stats {
    const self = peek() orelse return .{ .live_bytes = 0, .peak_bytes = 0 };
    lock();
    defer unlock();
    return self.allocation_stats;
}
pub fn resetTracking() void {
    const self = peek() orelse return;
    lock();
    if (self.handles.count() != 0)
        _ = c.fprintf(c.__stderrp, "ds4: Metal cleanup discarded %zu live tensor handles\n", @as(usize, self.handles.count()));
    self.handles.clearAndFree(allocator);
    self.allocation_stats = .{ .live_bytes = 0, .peak_bytes = 0 };
    unlock();
    c.ds4_metal_resource_state_slot().* = null;
    _ = c.pthread_mutex_destroy(&self.mutex);
    allocator.destroy(self);
}

pub fn alloc(bytes: u64) ?*Tensor {
    // Initialization precedes the zero/oversized check in the original API.
    if (c.ds4_gpu_init() == 0 or bytes == 0 or bytes > max_bytes) return null;
    const tensor = c.ds4_metal_resource_new(bytes) orelse return null;
    const tracked = track(tensor, bytes) orelse {
        _ = c.fprintf(c.__stderrp, "ds4: failed to track Metal tensor allocation\n");
        c.ds4_metal_resource_dispose(tensor);
        return null;
    };
    if (tracked.trace) logAllocation("alloc", bytes, tracked.snapshot);
    return tensor;
}
pub fn view(base: ?*const Tensor, offset: u64, bytes: u64) ?*Tensor {
    const parent = base orelse return null;
    const region = c.ds4_metal_resource_region(parent);
    if (offset > region.bytes or bytes > region.bytes - offset or
        region.offset > std.math.maxInt(u64) - offset) return null;
    const absolute_offset = region.offset + offset;
    if (absolute_offset > max_bytes) return null;
    const tensor = c.ds4_metal_resource_view(parent, absolute_offset, bytes) orelse {
        _ = c.fprintf(c.__stderrp, "ds4: failed to track Metal tensor view\n");
        return null;
    };
    if (track(tensor, 0) == null) {
        _ = c.fprintf(c.__stderrp, "ds4: failed to track Metal tensor view\n");
        c.ds4_metal_resource_dispose(tensor);
        return null;
    }
    return tensor;
}
pub fn free(tensor: ?*Tensor) void {
    const handle = tensor orelse return;
    const self = peek() orelse {
        _ = c.fprintf(c.__stderrp, "ds4: Metal tensor free ignored for unknown handle %p\n", handle);
        return;
    };
    lock();
    if (!self.handles.remove(@intFromPtr(handle))) {
        unlock();
        _ = c.fprintf(c.__stderrp, "ds4: Metal tensor free ignored for unknown handle %p\n", handle);
        return;
    }
    const region = c.ds4_metal_resource_region(handle);
    if (region.owner != 0)
        self.allocation_stats.live_bytes -|= region.bytes;
    const snapshot = self.allocation_stats;
    const trace = region.owner != 0 and traceEnabledLocked();
    unlock();
    if (trace) logAllocation("free", region.bytes, snapshot);
    c.ds4_metal_resource_dispose(handle);
}

pub fn byteSize(tensor: ?*const Tensor) u64 {
    return if (tensor) |handle| c.ds4_metal_resource_region(handle).bytes else 0;
}
pub fn contents(tensor: ?*Tensor) ?*anyopaque {
    return if (tensor) |handle| c.ds4_metal_resource_contents(handle) else null;
}
pub fn fillF32(tensor: ?*Tensor, value: f32, count: u64) bool {
    const handle = tensor orelse return false;
    if (count > byteSize(handle) / @sizeOf(f32)) return false;
    const pointer = contents(handle) orelse return count == 0;
    // Public views are byte-granular. Do not introduce an alignment rejection
    // where the C float-pointer loop allowed an unaligned shared-memory view.
    const output: [*]align(1) f32 = @ptrCast(pointer);
    @memset(output[0..@intCast(count)], value);
    return true;
}
fn fits(region_bytes: u64, offset: u64, length: u64) bool {
    return offset <= region_bytes and length <= region_bytes - offset;
}
pub fn write(tensor: ?*Tensor, offset: u64, data: ?*const anyopaque, length: u64) bool {
    const handle = tensor orelse return false;
    if ((data == null and length != 0) or !fits(byteSize(handle), offset, length)) return false;
    if (length != 0) {
        const destination: [*]u8 = @ptrCast(contents(handle).?);
        const source: [*]const u8 = @ptrCast(data.?);
        @memcpy(destination[@intCast(offset)..][0..@intCast(length)], source[0..@intCast(length)]);
    }
    return true;
}
pub fn read(tensor: ?*const Tensor, offset: u64, data: ?*anyopaque, length: u64) bool {
    const handle = tensor orelse return false;
    if ((data == null and length != 0) or !fits(byteSize(handle), offset, length)) return false;
    if (length != 0) {
        const destination: [*]u8 = @ptrCast(data.?);
        const source: [*]const u8 = @ptrCast(contents(@constCast(handle)).?);
        @memcpy(destination[0..@intCast(length)], source[@intCast(offset)..][0..@intCast(length)]);
    }
    return true;
}
pub fn copy(dst: ?*Tensor, dst_offset: u64, src: ?*const Tensor, src_offset: u64, length: u64) bool {
    const destination = dst orelse return false;
    const source = src orelse return false;
    if (c.ds4_gpu_init() == 0) return false;
    const destination_region = c.ds4_metal_resource_region(destination);
    const source_region = c.ds4_metal_resource_region(source);
    if (!fits(destination_region.bytes, dst_offset, length) or
        !fits(source_region.bytes, src_offset, length)) return false;
    // Empty copies succeed without an active batch; nonempty copies borrow it.
    if (length == 0) return true;
    return c.ds4_metal_resource_blit(destination, destination_region.offset +% dst_offset, source, source_region.offset +% src_offset, length) != 0;
}
