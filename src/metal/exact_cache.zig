//! Exact no-copy model views and their cache. Numerical kernels and command
//! ownership stay unchanged. Native keys replace formatted NSString keys;
//! the ARC boundary owns only buffer creation and retained-handle release.
const std = @import("std");
const c = @import("exact_cache_c");
const allocator = std.heap.c_allocator;
pub const Lifetime = c.ds4_gpu_exact_view_lifetime;
const Key = struct { model: usize, model_size: u64, offset: u64, bytes: u64 };
const State = struct {
    views: std.AutoHashMapUnmanaged(Key, *anyopaque) = .empty,
    enabled: bool = false,
    cache_bytes: u64 = 0,
    evictions: u64 = 0,
    over_limit: bool = false,
};
fn peek() ?*State {
    const value = c.ds4_metal_exact_cache_state_slot().* orelse return null;
    return @ptrCast(@alignCast(value));
}
fn state() *State {
    const slot = c.ds4_metal_exact_cache_state_slot();
    if (slot.* == null) {
        const value = allocator.create(State) catch {
            c.perror("ds4: exact cache state");
            c.exit(1);
        };
        value.* = .{};
        slot.* = value;
    }
    return @ptrCast(@alignCast(slot.*.?));
}
// Environment policy is shared; actual retained views/accounting are per owner.
var cached_limit: u64 = 0;
var limit_once: c.pthread_once_t = c.ds4_metal_exact_once_init;
const mib: u64 = 1024 * 1024;
const gib: u64 = 1024 * mib;

fn asGiB(bytes: u64) f64 {
    @setFloatMode(.optimized);
    return @as(f64, @floatFromInt(bytes)) / gib;
}
fn setting(name: [*:0]const u8, scale: u64, fallback: u64) u64 {
    const raw = c.getenv(name);
    if (raw == null or raw[0] == 0) return fallback;
    var end: [*c]u8 = null;
    const value = c.strtoull(raw, &end, 10);
    if (end == raw or end[0] != 0) return fallback;
    return value *| scale;
}
fn initializeLimit() callconv(.c) void {
    const from_gib = setting("DS4_METAL_EXACT_VIEW_CACHE_GIB", gib, 64 * gib);
    cached_limit = setting("DS4_METAL_EXACT_VIEW_CACHE_MIB", mib, from_gib);
}
fn limitBytes() u64 {
    _ = c.pthread_once(&limit_once, initializeLimit);
    return cached_limit;
}

pub fn init() void {
    cleanup(); // Replacing the old dictionary also released its retained values.
    state().enabled = true;
}
pub fn clear(reason: [*c]const u8) void {
    const self = peek() orelse return;
    if (self.enabled and self.views.count() != 0) {
        if (c.getenv("DS4_METAL_EXACT_VIEW_CACHE_PROFILE") != null)
            _ = c.fprintf(c.__stderrp, "ds4: Metal exact model view cache evict reason=%s entries=%lu bytes=%.2f GiB limit=%.2f GiB\n", if (reason == null) @as([*c]const u8, "unknown") else reason, @as(c_ulong, self.views.count()), asGiB(self.cache_bytes), asGiB(limitBytes()));
        var entries = self.views.valueIterator();
        while (entries.next()) |handle| c.ds4_metal_exact_view_release(handle.*);
        self.views.clearRetainingCapacity();
        self.evictions +%= 1;
    }
    self.cache_bytes = 0;
    self.over_limit = false;
}
pub fn maybeEvict(reason: [*c]const u8) void {
    const self = peek() orelse return;
    // Existing command-completion boundaries decide WHEN retirement is safe.
    // Do not evict during insertion: unretained encoders can still use a view.
    if (self.over_limit) clear(reason);
}
pub fn cleanup() void {
    const self = peek() orelse return;
    var entries = self.views.valueIterator();
    while (entries.next()) |handle| c.ds4_metal_exact_view_release(handle.*);
    self.views.clearAndFree(allocator);
    self.enabled = false;
    self.cache_bytes = 0;
    self.evictions = 0;
    self.over_limit = false;
    const value = self;
    c.ds4_metal_exact_cache_state_slot().* = null;
    allocator.destroy(value);
}
pub fn report() void {
    const self = peek() orelse return;
    if (!self.enabled or self.views.count() == 0) return;
    const limit = limitBytes();
    if (limit == 0) {
        _ = c.fprintf(c.__stderrp, "ds4:   exact model view cache %lu buffers %.2f GiB unlimited, %llu evictions (not copied)\n", @as(c_ulong, self.views.count()), asGiB(self.cache_bytes), self.evictions);
    } else {
        _ = c.fprintf(c.__stderrp, "ds4:   exact model view cache %lu buffers %.2f GiB / %.2f GiB, %llu evictions (not copied)\n", @as(c_ulong, self.views.count()), asGiB(self.cache_bytes), asGiB(limit), self.evictions);
    }
}
fn fail(message: [*:0]const u8) ?*anyopaque {
    _ = c.fprintf(c.__stderrp, "ds4: Metal %s\n", message);
    return null;
}

pub fn wrap(model: ?*const anyopaque, model_size: u64, offset: u64, bytes: u64, inner_offset: [*c]u64, lifetime: Lifetime) ?*anyopaque {
    const cached = lifetime == c.DS4_GPU_EXACT_VIEW_CACHED;
    const self = peek();
    @setFloatMode(.optimized);
    if (model == null or (cached and (self == null or !self.?.enabled)) or
        model_size == 0 or offset > model_size or bytes > model_size - offset)
        return fail("exact model range is outside the mapped model");
    const page: u64 = @intCast(c.sysconf(c._SC_PAGESIZE));
    const page_offset = offset & ~(page - 1);
    const leading = offset - page_offset;
    if (bytes > std.math.maxInt(u64) - leading or leading + bytes > std.math.maxInt(u64) - (page - 1))
        return fail("exact model range overflows page alignment");
    var view_bytes = (leading + bytes + page - 1) & ~(page - 1);
    view_bytes = @min(view_bytes, model_size - page_offset);
    if (leading + bytes > view_bytes)
        return fail("exact model range alignment exceeds mapped model");
    const max_buffer = c.ds4_metal_model_view_max_buffer_length();
    if (view_bytes > max_buffer) {
        _ = c.fprintf(c.__stderrp, "ds4: Metal exact model range %.2f GiB exceeds maxBufferLength %.2f GiB\n", asGiB(view_bytes), asGiB(max_buffer));
        return null;
    }
    const key: Key = .{ .model = @intFromPtr(model), .model_size = model_size, .offset = page_offset, .bytes = view_bytes };
    if (cached) if (self.?.views.get(key)) |handle| {
        if (inner_offset != null) inner_offset[0] = leading;
        return handle;
    };
    const address: *const anyopaque = @ptrFromInt(@intFromPtr(model) +% page_offset);
    const handle = c.ds4_metal_exact_view_create(address, view_bytes, lifetime) orelse {
        _ = c.fprintf(c.__stderrp, "ds4: Metal could not wrap exact mmaped model range at %.2f GiB, size %.2f MiB\n", asGiB(page_offset), @as(f64, @floatFromInt(view_bytes)) / mib);
        return null;
    };
    if (cached) {
        const cache = self.?;
        cache.views.put(allocator, key, handle) catch {
            c.ds4_metal_exact_view_release(handle);
            c.perror("ds4: Metal exact model view cache");
            c.exit(1);
        };
        cache.cache_bytes +|= view_bytes;
        const limit = limitBytes();
        if (limit != 0 and cache.cache_bytes > limit) cache.over_limit = true;
    }
    if (inner_offset != null) inner_offset[0] = leading;
    return handle;
}
