//! File-cache preparation primitives shared by layer and selected-range jobs.
//! These retain the current POSIX hints, page touches and bounded pread loop.
const std = @import("std");
const c = @import("gguf_c");
const Model = c.ds4_model;
const allocator = std.heap.c_allocator;
const maximum = std.math.maxInt(u64);

pub fn outOfMemory() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4: out of memory\n");
    c.exit(1);
}
pub fn now() f64 {
    var time: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &time);
    return @as(f64, @floatFromInt(time.tv_sec)) + @as(f64, @floatFromInt(time.tv_nsec)) * 1e-9;
}

const Aligned = struct { offset: u64, bytes: u64 };
fn pageRange(model: [*c]const Model, offset: u64, bytes: u64) ?Aligned {
    if (model == null or model.*.map == null or offset > model.*.size or bytes == 0 or bytes > model.*.size - offset) return null;
    const page: u64 = @intCast(c.sysconf(c._SC_PAGESIZE));
    const first = offset & ~(page - 1);
    const leading = offset - first;
    if (bytes > maximum - leading or leading + bytes > maximum - (page - 1)) return null;
    const count = @min((leading + bytes + page - 1) & ~(page - 1), model.*.size - first);
    if (count == 0 or count > std.math.maxInt(usize)) return null;
    return .{ .offset = first, .bytes = count };
}

pub fn touch(model: [*c]const Model, offset: u64, bytes: u64, touched: ?*u64, sink: ?*u8) bool {
    const range = pageRange(model, offset, bytes) orelse return false;
    const page: u64 = @intCast(c.sysconf(c._SC_PAGESIZE));
    const memory = model.*.map + range.offset;
    _ = c.posix_madvise(@constCast(memory), @intCast(range.bytes), c.POSIX_MADV_WILLNEED);
    const pointer: [*]const volatile u8 = @ptrCast(memory);
    var sum: u8 = if (sink) |value| value.* else 0;
    var position: u64 = 0;
    while (position < range.bytes) : (position += page) sum ^= pointer[position];
    sum ^= pointer[range.bytes - 1];
    if (sink) |value| value.* = sum;
    if (touched) |value| value.* +%= range.bytes;
    return true;
}

pub fn pread(model: [*c]const Model, offset: u64, bytes: u64, read_bytes: ?*u64, sink: ?*u8) bool {
    if (model == null or model.*.fd < 0 or offset > model.*.size or bytes == 0 or
        bytes > model.*.size - offset or offset > std.math.maxInt(i64)) return false;
    const buffer = allocator.alloc(u8, 1024 * 1024) catch outOfMemory();
    defer allocator.free(buffer);
    var position = offset;
    var remaining = bytes;
    var sum: u8 = if (sink) |value| value.* else 0;
    var ok = true;
    while (remaining != 0) {
        const wanted: usize = @intCast(@min(remaining, buffer.len));
        var count: isize = undefined;
        while (true) {
            count = c.pread(model.*.fd, buffer.ptr, wanted, @intCast(position));
            if (count >= 0 or c.__error().* != c.EINTR) break;
        }
        if (count <= 0) {
            ok = false;
            break;
        }
        const got: usize = @intCast(count);
        sum ^= buffer[0];
        sum ^= buffer[got - 1];
        position += got;
        remaining -= got;
        if (read_bytes) |value| value.* +|= got;
    }
    if (sink) |value| value.* = sum;
    return ok;
}

pub fn advise(model: [*c]const Model, offset: u64, bytes: u64, enabled: bool, advised: ?*u64) bool {
    if (!enabled) return true;
    const range = pageRange(model, offset, bytes) orelse return false;
    if (c.posix_madvise(@constCast(model.*.map + range.offset), @intCast(range.bytes), c.POSIX_MADV_WILLNEED) != 0) return false;
    if (advised) |value| value.* +|= range.bytes;
    return true;
}

pub fn readAhead(model: [*c]const Model, offset: u64, bytes: u64, enabled: bool) void {
    if (!enabled or model == null or model.*.fd < 0 or model.*.map == null or
        offset > model.*.size or bytes == 0 or bytes > model.*.size - offset) return;
    var position = offset;
    var remaining = bytes;
    while (remaining != 0) {
        const count = @min(remaining, std.math.maxInt(c_int));
        if (position > std.math.maxInt(i64)) break;
        var advisory: c.struct_radvisory = .{ .ra_offset = @intCast(position), .ra_count = @intCast(count) };
        _ = c.fcntl(model.*.fd, c.F_RDADVISE, &advisory);
        position += count;
        remaining -= count;
    }
}
