//! Bounded canonical-payload transfers. Model codecs own span ordering and
//! frontier validity; these leaves preserve FILE positions, remaining-byte
//! accounting and the caller's fixed scratch allocation.
const std = @import("std");
const c = @import("payload_io_c");
const Tensor = c.ds4_gpu_tensor;
pub const chunk_bytes = 8 * 1024 * 1024;

pub const File = struct {
    stream: *c.FILE,
    errors: [*c]u8,
    error_capacity: usize,

    pub fn fail(self: File, message: [:0]const u8) c_int {
        if (self.error_capacity != 0) _ = c.snprintf(self.errors, self.error_capacity, "%s", message.ptr);
        return 1;
    }
    pub fn write(self: File, data: ?*const anyopaque, bytes: u64) c_int {
        var pointer: [*c]const u8 = @ptrCast(data);
        var left = bytes;
        while (left != 0) {
            const size: usize = @intCast(@min(left, std.math.maxInt(usize)));
            if (c.fwrite(pointer, 1, size, self.stream) != size) return self.fail("failed to write session payload");
            pointer += size;
            left -= size;
        }
        return 0;
    }
    pub fn read(self: File, data: ?*anyopaque, bytes: u64, remaining: ?*u64) c_int {
        if (remaining) |bound| if (bound.* < bytes) return self.fail("truncated session payload");
        var pointer: [*c]u8 = @ptrCast(data);
        var left = bytes;
        while (left != 0) {
            const size: usize = @intCast(@min(left, std.math.maxInt(usize)));
            if (c.fread(pointer, 1, size, self.stream) != size) return self.fail("failed to read session payload");
            pointer += size;
            left -= size;
        }
        // A short physical read consumes FILE bytes and may modify the buffer,
        // but does not debit this call's logical bound, matching the C owner.
        if (remaining) |bound| bound.* -= bytes;
        return 0;
    }
    pub fn writeU32(self: File, value: u32) c_int {
        var storage: [4]u8 = undefined;
        std.mem.writeInt(u32, &storage, value, .little);
        return self.write(&storage, storage.len);
    }
    pub fn readU32(self: File, value: *u32, remaining: ?*u64) c_int {
        var storage: [4]u8 = undefined;
        if (self.read(&storage, storage.len, remaining) != 0) return 1;
        value.* = std.mem.readInt(u32, &storage, .little);
        return 0;
    }
    pub fn skip(self: File, bytes: u64, scratch: [*c]u8, capacity: usize, remaining: ?*u64) c_int {
        if (remaining) |bound| if (bound.* < bytes) return self.fail("truncated session payload");
        if (scratch == null or capacity == 0) return self.fail("session payload skip buffer is missing");
        var left = bytes;
        while (left != 0) {
            const size: usize = @intCast(@min(left, capacity));
            if (c.fread(scratch, 1, size, self.stream) != size) return self.fail("failed to skip session payload");
            left -= size;
        }
        if (remaining) |bound| bound.* -= bytes;
        return 0;
    }
};

pub fn copyFile(source: *c.FILE, destination: *c.FILE, bytes: u64, errors: [*c]u8, capacity: usize) c_int {
    const scratch = c.malloc(chunk_bytes) orelse {
        _ = c.fprintf(c.__stderrp, "ds4: out of memory\n");
        c.exit(1);
    };
    defer c.free(scratch);
    const file = File{ .stream = source, .errors = errors, .error_capacity = capacity };
    var left = bytes;
    while (left != 0) {
        const size: usize = @intCast(@min(left, chunk_bytes));
        if (c.fread(scratch, 1, size, source) != size) return file.fail("failed to read staged session payload");
        if (c.fwrite(scratch, 1, size, destination) != size) return file.fail("failed to write staged session payload");
        left -= size;
    }
    return 0;
}

fn spanFits(tensor: ?*const Tensor, offset: u64, bytes: u64) bool {
    if (tensor == null) return false;
    const size = c.ds4_gpu_tensor_bytes(tensor);
    return offset <= size and bytes <= size - offset;
}

pub fn writeTensor(file: File, tensor: ?*const Tensor, offset: u64, bytes: u64, scratch: [*c]u8, capacity: usize) c_int {
    if (!spanFits(tensor, offset, bytes)) return file.fail("session tensor is smaller than the payload");
    var done: u64 = 0;
    while (done < bytes) {
        const size: usize = @intCast(@min(bytes - done, capacity));
        if (c.ds4_gpu_tensor_read(tensor, offset + done, scratch, size) == 0) return file.fail("failed to read accelerator session tensor");
        if (file.write(scratch, size) != 0) return 1;
        done += size;
    }
    return 0;
}
pub fn readTensor(
    file: File,
    tensor: ?*Tensor,
    offset: u64,
    bytes: u64,
    scratch: [*c]u8,
    capacity: usize,
    remaining: ?*u64,
) c_int {
    if (!spanFits(tensor, offset, bytes)) return file.fail("session tensor is smaller than the payload");
    var done: u64 = 0;
    while (done < bytes) {
        const size: usize = @intCast(@min(bytes - done, capacity));
        if (file.read(scratch, size, remaining) != 0) return 1;
        if (c.ds4_gpu_tensor_write(tensor, offset + done, scratch, size) == 0) return file.fail("failed to restore accelerator session tensor");
        done += size;
    }
    return 0;
}

// Apple Silicon's previous owner uses NEON FCVT/FCVTN, including rounding and
// NaN payload handling. Strict casts lower to the same hardware conversion;
// no arithmetic or fast-math canonicalization is permitted around the bits.
pub fn halfToFloat(bits: u16) f32 {
    @setFloatMode(.strict);
    return @floatCast(@as(f16, @bitCast(bits)));
}
pub fn floatToHalf(value: f32) u16 {
    @setFloatMode(.strict);
    return @bitCast(@as(f16, @floatCast(value)));
}

const Conversion = struct {
    half: [*]align(1) u16,
    float: [*]align(1) f32,
    capacity: usize,

    fn init(scratch: [*c]u8, bytes: usize) ?Conversion {
        const count = (bytes / (@sizeOf(u16) + @sizeOf(f32))) & ~@as(usize, 1);
        if (count == 0) return null;
        return .{ .half = @ptrCast(scratch), .float = @ptrCast(scratch + count * @sizeOf(u16)), .capacity = count };
    }
};
fn halfSpanFits(tensor: ?*const Tensor, offset: u64, count: u64) bool {
    return count <= std.math.maxInt(u64) / @sizeOf(u16) and count <= std.math.maxInt(u64) / @sizeOf(f32) and
        spanFits(tensor, offset, count * @sizeOf(u16));
}

pub fn writeHalfAsFloat(file: File, tensor: ?*const Tensor, offset: u64, count: u64, scratch: [*c]u8, capacity: usize) c_int {
    if (!halfSpanFits(tensor, offset, count)) return file.fail("session tensor is smaller than the F16 payload");
    const conversion = Conversion.init(scratch, capacity) orelse return file.fail("session tensor conversion buffer is too small");
    var done: u64 = 0;
    while (done < count) {
        const size: usize = @intCast(@min(count - done, conversion.capacity));
        if (c.ds4_gpu_tensor_read(tensor, offset + done * @sizeOf(u16), conversion.half, size * @sizeOf(u16)) == 0)
            return file.fail("failed to read Metal F16 session tensor");
        for (0..size) |index| conversion.float[index] = halfToFloat(conversion.half[index]);
        if (file.write(conversion.float, @as(u64, size) * @sizeOf(f32)) != 0) return 1;
        done += size;
    }
    return 0;
}
pub fn readFloatAsHalf(
    file: File,
    tensor: ?*Tensor,
    offset: u64,
    count: u64,
    scratch: [*c]u8,
    capacity: usize,
    remaining: ?*u64,
) c_int {
    if (!halfSpanFits(tensor, offset, count)) return file.fail("session tensor is smaller than the F16 payload");
    const conversion = Conversion.init(scratch, capacity) orelse return file.fail("session tensor conversion buffer is too small");
    var done: u64 = 0;
    while (done < count) {
        const size: usize = @intCast(@min(count - done, conversion.capacity));
        if (file.read(conversion.float, @as(u64, size) * @sizeOf(f32), remaining) != 0) return 1;
        for (0..size) |index| conversion.half[index] = floatToHalf(conversion.float[index]);
        if (c.ds4_gpu_tensor_write(tensor, offset + done * @sizeOf(u16), conversion.half, size * @sizeOf(u16)) == 0)
            return file.fail("failed to restore Metal F16 session tensor");
        done += size;
    }
    return 0;
}
