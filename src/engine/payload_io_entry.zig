//! CPU-safe FILE entries. Do not merge this object with tensor transfers: a
//! host-only fixture must not pull Metal or a second engine into its link.
const c = @import("payload_io_c");
const io = @import("payload_io.zig");
fn file(stream: *c.FILE, errors: [*c]u8, capacity: usize) io.File {
    return .{ .stream = stream, .errors = errors, .error_capacity = capacity };
}
export fn ds4_native_payload_write_bytes(
    stream: *c.FILE,
    data: ?*const anyopaque,
    bytes: u64,
    errors: [*c]u8,
    capacity: usize,
) c_int {
    return file(stream, errors, capacity).write(data, bytes);
}
export fn ds4_native_payload_read_bytes(
    stream: *c.FILE,
    data: ?*anyopaque,
    bytes: u64,
    remaining: ?*u64,
    errors: [*c]u8,
    capacity: usize,
) c_int {
    return file(stream, errors, capacity).read(data, bytes, remaining);
}
export fn ds4_native_payload_write_u32(stream: *c.FILE, value: u32, errors: [*c]u8, capacity: usize) c_int {
    return file(stream, errors, capacity).writeU32(value);
}
export fn ds4_native_payload_read_u32(stream: *c.FILE, value: *u32, remaining: ?*u64, errors: [*c]u8, capacity: usize) c_int {
    return file(stream, errors, capacity).readU32(value, remaining);
}
export fn ds4_native_payload_skip_bytes(
    stream: *c.FILE,
    bytes: u64,
    scratch: [*c]u8,
    scratch_capacity: usize,
    remaining: ?*u64,
    errors: [*c]u8,
    capacity: usize,
) c_int {
    return file(stream, errors, capacity).skip(bytes, scratch, scratch_capacity, remaining);
}
export fn ds4_native_payload_copy_file(source: *c.FILE, destination: *c.FILE, bytes: u64, errors: [*c]u8, capacity: usize) c_int {
    return io.copyFile(source, destination, bytes, errors, capacity);
}
