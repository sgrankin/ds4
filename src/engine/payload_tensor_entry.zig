//! GPU-dependent tensor entries, separate from scalar FILE I/O.
const c = @import("payload_io_c");
const io = @import("payload_io.zig");
fn file(stream: *c.FILE, errors: [*c]u8, capacity: usize) io.File {
    return .{ .stream = stream, .errors = errors, .error_capacity = capacity };
}
export fn ds4_native_payload_write_tensor_span(
    stream: *c.FILE,
    tensor: ?*const c.ds4_gpu_tensor,
    offset: u64,
    bytes: u64,
    scratch: [*c]u8,
    scratch_capacity: usize,
    errors: [*c]u8,
    capacity: usize,
) c_int {
    return io.writeTensor(file(stream, errors, capacity), tensor, offset, bytes, scratch, scratch_capacity);
}
export fn ds4_native_payload_read_tensor_span(
    stream: *c.FILE,
    tensor: ?*c.ds4_gpu_tensor,
    offset: u64,
    bytes: u64,
    scratch: [*c]u8,
    scratch_capacity: usize,
    remaining: ?*u64,
    errors: [*c]u8,
    capacity: usize,
) c_int {
    return io.readTensor(file(stream, errors, capacity), tensor, offset, bytes, scratch, scratch_capacity, remaining);
}
export fn ds4_native_payload_write_f16_as_f32(
    stream: *c.FILE,
    tensor: ?*const c.ds4_gpu_tensor,
    offset: u64,
    count: u64,
    scratch: [*c]u8,
    scratch_capacity: usize,
    errors: [*c]u8,
    capacity: usize,
) c_int {
    return io.writeHalfAsFloat(file(stream, errors, capacity), tensor, offset, count, scratch, scratch_capacity);
}
export fn ds4_native_payload_read_f32_as_f16(
    stream: *c.FILE,
    tensor: ?*c.ds4_gpu_tensor,
    offset: u64,
    count: u64,
    scratch: [*c]u8,
    scratch_capacity: usize,
    remaining: ?*u64,
    errors: [*c]u8,
    capacity: usize,
) c_int {
    return io.readFloatAsHalf(file(stream, errors, capacity), tensor, offset, count, scratch, scratch_capacity, remaining);
}
