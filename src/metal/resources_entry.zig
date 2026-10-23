//! Sole public C facade. Resource ownership/policy is reusable without exports.
const owner = @import("resources.zig");
const Tensor = owner.Tensor;
export fn ds4_gpu_tensor_alloc(bytes: u64) ?*Tensor {
    return owner.alloc(bytes);
}
export fn ds4_gpu_tensor_alloc_managed(bytes: u64) ?*Tensor {
    return owner.alloc(bytes);
}
export fn ds4_gpu_should_use_managed_kv_cache(_: u64, _: u64) c_int {
    return 0;
}
export fn ds4_gpu_tensor_view(base: ?*const Tensor, offset: u64, bytes: u64) ?*Tensor {
    return owner.view(base, offset, bytes);
}
export fn ds4_gpu_tensor_free(tensor: ?*Tensor) void {
    owner.free(tensor);
}
export fn ds4_gpu_tensor_bytes(tensor: ?*const Tensor) u64 {
    return owner.byteSize(tensor);
}
export fn ds4_gpu_tensor_contents(tensor: ?*Tensor) ?*anyopaque {
    return owner.contents(tensor);
}
export fn ds4_gpu_tensor_fill_f32(tensor: ?*Tensor, value: f32, count: u64) c_int {
    return @intFromBool(owner.fillF32(tensor, value, count));
}
export fn ds4_gpu_tensor_write(tensor: ?*Tensor, offset: u64, data: ?*const anyopaque, bytes: u64) c_int {
    return @intFromBool(owner.write(tensor, offset, data, bytes));
}
export fn ds4_gpu_tensor_read(tensor: ?*const Tensor, offset: u64, data: ?*anyopaque, bytes: u64) c_int {
    return @intFromBool(owner.read(tensor, offset, data, bytes));
}
export fn ds4_gpu_tensor_copy(dst: ?*Tensor, dst_offset: u64, src: ?*const Tensor, src_offset: u64, bytes: u64) c_int {
    return @intFromBool(owner.copy(dst, dst_offset, src, src_offset, bytes));
}
export fn ds4_native_metal_tensor_stats() owner.Stats {
    return owner.stats();
}
export fn ds4_native_metal_tensor_tracking_reset() void {
    owner.resetTracking();
}
