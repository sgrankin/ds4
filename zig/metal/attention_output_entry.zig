//! Existing public GPU ABI. All policy/body code lives in the pure owner.
const c = @import("metal_c");
const owner = @import("attention_output.zig");
const Tensor = c.ds4_gpu_tensor;
fn request(out: ?*Tensor, low: ?*Tensor, map: ?*const anyopaque, size: u64, a: u64, b: u64, width: u64, rank: u64, groups: u32, outputs: u64, heads: ?*const Tensor, rows: u32) owner.Projection {
    return .{ .out = out, .low = low, .model = map, .size = size, .a = a, .b = b, .width = width, .rank = rank, .groups = groups, .outputs = outputs, .heads = heads, .rows = rows };
}
export fn ds4_gpu_attention_output_q8_batch_tensor(out: ?*Tensor, low: ?*Tensor, group_tmp: ?*Tensor, low_tmp: ?*Tensor, map: ?*const anyopaque, size: u64, a: u64, b: u64, width: u64, rank: u64, groups: u32, outputs: u64, heads: ?*const Tensor, rows: u32) c_int {
    var q = request(out, low, map, size, a, b, width, rank, groups, outputs, heads, rows);
    q.group_tmp = group_tmp;
    q.low_tmp = low_tmp;
    return @intFromBool(q.batch());
}
export fn ds4_gpu_dsv41_attention_output_batch(out: ?*Tensor, low: ?*Tensor, map: ?*const anyopaque, size: u64, a: u64, b: u64, heads: ?*const Tensor, rows: u32) c_int {
    var q = request(out, low, map, size, a, b, 4096, 1024, 8, 5120, heads, rows);
    q.group_tmp = low;
    q.low_tmp = low;
    q.round_low = true;
    return @intFromBool(q.batch());
}
export fn ds4_gpu_dsv41_attention_output_tp_batch(out: ?*Tensor, low: ?*Tensor, map: ?*const anyopaque, size: u64, a: u64, b: u64, heads: ?*const Tensor, rows: u32, rank: u32) c_int {
    const shard: u64 = 4 * 1024 * (4096 / 32 * 34);
    if (rank > 1 or a > size or shard * 2 > size - a) return 0;
    var q = request(out, low, map, size, a + rank * shard, b, 4096, 1024, 4, 5120, heads, rows);
    q.group_tmp = low;
    q.low_tmp = low;
    q.round_low = true;
    q.full_low = 8192;
    q.low_offset = @as(u64, rank) * 4096;
    return @intFromBool(q.batch());
}
export fn ds4_gpu_attention_output_q4_K_batch_tensor(out: ?*Tensor, low: ?*Tensor, group_tmp: ?*Tensor, low_tmp: ?*Tensor, map: ?*const anyopaque, size: u64, a: u64, b: u64, output_type: u32, width: u64, rank: u64, groups: u32, outputs: u64, heads: ?*const Tensor, rows: u32) c_int {
    _ = group_tmp;
    _ = low_tmp;
    return @intFromBool(request(out, low, map, size, a, b, width, rank, groups, outputs, heads, rows).q4Batch(output_type));
}
export fn ds4_gpu_attention_output_q8_batch_f16_tensor(out: ?*Tensor, low: ?*Tensor, map: ?*const anyopaque, size: u64, a: u64, b: u64, width: u64, rank: u64, groups: u32, outputs: u64, heads: ?*const Tensor, rows: u32) c_int {
    // This public action was an explicit unsupported stub; do not silently
    // introduce a half-output algorithm while migrating the working family.
    _ = out;
    _ = low;
    _ = map;
    _ = size;
    _ = a;
    _ = b;
    _ = width;
    _ = rank;
    _ = groups;
    _ = outputs;
    _ = heads;
    _ = rows;
    return 0;
}
export fn ds4_gpu_matmul_q8_0_kslice_tensor(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u64, first: u64, count: u64, outputs: u64, input: ?*const Tensor, input_offset: u64) c_int {
    return @intFromBool((owner.KSlice{ .out = out, .model = map, .size = size, .offset = offset, .width = width, .first = first, .count = count, .outputs = outputs, .input = input, .input_offset = input_offset }).scalar());
}
export fn ds4_gpu_matmul_q8_0_kslice_rows_tensor(out: ?*Tensor, map: ?*const anyopaque, size: u64, offset: u64, width: u64, outputs: u64, first: u64, count: u64, input: ?*const Tensor, rows: u64) c_int {
    return @intFromBool((owner.KSlice{ .out = out, .model = map, .size = size, .offset = offset, .width = width, .first = first, .count = count, .outputs = outputs, .input = input }).rows(rows, width, first));
}
export fn ds4_gpu_attention_output_q8_tp_tensor(out: ?*Tensor, low: ?*Tensor, map: ?*const anyopaque, size: u64, a: u64, b: u64, width: u64, rank: u64, total: u32, first: u32, count: u32, outputs: u64, heads: ?*const Tensor) c_int {
    return @intFromBool(request(out, low, map, size, a, b, width, rank, count, outputs, heads, 1).tpScalar(total, first));
}
export fn ds4_gpu_attention_output_low_q8_tensor(low: ?*Tensor, map: ?*const anyopaque, size: u64, a: u64, width: u64, rank: u64, groups: u32, heads: ?*const Tensor) c_int {
    return @intFromBool(request(null, low, map, size, a, 0, width, rank, groups, 0, heads, 1).lowOnly(false, 0));
}
export fn ds4_gpu_attention_output_low_q4_K_slice_tensor(low: ?*Tensor, map: ?*const anyopaque, size: u64, a: u64, width: u64, rank: u64, first: u32, count: u32, heads: ?*const Tensor) c_int {
    return @intFromBool(request(null, low, map, size, a, 0, width, rank, count, 0, heads, 1).lowOnly(true, first));
}
