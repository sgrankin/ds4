//! Sole public TP-control facade. Transport setup/thread/shutdown stays C.
const c = @import("metal_c");
const owner = @import("tp_control.zig");
const Tensor = c.ds4_gpu_tensor;
export fn ds4_gpu_signal_selected_readback_ready(value: ?*u64) c_int {
    return if (value) |out| @intFromBool(owner.signalSelected(out)) else 0;
}
export fn ds4_gpu_commit_and_wait_selected_readback(value: u64, label: ?[*:0]const u8) c_int {
    return @intFromBool(owner.commitSelected(value, label));
}
export fn ds4_gpu_wait_selected_readback_ready(value: u64, label: ?[*:0]const u8) c_int {
    return @intFromBool(owner.waitSelected(value, label));
}
export fn ds4_gpu_tp_flag_fold_request(layer: u32, gate: u32) void {
    owner.requestFold(layer, gate);
}
export fn ds4_native_metal_tp_fold_take(out: ?*const Tensor, bytes: u64, slot: *u32, value: *u32) c_int {
    return @intFromBool(owner.takeFold(out, bytes, slot, value));
}
export fn ds4_gpu_add_tensor_tp_flag(out: ?*Tensor, a: ?*const Tensor, b: ?*const Tensor, count: u32, layer: u32, gate: u32) c_int {
    return @intFromBool(owner.addFold(out, a, b, count, layer, gate));
}
export fn ds4_gpu_tp_gate_encode(layer: u32, gate: u32) c_int {
    return @intFromBool(owner.rowGate(layer, gate));
}
export fn ds4_gpu_tp_batch_gate_encode(layer: u32, rows: u32) c_int {
    return @intFromBool(owner.batch(layer, rows));
}
export fn ds4_gpu_tp_big_gate_encode(layer: u32, rows: u32, out: ?*const Tensor, input: ?*Tensor, bytes: u64) c_int {
    const sequence = owner.bigKick(layer, rows, out, input, bytes);
    return if (sequence == 0) 0 else @intFromBool(owner.bigWait(sequence));
}
export fn ds4_native_metal_tp_signal_batch_wait(label: ?[*:0]const u8) c_int {
    return @intFromBool(owner.signalBatchWait(label));
}
export fn ds4_gpu_tp_set_session_batch_mode(enabled: c_int) void {
    owner.setBatchMode(enabled);
}
export fn ds4_gpu_tp_decode_split_flush_safe() c_int {
    return @intFromBool(owner.splitFlushSafe());
}

const Prefetch = @import("tp_prefetch.zig");
export fn ds4_gpu_tp_gate_prefetch_plan(gate: u32, map: ?*const anyopaque, size: u64, offsets: [*c]const u64, lengths: [*c]const u64, count: u32) c_int {
    return Prefetch.plan(gate, map, size, offsets, lengths, count);
}
export fn ds4_native_metal_tp_prefetch_encode(gate: u32) c_int {
    return Prefetch.encode(gate);
}
