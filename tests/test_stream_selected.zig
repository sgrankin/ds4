//! Real pthread mailbox with small Metal API stand-ins: no model/device work.
const std = @import("std");
const c = @import("stream_selected_c");
const selected = @import("stream_selected");

const Mock = struct {
    service_notes: u32 = 0,
    wait_ok: bool = true,
    read_ok: bool = true,
    load_ok: bool = true,
    override_ok: bool = true,
    block_wait: bool = false,
    entered_wait: bool = false,
    release_wait: bool = false,
    event: u64 = 0,
    read_bytes: u64 = 0,
    waits: u32 = 0,
    reads: u32 = 0,
    loads: u32 = 0,
    overrides: u32 = 0,
    ids: [3]i32 = .{ 2, 5, 7 },
    loaded_ids: [3]i32 = @splat(-1),
    override_ids: [3]i32 = @splat(-1),
    table: c.ds4_gpu_stream_expert_table = std.mem.zeroes(c.ds4_gpu_stream_expert_table),
};
var mock: Mock = .{};
var gate: c.pthread_mutex_t = c.ds4_stream_selected_mutex_init;
var gate_changed: c.pthread_cond_t = c.ds4_stream_selected_cond_init;

export fn ds4_gpu_stream_expert_cache_note_service_thread() void {
    mock.service_notes += 1;
}
export fn ds4_gpu_wait_selected_readback_ready(event: u64, label: [*c]const u8) c_int {
    std.debug.assert(std.mem.eql(u8, std.mem.span(label), "selected-id async expert load"));
    mock.waits += 1;
    mock.event = event;
    if (mock.block_wait) {
        _ = c.pthread_mutex_lock(&gate);
        mock.entered_wait = true;
        _ = c.pthread_cond_signal(&gate_changed);
        while (!mock.release_wait) _ = c.pthread_cond_wait(&gate_changed, &gate);
        _ = c.pthread_mutex_unlock(&gate);
    }
    return @intFromBool(mock.wait_ok);
}
export fn ds4_gpu_tensor_read(_: ?*const c.ds4_gpu_tensor, offset: u64, output: ?*anyopaque, bytes: u64) c_int {
    mock.reads += 1;
    mock.read_bytes = bytes;
    std.debug.assert(offset == 0 and bytes == @sizeOf(@TypeOf(mock.ids)));
    if (!mock.read_ok) return 0;
    const destination: [*]u8 = @ptrCast(output.?);
    @memcpy(destination[0..bytes], std.mem.asBytes(&mock.ids));
    return 1;
}
export fn ds4_gpu_stream_expert_cache_begin_selected_load(table: [*c]const c.ds4_gpu_stream_expert_table, ids: [*c]const i32, count: u32) c_int {
    std.debug.assert(count == mock.ids.len);
    mock.loads += 1;
    mock.table = table.*;
    @memcpy(&mock.loaded_ids, ids[0..count]);
    return @intFromBool(mock.load_ok);
}
export fn ds4_gpu_routed_moe_set_selected_override(ids: [*c]const i32, count: u32) c_int {
    std.debug.assert(count == mock.ids.len);
    mock.overrides += 1;
    @memcpy(&mock.override_ids, ids[0..count]);
    return @intFromBool(mock.override_ok);
}

fn reset() void {
    // Called only after finish: the mailbox's mutex/condition handoff makes
    // worker writes visible, and it cannot read another job until start.
    const notes = mock.service_notes;
    mock = .{ .service_notes = notes };
}

test "selected worker owns one mailbox and retains IDs through failures" {
    const expect = std.testing.expect;
    const equal = std.testing.expectEqual;
    const tensor: *c.ds4_gpu_tensor = @ptrFromInt(0x1000);
    var model = std.mem.zeroes(c.ds4_model);
    model.map = @ptrFromInt(0x4000);
    model.size = 65536;
    var gate_weight = std.mem.zeroes(c.ds4_tensor);
    var up_weight = std.mem.zeroes(c.ds4_tensor);
    var down_weight = std.mem.zeroes(c.ds4_tensor);
    gate_weight.abs_offset = 128;
    up_weight.abs_offset = 256;
    down_weight.abs_offset = 512;
    var layer = std.mem.zeroes(c.ds4_layer_weights);
    layer.ffn_gate_exps = &gate_weight;
    layer.ffn_up_exps = &up_weight;
    layer.ffn_down_exps = &down_weight;
    var job = std.mem.zeroes(selected.Job);
    var other = std.mem.zeroes(selected.Job);

    try expect(!selected.finish(null));
    try expect(!selected.finish(&job));
    try expect(!selected.start(null, tensor, &model, &layer, 4, 11, 1024, 2048, 16, 3));
    try expect(!selected.start(&job, null, &model, &layer, 4, 11, 1024, 2048, 16, 3));
    try expect(!selected.start(&job, tensor, &model, &layer, 4, 0, 1024, 2048, 16, 3));

    mock.block_wait = true;
    try expect(selected.start(&job, tensor, &model, &layer, 4, 11, 1024, 2048, 16, 3));
    _ = c.pthread_mutex_lock(&gate);
    while (!mock.entered_wait) _ = c.pthread_cond_wait(&gate_changed, &gate);
    _ = c.pthread_mutex_unlock(&gate);
    try expect(job.active);
    try expect(!selected.start(&other, tensor, &model, &layer, 9, 12, 1, 2, 16, 3));
    try expect(!other.active);
    _ = c.pthread_mutex_lock(&gate);
    mock.release_wait = true;
    _ = c.pthread_cond_signal(&gate_changed);
    _ = c.pthread_mutex_unlock(&gate);
    try expect(selected.finish(&job));
    try expect(!job.active and job.ok and job.ids_ok);
    try equal(@as(u32, 1), mock.service_notes);
    try equal(@as(u64, 11), mock.event);
    try equal(@as(u64, 12), mock.read_bytes);
    try equal(@as(u32, 1), mock.loads);
    try equal(@as(u32, 1), mock.overrides);
    try std.testing.expectEqualSlices(i32, &mock.ids, job.selected_ids[0..3]);
    try std.testing.expectEqualSlices(i32, &mock.ids, &mock.loaded_ids);
    try std.testing.expectEqualSlices(i32, &mock.ids, &mock.override_ids);
    try equal(@as(u32, 4), mock.table.layer);
    try equal(@as(u32, 16), mock.table.n_total_expert);
    try equal(@as(u64, 128), mock.table.gate_offset);
    try equal(@as(u64, 256), mock.table.up_offset);
    try equal(@as(u64, 512), mock.table.down_offset);
    try equal(@as(u64, 1024), mock.table.gate_expert_bytes);
    try equal(@as(u64, 2048), mock.table.down_expert_bytes);
    try equal(model.map, @as([*c]const u8, @ptrCast(mock.table.model_map)));
    try equal(model.size, mock.table.model_size);
    try expect(!selected.finish(&job));

    reset();
    mock.wait_ok = false;
    try expect(selected.start(&job, tensor, &model, &layer, 4, 21, 1024, 2048, 16, 3));
    try expect(!selected.finish(&job));
    try expect(!job.active and !job.ok and !job.ids_ok);
    try equal(@as(u32, 0), mock.reads);
    try equal(@as(u32, 0), mock.loads);
    try equal(@as(u32, 0), mock.overrides);

    reset();
    mock.read_ok = false;
    try expect(selected.start(&job, tensor, &model, &layer, 4, 22, 1024, 2048, 16, 3));
    try expect(!selected.finish(&job));
    try expect(!job.ids_ok and !job.ok);
    try equal(@as(u32, 1), mock.reads);
    try equal(@as(u32, 0), mock.loads);

    reset();
    mock.ids[1] = 16;
    try expect(selected.start(&job, tensor, &model, &layer, 4, 23, 1024, 2048, 16, 3));
    try expect(!selected.finish(&job));
    try expect(!job.ids_ok and !job.ok);
    try equal(@as(u32, 0), mock.loads);
    try equal(@as(i32, 16), job.selected_ids[1]);

    reset();
    mock.load_ok = false;
    try expect(selected.start(&job, tensor, &model, &layer, 4, 24, 1024, 2048, 16, 3));
    try expect(!selected.finish(&job));
    try expect(job.ids_ok and !job.ok and !job.active);
    try std.testing.expectEqualSlices(i32, &mock.ids, job.selected_ids[0..3]);
    try equal(@as(u32, 1), mock.loads);
    try equal(@as(u32, 0), mock.overrides);

    reset();
    mock.override_ok = false;
    try expect(selected.start(&job, tensor, &model, &layer, 4, 25, 1024, 2048, 16, 3));
    try expect(!selected.finish(&job));
    try expect(job.ids_ok and job.ok and !job.active);
    try equal(@as(u32, 1), mock.overrides);

    reset();
    try expect(selected.start(&job, tensor, &model, &layer, 4, 26, 1024, 2048, 0, 3));
    try expect(!selected.finish(&job));
    try expect(!job.ids_ok and !job.ok);
    try equal(@as(u32, 0), mock.waits);

    reset();
    try expect(selected.start(&job, tensor, &model, &layer, 5, 27, 1024, 2048, 16, 3));
    try expect(selected.finish(&job));
    try equal(@as(u32, 1), mock.service_notes);
}
