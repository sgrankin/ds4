//! Real small files, mappings and pthread jobs. Only model-span selection and
//! Metal tensor/view publication are stand-ins; no inference/device work runs.
const std = @import("std");
const c = @import("pagein_c");
const pagein = @import("pagein");
const IO = pagein.IO;
const allocator = std.heap.c_allocator;
var source_spans: [2]c.ds4_model_map_span = undefined;
var spans_ok = true;
var span_requests: u32 = 0;
var last_decode = false;
var map_ok = true;
var mapped_count: u32 = 0;
var mapped_offsets: [2]u64 = undefined;
var mapped_sizes: [2]u64 = undefined;
var ids: [6]i32 = .{ 0, 2, 2, 5, 5, 0 };

export fn ds4_native_pagein_layer_spans(_: [*c]const c.ds4_weights, _: u32, decode: bool, out: [*c]c.ds4_model_map_span_vec) bool {
    span_requests += 1;
    last_decode = decode;
    if (!spans_ok) return false;
    const spans = allocator.dupe(c.ds4_model_map_span, &source_spans) catch unreachable;
    out.* = .{ .v = spans.ptr, .len = @intCast(spans.len), .cap = @intCast(spans.len), .max_tensor_bytes = 65536 };
    return true;
}
export fn ds4_gpu_set_model_map_spans(_: ?*const anyopaque, _: u64, offsets: [*c]const u64, sizes: [*c]const u64, count: u32, _: u64) c_int {
    std.debug.assert(count == source_spans.len);
    mapped_count = count;
    @memcpy(&mapped_offsets, offsets[0..count]);
    @memcpy(&mapped_sizes, sizes[0..count]);
    return @intFromBool(map_ok);
}
export fn ds4_gpu_tensor_read(_: ?*const c.ds4_gpu_tensor, offset: u64, output: ?*anyopaque, bytes: u64) c_int {
    std.debug.assert(offset == 0 and bytes == @sizeOf(@TypeOf(ids)));
    const destination: [*]u8 = @ptrCast(output.?);
    @memcpy(destination[0..bytes], std.mem.asBytes(&ids));
    return 1;
}

fn setting(name: [*:0]const u8, value: [*:0]const u8) void {
    std.debug.assert(c.setenv(name, value, 1) == 0);
}

test "page-in real I/O, range partitioning, selected coalescing and slot joins" {
    const expect = std.testing.expect;
    const equal = std.testing.expectEqual;
    for ([_][*:0]const u8{
        "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE",
        "DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREAD",
        "DS4_METAL_DISABLE_STREAMING_PREFILL_SELECTED_MADVISE",
    }) |name| _ = c.unsetenv(name);
    _ = c.mkdir(".workspaces", 0o700);
    _ = c.mkdir(".workspaces/pagein-tests", 0o700);
    var path: [256]u8 = undefined;
    _ = c.snprintf(&path, path.len, ".workspaces/pagein-tests/fixture-%d.bin", c.getpid());
    const fd = c.open(&path, c.O_CREAT | c.O_TRUNC | c.O_RDWR, @as(c_uint, 0o600));
    try expect(fd >= 0);
    defer _ = c.close(fd);
    defer _ = c.unlink(&path);
    const page: usize = @intCast(c.getpagesize());
    const bytes = page * 12;
    const contents = try allocator.alloc(u8, bytes);
    defer allocator.free(contents);
    for (contents, 0..) |*byte, index| byte.* = @truncate(index *% 13);
    try equal(@as(isize, @intCast(bytes)), c.write(fd, contents.ptr, contents.len));
    const mapping = c.mmap(null, bytes, c.PROT_READ, c.MAP_SHARED, fd, 0);
    try expect(mapping != c.MAP_FAILED);
    defer _ = c.munmap(mapping, bytes);
    var model = std.mem.zeroes(c.ds4_model);
    model.fd = fd;
    model.map = @ptrCast(mapping);
    model.size = bytes;
    var weights = std.mem.zeroes(c.ds4_weights);
    var touched: u64 = 0;
    var sink: u8 = 0;
    try expect(IO.pread(&model, 7, page + 5, &touched, &sink));
    try equal(@as(u64, page + 5), touched);
    try equal(contents[7] ^ contents[page + 11], sink);
    touched = 0;
    sink = 0;
    try expect(IO.touch(&model, page + 3, page + 5, &touched, &sink));
    try equal(@as(u64, page * 2), touched);
    try equal(contents[page] ^ contents[page * 2] ^ contents[page * 3 - 1], sink);
    try expect(!IO.touch(&model, bytes, 1, &touched, &sink));
    try expect(!IO.pread(&model, 0, 0, &touched, &sink));
    try expect(IO.advise(null, 0, 0, false, null));
    touched = std.math.maxInt(u64) - 1;
    try expect(IO.advise(&model, 1, 1, true, &touched));
    try equal(std.math.maxInt(u64), touched);
    IO.readAhead(&model, 0, bytes, true);
    model.size += page;
    touched = 0;
    try expect(!IO.pread(&model, bytes - 8, 16, &touched, &sink));
    try equal(@as(u64, 8), touched);
    model.size = bytes;

    source_spans = .{
        .{ .off = page + 17, .end = page * 5 + 13, .isolate = false },
        .{ .off = page * 8, .end = page * 9, .isolate = false },
    };
    const span_vec: c.ds4_model_map_span_vec = .{ .v = &source_spans, .len = 2, .cap = 2, .max_tensor_bytes = 65536 };
    try expect(pagein.install(&model, &span_vec, "fixture"));
    try equal(@as(u32, 2), mapped_count);
    try equal(source_spans[0].off, mapped_offsets[0]);
    try equal(source_spans[0].end - source_spans[0].off, mapped_sizes[0]);
    map_ok = false;
    try expect(!pagein.install(&model, &span_vec, "fixture"));
    map_ok = true;

    setting("DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_THREADS", "4");
    var job = std.mem.zeroes(pagein.Job);
    try expect(pagein.layerStart(&model, &weights, 0, 7, c.DS4_STREAM_PREPARE_PREAD | c.DS4_STREAM_PREPARE_DECODE, false, false, 4, &job));
    try expect(job.started and last_decode);
    try equal(@as(u32, 4), job.n_threads);
    try equal(@as(u32, 8), job.n_ranges);
    try equal(source_spans[0].off, job.ranges[0].off);
    try equal(source_spans[0].end, job.ranges[3].off + job.ranges[3].size);
    try expect(pagein.join(&job, false));
    try expect(!job.started and job.ranges == null and job.workers == null and job.threads == null);
    setting("DS4_METAL_STREAMING_PREFILL_LAYER_PREPARE_THREADS", "1");
    try expect(pagein.layerStart(&model, &weights, 0, 7, c.DS4_STREAM_PREPARE_PREAD, false, false, 4, &job));
    try expect(job.started and !last_decode and job.n_threads == 1);
    try expect(pagein.join(&job, false));

    var slots: [2]pagein.Slot = @splat(std.mem.zeroes(pagein.Slot));
    const flags = c.DS4_STREAM_PREPARE_PREAD;
    const requests = span_requests;
    try expect(pagein.prepareStart(&model, &weights, 0, 1, flags, false, false, 4, &slots));
    try expect(pagein.prepareStart(&model, &weights, 0, 1, flags, false, false, 4, &slots));
    try equal(requests + 1, span_requests);
    try expect(pagein.prepareStart(&model, &weights, 1, 1, flags, false, false, 4, &slots));
    try expect(!pagein.prepareStart(&model, &weights, 2, 1, flags, false, false, 4, &slots));
    try expect(pagein.prepareJoin(&model, &weights, 0, 1, flags, false, false, 4, &slots));
    try expect(pagein.prepareStart(&model, &weights, 2, 1, flags, false, false, 4, &slots));
    try expect(pagein.prepareJoinAll(&slots));
    try expect(!slots[0].active and !slots[1].active);
    try expect(pagein.prepareJoin(&model, &weights, 3, 1, flags, false, false, 4, &slots));
    model.fd = -1;
    try expect(pagein.layerStart(&model, &weights, 0, 1, flags, false, false, 4, &job));
    try expect(!pagein.join(&job, false));
    try expect(job.ranges == null and job.workers == null);
    model.fd = fd;
    spans_ok = false;
    try expect(!pagein.layerStart(&model, &weights, 0, 1, flags, false, false, 4, &job));
    spans_ok = true;
    setting("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE", "1");
    try expect(pagein.layerStart(null, null, 0, 1, flags, false, false, 4, &job));
    try expect(!job.started);
    _ = c.unsetenv("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE");

    var gate_weight = std.mem.zeroes(c.ds4_tensor);
    var up_weight = std.mem.zeroes(c.ds4_tensor);
    var down_weight = std.mem.zeroes(c.ds4_tensor);
    gate_weight.abs_offset = page;
    up_weight.abs_offset = page * 3;
    down_weight.abs_offset = page * 5;
    var layer = std.mem.zeroes(c.ds4_layer_weights);
    layer.ffn_gate_exps = &gate_weight;
    layer.ffn_up_exps = &up_weight;
    layer.ffn_down_exps = &down_weight;
    setting("DS4_METAL_ENABLE_STREAMING_PREFILL_SELECTED_MADVISE", "1");
    setting("DS4_METAL_STREAMING_PREFILL_SELECTED_PREPARE_GAP", "1");
    setting("DS4_METAL_STREAMING_PREFILL_SELECTED_PREPARE_THREADS", "3");
    const tensor: *c.ds4_gpu_tensor = @ptrFromInt(0x1000);
    try expect(pagein.selectedStart(&model, &layer, tensor, 0, 2, 64, 128, true, true, 8, 3, &job));
    try expect(job.started and job.madvise_only);
    try equal(@as(u32, 3), job.unique);
    try equal(@as(u32, 6), job.n_ranges);
    try equal(@as(u32, 3), job.n_threads);
    try equal(@as(u64, page), job.ranges[0].off);
    try equal(@as(u64, 64 * 3), job.ranges[0].size);
    try equal(@as(u64, page + 64 * 5), job.ranges[3].off);
    try expect(pagein.join(&job, true));
    ids[0] = -1;
    try expect(!pagein.selectedStart(&model, &layer, tensor, 0, 2, 64, 128, true, true, 8, 3, &job));
    try expect(!job.started and job.ranges == null);
}
