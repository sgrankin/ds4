//! Original BF16 PLE rows stay on SSD. Sorting groups nearby requests and
//! duplicate rows without changing output order; libdispatch joins every reader
//! before the caller stages its Metal command batch. No persistent row cache.
const std = @import("std");
const c = @import("gguf_c");
const Config = @import("config.zig").Config;
const Model = c.ds4_model;
const max_rows = 4096;
const max_readers = 16;

fn fail(code: c_int) bool {
    c.__error().* = code;
    return false;
}

pub fn readRow(model: *const Model, row: u32, output: []f32) bool {
    const weight = model.ngram_tensor;
    if (weight == null or model.ngram_fd < 0 or row >= weight.*.dim[1] or
        weight.*.type != c.DS4_TENSOR_BF16 or weight.*.dim[0] == 0 or weight.*.dim[0] > 160 or
        output.len < weight.*.dim[0]) return fail(c.EINVAL);
    var raw: [320]u8 = undefined;
    const bytes: usize = @intCast(weight.*.dim[0] * 2);
    const offset = weight.*.abs_offset + @as(u64, row) * bytes;
    var done: usize = 0;
    while (done < bytes) {
        const received = c.pread(model.ngram_fd, raw[done..].ptr, bytes - done, @intCast(offset + done));
        if (received < 0 and c.__error().* == c.EINTR) continue;
        if (received <= 0 or @as(usize, @intCast(received)) > bytes - done) {
            if (received == 0) c.__error().* = c.EIO;
            return false;
        }
        done += @intCast(received);
    }
    for (output[0..@intCast(weight.*.dim[0])], 0..) |*value, index| {
        const bits = (@as(u32, raw[index * 2]) | (@as(u32, raw[index * 2 + 1]) << 8)) << 16;
        if (bits & 0x7f800000 == 0x7f800000) return fail(c.EDOM);
        value.* = @bitCast(bits);
    }
    return true;
}

const Request = struct {
    row: u32,
    output: u32,

    fn before(_: void, a: Request, b: Request) bool {
        return a.row < b.row;
    }
};

const Batch = struct {
    model: *const Model,
    requests: []const Request,
    output: []f32,
    width: usize,
    readers: usize,
    errors: [max_readers]c_int = @splat(0),
};

fn readPart(context: ?*anyopaque, part: usize) callconv(.c) void {
    const batch: *Batch = @ptrCast(@alignCast(context.?));
    const first = batch.requests.len * part / batch.readers;
    const end = batch.requests.len * (part + 1) / batch.readers;
    var previous: []const f32 = &.{};
    for (first..end) |index| {
        const request = batch.requests[index];
        const start = @as(usize, request.output) * batch.width;
        const destination = batch.output[start..][0..batch.width];
        if (index > first and request.row == batch.requests[index - 1].row) {
            @memcpy(destination, previous);
        } else if (!readRow(batch.model, request.row, destination)) {
            batch.errors[part] = c.__error().*;
            return;
        }
        previous = destination;
    }
}

pub fn read(model: *const Model, rows: []const u32, output: []f32) bool {
    const weight = model.ngram_tensor;
    if (weight == null or model.ngram_fd < 0 or rows.len > std.math.maxInt(usize) / (160 * @sizeOf(f32))) return fail(c.EINVAL);
    const width: usize = @intCast(weight.*.dim[0]);
    if (width == 0 or width > 160 or output.len < rows.len * width) return fail(c.EINVAL);
    for (rows) |row| if (row >= weight.*.dim[1]) return fail(c.EINVAL);
    if (rows.len < 2) return rows.len == 0 or readRow(model, rows[0], output[0..width]);
    const requests = std.heap.c_allocator.alloc(Request, max_rows) catch return fail(c.ENOMEM);
    defer std.heap.c_allocator.free(requests);
    var offset: usize = 0;
    while (offset < rows.len) {
        const count = @min(max_rows, rows.len - offset);
        const active = requests[0..count];
        for (active, rows[offset..][0..count], 0..) |*request, row, index|
            request.* = .{ .row = row, .output = @intCast(index) };
        std.mem.sortUnstable(Request, active, {}, Request.before);
        var batch = Batch{
            .model = model,
            .requests = active,
            .output = output[offset * width ..][0 .. count * width],
            .width = width,
            .readers = @min(count, max_readers),
        };
        c.dispatch_apply_f(batch.readers, c.dispatch_get_global_queue(c.QOS_CLASS_USER_INITIATED, 0), &batch, readPart);
        for (batch.errors[0..batch.readers]) |code| if (code != 0) return fail(code);
        offset += count;
    }
    return true;
}

/// EOS cuts all older context. The trained uint64 multiply intentionally wraps
/// before the per-head prime modulus; checked arithmetic would change its hash.
pub fn step(config: *const Config, token: c_int, previous: *[c.DS4_MAX_PLE_NGRAM]c_int, rows: []u32) void {
    const shape = &config.shape;
    const per = shape.n_ple_heads_per_ngram;
    const eos: c_int = @intCast(shape.ple_eos_id);
    var context: [c.DS4_MAX_PLE_NGRAM]c_int = undefined;
    context[0] = token;
    var cut = false;
    for (1..shape.n_ple_ngram) |index| {
        const old = if (cut) eos else previous[index - 1];
        cut = cut or old == eos;
        context[index] = if (cut) eos else old;
    }
    for (2..shape.n_ple_ngram + 1) |n| {
        var mixed = @as(u64, @intCast(context[0])) *% config.ple.multiplier[0];
        for (1..n) |index| mixed ^= @as(u64, @intCast(context[index])) *% config.ple.multiplier[index];
        for (0..per) |group| {
            const head = (n - 2) * per + group;
            rows[head] = @as(u32, @intCast(mixed % config.ple.head_vocab[head])) +% config.ple.head_offset[head];
        }
    }
    var index: usize = previous.len - 1;
    while (index > 0) : (index -= 1) previous[index] = previous[index - 1];
    previous[0] = token;
}
