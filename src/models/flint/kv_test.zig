//! Exercise the real KV publish/assemble kernels without a model or a 65k
//! prefill. Two one-layer banks plus assembled K/V need approximately384MiB.
const std = @import("std");
const c = @import("flint_c");
const decode = @import("decode.zig");
const prefill = @import("prefill.zig");

const boundary: u32 = 65536;
const start: u32 = boundary - 16;
const rows: u32 = 32;
const total: u32 = start + rows;
const bank_elements: usize = 2 * 4 * boundary * 128;

fn pattern(value: bool, position: u32, head: u32, dimension: u32) u16 {
    var bits = position *% 0x85ebca6b ^ head *% 0xc2b2ae35 ^ dimension *% 0x9e3779b9;
    bits ^= bits >> 16;
    return @as(u16, if (value) 0xbf00 else 0x3f00) | @as(u16, @intCast(bits & 127));
}

fn binding(handle: *anyopaque, index: u32) c.flint_gpu_binding {
    return .{ .handle = handle, .offset = 0, .index = index };
}

const Fixture = struct {
    gpu: *c.flint_gpu,
    // bank0, bank1, two tiny fallback banks, current K/V, assembled K/V.
    storage: [8]*anyopaque,

    fn init(gpu: *c.flint_gpu) !Fixture {
        const sizes = [_]usize{
            bank_elements * 2, bank_elements * 2, 4,               4,
            rows * 512 * 2,    rows * 512 * 2,    total * 512 * 2, total * 512 * 2,
        };
        var result: Fixture = .{ .gpu = gpu, .storage = undefined };
        var completed: usize = 0;
        errdefer for (result.storage[0..completed]) |buffer| c.flint_gpu_buffer_free(buffer);
        for (&result.storage, sizes, 0..) |*buffer, size, i| {
            buffer.* = c.flint_gpu_buffer(gpu, size) orelse return error.GpuAllocation;
            completed += 1;
            // Deterministic old history; output buffers are fully written by
            // assemble and need no CPU pass over their 128MiB.
            if (i < 4) @memset(decode.contents(u8, buffer.*, size), 0);
        }
        return result;
    }

    fn deinit(self: *Fixture) void {
        for (self.storage) |buffer| c.flint_gpu_buffer_free(buffer);
        self.* = undefined;
    }

    fn fillCurrent(self: *Fixture, first: u32) void {
        const keys = decode.contents(u16, self.storage[4], rows * 512);
        const values = decode.contents(u16, self.storage[5], rows * 512);
        for (0..rows) |row| for (0..4) |head| for (0..128) |dim| {
            const index = (row * 4 + head) * 128 + dim;
            keys[index] = pattern(false, first + @as(u32, @intCast(row)), @intCast(head), @intCast(dim));
            values[index] = pattern(true, first + @as(u32, @intCast(row)), @intCast(head), @intCast(dim));
        };
    }

    fn dispatch(self: *Fixture, command: *anyopaque, name: [*:0]const u8, params: *const prefill.Kv, bindings: []const c.flint_gpu_binding, elements: u32) bool {
        return c.flint_gpu_dispatch(self.gpu, command, name, params, @sizeOf(prefill.Kv), bindings.ptr, bindings.len, &[_]u32{ (elements + 255) / 256, 1, 1 }, &[_]u32{ 256, 1, 1 });
    }

    fn banks(self: *Fixture) [4]c.flint_gpu_binding {
        return .{ binding(self.storage[0], 1), binding(self.storage[1], 2), binding(self.storage[2], 3), binding(self.storage[3], 4) };
    }

    fn publish(self: *Fixture, command: *anyopaque, params: *const prefill.Kv, assembled: bool) bool {
        const buffers = self.banks();
        const source: usize = if (assembled) 6 else 4;
        const bindings = [_]c.flint_gpu_binding{
            buffers[0],                       buffers[1],                           buffers[2], buffers[3],
            binding(self.storage[source], 5), binding(self.storage[source + 1], 6),
        };
        const count = if (params.is_global != 0) params.rows else @min(params.total, params.capacity);
        return self.dispatch(command, "flint_prefill_kv_publish", params, &bindings, count * 512);
    }

    fn assemble(self: *Fixture, command: *anyopaque, params: *const prefill.Kv) bool {
        const buffers = self.banks();
        const bindings = [_]c.flint_gpu_binding{
            buffers[0],                  buffers[1],                  buffers[2],                  buffers[3],
            binding(self.storage[4], 5), binding(self.storage[5], 6), binding(self.storage[6], 7), binding(self.storage[7], 8),
        };
        return self.dispatch(command, "flint_prefill_kv_assemble", params, &bindings, params.total * 512);
    }

    fn expect(actual: u16, expected: u16, phase: []const u8, position: u32, head: u32, dim: u32) !void {
        if (actual == expected) return;
        std.debug.print("flint: KV {s} position={d} head={d} dim={d}: 0x{x} != 0x{x}\n", .{ phase, position, head, dim, actual, expected });
        return error.KvLayoutMismatch;
    }

    fn globalBoundary(self: *Fixture) !void {
        self.fillCurrent(start);
        // Sample old history on every head, including the entry immediately
        // before the published range; these must survive the crossing write.
        const bank0 = decode.contents(u16, self.storage[0], bank_elements);
        for ([_]u32{ 0, 12345, start - 1 }) |position| for (0..4) |head| for ([_]u32{ 0, 31, 63, 64, 127 }) |dim| {
            const index = (head * boundary + position) * 128 + dim;
            bank0[index] = pattern(false, position, @intCast(head), dim);
            bank0[index + boundary * 512] = pattern(true, position, @intCast(head), dim);
        };
        const params: prefill.Kv = .{ .offset = 0, .capacity = boundary, .position = start, .rows = rows, .is_global = 1, .prior = start, .total = total, .initial_position = start };
        const command = c.flint_gpu_command_begin(self.gpu) orelse return error.GpuCommand;
        const dispatched = self.publish(command, &params, false) and self.assemble(command, &params);
        const completed = c.flint_gpu_command_end(self.gpu, command, true);
        if (!dispatched or !completed) return error.KvDispatch;
        const output_k = decode.contents(u16, self.storage[6], total * 512);
        const output_v = decode.contents(u16, self.storage[7], total * 512);
        for (0..rows) |row| for (0..4) |head| for (0..128) |dim| {
            const position = start + @as(u32, @intCast(row));
            const expected_k = pattern(false, position, @intCast(head), @intCast(dim));
            const expected_v = pattern(true, position, @intCast(head), @intCast(dim));
            const selected = self.storage[position / boundary];
            const raw = decode.contents(u16, selected, bank_elements);
            const index = (head * boundary + position % boundary) * 128 + dim;
            try expect(raw[index], expected_k, "global bank K", position, @intCast(head), @intCast(dim));
            try expect(raw[index + boundary * 512], expected_v, "global bank V", position, @intCast(head), @intCast(dim));
            const out = (head * total + position) * 128 + dim;
            try expect(output_k[out], expected_k, "global assembled K", position, @intCast(head), @intCast(dim));
            try expect(output_v[out], expected_v, "global assembled V", position, @intCast(head), @intCast(dim));
        };
        for ([_]u32{ 0, 12345, start - 1 }) |position| for (0..4) |head| for ([_]u32{ 0, 31, 63, 64, 127 }) |dim| {
            const out = (head * total + position) * 128 + dim;
            try expect(output_k[out], pattern(false, position, @intCast(head), dim), "global history K", position, @intCast(head), dim);
            try expect(output_v[out], pattern(true, position, @intCast(head), dim), "global history V", position, @intCast(head), dim);
        };
    }

    fn localRing(self: *Fixture) !void {
        const cap: u32 = 513;
        const position: u32 = 700;
        const initial: u32 = 600;
        const offset: u32 = 1024;
        const length: u32 = cap + rows;
        const oldest = position - cap;
        const cursor = (position - initial) % cap;
        const bank = decode.contents(u16, self.storage[0], bank_elements);
        for (0..cap) |row| for (0..4) |head| for (0..128) |dim| {
            const absolute = oldest + @as(u32, @intCast(row));
            const slot = (cursor + row) % cap;
            const index = offset + (head * cap + slot) * 128 + dim;
            bank[index] = pattern(false, absolute, @intCast(head), @intCast(dim));
            bank[index + cap * 512] = pattern(true, absolute, @intCast(head), @intCast(dim));
        };
        self.fillCurrent(position);
        const params: prefill.Kv = .{ .offset = offset, .capacity = cap, .position = position, .rows = rows, .is_global = 0, .prior = cap, .total = length, .initial_position = initial };
        const command = c.flint_gpu_command_begin(self.gpu) orelse return error.GpuCommand;
        const dispatched = self.assemble(command, &params) and self.publish(command, &params, true);
        const completed = c.flint_gpu_command_end(self.gpu, command, true);
        if (!dispatched or !completed) return error.KvDispatch;
        const output_k = decode.contents(u16, self.storage[6], length * 512);
        const output_v = decode.contents(u16, self.storage[7], length * 512);
        for (0..length) |row| for (0..4) |head| for (0..128) |dim| {
            const absolute = oldest + @as(u32, @intCast(row));
            const index = (head * length + row) * 128 + dim;
            try expect(output_k[index], pattern(false, absolute, @intCast(head), @intCast(dim)), "local assembled K", absolute, @intCast(head), @intCast(dim));
            try expect(output_v[index], pattern(true, absolute, @intCast(head), @intCast(dim)), "local assembled V", absolute, @intCast(head), @intCast(dim));
        };
        for (0..cap) |row| for (0..4) |head| for (0..128) |dim| {
            const absolute = position + rows - cap + @as(u32, @intCast(row));
            const index = offset + (head * cap + row) * 128 + dim;
            try expect(bank[index], pattern(false, absolute, @intCast(head), @intCast(dim)), "local compacted K", absolute, @intCast(head), @intCast(dim));
            try expect(bank[index + cap * 512], pattern(true, absolute, @intCast(head), @intCast(dim)), "local compacted V", absolute, @intCast(head), @intCast(dim));
        };
    }
};

pub fn run(gpu: *c.flint_gpu) !void {
    // The smoke caller has already compiled the actual prefill operations.
    var fixture = try Fixture.init(gpu);
    defer fixture.deinit();
    try fixture.globalBoundary();
    try fixture.localRing();
    std.debug.print("flint: GPU KV bank65536 crossing and chronological local-ring publish/assemble passed\n", .{});
}
