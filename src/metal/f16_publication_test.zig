//! CPU oracle for the real native F32-to-F16 publication controller.
const std = @import("std");
const native = @import("f16_publication.zig");
const api = @import("bridge.zig");
const Tensor = api.Tensor;
const Fake = struct { bytes: u64, offset: u64 = 0, buffer: bool = true };
var checks: usize = 0;
var cases: usize = 0;
var descriptors: usize = 0;
var init_ok = true;
var init_calls: usize = 0;
var acquire_ok = true;
var batch_active = false;
var disable_contig: c_int = 0;
var max_threads: u64 = 256;
var acquired: usize = 0;
var encoded: usize = 0;
var released: usize = 0;
var finished: usize = 0;
var fail_encode: usize = 0;
var release_ok: c_int = 1;
var token_live = false;
var token: u8 = 0;
var expected_src: ?*const Tensor = null;
var expected_dst: ?*const Tensor = null;
var expected_src_off: u64 = 0;
var expected_dst_off: u64 = 0;
var expected_count: u64 = 0;
var done: u64 = 0;

fn check(ok: bool) void {
    checks += 1;
    if (!ok) @panic("F16 publication descriptor/ownership mismatch");
}
fn tensor(value: *Fake) *Tensor {
    return @ptrCast(value);
}
fn fake(value: *const Tensor) *const Fake {
    return @ptrCast(@alignCast(value));
}
fn reset() void {
    init_ok = true;
    init_calls = 0;
    acquire_ok = true;
    batch_active = false;
    disable_contig = 0;
    max_threads = 256;
    acquired = 0;
    encoded = 0;
    released = 0;
    finished = 0;
    fail_encode = 0;
    release_ok = 1;
    token_live = false;
    done = 0;
}
pub export fn ds4_gpu_init() c_int {
    init_calls += 1;
    return @intFromBool(init_ok);
}
pub export fn ds4_gpu_tensor_bytes(t: ?*const Tensor) u64 {
    return if (t) |v| fake(v).bytes else 0;
}
pub export fn ds4_zig_metal_env_bool(name: [*:0]const u8) c_int {
    check(std.mem.eql(u8, std.mem.span(name), "DS4_METAL_DISABLE_CONTIG_F32_F16_COPY"));
    return disable_contig;
}
pub export fn ds4_zig_metal_copy_max_threads(pipeline: u32) u64 {
    const wanted: u32 = if (disable_contig <= 0) api.contiguous else api.generic;
    check(pipeline == wanted);
    return max_threads;
}
pub export fn ds4_zig_metal_command_acquire(owned: *c_int) ?*anyopaque {
    check(!token_live);
    acquired += 1;
    if (!acquire_ok) return null;
    owned.* = @intFromBool(!batch_active);
    token_live = true;
    return &token;
}
pub export fn ds4_zig_metal_copy_encode(command: ?*anyopaque, launch: *const api.CopyLaunch) c_int {
    check(token_live and command == @as(?*anyopaque, @ptrCast(&token)));
    encoded += 1;
    descriptors += 1;
    const n: u64 = @min(expected_count - done, @as(u64, 0xffffffff));
    const contig = disable_contig <= 0;
    const work = if (contig) (n + 3) / 4 else n;
    var nth: u64 = 32;
    while (nth < work and nth < max_threads) nth *= 2;
    nth = @min(nth, max_threads);
    nth = @min(nth, work);
    if (nth == 0) nth = 1;
    check(launch.args_bytes == (if (contig) @as(u64, 4) else 136));
    const bytes: [*]const u8 = @ptrCast(launch.args.?);
    if (contig) {
        const expected: u32 = @intCast(n);
        check(std.mem.eql(u8, bytes[0..4], std.mem.asBytes(&expected)));
    } else {
        // Original ds4_gpu_make_cpy_1d_args layout: all 17 words, no padding.
        const expected = [_]u64{ n, n, 1, 1, 1, 4, n * 4, n * 4, n * 4, n, 1, 1, 1, 2, n * 2, n * 2, n * 2 };
        check(std.mem.eql(u8, bytes[0..136], std.mem.asBytes(&expected)));
    }
    check(launch.pipeline == (if (contig) api.contiguous else api.generic));
    check(launch.src == expected_src and launch.dst == expected_dst);
    // View base offsets belong to the bridge and must not be added twice.
    check(launch.src_offset == expected_src_off + done * 4);
    check(launch.dst_offset == expected_dst_off + done * 2);
    check(launch.groups == (work + nth - 1) / nth and launch.threads == nth);
    done += n;
    return @intFromBool(encoded != fail_encode and fake(expected_src.?).buffer and fake(expected_dst.?).buffer);
}
pub export fn ds4_zig_metal_command_release(command: ?*anyopaque, owned: c_int, ok: c_int, label: [*:0]const u8) c_int {
    check(token_live and command == @as(?*anyopaque, @ptrCast(&token)));
    check(owned == @intFromBool(!batch_active));
    check(std.mem.eql(u8, std.mem.span(label), "tensor f32 to f16 copy"));
    released += 1;
    token_live = false;
    if (ok != 0 and owned != 0) finished += 1;
    return if (ok != 0) release_ok else 0;
}
fn invoke(dst: *Fake, dst_off: u64, src: *Fake, src_off: u64, count: u64) c_int {
    cases += 1;
    expected_dst = tensor(dst);
    expected_src = tensor(src);
    expected_dst_off = dst_off;
    expected_src_off = src_off;
    expected_count = count;
    done = 0;
    return native.ds4_gpu_tensor_copy_f32_to_f16(tensor(dst), dst_off, tensor(src), src_off, count);
}

pub fn main() void {
    const counts = [_]u64{ 1, 3, 4, 5, 33, 1025, 0xffffffff + @as(u64, 9), 2 * @as(u64, 0xffffffff) + 17 };
    for ([_]c_int{ 0, 1 }) |disabled| {
        for ([_]u64{ 31, 256, 1024 }) |maximum| {
            for ([_]bool{ false, true }) |borrowed| {
                for (counts) |n| {
                    reset();
                    disable_contig = disabled;
                    max_threads = maximum;
                    batch_active = borrowed;
                    var src = Fake{ .bytes = n * 4 + 20, .offset = 128 };
                    var dst = Fake{ .bytes = n * 2 + 14, .offset = 64 };
                    check(invoke(&dst, 14, &src, 20, n) == 1);
                    check(done == n and acquired == 1 and released == 1 and !token_live);
                    check(encoded == (n + 0xffffffff - 1) / 0xffffffff);
                    check(finished == @intFromBool(!borrowed) and batch_active == borrowed);
                }
            }
        }
    }
    reset();
    var src = Fake{ .bytes = 0, .buffer = false };
    var dst = Fake{ .bytes = 0, .buffer = false };
    check(native.ds4_gpu_tensor_copy_f32_to_f16(null, 0, tensor(&src), 0, 0) == 0 and init_calls == 0);
    check(invoke(&dst, ~@as(u64, 0), &src, ~@as(u64, 0), 0) == 1 and init_calls == 1 and acquired == 0);
    init_ok = false;
    check(invoke(&dst, 0, &src, 0, 0) == 0 and acquired == 0);
    reset();
    src = .{ .bytes = 16 };
    dst = .{ .bytes = 8 };
    check(invoke(&dst, 0, &src, 0, ~@as(u64, 0) / 4 + 1) == 0);
    check(invoke(&dst, 0, &src, 1, 4) == 0);
    check(invoke(&dst, 1, &src, 0, 4) == 0);
    check(invoke(&dst, 9, &src, 0, 1) == 0 and acquired == 0);
    reset();
    acquire_ok = false;
    check(invoke(&dst, 0, &src, 0, 4) == 0 and acquired == 1 and released == 0);
    for ([_]bool{ false, true }) |borrowed| {
        reset();
        batch_active = borrowed;
        fail_encode = 2;
        const n: u64 = 2 * @as(u64, 0xffffffff) + 17;
        src.bytes = n * 4;
        dst.bytes = n * 2;
        check(invoke(&dst, 0, &src, 0, n) == 0);
        check(encoded == 2 and acquired == 1 and released == 1 and finished == 0 and !token_live);
        check(batch_active == borrowed);
    }
    reset();
    src = .{ .bytes = 16, .buffer = false };
    dst = .{ .bytes = 8 };
    check(invoke(&dst, 0, &src, 0, 4) == 0 and released == 1 and finished == 0);
    reset();
    src.buffer = true;
    release_ok = 0;
    check(invoke(&dst, 0, &src, 0, 4) == 0 and released == 1);
    _ = std.c.printf("F16 publication: %zu cases / %zu descriptors / %zu checks PASS\n", cases, descriptors, checks);
}
