//! Synthetic logits only: mask placement, sparse/empty groups, finite values,
//! exact counter-RNG parity with the original kernels, and a small warm timing.
const std = @import("std");
const c = @import("flint_c");
const decode = @import("decode.zig");

const Fixture = struct {
    gpu: *c.flint_gpu,
    storage: [7]*anyopaque,
    gpu_seconds: f64 = 0,

    fn init(gpu: *c.flint_gpu) !Fixture {
        const sizes = [_]usize{ decode.vocab * 4, (decode.packet_width + 1) * 4 * 4, 128 * 128 * 8, 128 * 2 * 4, 17 * 4, decode.workers * decode.packet_width * 4, decode.mask_words * decode.packet_width * 4 };
        var result: Fixture = .{ .gpu = gpu, .storage = undefined };
        var completed: usize = 0;
        errdefer for (result.storage[0..completed]) |buffer| c.flint_gpu_buffer_free(buffer);
        for (&result.storage, sizes) |*buffer, size| {
            buffer.* = c.flint_gpu_buffer(gpu, size) orelse return error.GpuAllocation;
            @memset(decode.contents(u8, buffer.*, size), 0);
            completed += 1;
        }
        return result;
    }

    fn deinit(self: *Fixture) void {
        for (self.storage) |buffer| c.flint_gpu_buffer_free(buffer);
    }

    fn logits(self: *Fixture) []f32 {
        return decode.contents(f32, self.storage[0], decode.vocab);
    }

    fn mask(self: *Fixture) []u32 {
        return decode.contents(u32, self.storage[6], decode.mask_words);
    }

    fn permit(self: *Fixture, id: u32) void {
        self.mask()[id >> 5] |= @as(u32, 1) << @intCast(id & 31);
    }

    fn choose(self: *Fixture, options: decode.SampleOptions, masked: bool, fp32: bool) !u32 {
        const config = try decode.SampleConfig.init(options, 0, 1);
        const parameters = decode.contents(u32, self.storage[1], 8);
        parameters[0..4].* = .{ 42, 17, 0, 49 };
        @memset(parameters[4..8], 0);
        @memset(decode.contents(u32, self.storage[4], 17), 0);
        const buffers: decode.SampleBuffers = .{
            .parameters = self.storage[1],
            .candidates = self.storage[2],
            .normalizers = self.storage[3],
            .record = self.storage[4],
            .diagnostic = self.storage[5],
        };
        const command = c.flint_gpu_command_begin(self.gpu) orelse return error.GpuCommand;
        const dispatched = if (masked)
            decode.encodeMaskedSampling(self.gpu, command, buffers, self.storage[6], &config, self.storage[0], 0, fp32)
        else
            decode.encodeSampling(self.gpu, command, buffers, &config, self.storage[0], 0, fp32);
        const completed = c.flint_gpu_command_end_profile(self.gpu, command, &self.gpu_seconds);
        if (!dispatched or !completed) return error.SamplerDispatch;
        if (masked) {
            for (decode.contents(f32, self.storage[3], 256), 0..) |value, i| {
                if (std.math.isNan(value) or (i % 2 == 1 and (!std.math.isFinite(value) or value < 0))) return error.NonfiniteNormalizer;
            }
        }
        const record = decode.contents(u32, self.storage[4], 17);
        if (record[16] == 0) return error.EmptyMaskOrNonfinite;
        if (record[16] != 1 or record[0] != 42 or record[1] >= decode.vocab or parameters[4] != record[1] or parameters[5] != 18) return error.SamplerHandoff;
        if (masked and (self.mask()[record[1] >> 5] & (@as(u32, 1) << @intCast(record[1] & 31))) == 0) return error.MaskedTokenSelected;
        var stop = false;
        for (options.eos) |id| stop = stop or id == record[1];
        if (parameters[7] != @as(u32, if (stop) 0 else 49)) return error.SamplerStop;
        return record[1];
    }
};

fn now() f64 {
    var time: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &time);
    return @as(f64, @floatFromInt(time.tv_sec)) + @as(f64, @floatFromInt(time.tv_nsec)) * 1e-9;
}

fn run(fixture: *Fixture) !void {
    @memset(fixture.mask(), 0xffffffff);
    const logits = fixture.logits();
    @memset(logits, @log(@as(f32, 0.7) / @as(f32, @floatFromInt(decode.vocab - 2))));
    logits[7] = @log(@as(f32, 0.2));
    logits[19] = @log(@as(f32, 0.1));
    for ([_]decode.SampleOptions{
        .{ .temperature = 0 },
        .{ .top_k = 20, .top_p = 0.5 },
        .{ .top_k = 128, .top_p = 1, .min_p = 0.05 },
        .{ .top_k = 0, .top_p = 1 },
    }) |base| {
        for (0..16) |seed| {
            var options = base;
            options.seed = seed;
            const ordinary = try fixture.choose(options, false, true);
            const masked = try fixture.choose(options, true, true);
            if (ordinary != masked) return error.FullMaskCounterParity;
        }
    }

    // A forbidden huge logit cannot affect either the normalizer or top-k.
    @memset(fixture.mask(), 0);
    fixture.permit(7);
    fixture.permit(19);
    logits[1] = 100;
    for (0..16) |seed| {
        if (try fixture.choose(.{ .top_k = 20, .top_p = 0.5, .seed = seed }, true, true) != 7) return error.MaskedNucleus;
        if (try fixture.choose(.{ .top_k = 128, .top_p = 1, .min_p = 0.75, .seed = seed }, true, true) != 7) return error.MaskedMinP;
    }

    // Full-vocabulary mass must still precede top-k even with sparse groups.
    for (32768..33768) |id| {
        fixture.permit(@intCast(id));
        logits[id] = @log(@as(f32, 0.7) / 1000);
    }
    var second: u32 = 0;
    for (0..64) |seed| {
        const choice = try fixture.choose(.{ .top_k = 2, .top_p = 0.5, .seed = seed }, true, true);
        if (choice != 7 and choice != 19) return error.MaskedTopK;
        second += @intFromBool(choice == 19);
    }
    if (second == 0) return error.MaskedNucleusOrder;

    // Most groups have no mass. Extreme finite logits and NaNs must not make
    // the masked groups contribute NaN or select an invalid/disallowed ID.
    @memset(fixture.mask(), 0);
    fixture.permit(decode.vocab - 1);
    @memset(logits, std.math.nan(f32));
    logits[decode.vocab - 1] = -1000;
    if (try fixture.choose(.{ .top_k = 128 }, true, true) != decode.vocab - 1) return error.SparseFiniteChoice;
    if (try fixture.choose(.{ .temperature = 0, .eos = &.{decode.vocab - 1} }, true, true) != decode.vocab - 1) return error.SparseGreedy;
    @memset(fixture.mask(), 0);
    try std.testing.expectError(error.EmptyMaskOrNonfinite, fixture.choose(.{}, true, true));
    @memset(fixture.mask(), 0xffffffff);
    @memset(logits, std.math.nan(f32));
    try std.testing.expectError(error.EmptyMaskOrNonfinite, fixture.choose(.{}, true, true));

    // Decode samples its original BF16 frontier buffer; prefill uses FP32.
    const bf16 = decode.contents(u16, fixture.storage[0], decode.vocab);
    @memset(bf16, @truncate(@as(u32, @bitCast(@as(f32, -20))) >> 16));
    bf16[123] = @truncate(@as(u32, @bitCast(@as(f32, 3))) >> 16);
    @memset(fixture.mask(), 0);
    fixture.permit(123);
    if (try fixture.choose(.{ .top_k = 20 }, true, false) != 123) return error.MaskedBF16;
    if (try fixture.choose(.{ .top_k = 128 }, true, false) != 123) return error.MaskedBF16General;
    std.debug.print("Flint masked GPU sampler: parity/filtering/sparse/empty/NaN/BF16/EOS passed; second {d}/64\n", .{second});
}

fn benchmark(fixture: *Fixture) !void {
    @memset(fixture.mask(), 0xffffffff);
    @memset(fixture.logits(), -5);
    fixture.logits()[7] = 0;
    fixture.logits()[19] = -1;
    var wall: [2][2]f64 = undefined;
    var gpu: [2][2]f64 = undefined;
    var repeats: [2]usize = @splat(0);
    for ([_]bool{ false, true, true, false }) |masked| {
        const arm: usize = @intFromBool(masked);
        const start = now();
        var total_gpu: f64 = 0;
        for (0..128) |seed| {
            _ = try fixture.choose(.{ .seed = seed }, masked, true);
            total_gpu += fixture.gpu_seconds;
        }
        wall[arm][repeats[arm]] = (now() - start) / 128;
        gpu[arm][repeats[arm]] = total_gpu / 128;
        repeats[arm] += 1;
    }
    const ordinary_wall = (wall[0][0] + wall[0][1]) / 2;
    const masked_wall = (wall[1][0] + wall[1][1]) / 2;
    const ordinary_gpu = (gpu[0][0] + gpu[0][1]) / 2;
    const masked_gpu = (gpu[1][0] + gpu[1][1]) / 2;
    std.debug.print("Synthetic warm ABBA, all-allowed mask, T1/p.95/k20: wall {d:.2}->{d:.2} us, GPU {d:.2}->{d:.2} us per sampler; no model/whole-request claim\n", .{ ordinary_wall * 1e6, masked_wall * 1e6, ordinary_gpu * 1e6, masked_gpu * 1e6 });
}

const packet_source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\kernel void flint_fixture_forward(constant uint& step [[buffer(0)]],
    \\    device atomic_uint* markers [[buffer(1)]]) {
    \\    atomic_store_explicit(markers + step, 1u, memory_order_relaxed);
    \\}
;

const PacketMatcher = struct {
    markers: []u32,
    accepted: usize = 0,
    masks: usize = 0,
    rolled_back: usize = 0,
    fail_mask: ?usize = null,
    fail_accept: ?usize = null,
    overlap: bool = true,

    const choices = [_]u32{ 7, 19, 123, 4096, 8000, 17000, 50000, decode.vocab - 1 };

    fn mask(raw: ?*anyopaque, words: [*c]u32, count: usize) callconv(.c) c_int {
        const self: *PacketMatcher = @ptrCast(@alignCast(raw.?));
        if (count != decode.mask_words) return -1;
        if (self.masks != 0) {
            // The GPU's following forward must be able to run while its CPU
            // mask is in progress. A sampler-before-forward fence deadlocks
            // this check, which exits with an explicit failure after 100 ms.
            const start = now();
            const marker: *u32 = &self.markers[self.accepted];
            while (@atomicLoad(u32, marker, .acquire) == 0) {
                if (now() - start > 0.1) {
                    self.overlap = false;
                    return -1;
                }
            }
        }
        self.masks += 1;
        if (self.fail_mask == self.accepted) return -1;
        @memset(words[0..count], 0);
        const token = choices[self.accepted];
        words[token >> 5] |= @as(u32, 1) << @intCast(token & 31);
        return 0;
    }

    fn accept(raw: ?*anyopaque, token: c_int) callconv(.c) c_int {
        const self: *PacketMatcher = @ptrCast(@alignCast(raw.?));
        if (self.fail_accept == self.accepted or token != choices[self.accepted]) return -1;
        self.accepted += 1;
        return 0;
    }

    fn rollback(raw: ?*anyopaque, count: usize) callconv(.c) c_int {
        const self: *PacketMatcher = @ptrCast(@alignCast(raw.?));
        if (count > self.accepted) return -1;
        self.accepted -= count;
        self.rolled_back += count;
        return 0;
    }
};

fn packetCase(fixture: *Fixture, count: u32, eos: []const u32, fail_mask: ?usize, fail_accept: ?usize) !void {
    const markers = c.flint_gpu_buffer(fixture.gpu, decode.packet_width * 4) orelse return error.GpuAllocation;
    defer c.flint_gpu_buffer_free(markers);
    const marker_values = decode.contents(u32, markers, decode.packet_width);
    @memset(marker_values, 0);
    var matcher: PacketMatcher = .{ .markers = marker_values, .fail_mask = fail_mask, .fail_accept = fail_accept };
    const callbacks: c.ds4_token_constraint = .{ .context = &matcher, .mask = PacketMatcher.mask, .accept = PacketMatcher.accept, .rollback = PacketMatcher.rollback };
    const params = decode.contents(u32, fixture.storage[1], (decode.packet_width + 1) * 4);
    @memset(params, 0);
    params[0..4].* = .{ 42, 17, 0, 49 };
    const record = decode.contents(u32, fixture.storage[4], 17);
    @memset(record, 0);
    @memset(decode.contents(u32, fixture.storage[5], decode.workers * decode.packet_width), 0);
    const masks = decode.contents(u32, fixture.storage[6], decode.mask_words * decode.packet_width);
    @memset(masks, 0);
    @memset(fixture.logits(), 0);
    var grammar: decode.ConstraintPacket = .{ .constraint = &callbacks, .parameters = params, .record = record, .masks = masks };
    try grammar.fill(0);
    const handoff = c.flint_gpu_handoff_create(fixture.gpu, &grammar, decode.ConstraintPacket.ready) orelse return error.GpuHandoff;
    const command = c.flint_gpu_command_begin(fixture.gpu) orelse return error.GpuCommand;
    var dispatched = true;
    const buffers: decode.SampleBuffers = .{
        .parameters = fixture.storage[1],
        .candidates = fixture.storage[2],
        .normalizers = fixture.storage[3],
        .record = fixture.storage[4],
        .diagnostic = fixture.storage[5],
    };
    for (0..count) |i| {
        const step: u32 = @intCast(i);
        const binding: c.flint_gpu_binding = .{ .handle = markers, .index = 1, .offset = 0 };
        dispatched = c.flint_gpu_dispatch(fixture.gpu, command, "flint_fixture_forward", &step, @sizeOf(u32), &binding, 1, &[_]u32{ 1, 1, 1 }, &[_]u32{ 1, 1, 1 });
        if (!dispatched) break;
        if (step != 0) c.flint_gpu_handoff_wait(handoff, command, step - 1);
        const config = try decode.SampleConfig.init(.{ .temperature = 0, .eos = eos }, step, count);
        dispatched = decode.encodeMaskedSamplingAt(fixture.gpu, command, buffers, fixture.storage[6], @as(u64, step) * decode.mask_words * 4, &config, fixture.storage[0], 0, true);
        if (!dispatched) break;
        if (step + 1 < count) c.flint_gpu_handoff_signal(handoff, command, step);
    }
    const completed = c.flint_gpu_command_end(fixture.gpu, command, true);
    c.flint_gpu_handoff_free(handoff);
    if (!dispatched or !completed) return error.PacketDispatch;
    if (!matcher.overlap) return error.GrammarForwardOverlap;
    if (fail_mask != null or fail_accept != null) {
        if (grammar.failure == null) return error.MissingCallbackFailure;
        const advanced = grammar.advanced;
        try grammar.rollback();
        if (matcher.accepted != 0 or matcher.rolled_back != advanced) return error.CallbackRollback;
        return;
    }
    const expected: u32 = if (eos.len == 0) count else 2;
    if (record[16] != expected or matcher.accepted != expected - 1 or grammar.advanced != matcher.accepted) return error.GrammarCommitCount;
    for (0..expected) |step| {
        const input = if (step == 0) 42 else PacketMatcher.choices[step - 1];
        if (record[2 * step] != input or record[2 * step + 1] != PacketMatcher.choices[step]) return error.GrammarPacketTokens;
    }
}

fn packetHandoff(fixture: *Fixture) !void {
    var errors: [4096]u8 = @splat(0);
    if (!c.flint_gpu_compile(fixture.gpu, packet_source.ptr, packet_source.len, &errors, errors.len)) {
        std.debug.print("{s}\n", .{std.mem.sliceTo(&errors, 0)});
        return error.PacketFixtureCompilation;
    }
    try packetCase(fixture, 1, &.{}, null, null);
    try packetCase(fixture, 8, &.{}, null, null);
    try packetCase(fixture, 8, &.{19}, null, null);
    try packetCase(fixture, 8, &.{}, 2, null);
    try packetCase(fixture, 8, &.{}, null, 2);
    std.debug.print("Flint constrained packet: 1/8 steps, forward-mask overlap, EOS, callback failure/rollback passed (synthetic logits, no model)\n", .{});
}

pub fn main() !void {
    const gpu = c.flint_gpu_create() orelse return error.MetalUnavailable;
    defer c.flint_gpu_free(gpu);
    var errors: [4096]u8 = @splat(0);
    if (!c.flint_gpu_compile(gpu, decode.sampling_source.ptr, decode.sampling_source.len, &errors, errors.len)) {
        std.debug.print("{s}\n", .{std.mem.sliceTo(&errors, 0)});
        return error.SamplerCompilation;
    }
    var fixture = try Fixture.init(gpu);
    defer fixture.deinit();
    try run(&fixture);
    try packetHandoff(&fixture);
    try benchmark(&fixture);
}
