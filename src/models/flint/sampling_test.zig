//! A no-model GPU qualification of sampler ordering, token handoff and stopping.
//! About one MiB of buffers; shares the production encoder and parameter record.
const std = @import("std");
const c = @import("flint_c");
const decode = @import("decode.zig");

const Fixture = struct {
    gpu: *c.flint_gpu,
    storage: [6]*anyopaque,

    fn init(gpu: *c.flint_gpu) !Fixture {
        const sizes = [_]usize{ decode.vocab * 4, 8 * 4, 128 * 128 * 8, 128 * 2 * 4, 17 * 4, decode.workers * 4 };
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
        self.* = undefined;
    }

    fn choose(self: *Fixture, options: decode.SampleOptions, active: bool) !u32 {
        const config = try decode.SampleConfig.init(options, 0, 1);
        const parameters = decode.contents(u32, self.storage[1], 8);
        parameters[0..4].* = .{ 42, 17, 0, if (active) 49 else 0 };
        @memset(parameters[4..8], 0);
        @memset(decode.contents(u32, self.storage[4], 17), 0);
        const command = c.flint_gpu_command_begin(self.gpu) orelse return error.GpuCommand;
        const dispatched = decode.encodeSampling(self.gpu, command, .{
            .parameters = self.storage[1],
            .candidates = self.storage[2],
            .normalizers = self.storage[3],
            .record = self.storage[4],
            .diagnostic = self.storage[5],
        }, &config, self.storage[0], 0, true);
        const completed = c.flint_gpu_command_end(self.gpu, command, true);
        if (!dispatched or !completed) return error.SamplerDispatch;
        const record = decode.contents(u32, self.storage[4], 17);
        if (!active) {
            if (record[16] != 0 or parameters[7] != 0) return error.InactiveSamplerWroteToken;
            return 0;
        }
        if (record[16] != 1 or record[0] != 42 or record[1] >= decode.vocab or parameters[4] != record[1] or parameters[5] != 18 or parameters[6] != 0) return error.SamplerHandoff;
        var stop = false;
        for (options.eos) |eos| stop = stop or record[1] == eos;
        if (parameters[7] != @as(u32, if (stop) 0 else 49)) return error.SamplerStop;
        return record[1];
    }
};

pub fn run(gpu: *c.flint_gpu) !void {
    // Caller compiles decode.compile(gpu) once before this fixture.
    var fixture = try Fixture.init(gpu);
    defer fixture.deinit();
    const logits = decode.contents(f32, fixture.storage[0], decode.vocab);
    // The two largest words carry only 30% of the full-vocabulary mass. At
    // top-p .5, both survive before top-k2; normalizing top-k first would wrongly
    // leave only the first word. This protects the actual sampling order.
    @memset(logits, @log(@as(f32, 0.7) / @as(f32, @floatFromInt(decode.vocab - 2))));
    logits[7] = @log(@as(f32, 0.2));
    logits[19] = @log(@as(f32, 0.1));
    if (try fixture.choose(.{ .temperature = 0 }, true) != 7) return error.SamplerArgmax;
    var observed_second: u32 = 0;
    for (0..64) |seed| {
        const chosen = try fixture.choose(.{ .top_k = 2, .top_p = 0.5, .seed = seed }, true);
        if (chosen != 7 and chosen != 19) return error.SamplerTopK;
        observed_second += @intFromBool(chosen == 19);
    }
    if (observed_second == 0) return error.SamplerNucleusOrder;
    for (0..8) |seed| {
        if (try fixture.choose(.{ .top_k = 20, .top_p = 0.1, .seed = seed }, true) != 7) return error.SamplerNucleus;
        if (try fixture.choose(.{ .top_k = 20, .top_p = 1, .min_p = 0.75, .seed = seed }, true) != 7) return error.SamplerMinP;
    }
    // The general-k variant and EOS/inactive records use the same production
    // handoff; neither needs a full model to prove that it does not advance.
    if (try fixture.choose(.{ .top_k = 128, .top_p = 0.1, .eos = &.{7} }, true) != 7) return error.SamplerGeneralK;
    _ = try fixture.choose(.{}, false);
    std.debug.print("flint: GPU sampler ordering/handoff/EOS passed (second word {d}/64)\n", .{observed_second});
}
