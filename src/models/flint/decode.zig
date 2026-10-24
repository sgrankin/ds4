//! Native Flint resident decoder, derived from recursive-oss/rec e53663b262af
//! (Apache-2.0). Zig owns the Metal buffers; no tensor runtime is involved.
//! A decoder belongs to the model, and each conversation has its own Session.
const std = @import("std");
const c = @import("flint_c");
const weights = @import("weights.zig");
const profiles = @import("profile.zig");
pub const Profile = profiles.Profile;

pub const vocab: u32 = 150144;
pub const layers: u32 = 49;
/// Buffers keep the largest profile's stride; a smaller resident grid uses only
/// its first workerCount() diagnostics. The sampler can therefore share its ABI.
pub const workers: u32 = 76;
pub const packet_width: u32 = 8;
pub const block_tokens: u32 = 65536;
pub const context_limit: u32 = block_tokens * 4;
pub const shader_source = @embedFile("../../metal/flint/decode.metal");
const m4_pro_source = @embedFile("../../metal/flint/decode-m4-pro.metal");
const m5_pro_source = @embedFile("../../metal/flint/decode-m5-pro.metal");
const steering_source = @embedFile("../../metal/flint/decode-steer.metal");
const m4_pro_steering_source = @embedFile("../../metal/flint/decode-m4-pro-steer.metal");
const m5_pro_steering_source = @embedFile("../../metal/flint/decode-m5-pro-steer.metal");
const additive_source = @embedFile("../../metal/flint/decode-additive.metal");
const m4_pro_additive_source = @embedFile("../../metal/flint/decode-m4-pro-additive.metal");
const m5_pro_additive_source = @embedFile("../../metal/flint/decode-m5-pro-additive.metal");
pub const sampling_source = @embedFile("../../metal/flint/sampling.metal");
const metadata_bytes: usize = 50 * 64 * 4;
const parameters_bytes: usize = (packet_width + 1) * 4 * 4;
const control_bytes: usize = packet_width * 1024 * 4;
// Six packet fields plus attention/FFN steering scales. Ordinary kernels never
// read the final two fields, and keep their original 30-binding entrypoints.
const sampling_bytes: usize = 8 * 4;
pub const output_bytes = [_]usize{
    packet_width * vocab * 2,
    packet_width * 50 * 2048 * 2,
    packet_width * workers * 4,
    17 * 4,
    78848 * 2,
    32 * 1024 * 128 / 2 * 4,
    32 * 1024 * 2 * 4,
    1280 * 4,
};
/// Everything owned by a decode session except KV. Model-shared SiLU and the
/// serving facade's prefill/checkpoint allocations are accounted separately.
pub const session_scratch_bytes: usize = blk: {
    var bytes = metadata_bytes + parameters_bytes + control_bytes + sampling_bytes;
    for (output_bytes) |size| bytes += size;
    break :blk bytes;
};

fn allocation(gpu: *c.flint_gpu, bytes: usize) !*anyopaque {
    return c.flint_gpu_buffer(gpu, bytes) orelse error.GpuAllocation;
}

pub fn contents(comptime T: type, buffer: *anyopaque, count: usize) []T {
    return @as([*]T, @ptrCast(@alignCast(c.flint_gpu_contents(buffer))))[0..count];
}

fn gpuError(gpu: *c.flint_gpu, operation: []const u8) void {
    const message = c.flint_gpu_error(gpu);
    std.debug.print("flint: {s}: {s}\n", .{ operation, if (message != null) std.mem.span(message) else "Metal failure" });
}

/// Compile without allocating a model or a KV cache; used by the qualification
/// executable as well as model initialization.
pub fn compile(gpu: *c.flint_gpu, profile: Profile) !void {
    var error_text: [4096]u8 = @splat(0);
    const decoder_source: []const u8 = switch (profile) {
        .m4_pro => m4_pro_source,
        .m5_pro => m5_pro_source,
        .m5_max => shader_source,
    };
    for ([_][]const u8{ decoder_source, sampling_source }) |source| {
        if (!c.flint_gpu_compile(gpu, source.ptr, source.len, &error_text, error_text.len)) {
            std.debug.print("flint: decode compilation: {s}\n", .{std.mem.sliceTo(&error_text, 0)});
            return error.CompileDecode;
        }
    }
}

/// Active resident entrypoints live separately so ordinary model startup does
/// not compile additional megakernels. A zero-scale preload compiles them once
/// too, making later live scale changes allocation/compilation-free.
pub fn compileSteering(gpu: *c.flint_gpu, profile: Profile) !void {
    const source: []const u8 = switch (profile) {
        .m4_pro => m4_pro_steering_source,
        .m5_pro => m5_pro_steering_source,
        .m5_max => steering_source,
    };
    var error_text: [4096]u8 = @splat(0);
    if (!c.flint_gpu_compile(gpu, source.ptr, source.len, &error_text, error_text.len)) {
        std.debug.print("flint: steering decode compilation: {s}\n", .{std.mem.sliceTo(&error_text, 0)});
        return error.CompileSteeringDecode;
    }
    const kernels = switch (profile) {
        .m4_pro => [_][*:0]const u8{ "flint_decode_scalar_steer_m4_pro", "flint_decode_greedy8_steer_m4_pro" },
        .m5_pro => [_][*:0]const u8{ "flint_decode_scalar_steer_m5_pro", "flint_decode_greedy8_steer_m5_pro" },
        .m5_max => [_][*:0]const u8{ "flint_decode_scalar_steer", "flint_decode_greedy8_steer" },
    };
    for (kernels) |kernel| if (!c.flint_gpu_prepare(gpu, kernel)) {
        gpuError(gpu, "prepare steering pipeline");
        return error.CompileSteeringPipeline;
    };
}

/// Additive residual steering has no reduction or threadgroup scratch.
pub fn compileAdditiveSteering(gpu: *c.flint_gpu, profile: Profile) !void {
    const selected: []const u8 = switch (profile) {
        .m4_pro => m4_pro_additive_source,
        .m5_pro => m5_pro_additive_source,
        .m5_max => additive_source,
    };
    var error_text: [4096]u8 = @splat(0);
    if (!c.flint_gpu_compile(gpu, selected.ptr, selected.len, &error_text, error_text.len)) {
        std.debug.print("flint: additive steering compilation: {s}\n", .{std.mem.sliceTo(&error_text, 0)});
        return error.CompileAdditiveSteering;
    }
    const kernels = switch (profile) {
        .m4_pro => [_][*:0]const u8{ "flint_decode_scalar_additive_m4_pro", "flint_decode_greedy8_additive_m4_pro" },
        .m5_pro => [_][*:0]const u8{ "flint_decode_scalar_additive_m5_pro", "flint_decode_greedy8_additive_m5_pro" },
        .m5_max => [_][*:0]const u8{ "flint_decode_scalar_additive", "flint_decode_greedy8_additive" },
    };
    for (kernels) |kernel| if (!c.flint_gpu_prepare(gpu, kernel)) {
        gpuError(gpu, "prepare additive steering pipeline");
        return error.CompileAdditiveSteeringPipeline;
    };
}

pub const Decoder = struct {
    model: *const weights.Weights,
    profile: Profile,
    silu: *anyopaque,
    steering: ?*anyopaque = null,
    additive_steering: bool = false,

    pub fn init(model: *const weights.Weights, profile: Profile) !Decoder {
        // A larger device can qualify a smaller schedule. Never launch more
        // resident groups than the physical device's supported profile allows.
        const device = c.flint_gpu_device_name(model.gpu);
        if (device == null) return error.UnsupportedResidentDevice;
        _ = try profiles.select(std.mem.span(device), profile.name());
        try compile(model.gpu, profile);
        const silu = try allocation(model.gpu, 65536 * 2);
        errdefer c.flint_gpu_buffer_free(silu);
        const command = c.flint_gpu_command_begin(model.gpu) orelse return error.GpuCommand;
        const binding: c.flint_gpu_binding = .{ .handle = silu, .offset = 0, .index = 0 };
        const dispatched = c.flint_gpu_dispatch(model.gpu, command, "flint_init_silu", null, 0, &binding, 1, &[_]u32{ 256, 1, 1 }, &[_]u32{ 256, 1, 1 });
        const completed = c.flint_gpu_command_end(model.gpu, command, true);
        if (!dispatched or !completed) {
            gpuError(model.gpu, "SiLU initialization");
            return error.GpuDispatch;
        }
        return .{ .model = model, .profile = profile, .silu = silu };
    }

    fn scalarKernel(self: *const Decoder, active: bool) [*:0]const u8 {
        if (active and self.additive_steering) return switch (self.profile) {
            .m4_pro => "flint_decode_scalar_additive_m4_pro",
            .m5_pro => "flint_decode_scalar_additive_m5_pro",
            .m5_max => "flint_decode_scalar_additive",
        };
        if (active) return switch (self.profile) {
            .m4_pro => "flint_decode_scalar_steer_m4_pro",
            .m5_pro => "flint_decode_scalar_steer_m5_pro",
            .m5_max => "flint_decode_scalar_steer",
        };
        return switch (self.profile) {
            .m4_pro => "flint_decode_scalar_m4_pro",
            .m5_pro => "flint_decode_scalar_m5_pro",
            .m5_max => "flint_decode_scalar",
        };
    }

    fn greedyKernel(self: *const Decoder, active: bool) [*:0]const u8 {
        if (active and self.additive_steering) return switch (self.profile) {
            .m4_pro => "flint_decode_greedy8_additive_m4_pro",
            .m5_pro => "flint_decode_greedy8_additive_m5_pro",
            .m5_max => "flint_decode_greedy8_additive",
        };
        if (active) return switch (self.profile) {
            .m4_pro => "flint_decode_greedy8_steer_m4_pro",
            .m5_pro => "flint_decode_greedy8_steer_m5_pro",
            .m5_max => "flint_decode_greedy8_steer",
        };
        return switch (self.profile) {
            .m4_pro => "flint_decode_greedy8_m4_pro",
            .m5_pro => "flint_decode_greedy8_m5_pro",
            .m5_max => "flint_decode_greedy8",
        };
    }

    pub fn deinit(self: *Decoder) void {
        if (self.steering) |buffer| c.flint_gpu_buffer_free(buffer);
        c.flint_gpu_buffer_free(self.silu);
        self.* = undefined;
    }
};

pub const Packet = struct {
    tokens: [packet_width]u32 = @splat(0),
    len: u32 = 0,
};

pub const SampleOptions = struct {
    temperature: f32 = 1,
    top_p: f32 = 0.95,
    top_k: u32 = 20,
    min_p: f32 = 0,
    seed: u64 = 0,
    eos: []const u32 = &.{},
};

pub const SampleConfig = extern struct {
    step: u32,
    count: u32,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    seed_lo: u32,
    seed_hi: u32,
    eos_count: u32,
    eos: [4]u32,
    min_p: f32,
    padding: [3]u32 = @splat(0),

    pub fn init(options: SampleOptions, step: u32, count: u32) !SampleConfig {
        if (!std.math.isFinite(options.temperature) or options.temperature < 0 or !std.math.isFinite(options.top_p) or options.top_p < 0 or options.top_p > 1 or !std.math.isFinite(options.min_p) or options.min_p < 0 or options.min_p > 1 or options.eos.len > 4) return error.InvalidSampler;
        if (options.top_k > 128 or (options.temperature != 0 and options.top_k == 0 and ((options.top_p > 0 and options.top_p < 1) or options.min_p > 0))) return error.UnsupportedSampler;
        var result: SampleConfig = .{
            .step = step,
            .count = count,
            .temperature = options.temperature,
            .top_p = options.top_p,
            .top_k = if (options.temperature == 0) 1 else options.top_k,
            .seed_lo = @truncate(options.seed),
            .seed_hi = @truncate(options.seed >> 32),
            .eos_count = @intCast(options.eos.len),
            .eos = @splat(0),
            .min_p = options.min_p,
        };
        for (options.eos, 0..) |value, i| {
            if (value >= vocab) return error.InvalidToken;
            result.eos[i] = value;
        }
        return result;
    }
};

pub const SampleBuffers = struct {
    parameters: *anyopaque,
    candidates: *anyopaque,
    normalizers: *anyopaque,
    record: *anyopaque,
    diagnostic: *anyopaque,
};

/// The sampler is also usable on a synthetic vocabulary without loading weights
/// or allocating KV. All buffers belong to the caller; both dispatches are
/// encoded into its existing command buffer.
pub fn encodeSampling(gpu: *c.flint_gpu, command: *anyopaque, buffers: SampleBuffers, config: *const SampleConfig, logits: *anyopaque, offset: u64, fp32: bool) bool {
    const small = config.top_k <= 20;
    const partial: [*:0]const u8 = if (fp32)
        (if (small) "flint_sample_partial20_f32" else "flint_sample_partial128_f32")
    else
        (if (small) "flint_sample_partial20_bf16" else "flint_sample_partial128_bf16");
    const first = [_]c.flint_gpu_binding{
        .{ .handle = logits, .offset = offset, .index = 1 },
        .{ .handle = buffers.candidates, .offset = 0, .index = 2 },
        .{ .handle = buffers.normalizers, .offset = 0, .index = 3 },
        .{ .handle = buffers.parameters, .offset = @as(u64, config.step) * 16, .index = 4 },
    };
    if (!c.flint_gpu_dispatch(gpu, command, partial, config, @sizeOf(SampleConfig), &first, first.len, &[_]u32{ 128, 1, 1 }, &[_]u32{ 256, 1, 1 })) return false;
    const second = [_]c.flint_gpu_binding{
        .{ .handle = buffers.candidates, .offset = 0, .index = 1 },
        .{ .handle = buffers.normalizers, .offset = 0, .index = 2 },
        .{ .handle = buffers.parameters, .offset = @as(u64, config.step) * 16, .index = 3 },
        .{ .handle = buffers.parameters, .offset = @as(u64, config.step + 1) * 16, .index = 4 },
        .{ .handle = buffers.record, .offset = 0, .index = 5 },
        .{ .handle = buffers.diagnostic, .offset = @as(u64, config.step) * workers * 4, .index = 6 },
    };
    return c.flint_gpu_dispatch(gpu, command, if (small) "flint_sample_finish20" else "flint_sample_finish128", config, @sizeOf(SampleConfig), &second, second.len, &[_]u32{ 1, 1, 1 }, &[_]u32{ 256, 1, 1 });
}

pub const Session = struct {
    decoder: *Decoder,
    /// KV layout is public so prefill can write the same buffers directly.
    /// Per layer: [K/V, 4 heads, capacity, 128 BF16]; offsets count BF16 values.
    metadata: [50][64]u32,
    metadata_buffer: *anyopaque,
    kv: [4]*anyopaque,
    position: u32 = 0,
    /// Sliding rings are compacted chronologically after prefill. The resident
    /// decoder starts overwriting slot zero when the prefill filled the ring.
    initial_position: u32 = 0,
    capacity: u32 = 0,
    limit: u32,
    usable: bool = true,

    parameters: *anyopaque,
    control: *anyopaque,
    sampling: *anyopaque,
    // Binding22..29: logits, trace, diagnostics, packet/profile, workspace,
    // attention partials/statistics and expert routing handoff flags.
    outputs: [8]*anyopaque,
    last_logits_offset: usize = 0,
    steering_attn: f32 = 0,
    steering_ffn: f32 = 0,

    pub fn init(decoder: *Decoder, limit: u32) !Session {
        if (limit == 0 or limit > context_limit) return error.ContextLimit;
        const gpu = decoder.model.gpu;
        var kv: [4]*anyopaque = undefined;
        var allocated: usize = 0;
        errdefer for (kv[0..allocated]) |buffer| c.flint_gpu_buffer_free(buffer);
        for (&kv) |*buffer| {
            buffer.* = try allocation(gpu, 4);
            contents(u32, buffer.*, 1)[0] = 0;
            allocated += 1;
        }
        const metadata_buffer = try allocation(gpu, metadata_bytes);
        errdefer c.flint_gpu_buffer_free(metadata_buffer);
        const parameters = try allocation(gpu, parameters_bytes);
        errdefer c.flint_gpu_buffer_free(parameters);
        const control = try allocation(gpu, control_bytes);
        errdefer c.flint_gpu_buffer_free(control);
        const sampling = try allocation(gpu, sampling_bytes);
        errdefer c.flint_gpu_buffer_free(sampling);
        var outputs: [8]*anyopaque = undefined;
        var output_count: usize = 0;
        errdefer for (outputs[0..output_count]) |buffer| c.flint_gpu_buffer_free(buffer);
        for (&outputs, output_bytes) |*buffer, size| {
            buffer.* = try allocation(gpu, size);
            output_count += 1;
        }
        var self: Session = .{
            .decoder = decoder,
            .metadata = decoder.model.metadata,
            .metadata_buffer = metadata_buffer,
            .kv = kv,
            .limit = limit,
            .parameters = parameters,
            .control = control,
            .sampling = sampling,
            .outputs = outputs,
        };
        var global_count: u32 = 0;
        for (self.metadata[0..layers]) |row| global_count += @intFromBool(row[1] != 0);
        if (global_count != 14) return error.UnsupportedAttentionLayout;
        var global_index: u32 = 0;
        var local_index: u32 = 0;
        for (self.metadata[0..layers]) |*row| {
            if (row[1] != 0) {
                row[61] = global_index * block_tokens * 1024;
                row[62] = block_tokens;
                global_index += 1;
            } else {
                row[61] = global_count * block_tokens * 1024 + local_index * 513 * 1024;
                row[62] = 513;
                local_index += 1;
            }
        }
        @memcpy(contents([64]u32, self.metadata_buffer, 50), &self.metadata);
        // The shader's global KV addressing uses 64k-token blocks. Allocate only
        // block zero initially; the other bindings remain tiny valid buffers.
        try self.grow(1);
        return self;
    }

    pub fn deinit(self: *Session) void {
        for (self.kv) |buffer| c.flint_gpu_buffer_free(buffer);
        for (self.outputs) |buffer| c.flint_gpu_buffer_free(buffer);
        c.flint_gpu_buffer_free(self.metadata_buffer);
        c.flint_gpu_buffer_free(self.parameters);
        c.flint_gpu_buffer_free(self.control);
        c.flint_gpu_buffer_free(self.sampling);
        self.* = undefined;
    }

    pub fn grow(self: *Session, required: u32) !void {
        if (!self.usable) return error.InvalidatedSession;
        if (required > self.limit) return error.ContextLimit;
        if (required <= self.capacity) return;
        const count = (required + block_tokens - 1) / block_tokens;
        while (self.capacity / block_tokens < count) {
            const index = self.capacity / block_tokens;
            const global_values = @as(usize, 14) * block_tokens * 1024;
            const local_values = if (index == 0) @as(usize, 35) * 513 * 1024 else 0;
            const buffer = try allocation(self.decoder.model.gpu, (global_values + local_values) * 2);
            c.flint_gpu_buffer_free(self.kv[index]);
            self.kv[index] = buffer;
            self.capacity = (index + 1) * block_tokens;
        }
    }

    /// The result borrows the output buffer and remains valid until this session
    /// decodes again. BF16 bits are returned without a CPU vocabulary conversion.
    pub fn lastLogits(self: *Session) []const u16 {
        return contents(u16, self.outputs[0], packet_width * vocab)[self.last_logits_offset..][0..vocab];
    }

    fn steeringActive(self: *const Session) bool {
        return self.decoder.steering != null and (self.steering_attn != 0 or self.steering_ffn != 0);
    }

    fn publishSteering(self: *Session) void {
        contents(f32, self.sampling, 8)[6..8].* = .{ self.steering_attn, self.steering_ffn };
    }

    fn bindings(self: *Session, step: u32) [31]c.flint_gpu_binding {
        var result: [31]c.flint_gpu_binding = undefined;
        for (self.decoder.model.banks, 0..) |buffer, i| result[i] = .{ .handle = buffer, .offset = 0, .index = @intCast(i) };
        result[13] = .{ .handle = self.metadata_buffer, .offset = 0, .index = 13 };
        for (self.kv, 14..) |buffer, i| result[i] = .{ .handle = buffer, .offset = 0, .index = @intCast(i) };
        result[18] = .{ .handle = self.parameters, .offset = @as(u64, step) * 16, .index = 18 };
        result[19] = .{ .handle = self.decoder.silu, .offset = 0, .index = 19 };
        result[20] = .{ .handle = self.control, .offset = @as(u64, step) * 4096, .index = 20 };
        result[21] = .{ .handle = self.sampling, .offset = 0, .index = 21 };
        for (self.outputs, 22..) |buffer, i| result[i] = .{ .handle = buffer, .offset = 0, .index = @intCast(i) };
        result[22].offset = @as(u64, step) * vocab * 2;
        result[23].offset = @as(u64, step) * 50 * 2048 * 2;
        result[24].offset = @as(u64, step) * workers * 4;
        result[30] = .{ .handle = self.decoder.steering, .offset = 0, .index = 30 };
        return result;
    }

    fn checkDiagnostics(self: *Session, count: u32) !void {
        const diagnostic = contents(u32, self.outputs[2], count * workers);
        for (0..count) |step| for (diagnostic[step * workers ..][0..self.decoder.profile.workerCount()], 0..) |failure, group| {
            if (failure == 0) continue;
            self.usable = false;
            std.debug.print("flint: resident synchronization failed at step {d}, group {d}, phase {d}\n", .{ step, group, failure });
            return error.ResidentSynchronization;
        };
    }

    fn dispatch(self: *Session, token: u32, count: u32, packet: bool, eos: []const u32) !void {
        if (token >= vocab or eos.len > 4) return error.InvalidToken;
        if (count == 0 or count > packet_width) return error.InvalidPacket;
        try self.grow(self.position + count);
        const params = contents(u32, self.parameters, 4);
        params[0..4].* = .{ token, self.position, self.initial_position, layers };
        // A private fresh synchronization epoch is essential. Reusing the
        // allocation is fine; reusing its atomic counters is not.
        @memset(contents(u32, self.control, 1024), 0);
        @memset(contents(u32, self.outputs[2], workers), 0);
        @memset(contents(u32, self.outputs[3], 17), 0);
        if (packet) {
            const sample = contents(f32, self.sampling, 6);
            @memset(sample, 0);
            sample[0] = @floatFromInt(count);
            sample[1] = @floatFromInt(eos.len);
            for (eos, 0..) |value, i| {
                if (value >= vocab) return error.InvalidToken;
                sample[i + 2] = @floatFromInt(value);
            }
        } else contents(u32, self.sampling, 1)[0] = 1;
        self.publishSteering();
        const bound = self.bindings(0);
        const gpu = self.decoder.model.gpu;
        const command = c.flint_gpu_command_begin(gpu) orelse return error.GpuCommand;
        const active = self.steeringActive();
        const dispatched = c.flint_gpu_dispatch(gpu, command, if (packet) self.decoder.greedyKernel(active) else self.decoder.scalarKernel(active), null, 0, &bound, if (active) 31 else 30, &[_]u32{ self.decoder.profile.workerCount(), 1, 1 }, &[_]u32{ 256, 1, 1 });
        const completed = c.flint_gpu_command_end(gpu, command, true);
        if (!dispatched or !completed) {
            self.usable = false;
            gpuError(gpu, "resident decode");
            return error.GpuDispatch;
        }
        try self.checkDiagnostics(1);
    }

    pub fn forward(self: *Session, token: u32) ![]const u16 {
        try self.dispatch(token, 1, false, &.{});
        self.position += 1;
        self.last_logits_offset = 0;
        return self.lastLogits();
    }

    pub fn greedy(self: *Session, token: u32, count: u32, eos: []const u32) !Packet {
        try self.dispatch(token, count, true, eos);
        return self.finishPacket(token, count);
    }

    fn finishPacket(self: *Session, token: u32, count: u32) !Packet {
        const record = contents(u32, self.outputs[3], 17);
        var result: Packet = .{ .len = record[16] };
        if (result.len == 0 or result.len > count) {
            self.usable = false;
            return error.InvalidResidentPacket;
        }
        var expected = token;
        for (0..result.len) |i| {
            if (record[2 * i] != expected or record[2 * i + 1] >= vocab) {
                self.usable = false;
                return error.InvalidResidentPacket;
            }
            result.tokens[i] = record[2 * i + 1];
            expected = result.tokens[i];
        }
        self.position += result.len;
        self.last_logits_offset = (result.len - 1) * vocab;
        return result;
    }

    fn encodeSample(self: *Session, command: *anyopaque, config: *const SampleConfig, logits: *anyopaque, offset: u64, fp32: bool) bool {
        return encodeSampling(self.decoder.model.gpu, command, .{
            .parameters = self.parameters,
            .candidates = self.outputs[5],
            .normalizers = self.outputs[6],
            .record = self.outputs[3],
            .diagnostic = self.outputs[2],
        }, config, logits, offset, fp32);
    }

    /// Encode the complete packet once: resident forward -> parallel top-k/
    /// normalization -> sampling/token handoff, repeated without CPU readback.
    pub fn sampled(self: *Session, token: u32, count: u32, options: SampleOptions) !Packet {
        if (options.temperature == 0) return self.greedy(token, count, options.eos);
        if (token >= vocab) return error.InvalidToken;
        if (count == 0 or count > packet_width) return error.InvalidPacket;
        _ = try SampleConfig.init(options, 0, count);
        try self.grow(self.position + count);
        const params = contents(u32, self.parameters, (packet_width + 1) * 4);
        @memset(params, 0);
        params[0..4].* = .{ token, self.position, self.initial_position, layers };
        @memset(contents(u32, self.control, packet_width * 1024), 0);
        @memset(contents(u32, self.outputs[2], packet_width * workers), 0);
        @memset(contents(u32, self.outputs[3], 17), 0);
        self.publishSteering();
        const gpu = self.decoder.model.gpu;
        const command = c.flint_gpu_command_begin(gpu) orelse return error.GpuCommand;
        var dispatched = true;
        for (0..count) |i| {
            const step: u32 = @intCast(i);
            const bound = self.bindings(step);
            const active = self.steeringActive();
            dispatched = c.flint_gpu_dispatch(gpu, command, self.decoder.scalarKernel(active), null, 0, &bound, if (active) 31 else 30, &[_]u32{ self.decoder.profile.workerCount(), 1, 1 }, &[_]u32{ 256, 1, 1 });
            if (!dispatched) break;
            const config = try SampleConfig.init(options, step, count);
            dispatched = self.encodeSample(command, &config, self.outputs[0], @as(u64, step) * vocab * 2, false);
            if (!dispatched) break;
        }
        const completed = c.flint_gpu_command_end(gpu, command, true);
        if (!dispatched or !completed) {
            self.usable = false;
            gpuError(gpu, "filtered resident packet");
            return error.GpuDispatch;
        }
        try self.checkDiagnostics(count);
        return self.finishPacket(token, count);
    }

    /// Sample the FP32 logits produced by prefill without advancing the cache.
    /// Its selected token is the input to the first resident decode packet.
    pub fn sampleLogits(self: *Session, logits: *anyopaque, offset: u64, options: SampleOptions) !u32 {
        const config = try SampleConfig.init(options, 0, 1);
        if (!self.usable) return error.InvalidatedSession;
        const params = contents(u32, self.parameters, 8);
        @memset(params, 0);
        params[0..4].* = .{ 0, self.position -| 1, self.initial_position, layers };
        @memset(contents(u32, self.outputs[2], workers), 0);
        @memset(contents(u32, self.outputs[3], 17), 0);
        const gpu = self.decoder.model.gpu;
        const command = c.flint_gpu_command_begin(gpu) orelse return error.GpuCommand;
        const dispatched = self.encodeSample(command, &config, logits, offset, true);
        const completed = c.flint_gpu_command_end(gpu, command, true);
        if (!dispatched or !completed) {
            gpuError(gpu, "prefill sampling");
            return error.GpuDispatch;
        }
        const record = contents(u32, self.outputs[3], 17);
        if (record[16] != 1 or record[1] >= vocab) return error.InvalidResidentPacket;
        return record[1];
    }
};
