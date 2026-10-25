//! Native build/driver smoke. Model inference qualifications extend this same
//! executable as the execution pieces are connected.
const std = @import("std");
const c = @import("flint_c");
const weights = @import("weights.zig");
const shaders = @import("prefill_shaders.zig");
const decode = @import("decode.zig");
const sampling_test = @import("sampling_test.zig");
const kv_test = @import("kv_test.zig");
const profiles = @import("profile.zig");
const profile_test = @import("profile_test.zig");
const prefill_test = @import("prefill_test.zig");
const steering_test = @import("steering_test.zig");
const a = std.heap.c_allocator;

export fn main(argc: c_int, argv: [*][*:0]u8) c_int {
    run(argc, argv) catch |err| {
        std.debug.print("flint smoke: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}

fn run(argc: c_int, argv: [*][*:0]u8) !void {
    try profile_test.run();
    if (argc > 1 and std.mem.eql(u8, std.mem.span(argv[1]), "--flint-profile")) {
        if (argc < 4) return error.MissingProfileArgument;
        return runSelected(argc - 2, argv + 2, std.mem.span(argv[2]));
    }
    return runSelected(argc, argv, null);
}

fn runSelected(argc: c_int, argv: [*][*:0]u8, requested_profile: ?[:0]const u8) !void {
    if (argc > 2 and !std.mem.startsWith(u8, std.mem.span(argv[1]), "--")) {
        if (argc > 4 and std.mem.eql(u8, std.mem.span(argv[4]), "quality"))
            return quality(std.mem.span(argv[1]), std.mem.span(argv[2]), std.mem.span(argv[3]), requested_profile);
        return benchmark(std.mem.span(argv[1]), std.mem.span(argv[2]), if (argc > 3) std.mem.span(argv[3]) else ".workspaces/flint-native", if (argc > 4 and std.mem.eql(u8, std.mem.span(argv[4]), "default")) 1 else 0, argc > 4 and std.mem.eql(u8, std.mem.span(argv[4]), "scalar"), requested_profile);
    }
    const gpu = c.flint_gpu_create() orelse return error.MetalUnavailable;
    defer c.flint_gpu_free(gpu);
    const device = c.flint_gpu_device_name(gpu);
    if (device == null) return error.MetalUnavailable;
    const profile = try profiles.select(std.mem.span(device), requested_profile);
    const decode_only = argc > 1 and std.mem.eql(u8, std.mem.span(argv[1]), "--decode-only");
    if (!decode_only) {
        const source = try shaders.sourceFor(a, if (profile.usesNax()) .nax else .simd);
        defer a.free(source);
        var error_text: [4096]u8 = @splat(0);
        if (!c.flint_gpu_compile(gpu, source.ptr, source.len, &error_text, error_text.len)) {
            std.debug.print("{s}\n", .{std.mem.sliceTo(&error_text, 0)});
            return error.CompilePrefill;
        }
        std.debug.print("flint: native prefill shader compiled on {s}\n", .{c.flint_gpu_device_name(gpu)});
        try kv_test.run(gpu);
        if (profile.usesNax()) try prefill_test.longGather(gpu) else try prefill_test.run(gpu);
    }
    try decode.compile(gpu, profile);
    std.debug.print("flint: native decode profile {s} compiled\n", .{profile.name()});
    try sampling_test.run(gpu);
    if (!decode_only) {
        try decode.compileSteering(gpu, profile);
        try decode.compileAdditiveSteering(gpu, profile);
        try steering_test.run(gpu);
    }
    if (argc > 1 and !std.mem.startsWith(u8, std.mem.span(argv[1]), "--")) {
        var model = try weights.load(std.mem.span(argv[1]));
        defer model.deinit();
        std.debug.print("flint: native model banks ready\n", .{});
    }
}

fn seconds() f64 {
    var time: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &time);
    return @as(f64, @floatFromInt(time.tv_sec)) + @as(f64, @floatFromInt(time.tv_nsec)) * 1e-9;
}

fn writeFile(path: []const u8, bytes: []const u8) !void {
    const name = try a.dupeSentinel(u8, path, 0);
    defer a.free(name);
    const file = c.fopen(name, "wb") orelse return error.OpenOutput;
    defer _ = c.fclose(file);
    if (c.fwrite(bytes.ptr, 1, bytes.len, file) != bytes.len) return error.WriteOutput;
}

fn argmax(values: []const f32) c_int {
    var best: usize = 0;
    for (values, 0..) |value, i| if (value > values[best]) {
        best = i;
    };
    return @intCast(best);
}

fn quality(directory: []const u8, input_path: []const u8, output_prefix: []const u8, profile: ?[:0]const u8) !void {
    const name = try a.dupeSentinel(u8, directory, 0);
    defer a.free(name);
    var error_text: [4096]u8 = @splat(0);
    const engine = c.flint_engine_open(name, 4096, if (profile) |value| value.ptr else null, &error_text, error_text.len) orelse return error.LoadModel;
    defer c.flint_engine_close(engine);
    const bytes = try weights.readFile(a, input_path);
    defer a.free(bytes);
    const json = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer json.deinit();
    const session = c.flint_session_create(engine, 8192) orelse return error.CreateSession;
    defer c.flint_session_free(session);
    for (json.value.array.items) |item| {
        const values = item.object.get("token_ids").?.array.items;
        const ids = try a.alloc(c_int, values.len);
        defer a.free(ids);
        for (ids, values) |*id, value| id.* = @intCast(value.integer);
        const prompt = c.ds4_tokens{ .v = ids.ptr, .len = @intCast(ids.len), .cap = @intCast(ids.len) };
        c.flint_session_reset(session);
        if (c.flint_session_sync(session, &prompt, null, null, null, null, &error_text, error_text.len) != 0) {
            std.debug.print("{s}\n", .{std.mem.sliceTo(&error_text, 0)});
            return error.Prefill;
        }
        const path = try std.fmt.allocPrint(a, "{s}-{s}.f32", .{ output_prefix, item.object.get("id").?.string });
        defer a.free(path);
        try writeFile(path, std.mem.sliceAsBytes(c.flint_session_logits(session)[0..150144]));
        std.debug.print("flint: quality {s}, {d} tokens\n", .{ item.object.get("id").?.string, ids.len });
    }
}

fn benchmark(directory: []const u8, input_path: []const u8, output_prefix: []const u8, temperature: f32, scalar: bool, profile: ?[:0]const u8) !void {
    const terminated = try a.dupeSentinel(u8, directory, 0);
    defer a.free(terminated);
    var error_text: [4096]u8 = @splat(0);
    const engine = c.flint_engine_open(terminated, 4096, if (profile) |value| value.ptr else null, &error_text, error_text.len) orelse {
        std.debug.print("{s}\n", .{std.mem.sliceTo(&error_text, 0)});
        return error.LoadModel;
    };
    defer c.flint_engine_close(engine);
    const bytes = try weights.readFile(a, input_path);
    defer a.free(bytes);
    const json = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer json.deinit();
    const values = json.value.object.get("token_ids").?.array.items;
    const ids = try a.alloc(c_int, values.len);
    defer a.free(ids);
    for (ids, values) |*id, value| id.* = @intCast(value.integer);
    const prompt = c.ds4_tokens{ .v = ids.ptr, .len = @intCast(ids.len), .cap = @intCast(ids.len) };
    const session = c.flint_session_create(engine, 262144) orelse return error.CreateSession;
    defer c.flint_session_free(session);
    const logits = c.flint_session_logits(session)[0..150144];
    for (0..3) |repeat| {
        c.flint_session_reset(session);
        const start = seconds();
        if (c.flint_session_sync(session, &prompt, null, null, null, null, &error_text, error_text.len) != 0) {
            std.debug.print("{s}\n", .{std.mem.sliceTo(&error_text, 0)});
            return error.Prefill;
        }
        const prefill_end = seconds();
        if (repeat == 0) {
            const path = try std.fmt.allocPrint(a, "{s}-logits.f32", .{output_prefix});
            defer a.free(path);
            try writeFile(path, std.mem.sliceAsBytes(logits));
        }
        var generated: [128]c_int = undefined;
        var count: usize = 0;
        var rng: u64 = 0;
        const first = seconds();
        while (count < generated.len - 1) {
            const token = if (temperature == 0) argmax(logits) else c.ds4_sample_logits(logits.ptr, 150144, temperature, 20, 0.95, 0, &rng);
            if (c.flint_tokenizer_is_stop(c.flint_engine_tokenizer(engine), token)) break;
            if (scalar) {
                if (c.flint_session_eval(session, token, &error_text, error_text.len) != 0) {
                    std.debug.print("{s}\n", .{std.mem.sliceTo(&error_text, 0)});
                    return error.Decode;
                }
                generated[count] = token;
                count += 1;
                continue;
            }
            var accepted: [8]c_int = undefined;
            const limit: u32 = @intCast(@min(accepted.len, generated.len - 1 - count));
            const n = c.flint_session_packet(session, token, limit, temperature, 20, 0.95, 0, &rng, &accepted, &error_text, error_text.len);
            if (n <= 0) {
                std.debug.print("{s}\n", .{std.mem.sliceTo(&error_text, 0)});
                return error.Decode;
            }
            @memcpy(generated[count..][0..@intCast(n)], accepted[0..@intCast(n)]);
            count += @intCast(n);
        }
        generated[count] = if (temperature == 0) argmax(logits) else c.ds4_sample_logits(logits.ptr, 150144, temperature, 20, 0.95, 0, &rng);
        count += 1;
        const end = seconds();
        const record = .{ .repeat = repeat, .temperature = temperature, .scalar = scalar, .prompt_tokens = ids.len, .prefill_seconds = prefill_end - start, .decode_seconds = end - first, .decode_tokens_per_second = @as(f64, @floatFromInt(count - 1)) / (end - first), .request_seconds = end - start, .token_ids = generated[0..count] };
        const result = try std.json.Stringify.valueAlloc(a, record, .{});
        defer a.free(result);
        const path = try std.fmt.allocPrint(a, "{s}-{d}.json", .{ output_prefix, repeat });
        defer a.free(path);
        try writeFile(path, result);
        std.debug.print("{s}\n", .{result});
    }
}
