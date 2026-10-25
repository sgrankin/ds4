//! Native vision-tower oracle: prepared HF patches in, merged FP32 features out.
const std = @import("std");
const c = @import("clef_runtime_c");
const Backend = @import("models/clef/backend.zig");
const a = std.heap.c_allocator;

fn read(path: [*:0]const u8) ![]f32 {
    const file = c.fopen(path, "rb") orelse return error.OpenPatches;
    defer _ = c.fclose(file);
    if (c.fseek(file, 0, c.SEEK_END) != 0) return error.ReadPatches;
    const bytes = c.ftell(file);
    if (bytes < 0 or @rem(bytes, 4) != 0 or c.fseek(file, 0, c.SEEK_SET) != 0) return error.InvalidPatches;
    const values = try a.alloc(f32, @intCast(@divTrunc(bytes, 4)));
    errdefer a.free(values);
    if (c.fread(values.ptr, 4, values.len, file) != values.len) return error.ReadPatches;
    return values;
}
fn write(path: [*:0]const u8, values: []const f32) !void {
    const file = c.fopen(path, "wb") orelse return error.OpenOutput;
    defer _ = c.fclose(file);
    if (c.fwrite(values.ptr, 4, values.len, file) != values.len or c.fflush(file) != 0) return error.WriteOutput;
}
fn now() f64 {
    var stamp: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &stamp);
    return @as(f64, @floatFromInt(stamp.tv_sec)) + @as(f64, @floatFromInt(stamp.tv_nsec)) / 1e9;
}
fn run(argv: [*c][*c]u8) !void {
    const engine = try Backend.Engine.open(std.mem.span(argv[1]));
    defer engine.close();
    const patches = try read(@ptrCast(argv[2]));
    defer a.free(patches);
    var grid: [3]u32 = undefined;
    for (&grid, 3..) |*n, i| n.* = try std.fmt.parseInt(u32, std.mem.span(argv[i]), 10);
    const features = try engine.visionFeatures(patches, grid, null, null);
    for (features.features) |v| if (!std.math.isFinite(v)) return error.NonFiniteVision;
    try write(@ptrCast(argv[6]), features.features);
    for (features.deepstack, 0..) |rows, i| {
        for (rows) |v| if (!std.math.isFinite(v)) return error.NonFiniteVision;
        const name = try std.fmt.allocPrintSentinel(a, "{s}.deepstack-{d}.f32", .{ std.mem.span(argv[6]), i }, 0);
        defer a.free(name);
        try write(name, rows);
    }
    var milliseconds: [4]f64 = undefined;
    for (&milliseconds) |*elapsed| {
        const start = now();
        _ = try engine.visionFeatures(patches, grid, null, null);
        elapsed.* = (now() - start) * 1000;
    }
    std.debug.print("Clef vision warm milliseconds: {any}\n", .{milliseconds});
    std.debug.print("Clef vision: grid {any}, {d} merged rows × {d}, {d} DeepStack outputs, all finite.\n", .{ grid, features.features.len / engine.config.H, engine.config.H, features.deepstack.len });
}
export fn main(argc: c_int, argv: [*c][*c]u8) c_int {
    if (argc != 7) {
        std.debug.print("Usage: clef-vision-smoke MODEL_DIR PATCHES.f32 T GRID_H GRID_W OUTPUT.f32\n", .{});
        return 2;
    }
    run(argv) catch |err| {
        std.debug.print("Clef vision: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}
