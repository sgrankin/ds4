//! Numerical-oracle bridge: JSON token IDs in, row-major FP32 hidden states out.
const std = @import("std");
const c = @import("clef_c");
const Backend = @import("models/clef/backend.zig");
const a = std.heap.c_allocator;

fn run(model: []const u8, input_path: [*c]const u8, output_path: [*c]const u8) !void {
    const input = c.fopen(input_path, "rb") orelse return error.OpenInput;
    defer _ = c.fclose(input);
    if (c.fseek(input, 0, c.SEEK_END) != 0) return error.ReadInput;
    const size = c.ftell(input);
    if (size < 0 or size > 1024 * 1024) return error.InvalidInputLength;
    c.rewind(input);
    const bytes = try a.alloc(u8, @intCast(size));
    defer a.free(bytes);
    if (c.fread(bytes.ptr, 1, bytes.len, input) != bytes.len) return error.ReadInput;
    const ids = try std.json.parseFromSlice([]u32, a, bytes, .{});
    defer ids.deinit();
    const engine = try Backend.Engine.open(model);
    defer engine.close();
    const hidden = try engine.infer(ids.value);
    const output = c.fopen(output_path, "wb") orelse return error.OpenOutput;
    defer _ = c.fclose(output);
    if (c.fwrite(hidden.ptr, @sizeOf(f32), hidden.len, output) != hidden.len) return error.WriteOutput;
    if (c.fflush(output) != 0) return error.WriteOutput;
    std.debug.print("clef: wrote {d} final hidden values for {d} input tokens\n", .{ hidden.len, ids.value.len });
}

export fn main(argc: c_int, argv: [*c][*c]u8) c_int {
    if (argc != 4) {
        std.debug.print("Usage: clef-backend-smoke MODEL_DIR IDS.json HIDDEN.f32\n", .{});
        return 2;
    }
    run(std.mem.span(argv[1]), argv[2], argv[3]) catch |err| {
        std.debug.print("clef-backend-smoke: {s}\n", .{@errorName(err)});
        return 1;
    };
    return 0;
}
