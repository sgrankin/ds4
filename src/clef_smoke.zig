//! Repeated real decision calls without a network: one JSON request per line.
const std = @import("std");
const c = @import("clef_runtime_c");

fn now() f64 {
    var stamp: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &stamp);
    return @as(f64, @floatFromInt(stamp.tv_sec)) + @as(f64, @floatFromInt(stamp.tv_nsec)) / 1e9;
}

export fn main(argc: c_int, argv: [*c][*c]u8) c_int {
    if (argc < 3 or argc > 4) {
        std.debug.print("Usage: clef-smoke MODEL_DIR REQUESTS.jsonl [--openai]\n", .{});
        return 2;
    }
    const openai = argc == 4 and std.mem.eql(u8, std.mem.span(argv[3]), "--openai");
    if (argc == 4 and !openai) return 2;
    var message: [512]u8 = @splat(0);
    const engine = c.ds4_decision_engine_open(argv[1], 16384, &message, message.len) orelse {
        std.debug.print("clef-smoke: {s}\n", .{std.mem.sliceTo(&message, 0)});
        return 1;
    };
    defer c.ds4_decision_engine_close(engine);
    const file = c.fopen(argv[2], "rb") orelse return 1;
    defer _ = c.fclose(file);
    var line: [*c]u8 = null;
    var capacity: usize = 0;
    defer c.free(line);
    var count: usize = 0;
    while (true) {
        const length = c.getline(&line, &capacity, file);
        if (length < 0) break;
        if (std.mem.trim(u8, line[0..@intCast(length)], " \r\n\t").len == 0) continue;
        var output: [*c]u8 = null;
        const start = now();
        const status = c.ds4_decision_engine_request(engine, line, openai, &output, &message, message.len, null, null);
        defer c.free(output);
        if (status != 200) {
            std.debug.print("clef-smoke: request {d}, status {d}: {s}\n", .{ count, status, std.mem.sliceTo(&message, 0) });
            return 1;
        }
        _ = c.fputs(output, c.__stdoutp);
        _ = c.fputc('\n', c.__stdoutp);
        _ = c.fflush(c.__stdoutp);
        std.debug.print("clef-smoke: request {d}, {d:.3} ms\n", .{ count, (now() - start) * 1000 });
        count += 1;
    }
    if (c.ferror(file) != 0 or count == 0) return 1;
    return 0;
}
