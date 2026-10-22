//! Slash-command admission and save questions. Keep the existing libc numeric
//! grammar and distinguish leaving a session from exiting the process.
const std = @import("std");
const c = @import("agent_runtime_c");
const Ui = @import("ui_text.zig");
const Sessions = @import("worker_sessions.zig").Sessions;

pub fn now() f64 {
    var stamp: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &stamp);
    return @as(f64, @floatFromInt(stamp.tv_sec)) + @as(f64, @floatFromInt(stamp.tv_nsec)) * 1.0e-9;
}
pub fn power(value: [*c]const u8, output: *c_int) bool {
    var end: [*c]u8 = null;
    const parsed = c.strtol(value, &end, 10);
    if (value[0] == 0 or end[0] != 0 or parsed < 1 or parsed > 100) return false;
    output.* = @intCast(parsed);
    return true;
}
pub fn steering(value: [*c]const u8, output: *f32) bool {
    var end: [*c]u8 = null;
    c.__error().* = 0;
    const parsed = c.strtof(value, &end);
    if (value[0] == 0 or end[0] != 0 or c.__error().* == c.ERANGE or !std.math.isFinite(parsed) or parsed < -100 or parsed > 100) return false;
    output.* = parsed;
    return true;
}
pub fn hints(value: [*c]const u8, output: *bool) bool {
    if (std.mem.eql(u8, Ui.text(value), "on")) output.* = true else if (std.mem.eql(u8, Ui.text(value), "off")) output.* = false else return false;
    return true;
}
pub fn withArguments(command: []const u8, name: []const u8) bool {
    return std.mem.startsWith(u8, command, name) and (command.len == name.len or c.isspace(command[name.len]) != 0);
}
pub fn known(command: []const u8) bool {
    for ([_][]const u8{ "/help", "/save", "/compact", "/list", "/quit", "/exit", "/new" }) |name| {
        if (std.mem.eql(u8, command, name)) return true;
    }
    for ([_][]const u8{ "/power", "/think", "/steer", "/hints", "/switch", "/del", "/strip", "/history" }) |name| {
        if (withArguments(command, name)) return true;
    }
    return false;
}
pub fn integerDefault(value: [*c]const u8, fallback: c_int, lower: c_int, upper: c_int) c_int {
    if (value == null or value[0] == 0) return fallback;
    var end: [*c]u8 = null;
    const parsed = c.strtol(value, &end, 10);
    if (end == value) return fallback;
    while (end[0] == ' ' or end[0] == '\t' or end[0] == '\r' or end[0] == '\n') end += 1;
    if (end[0] != 0) return fallback;
    return @intCast(@min(@max(parsed, lower), upper));
}
pub fn help() void {
    for ([_][*:0]const u8{
        "Commands:",
        "  /help        Show this help.",
        "  /save        Save the current session.",
        "  /compact     Compact the current session context now.",
        "  /list        List saved sessions.",
        "  /switch SHA  Load a saved session and show recent history.",
        "  /del SHA     Delete a saved session.",
        "  /strip SHA   Strip KV payload; /switch rebuilds it by prefill.",
        "  /history [N] Show N recent user turns from the current session.",
        "  /power N     Set GPU duty cycle percentage, 1..100.",
        "  /hints on|off Enable or disable brief programming hints; starts off.",
        "  /steer [F]   Show or set FFN steering for subsequent tokens.",
        "  /new         Start a fresh session from the system prompt.",
        "  /quit, /exit Exit.",
        "  Ctrl+C       Interrupt generation; clear edited text.",
        "  Enter        Queue text while the agent is busy.",
        "  Ctrl+X       Edit the first queued prompt.",
        "  ESC          Interrupt and send queued prompt immediately.",
        "  Ctrl+D       Exit from an empty prompt.",
    }) |line| _ = c.puts(line);
}

const Automatic = enum { none, no, yes };
pub const Question = struct {
    timeout_seconds: c_int = 0,
    automatic: Automatic = .none,
};
/// Ordinary questions block indefinitely. Browser approval has a deadline so
/// an unattended browser start cannot stall the Agent's worker indefinitely.
pub fn yesNo(prompt: [*c]const u8, options: Question, timed_out: ?*bool) bool {
    const timeout = options.timeout_seconds > 0 and options.automatic != .none;
    const deadline = if (timeout) now() + @as(f64, @floatFromInt(options.timeout_seconds)) else 0;
    if (timed_out) |output| output.* = false;
    while (true) {
        _ = c.printf("%s", prompt);
        if (timeout) {
            const remaining: c_int = @max(0, @as(c_int, @intFromFloat(deadline - now() + 0.999)));
            _ = c.printf("[auto-%s in %ds] ", @as([*:0]const u8, if (options.automatic == .yes) "yes" else "no"), remaining);
        }
        _ = c.fflush(c.__stdoutp);
        if (timeout) {
            const remaining = deadline - now();
            if (remaining <= 0) {
                if (timed_out) |output| output.* = true;
                _ = c.printf("\n");
                return options.automatic == .yes;
            }
            var descriptor = c.struct_pollfd{ .fd = c.STDIN_FILENO, .events = c.POLLIN, .revents = 0 };
            const timeout_ms: c_int = @as(c_int, @intFromFloat(remaining * 1000)) + 1;
            var result: c_int = undefined;
            while (true) {
                result = c.poll(&descriptor, 1, timeout_ms);
                if (result >= 0 or c.__error().* != c.EINTR) break;
            }
            if (result == 0) {
                if (timed_out) |output| output.* = true;
                _ = c.printf("\n");
                return options.automatic == .yes;
            }
            if (result < 0) return false;
        }
        // editor_start can leave stdin nonblocking. fgets must temporarily
        // wait for one complete answer, then restore the exact old flags.
        const saved = c.fcntl(c.STDIN_FILENO, c.F_GETFL, @as(c_int, 0));
        const nonblocking = saved >= 0 and saved & c.O_NONBLOCK != 0;
        if (nonblocking) _ = c.fcntl(c.STDIN_FILENO, c.F_SETFL, saved & ~@as(c_int, c.O_NONBLOCK));
        var buffer: [32]u8 = undefined;
        const got_line = c.fgets(&buffer, buffer.len, c.__stdinp) != null;
        if (nonblocking) _ = c.fcntl(c.STDIN_FILENO, c.F_SETFL, saved);
        if (!got_line) return false;
        const answer = std.mem.trimStart(u8, Ui.text(&buffer), " \t");
        if (answer.len != 0) switch (answer[0]) {
            'y', 'Y' => return true,
            'n', 'N' => return false,
            else => {},
        };
    }
}
/// Fresh system-only sessions do not need a save question. Declining a save
/// before /new or /switch means continuing with the requested session change.
pub fn beforeLeaving(worker: *c.agent_worker) bool {
    const sessions = Sessions{ .worker = worker };
    if (!sessions.needsSave()) return true;
    if (!yesNo("Save current session? (y/n) ", .{}, null)) return true;
    var errors: [160]u8 = @splat(0);
    if (sessions.save(&errors)) return true;
    _ = c.printf("save failed: %s\n", &errors);
    return yesNo("Continue anyway? (y/n) ", .{}, null);
}
pub const Exit = enum { cancel, clean, now };
/// Once the terminal is restored, declining an exit save can let the OS retire
/// model/Metal resources immediately rather than waiting for orderly teardown.
pub fn beforeExiting(worker: *c.agent_worker) Exit {
    const sessions = Sessions{ .worker = worker };
    if (!sessions.needsSave()) return .clean;
    if (!yesNo("Save current session? (y/n) ", .{}, null)) return .now;
    var errors: [160]u8 = @splat(0);
    if (sessions.save(&errors)) return .clean;
    _ = c.printf("save failed: %s\n", &errors);
    return if (yesNo("Continue anyway? (y/n) ", .{}, null)) .now else .cancel;
}
