//! Existing trace wire text over the actual Worker FILE and mutex.
const std = @import("std");
const c = @import("agent_runtime_c");

fn timestamp(stream: [*c]c.FILE) void {
    var stamp: c.struct_timespec = undefined;
    var calendar: c.struct_tm = undefined;
    _ = c.clock_gettime(c.CLOCK_REALTIME, &stamp);
    _ = c.localtime_r(&stamp.tv_sec, &calendar);
    _ = c.fprintf(stream, "%04d-%02d-%02d %02d:%02d:%02d.%03ld", calendar.tm_year + 1900, calendar.tm_mon + 1, calendar.tm_mday, calendar.tm_hour, calendar.tm_min, calendar.tm_sec, @as(c_long, @divTrunc(stamp.tv_nsec, 1000000)));
}
fn escaped(stream: [*c]c.FILE, value: []const u8) void {
    for (value) |byte| switch (byte) {
        '\\' => {
            _ = c.fputs("\\\\", stream);
        },
        '\n' => {
            _ = c.fputs("\\n", stream);
        },
        '\r' => {
            _ = c.fputs("\\r", stream);
        },
        '\t' => {
            _ = c.fputs("\\t", stream);
        },
        '"' => {
            _ = c.fputs("\\\"", stream);
        },
        else => {
            if (byte < 32 or byte == 127) _ = c.fprintf(stream, "\\x%02x", @as(c_uint, byte)) else _ = c.fputc(byte, stream);
        },
    };
}
pub const Trace = struct {
    worker: *c.agent_worker,
    pub fn log(self: Trace, format: [*:0]const u8, arguments: anytype) void {
        const worker = self.worker;
        if (worker.trace == null) return;
        _ = c.pthread_mutex_lock(&worker.mu);
        defer _ = c.pthread_mutex_unlock(&worker.mu);
        timestamp(worker.trace);
        _ = c.fputc(' ', worker.trace);
        _ = @call(.auto, c.fprintf, .{ worker.trace, format } ++ arguments);
        _ = c.fputc('\n', worker.trace);
        _ = c.fflush(worker.trace);
    }
    pub fn text(self: Trace, label: [*c]const u8, value: []const u8) void {
        const worker = self.worker;
        if (worker.trace == null) return;
        _ = c.pthread_mutex_lock(&worker.mu);
        defer _ = c.pthread_mutex_unlock(&worker.mu);
        timestamp(worker.trace);
        _ = c.fprintf(worker.trace, " %s=\"", @as([*c]const u8, if (label != null) label else "text"));
        escaped(worker.trace, value);
        _ = c.fputs("\"\n", worker.trace);
        _ = c.fflush(worker.trace);
    }
    pub fn token(self: Trace, id: c_int, value: []const u8, index: c_int) void {
        const worker = self.worker;
        if (worker.trace == null) return;
        _ = c.pthread_mutex_lock(&worker.mu);
        defer _ = c.pthread_mutex_unlock(&worker.mu);
        timestamp(worker.trace);
        _ = c.fprintf(worker.trace, " token index=%d id=%d bytes=%zu text=\"", index, id, value.len);
        escaped(worker.trace, value);
        _ = c.fputs("\" hex=", worker.trace);
        for (value) |byte| _ = c.fprintf(worker.trace, "%02x", @as(c_uint, byte));
        _ = c.fputc('\n', worker.trace);
        _ = c.fflush(worker.trace);
    }
    pub fn tokens(self: Trace, label: [*c]const u8, value: *const c.ds4_tokens, start: c_int) void {
        if (self.worker.trace == null) return;
        const begin: c_int = @max(0, @min(start, value.len));
        self.log("tokens label=%s start=%d len=%d", .{ if (label != null) label else @as([*c]const u8, ""), begin, value.len });
        var index = begin;
        while (index < value.len) : (index += 1) {
            var length: usize = 0;
            const decoded = c.ds4_token_text(self.worker.engine, value.v[@intCast(index)], &length);
            defer c.free(decoded);
            self.token(value.v[@intCast(index)], if (length == 0) &.{} else decoded[0..length], index);
        }
    }
};
