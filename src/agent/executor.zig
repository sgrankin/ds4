//! Complete one-call tool dispatch and its job-list/Web observation glue.
//! Context import, fit/compact/commit and the surrounding Worker turn stay out.
const std = @import("std");
const c = @import("agent_tool_actions_c");
const allocator = std.heap.c_allocator;
pub const Runtime = c.ds4_agent_tool_runtime;
pub const Job = c.agent_bash_job;
const Call = c.agent_tool_call;

fn fatal(label: [*:0]const u8) noreturn {
    c.perror(label);
    c.exit(1);
}
fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn same(pointer: [*c]const u8, value: []const u8) bool {
    return std.mem.eql(u8, text(pointer), value);
}
fn copy(value: []const u8) [*c]u8 {
    return (allocator.dupeSentinel(u8, value, 0) catch fatal("ds4-agent: malloc")).ptr;
}
fn argument(call: *const Call, name: [*:0]const u8) [*c]const u8 {
    return c.ds4_agent_tool_arg_value(call, name);
}
const Buffer = struct {
    list: std.ArrayList(u8) = .empty,
    fn deinit(self: *Buffer) void {
        self.list.deinit(allocator);
    }
    fn append(self: *Buffer, value: []const u8) void {
        self.list.appendSlice(allocator, value) catch fatal("ds4-agent: realloc");
    }
    fn take(self: *Buffer) [*c]u8 {
        return (self.list.toOwnedSliceSentinel(allocator, 0) catch fatal("ds4-agent: realloc")).ptr;
    }
};
fn integer(pointer: [*c]const u8, default: c_int, minimum: c_int, maximum: c_int) c_int {
    if (pointer == null or pointer[0] == 0) return default;
    var end: [*c]u8 = null;
    const value = c.strtol(pointer, &end, 10);
    if (end == pointer) return default;
    while (end[0] != 0 and std.mem.findScalar(u8, " \t\r\n", end[0]) != null) end += 1;
    if (end[0] != 0) return default;
    return @intCast(@max(minimum, @min(maximum, value)));
}
fn timeout(pointer: [*c]const u8) c_int {
    if (pointer == null or pointer[0] == 0) return 3600;
    var end: [*c]u8 = null;
    const value = c.strtod(pointer, &end);
    if (end == pointer or value <= 0 or !std.math.isFinite(value)) return 3600;
    return @intFromFloat(@max(1.0, @min(86400.0, value)));
}
fn now() f64 {
    var clock: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &clock);
    return @as(f64, @floatFromInt(clock.tv_sec)) + @as(f64, @floatFromInt(clock.tv_nsec)) * 1e-9;
}
fn snapshot(job: *Job) c.ds4_agent_bash_snapshot {
    var value: c.ds4_agent_bash_snapshot = undefined;
    c.ds4_agent_bash_get_snapshot(job.process, false, &value);
    return value;
}
fn freeJob(job: *Job) void {
    c.ds4_agent_bash_free(job.process);
    allocator.destroy(job);
}
pub fn freeJobs(runtime: *Runtime) void {
    var current = runtime.jobs;
    while (current != null) {
        const job: *Job = @ptrCast(current);
        current = job.next;
        freeJob(job);
    }
    runtime.jobs = null;
}
pub fn findJob(runtime: *Runtime, id: c_int, pid: c.pid_t) ?*Job {
    var current = runtime.jobs;
    while (current != null) {
        const job: *Job = @ptrCast(current);
        const value = snapshot(job);
        if ((id > 0 and value.id == id) or (id <= 0 and pid > 0 and value.pid == pid)) return job;
        current = job.next;
    }
    return null;
}
pub fn removeJob(runtime: *Runtime, target: *Job) void {
    var link = &runtime.jobs;
    while (link.* != null) {
        const job: *Job = @ptrCast(link.*);
        if (job == target) {
            link.* = job.next;
            job.next = null;
            freeJob(job);
            return;
        }
        link = &job.next;
    }
}
pub fn startJob(runtime: *Runtime, command: [*c]const u8, seconds: c_int, terminal_changed: c.ds4_agent_bash_terminal_fn, context: ?*anyopaque, errors: []u8) ?*Job {
    const process = c.ds4_agent_bash_start(command, seconds, &runtime.next_job_id, terminal_changed, context, errors.ptr, errors.len) orelse return null;
    const job = allocator.create(Job) catch fatal("ds4-agent: malloc");
    job.* = .{ .process = process, .next = runtime.jobs };
    runtime.jobs = job;
    return job;
}

/// Preserve safe SGR styling but remove terminal-control commands from child
/// output. Invalid UTF8/C1 bytes are escaped individually, as in the old sink.
pub fn terminalSafe(bytes: []const u8) [*c]u8 {
    var output: Buffer = .{};
    defer output.deinit();
    var position: usize = 0;
    while (position < bytes.len) {
        const start = position;
        const byte = bytes[position];
        position += 1;
        if (byte >= 0x80) {
            var codepoint: u32 = undefined;
            const length = c.linenoiseUtf8Decode(bytes.ptr + start, bytes.len - start, &codepoint);
            if (length > 1 and !(codepoint >= 0x80 and codepoint <= 0x9f)) {
                output.append(bytes[start..][0..length]);
                position += length - 1;
                continue;
            }
        } else if (byte == 0x1b) {
            if (position == bytes.len) break;
            var kind = bytes[position];
            position += 1;
            if (kind == '[') {
                var sgr = true;
                while (position < bytes.len and bytes[position] < 0x40) : (position += 1) {
                    if (c.isdigit(bytes[position]) == 0 and bytes[position] != ';' and bytes[position] != ':') sgr = false;
                }
                if (position < bytes.len) {
                    if (bytes[position] == 'm' and sgr and position - start < 96) output.append(bytes[start .. position + 1]);
                    position += 1;
                }
            } else if (std.mem.findScalar(u8, "]P_^X", kind) != null) {
                while (position < bytes.len) {
                    const character = bytes[position];
                    position += 1;
                    if (character == 7 and kind == ']') break;
                    if (position >= 2 and bytes[position - 2] == 0x1b and bytes[position - 1] == '\\') break;
                }
            } else {
                while (kind >= 0x20 and kind <= 0x2f and position < bytes.len) {
                    kind = bytes[position];
                    position += 1;
                }
            }
            continue;
        } else if (!((byte < 32 and byte != '\n' and byte != '\r' and byte != '\t') or byte == 127)) {
            output.append(&.{byte});
            continue;
        }
        var escaped: [5]u8 = undefined;
        _ = c.snprintf(&escaped, escaped.len, "\\x%02x", @as(c_uint, byte));
        output.append(text(&escaped));
    }
    return output.take();
}

pub const Executor = struct {
    runtime: *Runtime,
    context_size: c_int,
    allow_upto: bool,
    non_interactive: bool,
    use_color: bool,
    context: ?*anyopaque,
    emit: c.ds4_agent_presentation_emit_fn,
    cancelled: c.ds4_session_cancel_fn,
    terminal_changed: c.ds4_agent_bash_terminal_fn,

    fn write(self: Executor, value: []const u8) void {
        if (self.emit) |emit| emit(self.context, value.ptr, value.len);
    }
    fn interrupted(self: Executor) bool {
        return if (self.cancelled) |cancel| cancel(self.context) else false;
    }
    fn status(self: Executor, prefix: []const u8, value: []const u8) void {
        if (self.non_interactive) return;
        self.write(if (self.use_color) "\x1b[33m✦ \x1b[38;5;218m" else "✦ ");
        self.write(prefix);
        self.write(value);
        self.write("...");
        self.write(if (self.use_color) "\x1b[0m\n" else "\n");
    }
    fn failure(tool: []const u8, detail: []const u8) [*c]u8 {
        var out: Buffer = .{};
        defer out.deinit();
        out.append("Tool error: ");
        out.append(tool);
        out.append(" failed: ");
        out.append(detail);
        out.append("\n");
        return out.take();
    }
    fn publishObservation(self: Executor, observation: [*c]const u8) void {
        const value = text(observation);
        if (value.len == 0) return;
        var body: ?[]const u8 = null;
        var close: []const u8 = "</head>";
        var label = std.mem.find(u8, value, "\n<head ");
        if (label == null) {
            label = std.mem.find(u8, value, "\n<tail ");
            close = "</tail>";
        }
        if (label) |offset| {
            if (std.mem.findPos(u8, value, offset, ">\n")) |end| {
                self.write("\x1b[90m");
                self.write(if (std.mem.startsWith(u8, value[offset..], "\n<head ")) "[showing first output lines]\n" else "[showing last output lines]\n");
                self.write("\x1b[0m");
                body = value[end + 2 ..];
            }
        } else if (std.mem.find(u8, value, "\n<output>\n")) |offset| {
            body = value[offset + "\n<output>\n".len ..];
            close = "</output>";
        }
        const rest = body orelse return;
        const visible = rest[0..(std.mem.find(u8, rest, close) orelse rest.len)];
        if (visible.len == 0) return;
        const failed = std.mem.find(u8, value, "status=done") != null and std.mem.find(u8, value, "exit_status=0\n") == null;
        if (failed) self.write("\x1b[38;5;208m");
        const safe = terminalSafe(visible);
        defer c.free(safe);
        const shown = text(safe);
        self.write(shown);
        if (shown.len != 0 and shown[shown.len - 1] != '\n') self.write("\n");
        self.write("\x1b[0m");
    }
    fn refresh(self: Executor, job: *Job, seconds: c_int) void {
        const started = now();
        while (c.ds4_agent_bash_running(job.process) and now() - started < @as(f64, @floatFromInt(seconds))) {
            if (self.interrupted()) break;
            _ = c.usleep(20000);
        }
    }
    fn jobResult(self: Executor, job: *Job, wait: bool, seconds: c_int, stop: bool) [*c]u8 {
        if (stop and c.ds4_agent_bash_running(job.process)) {
            c.ds4_agent_bash_signal(job.process, c.SIGTERM);
            const started = now();
            while (c.ds4_agent_bash_running(job.process) and now() - started < 1.0) _ = c.usleep(20000);
            c.ds4_agent_bash_signal(job.process, c.SIGKILL);
        }
        if (wait or stop) self.refresh(job, seconds);
        var done = false;
        const observation = c.ds4_agent_bash_observation(job.process, true, &done);
        self.publishObservation(observation);
        if (done) removeJob(self.runtime, job);
        return observation;
    }
    fn google(self: Executor, call: *const Call) [*c]u8 {
        const query = argument(call, "query");
        if (text(query).len == 0) return copy("Tool error: google_search requires query\n");
        var errors: [256]u8 = @splat(0);
        self.status("Searching Google for ", text(query));
        const markdown = c.ds4_web_google_search(self.runtime.web, query, &errors, errors.len);
        return if (markdown != null) markdown else failure("google_search", if (errors[0] != 0) text(&errors) else "unknown error");
    }
    // Browser lifetime/permission remains the real Web library's responsibility.
    // Agent observations keep the old bounded head plus secure temp-file shape,
    // so longer rendered pages can be inspected using the ordinary read tool.
    fn visit(self: Executor, call: *const Call) [*c]u8 {
        const url = argument(call, "url");
        if (text(url).len == 0) return copy("Tool error: visit_page requires url\n");
        var errors: [256]u8 = @splat(0);
        self.status("Opening page ", text(url));
        const markdown = c.ds4_web_visit_page(self.runtime.web, url, &errors, errors.len);
        if (markdown == null) return failure("visit_page", if (errors[0] != 0) text(&errors) else "unknown error");
        defer c.free(markdown);
        var path: [c.PATH_MAX]u8 = undefined;
        if (!writeTemporary(markdown, &path, &errors)) return failure("visit_page", if (errors[0] != 0) text(&errors) else "could not store rendered page");
        const value = text(markdown);
        var total_lines: c_int = 0;
        for (value) |byte| {
            if (byte == '\n') total_lines += 1;
        }
        if (value.len != 0 and value[value.len - 1] != '\n') total_lines += 1;
        var used: usize = 0;
        var shown_lines: c_int = 0;
        while (used < value.len and used < c.DS4_AGENT_WEB_HEAD_BYTES and shown_lines < c.DS4_AGENT_WEB_HEAD_LINES) {
            if (value[used] == '\n') shown_lines += 1;
            used += 1;
        }
        const byte_limited = used < value.len and used >= c.DS4_AGENT_WEB_HEAD_BYTES;
        if (used != 0 and value[used - 1] != '\n' and shown_lines < c.DS4_AGENT_WEB_HEAD_LINES) shown_lines += 1;
        const truncated = byte_limited or shown_lines < total_lines;
        var out: Buffer = .{};
        defer out.deinit();
        var line: [c.PATH_MAX + 256]u8 = undefined;
        _ = c.snprintf(&line, line.len, "visit_page url=%s\noutput_path=%s (%zu bytes, %d lines)\n", url, &path, value.len, total_lines);
        out.append(text(&line));
        if (truncated) {
            _ = c.snprintf(&line, line.len, "<head -%d %s>\n", @as(c_int, c.DS4_AGENT_WEB_HEAD_LINES), &path);
            out.append(text(&line));
        } else out.append("<markdown>\n");
        out.append(value[0..used]);
        if (used != 0 and value[used - 1] != '\n') out.append("\n");
        if (truncated) out.append("</head>\nUse read path=<output_path> start_line=<line> max_lines=<count> raw=true to inspect more rendered Markdown.\n") else out.append("</markdown>\n");
        return out.take();
    }
    /// Side effects and model-visible observations only; the streaming owner
    /// has already presented the call. Tools do not import context here.
    pub fn execute(self: Executor, call: *const Call) [*c]u8 {
        if (call.name == null) return copy("Tool error: missing tool name\n");
        if (same(call.name, "read")) return c.ds4_agent_tool_read(&self.runtime.pager, self.context_size, call);
        if (same(call.name, "more")) return c.ds4_agent_tool_more(&self.runtime.pager, self.context_size, call);
        if (same(call.name, "write")) return c.ds4_agent_edit_write(call);
        if (same(call.name, "list")) return c.ds4_agent_tool_list(call);
        if (same(call.name, "edit")) return c.ds4_agent_edit_apply(self.allow_upto, call);
        if (same(call.name, "search")) return c.ds4_agent_tool_search(call);
        if (same(call.name, "google_search")) return self.google(call);
        if (same(call.name, "visit_page")) return self.visit(call);
        if (same(call.name, "bash")) {
            const command = argument(call, "command");
            if (text(command).len == 0) return copy("Tool error: bash requires command\n");
            const seconds = timeout(argument(call, "timeout_sec"));
            const refresh_seconds = integer(argument(call, "refresh_sec"), 60, 1, 3600);
            var errors: [160]u8 = @splat(0);
            const job = startJob(self.runtime, command, seconds, self.terminal_changed, self.context, &errors) orelse {
                var out: Buffer = .{};
                defer out.deinit();
                out.append("Tool error: bash failed to start: ");
                out.append(if (errors[0] != 0) text(&errors) else "unknown error");
                out.append("\n");
                return out.take();
            };
            return self.jobResult(job, true, refresh_seconds, false);
        }
        if (same(call.name, "bash_status") or same(call.name, "bash_stop")) {
            const id = integer(argument(call, "job"), 0, 0, std.math.maxInt(c_int));
            const pid: c.pid_t = @intCast(integer(argument(call, "pid"), 0, 0, std.math.maxInt(c_int)));
            const job = findJob(self.runtime, id, pid) orelse {
                var missing: [128]u8 = undefined;
                _ = c.snprintf(&missing, missing.len, "Tool error: bash job not found: job=%d pid=%ld\n", id, @as(c_long, pid));
                return copy(text(&missing));
            };
            var seconds = integer(argument(call, "refresh_sec"), 0, 0, 3600);
            const stop = same(call.name, "bash_stop");
            const wait = stop or seconds > 0;
            if (stop and seconds == 0) seconds = 1;
            return self.jobResult(job, wait, seconds, stop);
        }
        var header: [256]u8 = undefined;
        _ = c.snprintf(&header, header.len, "\n[tool:%s] unknown tool\n", call.name);
        self.write(text(&header));
        var out: Buffer = .{};
        defer out.deinit();
        out.append("Tool error: unknown tool: ");
        out.append(text(call.name));
        out.append("\n");
        return out.take();
    }
};

fn writeTemporary(rendered: [*c]const u8, path: *[c.PATH_MAX]u8, errors: *[256]u8) bool {
    var temporary: [c.PATH_MAX]u8 = undefined;
    _ = c.snprintf(&temporary, temporary.len, "/tmp/%s_XXXXXX", @as([*:0]const u8, "ds4_agent_web"));
    const descriptor = c.mkstemp(&temporary);
    if (descriptor < 0) {
        _ = c.snprintf(errors, errors.len, "failed to create temporary file: %s", c.strerror(c.__error().*));
        return false;
    }
    const value = text(rendered);
    var written: usize = 0;
    while (written < value.len) {
        const count = c.write(descriptor, value.ptr + written, value.len - written);
        if (count < 0 and c.__error().* == c.EINTR) continue;
        if (count <= 0) {
            _ = c.snprintf(errors, errors.len, "failed to write temporary file: %s", c.strerror(c.__error().*));
            _ = c.close(descriptor);
            _ = c.unlink(&temporary);
            return false;
        }
        written += @intCast(count);
    }
    if (c.close(descriptor) != 0) {
        _ = c.snprintf(errors, errors.len, "failed to close temporary file: %s", c.strerror(c.__error().*));
        _ = c.unlink(&temporary);
        return false;
    }
    _ = c.snprintf(path, path.len, "%s", &temporary);
    return true;
}

/// Context compaction must not lose a still-running external job. Observing
/// jobs marks their head/tail state exactly once and retires completed nodes.
pub fn compactionObservation(runtime: *Runtime) [*c]u8 {
    if (runtime.jobs == null) return null;
    var out: Buffer = .{};
    defer out.deinit();
    out.append("Bash job update after context compaction. Running jobs still need explicit bash_status or bash_stop if relevant.\n");
    var current = runtime.jobs;
    while (current != null) {
        const job: *Job = @ptrCast(current);
        current = job.next;
        var done = false;
        const observation = c.ds4_agent_bash_observation(job.process, true, &done);
        const value = snapshot(job);
        var heading: [64]u8 = undefined;
        _ = c.snprintf(&heading, heading.len, "\nJob %d:\n", value.id);
        out.append(text(&heading));
        out.append(text(observation));
        c.free(observation);
        if (done) removeJob(runtime, job);
    }
    return out.take();
}
