//! Shell job ownership, independent of the agent Worker/model/UI layouts.
//! One monitor owns child reaping and descriptor lifetime; observations copy
//! a small coherent snapshot under its mutex, then read disk outside the lock.
const std = @import("std");
const c = @import("agent_bash_c");
const allocator = std.heap.c_allocator;
pub const Snapshot = c.ds4_agent_bash_snapshot;
const head_bytes = c.DS4_AGENT_BASH_HEAD_BYTES;
const head_lines = c.DS4_AGENT_BASH_HEAD_LINES;
const tail_bytes = c.DS4_AGENT_BASH_TAIL_BYTES;
const progress_lines = c.DS4_AGENT_BASH_PROGRESS_TAIL_LINES;
const final_lines = c.DS4_AGENT_BASH_FINAL_TAIL_LINES;

fn now() f64 {
    var time: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &time);
    return @as(f64, @floatFromInt(time.tv_sec)) + @as(f64, @floatFromInt(time.tv_nsec)) * 1e-9;
}
fn errnoValue() c_int {
    return c.__error().*;
}
fn fatal(label: [*:0]const u8) noreturn {
    c.perror(label);
    c.exit(1);
}
fn duplicate(value: []const u8) [*c]u8 {
    return (allocator.dupeSentinel(u8, value, 0) catch fatal("ds4-agent: malloc")).ptr;
}
fn message(errors: []u8, what: [*:0]const u8, code: c_int) void {
    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s: %s", what, c.strerror(code));
}
const Text = struct {
    list: std.ArrayList(u8) = .empty,

    fn deinit(self: *Text) void {
        self.list.deinit(allocator);
    }
    fn append(self: *Text, value: []const u8) void {
        self.list.appendSlice(allocator, value) catch fatal("ds4-agent: realloc");
    }
    fn byte(self: *Text, value: u8) void {
        self.list.append(allocator, value) catch fatal("ds4-agent: realloc");
    }
    fn take(self: *Text) [*c]u8 {
        return (self.list.toOwnedSliceSentinel(allocator, 0) catch fatal("ds4-agent: malloc")).ptr;
    }
    fn format(self: *Text, comptime pattern: [:0]const u8, args: anytype) void {
        var buffer: [c.PATH_MAX + 256]u8 = undefined;
        const count = @call(.auto, c.snprintf, .{ &buffer, buffer.len, pattern.ptr } ++ args);
        if (count > 0) self.append(buffer[0..@min(@as(usize, @intCast(count)), buffer.len - 1)]);
    }
    fn body(self: *Text, value: [*c]const u8) void {
        const visible = std.mem.span(value);
        self.append(visible);
        if (visible.len != 0 and visible[visible.len - 1] != '\n') self.append("\n");
    }
};

pub const Job = struct {
    data: Snapshot,
    thread: c.pthread_t,
    mutex: c.pthread_mutex_t,
    thread_started: bool = false,
    pipe_fd: c_int,
    tmp_fd: c_int,
    stop_deadline: f64 = 0,
    reaped: bool = false,
    pipe_eof: bool = false,
    child_status: c_int = 0,
    terminal_changed: c.ds4_agent_bash_terminal_fn,
    userdata: ?*anyopaque,

    pub fn running(self: *Job) bool {
        _ = c.pthread_mutex_lock(&self.mutex);
        defer _ = c.pthread_mutex_unlock(&self.mutex);
        return self.data.running;
    }
    pub fn snapshot(self: *Job, mark_observed: bool) Snapshot {
        _ = c.pthread_mutex_lock(&self.mutex);
        defer _ = c.pthread_mutex_unlock(&self.mutex);
        const result = self.data;
        if (mark_observed) self.data.observed_once = true;
        return result;
    }
    pub fn signal(self: *Job, number: c_int) void {
        _ = c.pthread_mutex_lock(&self.mutex);
        defer _ = c.pthread_mutex_unlock(&self.mutex);
        if (self.data.running) {
            if (number == c.SIGKILL and self.stop_deadline == 0) self.stop_deadline = now() + 1;
            _ = c.kill(-self.data.pid, number);
            if (!self.reaped) _ = c.kill(self.data.pid, number);
        }
    }
    pub fn deinit(self: *Job) void {
        self.signal(c.SIGKILL);
        if (self.thread_started) {
            _ = c.pthread_join(self.thread, null);
        } else if (self.data.pid > 0) {
            while (c.waitpid(self.data.pid, null, 0) < 0 and errnoValue() == c.EINTR) {}
        }
        self.closeDescriptors();
        _ = c.pthread_mutex_destroy(&self.mutex);
        allocator.destroy(self);
        // The output path intentionally survives: tools may read its full log.
    }
    fn closeDescriptors(self: *Job) void {
        if (self.pipe_fd >= 0) _ = c.close(self.pipe_fd);
        if (self.tmp_fd >= 0) _ = c.close(self.tmp_fd);
        self.pipe_fd = -1;
        self.tmp_fd = -1;
    }
    fn noteOutput(self: *Job, text: []const u8) void {
        for (text) |character| {
            if (character == '\n' and self.data.newline_count < c.INT_MAX - 1) self.data.newline_count += 1;
        }
        if (text.len != 0) self.data.last_byte = text[text.len - 1];
        self.data.bytes +%= text.len; // C size_t accounting is unsigned.
    }
    fn drain(self: *Job) void {
        if (self.pipe_fd < 0) return;
        var buffer: [4096]u8 = undefined;
        var drained: usize = 0;
        // Yield even for an endless writer, so deadlines still get checked.
        while (drained < 256 * 1024) {
            const count = c.read(self.pipe_fd, &buffer, buffer.len);
            if (count > 0) {
                const length: usize = @intCast(count);
                drained += length;
                var position: usize = 0;
                while (position < length) {
                    const written = c.write(self.tmp_fd, buffer[position..length].ptr, length - position);
                    if (written < 0 and errnoValue() == c.EINTR) continue;
                    if (written <= 0) {
                        self.data.output_error = if (written < 0) errnoValue() else c.EIO;
                        return;
                    }
                    const end = position + @as(usize, @intCast(written));
                    self.noteOutput(buffer[position..end]);
                    position = end;
                }
                continue;
            }
            if (count == 0) self.pipe_eof = true;
            if (count < 0 and errnoValue() == c.EINTR) continue;
            if (count < 0 and errnoValue() != c.EAGAIN and errnoValue() != c.EWOULDBLOCK) self.data.output_error = errnoValue();
            break;
        }
    }
    fn notifyTerminal(self: *Job) void {
        // A child can open /dev/tty even with stdin=/dev/null. Ask the UI to
        // check raw mode at a safe point; never touch linenoise from this job.
        if (self.terminal_changed) |notify| notify(self.userdata);
    }
    fn finalize(self: *Job, status: c_int) void {
        self.drain();
        self.closeDescriptors();
        self.data.exit_status = if (c.WIFEXITED(status)) c.WEXITSTATUS(status) else if (c.WIFSIGNALED(status)) 128 + c.WTERMSIG(status) else -1;
        self.data.running = false;
        self.data.end_time = now();
        self.notifyTerminal();
    }
    fn poll(self: *Job) void {
        if (!self.data.running) return;
        self.drain();
        var status: c_int = 0;
        const result = if (self.reaped) @as(c.pid_t, 0) else c.waitpid(self.data.pid, &status, c.WNOHANG);
        if (result == self.data.pid) {
            self.reaped = true;
            self.child_status = status;
        }
        if (result < 0 and errnoValue() != c.EINTR) {
            self.data.exit_status = -1;
            self.data.running = false;
            self.data.end_time = now();
            self.closeDescriptors();
            self.notifyTerminal();
            return;
        }
        if (self.reaped and self.pipe_eof) {
            self.finalize(self.child_status);
            return;
        }
        if (!self.data.timed_out and (self.data.output_error != 0 or now() - self.data.start_time >= self.data.timeout_sec)) {
            self.data.timed_out = self.data.output_error == 0;
            _ = c.kill(-self.data.pid, c.SIGKILL);
            if (!self.reaped) {
                _ = c.kill(self.data.pid, c.SIGKILL);
                while (c.waitpid(self.data.pid, &status, 0) < 0 and errnoValue() == c.EINTR) {}
                self.reaped = true;
                self.child_status = status;
            }
        }
        if (self.data.output_error != 0 or (self.data.timed_out and now() - self.data.start_time > self.data.timeout_sec + 1) or (self.stop_deadline != 0 and now() >= self.stop_deadline)) {
            // Escaped descendants must not keep inherited stdout alive forever.
            if (self.data.output_error == 0 and !self.pipe_eof) self.data.output_error = c.EIO;
            self.finalize(self.child_status);
        }
    }
    fn monitor(argument: ?*anyopaque) callconv(.c) ?*anyopaque {
        const self: *Job = @ptrCast(@alignCast(argument.?));
        while (true) {
            _ = c.pthread_mutex_lock(&self.mutex);
            self.poll();
            const active = self.data.running;
            const fd = self.pipe_fd;
            _ = c.pthread_mutex_unlock(&self.mutex);
            if (!active) break;
            var descriptor = c.struct_pollfd{ .fd = fd, .events = c.POLLIN, .revents = 0 };
            const result = c.poll(&descriptor, 1, 50);
            // Closed stdout must not turn the deadline monitor into a busy loop.
            if (result > 0 and descriptor.revents & c.POLLHUP != 0 and descriptor.revents & c.POLLIN == 0) _ = c.usleep(50000);
        }
        return null;
    }
    pub fn observation(self: *Job, mark_observed: bool, done: ?*bool) [*c]u8 {
        return observationOf(self.snapshot(mark_observed), done);
    }
};

/// Never fork the inference process: copying wired/private model mappings can
/// exhaust memory before exec, even with copy-on-write. Spawn a fresh group.
fn spawn(pid: *c.pid_t, tmpfd: c_int, pipefd: [2]c_int, command: [*c]const u8) c_int {
    var attributes: c.posix_spawnattr_t = undefined;
    var actions: c.posix_spawn_file_actions_t = undefined;
    var result = c.posix_spawnattr_init(&attributes);
    if (result != 0) return result;
    defer _ = c.posix_spawnattr_destroy(&attributes);
    result = c.posix_spawn_file_actions_init(&actions);
    if (result != 0) return result;
    defer _ = c.posix_spawn_file_actions_destroy(&actions);
    result = c.posix_spawnattr_setpgroup(&attributes, 0);
    if (result == 0) result = c.posix_spawnattr_setflags(&attributes, c.POSIX_SPAWN_SETPGROUP);
    if (result == 0) result = c.posix_spawn_file_actions_addclose(&actions, tmpfd);
    if (result == 0) result = c.posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    if (result == 0) result = c.posix_spawn_file_actions_adddup2(&actions, pipefd[1], c.STDOUT_FILENO);
    if (result == 0) result = c.posix_spawn_file_actions_adddup2(&actions, pipefd[1], c.STDERR_FILENO);
    if (result == 0) result = c.posix_spawn_file_actions_addclose(&actions, pipefd[1]);
    // Tools must not inherit or alter the live terminal's input mode.
    if (result == 0) result = c.posix_spawn_file_actions_addopen(&actions, c.STDIN_FILENO, "/dev/null", c.O_RDONLY, 0);
    const script: [*c]const u8 = if (command != null) command else "";
    var arguments = [_][*c]u8{ @constCast("sh"), @constCast("-c"), @constCast(script), null };
    if (result == 0) result = c.posix_spawn(pid, "/bin/sh", &actions, &attributes, &arguments, c.environ);
    return result;
}

pub fn start(command: [*c]const u8, timeout_sec: c_int, next_id: *c_int, terminal_changed: c.ds4_agent_bash_terminal_fn, userdata: ?*anyopaque, errors: []u8) ?*Job {
    var path = "/tmp/ds4_agent_output_XXXXXX".*;
    var tmpfd = c.mkstemp(&path);
    if (tmpfd < 0) {
        message(errors, "failed to create temporary output file", errnoValue());
        return null;
    }
    var retained = false;
    defer if (!retained) {
        _ = c.unlink(&path);
    };
    defer if (tmpfd >= 0) {
        _ = c.close(tmpfd);
    };
    var pipefd: [2]c_int = .{ -1, -1 };
    if (c.pipe(&pipefd) != 0) {
        message(errors, "failed to create pipe", errnoValue());
        return null;
    }
    defer {
        for (pipefd) |fd| {
            if (fd >= 0) _ = c.close(fd);
        }
    }
    var result: c_int = 0;
    // Keep both dup2 sources above stdio, even if the agent started with its
    // standard descriptors closed. Preserve the original relocation ordering.
    if (pipefd[1] <= c.STDERR_FILENO) {
        const fd = c.fcntl(pipefd[1], c.F_DUPFD_CLOEXEC, @as(c_int, c.STDERR_FILENO + 1));
        if (fd < 0) result = errnoValue() else {
            _ = c.close(pipefd[1]);
            pipefd[1] = fd;
        }
    }
    if (result == 0 and (c.fcntl(tmpfd, c.F_SETFD, @as(c_int, c.FD_CLOEXEC)) < 0 or c.fcntl(pipefd[0], c.F_SETFD, @as(c_int, c.FD_CLOEXEC)) < 0 or c.fcntl(pipefd[1], c.F_SETFD, @as(c_int, c.FD_CLOEXEC)) < 0)) result = errnoValue();
    var pid: c.pid_t = undefined;
    if (result == 0) result = spawn(&pid, tmpfd, pipefd, command);
    if (result != 0) {
        message(errors, "failed to spawn shell", result);
        return null;
    }
    _ = c.close(pipefd[1]);
    pipefd[1] = -1;
    const flags = c.fcntl(pipefd[0], c.F_GETFL, @as(c_int, 0));
    if (flags >= 0) _ = c.fcntl(pipefd[0], c.F_SETFL, flags | c.O_NONBLOCK);
    const job = allocator.create(Job) catch fatal("ds4-agent: malloc");
    job.* = .{
        .data = std.mem.zeroes(Snapshot),
        .thread = std.mem.zeroes(c.pthread_t),
        .mutex = std.mem.zeroes(c.pthread_mutex_t),
        .pipe_fd = pipefd[0],
        .tmp_fd = tmpfd,
        .terminal_changed = terminal_changed,
        .userdata = userdata,
    };
    pipefd[0] = -1;
    tmpfd = -1;
    _ = c.pthread_mutex_init(&job.mutex, null);
    if (next_id.* <= 0) next_id.* = 1;
    job.data.id = next_id.*;
    next_id.* +%= 1;
    job.data.pid = pid;
    _ = c.snprintf(&job.data.path, job.data.path.len, "%s", &path);
    job.data.start_time = now();
    job.data.timeout_sec = @floatFromInt(timeout_sec);
    job.data.exit_status = -1;
    job.data.running = true;
    const thread_result = c.pthread_create(&job.thread, null, Job.monitor, job);
    if (thread_result != 0) {
        message(errors, "failed to monitor shell command", thread_result);
        job.deinit();
        return null;
    }
    job.thread_started = true;
    retained = true;
    return job;
}

const Head = struct { text: [*c]u8, lines: c_int = 0, byte_limited: bool = false };
fn readHead(snapshot: *const Snapshot, max_lines: c_int, max_bytes: usize) Head {
    if (snapshot.path[0] == 0 or snapshot.bytes == 0) return .{ .text = duplicate("") };
    const file = c.fopen(&snapshot.path, "rb") orelse return .{ .text = duplicate("<failed to reopen output file>\n") };
    defer _ = c.fclose(file);
    var text = Text{};
    defer text.deinit();
    var lines: c_int = 0;
    const available = @min(snapshot.bytes, max_bytes);
    while (lines < max_lines and text.list.items.len < available) {
        const character = c.fgetc(file);
        if (character == c.EOF) {
            if (c.ferror(file) != 0 and errnoValue() == c.EINTR) {
                c.clearerr(file);
                continue;
            }
            break;
        }
        text.byte(@intCast(character));
        if (character == '\n') lines += 1;
    }
    const count = text.list.items.len;
    return .{
        .lines = lines + @as(c_int, @intFromBool(count != 0 and text.list.items[count - 1] != '\n')),
        .byte_limited = count >= max_bytes and count < snapshot.bytes,
        .text = text.take(),
    };
}
fn readTail(snapshot: *const Snapshot, max_lines: c_int) [*c]u8 {
    if (snapshot.path[0] == 0 or snapshot.bytes == 0) return duplicate("");
    const file = c.fopen(&snapshot.path, "rb") orelse return duplicate("<failed to reopen output file>\n");
    defer _ = c.fclose(file);
    var text = Text{};
    defer text.deinit();
    var take: usize = @min(snapshot.bytes, tail_bytes);
    if (c.fseeko(file, @intCast(snapshot.bytes - take), c.SEEK_SET) != 0) return duplicate("<failed to seek output file>\n");
    var buffer: [2048]u8 = undefined;
    while (take != 0) {
        const want = @min(take, buffer.len);
        const count = c.fread(&buffer, 1, want, file);
        if (count != 0) text.append(buffer[0..count]);
        take -= count;
        if (count < want) {
            if (c.ferror(file) != 0 and errnoValue() == c.EINTR) {
                c.clearerr(file);
                continue;
            }
            break;
        }
    }
    const tail = text.list.items;
    if (tail.len == 0) return duplicate("");
    var start_index: usize = 0;
    var newlines: c_int = if (tail[tail.len - 1] == '\n') 0 else 1;
    var position = tail.len;
    while (position > 0) : (position -= 1) {
        if (tail[position - 1] == '\n') {
            newlines += 1;
            if (newlines > max_lines) {
                start_index = position;
                break;
            }
        }
    }
    // A byte-limited tail may start inside a UTF8 character.
    while (start_index < tail.len and tail[start_index] & 0xc0 == 0x80) start_index += 1;
    return duplicate(std.mem.sliceTo(tail[start_index..], 0));
}
fn observationOf(snapshot: Snapshot, done: ?*bool) [*c]u8 {
    if (done) |value| value.* = !snapshot.running;
    const display_lines: c_int = if (snapshot.bytes == 0) 0 else snapshot.newline_count + @as(c_int, @intFromBool(snapshot.last_byte != '\n'));
    const elapsed = (if (snapshot.running) now() else snapshot.end_time) - snapshot.start_time;
    var text = Text{};
    defer text.deinit();
    if (snapshot.running) {
        text.format("bash job=%d pid=%ld status=running elapsed_sec=%.1f timeout_sec=%.0f\n", .{ snapshot.id, @as(c_long, snapshot.pid), elapsed, snapshot.timeout_sec });
    } else {
        text.format("bash job=%d pid=%ld status=done elapsed_sec=%.1f timed_out=%d\n", .{ snapshot.id, @as(c_long, snapshot.pid), elapsed, @as(c_int, @intFromBool(snapshot.timed_out)) });
        text.format("exit_status=%d\n", .{snapshot.exit_status});
    }
    if (snapshot.output_error != 0) {
        text.append("Tool error: command output could not be captured completely: ");
        text.append(std.mem.span(c.strerror(snapshot.output_error)));
        text.append("\n");
    }
    if (snapshot.bytes == 0) {
        text.append("<output>\n</output>\n");
    } else if (!snapshot.observed_once) {
        const head = readHead(&snapshot, head_lines, head_bytes);
        defer c.free(head.text);
        const truncated = head.byte_limited or display_lines > head.lines;
        if (!snapshot.running and !truncated) {
            text.append("<output>\n");
            text.body(head.text);
            text.append("</output>\n");
        } else {
            text.format("output_path=%s (%zu bytes, %d lines)\n", .{ @as([*c]const u8, if (snapshot.path[0] != 0) &snapshot.path else "<unavailable>"), snapshot.bytes, display_lines });
            text.format("<head -%d %s>\n", .{ @as(c_int, head_lines), &snapshot.path });
            text.body(head.text);
            text.append("</head>\n");
        }
    } else {
        const count: c_int = if (snapshot.running) progress_lines else final_lines;
        const tail = readTail(&snapshot, count);
        defer c.free(tail);
        text.format("output_path=%s (%zu bytes, %d lines)\n", .{ @as([*c]const u8, if (snapshot.path[0] != 0) &snapshot.path else "<unavailable>"), snapshot.bytes, display_lines });
        text.format("<tail -%d %s>\n", .{ count, &snapshot.path });
        text.body(tail);
        text.append("</tail>\n");
    }
    if (snapshot.running) text.format("\nUse bash_status job=%d to get info before refresh time; use bash_stop job=%d to stop execution\n", .{ snapshot.id, snapshot.id });
    return text.take();
}
