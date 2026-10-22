//! Owned prompt queue and terminal-only footer/text rendering. None of these
//! actions mutate model tokens, Session, tool state or conversation ownership.
const std = @import("std");
const c = @import("agent_runtime_c");
const allocator = std.heap.c_allocator;
pub fn text(value: [*c]const u8) []const u8 {
    return if (value == null) &.{} else std.mem.span(value);
}
pub fn fatal() noreturn {
    c.perror("ds4-agent: realloc");
    c.exit(1);
}
pub fn tty() bool {
    return c.isatty(c.STDOUT_FILENO) != 0;
}
pub fn copy(output: []u8, value: []const u8) void {
    if (output.len == 0) return;
    const length = @min(output.len - 1, value.len);
    @memcpy(output[0..length], value[0..length]);
    output[length] = 0;
}
fn items(comptime T: type, pointer: [*c]T, length: usize) []T {
    return if (pointer == null and length == 0) &.{} else @as([*]T, @ptrCast(pointer))[0..length];
}
/// Preserve allocated-empty base/capacity when adopting a C-compatible vector.
pub fn append(buffer: *c.agent_input_buf, value: []const u8) void {
    if (value.len == 0) return;
    var list: std.ArrayList(u8) = .empty;
    list.items = items(u8, buffer.ptr, buffer.len);
    list.capacity = buffer.cap;
    list.appendSlice(allocator, value) catch fatal();
    list.ensureUnusedCapacity(allocator, 1) catch fatal();
    list.items.ptr[list.items.len] = 0;
    buffer.* = .{ .ptr = list.items.ptr, .len = list.items.len, .cap = list.capacity };
}
pub fn take(buffer: *c.agent_input_buf) [*c]u8 {
    const result = if (buffer.ptr != null) buffer.ptr else (allocator.dupeSentinel(u8, "", 0) catch fatal()).ptr;
    buffer.* = std.mem.zeroes(c.agent_input_buf);
    return result;
}
pub fn free(buffer: *c.agent_input_buf) void {
    c.free(buffer.ptr);
    buffer.* = std.mem.zeroes(c.agent_input_buf);
}
fn duplicate(value: []const u8) [*c]u8 {
    return (allocator.dupeSentinel(u8, value, 0) catch fatal()).ptr;
}
pub fn echo(value: [*c]const u8) [*c]u8 {
    var output = std.mem.zeroes(c.agent_input_buf);
    append(&output, if (tty()) "\x1b[1;91m*\x1b[1;97m " else "* ");
    append(&output, text(value));
    append(&output, if (tty()) "\x1b[0m\n\n" else "\n\n");
    return take(&output);
}
pub fn contextSize(value: c_int, output: []u8) void {
    if (output.len == 0) return;
    if (value >= 1000) {
        if (@rem(value, 1000) == 0) _ = c.snprintf(output.ptr, output.len, "%dk", @divTrunc(value, 1000)) else _ = c.snprintf(output.ptr, output.len, "%.1fk", @as(f64, @floatFromInt(value)) / 1000);
    } else _ = c.snprintf(output.ptr, output.len, "%d", value);
}
const Writer = struct {
    output: []u8,
    position: usize = 0,
    fn put(self: *Writer, value: []const u8) void {
        if (self.output.len == 0 or self.position >= self.output.len - 1) return;
        const length = @min(value.len, self.output.len - self.position - 1);
        @memcpy(self.output[self.position..][0..length], value[0..length]);
        self.position += length;
        self.output[self.position] = 0;
    }
};
pub fn progressBar(done_input: c_int, total_input: c_int, rate: f64, output: []u8, color: bool) void {
    if (output.len == 0) return;
    const total = @max(total_input, 1);
    const done = std.math.clamp(done_input, 0, total);
    var filled: c_int = @intCast(@divTrunc(@as(i64, done) * c.AGENT_PROGRESS_BAR_WIDTH, total));
    filled = std.math.clamp(filled, 0, c.AGENT_PROGRESS_BAR_WIDTH);
    if (color and filled == 0 and done < total) filled = 1;
    var speed: [32]u8 = @splat(0);
    if (rate > 0 and filled < c.AGENT_PROGRESS_BAR_WIDTH) _ = c.snprintf(&speed, speed.len, " %.0ft/s", rate);
    const speed_text = text(&speed);
    var writer = Writer{ .output = output };
    writer.put("[");
    if (color) writer.put(text(c.AGENT_STATUS_BAR_FILL));
    var index: c_int = 0;
    while (index < c.AGENT_PROGRESS_BAR_WIDTH and writer.position + 1 < output.len) : (index += 1) {
        if (color and index == filled) writer.put(text(c.AGENT_STATUS_STYLE_START));
        if (index >= filled and @as(usize, @intCast(index - filled)) < speed_text.len) writer.put(speed_text[@intCast(index - filled)..][0..1]) else writer.put(if (index < filled) "▶" else "·");
    }
    if (color) writer.put(text(c.AGENT_STATUS_STYLE_START));
    writer.put("]");
    output[@min(writer.position, output.len - 1)] = 0;
}
/// A prefill operation keeps one label so progress updates do not churn it.
pub fn prefillLabel(status: ?*const c.agent_status) [*:0]const u8 {
    const labels = [_][*:0]const u8{ "reading", "absorbing", "studying", "gathering", "crunching", "scrutinizing" };
    return labels[(if (status) |state| state.prefill_label else @as(c_uint, 0)) % labels.len];
}
/// Keep the one-line status compact: linenoise redraws it on progress updates.
pub fn statusText(status: *const c.agent_status, output: []u8) void {
    if (output.len == 0) return;
    var used: [32]u8 = undefined;
    var total: [32]u8 = undefined;
    var power: [32]u8 = @splat(0);
    contextSize(status.ctx_used, &used);
    contextSize(status.ctx_size, &total);
    if (status.power_percent > 0 and status.power_percent < 100) _ = c.snprintf(&power, power.len, " | ⚡ %d%%", status.power_percent);
    switch (status.state) {
        c.AGENT_WORKER_PREFILL => {
            const count: c_int = @max(status.prefill_total, 1);
            const done = @min(status.prefill_done, count);
            var bar: [c.AGENT_PROGRESS_BAR_MAX_BYTES]u8 = undefined;
            progressBar(done, count, status.prefill_tps, &bar, tty());
            _ = c.snprintf(output.ptr, output.len, "ctx %s/%s | %s %s %d/%d %.1f%%%s", &used, &total, prefillLabel(status), &bar, done, count, 100 * @as(f64, @floatFromInt(done)) / @as(f64, @floatFromInt(count)), &power);
        },
        c.AGENT_WORKER_GENERATING => {
            _ = c.snprintf(output.ptr, output.len, "ctx %s/%s | generation %d tokens%s %.1f t/s%s", &used, &total, status.generated, @as([*:0]const u8, if (status.greedy_sampling) " ❄️" else ""), status.gen_tps, &power);
        },
        c.AGENT_WORKER_COMPACTING => {
            _ = c.snprintf(output.ptr, output.len, "ctx %s/%s | COMPACTING summary %d tokens %.1f t/s%s", &used, &total, status.generated, status.gen_tps, &power);
        },
        c.AGENT_WORKER_DRAINING => {
            _ = c.snprintf(output.ptr, output.len, "ctx %s/%s | stopping after distributed cluster drains%s", &used, &total, &power);
        },
        c.AGENT_WORKER_SAVING => {
            _ = c.snprintf(output.ptr, output.len, "ctx %s/%s | saving session%s", &used, &total, &power);
        },
        c.AGENT_WORKER_ERROR => {
            _ = c.snprintf(output.ptr, output.len, "ctx %s/%s | error: %s%s", &used, &total, if (status.@"error"[0] != 0) @as([*c]const u8, &status.@"error") else @as([*c]const u8, "unknown error"), &power);
        },
        c.AGENT_WORKER_STOPPED => {
            _ = c.snprintf(output.ptr, output.len, "ctx %s/%s | interrupted%s", &used, &total, &power);
        },
        else => {
            _ = c.snprintf(output.ptr, output.len, "ctx %s/%s | idle%s", &used, &total, &power);
        },
    }
}

pub const Queue = struct {
    state: *c.agent_prompt_queue,
    fn adopt(self: Queue) std.ArrayList([*c]u8) {
        var result: std.ArrayList([*c]u8) = .empty;
        result.items = items([*c]u8, self.state.v, self.state.len);
        result.capacity = self.state.cap;
        return result;
    }
    fn keep(self: Queue, list: std.ArrayList([*c]u8)) void {
        self.state.* = .{ .v = list.items.ptr, .len = list.items.len, .cap = list.capacity };
    }
    pub fn push(self: Queue, value: [*c]const u8) void {
        var list = self.adopt();
        list.append(allocator, duplicate(text(value))) catch fatal();
        self.keep(list);
    }
    pub fn pop(self: Queue) [*c]u8 {
        if (self.state.len == 0) return null;
        var list = self.adopt();
        const value = list.orderedRemove(0);
        self.keep(list);
        return value;
    }
    pub fn pushFront(self: Queue, value: [*c]u8) void {
        var list = self.adopt();
        list.insert(allocator, 0, value) catch fatal();
        self.keep(list);
    }
    pub fn takeAll(self: Queue) [*c]u8 {
        if (self.state.len == 0) return null;
        if (self.state.len == 1) return self.pop();
        var output = std.mem.zeroes(c.agent_input_buf);
        for (items([*c]u8, self.state.v, self.state.len), 0..) |value, index| {
            if (index != 0) append(&output, "\n\n");
            var header: [64]u8 = undefined;
            _ = c.snprintf(&header, header.len, "Queued user message %zu:\n", index + 1);
            append(&output, text(&header));
            append(&output, text(value));
            c.free(value);
        }
        self.state.len = 0;
        return take(&output);
    }
    pub fn echoAll(self: Queue) [*c]u8 {
        if (self.state.len == 0) return null;
        var output = std.mem.zeroes(c.agent_input_buf);
        for (items([*c]u8, self.state.v, self.state.len)) |value| {
            const rendered = echo(value);
            append(&output, text(rendered));
            c.free(rendered);
        }
        return take(&output);
    }
    pub fn deinit(self: Queue) void {
        for (items([*c]u8, self.state.v, self.state.len)) |value| c.free(value);
        c.free(@ptrCast(self.state.v));
        self.state.* = std.mem.zeroes(c.agent_prompt_queue);
    }
};
/// Queued prompts add a bounded grapheme preview above the status row.
pub fn footer(status: *const c.agent_status, queue: ?*const c.agent_prompt_queue, columns: c_int, output: []u8) void {
    var status_line: [512]u8 = undefined;
    statusText(status, &status_line);
    if (queue == null or queue.?.len == 0) {
        copy(output, text(&status_line));
        return;
    }
    const queued = text(queue.?.v[0]);
    const cols: usize = @intCast(@max(columns, 1));
    const max_rows = 3;
    const suffix = " (ctrl+x to edit, ESC to send ASAP)";
    const budget = cols * max_rows;
    const reserve = suffix.len + 4;
    const preview_bytes = output.len -| 1024;
    var preview = std.mem.zeroes(c.agent_input_buf);
    defer free(&preview);
    append(&preview, "queued: ");
    var position: usize = 0;
    var cells: usize = 8;
    while (position < queued.len) {
        var width: c_int = undefined;
        const length = c.linenoiseNextGrapheme(queued.ptr + position, queued.len - position, &width);
        if (length == 0) break;
        const control = queued[position] < 32 or queued[position] == 127;
        if (control) width = 1;
        const grapheme_width: usize = @intCast(width);
        if (cells + grapheme_width + reserve > budget or preview.len + length > preview_bytes) break;
        append(&preview, if (control) " " else queued[position..][0..length]);
        cells += grapheme_width;
        position += length;
    }
    if (position < queued.len) append(&preview, "...");
    append(&preview, suffix);
    var rendered = std.mem.zeroes(c.agent_input_buf);
    defer free(&rendered);
    const bytes = text(preview.ptr);
    position = 0;
    for (0..max_rows) |row| {
        if (position >= bytes.len) break;
        if (row != 0) append(&rendered, "\n");
        if (tty()) append(&rendered, text(c.AGENT_QUEUE_STYLE));
        cells = 0;
        while (position < bytes.len) {
            var width: c_int = undefined;
            const length = c.linenoiseNextGrapheme(bytes.ptr + position, bytes.len - position, &width);
            if (length == 0 or cells + @as(usize, @intCast(width)) > cols) break;
            append(&rendered, bytes[position..][0..length]);
            cells += @intCast(width);
            position += length;
        }
        if (tty()) append(&rendered, "\x1b[0m");
    }
    append(&rendered, "\n");
    if (tty()) append(&rendered, text(c.AGENT_STATUS_STYLE_START));
    append(&rendered, text(&status_line));
    copy(output, text(rendered.ptr));
}
pub fn welcome(config: *const c.agent_config, output: []u8) void {
    if (output.len == 0) return;
    var context: [32]u8 = undefined;
    contextSize(config.gen.ctx_size, &context);
    _ = c.snprintf(output.ptr, output.len, if (tty()) "\x1b[1;97mDwarf\x1b[1;94mStar\x1b[0m 🐋 Agent, context %s tokens\n\n" else "DwarfStar Agent, context %s tokens\n\n", &context);
}
