//! Main-thread terminal owner. The original UI deliberately counts output bytes
//! as cells; keep that projection and the two-pane redraw/control protocol.
const std = @import("std");
const c = @import("eval_c");
const Common = @import("common.zig");
const Input = @import("input.zig").Input;
const allocator = std.heap.c_allocator;
pub const reset = "\x1b[0m";
pub const bold = "\x1b[1m";
pub const dim = "\x1b[90m";
pub const red = "\x1b[31m";
pub const green = "\x1b[32m";
pub const yellow = "\x1b[33m";
pub const blue = "\x1b[34m";
pub const cyan = "\x1b[36m";
const restore_bytes = reset ++ "\x1b[?25h\x1b[?1049l";
var global_ui: ?*Ui = null;
var input: Input = .{};

pub fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-eval: out of memory\n");
    c.exit(1);
}
pub fn now() f64 {
    var timestamp: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &timestamp);
    return @as(f64, @floatFromInt(timestamp.tv_sec)) + @as(f64, @floatFromInt(timestamp.tv_nsec)) / 1000000000.0;
}
fn runClock() f64 {
    var timestamp: c.struct_timespec = undefined;
    // Darwin uptime excludes system sleep: elapsed benchmark time should not
    // include a period when the laptop lid was closed.
    _ = c.clock_gettime(if (@hasDecl(c, "CLOCK_UPTIME_RAW")) c.CLOCK_UPTIME_RAW else c.CLOCK_MONOTONIC, &timestamp);
    return @as(f64, @floatFromInt(timestamp.tv_sec)) + @as(f64, @floatFromInt(timestamp.tv_nsec)) / 1000000000.0;
}
fn puts(bytes: [*:0]const u8) void {
    _ = c.fputs(bytes, c.__stdoutp);
}
fn move(row: c_int, column: c_int) void {
    _ = c.printf("\x1b[%d;%dH", row, column);
}
fn clearLine() void {
    puts("\x1b[K");
}
pub fn statusName(status: c.eval_status) [*:0]const u8 {
    return switch (status) {
        c.EVAL_PENDING => "PEND",
        c.EVAL_PREFILL => "FILL",
        c.EVAL_THINKING => "RUN",
        c.EVAL_SKIPPED => "SKIP",
        c.EVAL_STOPPED => "STOP",
        c.EVAL_INCOMPLETE => "INC",
        c.EVAL_PASSED => "PASS",
        c.EVAL_FAILED => "FAIL",
        else => "?",
    };
}
pub fn running(status: c.eval_status) bool {
    return status == c.EVAL_PREFILL or status == c.EVAL_THINKING;
}
fn statusColor(status: c.eval_status) void {
    switch (status) {
        c.EVAL_PENDING, c.EVAL_SKIPPED => puts(dim),
        c.EVAL_PREFILL => puts(cyan),
        c.EVAL_THINKING => puts(yellow),
        c.EVAL_STOPPED, c.EVAL_INCOMPLETE => puts(yellow ++ bold),
        c.EVAL_PASSED => puts(green ++ bold),
        c.EVAL_FAILED => puts(red ++ bold),
        else => {},
    }
}
fn trimmed(bytes: []const u8, width: c_int) void {
    if (width <= 0) return;
    const count: usize = @intCast(width);
    if (bytes.len <= count) {
        _ = c.fwrite(bytes.ptr, 1, bytes.len, c.__stdoutp);
        return;
    }
    if (width <= 3) {
        for (0..count) |_| {
            _ = c.fputc('.', c.__stdoutp);
        }
        return;
    }
    _ = c.fwrite(bytes.ptr, 1, count - 3, c.__stdoutp);
    puts("...");
}
fn shortCount(output: []u8, count: c_int) void {
    if (count >= 1000000) _ = c.snprintf(output.ptr, output.len, "%dm", @divTrunc(count + 500000, 1000000)) else if (count >= 10000) _ = c.snprintf(output.ptr, output.len, "%dk", @divTrunc(count + 500, 1000)) else _ = c.snprintf(
        output.ptr,
        output.len,
        "%d",
        count,
    );
}
fn phaseName(phase: []const u8) [*:0]const u8 {
    if (std.mem.eql(u8, phase, "prefill")) return "FILL";
    if (std.mem.eql(u8, phase, "thinking")) return "THINK";
    if (std.mem.eql(u8, phase, "answer")) return "ANS";
    if (std.mem.eql(u8, phase, "passed")) return "PASS";
    if (std.mem.eql(u8, phase, "failed")) return "FAIL";
    return "RUN";
}
fn restore() callconv(.c) void {
    input.stop();
    const ui = global_ui orelse return;
    if (!ui.active) return;
    puts(restore_bytes);
    _ = c.fflush(c.__stdoutp);
    ui.active = false;
}
fn signalRestore(signal: c_int) callconv(.c) void {
    // Full restore joins the reader and uses stdio, so it is not signal safe.
    input.signalStop();
    if (global_ui) |ui| if (ui.active) {
        _ = c.write(c.STDOUT_FILENO, restore_bytes.ptr, restore_bytes.len);
        ui.active = false;
    };
    c.ds4_eval_reraise(signal);
}

pub const Ui = struct {
    cols: c_int = 0,
    rows: c_int = 0,
    left_w: c_int = 0,
    right_x: c_int = 0,
    right_w: c_int = 0,
    body_y: c_int = 0,
    body_h: c_int = 0,
    active: bool = false,
    enabled: bool = false,
    cases: []const c.eval_case = &.{},
    status: []c.eval_status = &.{},
    guess: [][c.EVAL_ANSWER_MAX]u8 = &.{},
    prompt_tokens: []c_int = &.{},
    generated_tokens: []c_int = &.{},
    active_case: c_int = 0,
    generated: c_int = 0,
    max_tokens: c_int = 0,
    think_max_tokens: c_int = 0,
    prefill_current: c_int = 0,
    prefill_total: c_int = 0,
    phase_start_sec: f64 = 0,
    speed_tps: f64 = 0,
    run_elapsed_sec: f64 = 0,
    run_last_sec: f64 = 0,
    run_clock_active: bool = false,
    selection_active: bool = false,
    selected_case: c_int = 0,
    requested_case: c_int = -1,
    paused: bool = false,
    quit_requested: bool = false,
    stream: std.ArrayList(u8) = .empty,
    styles: std.ArrayList(u8) = .empty,
    in_think: bool = false,
    pending_tag: [16]u8 = undefined,
    pending_tag_len: usize = 0,

    pub fn clockTick(self: *Ui) void {
        if (!self.run_clock_active) return;
        const current = runClock();
        const interval = current - self.run_last_sec;
        if (interval > 0) self.run_elapsed_sec += interval;
        self.run_last_sec = current;
    }
    pub fn clockStart(self: *Ui) void {
        if (self.run_clock_active) return;
        self.run_last_sec = runClock();
        self.run_clock_active = true;
    }
    pub fn clockStop(self: *Ui) void {
        if (!self.run_clock_active) return;
        self.clockTick();
        self.run_clock_active = false;
    }
    pub fn elapsed(self: *const Ui) f64 {
        var seconds = self.run_elapsed_sec;
        if (self.run_clock_active) {
            const delta = runClock() - self.run_last_sec;
            if (delta > 0) seconds += delta;
        }
        return seconds;
    }
    // Do not write the physical last column: deferred terminal wrap would spill
    // the right pane's next byte into the next row's left pane.
    fn rightWidth(self: *const Ui) c_int {
        return if (self.right_w > 1) self.right_w - 1 else self.right_w;
    }
    fn clearLeft(self: *Ui, row: c_int) void {
        move(row, 1);
        for (0..@intCast(self.left_w)) |_| {
            _ = c.fputc(' ', c.__stdoutp);
        }
        move(row, self.left_w + 1);
        puts(dim ++ "|" ++ reset);
        move(row, 1);
    }
    fn title(self: *Ui) void {
        move(1, 1);
        self.clearLeft(1);
        const minutes: c_ulonglong = @intFromFloat(@max(self.elapsed(), 0) / 60.0);
        puts("ds4-eval (" ++ bold ++ "p" ++ reset ++ ")ause (" ++ bold ++ "q" ++ reset ++ ")uit");
        _ = c.printf(" %02lluh:%02llum", minutes / 60, minutes % 60);
        if (self.paused) puts(" " ++ red ++ bold ++ "PAUSED" ++ reset);
    }
    fn frame(self: *Ui) void {
        puts("\x1b[2J");
        self.title();
        move(1, self.right_x);
        puts(bold ++ "live sampled tokens" ++ reset);
        var row: c_int = 1;
        while (row <= self.rows) : (row += 1) {
            move(row, self.left_w + 1);
            puts(dim ++ "|" ++ reset);
        }
    }
    fn questionPreview(self: *Ui, case: *const c.eval_case) void {
        const question = Common.text(case.question);
        var position: usize = 0;
        const width = self.rightWidth();
        var row: c_int = 0;
        while (row < 3) : (row += 1) {
            move(2 + row, self.right_x);
            clearLine();
            puts(blue);
            var printed: c_int = 0;
            var last_space = true;
            while (position < question.len and printed < width) {
                var byte = question[position];
                position += 1;
                if (c.isspace(byte) != 0) {
                    if (last_space) continue;
                    byte = ' ';
                    last_space = true;
                } else last_space = false;
                _ = c.fputc(byte, c.__stdoutp);
                printed += 1;
            }
            if (row == 2) {
                while (position < question.len and c.isspace(question[position]) != 0) position += 1;
                if (position < question.len and width >= 3) {
                    move(2 + row, self.right_x + if (printed >= 3) printed - 3 else @as(c_int, 0));
                    puts("...");
                }
            }
            puts(reset);
        }
        move(5, self.right_x);
        clearLine();
    }
    fn left(self: *Ui) void {
        var passed: c_int = 0;
        var failed: c_int = 0;
        var incomplete: c_int = 0;
        for (self.status) |status| {
            if (status == c.EVAL_PASSED) passed += 1 else if (status == c.EVAL_FAILED) failed += 1 else if (status == c.EVAL_INCOMPLETE) incomplete += 1;
        }
        self.title();
        move(2, 1);
        self.clearLeft(2);
        _ = c.printf(
            "score %s%d%s/%d  failed %s%d%s  incomplete %s%d%s",
            green,
            passed,
            reset,
            @as(c_int, @intCast(self.cases.len)),
            if (failed != 0) @as([*:0]const u8, red) else @as([*:0]const u8, dim),
            failed,
            reset,
            if (incomplete != 0) @as([*:0]const u8, yellow) else @as([*:0]const u8, dim),
            incomplete,
            reset,
        );
        const visible_rows = if (self.rows >= 4) self.rows - 3 else 0;
        const count: c_int = @intCast(self.cases.len);
        const shown = @min(count, visible_rows);
        var first: c_int = 0;
        if (shown > 0 and count > shown) {
            // Follow the selection cursor immediately, not just the active run.
            const anchor = if (self.selected_case < 0) self.active_case else @min(self.selected_case, count - 1);
            first = std.math.clamp(anchor - @divTrunc(shown, 2), 0, count - shown);
        }
        var row: c_int = 0;
        while (row < visible_rows) : (row += 1) {
            const screen_row = 4 + row;
            const index: usize = @intCast(first + row);
            move(screen_row, 1);
            self.clearLeft(screen_row);
            if (index >= self.cases.len) continue;
            if (@as(c_int, @intCast(index)) == self.active_case) puts(bold);
            statusColor(self.status[index]);
            _ = c.printf(
                "%c%2d %-4s",
                @as(c_int, if (self.selection_active and @as(c_int, @intCast(index)) == self.selected_case) '>' else ' '),
                @as(c_int, @intCast(index + 1)),
                statusName(self.status[index]),
            );
            puts(reset);
            const answer_column = self.left_w - 18 + 1;
            const title_width = answer_column - 11;
            if (title_width > 0) {
                _ = c.fputc(' ', c.__stdoutp);
                var text: [512]u8 = undefined;
                _ = c.snprintf(&text, text.len, "%s: %s", self.cases[index].source, self.cases[index].title);
                trimmed(Common.written(&text), title_width);
            }
            if (self.status[index] == c.EVAL_FAILED or self.status[index] == c.EVAL_PASSED or self.status[index] == c.EVAL_INCOMPLETE) {
                var answers: [64]u8 = undefined;
                _ = c.snprintf(
                    &answers,
                    answers.len,
                    "%s/%s",
                    if (self.guess[index][0] != 0) @as([*c]const u8, &self.guess[index]) else @as([*c]const u8, "?"),
                    self.cases[index].answer,
                );
                move(screen_row, answer_column);
                trimmed(Common.written(&answers), 18);
            }
            puts(reset);
        }
    }
    pub fn resetStream(self: *Ui, case: *const c.eval_case, in_think: bool) void {
        self.stream.clearRetainingCapacity();
        self.styles.clearRetainingCapacity();
        self.in_think = in_think;
        self.pending_tag_len = 0;
        self.generated = 0;
        self.prefill_current = 0;
        self.prefill_total = 0;
        self.phase_start_sec = now();
        self.speed_tps = 0;
        var row: c_int = 2;
        while (row <= self.rows) : (row += 1) {
            move(row, self.right_x);
            clearLine();
        }
        self.questionPreview(case);
    }
    fn visible(self: *Ui, bytes: []const u8) void {
        self.stream.appendSlice(allocator, bytes) catch fatal();
        self.styles.appendNTimes(allocator, if (self.in_think) 1 else 0, bytes.len) catch fatal();
    }
    pub fn append(self: *Ui, bytes: []const u8, finish: bool) void {
        const temporary = allocator.alloc(u8, @max(self.pending_tag_len + bytes.len, 1)) catch fatal();
        defer allocator.free(temporary);
        @memcpy(temporary[0..self.pending_tag_len], self.pending_tag[0..self.pending_tag_len]);
        @memcpy(temporary[self.pending_tag_len..][0..bytes.len], bytes);
        const total = self.pending_tag_len + bytes.len;
        self.pending_tag_len = 0;
        var position: usize = 0;
        while (position < total) {
            const rest = temporary[position..total];
            if (std.mem.startsWith(u8, rest, "<think>")) {
                self.in_think = true;
                position += 7;
                continue;
            }
            if (std.mem.startsWith(u8, rest, "</think>")) {
                self.in_think = false;
                self.visible("\n");
                position += 8;
                continue;
            }
            if (!finish and rest[0] == '<' and (std.mem.startsWith(u8, "<think>", rest) or std.mem.startsWith(u8, "</think>", rest))) {
                if (rest.len < self.pending_tag.len) {
                    @memcpy(self.pending_tag[0..rest.len], rest);
                    self.pending_tag_len = rest.len;
                }
                break;
            }
            self.visible(rest[0..1]);
            position += 1;
        }
    }
    pub fn drawStream(self: *Ui) void {
        if (!self.enabled) return;
        const Span = struct { start: usize, end: usize };
        var lines: std.ArrayList(Span) = .empty;
        defer lines.deinit(allocator);
        var start: usize = 0;
        var column: c_int = 0;
        const width = self.rightWidth();
        for (self.stream.items, 0..) |byte, index| {
            var end = index;
            const end_line = byte == '\n' or blk: {
                column += 1;
                if (column >= width) {
                    end = index + 1;
                    break :blk true;
                }
                break :blk false;
            };
            if (end_line) {
                lines.append(allocator, .{ .start = start, .end = end }) catch fatal();
                start = index + 1;
                column = 0;
            }
        }
        if (start <= self.stream.items.len) lines.append(allocator, .{ .start = start, .end = self.stream.items.len }) catch fatal();
        const first = lines.items.len - @min(lines.items.len, @as(usize, @intCast(self.body_h)));
        var row: c_int = 0;
        while (row < self.body_h) : (row += 1) {
            move(self.body_y + row, self.right_x);
            clearLine();
            const index = first + @as(usize, @intCast(row));
            if (index >= lines.items.len) continue;
            const span = lines.items[index];
            var style: u8 = 255;
            var printed: c_int = 0;
            for (span.start..span.end) |offset| {
                if (printed >= width) break;
                const next_style = self.styles.items[offset];
                if (next_style != style) {
                    puts(if (next_style != 0) dim else reset);
                    style = next_style;
                }
                var byte = self.stream.items[offset];
                if (byte == '\r' or byte == '\n') continue;
                if (byte == '\t') byte = ' ';
                _ = c.fputc(byte, c.__stdoutp);
                printed += 1;
            }
            puts(reset);
        }
    }
    fn rightStatus(self: *Ui, phase: []const u8) void {
        move(1, self.right_x);
        clearLine();
        const index: usize = @intCast(self.active_case);
        const case = &self.cases[index];
        var id: [13]u8 = undefined;
        var current: [16]u8 = undefined;
        var maximum: [16]u8 = undefined;
        var line: [512]u8 = undefined;
        _ = c.snprintf(&id, id.len, "%.12s", case.id);
        if (self.prefill_total > 0 and self.status[index] == c.EVAL_PREFILL) {
            const percent = 100.0 * @as(f64, @floatFromInt(self.prefill_current)) / @as(f64, @floatFromInt(self.prefill_total));
            shortCount(&current, self.prefill_current);
            shortCount(&maximum, self.prefill_total);
            _ = c.snprintf(
                &line,
                line.len,
                "Speed %.2f t/s  %s %s/%s %.0f%%  %s/%s",
                self.speed_tps,
                phaseName(phase),
                &current,
                &maximum,
                percent,
                case.source,
                &id,
            );
        } else {
            const thinking = std.mem.eql(u8, phase, "thinking");
            const limit = @max(if (thinking) self.think_max_tokens else self.max_tokens, 0);
            const generated = if (thinking and self.generated > limit) limit else self.generated;
            shortCount(&current, generated);
            shortCount(&maximum, limit);
            _ = c.snprintf(&line, line.len, "Speed %.2f t/s  %s %s/%s  %s/%s", self.speed_tps, phaseName(phase), &current, &maximum, case.source, &id);
        }
        trimmed(Common.written(&line), self.rightWidth());
    }
    pub fn refresh(self: *Ui, phase: []const u8) void {
        if (!self.enabled) return;
        self.left();
        self.rightStatus(phase);
        self.drawStream();
        _ = c.fflush(c.__stdoutp);
    }
    pub fn consume(self: *Ui) void {
        if (!self.enabled or !input.enabled) return;
        const keys = input.take();
        if (keys.quit) {
            self.quit_requested = true;
            self.paused = false;
            return;
        }
        if (keys.pause) self.paused = !self.paused;
        const count: c_int = @intCast(self.cases.len);
        if (keys.move != 0) {
            if (!self.selection_active) {
                self.selected_case = self.active_case;
                self.selection_active = true;
            }
            self.selected_case = std.math.clamp(self.selected_case + keys.move, 0, count - 1);
        }
        if (keys.enter) {
            if (!self.selection_active) {
                self.selected_case = self.active_case;
                self.selection_active = true;
            }
            if (self.selected_case != self.active_case or !running(self.status[@intCast(self.active_case)])) self.requested_case = self.selected_case;
        }
    }
    pub fn startRun(self: *Ui, cases: []const c.eval_case, max_tokens: c_int, enabled: bool) void {
        self.* = .{ .enabled = enabled, .cases = cases, .max_tokens = max_tokens };
        self.status = allocator.alloc(c.eval_status, cases.len) catch fatal();
        @memset(self.status, c.EVAL_PENDING);
        self.guess = allocator.alloc([c.EVAL_ANSWER_MAX]u8, cases.len) catch fatal();
        @memset(std.mem.sliceAsBytes(self.guess), 0);
        self.prompt_tokens = allocator.alloc(c_int, cases.len) catch fatal();
        @memset(self.prompt_tokens, 0);
        self.generated_tokens = allocator.alloc(c_int, cases.len) catch fatal();
        @memset(self.generated_tokens, 0);
        if (!enabled) return;
        var size: c.struct_winsize = undefined;
        if (c.ioctl(c.STDOUT_FILENO, c.TIOCGWINSZ, &size) != 0 or size.ws_col == 0 or size.ws_row == 0) {
            self.cols = 80;
            self.rows = 24;
        } else {
            self.cols = size.ws_col;
            self.rows = size.ws_row;
        }
        if (self.cols < 90 or self.rows < 18) {
            self.enabled = false;
            return;
        }
        self.left_w = std.math.clamp(@divTrunc(self.cols, 2), 42, 72);
        self.right_x = self.left_w + 3;
        self.right_w = self.cols - self.right_x + 1;
        self.body_y = 6;
        self.body_h = self.rows - self.body_y + 1;
        global_ui = self;
        _ = c.atexit(restore);
        c.ds4_eval_install_signals(signalRestore);
        puts("\x1b[?1049h\x1b[?25l");
        self.active = true;
        input.start();
        self.frame();
        self.refresh("idle");
    }
    pub fn deinit(self: *Ui) void {
        if (self.active) restore();
        allocator.free(self.status);
        allocator.free(self.guess);
        allocator.free(self.prompt_tokens);
        allocator.free(self.generated_tokens);
        self.stream.deinit(allocator);
        self.styles.deinit(allocator);
        self.* = .{};
    }
};

pub fn restoreTerminal() void {
    restore();
}
