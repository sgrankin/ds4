//! Live linenoise widget and asynchronous output layout. The real Editor and
//! vendor linenoiseState are shared once; no Worker or model state enters here.
const std = @import("std");
const c = @import("agent_runtime_c");
const Ui = @import("ui_text.zig");
const allocator = std.heap.c_allocator;
fn now() f64 {
    var stamp: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &stamp);
    return @as(f64, @floatFromInt(stamp.tv_sec)) + @as(f64, @floatFromInt(stamp.tv_nsec)) * 1e-9;
}
pub fn write(fd: c_int, bytes: []const u8) void {
    if (fd == c.STDOUT_FILENO) {
        _ = c.linenoiseWrite(fd, bytes.ptr, bytes.len);
        return;
    }
    var remaining = bytes;
    while (remaining.len != 0) {
        const count = c.write(fd, remaining.ptr, remaining.len);
        if (count <= 0) {
            if (count < 0 and c.__error().* == c.EINTR) continue;
            return;
        }
        remaining = remaining[@intCast(count)..];
    }
}
fn output(bytes: []const u8) void {
    write(c.STDOUT_FILENO, bytes);
}
fn sequence(comptime pattern: [*:0]const u8, arguments: anytype) void {
    var bytes: [96]u8 = undefined;
    const length = @call(.auto, c.snprintf, .{ &bytes, bytes.len, pattern } ++ arguments);
    if (length > 0) output(bytes[0..@intCast(length)]);
}
pub fn terminalSize(rows: *c_int, cols: *c_int) bool {
    var size: c.struct_winsize = undefined;
    if (c.ioctl(c.STDOUT_FILENO, c.TIOCGWINSZ, &size) != 0 or size.ws_row < 1 or size.ws_col < 1) return false;
    rows.* = size.ws_row;
    cols.* = size.ws_col;
    return true;
}
/// Only definitely complete cursor-position replies may be swallowed; real
/// keystrokes can arrive between their bytes.
pub fn candidate(bytes: []const u8) c.cpr_state {
    if (bytes.len == 0) return c.CPR_PARTIAL;
    if (bytes[0] != 0x1b) return c.CPR_INVALID;
    if (bytes.len == 1) return c.CPR_PARTIAL;
    if (bytes[1] != '[') return c.CPR_INVALID;
    if (bytes.len == 2) return c.CPR_PARTIAL;
    var at: usize = 2;
    if (!std.ascii.isDigit(bytes[at])) return c.CPR_INVALID;
    while (at < bytes.len and std.ascii.isDigit(bytes[at])) at += 1;
    if (at == bytes.len) return c.CPR_PARTIAL;
    if (bytes[at] != ';') return c.CPR_INVALID;
    at += 1;
    if (at == bytes.len) return c.CPR_PARTIAL;
    if (!std.ascii.isDigit(bytes[at])) return c.CPR_INVALID;
    while (at < bytes.len and std.ascii.isDigit(bytes[at])) at += 1;
    if (at == bytes.len) return c.CPR_PARTIAL;
    return if (at + 1 == bytes.len and bytes[at] == 'R') c.CPR_COMPLETE else c.CPR_INVALID;
}
pub const CursorReply = struct { start: usize, end: usize, row: c_int, col: c_int };
pub fn findReply(bytes: []const u8) ?CursorReply {
    var start: usize = 0;
    while (start + 5 < bytes.len) : (start += 1) {
        if (bytes[start] != 0x1b or bytes[start + 1] != '[') continue;
        var at = start + 2;
        var row: c_int = 0;
        var col: c_int = 0;
        if (!std.ascii.isDigit(bytes[at])) continue;
        while (at < bytes.len and std.ascii.isDigit(bytes[at])) : (at += 1) row = row *% 10 +% @as(c_int, bytes[at] - '0');
        if (at >= bytes.len or bytes[at] != ';') continue;
        at += 1;
        if (at >= bytes.len or !std.ascii.isDigit(bytes[at])) continue;
        while (at < bytes.len and std.ascii.isDigit(bytes[at])) : (at += 1) col = col *% 10 +% @as(c_int, bytes[at] - '0');
        if (at < bytes.len and bytes[at] == 'R') return .{ .start = start, .end = at + 1, .row = row, .col = col };
    }
    return null;
}
pub fn terminalText(bytes: []const u8) void {
    var start: usize = 0;
    for (bytes, 0..) |byte, index| {
        if (byte != '\n') continue;
        if (index > start) output(bytes[start..index]);
        output("\r\n");
        start = index + 1;
    }
    if (start < bytes.len) output(bytes[start..]);
}
fn cursor(row: c_int, col: c_int) void {
    sequence("\x1b[%d;%dH", .{ row, col });
}
fn clearRow(row: c_int) void {
    cursor(row, 1);
    output("\r\x1b[0K");
}
fn scrollMargin(bottom: c_int) void {
    sequence("\x1b[1;%dr", .{bottom});
}
fn scrollUp(bottom: c_int, lines: c_int) void {
    if (lines <= 0) return;
    scrollMargin(bottom);
    cursor(bottom, 1);
    for (0..@intCast(lines)) |_| output("\n");
}

pub const Editor = struct {
    state: *c.agent_editor,

    fn flushCandidate(self: Editor) void {
        const state = self.state;
        if (state.cpr_len == 0) return;
        _ = c.linenoiseEditQueueInput(&state.edit, &state.cpr_buf, state.cpr_len);
        state.cpr_len = 0;
    }
    /// Unlike blocking linenoise(), the outer agent loop reads paste in chunks.
    /// Queue the entire envelope, but postpone EditFeed until ESC[201~ arrives
    /// so pasted newlines can never be mistaken for the Enter key.
    fn trackPaste(self: Editor, byte: u8) void {
        const state = self.state;
        const paste_begin = "\x1b[200~";
        const end = "\x1b[201~";
        if (state.paste_tail_len == state.paste_tail.len) {
            std.mem.copyForwards(u8, state.paste_tail[0 .. state.paste_tail.len - 1], state.paste_tail[1..]);
            state.paste_tail_len -= 1;
        }
        state.paste_tail[state.paste_tail_len] = byte;
        state.paste_tail_len += 1;
        const tail = state.paste_tail[0..state.paste_tail_len];
        if (std.mem.endsWith(u8, tail, paste_begin)) {
            state.paste_open = true;
            state.paste_start_pending = false;
        } else if (std.mem.endsWith(u8, tail, end)) {
            state.paste_open = false;
            state.paste_start_pending = false;
        } else {
            state.paste_start_pending = false;
            if (state.paste_open) return;
            var length = @min(tail.len, paste_begin.len - 1);
            while (length > 0) : (length -= 1) {
                if (std.mem.endsWith(u8, tail, paste_begin[0..length])) {
                    state.paste_start_pending = true;
                    break;
                }
            }
        }
    }
    fn filterByte(self: Editor, byte: u8) void {
        const state = self.state;
        if (state.cpr_len != 0 or byte == 0x1b) {
            if (state.cpr_len == state.cpr_buf.len) self.flushCandidate();
            state.cpr_buf[state.cpr_len] = byte;
            state.cpr_len += 1;
            switch (candidate(state.cpr_buf[0..state.cpr_len])) {
                c.CPR_COMPLETE => state.cpr_len = 0, // Late terminal reply.
                c.CPR_INVALID => self.flushCandidate(),
                else => {},
            }
            return;
        }
        _ = c.linenoiseEditQueueInput(&state.edit, &byte, 1);
    }
    pub fn queueBytes(self: Editor, bytes: []const u8) void {
        for (bytes) |byte| {
            self.trackPaste(byte);
            self.filterByte(byte);
        }
    }
    pub fn readStdin(self: Editor) void {
        var buffer: [256]u8 = undefined;
        while (true) {
            const length = c.read(c.STDIN_FILENO, &buffer, buffer.len);
            if (length > 0) {
                self.queueBytes(buffer[0..@intCast(length)]);
                continue;
            }
            if (length < 0 and c.__error().* == c.EINTR) continue;
            return;
        }
    }
    pub fn takeByte(self: Editor, byte: u8) bool {
        const edit = &self.state.edit;
        var position = edit.queued_input_pos;
        while (position < edit.queued_input_len) : (position += 1) {
            if (edit.queued_input[position] != byte) continue;
            std.mem.copyForwards(u8, edit.queued_input[position .. edit.queued_input_len - 1], edit.queued_input[position + 1 .. edit.queued_input_len]);
            edit.queued_input_len -= 1;
            edit.queued_input_pos = @min(edit.queued_input_pos, edit.queued_input_len);
            return true;
        }
        return false;
    }
    pub fn takeEscape(self: Editor) bool {
        if (self.state.cpr_len != 1 or self.state.cpr_buf[0] != 0x1b) return false;
        self.state.cpr_len = 0;
        return true;
    }
    pub fn replaceInput(self: Editor, value: [*c]const u8) void {
        if (self.state.hidden) self.show();
        c.linenoiseEditClear(&self.state.edit);
        if (Ui.text(value).len != 0) _ = c.linenoiseEditInsert(&self.state.edit, value, Ui.text(value).len);
    }
    /// Match linenoise's character/grapheme widths across worker chunks.
    /// Partial UTF8 and ANSI sequences stay pending until more output arrives.
    pub fn noteOutput(self: Editor, bytes: []const u8) void {
        const state = self.state;
        const cols: c_int = if (state.edit.cols > 0) @intCast(state.edit.cols) else 80;
        for (bytes) |byte| {
            if (state.output_escape != 0) {
                if (state.output_escape == 1) state.output_escape = if (byte == '[') 2 else 0 else if (byte >= 0x40 and byte <= 0x7e) state.output_escape = 0;
                continue;
            }
            if (byte == 0x1b and state.output_utf8_len == 0) {
                state.output_escape = 1;
                continue;
            }
            var codepoint: u32 = byte;
            if (state.output_utf8_len != 0 or byte >= 0x80) {
                if (state.output_utf8_len != 0 and byte & 0xc0 != 0x80) {
                    state.output_utf8_len = 0;
                    state.output_col = @rem(state.output_col + 1, cols);
                }
                state.output_utf8[state.output_utf8_len] = byte;
                state.output_utf8_len += 1;
                if (c.linenoiseUtf8Decode(&state.output_utf8, state.output_utf8_len, &codepoint) == 0) continue;
                state.output_utf8_len = 0;
            }
            if (codepoint == '\n' or codepoint == '\r') {
                state.output_col = 0;
                state.output_pending_wrap = false;
                if (codepoint == '\n') state.output_line_open = false;
                state.output_zwj = false;
                state.output_regional = false;
                state.output_glyph_width = 0;
                continue;
            }
            if (codepoint == '\x08') {
                if (state.output_pending_wrap) state.output_col = cols - 1 else if (state.output_col > 0) state.output_col -= 1;
                state.output_pending_wrap = false;
                continue;
            }
            if (codepoint == '\t') {
                const col = if (state.output_pending_wrap) cols - 1 else state.output_col;
                state.output_col = @min((col | 7) + 1, cols - 1);
                state.output_pending_wrap = false;
                state.output_line_open = true;
                continue;
            }
            var width = c.linenoiseCharacterWidth(codepoint);
            const regional = codepoint >= 0x1f1e6 and codepoint <= 0x1f1ff;
            if (codepoint == 0x200d) {
                state.output_zwj = true;
                continue;
            }
            if (codepoint == 0xfe0f and state.output_glyph_width == 1) {
                width = 1;
                state.output_glyph_width = 2;
            } else if (width != 0) {
                if (state.output_zwj or (regional and state.output_regional)) {
                    width = 0;
                    state.output_zwj = false;
                    state.output_regional = false;
                } else {
                    state.output_glyph_width = width;
                    state.output_regional = regional;
                }
            }
            if (width > 0) {
                if (state.output_col + width > cols) state.output_col = 0;
                state.output_col = @rem(state.output_col + width, cols);
                state.output_pending_wrap = state.output_col == 0;
                state.output_line_open = true;
            }
        }
    }
    fn queryCursor(self: Editor, column: *c_int) bool {
        if (c.isatty(c.STDIN_FILENO) == 0 or !Ui.tty()) return false;
        var bytes: [512]u8 = undefined;
        var length: usize = 0;
        output("\x1b[6n");
        for (0..8) |_| {
            var request = c.struct_pollfd{ .fd = c.STDIN_FILENO, .events = c.POLLIN, .revents = 0 };
            const ready = c.poll(&request, 1, 5);
            if (ready <= 0) continue;
            while (true) {
                const count = c.read(c.STDIN_FILENO, bytes[length..].ptr, bytes.len - length);
                if (count > 0) {
                    length += @intCast(count);
                    if (findReply(bytes[0..length])) |reply| {
                        if (reply.start != 0) self.queueBytes(bytes[0..reply.start]);
                        if (reply.end < length) self.queueBytes(bytes[reply.end..length]);
                        column.* = reply.col;
                        return reply.col > 0;
                    }
                    if (length == bytes.len) break;
                    continue;
                }
                if (count < 0 and c.__error().* == c.EINTR) continue;
                break;
            }
        }
        if (length != 0) self.queueBytes(bytes[0..length]);
        return false;
    }
    fn moveToOutput(self: Editor) void {
        output("\x1b[1A");
        sequence("\x1b[%dG", .{self.state.output_col + 1});
    }
    fn saveOutput(self: Editor) void {
        if (self.state.scroll_region) {
            output("\x1b7");
            self.state.output_cursor_saved = true;
        }
    }
    fn restoreOutput(self: Editor) void {
        if (!self.state.scroll_region) return;
        if (self.state.output_cursor_saved) output("\x1b8") else cursor(self.state.output_bottom, 1);
    }
    fn promptRow(self: Editor) void {
        if (self.state.scroll_region) cursor(self.state.prompt_row, 1);
    }
    fn promptCursor(self: Editor) void {
        if (!self.state.scroll_region) return;
        if (self.state.edit.screen_cursor_row > 0 and self.state.edit.screen_cursor_col > 0) cursor(self.state.edit.screen_cursor_row, self.state.edit.screen_cursor_col) else self.promptRow();
    }
    fn clearPrompt(self: Editor) void {
        const state = self.state;
        if (!state.scroll_region) return;
        var row = state.prompt_row;
        while (row <= state.term_rows) : (row += 1) clearRow(row);
        // Absolute reserved rows are ours. Reset linenoise bookkeeping so its
        // next show writes only into those rows, independent of old cursor state.
        state.edit.oldrows = 0;
        state.edit.oldstatusrows = 0;
        state.edit.oldrpos = 1;
        state.edit.oldpos = state.edit.pos;
    }
    pub fn setLayout(self: Editor, requested_rows: c_int, shrink: bool, scroll: bool) bool {
        const state = self.state;
        if (!state.scroll_region) return false;
        var rows: c_int = 0;
        var cols: c_int = 0;
        if (!terminalSize(&rows, &cols) or rows < 8 or cols < 20) return false;
        var reserved = std.math.clamp(requested_rows, 2, rows - 2);
        if (!shrink and state.reserved_rows > 0 and state.term_rows == rows and state.term_cols == cols and reserved < state.reserved_rows) reserved = state.reserved_rows;
        const bottom = rows - reserved;
        const prompt = bottom + 1;
        if (state.term_rows == rows and state.term_cols == cols and state.output_bottom == bottom and state.prompt_row == prompt and state.reserved_rows == reserved) return true;
        // Growing prompt rows first scrolls their former output upward. A
        // shrinking prompt exposes more output space; it restores no old text.
        var scrolled = false;
        if (scroll and state.term_rows == rows and state.term_cols == cols and state.output_bottom > 0 and bottom < state.output_bottom) {
            scrollUp(state.output_bottom, state.output_bottom - bottom);
            scrolled = true;
        }
        scrollMargin(bottom);
        state.term_rows = rows;
        state.term_cols = cols;
        state.output_bottom = bottom;
        state.prompt_row = prompt;
        state.reserved_rows = reserved;
        state.output_cursor_saved = false;
        state.output_at_scroll_boundary = scrolled;
        var row = prompt;
        while (row <= rows) : (row += 1) clearRow(row);
        // A partial line keeps its column when the prompt grows, otherwise the
        // next streamed token would overwrite it from the first column.
        const column = std.math.clamp(if (state.output_line_open) state.output_col + 1 else 1, 1, cols);
        cursor(bottom, column);
        self.saveOutput();
        self.promptRow();
        return true;
    }
    pub fn configure(self: Editor) bool {
        const state = self.state;
        if (state.scroll_region) return true;
        if (c.isatty(c.STDIN_FILENO) == 0 or !Ui.tty()) return false;
        var rows: c_int = 0;
        var cols: c_int = 0;
        if (!terminalSize(&rows, &cols) or rows < 8 or cols < 20) return false;
        state.term_rows = 0;
        state.term_cols = 0;
        state.output_bottom = 0;
        state.prompt_row = 0;
        state.reserved_rows = 0;
        state.output_cursor_saved = false;
        state.output_at_scroll_boundary = false;
        state.scroll_region = true;
        if (!self.setLayout(2, true, false)) return false;
        // Startup logs already exist above the editor. Create an append line
        // rather than overwriting the last startup message with the first token.
        scrollUp(state.output_bottom, 1);
        state.output_cursor_saved = false;
        cursor(state.output_bottom, 1);
        self.saveOutput();
        self.promptRow();
        return true;
    }
    pub fn restoreLayout(self: Editor) void {
        const state = self.state;
        if (!state.scroll_region) return;
        output("\x1b[0m");
        output("\x1b[r");
        cursor(state.term_rows, 1);
        output("\r\x1b[0K\r\n");
        state.scroll_region = false;
        state.output_cursor_saved = false;
        state.term_rows = 0;
        state.term_cols = 0;
        state.output_bottom = 0;
        state.prompt_row = 0;
        state.reserved_rows = 0;
        state.output_at_scroll_boundary = false;
    }
    pub fn start(self: Editor, prompt: [*c]const u8, status: [*c]const u8, initial: [*c]const u8) c_int {
        const state = self.state;
        const input = allocator.alloc(u8, c.AGENT_INPUT_INITIAL_BUFLEN) catch Ui.fatal();
        Ui.copy(&state.prompt, Ui.text(prompt));
        Ui.copy(&state.status, Ui.text(status));
        const had_scroll = state.scroll_region;
        if (self.configure()) {
            if (had_scroll) _ = self.setLayout(2, true, false);
            self.promptRow();
        }
        if (c.linenoiseEditStart(&state.edit, c.STDIN_FILENO, c.STDOUT_FILENO, input.ptr, input.len, &state.prompt) != 0) {
            self.restoreLayout();
            allocator.free(input);
            return -1;
        }
        self.installStatus();
        c.linenoiseEditSetLayoutCallback(&state.edit, layoutChanged, state);
        if (c.isatty(state.edit.ifd) != 0 or c.getenv("LINENOISE_ASSUME_TTY") != null) {
            c.linenoiseHide(&state.edit);
            c.linenoiseShow(&state.edit);
        }
        state.input = input.ptr;
        state.edit.buflen_max = c.AGENT_INPUT_MAX_BUFLEN;
        state.active = true;
        const flags = c.fcntl(c.STDIN_FILENO, c.F_GETFL, @as(c_int, 0));
        state.old_stdin_flags = if (flags >= 0 and c.fcntl(c.STDIN_FILENO, c.F_SETFL, flags | c.O_NONBLOCK) == 0) flags else -1;
        if (Ui.text(initial).len != 0) _ = c.linenoiseEditInsert(&state.edit, initial, Ui.text(initial).len);
        self.resetOutput();
        return 0;
    }
    fn resetOutput(self: Editor) void {
        const state = self.state;
        state.hidden = false;
        state.output_line_open = false;
        state.prompt_below_output = false;
        state.output_col = 0;
        state.output_pending_wrap = false;
        state.output_utf8_len = 0;
        state.output_escape = 0;
        state.output_zwj = false;
        state.output_regional = false;
        state.output_glyph_width = 0;
        state.cpr_len = 0;
        state.paste_open = false;
        state.paste_start_pending = false;
        state.paste_tail_len = 0;
    }
    pub fn stop(self: Editor) void {
        const state = self.state;
        if (!state.active) return;
        if (state.deferred_output.len != 0) _ = self.writeScroll(&.{}, true);
        // linenoise is a live input widget, not persistent scrollback. Hide it
        // before shutdown so a submitted input is not duplicated on reopening.
        if (!state.hidden and (c.isatty(state.edit.ifd) != 0 or c.getenv("LINENOISE_ASSUME_TTY") != null)) self.hide();
        c.linenoiseEditStop(&state.edit);
        if (state.old_stdin_flags >= 0) _ = c.fcntl(c.STDIN_FILENO, c.F_SETFL, state.old_stdin_flags);
        c.free(state.edit.buf);
        state.input = null;
        state.active = false;
        self.resetOutput();
    }
    pub fn hide(self: Editor) void {
        const state = self.state;
        if (!state.active or state.hidden) return;
        if (state.scroll_region) {
            self.clearPrompt();
            self.restoreOutput();
            state.hidden = true;
            return;
        }
        c.linenoiseHide(&state.edit);
        if (state.prompt_below_output) {
            self.moveToOutput();
            state.prompt_below_output = false;
        }
        state.hidden = true;
    }
    pub fn show(self: Editor) void {
        const state = self.state;
        if (!state.active or !state.hidden) return;
        if (state.scroll_region) {
            self.saveOutput();
            self.promptRow();
            output("\x1b[0m");
            c.linenoiseShow(&state.edit);
            state.hidden = false;
            return;
        }
        if (state.output_line_open) {
            output("\r\n");
            state.prompt_below_output = true;
        } else state.prompt_below_output = false;
        // Streamed tools can leave SGR active. The next tool byte restores its
        // own palette, while every prompt redraw begins in normal attributes.
        output("\x1b[0m");
        c.linenoiseShow(&state.edit);
        state.prompt_dirty = false;
        state.status_dirty = false;
        state.last_prompt_redraw_time = now();
        state.hidden = false;
    }
    fn installStatus(self: Editor) void {
        const state = self.state;
        const multiline = std.mem.findScalar(u8, Ui.text(&state.status), '\n') != null;
        _ = c.linenoiseEditSetStatus(&state.edit, &state.status, if (Ui.tty() and !multiline) c.AGENT_STATUS_STYLE_START else @as([*:0]const u8, ""), if (Ui.tty() and state.status[0] != 0) c.AGENT_STATUS_STYLE_END else @as([*:0]const u8, ""));
    }
    fn updatePrompt(self: Editor, prompt: [*c]const u8) void {
        const state = self.state;
        if (!std.mem.eql(u8, Ui.text(&state.prompt), Ui.text(prompt))) state.prompt_dirty = true;
        Ui.copy(&state.prompt, Ui.text(prompt));
        state.edit.prompt = &state.prompt;
        state.edit.plen = Ui.text(&state.prompt).len;
    }
    fn updateStatus(self: Editor, status: [*c]const u8) void {
        if (!std.mem.eql(u8, Ui.text(&self.state.status), Ui.text(status))) self.state.status_dirty = true;
        Ui.copy(&self.state.status, Ui.text(status));
        self.installStatus();
    }
    fn redraw(self: Editor) void {
        const state = self.state;
        if (!state.active or !state.scroll_region) return;
        c.linenoiseBeginUpdate(c.STDOUT_FILENO);
        self.clearPrompt();
        self.promptRow();
        output("\x1b[0m");
        c.linenoiseShow(&state.edit);
        state.prompt_dirty = false;
        state.status_dirty = false;
        state.last_prompt_redraw_time = now();
        c.linenoiseEndUpdate();
    }
    pub fn flush(self: Editor, force_input: bool) void {
        const state = self.state;
        if (!state.active or state.hidden) return;
        var force = force_input;
        var rows: c_int = undefined;
        var cols: c_int = undefined;
        if (state.scroll_region and terminalSize(&rows, &cols) and (rows != state.term_rows or cols != state.term_cols)) {
            state.edit.cols = @intCast(cols);
            state.prompt_dirty = true;
            force = true;
        }
        if (!state.prompt_dirty and !state.status_dirty) return;
        if (!force and state.last_prompt_redraw_time > 0 and now() - state.last_prompt_redraw_time < c.AGENT_STATUS_REDRAW_INTERVAL_SEC) return;
        c.linenoiseBeginUpdate(c.STDOUT_FILENO);
        if (state.scroll_region) {
            if (state.prompt_dirty or c.linenoiseRefreshStatus(&state.edit) == 0) self.redraw();
        } else {
            self.hide();
            self.show();
        }
        state.prompt_dirty = false;
        state.status_dirty = false;
        state.last_prompt_redraw_time = now();
        c.linenoiseEndUpdate();
    }
    pub fn setPromptStatus(self: Editor, prompt: [*c]const u8, status: [*c]const u8) void {
        if (!std.mem.eql(u8, Ui.text(&self.state.prompt), Ui.text(prompt))) self.updatePrompt(prompt);
        if (!std.mem.eql(u8, Ui.text(&self.state.status), Ui.text(status))) self.updateStatus(status);
        self.flush(false);
    }
    pub fn writeScroll(self: Editor, bytes: []const u8, settle: bool) bool {
        const state = self.state;
        Ui.append(&state.deferred_output, bytes);
        if (state.deferred_output.len == 0) return false;
        var predicted = state.*;
        (Editor{ .state = &predicted }).noteOutput(state.deferred_output.ptr[0..state.deferred_output.len]);
        if ((predicted.output_utf8_len != 0 or predicted.output_escape != 0) and settle) {
            if (predicted.output_utf8_len != 0) {
                state.deferred_output.len -= predicted.output_utf8_len;
                Ui.append(&state.deferred_output, "\xef\xbf\xbd");
            } else {
                while (state.deferred_output.len != 0) {
                    state.deferred_output.len -= 1;
                    if (state.deferred_output.ptr[state.deferred_output.len] == 0x1b) break;
                }
            }
            predicted = state.*;
            (Editor{ .state = &predicted }).noteOutput(state.deferred_output.ptr[0..state.deferred_output.len]);
        }
        const at_boundary = predicted.output_pending_wrap;
        if (predicted.output_utf8_len != 0 or predicted.output_escape != 0) return false;
        // DECSC/DECRC may lose pending auto-wrap. Retain a margin-ending chunk
        // until more content can trigger its wrap within the same terminal write.
        if (at_boundary and !settle) return false;
        c.linenoiseBeginUpdate(c.STDOUT_FILENO);
        self.restoreOutput();
        terminalText(state.deferred_output.ptr[0..state.deferred_output.len]);
        self.noteOutput(state.deferred_output.ptr[0..state.deferred_output.len]);
        if (at_boundary) {
            output("\r\n");
            state.output_col = 0;
            state.output_line_open = false;
            state.output_pending_wrap = false;
        }
        Ui.free(&state.deferred_output);
        self.saveOutput();
        output("\x1b[0m");
        self.promptCursor();
        c.linenoiseEndUpdate();
        state.output_at_scroll_boundary = true;
        return true;
    }
    /// Central terminal contract: append model/tool output in upper scroll
    /// rows and return to the real prompt cursor. Fallback terminals hide/show.
    pub fn writeAsync(self: Editor, bytes: []const u8, prompt: [*c]const u8, status: [*c]const u8, force: bool) void {
        const state = self.state;
        if (state.scroll_region and state.active and !state.hidden and (bytes.len != 0 or state.deferred_output.len != 0)) {
            self.flush(false);
            const prompt_changed = !std.mem.eql(u8, Ui.text(&state.prompt), Ui.text(prompt));
            const status_changed = !std.mem.eql(u8, Ui.text(&state.status), Ui.text(status));
            c.linenoiseBeginUpdate(c.STDOUT_FILENO);
            _ = self.writeScroll(bytes, force);
            if (prompt_changed) self.updatePrompt(prompt);
            if (status_changed) self.updateStatus(status);
            self.flush(force);
            c.linenoiseEndUpdate();
            return;
        }
        self.hide();
        if (bytes.len != 0) {
            terminalText(bytes);
            if (state.scroll_region) state.output_at_scroll_boundary = true;
            if (!state.scroll_region) {
                if (bytes[bytes.len - 1] == '\n' or bytes[bytes.len - 1] == '\r') {
                    state.output_col = 0;
                    state.output_line_open = false;
                } else {
                    var col: c_int = 0;
                    if (self.queryCursor(&col)) {
                        const cols: c_int = if (state.edit.cols > 0) @intCast(state.edit.cols) else 80;
                        state.output_col = if (col > 0) col - 1 else 0;
                        state.output_line_open = true;
                        if (state.output_col + 1 >= cols) {
                            output("\r\n");
                            state.output_col = 0;
                        }
                    } else self.noteOutput(bytes);
                }
            }
        }
        if (state.active) {
            self.updatePrompt(prompt);
            self.updateStatus(status);
            if (force or bytes.len != 0) self.show();
        }
    }
    pub fn cancelInput(self: Editor, prompt: [*c]const u8, status: [*c]const u8) void {
        if (!self.state.active) return;
        if (self.state.hidden) self.show();
        c.linenoiseEditClear(&self.state.edit);
        self.writeAsync(if (Ui.tty()) "\x1b[1;33mpress Ctrl+D to exit\x1b[0m\n" else "press Ctrl+D to exit\n", prompt, status, true);
    }
};

fn layoutChanged(_: [*c]c.struct_linenoiseState, prompt_rows: usize, status_rows: usize, context: ?*anyopaque) callconv(.c) c_int {
    if (context == null) return 0;
    const editor = Editor{ .state = @ptrCast(@alignCast(context.?)) };
    if (!editor.state.scroll_region) return 0;
    const reserved: c_int = @bitCast(@as(u32, @truncate(@max(prompt_rows, 1) +% status_rows)));
    return if (editor.setLayout(reserved, true, true)) editor.state.prompt_row else 0;
}
