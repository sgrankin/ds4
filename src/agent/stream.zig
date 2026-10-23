//! One streaming terminal projection for assistant text and executable tools.
//! Parser state remains authoritative. Raw DSML is hidden only in the display;
//! the sampled transcript and parser's owned raw/call buffers stay unchanged.
const std = @import("std");
const c = @import("agent_presentation_c");
const presentation = @import("presentation.zig");
const Renderer = presentation.Renderer;
pub const Record = c.agent_stream_renderer;
const Visualizer = c.agent_tool_visualizer;
const Kind = c.agent_tool_param_kind;
const search = c.AGENT_DSML_SEARCH;
const structural = c.AGENT_DSML_STRUCTURAL;
const value_state = c.AGENT_DSML_PARAM_VALUE;
const done = c.AGENT_DSML_DONE;
const invalid = c.AGENT_DSML_ERROR;

fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn same(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
fn copyText(destination: []u8, source: []const u8) void {
    const count = @min(destination.len - 1, source.len);
    @memcpy(destination[0..count], source[0..count]);
    destination[count] = 0;
}
fn append(destination: []u8, byte: u8) void {
    const length = std.mem.indexOfScalar(u8, destination, 0) orelse destination.len;
    if (length + 1 >= destination.len) return;
    destination[length] = byte;
    destination[length + 1] = 0;
}
fn xmlSyntax(syntax: c.agent_tool_syntax) bool {
    return syntax == c.AGENT_TOOL_SYNTAX_GLM or syntax == c.AGENT_TOOL_SYNTAX_QWEN;
}
fn boolValue(pointer: [*c]const u8) bool {
    return pointer != null and (c.strcasecmp(pointer, "true") == 0 or
        c.strcasecmp(pointer, "yes") == 0 or c.strcmp(pointer, "1") == 0);
}
fn oneOf(value: []const u8, choices: []const []const u8) bool {
    for (choices) |choice| if (same(value, choice)) return true;
    return false;
}
fn paramKind(tool: []const u8, name: []const u8) Kind {
    if (same(tool, "bash") and same(name, "command")) return c.AGENT_TOOL_PARAM_BASH_COMMAND;
    if (same(tool, "edit") and same(name, "old")) return c.AGENT_TOOL_PARAM_DIFF_OLD;
    if (same(tool, "edit") and same(name, "new")) return c.AGENT_TOOL_PARAM_DIFF_NEW;
    if (oneOf(name, &.{ "path", "file", "filename" })) return c.AGENT_TOOL_PARAM_PATH;
    if (oneOf(name, &.{ "line", "start_line", "end_line", "offset", "start", "end", "count", "max_lines", "timeout_sec", "refresh_sec" })) return c.AGENT_TOOL_PARAM_OFFSET;
    if (oneOf(name, &.{ "content", "text" })) return c.AGENT_TOOL_PARAM_CONTENT;
    return c.AGENT_TOOL_PARAM_NORMAL;
}
fn paramColor(kind: Kind) []const u8 {
    return switch (kind) {
        c.AGENT_TOOL_PARAM_PATH, c.AGENT_TOOL_PARAM_DIFF_NEW => "\x1b[32m",
        c.AGENT_TOOL_PARAM_OFFSET => "\x1b[33m",
        c.AGENT_TOOL_PARAM_CONTENT => "\x1b[34m",
        c.AGENT_TOOL_PARAM_DIFF_OLD => "\x1b[31m",
        c.AGENT_TOOL_PARAM_BASH_COMMAND => "\x1b[1;36m",
        else => "\x1b[37m",
    };
}
fn toolPrefix(name: []const u8) ?[]const u8 {
    const names = [_][]const u8{ "bash", "read", "write", "edit", "search", "google_search", "visit_page" };
    const prefixes = [_][]const u8{ "$ ", "read ", "write ", "edit ", "search ", "google ", "visit " };
    for (names, prefixes) |tool, prefix| if (same(name, tool)) return prefix;
    return null;
}
const Diff = struct { prefix: [*:0]const u8, color: [*:0]const u8 };
fn diff(kind: Kind) ?Diff {
    return switch (kind) {
        c.AGENT_TOOL_PARAM_DIFF_OLD => .{ .prefix = "- ", .color = "\x1b[31m" },
        c.AGENT_TOOL_PARAM_DIFF_NEW => .{ .prefix = "+ ", .color = "\x1b[32m" },
        else => null,
    };
}

pub const Stream = struct {
    record: *Record,

    fn renderer(self: Stream) Renderer {
        return .{ .record = @ptrCast(self.record.renderer) };
    }
    fn parser(self: Stream) *c.agent_dsml_parser {
        return @ptrCast(self.record.parser);
    }
    fn trace(self: Stream, prefix: [*:0]const u8, detail: [*c]const u8) void {
        var message: [512]u8 = undefined;
        _ = c.snprintf(&message, message.len, "%s%s", prefix, detail);
        c.ds4_agent_stream_trace(self.renderer().record.worker, &message);
    }
    fn traceDone(self: Stream) void {
        var message: [64]u8 = undefined;
        _ = c.snprintf(&message, message.len, "dsml done calls=%d", self.parser().calls.len);
        c.ds4_agent_stream_trace(self.renderer().record.worker, &message);
    }
    fn write(self: Stream, bytes: []const u8) void {
        self.renderer().plain(bytes);
        if (bytes.len != 0) self.record.viz.last_output_newline = bytes[bytes.len - 1] == '\n';
    }
    fn start(self: Stream) void {
        const state = self.record;
        const line_open = !self.renderer().record.last_output_newline;
        state.viz = std.mem.zeroes(Visualizer);
        state.viz.active = true;
        state.viz.at_line_start = true;
        state.viz.last_output_newline = true;
        if (state.replay) {
            if (line_open) self.write("\n");
        } else if (self.renderer().record.use_color) {
            // A stanza may follow prose on the current row. Cursor control is
            // live-terminal UI only; plain stdout must stay safe for pipes.
            self.write("\r\x1b[2K");
        } else if (line_open) {
            self.write("\n");
        }
        state.viz.last_output_newline = true;
    }
    fn linePrefix(self: Stream) void {
        if (!self.record.viz.last_output_newline) self.write("\n");
        self.write("🛠️ ");
        self.record.viz.at_line_start = false;
    }
    fn tool(self: Stream, name: [*c]const u8) void {
        const v = &self.record.viz;
        if (v.tool_announced and same(text(&v.tool_name), text(name))) return;
        if (v.tool_announced and !v.last_output_newline) self.write("\n");
        copyText(&v.tool_name, if (name == null) "tool" else text(name));
        v.tool_announced = true;
        v.read_style = same(text(&v.tool_name), "read");
        self.linePrefix();
        if (v.read_style) {
            self.renderer().color("\x1b[1;37m");
            self.write("Reading ");
            self.renderer().color("\x1b[32m");
            v.read_prefix_rendered = true;
            return;
        }
        self.renderer().color(if (same(text(&v.tool_name), "bash")) "\x1b[1;36m" else "\x1b[1;37m");
        if (toolPrefix(text(&v.tool_name))) |prefix| {
            self.write(prefix);
        } else {
            self.write(text(&v.tool_name));
            self.write(" ");
        }
        self.renderer().color("\x1b[0m");
    }
    fn readByte(self: Stream, byte: u8) void {
        const v = &self.record.viz;
        const name = text(&v.param_name);
        if (same(name, "path")) {
            append(&v.read_path, byte);
            if (v.read_prefix_rendered) self.write(&.{byte});
        } else if (same(name, "start_line")) {
            append(&v.read_start, byte);
        } else if (same(name, "max_lines")) {
            append(&v.read_max, byte);
        } else if (same(name, "whole")) {
            append(&v.read_whole, byte);
        }
    }
    fn renderRead(self: Stream) void {
        const v = &self.record.viz;
        if (!v.read_style or v.read_line_rendered) return;
        if (!v.read_prefix_rendered) {
            self.linePrefix();
            self.renderer().color("\x1b[1;37m");
            self.write("Reading ");
            self.renderer().color("\x1b[32m");
            self.write(if (v.read_path[0] != 0) text(&v.read_path) else "<unknown>");
        } else if (v.read_path[0] == 0) {
            self.renderer().color("\x1b[32m");
            self.write("<unknown>");
        }
        self.renderer().color("\x1b[33m");
        if (boolValue(&v.read_whole)) {
            if (v.read_start[0] == 0 or same(text(&v.read_start), "1")) {
                self.write(" (whole file)");
            } else {
                self.write(" ");
                self.write(text(&v.read_start));
                self.write(":EOF");
            }
        } else {
            var default_lines: [16]u8 = undefined;
            const worker = if (self.record.renderer != null) self.renderer().record.worker else null;
            _ = c.snprintf(&default_lines, default_lines.len, "%d", c.ds4_agent_stream_read_lines(worker));
            self.write(" ");
            self.write(if (v.read_start[0] != 0) text(&v.read_start) else "1");
            self.write(":");
            self.write(if (v.read_max[0] != 0) text(&v.read_max) else text(&default_lines));
        }
        self.renderer().color("\x1b[1;37m");
        self.write("...");
        self.renderer().color("\x1b[0m");
        self.write("\n");
        v.read_line_rendered = true;
    }
    fn codeBody(v: *const Visualizer) bool {
        const name = text(&v.tool_name);
        return (same(name, "write") and v.param_kind == c.AGENT_TOOL_PARAM_CONTENT) or
            (same(name, "edit") and (v.param_kind == c.AGENT_TOOL_PARAM_DIFF_OLD or
                v.param_kind == c.AGENT_TOOL_PARAM_DIFF_NEW or v.param_kind == c.AGENT_TOOL_PARAM_CONTENT));
    }
    fn codePrefix(self: Stream) void {
        const v = &self.record.viz;
        if (!v.at_line_start) return;
        const prefix = diff(v.param_kind) orelse return;
        self.renderer().color(text(prefix.color));
        self.renderer().write(text(prefix.prefix));
        self.renderer().color("\x1b[0m");
        self.renderer().record.wrote_visible_output = true;
        self.renderer().record.last_output_newline = false;
        v.last_output_newline = false;
        v.at_line_start = false;
    }
    fn codeBegin(self: Stream) void {
        const v = &self.record.viz;
        self.renderer().codeBegin(presentation.syntaxForPath(text(&v.tool_path)));
        self.renderer().setCodeUpto(same(text(&v.tool_name), "edit") and v.param_kind == c.AGENT_TOOL_PARAM_DIFF_OLD);
        v.code_param_active = true;
        if (diff(v.param_kind)) |prefix| {
            // Diff prefixes are UI, not code. Keep them outside the syntax
            // buffer so a repaint highlights only the actual edited line.
            self.renderer().setCodePrefix(prefix.prefix, prefix.color);
            self.codePrefix();
        }
    }
    fn codeEnd(self: Stream) void {
        const v = &self.record.viz;
        if (!v.code_param_active) return;
        self.renderer().codeEnd();
        v.code_param_active = false;
        v.at_line_start = true;
        v.last_output_newline = self.renderer().record.last_output_newline;
    }
    fn codeByte(self: Stream, byte: u8) void {
        self.codePrefix();
        self.renderer().codeByte(byte);
        self.record.viz.last_output_newline = byte == '\n';
        self.record.viz.at_line_start = byte == '\n';
    }
    fn paramBegin(self: Stream, name: [*c]const u8) void {
        const v = &self.record.viz;
        const p = self.parser();
        if (!v.tool_announced and p.current.name != null) self.tool(p.current.name);
        copyText(&v.param_name, text(name));
        v.param_kind = paramKind(text(&v.tool_name), text(&v.param_name));
        v.param_active = true;
        v.param_end_len = 0;
        if (v.read_style) return;
        if (diff(v.param_kind) != null) {
            if (!v.last_output_newline) self.write("\n");
            v.at_line_start = true;
            self.codeBegin();
            return;
        }
        if (v.param_kind == c.AGENT_TOOL_PARAM_CONTENT) {
            if (!v.last_output_newline) self.write("\n");
            if (!same(text(&v.tool_name), "write")) {
                self.renderer().color("\x1b[1;37m");
                self.write(text(&v.param_name));
                self.write(":\n");
            }
            v.at_line_start = true;
            if (codeBody(v)) self.codeBegin() else self.renderer().color("\x1b[34m");
            return;
        }
        if (v.param_kind != c.AGENT_TOOL_PARAM_BASH_COMMAND) {
            if (!v.at_line_start) self.write(" ");
            self.renderer().color("\x1b[1;37m");
            self.write(text(&v.param_name));
            self.write("=");
        } else {
            self.renderer().color(paramColor(c.AGENT_TOOL_PARAM_BASH_COMMAND));
            return;
        }
        self.renderer().color(paramColor(v.param_kind));
    }
    fn paramEnd(self: Stream) void {
        const v = &self.record.viz;
        v.param_end_len = 0;
        if (v.code_param_active) self.codeEnd();
        if (!v.read_style) self.renderer().color("\x1b[0m");
        v.param_active = false;
        v.param_name[0] = 0;
    }
    fn paramRaw(self: Stream, byte: u8) void {
        const v = &self.record.viz;
        if (v.read_style) {
            self.readByte(byte);
            return;
        }
        if (v.param_kind == c.AGENT_TOOL_PARAM_PATH) append(&v.tool_path, byte);
        if (v.code_param_active) {
            self.codeByte(byte);
            return;
        }
        if (v.param_kind == c.AGENT_TOOL_PARAM_BASH_COMMAND) {
            self.write(&.{byte});
            v.at_line_start = byte == '\n';
            return;
        }
        if (diff(v.param_kind) != null) {
            self.codeBegin();
            self.codeByte(byte);
            return;
        }
        self.write(&.{byte});
        v.at_line_start = byte == '\n';
    }
    fn restoreParamColor(self: Stream) void {
        const v = &self.record.viz;
        if (v.active and v.param_active and !v.read_style) self.renderer().color(paramColor(v.param_kind));
    }
    // Project large write/edit parameters immediately while withholding only
    // a possible closing tag, including forms split across arbitrary tokens.
    fn paramByte(self: Stream, byte: u8) void {
        const v = &self.record.viz;
        if (v.param_end_len != 0 or byte == '<') {
            if (v.param_end_len == v.param_end_tail.len) {
                const keep = v.param_end_len;
                v.param_end_len = 0;
                for (v.param_end_tail[0..keep]) |held| self.paramRaw(held);
                if (byte != '<') {
                    self.paramRaw(byte);
                    return;
                }
            }
            if (v.param_end_len < v.param_end_tail.len) {
                v.param_end_tail[v.param_end_len] = byte;
                v.param_end_len += 1;
            }
            var complete = false;
            if (c.ds4_agent_tool_value_close_tail(self.parser().syntax, &v.param_end_tail, v.param_end_len, &complete)) {
                if (complete) self.paramEnd();
                return;
            }
            const keep = v.param_end_len;
            v.param_end_len = 0;
            for (v.param_end_tail[0..keep]) |held| self.paramRaw(held);
            return;
        }
        self.paramRaw(byte);
    }
    fn finishViz(self: Stream, status: []const u8) void {
        const v = &self.record.viz;
        if (!v.active) return;
        if (v.param_active) self.paramEnd();
        if (status.len == 0) self.renderRead();
        if (status.len != 0) {
            if (!v.last_output_newline) self.write("\n");
            self.renderer().color("\x1b[90m");
            self.write(status);
            self.renderer().color("\x1b[0m");
        }
        if (!v.last_output_newline) self.write("\n");
        v.active = false;
    }
    fn dumpInvalid(self: Stream) void {
        const v = &self.record.viz;
        if (!v.active) return;
        // Friendly projection is not evidence. On failure show the exact raw
        // stanza so the next fix can inspect what the parser really rejected.
        if (v.param_active) {
            v.param_active = false;
            v.param_end_len = 0;
            v.param_name[0] = 0;
        }
        if (!v.last_output_newline) self.write("\n");
        self.renderer().color("\x1b[1;31m");
        const p = self.parser();
        if (p.raw != null and p.raw_len != 0) self.write(p.raw[0..p.raw_len]) else self.write("<empty DSML>");
        self.renderer().color("\x1b[0m");
        if (!v.last_output_newline) self.write("\n");
    }
    fn ignored(self: Stream, detail: [*:0]const u8) void {
        const state = self.record;
        state.dsml_in_think = true;
        state.dsml_in_think_reported = true;
        self.trace("dsml ignored inside thinking: ", detail);
        if (!self.renderer().record.last_output_newline) self.renderer().plain("\n");
        self.renderer().color("\x1b[1;31m");
        self.renderer().plain("[tool call ignored: ");
        self.renderer().plain(std.mem.span(detail));
        self.renderer().plain("]\n");
        self.renderer().color("\x1b[0m");
        c.ds4_agent_tool_parser_reset(state.parser);
        state.dsml_active = false;
        state.dsml_ignored = false;
    }
    fn malformed(self: Stream) void {
        const p = self.parser();
        if (p.state == invalid) return;
        const message = "DSML markup outside a valid tool_calls block";
        c.ds4_agent_tool_parser_error(p, message);
        self.trace("malformed dsml in assistant output: ", message);
        if (!self.renderer().record.last_output_newline) self.renderer().plain("\n");
        self.renderer().color("\x1b[1;31m");
        self.renderer().plain("[invalid tool call: ");
        self.renderer().plain(message);
        self.renderer().plain("]\n");
        self.renderer().color("\x1b[0m");
    }
    // Parser progress is the source of truth; events decide only what to show.
    fn events(self: Stream) void {
        const v = &self.record.viz;
        const p = self.parser();
        if (!v.tool_announced and p.current.name != null) self.tool(p.current.name);
        if (v.tool_announced and p.current.name == null and !v.param_active) {
            self.renderRead();
            if (!v.last_output_newline) self.write("\n");
            v.read_style = false;
            v.read_prefix_rendered = false;
            v.read_line_rendered = false;
            v.read_path[0] = 0;
            v.read_start[0] = 0;
            v.read_max[0] = 0;
            v.read_whole[0] = 0;
            v.tool_announced = false;
        }
        if (!v.param_active and p.state == value_state and p.param_name != null) self.paramBegin(p.param_name);
    }
    fn preflightClosed(self: Stream) void {
        const state = self.record;
        if (state.replay or state.dsml_ignored or state.tool_preflight_error) return;
        const p = self.parser();
        const v = &state.viz;
        if (!v.param_active or !same(text(&v.param_name), "old") or
            p.current.name == null or !same(text(p.current.name), "edit")) return;
        var error_text: [256]u8 = @splat(0);
        if (c.ds4_agent_stream_preflight(self.renderer().record.worker, &p.current, &error_text, error_text.len)) return;
        state.tool_preflight_error = true;
        const detail: [*c]const u8 = if (error_text[0] != 0) &error_text else "old text is not a unique match";
        _ = c.snprintf(&state.tool_preflight_error_msg, state.tool_preflight_error_msg.len, "edit old selector failed before new was generated: %s", detail);
        self.trace("edit old preflight failed: ", &state.tool_preflight_error_msg);
    }
    fn feed(self: Stream, byte: u8) void {
        const state = self.record;
        const p = self.parser();
        const was_param = !state.dsml_ignored and state.viz.param_active;
        c.ds4_agent_tool_parser_feed(p, &byte, 1);
        if (!state.dsml_ignored) {
            self.events();
            if (was_param) self.paramByte(byte);
            if (was_param and p.state != value_state and state.viz.param_active) {
                self.preflightClosed();
                self.paramEnd();
            }
        }
        if (p.state == done) {
            if (state.dsml_ignored) {
                self.ignored("tool calling is not allowed inside <think></think>");
            } else {
                self.traceDone();
                self.finishViz(&.{});
                state.dsml_active = false;
            }
        } else if (p.state == invalid) {
            if (state.dsml_ignored) {
                self.ignored("malformed tool call inside <think></think>");
            } else {
                const error_text: [*c]const u8 = if (p.@"error"[0] != 0) &p.@"error" else "parse error";
                var status: [220]u8 = undefined;
                _ = c.snprintf(&status, status.len, "[invalid tool call: %s]\n", error_text);
                self.trace("dsml error ", error_text);
                self.dumpInvalid();
                self.finishViz(text(&status));
                state.dsml_active = false;
            }
        }
    }
    // Tolerated opening typos seed canonical parser bytes; all parsing after
    // detection stays strict. A direct invoke acts as an implicit calls block.
    fn startDsml(self: Stream, ignored_block: bool) void {
        const state = self.record;
        state.dsml_active = true;
        state.dsml_ignored = ignored_block;
        if (ignored_block) state.dsml_in_think = true;
        state.dsml_start_len = 0;
        state.post_think_gap = false;
        var message: [96]u8 = undefined;
        const name: [*:0]const u8 = if (state.syntax == c.AGENT_TOOL_SYNTAX_QWEN) "qwen" else if (state.syntax == c.AGENT_TOOL_SYNTAX_GLM) "glm" else "dsml";
        _ = c.snprintf(&message, message.len, "%s tool start detected%s", name, @as([*:0]const u8, if (ignored_block) " inside thinking" else ""));
        c.ds4_agent_stream_trace(self.renderer().record.worker, &message);
        c.ds4_agent_tool_parser_start(state.parser);
        if (!ignored_block) {
            self.start();
            self.events();
        }
    }
    fn flushStart(self: Stream) void {
        const state = self.record;
        if (state.dsml_start_len == 0) return;
        state.post_think_gap = false;
        for (state.dsml_start_tail[0..state.dsml_start_len]) |byte| {
            self.renderer().character(byte);
            self.notePlain(byte);
            if (self.parser().state == invalid) break;
        }
        state.dsml_start_len = 0;
    }
    fn noteThink(self: Stream, byte: u8) void {
        const state = self.record;
        if (state.in_think and !state.dsml_in_think and marker(&state.think_dsml, byte)) state.dsml_in_think = true;
    }
    fn notePlain(self: Stream, byte: u8) void {
        const state = self.record;
        if (self.parser().state == invalid or state.dsml_active or state.in_think or state.dsml_in_think) return;
        if (marker(&state.plain_dsml, byte)) self.malformed();
    }
    // Hold only short possible DSML openings, never whole assistant output.
    fn normal(self: Stream, byte: u8) void {
        const state = self.record;
        const opening = text(c.ds4_agent_tool_start(state.syntax));
        const invoke = if (state.syntax == c.AGENT_TOOL_SYNTAX_DSML41) "<｜DSML｜ invoke" else "<｜DSML｜invoke";
        if (self.parser().state == invalid) return;
        self.noteThink(byte);
        // Suppress only the whitespace window immediately after </think>.
        // Once prose starts, even ordinary subsequent blank lines are visible.
        if (state.post_think_gap and state.dsml_start_len == 0 and
            std.mem.indexOfScalar(u8, " \t\r\n", byte) != null) return;
        if (state.dsml_start_len != 0 or byte == opening[0]) {
            if (state.dsml_start_len < state.dsml_start_tail.len) {
                state.dsml_start_tail[state.dsml_start_len] = byte;
                state.dsml_start_len += 1;
            }
            if (startMatch(state.syntax, state.dsml_start_tail[0..state.dsml_start_len])) |match| {
                if (match.complete) {
                    self.startDsml(state.in_think);
                    if (state.syntax != c.AGENT_TOOL_SYNTAX_GLM and match.implicit_invoke) {
                        for (invoke) |part| self.feed(part);
                    }
                }
                return;
            }
            if (state.dsml_start_len > 1 and state.dsml_start_tail[state.dsml_start_len - 1] == opening[0]) {
                state.post_think_gap = false;
                for (state.dsml_start_tail[0 .. state.dsml_start_len - 1]) |part| {
                    self.renderer().character(part);
                    self.notePlain(part);
                    if (self.parser().state == invalid) break;
                }
                if (self.parser().state == invalid) {
                    state.dsml_start_len = 0;
                    return;
                }
                state.dsml_start_tail[0] = opening[0];
                state.dsml_start_len = 1;
                return;
            }
            self.flushStart();
            return;
        }
        state.post_think_gap = false;
        self.renderer().character(byte);
        self.notePlain(byte);
    }
    fn consume(self: Stream, bytes: []const u8, finish: bool) void {
        const state = self.record;
        const count = state.pending_len + bytes.len;
        const memory = c.malloc(@max(count, 1)) orelse {
            c.perror("ds4-agent: malloc");
            c.exit(1);
        };
        defer c.free(memory);
        const buffer: []u8 = @as([*]u8, @ptrCast(memory))[0..count];
        @memcpy(buffer[0..state.pending_len], state.pending[0..state.pending_len]);
        @memcpy(buffer[state.pending_len..], bytes);
        state.pending_len = 0;
        // A prompt redraw can reset terminal attributes between chunks. Make
        // every fragment self-contained before projecting its visible bytes.
        if (bytes.len != 0) self.restoreParamColor();
        if (bytes.len != 0 and !state.dsml_active) self.renderer().restore();
        var position: usize = 0;
        while (position < buffer.len) {
            const rest = buffer[position..];
            if (!state.dsml_active and std.mem.startsWith(u8, rest, "<think>")) {
                self.flushStart();
                self.renderer().hintEnd();
                state.post_think_gap = false;
                state.in_think = true;
                self.renderer().record.in_think = true;
                position += "<think>".len;
                continue;
            }
            if (!state.dsml_active and std.mem.startsWith(u8, rest, "</think>")) {
                self.flushStart();
                self.renderer().hintEnd();
                state.in_think = false;
                self.renderer().record.in_think = false;
                self.renderer().reset();
                if (!self.renderer().record.last_output_newline) self.renderer().write("\n");
                self.renderer().write("\n");
                self.renderer().record.last_output_newline = true;
                state.post_think_gap = true;
                position += "</think>".len;
                continue;
            }
            if (!finish and !state.dsml_active and rest[0] == '<' and
                (partial(rest, "<think>") or partial(rest, "</think>")))
            {
                if (rest.len < state.pending.len) {
                    @memcpy(state.pending[0..rest.len], rest);
                    state.pending_len = rest.len;
                }
                break;
            }
            if (state.dsml_active) self.feed(rest[0]) else self.normal(rest[0]);
            position += 1;
        }
    }
    pub fn writeText(self: Stream, bytes: []const u8, finish: bool) void {
        self.consume(bytes, finish);
        if (!finish) return;
        const state = self.record;
        self.flushStart();
        state.post_think_gap = false;
        if (state.dsml_active) c.ds4_agent_tool_parser_finish(state.parser);
        if (state.dsml_active) {
            if (self.parser().state == done) {
                if (state.dsml_ignored) {
                    self.ignored("tool calling is not allowed inside <think></think>");
                } else {
                    self.traceDone();
                    self.finishViz(&.{});
                    state.dsml_active = false;
                }
            } else if (state.dsml_ignored) {
                self.ignored("tool calling is not allowed inside <think></think>");
            } else {
                self.finishViz(if (state.tool_preflight_error) "[tool call stopped: edit old selector failed]\n" else "[tool call interrupted]\n");
                state.dsml_active = false;
            }
        }
        if (state.dsml_in_think and !state.dsml_in_think_reported) self.ignored("tool calling is not allowed inside <think></think>");
    }
};

fn partial(bytes: []const u8, prefix: []const u8) bool {
    return bytes.len < prefix.len and std.mem.startsWith(u8, prefix, bytes);
}
const Match = struct { complete: bool, implicit_invoke: bool = false };
fn startMatch(syntax: c.agent_tool_syntax, tail: []const u8) ?Match {
    if (xmlSyntax(syntax)) {
        const open = "<tool_call>";
        return if (tail.len <= open.len and std.mem.startsWith(u8, open, tail)) .{ .complete = tail.len == open.len } else null;
    }
    const v41 = syntax == c.AGENT_TOOL_SYNTAX_DSML41;
    const forms = [_]struct { bytes: []const u8, implicit: bool }{
        .{ .bytes = text(c.ds4_agent_tool_start(syntax)), .implicit = false },
        .{ .bytes = if (v41) "<DSML｜ calls>" else "<DSML｜tool_calls>", .implicit = false },
        .{ .bytes = if (v41) "<｜DSML｜ invoke" else "<｜DSML｜invoke", .implicit = true },
        .{ .bytes = if (v41) "<DSML｜ invoke" else "<DSML｜invoke", .implicit = true },
    };
    for (forms) |form| {
        if (tail.len <= form.bytes.len and std.mem.startsWith(u8, form.bytes, tail))
            return .{ .complete = tail.len == form.bytes.len, .implicit_invoke = form.implicit };
    }
    return null;
}
// Marker detection has no policy: in thinking it records an early tool attempt;
// outside thinking a stray DSML marker becomes a recoverable parser error.
fn marker(detector: *c.agent_dsml_marker_detector, byte: u8) bool {
    if (detector.len == detector.tail.len) {
        std.mem.copyForwards(u8, detector.tail[0 .. detector.tail.len - 1], detector.tail[1..]);
        detector.len -= 1;
    }
    detector.tail[detector.len] = byte;
    detector.len += 1;
    const tail = detector.tail[0..detector.len];
    for ([_][]const u8{ "｜DSML｜", "|DSML|", "<DSML｜", "</DSML｜" }) |needle|
        if (std.mem.endsWith(u8, tail, needle)) return true;
    return false;
}

/// Grammar bytes are greedy, argument values retain normal sampling. A possible
/// partial close is grammar only after the parser identifies it. All state is
/// local to one round, so EOS/cancellation cannot leave later rounds greedy.
pub fn wantsGreedy(maybe_state: ?*const Record) bool {
    const state = maybe_state orelse return false;
    if (state.parser == null) return false;
    const p = state.parser;
    if (p.*.state == invalid or p.*.state == done) return false;
    // A single '<' is common prose/code; the second matching byte is specific.
    if (state.dsml_start_len > 1) return true;
    if (!state.dsml_active) return false;
    if (p.*.state == structural) return true;
    return p.*.state == value_state and p.*.param_close_prefix;
}
pub fn needsLookahead(state: *const Record) bool {
    return !state.dsml_active and state.parser.*.state == search and
        (state.pending_len != 0 or state.dsml_start_len != 0);
}
