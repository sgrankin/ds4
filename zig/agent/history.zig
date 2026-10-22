//! Saved transcript presentation only. Replay uses the same streaming renderer
//! as live output, but never executes tools or changes transcript/KV state.
const std = @import("std");
const c = @import("agent_presentation_c");
const presentation = @import("presentation.zig");
const streaming = @import("stream.zig");
const allocator = std.heap.c_allocator;
const max_turns = c.DS4_AGENT_HISTORY_MAX_TURNS;
const assistant_lines = c.DS4_AGENT_HISTORY_ASSISTANT_MAX_LINES;
const assistant_bytes = c.DS4_AGENT_HISTORY_ASSISTANT_MAX_BYTES;
const user_marker = "<｜User｜>";
const assistant_marker = "<｜Assistant｜>";
const eos_marker = "<｜end▁of▁sentence｜>";
const summary_start = "[ds4-agent compacted earlier conversation. Durable task-state summary follows.]";
const summary_end = "[End compacted summary. Recent conversation continues verbatim below.]";

const Kind = enum { user, assistant, eos };
const Marker = struct { kind: Kind, position: usize, length: usize };
const Event = struct { marker: Marker, content: []const u8, human: bool };

fn nextMarker(bytes: []const u8, position: usize) ?Marker {
    var result: ?Marker = null;
    const forms = [_]struct { kind: Kind, bytes: []const u8 }{
        .{ .kind = .user, .bytes = user_marker },
        .{ .kind = .assistant, .bytes = assistant_marker },
        .{ .kind = .eos, .bytes = eos_marker },
    };
    for (forms) |form| {
        if (std.mem.findPos(u8, bytes, position, form.bytes)) |offset| {
            if (result == null or offset < result.?.position)
                result = .{ .kind = form.kind, .position = offset, .length = form.bytes.len };
        }
    }
    return result;
}
fn trim(bytes: []const u8) []const u8 {
    var first: usize = 0;
    var end = bytes.len;
    while (first < end and c.isspace(bytes[first]) != 0) first += 1;
    while (end > first and c.isspace(bytes[end - 1]) != 0) end -= 1;
    return bytes[first..end];
}

// Tool results are stored as user-role turns. Skip these pseudo-user turns
// when counting human conversation, and display their payload without wrappers.
fn toolPayload(bytes: []const u8) ?[]const u8 {
    const value = trim(bytes);
    const open = "<tool_result>";
    const close = "</tool_result>";
    if (!std.mem.startsWith(u8, value, open)) return null;
    const payload = value[open.len..];
    return if (std.mem.endsWith(u8, payload, close)) payload[0 .. payload.len - close.len] else payload;
}
fn toolUser(bytes: []const u8) bool {
    const value = trim(bytes);
    return toolPayload(value) != null or std.mem.startsWith(u8, value, "Tool:") or
        std.mem.startsWith(u8, value, "Tool result");
}
fn collect(bytes: []const u8) std.ArrayList(Event) {
    var events: std.ArrayList(Event) = .empty;
    var position: usize = 0;
    while (nextMarker(bytes, position)) |marker| {
        const content_start = marker.position + marker.length;
        const next = nextMarker(bytes, content_start);
        const content_end = if (next) |found| found.position else bytes.len;
        const content = bytes[content_start..content_end];
        events.append(allocator, .{ .marker = marker, .content = content, .human = marker.kind == .user and !toolUser(content) }) catch {
            c.perror("ds4-agent: realloc");
            c.exit(1);
        };
        position = content_end;
    }
    return events;
}
const Start = struct { index: usize, tool_only: bool = false };

// Compacted tails may contain only tool/assistant events. Include the assistant
// call before the oldest retained tool result rather than showing an empty tail
// or hiding the call that caused that result.
fn startForTurns(events: []const Event, turns: usize) Start {
    var humans: usize = 0;
    var users: usize = 0;
    for (events) |event| {
        if (event.human) humans += 1;
        if (event.marker.kind == .user) users += 1;
    }
    if (users == 0) return .{ .index = events.len };
    var skip = if (humans != 0) humans -| turns else users -| turns;
    var selected: usize = 0;
    for (events, 0..) |event, index| {
        if (if (humans != 0) event.human else event.marker.kind == .user) {
            if (skip == 0) {
                selected = index;
                break;
            }
            skip -= 1;
        }
    }
    if (humans != 0) return .{ .index = selected };
    var index = selected;
    while (index != 0) {
        index -= 1;
        switch (events[index].marker.kind) {
            .user => break,
            .assistant => {
                selected = index;
                break;
            },
            .eos => {},
        }
    }
    return .{ .index = selected, .tool_only = true };
}
fn latestSummary(bytes: []const u8) ?[]const u8 {
    var position: usize = 0;
    var latest: ?[]const u8 = null;
    while (std.mem.findPos(u8, bytes, position, summary_start)) |start| {
        const content_start = start + summary_start.len;
        const end = std.mem.findPos(u8, bytes, content_start, summary_end) orelse break;
        latest = bytes[content_start..end];
        position = end + summary_end.len;
    }
    const value = trim(latest orelse return null);
    return if (value.len != 0) value else null;
}
const Tail = struct { bytes: []const u8, truncated: bool };
fn tail(bytes: []const u8, max_lines: usize, max_bytes: usize) Tail {
    if (bytes.len == 0) return .{ .bytes = bytes, .truncated = false };
    var start: usize = 0;
    var truncated = false;
    if (max_bytes != 0 and bytes.len > max_bytes) {
        start = bytes.len - max_bytes;
        truncated = true;
    }
    if (max_lines != 0) {
        var position = bytes.len;
        if (bytes[position - 1] == '\n') position -= 1;
        var lines: usize = 0;
        while (position != 0) {
            position -= 1;
            if (bytes[position] == '\n') {
                lines += 1;
                if (lines == max_lines) {
                    truncated = true;
                    start = @max(start, position + 1);
                    break;
                }
            }
        }
    }
    // A byte budget can land inside a multibyte character. Retain only complete
    // leading UTF8 units, without changing the original truncation decision.
    while (start < bytes.len and bytes[start] & 0xc0 == 0x80) start += 1;
    return .{ .bytes = bytes[start..], .truncated = truncated };
}

pub const History = struct {
    engine: ?*c.ds4_engine,
    syntax: c.agent_tool_syntax,
    context: ?*anyopaque,
    emit: c.ds4_agent_presentation_emit_fn,
    use_color: bool,

    fn write(self: History, bytes: []const u8) void {
        if (self.emit) |output| output(self.context, bytes.ptr, bytes.len);
    }
    fn limited(self: History, bytes: []const u8, lines: usize, limit: usize) void {
        const selected = tail(bytes, lines, limit);
        if (selected.truncated) self.write("\n... earlier history truncated; showing tail ...\n");
        self.write(selected.bytes);
        if (selected.bytes.len != 0 and selected.bytes[selected.bytes.len - 1] != '\n') self.write("\n");
    }
    fn summary(self: History, bytes: []const u8) void {
        const value = latestSummary(bytes) orelse return;
        self.write(if (self.use_color) "\n\x1b[1;95mCompacted Summary:\x1b[0m\n" else "\nCompacted Summary:\n");
        self.limited(value, 80, 12000);
    }
    fn assistant(self: History, bytes: []const u8) void {
        const value = trim(bytes);
        if (value.len == 0) return;
        const source_truncated = tail(value, assistant_lines, assistant_bytes).truncated;
        var capture = std.mem.zeroes(presentation.Capture);
        capture.cap = if (source_truncated) assistant_bytes else 0;
        var renderer_record = std.mem.zeroes(presentation.Record);
        renderer_record.engine = self.engine;
        renderer_record.worker = self.context;
        renderer_record.emit = self.emit;
        renderer_record.format_thinking = true;
        // History is the original terminal stream, not another transcript
        // format. Tool calls are dry-rendered; saved KV alone governs resume.
        renderer_record.format_markdown = true;
        renderer_record.use_color = self.use_color and !source_truncated;
        renderer_record.last_output_newline = true;
        renderer_record.capture = if (source_truncated) &capture else null;
        var parser = std.mem.zeroes(c.agent_dsml_parser);
        parser.syntax = self.syntax;
        parser.state = c.AGENT_DSML_SEARCH;
        defer c.ds4_agent_tool_parser_free(&parser);
        var stream_record = std.mem.zeroes(streaming.Record);
        stream_record.renderer = &renderer_record;
        stream_record.parser = &parser;
        stream_record.syntax = self.syntax;
        stream_record.replay = true;
        (streaming.Stream{ .record = &stream_record }).writeText(value, true);
        (presentation.Renderer{ .record = &renderer_record }).finish();
        if (source_truncated) {
            var length: usize = 0;
            const rendered = presentation.captureTake(&capture, &length);
            defer c.free(rendered);
            const selected = tail(rendered[0..length], assistant_lines, assistant_bytes);
            if (self.use_color) self.write("\x1b[90m");
            self.write("\n... earlier assistant history truncated; showing tail ...\n");
            self.write(selected.bytes);
            if (length != 0 and rendered[length - 1] != '\n') self.write("\n");
            if (self.use_color) self.write("\x1b[0m");
        }
    }
    pub fn render(self: History, bytes: []const u8, requested_turns: c_int) void {
        if (requested_turns <= 0) return;
        const turns: c_int = @min(requested_turns, max_turns);
        self.summary(bytes);
        var events = collect(bytes);
        defer events.deinit(allocator);
        const selected = startForTurns(events.items, @intCast(turns));
        if (selected.index == events.items.len) {
            self.write("\n(no user history)\n");
            return;
        }
        self.write(if (self.use_color) "\n\x1b[90m" else "\n");
        if (selected.tool_only) {
            self.write("--- session history: recent tool/assistant events ---\n");
        } else {
            var heading: [128]u8 = undefined;
            const length = c.snprintf(&heading, heading.len, "--- session history: last %d user turn%s ---\n", turns, @as([*:0]const u8, if (turns == 1) "" else "s"));
            self.write(heading[0..@intCast(length)]);
        }
        if (self.use_color) self.write("\x1b[0m");
        for (events.items[selected.index..]) |event| {
            const content = trim(event.content);
            switch (event.marker.kind) {
                .user => {
                    if (toolUser(content)) {
                        self.write(if (self.use_color) "\x1b[90mTool result:\n" else "Tool result:\n");
                        self.limited(toolPayload(content) orelse content, 12, 3000);
                        if (self.use_color) self.write("\x1b[0m");
                    } else {
                        self.write(if (self.use_color) "\x1b[1;32mUser:\x1b[0m\n" else "User:\n");
                        self.limited(content, 24, 6000);
                    }
                },
                .assistant => {
                    self.write(if (self.use_color) "\x1b[1;37mAssistant:\x1b[0m\n" else "Assistant:\n");
                    self.assistant(content);
                },
                .eos => {},
            }
        }
        self.write(if (self.use_color) "\x1b[90m--- end history ---\x1b[0m\n" else "--- end history ---\n");
    }
};
