//! Incremental sampling policy over the actual sampled bytes. Syntax is greedy
//! while string payloads keep their requested sampling. Malformed DSML remains
//! the completed-reply parser's responsibility; this recognizer never repairs
//! or rewrites the model transcript.
const std = @import("std");
const c = @import("server_decode_c");
const allocator = std.heap.c_allocator;
const Tracker = c.dsml_decode_tracker;
const Syntax = c.dsml_syntax;

const syntaxes = [_]Syntax{
    .{ .tool_calls_start = c.DS41_TOOL_CALLS_START, .tool_calls_end = c.DS41_TOOL_CALLS_END, .invoke_start = c.DS41_INVOKE_START, .invoke_end = c.DS41_INVOKE_END, .param_start = c.DS41_PARAM_START, .param_end = c.DS41_PARAM_END },
    .{ .tool_calls_start = c.DS4_TOOL_CALLS_START, .tool_calls_end = c.DS4_TOOL_CALLS_END, .invoke_start = c.DS4_INVOKE_START, .invoke_end = c.DS4_INVOKE_END, .param_start = c.DS4_PARAM_START, .param_end = c.DS4_PARAM_END },
    .{ .tool_calls_start = c.DS4_TOOL_CALLS_START_SHORT, .tool_calls_end = c.DS4_TOOL_CALLS_END_SHORT, .invoke_start = c.DS4_INVOKE_START_SHORT, .invoke_end = c.DS4_INVOKE_END_SHORT, .param_start = c.DS4_PARAM_START_SHORT, .param_end = c.DS4_PARAM_END_SHORT },
    .{ .tool_calls_start = "<tool_calls>", .tool_calls_end = "</tool_calls>", .invoke_start = "<invoke", .invoke_end = "</invoke>", .param_start = "<parameter", .param_end = "</parameter>" },
};
const glm_syntax: Syntax = .{ .tool_calls_start = "<tool_call>", .tool_calls_end = "</tool_call>", .invoke_start = "<arg_key>", .invoke_end = "</arg_key>", .param_start = "<arg_value>", .param_end = "</arg_value>" };
const qwen_syntax: Syntax = .{ .tool_calls_start = "<tool_call>", .tool_calls_end = "</tool_call>", .invoke_start = "<function=", .invoke_end = "</function>", .param_start = "<parameter=", .param_end = "</parameter>" };

fn text(pointer: [*c]const u8) []const u8 {
    return std.mem.span(pointer);
}
fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
    c.exit(1);
}
fn at(raw: []const u8, offset: usize, literal: [*c]const u8) bool {
    return offset <= raw.len and std.mem.startsWith(u8, raw[offset..], text(literal));
}
fn partial(raw: []const u8, offset: usize, literal: [*c]const u8) bool {
    const expected = text(literal);
    return offset <= raw.len and raw.len - offset < expected.len and
        std.mem.startsWith(u8, expected, raw[offset..]);
}
fn suffixPartial(raw: []const u8, literal: []const u8, minimum: usize) bool {
    if (raw.len == 0 or literal.len == 0) return false;
    const maximum = @min(raw.len, literal.len - 1);
    var length = minimum;
    while (length <= maximum) : (length += 1) {
        if (std.mem.eql(u8, raw[raw.len - length ..], literal[0..length])) return true;
    }
    return false;
}
fn maximumStart() usize {
    var longest: usize = 0;
    for (&syntaxes) |*syntax| longest = @max(longest, text(syntax.tool_calls_start).len);
    return longest;
}
const Start = struct { after: usize, syntax: *const Syntax };
fn findStart(raw: []const u8, offset: usize, family: c.server_model_syntax) ?Start {
    if (offset > raw.len) return null;
    if (family == c.SERVER_MODEL_SYNTAX_GLM or family == c.SERVER_MODEL_SYNTAX_QWEN) {
        const syntax = if (family == c.SERVER_MODEL_SYNTAX_QWEN) &qwen_syntax else &glm_syntax;
        const found = std.mem.findPos(u8, raw, offset, text(syntax.tool_calls_start)) orelse return null;
        return .{ .after = found + text(syntax.tool_calls_start).len, .syntax = syntax };
    }
    var best: ?Start = null;
    var best_position = raw.len;
    for (&syntaxes) |*syntax| {
        if (std.mem.findPos(u8, raw, offset, text(syntax.tool_calls_start))) |found| {
            if (best == null or found < best_position) {
                best_position = found;
                best = .{ .after = found + text(syntax.tool_calls_start).len, .syntax = syntax };
            }
        }
    }
    return best;
}
fn stringAttribute(raw: []const u8, start: usize, end: usize) bool {
    if (end <= start or end > raw.len) return false;
    const tag = allocator.dupeSentinel(u8, raw[start..end], 0) catch fatal();
    defer allocator.free(tag);
    const value = c.ds4_server_reply_attribute(tag.ptr, "string");
    defer c.free(value);
    return value != null and std.mem.eql(u8, text(value), "true");
}
pub fn init(tracker: *Tracker) void {
    tracker.* = std.mem.zeroes(Tracker);
    tracker.mode = c.DSML_TRACK_SEARCH;
    tracker.decode = c.DSML_DECODE_OUTSIDE;
}
pub fn isTool(state: c.dsml_decode_state) bool {
    return state != c.DSML_DECODE_OUTSIDE;
}
pub fn isPayload(state: c.dsml_decode_state) bool {
    return state == c.DSML_DECODE_STRING_BODY or state == c.DSML_DECODE_JSON_STRING;
}

/// Feed monotonically extended sampled text. A partial closing marker retains
/// its cursor for the next update. Even a lone '<' is retained without making
/// ordinary payload code/HTML switch to greedy structural sampling.
pub fn update(tracker: *Tracker, raw: []const u8) void {
    while (true) {
        switch (tracker.mode) {
            c.DSML_TRACK_DONE => {
                tracker.decode = c.DSML_DECODE_OUTSIDE;
                return;
            },
            c.DSML_TRACK_SEARCH => {
                const found = findStart(raw, tracker.pos, tracker.model_syntax) orelse {
                    const hold = maximumStart();
                    tracker.pos = if (raw.len > hold) raw.len - hold else 0;
                    tracker.decode = c.DSML_DECODE_OUTSIDE;
                    return;
                };
                tracker.syn = found.syntax;
                tracker.pos = found.after;
                tracker.mode = c.DSML_TRACK_STRUCTURAL;
                tracker.decode = c.DSML_DECODE_STRUCTURAL;
            },
            c.DSML_TRACK_STRING_BODY => {
                const syntax = tracker.syn.*;
                while (tracker.pos < raw.len) {
                    if (at(raw, tracker.pos, syntax.param_end)) {
                        tracker.pos += text(syntax.param_end).len;
                        tracker.mode = c.DSML_TRACK_STRUCTURAL;
                        tracker.decode = c.DSML_DECODE_STRUCTURAL;
                        break;
                    }
                    if (partial(raw, tracker.pos, syntax.param_end)) {
                        tracker.decode = if (raw.len - tracker.pos >= 2) c.DSML_DECODE_STRUCTURAL else c.DSML_DECODE_STRING_BODY;
                        return;
                    }
                    tracker.pos += 1;
                }
                if (tracker.mode == c.DSML_TRACK_STRING_BODY) {
                    tracker.decode = c.DSML_DECODE_STRING_BODY;
                    return;
                }
            },
            c.DSML_TRACK_JSON_PARAM => {
                const syntax = tracker.syn.*;
                while (tracker.pos < raw.len) {
                    if (!tracker.json_in_string) {
                        if (at(raw, tracker.pos, syntax.param_end)) {
                            tracker.pos += text(syntax.param_end).len;
                            tracker.mode = c.DSML_TRACK_STRUCTURAL;
                            tracker.decode = c.DSML_DECODE_STRUCTURAL;
                            break;
                        }
                        if (partial(raw, tracker.pos, syntax.param_end)) {
                            tracker.decode = if (raw.len - tracker.pos >= 2) c.DSML_DECODE_STRUCTURAL else c.DSML_DECODE_JSON_STRUCTURAL;
                            return;
                        }
                    }
                    const character = raw[tracker.pos];
                    tracker.pos += 1;
                    if (tracker.json_in_string) {
                        if (tracker.json_escaped) {
                            tracker.json_escaped = false;
                        } else if (character == '\\') {
                            tracker.json_escaped = true;
                        } else if (character == '"') {
                            tracker.json_in_string = false;
                        }
                    } else if (character == '"') {
                        tracker.json_in_string = true;
                    }
                }
                if (tracker.mode == c.DSML_TRACK_JSON_PARAM) {
                    tracker.decode = if (tracker.json_in_string) c.DSML_DECODE_JSON_STRING else c.DSML_DECODE_JSON_STRUCTURAL;
                    return;
                }
            },
            c.DSML_TRACK_STRUCTURAL => {
                const syntax = tracker.syn.*;
                while (tracker.pos < raw.len and c.isspace(raw[tracker.pos]) != 0) tracker.pos += 1;
                if (tracker.pos >= raw.len) {
                    tracker.decode = c.DSML_DECODE_STRUCTURAL;
                    return;
                }
                if (at(raw, tracker.pos, syntax.tool_calls_end)) {
                    tracker.mode = c.DSML_TRACK_DONE;
                    tracker.pos += text(syntax.tool_calls_end).len;
                    tracker.decode = c.DSML_DECODE_OUTSIDE;
                    return;
                }
                if (at(raw, tracker.pos, syntax.invoke_end)) {
                    tracker.pos += text(syntax.invoke_end).len;
                    continue;
                }
                if (at(raw, tracker.pos, syntax.invoke_start)) {
                    const end = std.mem.findScalarPos(u8, raw, tracker.pos, '>') orelse {
                        tracker.decode = c.DSML_DECODE_STRUCTURAL;
                        return;
                    };
                    tracker.pos = end + 1;
                    continue;
                }
                if (at(raw, tracker.pos, syntax.param_start)) {
                    const end = std.mem.findScalarPos(u8, raw, tracker.pos, '>') orelse {
                        tracker.decode = c.DSML_DECODE_STRUCTURAL;
                        return;
                    };
                    const is_string = tracker.model_syntax == c.SERVER_MODEL_SYNTAX_GLM or tracker.model_syntax == c.SERVER_MODEL_SYNTAX_QWEN or stringAttribute(raw, tracker.pos, end + 1);
                    tracker.pos = end + 1;
                    if (is_string) {
                        tracker.mode = c.DSML_TRACK_STRING_BODY;
                        tracker.decode = c.DSML_DECODE_STRING_BODY;
                    } else {
                        tracker.mode = c.DSML_TRACK_JSON_PARAM;
                        tracker.json_in_string = false;
                        tracker.json_escaped = false;
                        tracker.decode = c.DSML_DECODE_JSON_STRUCTURAL;
                    }
                    continue;
                }
                if (partial(raw, tracker.pos, syntax.tool_calls_end) or partial(raw, tracker.pos, syntax.invoke_start) or partial(raw, tracker.pos, syntax.invoke_end) or partial(raw, tracker.pos, syntax.param_start) or partial(raw, tracker.pos, syntax.param_end)) {
                    tracker.decode = c.DSML_DECODE_STRUCTURAL;
                    return;
                }
                if (tracker.model_syntax == c.SERVER_MODEL_SYNTAX_GLM) {
                    // GLM puts function and argument names between tags.
                    tracker.pos += 1;
                    continue;
                }
                tracker.decode = c.DSML_DECODE_STRUCTURAL;
                return;
            },
            else => return,
        }
    }
}

pub fn observe(tracker: *const Tracker, scan: [*c]const u8, saw_start: *bool, saw_end: *bool, orphan: ?*bool) void {
    if (tracker.syn != null) saw_start.* = true;
    if (tracker.mode == c.DSML_TRACK_DONE) saw_end.* = true else if (!saw_start.* and scan != null and c.ds4_server_reply_tool_end(scan) != null) {
        if (orphan) |flag| flag.* = true;
    }
}

/// Generation deliberately ignores DSML while thinking. Check its current raw
/// reasoning too, so a soft reasoning cutoff never splits a tool/think marker.
pub fn boundarySafe(sampled: *const c.buf, tracker: *const Tracker) bool {
    if (tracker.decode != c.DSML_DECODE_OUTSIDE or tracker.mode != c.DSML_TRACK_SEARCH) return false;
    const raw = if (sampled.ptr == null) "" else sampled.ptr[0..sampled.len];
    var probe: Tracker = undefined;
    init(&probe);
    probe.model_syntax = tracker.model_syntax;
    update(&probe, raw);
    if (isTool(probe.decode)) return false;
    for (&syntaxes) |*syntax| {
        const markers = .{ syntax.tool_calls_start, syntax.tool_calls_end, syntax.invoke_start, syntax.invoke_end, syntax.param_start, syntax.param_end };
        inline for (markers) |marker| if (suffixPartial(raw, text(marker), 1)) return false;
    }
    return !suffixPartial(raw, "<think>", 1) and !suffixPartial(raw, "</think>", 1);
}
