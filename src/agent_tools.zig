//! Streaming DSML/V4.1, GLM and Qwen tool syntax over libc-owned records.
//! Only completed calls move into the output vector. Incomplete input waits;
//! malformed input retains its raw stanza and owners for a retryable error.
const std = @import("std");
const c = @import("agent_tools_c");
const Call = c.agent_tool_call;
const Calls = c.agent_tool_calls;
const Arg = c.agent_tool_arg;
const Parser = c.agent_dsml_parser;
const Syntax = c.agent_tool_syntax;
const search = c.AGENT_DSML_SEARCH;
const structural = c.AGENT_DSML_STRUCTURAL;
const value_state = c.AGENT_DSML_PARAM_VALUE;
const done = c.AGENT_DSML_DONE;
const invalid = c.AGENT_DSML_ERROR;
const whitespace = " \t\r\n";
const bar = "｜";

fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) "" else std.mem.span(pointer);
}
fn span(pointer: [*c]const u8, length: usize) []const u8 {
    return if (length == 0) "" else pointer[0..length];
}
fn visible(bytes: []const u8) []const u8 {
    return std.mem.sliceTo(bytes, 0); // C strstr/strchr stop at an embedded NUL.
}
fn equal(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
fn starts(bytes: []const u8, prefix: []const u8) bool {
    return std.mem.startsWith(u8, bytes, prefix);
}
fn partial(bytes: []const u8, prefix: []const u8) bool {
    return bytes.len < prefix.len and starts(prefix, bytes);
}
fn skipSpace(bytes: []const u8, position: usize) usize {
    var next = position;
    while (next < bytes.len and std.mem.indexOfScalar(u8, whitespace, bytes[next]) != null) next += 1;
    return next;
}
fn oom(message: [*:0]const u8) noreturn {
    c.perror(message);
    c.exit(1);
}
fn duplicate(bytes: []const u8) [*c]u8 {
    const memory = c.malloc(bytes.len + 1) orelse oom("ds4-agent: malloc");
    const output: [*]u8 = @ptrCast(memory);
    @memcpy(output[0..bytes.len], bytes);
    output[bytes.len] = 0;
    return output;
}
fn grow(comptime T: type, pointer: [*c]T, count: usize) [*c]T {
    const memory = c.realloc(pointer, @max(1, count * @sizeOf(T))) orelse oom("ds4-agent: realloc");
    return @ptrCast(@alignCast(memory));
}

export fn ds4_agent_tool_call_free(maybe_call: ?*Call) void {
    const call = maybe_call orelse return;
    c.free(call.name);
    if (call.argc > 0) {
        for (call.args[0..@intCast(call.argc)]) |arg| {
            c.free(arg.name);
            c.free(arg.value);
        }
    }
    c.free(call.args);
    call.* = std.mem.zeroes(Call);
}
export fn ds4_agent_tool_calls_free(maybe_calls: ?*Calls) void {
    const calls = maybe_calls orelse return;
    if (calls.len > 0) {
        for (calls.v[0..@intCast(calls.len)]) |*call| ds4_agent_tool_call_free(call);
    }
    c.free(calls.v);
    calls.* = std.mem.zeroes(Calls);
}
export fn ds4_agent_tool_call_add_arg(call: *Call, name: [*c]const u8, value: [*c]const u8, length: usize, is_string: bool, end_tag: [*c]const u8) void {
    if (call.argc == call.argcap) {
        call.argcap = if (call.argcap == 0) 4 else call.argcap * 2;
        call.args = grow(Arg, call.args, @intCast(call.argcap));
    }
    const arg = &call.args[@intCast(call.argc)];
    arg.* = .{ .name = duplicate(text(name)), .value = duplicate(span(value, length)), .is_string = is_string };
    call.argc += 1;
    if (is_string) c.ds4_tool_text_unescape(arg.value, end_tag);
}
export fn ds4_agent_tool_calls_push(calls: *Calls, call: *Call) void {
    if (call.name == null) return;
    if (calls.len == calls.cap) {
        calls.cap = if (calls.cap == 0) 2 else calls.cap * 2;
        calls.v = grow(Call, calls.v, @intCast(calls.cap));
    }
    calls.v[@intCast(calls.len)] = call.*;
    calls.len += 1;
    call.* = std.mem.zeroes(Call); // Transfer every string/vector owner once.
}
export fn ds4_agent_tool_arg_value(call: *const Call, name: [*c]const u8) [*c]const u8 {
    if (call.argc > 0) {
        for (call.args[0..@intCast(call.argc)]) |arg| {
            if (arg.name != null and equal(text(arg.name), text(name))) return if (arg.value != null) arg.value else "";
        }
    }
    return null;
}
export fn ds4_agent_tool_parser_free(maybe_parser: ?*Parser) void {
    const parser = maybe_parser orelse return;
    const syntax = parser.syntax;
    c.free(parser.raw);
    ds4_agent_tool_call_free(&parser.current);
    c.free(parser.param_name);
    ds4_agent_tool_calls_free(&parser.calls);
    parser.* = std.mem.zeroes(Parser);
    parser.syntax = syntax;
}
export fn ds4_agent_tool_parser_reset(parser: *Parser) void {
    ds4_agent_tool_parser_free(parser);
    parser.state = search;
}
export fn ds4_agent_tool_parser_error(parser: *Parser, message: [*c]const u8) void {
    parser.state = invalid;
    const bytes = text(message);
    const length = @min(bytes.len, parser.@"error".len - 1);
    @memcpy(parser.@"error"[0..length], bytes[0..length]);
    parser.@"error"[length] = 0;
}
fn setError(parser: *Parser, message: [:0]const u8) void {
    ds4_agent_tool_parser_error(parser, message.ptr);
}
export fn ds4_agent_tool_start(syntax: Syntax) [*c]const u8 {
    if (syntax == c.AGENT_TOOL_SYNTAX_GLM or syntax == c.AGENT_TOOL_SYNTAX_QWEN) return "<tool_call>";
    return if (syntax == c.AGENT_TOOL_SYNTAX_DSML41) "<｜DSML｜ calls>" else "<｜DSML｜tool_calls>";
}
export fn ds4_agent_tool_tag_name(syntax: Syntax, name: [*c]const u8) [*c]const u8 {
    if (syntax != c.AGENT_TOOL_SYNTAX_DSML41) return name;
    if (equal(text(name), "tool_calls")) return " calls";
    if (equal(text(name), "invoke")) return " invoke";
    return " parameter";
}
export fn ds4_agent_tool_parameter_close(syntax: Syntax) [*c]const u8 {
    return if (syntax == c.AGENT_TOOL_SYNTAX_DSML41) "</｜DSML｜ parameter>" else "</｜DSML｜parameter>";
}

fn raw(parser: *const Parser) []const u8 {
    return span(parser.raw, parser.raw_len);
}
fn appendRaw(parser: *Parser, bytes: []const u8) void {
    if (bytes.len == 0) return;
    const needed = parser.raw_len + bytes.len + 1;
    if (needed > parser.raw_cap) {
        var capacity = if (parser.raw_cap == 0) @as(usize, 512) else parser.raw_cap * 2;
        while (capacity < needed) capacity *= 2;
        parser.raw = grow(u8, parser.raw, capacity);
        parser.raw_cap = capacity;
    }
    @memcpy(parser.raw[parser.raw_len..][0..bytes.len], bytes);
    parser.raw_len += bytes.len;
    parser.raw[parser.raw_len] = 0;
}
fn attribute(tag: []const u8, comptime name: []const u8) ?[]const u8 {
    const bytes = visible(tag);
    const begin = (std.mem.indexOf(u8, bytes, name ++ "=\"") orelse return null) + name.len + 2;
    const length = std.mem.indexOfScalar(u8, bytes[begin..], '"') orelse return null;
    return bytes[begin..][0..length];
}
fn openTag(tag: []const u8, name: []const u8) bool {
    var buffer: [64]u8 = undefined;
    const prefix = std.mem.printSentinel(&buffer, "<｜DSML｜{s}", .{name}, 0) catch unreachable;
    return starts(tag, prefix) and tag.len > prefix.len and
        (tag[prefix.len] == '>' or std.mem.indexOfScalar(u8, whitespace, tag[prefix.len]) != null);
}
fn closeTag(bytes: []const u8, name: []const u8) ?usize {
    var buffer: [64]u8 = undefined;
    const prefix = std.mem.printSentinel(&buffer, "</｜DSML｜{s}", .{name}, 0) catch unreachable;
    if (!starts(bytes, prefix)) return null;
    var position = skipSpace(bytes, prefix.len);
    if (starts(bytes[position..], bar)) position += bar.len;
    position = skipSpace(bytes, position);
    return if (position < bytes.len and bytes[position] == '>') position + 1 else null;
}
const Closing = struct { start: usize, end: usize };
fn findClose(bytes: []const u8, name: []const u8) ?Closing {
    const input = visible(bytes);
    var position: usize = 0;
    while (std.mem.indexOfPos(u8, input, position, "</｜DSML｜")) |found| {
        if (closeTag(input[found..], name)) |length| return .{ .start = found, .end = found + length };
        position = found + 1;
    }
    return null;
}

// Online rendering must hide a partial close while leaving argument text live.
// DSML accepts whitespace and an optional fullwidth bar only on closing tags.
export fn ds4_agent_tool_value_close_tail(syntax: Syntax, tail: [*c]const u8, length: usize, complete: *bool) bool {
    const bytes = span(tail, length);
    complete.* = false;
    if (syntax == c.AGENT_TOOL_SYNTAX_QWEN or syntax == c.AGENT_TOOL_SYNTAX_GLM) {
        const closing: []const u8 = if (syntax == c.AGENT_TOOL_SYNTAX_QWEN) "</parameter>" else "</arg_value>";
        if (bytes.len > closing.len or !starts(closing, bytes)) return false;
        complete.* = bytes.len == closing.len;
        return true;
    }
    const closing = text(ds4_agent_tool_parameter_close(syntax));
    const prefix = closing[0 .. closing.len - 1];
    if (bytes.len <= prefix.len) return starts(prefix, bytes);
    if (!starts(bytes, prefix)) return false;
    var position = skipSpace(bytes, prefix.len);
    if (position < bytes.len and bytes.len - position <= bar.len and starts(bar, bytes[position..])) return true;
    if (starts(bytes[position..], bar)) position += bar.len;
    while (position < bytes.len) : (position += 1) {
        if (bytes[position] == '>') {
            complete.* = position == bytes.len - 1;
            return complete.*;
        }
        if (std.mem.indexOfScalar(u8, whitespace, bytes[position]) == null) return false;
    }
    return true;
}
fn updateClosePrefix(parser: *Parser) void {
    parser.param_close_prefix = false;
    if (parser.state != value_state or parser.raw_len <= parser.param_value_start) return;
    const value = raw(parser)[parser.param_value_start..];
    const last = std.mem.lastIndexOfScalar(u8, value, '<') orelse return;
    const tail = value[last..];
    if (tail.len > 64) return;
    var complete = false;
    parser.param_close_prefix = ds4_agent_tool_value_close_tail(parser.syntax, tail.ptr, tail.len, &complete) and !complete;
}
fn addValue(parser: *Parser, bytes: []const u8, is_string: bool, closing: [*c]const u8, after: usize) void {
    ds4_agent_tool_call_add_arg(&parser.current, if (parser.param_name != null) parser.param_name else "", bytes.ptr, bytes.len, is_string, closing);
    c.free(parser.param_name);
    parser.param_name = null;
    parser.param_close_prefix = false;
    parser.parse_pos = after;
    parser.state = structural;
}

fn parseGlm(parser: *Parser) void {
    const start = "<tool_call>";
    const close = "</tool_call>";
    const key_open = "<arg_key>";
    const key_close = "</arg_key>";
    const value_open = "<arg_value>";
    const value_close = "</arg_value>";
    if (!starts(raw(parser), start)) return;
    while (parser.state == structural or parser.state == value_state) {
        const input = raw(parser);
        if (parser.state == value_state) {
            const value = input[parser.param_value_start..];
            const end = std.mem.indexOf(u8, visible(value), value_close) orelse return;
            addValue(parser, value[0..end], true, value_close, parser.param_value_start + end + value_close.len);
            continue;
        }
        parser.parse_pos = skipSpace(input, parser.parse_pos);
        if (parser.parse_pos == input.len) return;
        var current = input[parser.parse_pos..];
        if (parser.glm_after_call) {
            if (starts(current, start)) {
                parser.parse_pos += start.len;
                parser.glm_after_call = false;
                continue;
            }
            if (partial(current, start)) return;
            parser.glm_after_call = false;
            parser.state = done;
            return;
        }
        if (parser.current.name == null) {
            const delimiter = std.mem.indexOfScalar(u8, current, '<') orelse return;
            const tag = current[delimiter..];
            if (!starts(tag, key_open) and !starts(tag, close)) {
                if (partial(tag, key_open) or partial(tag, close)) return;
                setError(parser, "expected <arg_key> or </tool_call> in GLM tool call");
                return;
            }
            const name = std.mem.trim(u8, current[0..delimiter], whitespace);
            if (name.len == 0) {
                setError(parser, "GLM tool call without function name");
                return;
            }
            ds4_agent_tool_call_free(&parser.current);
            parser.current.name = duplicate(name);
            parser.parse_pos += delimiter;
            current = input[parser.parse_pos..];
        }
        if (starts(current, close)) {
            parser.parse_pos += close.len;
            ds4_agent_tool_calls_push(&parser.calls, &parser.current);
            parser.glm_after_call = true;
            continue;
        }
        if (partial(current, close)) return;
        if (!starts(current, key_open)) {
            if (partial(current, key_open)) return;
            setError(parser, "expected <arg_key> in GLM tool call");
            return;
        }
        const content = current[key_open.len..];
        const end = std.mem.indexOf(u8, visible(content), key_close) orelse return;
        const name = std.mem.trim(u8, content[0..end], whitespace);
        if (name.len == 0) {
            setError(parser, "empty <arg_key> in GLM tool call");
            return;
        }
        const key = duplicate(name);
        c.ds4_tool_text_unescape(key, key_close);
        const after_key = parser.parse_pos + key_open.len + end + key_close.len;
        const after_space = skipSpace(input, after_key);
        if (!starts(input[after_space..], value_open)) {
            c.free(key);
            if (!partial(input[after_space..], value_open)) setError(parser, "expected <arg_value> in GLM tool call");
            return;
        }
        c.free(parser.param_name);
        parser.param_name = key;
        parser.param_is_string = true;
        parser.param_value_start = after_space + value_open.len;
        parser.parse_pos = parser.param_value_start;
        parser.param_close_prefix = false;
        parser.state = value_state;
    }
}

// This is the existing typing heuristic, not strict JSON validation. Bracketed
// values and libc strtod spellings retain their original accepted grammar.
fn qwenJson(value: []const u8) bool {
    const bytes = std.mem.trimEnd(u8, std.mem.trimStart(u8, value, " \t"), " \t\n");
    if (bytes.len == 0) return false;
    if ((bytes[0] == '{' and bytes[bytes.len - 1] == '}') or (bytes[0] == '[' and bytes[bytes.len - 1] == ']')) return true;
    if (equal(bytes, "true") or equal(bytes, "false") or equal(bytes, "null")) return true;
    var buffer: [64]u8 = undefined;
    if (bytes.len >= buffer.len) return false;
    @memcpy(buffer[0..bytes.len], bytes);
    buffer[bytes.len] = 0;
    var end: [*c]u8 = null;
    _ = c.strtod(&buffer, &end);
    return end != null and end[0] == 0 and end != @as([*c]u8, &buffer);
}
fn parseQwen(parser: *Parser) void {
    const start = "<tool_call>";
    const close = "</tool_call>";
    const function_open = "<function=";
    const function_close = "</function>";
    const parameter_open = "<parameter=";
    const parameter_close = "</parameter>";
    if (!starts(raw(parser), start)) return;
    while (parser.state == structural or parser.state == value_state) {
        const input = raw(parser);
        if (parser.state == value_state) {
            const value = input[parser.param_value_start..];
            const end = std.mem.indexOf(u8, visible(value), parameter_close) orelse return;
            var body = value[0..end];
            if (body.len != 0 and body[0] == '\n') body = body[1..];
            if (body.len != 0 and body[body.len - 1] == '\n') body = body[0 .. body.len - 1];
            addValue(parser, body, !qwenJson(body), parameter_close, parser.param_value_start + end + parameter_close.len);
            continue;
        }
        parser.parse_pos = skipSpace(input, parser.parse_pos);
        if (parser.parse_pos == input.len) return;
        const current = input[parser.parse_pos..];
        if (parser.glm_after_call) {
            if (starts(current, start)) {
                parser.parse_pos += start.len;
                parser.glm_after_call = false;
                continue;
            }
            if (partial(current, start)) return;
            parser.glm_after_call = false;
            parser.state = done;
            return;
        }
        if (parser.current.name == null) {
            if (!starts(current, function_open)) {
                if (partial(current, function_open)) return;
                setError(parser, "expected <function=...> in Qwen tool call");
                return;
            }
            const name_bytes = current[function_open.len..];
            const end = std.mem.indexOfScalar(u8, name_bytes, '>') orelse return;
            const name = std.mem.trim(u8, name_bytes[0..end], whitespace);
            if (name.len == 0) {
                setError(parser, "Qwen tool call without function name");
                return;
            }
            ds4_agent_tool_call_free(&parser.current);
            parser.current.name = duplicate(name);
            parser.parse_pos += function_open.len + end + 1;
            continue;
        }
        if (starts(current, function_close)) {
            const after = skipSpace(input, parser.parse_pos + function_close.len);
            if (after == input.len or partial(input[after..], close)) return;
            if (!starts(input[after..], close)) {
                setError(parser, "expected </tool_call> after </function>");
                return;
            }
            parser.parse_pos = after + close.len;
            ds4_agent_tool_calls_push(&parser.calls, &parser.current);
            parser.glm_after_call = true;
            continue;
        }
        if (partial(current, function_close)) return;
        if (!starts(current, parameter_open)) {
            if (partial(current, parameter_open)) return;
            setError(parser, "expected <parameter=...> or </function> in Qwen tool call");
            return;
        }
        const key_bytes = current[parameter_open.len..];
        const end = std.mem.indexOfScalar(u8, key_bytes, '>') orelse return;
        const key = std.mem.trim(u8, key_bytes[0..end], whitespace);
        if (key.len == 0) {
            setError(parser, "empty <parameter=> name in Qwen tool call");
            return;
        }
        c.free(parser.param_name);
        parser.param_name = duplicate(key);
        parser.param_is_string = true;
        parser.param_value_start = parser.parse_pos + parameter_open.len + end + 1;
        parser.parse_pos = parser.param_value_start;
        parser.param_close_prefix = false;
        parser.state = value_state;
    }
}

fn parseDsml(parser: *Parser) void {
    const calls = text(ds4_agent_tool_tag_name(parser.syntax, "tool_calls"));
    const invoke = text(ds4_agent_tool_tag_name(parser.syntax, "invoke"));
    const parameter = text(ds4_agent_tool_tag_name(parser.syntax, "parameter"));
    while (parser.state == structural or parser.state == value_state) {
        const input = raw(parser);
        if (parser.state == value_state) {
            const value = input[parser.param_value_start..];
            const close = findClose(value, parameter) orelse return;
            const closing = ds4_agent_tool_parameter_close(parser.syntax);
            addValue(parser, value[0..close.start], parser.param_is_string, closing, parser.param_value_start + close.end);
            continue;
        }
        parser.parse_pos = skipSpace(input, parser.parse_pos);
        if (parser.parse_pos == input.len) return;
        const current = input[parser.parse_pos..];
        if (closeTag(current, calls)) |length| {
            ds4_agent_tool_calls_push(&parser.calls, &parser.current);
            parser.parse_pos += length;
            parser.state = done;
            return;
        }
        if (closeTag(current, invoke)) |length| {
            ds4_agent_tool_calls_push(&parser.calls, &parser.current);
            parser.parse_pos += length;
            continue;
        }
        const end = std.mem.indexOfScalar(u8, visible(current), '>') orelse return;
        const tag = current[0 .. end + 1];
        if (openTag(tag, invoke)) {
            ds4_agent_tool_call_free(&parser.current);
            parser.current.name = if (attribute(tag, "name")) |name| duplicate(name) else null;
            if (parser.current.name == null) {
                setError(parser, "tool invoke without name");
                return;
            }
            parser.parse_pos += tag.len;
        } else if (openTag(tag, parameter)) {
            c.free(parser.param_name);
            parser.param_name = if (attribute(tag, "name")) |name| duplicate(name) else null;
            parser.param_is_string = if (attribute(tag, "string")) |value| equal(value, "true") else false;
            if (parser.param_name == null) {
                setError(parser, "tool parameter without name");
                return;
            }
            parser.parse_pos += tag.len;
            parser.param_value_start = parser.parse_pos;
            parser.param_close_prefix = false;
            parser.state = value_state;
        } else {
            parser.state = invalid;
            // snprintf's precision counts bytes, and stops at a NUL as usual.
            const excerpt = visible(tag)[0..@min(80, visible(tag).len)];
            _ = std.mem.printSentinel(&parser.@"error", "unexpected DSML tag: {s}", .{excerpt}, 0) catch unreachable;
            return;
        }
    }
}
fn parse(parser: *Parser) void {
    switch (parser.syntax) {
        c.AGENT_TOOL_SYNTAX_GLM => parseGlm(parser),
        c.AGENT_TOOL_SYNTAX_QWEN => parseQwen(parser),
        else => parseDsml(parser),
    }
}
export fn ds4_agent_tool_parser_start(parser: *Parser) void {
    const start = text(ds4_agent_tool_start(parser.syntax));
    parser.state = structural;
    parser.search_len = 0;
    appendRaw(parser, start);
    parser.parse_pos = start.len;
}
export fn ds4_agent_tool_parser_feed(parser: *Parser, input: [*c]const u8, length: usize) void {
    if (parser.state == done or parser.state == invalid) return;
    const start = text(ds4_agent_tool_start(parser.syntax));
    for (span(input, length)) |byte| {
        if (parser.state == search) {
            if (parser.search_len == parser.search_tail.len) {
                parser.search_len -= 1;
                std.mem.copyForwards(u8, parser.search_tail[0..parser.search_len], parser.search_tail[1..]);
            }
            parser.search_tail[parser.search_len] = byte;
            parser.search_len += 1;
            if (parser.search_len >= start.len and equal(parser.search_tail[parser.search_len - start.len .. parser.search_len], start))
                ds4_agent_tool_parser_start(parser);
            continue;
        }
        // Preserve the remainder of this same chunk after DONE/ERROR for raw
        // diagnostics. Only a subsequent feed call ignores further input.
        appendRaw(parser, &.{byte});
        parse(parser);
        if (parser.state == value_state) updateClosePrefix(parser) else parser.param_close_prefix = false;
    }
}
export fn ds4_agent_tool_parser_finish(maybe_parser: ?*Parser) void {
    const parser = maybe_parser orelse return;
    if (parser.state == done or parser.state == invalid or !parser.glm_after_call or
        (parser.syntax != c.AGENT_TOOL_SYNTAX_GLM and parser.syntax != c.AGENT_TOOL_SYNTAX_QWEN)) return;
    parser.parse_pos = skipSpace(raw(parser), parser.parse_pos);
    if (parser.parse_pos >= parser.raw_len) {
        parser.glm_after_call = false;
        parser.state = done;
    }
}
