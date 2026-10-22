//! Alternating USER:/ASSISTANT: prefix-file parsing, with libc-owned turns.
//! This parses the file grammar only. Native chat token/control rendering stays
//! in the public header's append helper and is not reimplemented here.
//!
//! Parse/load require an empty output (free a previous result first). Parsing
//! zeroes it before validating input; any partially built turns are freed on
//! failure. An invalid load path returns before touching output, matching C.
//! Error text is best effort into the caller's optional bounded buffer.
const std = @import("std");
const c = @import("c_api");
const Turn = c.ds4_prompt_prefix_turn;
const Prefix = c.ds4_prompt_prefix;
const Role = c.ds4_prompt_prefix_role;

fn setError(buf: ?[*]u8, cap: usize, comptime fmt: [*:0]const u8, args: anytype) void {
    if (buf != null and cap != 0) _ = @call(.auto, c.snprintf, .{ buf.?, cap, fmt } ++ args);
}

const Marker = struct { role: Role, len: usize };
fn markerAt(data: []const u8, pos: usize) ?Marker {
    if (std.mem.startsWith(u8, data[pos..], "USER:")) return .{ .role = c.DS4_PROMPT_PREFIX_USER, .len = 5 };
    if (std.mem.startsWith(u8, data[pos..], "ASSISTANT:")) return .{ .role = c.DS4_PROMPT_PREFIX_ASSISTANT, .len = 10 };
    return null;
}
fn roleName(role: Role) [*:0]const u8 {
    return if (role == c.DS4_PROMPT_PREFIX_USER) "USER" else "ASSISTANT";
}

export fn ds4_prompt_prefix_free(maybe_prefix: ?*Prefix) void {
    const prefix = maybe_prefix orelse return;
    if (prefix.turns != null) {
        for (prefix.turns[0..prefix.count]) |turn| c.free(turn.content);
        c.free(prefix.turns);
    }
    prefix.* = .{ .turns = null, .count = 0 };
}

export fn ds4_prompt_prefix_parse(maybe_out: ?*Prefix, maybe_data: ?[*]const u8, len: usize, err: ?[*]u8, err_cap: usize) c_int {
    const out = maybe_out orelse {
        setError(err, err_cap, "invalid prefix output", .{});
        return -1;
    };
    out.* = .{ .turns = null, .count = 0 };
    if (maybe_data == null or len == 0) {
        setError(err, err_cap, "prefix file is empty", .{});
        return -1;
    }
    const data = maybe_data.?[0..len];
    if (std.mem.indexOfScalar(u8, data, 0) != null) {
        setError(err, err_cap, "prefix file contains a NUL byte", .{});
        return -1;
    }
    // The optional UTF-8 BOM is accepted only at the start. Embedded NUL bytes
    // were rejected above; all other content bytes are retained verbatim.
    var pos: usize = if (std.mem.startsWith(u8, data, "\xef\xbb\xbf")) 3 else 0;
    var line: usize = 1;
    while (pos < len) {
        const marker = markerAt(data, pos) orelse {
            setError(err, err_cap, "line %zu must start with USER: or ASSISTANT:", .{line});
            ds4_prompt_prefix_free(out);
            return -1;
        };
        const expected: Role = if (out.count % 2 == 0) c.DS4_PROMPT_PREFIX_USER else c.DS4_PROMPT_PREFIX_ASSISTANT;
        if (marker.role != expected) {
            setError(err, err_cap, "line %zu: expected %s:", .{ line, roleName(expected) });
            ds4_prompt_prefix_free(out);
            return -1;
        }
        // Skip at most one space/tab after the marker, not arbitrary trimming.
        var content_start = pos + marker.len;
        if (content_start < len and (data[content_start] == ' ' or data[content_start] == '\t')) content_start += 1;
        // Only a marker at the start of a later line starts another turn;
        // inline marker-like text remains content. Roles must alternate.
        var next = len;
        var next_line = line;
        var scan_line = line;
        var i = content_start;
        while (i < len) : (i += 1) {
            if (data[i] != '\n') continue;
            scan_line += 1;
            if (i + 1 < len and markerAt(data, i + 1) != null) {
                next = i + 1;
                next_line = scan_line;
                break;
            }
        }
        // Drop exactly one terminal LF (and its optional CR), preserving any
        // preceding newlines, indentation and other content whitespace.
        var content_end = next;
        if (content_end > content_start and data[content_end - 1] == '\n') {
            content_end -= 1;
            if (content_end > content_start and data[content_end - 1] == '\r') content_end -= 1;
        }
        if (content_end == content_start) {
            setError(err, err_cap, "line %zu has an empty %s turn", .{ line, roleName(marker.role) });
            ds4_prompt_prefix_free(out);
            return -1;
        }
        if (out.count == std.math.maxInt(usize) / @sizeOf(Turn)) {
            setError(err, err_cap, "too many turns in prefix file", .{});
            ds4_prompt_prefix_free(out);
            return -1;
        }
        // Keep the old turn allocation published until realloc succeeds.
        // count advances only after the new content is owned, so failure cleanup
        // visits exactly the fully initialized turns.
        const resized = c.realloc(out.turns, (out.count + 1) * @sizeOf(Turn)) orelse {
            setError(err, err_cap, "out of memory reading prefix file", .{});
            ds4_prompt_prefix_free(out);
            return -1;
        };
        out.turns = @ptrCast(@alignCast(resized));
        // c_allocator's u8 result is malloc/free-compatible; transfer each
        // sentinel string into the public C record rather than borrowing input.
        const content = std.heap.c_allocator.dupeSentinel(u8, data[content_start..content_end], 0) catch {
            setError(err, err_cap, "out of memory reading prefix file", .{});
            ds4_prompt_prefix_free(out);
            return -1;
        };
        out.turns[out.count] = .{ .role = marker.role, .content = content.ptr };
        out.count += 1;
        pos = next;
        if (next != len) line = next_line;
    }
    if (out.count == 0) {
        setError(err, err_cap, "prefix file contains no turns", .{});
        return -1;
    }
    if (out.turns[out.count - 1].role != c.DS4_PROMPT_PREFIX_ASSISTANT) {
        setError(err, err_cap, "prefix file must end with an ASSISTANT turn", .{});
        ds4_prompt_prefix_free(out);
        return -1;
    }
    return 0;
}

export fn ds4_prompt_prefix_load(maybe_out: ?*Prefix, maybe_path: ?[*:0]const u8, err: ?[*]u8, err_cap: usize) c_int {
    if (maybe_out == null or maybe_path == null or maybe_path.?[0] == 0) {
        setError(err, err_cap, "invalid prefix file path", .{});
        return -1;
    }
    const out = maybe_out.?;
    const path = maybe_path.?;
    out.* = .{ .turns = null, .count = 0 };
    const fp = c.fopen(path, "rb");
    if (fp == null) {
        setError(err, err_cap, "cannot open prefix file %s: %s", .{ path, c.strerror(c.__error().*) });
        return -1;
    }
    if (c.fseek(fp, 0, c.SEEK_END) != 0) {
        setError(err, err_cap, "cannot seek prefix file %s: %s", .{ path, c.strerror(c.__error().*) });
        _ = c.fclose(fp);
        return -1;
    }
    const size = c.ftell(fp);
    if (size < 0 or @as(u64, @intCast(size)) > std.math.maxInt(usize) - 1) {
        setError(err, err_cap, "cannot size prefix file %s", .{path});
        _ = c.fclose(fp);
        return -1;
    }
    if (c.fseek(fp, 0, c.SEEK_SET) != 0) {
        setError(err, err_cap, "cannot rewind prefix file %s: %s", .{ path, c.strerror(c.__error().*) });
        _ = c.fclose(fp);
        return -1;
    }
    const len: usize = @intCast(size);
    const data: [*]u8 = @ptrCast(c.malloc(len + 1) orelse {
        setError(err, err_cap, "out of memory reading prefix file", .{});
        _ = c.fclose(fp);
        return -1;
    });
    defer c.free(@ptrCast(data));
    if (len != 0 and c.fread(data, 1, len, fp) != len) {
        setError(err, err_cap, "cannot read prefix file %s", .{path});
        _ = c.fclose(fp);
        return -1;
    }
    if (c.fclose(fp) != 0) {
        setError(err, err_cap, "cannot close prefix file %s: %s", .{ path, c.strerror(c.__error().*) });
        return -1;
    }
    data[len] = 0;
    // Parsing makes independent content copies before this file buffer is
    // released. Preserve the original parse message, then prefix its file path.
    const rc = ds4_prompt_prefix_parse(out, data, len, err, err_cap);
    if (rc != 0 and err != null and err_cap != 0 and err.?[0] != 0) {
        var detail: [256]u8 = undefined;
        _ = c.snprintf(&detail, detail.len, "%s", err.?);
        setError(err, err_cap, "%s: %s", .{ path, @as([*:0]const u8, @ptrCast(&detail)) });
    }
    return rc;
}
