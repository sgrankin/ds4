//! File/pager/search actions and owned tool observations. Files are streamed:
//! neither a large file nor a recursive tree is accumulated in memory. Worker
//! execution and UI publication remain whole opaque actions during migration.
const std = @import("std");
const c = @import("agent_tool_actions_c");
const allocator = std.heap.c_allocator;
pub const Pager = c.ds4_agent_pager;
pub const Call = c.agent_tool_call;
pub const Observation = c.agent_tool_observation;
const max_bytes = c.AGENT_TOOL_MAX_BYTES;

fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn items(comptime T: type, pointer: [*c]T, count: usize) []T {
    // Owning ArrayLists may have capacity but zero live items. Preserve that
    // allocation's pointer when adopting it; only NULL/zero is a static empty.
    return if (count == 0 and pointer == null) &.{} else pointer[0..count];
}
fn same(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
fn oom(label: [*:0]const u8) noreturn {
    c.perror(label);
    c.exit(1);
}
fn allocate(comptime T: type, count: usize) [*]T {
    const memory = c.malloc(@max(1, count * @sizeOf(T))) orelse oom("ds4-agent: malloc");
    return @ptrCast(@alignCast(memory));
}
fn duplicate(bytes: []const u8) [*c]u8 {
    const output = allocate(u8, bytes.len + 1);
    @memcpy(output[0..bytes.len], bytes);
    output[bytes.len] = 0;
    return output;
}
fn errorText(destination: []u8, message: [*:0]const u8) void {
    if (destination.len != 0) _ = c.snprintf(destination.ptr, destination.len, "%s", message);
}
fn arg(call: *const Call, name: [*:0]const u8) [*c]const u8 {
    return c.ds4_agent_tool_arg_value(call, name);
}
fn integer(pointer: [*c]const u8, default: c_int, minimum: c_int, maximum: c_int) c_int {
    if (pointer == null or pointer[0] == 0) return default;
    var end: [*c]u8 = null;
    const value = c.strtol(pointer, &end, 10);
    if (end == pointer) return default;
    while (end[0] != 0 and std.mem.indexOfScalar(u8, " \t\r\n", end[0]) != null) end += 1;
    if (end[0] != 0) return default;
    return @intCast(@max(minimum, @min(maximum, value)));
}
fn boolean(pointer: [*c]const u8, default: bool) bool {
    if (pointer == null or pointer[0] == 0) return default;
    if (c.strcasecmp(pointer, "true") == 0 or c.strcasecmp(pointer, "yes") == 0 or c.strcmp(pointer, "1") == 0) return true;
    if (c.strcasecmp(pointer, "false") == 0 or c.strcasecmp(pointer, "no") == 0 or c.strcmp(pointer, "0") == 0) return false;
    return default;
}

// Private standard buffer with the original byte-limit/UTF8-tail policy. The
// returned allocation is libc-owned; no native container escapes the facade.
pub const Buffer = struct {
    data: std.ArrayList(u8) = .empty,
    limit: usize = 0,
    truncated: bool = false,

    pub fn deinit(self: *Buffer) void {
        self.data.deinit(allocator);
        self.* = .{};
    }
    pub fn append(self: *Buffer, bytes: []const u8) void {
        if (bytes.len == 0 or self.truncated) return;
        const maximum = if (self.limit != 0) self.limit else std.math.maxInt(usize) - 1;
        var count = bytes.len;
        if (count > maximum - self.data.items.len) {
            count = maximum -| self.data.items.len;
            while (count != 0 and bytes[count] & 0xc0 == 0x80) count -= 1;
            self.truncated = true;
        }
        if (count == 0) return;
        self.data.ensureUnusedCapacity(allocator, count + 1) catch oom("ds4-agent: realloc");
        self.data.appendSliceAssumeCapacity(bytes[0..count]);
        self.data.items.ptr[self.data.items.len] = 0;
    }
    fn prepend(self: *Buffer, bytes: []const u8) void {
        self.data.ensureUnusedCapacity(allocator, bytes.len + 1) catch oom("ds4-agent: realloc");
        const old_length = self.data.items.len;
        self.data.items.len += bytes.len;
        std.mem.copyBackwards(u8, self.data.items.ptr[bytes.len..][0 .. old_length + 1], self.data.items.ptr[0 .. old_length + 1]);
        @memcpy(self.data.items[0..bytes.len], bytes);
    }
    pub fn take(self: *Buffer) [*c]u8 {
        if (self.truncated) {
            self.truncated = false;
            self.limit = 0;
            self.append("\n[Output truncated at the tool byte limit. Narrow the request.]\n");
        }
        if (self.data.capacity == 0) return duplicate(&.{});
        const output = self.data.items.ptr;
        self.* = .{};
        return output;
    }
};

pub fn defaultLines(context_size: c_int) c_int {
    if (context_size > 0 and context_size <= c.AGENT_READ_SMALL_CONTEXT_MAX) return c.AGENT_READ_DEFAULT_LINES_SMALL;
    if (context_size > 0 and context_size <= c.AGENT_READ_MEDIUM_CONTEXT_MAX) return c.AGENT_READ_DEFAULT_LINES_MEDIUM;
    return c.AGENT_READ_DEFAULT_LINES_LARGE;
}
pub fn setPager(pager: *Pager, path: [*c]const u8, next_line: c_int, bare: bool) void {
    // more resumes from this very array; do not overwrite an aliased source.
    if (path != @as([*c]const u8, &pager.more_path))
        _ = c.snprintf(&pager.more_path, pager.more_path.len, "%s", if (path == null) @as([*c]const u8, "") else path);
    pager.more_next_line = next_line;
    pager.more_bare = bare;
    pager.more_valid = path != null and path[0] != 0 and next_line > 0;
    pager.more_byte_offset = 0;
    pager.more_mid_line = false;
}

const Read = struct {
    pager: *Pager,
    context_size: c_int,
    path: [*c]const u8,
    start: c_int,
    count: c_int,
    whole: bool,
    bare: bool,
    save_more: bool,
    offset: c.off_t,
    mid_line: bool,
    saved_errno: c_int = 0,

    fn failed(self: *Read) error{ReadFailed} {
        self.saved_errno = c.__error().*;
        return error.ReadFailed;
    }
    fn opened(self: *Read, file: *c.FILE, out: *Buffer) error{ ReadFailed, TextError }!void {
        var body = Buffer{};
        defer body.deinit();
        var status: c.struct_stat = undefined;
        if (c.fstat(c.fileno(file), &status) != 0) return self.failed();
        if ((status.st_mode & c.S_IFMT) != c.S_IFREG) {
            c.__error().* = c.EINVAL;
            return self.failed();
        }
        const start = @max(self.start, 1);
        const maximum = if (self.count <= 0) defaultLines(self.context_size) else self.count;
        var line: c_int = 1;
        var lines: c_int = 0;
        if (self.offset > 0) {
            if (c.fseeko(file, self.offset, c.SEEK_SET) != 0) return self.failed();
            line = start;
        }
        while (line < start) {
            const byte = c.fgetc(file);
            if (byte == c.EOF) break;
            if (byte == '\r') {
                const next = c.fgetc(file);
                if (next != '\n' and next != c.EOF) _ = c.ungetc(next, file);
            }
            if (byte == '\n' or byte == '\r') line += 1;
        }
        var beginning = !self.mid_line;
        var prefix = true;
        var last_line: c_int = 0;
        while (self.whole or lines < maximum) {
            const byte = c.fgetc(file);
            if (byte == c.EOF) break;
            // Keep a possible UTF8 continuation tail bounded even for malformed
            // input; do not split a normal code point at the page boundary.
            if (body.data.items.len >= max_bytes - 4096 and
                (byte & 0xc0 != 0x80 or body.data.items.len >= max_bytes - 4096 + 3))
            {
                _ = c.ungetc(byte, file);
                break;
            }
            if (byte == 0) {
                out.append("Tool error: read encountered binary data\n");
                return error.TextError;
            }
            last_line = line;
            if (prefix and !self.bare) {
                var label: [80]u8 = undefined;
                _ = c.snprintf(&label, label.len, "%d%s ", line, @as([*:0]const u8, if (beginning) "" else " (continued)"));
                body.append(text(&label));
            }
            prefix = false;
            beginning = false;
            var character: u8 = @intCast(byte);
            if (byte == '\r') {
                const next = c.fgetc(file);
                if (self.bare) body.append(&.{character});
                if (next == '\n') {
                    character = '\n';
                } else {
                    if (next != c.EOF) _ = c.ungetc(next, file);
                    character = if (self.bare) 0 else '\n';
                }
            }
            if (character != 0) body.append(&.{character});
            if (byte == '\n' or byte == '\r') {
                if (line == std.math.maxInt(c_int)) {
                    c.__error().* = c.EOVERFLOW;
                    return self.failed();
                }
                line += 1;
                lines += 1;
                beginning = true;
                prefix = true;
            }
        }
        if (c.ferror(file) != 0) return self.failed();
        const next_offset = c.ftello(file);
        if (next_offset < 0) return self.failed();
        const has_more = c.fgetc(file) != c.EOF;
        if (c.ferror(file) != 0) return self.failed();
        if (self.whole and has_more) {
            out.append("Tool error: whole read exceeds the 128 KiB output limit; use read and more for chunks\n");
            return error.TextError;
        }
        if (!self.bare) {
            var header: [c.PATH_MAX + 128]u8 = undefined;
            _ = c.snprintf(&header, header.len, "%s: lines %d-%d%s\n", self.path, @as(c_int, if (last_line != 0) start else 0), last_line, @as([*:0]const u8, if (has_more) " (partial read)" else " (end of file)"));
            out.append(text(&header));
        }
        out.append(body.data.items);
        if (has_more) {
            var note: [256]u8 = undefined;
            _ = c.snprintf(&note, note.len, "\n[Read truncated. continue_offset=%d; continue_byte_offset=%lld%s. Call more to continue.]\n", line, @as(c_longlong, next_offset), @as([*:0]const u8, if (beginning) "" else " (within line)"));
            out.append(text(&note));
        }
        if (self.save_more) {
            setPager(self.pager, if (has_more) self.path else null, if (has_more) line else 0, self.bare);
            self.pager.more_byte_offset = next_offset;
            self.pager.more_mid_line = !beginning;
        }
    }
    fn run(self: *Read) [*c]u8 {
        if (self.path == null or self.path[0] == 0) return duplicate("Tool error: read requires path\n");
        var out = Buffer{};
        defer out.deinit();
        if (c.ds4_agent_file_open_regular(self.path)) |file| {
            defer _ = c.fclose(file);
            self.opened(file, &out) catch |err| {
                if (err == error.ReadFailed) {
                    out.append("Tool error: read failed: ");
                    out.append(text(c.strerror(self.saved_errno)));
                    out.append("\n");
                }
                if (self.save_more) setPager(self.pager, null, 0, false);
            };
        } else {
            const saved = c.__error().*;
            out.append("Tool error: read failed: ");
            out.append(text(c.strerror(saved)));
            out.append("\n");
            if (self.save_more) setPager(self.pager, null, 0, false);
        }
        return out.take();
    }
};

pub fn readRange(pager: *Pager, context_size: c_int, path: [*c]const u8, start: c_int, count: c_int, whole: bool, bare: bool, set_more: bool, offset: c.off_t, mid_line: bool) [*c]u8 {
    var request = Read{ .pager = pager, .context_size = context_size, .path = path, .start = start, .count = count, .whole = whole, .bare = bare, .save_more = set_more, .offset = offset, .mid_line = mid_line };
    return request.run();
}
pub fn read(pager: *Pager, context_size: c_int, call: *const Call) [*c]u8 {
    return readRange(pager, context_size, arg(call, "path"), integer(arg(call, "start_line"), 1, 1, std.math.maxInt(c_int)), integer(arg(call, "max_lines"), defaultLines(context_size), 1, std.math.maxInt(c_int)), boolean(arg(call, "whole"), false), boolean(arg(call, "raw"), false), true, 0, false);
}
pub fn more(pager: *Pager, context_size: c_int, call: *const Call) [*c]u8 {
    const count = integer(arg(call, "count"), defaultLines(context_size), 1, std.math.maxInt(c_int));
    if (!pager.more_valid) return duplicate("Tool error: no previous output to continue\n");
    return readRange(pager, context_size, &pager.more_path, pager.more_next_line, count, false, pager.more_bare, true, pager.more_byte_offset, pager.more_mid_line);
}

pub fn list(call: *const Call) [*c]u8 {
    var path = arg(call, "path");
    if (path == null or path[0] == 0) path = ".";
    var out = Buffer{ .limit = max_bytes };
    defer out.deinit();
    const directory = c.opendir(path) orelse {
        const saved = c.__error().*;
        out.append("Tool error: opendir failed: ");
        out.append(text(c.strerror(saved)));
        out.append("\n");
        return out.take();
    };
    defer _ = c.closedir(directory);
    var header: [c.PATH_MAX + 64]u8 = undefined;
    _ = c.snprintf(&header, header.len, "%s:\n", path);
    out.append(text(&header));
    var shown: c_int = 0;
    while (true) {
        const entry = c.readdir(directory);
        if (entry == null) break;
        if (shown >= 300) {
            out.append("... more entries omitted ...\n");
            break;
        }
        const name: [*c]const u8 = &entry.*.d_name;
        if (same(text(name), ".") or same(text(name), "..")) continue;
        var full: [c.PATH_MAX]u8 = undefined;
        _ = c.snprintf(&full, full.len, "%s/%s", path, name);
        var status: c.struct_stat = undefined;
        if (c.lstat(&full, &status) != 0) continue;
        const kind: c_int = switch (status.st_mode & c.S_IFMT) {
            c.S_IFDIR => 'd',
            c.S_IFLNK => 'l',
            c.S_IFREG => '-',
            else => '?',
        };
        var line: [c.PATH_MAX + 96]u8 = undefined;
        _ = c.snprintf(&line, line.len, "%c %10lld %s%s\n", kind, @as(c_longlong, status.st_size), name, @as([*:0]const u8, if ((status.st_mode & c.S_IFMT) == c.S_IFDIR) "/" else ""));
        out.append(text(&line));
        shown += 1;
    }
    return out.take();
}

fn literal(haystack: []const u8, needle: []const u8, case_sensitive: bool) bool {
    if (case_sensitive) return std.mem.indexOf(u8, haystack, needle) != null;
    if (needle.len > haystack.len) return false;
    for (0..haystack.len - needle.len + 1) |position| {
        var matched = true;
        for (haystack[position..][0..needle.len], needle) |a, b| {
            if (c.tolower(a) != c.tolower(b)) {
                matched = false;
                break;
            }
        }
        if (matched) return true;
    }
    return false;
}
const Line = union(enum) { end, value: [*c]u8, io_error: c_int, rejected };
// Only the current line and six context lines are resident. Binary or oversized
// lines produce explicit incomplete-search evidence rather than disappearing.
fn readLine(file: *c.FILE) Line {
    var output = Buffer{};
    defer output.deinit();
    var byte: c_int = undefined;
    while (true) {
        byte = c.fgetc(file);
        if (byte == c.EOF) break;
        if (byte == 0 or output.data.items.len == max_bytes) return .rejected;
        if (byte == '\n' or byte == '\r') {
            if (byte == '\r') {
                const next = c.fgetc(file);
                if (next != '\n' and next != c.EOF) _ = c.ungetc(next, file);
            }
            break;
        }
        output.append(&.{@as(u8, @intCast(byte))});
    }
    if (c.ferror(file) != 0) return .{ .io_error = c.__error().* };
    if (byte == c.EOF and output.data.items.len == 0) return .end;
    return .{ .value = output.take() };
}
const Search = struct {
    query: [*c]const u8,
    glob: [*c]const u8,
    regex: c.regex_t = std.mem.zeroes(c.regex_t),
    use_regex: bool,
    case_sensitive: bool,
    context: c_int,
    maximum: c_int,
    results: c_int = 0,
    skipped: usize = 0,
    first_skip: [c.PATH_MAX + 128]u8 = @splat(0),
    output: Buffer = .{ .limit = max_bytes },

    fn skip(self: *Search, path: [*c]const u8, reason: [*c]const u8) void {
        self.skipped +%= 1;
        if (self.first_skip[0] == 0) _ = c.snprintf(&self.first_skip, self.first_skip.len, "%s: %s", path, reason);
    }
    fn matches(self: *Search, line: [*c]const u8) bool {
        return if (self.use_regex) c.regexec(&self.regex, line, 0, null, 0) == 0 else literal(text(line), text(self.query), self.case_sensitive);
    }
    fn emit(self: *Search, line: [*c]const u8, number: c_int) void {
        var label: [64]u8 = undefined;
        _ = c.snprintf(&label, label.len, "  %d ", number);
        self.output.append(text(&label));
        self.output.append(text(line));
        self.output.append("\n");
    }
    fn file(self: *Search, path: [*c]const u8) void {
        if (self.results >= self.maximum or self.output.truncated) return;
        if (self.glob != null and self.glob[0] != 0) {
            const base = c.strrchr(path, '/');
            if (c.fnmatch(self.glob, if (base == null) path else base + 1, 0) != 0 and c.fnmatch(self.glob, path, 0) != 0) return;
        }
        const stream = c.ds4_agent_file_open_regular(path) orelse {
            self.skip(path, c.strerror(c.__error().*));
            return;
        };
        defer _ = c.fclose(stream);
        var ring: [6][*c]u8 = @splat(null);
        defer for (ring) |line| c.free(line);
        var printed = false;
        var number: c_int = 0;
        var last: c_int = 0;
        var after: c_int = 0;
        while ((self.results < self.maximum or after != 0) and !self.output.truncated) {
            const line = switch (readLine(stream)) {
                .end => break,
                .io_error => |reason| {
                    self.skip(path, c.strerror(reason));
                    break;
                },
                .rejected => {
                    self.skip(path, "binary data or line exceeds 128 KiB");
                    break;
                },
                .value => |value| value,
            };
            if (number == std.math.maxInt(c_int)) {
                c.free(line);
                self.skip(path, "line number limit reached");
                break;
            }
            number += 1;
            const slot: usize = @intCast(@mod(number, 6));
            c.free(ring[slot]);
            ring[slot] = line;
            const match = self.results < self.maximum and self.matches(line);
            if (match and !printed) {
                self.output.append(text(path));
                self.output.append("\n");
                printed = true;
            }
            if (match) {
                var first = number - self.context;
                if (first <= last) first = last + 1;
                var current = first;
                while (current <= number) : (current += 1) self.emit(ring[@intCast(@mod(current, 6))], current);
                last = number;
                after = self.context;
                self.results += 1;
            } else if (after != 0) {
                self.emit(line, number);
                last = number;
                after -= 1;
            }
        }
        if (printed) self.output.append("\n");
    }
    fn walk(self: *Search, name: [*c]const u8, depth: c_int) void {
        if (self.results >= self.maximum or self.output.truncated) return;
        if (depth > 24) {
            self.skip(name, "directory depth limit reached");
            return;
        }
        var status: c.struct_stat = undefined;
        if ((if (depth == 0) c.stat(name, &status) else c.lstat(name, &status)) != 0) {
            self.skip(name, c.strerror(c.__error().*));
            return;
        }
        const kind = status.st_mode & c.S_IFMT;
        if (kind == c.S_IFREG) {
            self.file(name);
            return;
        }
        if (kind != c.S_IFDIR) {
            self.skip(name, "not a regular file or directory (nested symlinks are not followed)");
            return;
        }
        const directory = c.opendir(name) orelse {
            self.skip(name, c.strerror(c.__error().*));
            return;
        };
        defer _ = c.closedir(directory);
        while (self.results < self.maximum and !self.output.truncated) {
            c.__error().* = 0;
            const entry = c.readdir(directory);
            if (entry == null) {
                if (c.__error().* != 0) self.skip(name, c.strerror(c.__error().*));
                break;
            }
            const child_name: [*c]const u8 = &entry.*.d_name;
            if (same(text(child_name), ".") or same(text(child_name), "..") or same(text(child_name), ".git")) continue;
            var child: [c.PATH_MAX]u8 = undefined;
            if (c.snprintf(&child, child.len, "%s/%s", name, child_name) >= @as(c_int, @intCast(child.len))) {
                self.skip(name, "path exceeds PATH_MAX");
                continue;
            }
            self.walk(&child, depth + 1);
        }
    }
};

pub fn search(call: *const Call) [*c]u8 {
    const query = arg(call, "query");
    if (query == null or query[0] == 0) return duplicate("Tool error: search requires query\n");
    var path = arg(call, "path");
    if (path == null or path[0] == 0) path = ".";
    const mode = arg(call, "mode");
    if (mode != null and !same(text(mode), "literal") and !same(text(mode), "regex")) return duplicate("Tool error: search mode must be literal or regex (POSIX extended)\n");
    var root: c.struct_stat = undefined;
    if (c.stat(path, &root) != 0 or ((root.st_mode & c.S_IFMT) != c.S_IFDIR and (root.st_mode & c.S_IFMT) != c.S_IFREG)) {
        var error_output = Buffer{};
        error_output.append("Tool error: search path is missing, unreadable, or not a regular file/directory: ");
        error_output.append(text(path));
        error_output.append("\n");
        return error_output.take();
    }
    var request = Search{ .query = query, .glob = arg(call, "glob"), .use_regex = mode != null and same(text(mode), "regex"), .case_sensitive = boolean(arg(call, "case_sensitive"), true), .context = integer(arg(call, "context"), 0, 0, 5), .maximum = integer(arg(call, "max_results"), 50, 1, 500) };
    defer request.output.deinit();
    if (request.use_regex) {
        var flags: c_int = c.REG_EXTENDED | c.REG_NOSUB;
        if (!request.case_sensitive) flags |= c.REG_ICASE;
        const result = c.regcomp(&request.regex, query, flags);
        if (result != 0) {
            var reason: [256]u8 = undefined;
            _ = c.regerror(result, &request.regex, &reason, reason.len);
            var output = Buffer{};
            output.append("Tool error: invalid regex: ");
            output.append(text(&reason));
            output.append("\n");
            return output.take();
        }
    }
    defer if (request.use_regex) c.regfree(&request.regex);
    request.walk(path, 0);
    if (request.output.data.capacity == 0) {
        request.output.append("No matches in searched text\n");
    } else {
        var header: [96]u8 = undefined;
        _ = c.snprintf(&header, header.len, "%d match%s shown\n\n", request.results, @as([*:0]const u8, if (request.results == 1) "" else "es"));
        request.output.prepend(text(&header));
    }
    request.output.limit = 0;
    if (request.skipped != 0 or request.results >= request.maximum) {
        const truncated = request.output.truncated;
        request.output.truncated = false;
        var note: [c.PATH_MAX + 256]u8 = undefined;
        _ = c.snprintf(&note, note.len, "\nSearch incomplete: %zu skipped paths%s. %s\n", request.skipped, @as([*:0]const u8, if (request.results >= request.maximum) "; match limit reached" else ""), &request.first_skip);
        request.output.append(text(&note));
        request.output.truncated = truncated;
    }
    return request.output.take();
}

fn textParts(observation: *Observation) std.ArrayList(c.agent_tool_text_part) {
    var parts: std.ArrayList(c.agent_tool_text_part) = .empty;
    parts.items = items(c.agent_tool_text_part, observation.parts, observation.part_count);
    parts.capacity = observation.part_cap;
    return parts;
}
fn publishParts(observation: *Observation, parts: std.ArrayList(c.agent_tool_text_part)) void {
    observation.parts = parts.items.ptr;
    observation.part_count = parts.items.len;
    observation.part_cap = parts.capacity;
}
pub fn initObservation(observation: *Observation) void {
    observation.* = std.mem.zeroes(Observation);
    observation.parts = allocate(c.agent_tool_text_part, 1);
    observation.parts[0] = std.mem.zeroes(c.agent_tool_text_part);
    observation.part_count = 1;
    observation.part_cap = 1;
}
pub fn puts(observation: *Observation, value: [*c]const u8) void {
    if (observation.part_count == 0) initObservation(observation);
    const part: *c.agent_tool_text_part = @ptrCast(&observation.parts[observation.part_count - 1]);
    const bytes = text(value);
    var buffer: Buffer = .{};
    buffer.data.items = items(u8, part.text, part.len);
    buffer.data.capacity = part.cap;
    buffer.data.ensureUnusedCapacity(allocator, bytes.len + 1) catch oom("ds4-agent: realloc");
    buffer.data.appendSliceAssumeCapacity(bytes);
    buffer.data.items.ptr[buffer.data.items.len] = 0;
    part.text = buffer.data.items.ptr;
    part.len = buffer.data.items.len;
    part.cap = buffer.data.capacity;
}
pub fn addImage(observation: *Observation, embedding: *c.ds4_vision_embedding) void {
    var images: std.ArrayList(c.ds4_vision_embedding) = .empty;
    images.items = items(c.ds4_vision_embedding, observation.images, observation.image_count);
    images.capacity = observation.image_cap;
    images.append(allocator, embedding.*) catch oom("ds4-agent: realloc");
    observation.images = images.items.ptr;
    observation.image_count = images.items.len;
    observation.image_cap = images.capacity;
    embedding.* = std.mem.zeroes(c.ds4_vision_embedding);
    var parts = textParts(observation);
    parts.append(allocator, std.mem.zeroes(c.agent_tool_text_part)) catch oom("ds4-agent: realloc");
    publishParts(observation, parts);
}
pub fn freeObservation(maybe_observation: ?*Observation) void {
    const observation = maybe_observation orelse return;
    for (items(c.agent_tool_text_part, observation.parts, observation.part_count)) |part| c.free(part.text);
    for (items(c.ds4_vision_embedding, observation.images, observation.image_count)) |*embedding| c.ds4_vision_embedding_free(embedding);
    c.free(observation.parts);
    c.free(observation.images);
    observation.* = std.mem.zeroes(Observation);
}
fn viewImage(observation: *Observation, worker: ?*anyopaque, engine: ?*c.ds4_engine, call: *const Call) void {
    const path = arg(call, "path");
    if (path == null or path[0] == 0) {
        puts(observation, "Tool error: view_image requires path\n");
        return;
    }
    if (!c.ds4_engine_has_vision(engine)) {
        puts(observation, "Tool error: view_image requires ds4-agent --vision FILE\n");
        return;
    }
    var embedding = std.mem.zeroes(c.ds4_vision_embedding);
    var error_buffer: [256]u8 = @splat(0);
    if (c.ds4_engine_vision_encode_file(engine, path, &embedding, &error_buffer, error_buffer.len) == 0) {
        puts(observation, "Tool error: view_image failed: ");
        puts(observation, if (error_buffer[0] != 0) @as([*c]const u8, &error_buffer) else "unable to decode image");
        puts(observation, "\n");
        return;
    }
    var shown: [c.PATH_MAX + 80]u8 = undefined;
    _ = c.snprintf(&shown, shown.len, "\n[tool:view_image] %s\n", path);
    c.ds4_agent_tool_publish(worker, &shown, text(&shown).len);
    addImage(observation, &embedding);
    const image = observation.images[observation.image_count - 1];
    var meta: [192]u8 = undefined;
    _ = c.snprintf(&meta, meta.len, "\nImage observation attached (%ux%u, %u visual tokens).\n", image.width, image.height, image.token_count);
    puts(observation, &meta);
}
pub fn execute(observation: *Observation, worker: ?*anyopaque, engine: ?*c.ds4_engine, calls: *const c.agent_tool_calls) void {
    initObservation(observation);
    for (items(c.agent_tool_call, calls.v, if (calls.len > 0) @intCast(calls.len) else 0), 0..) |*call, index| {
        var header: [128]u8 = undefined;
        _ = c.snprintf(&header, header.len, "Tool result %d (%s):\n", @as(c_int, @intCast(index + 1)), if (call.name == null) @as([*c]const u8, "unknown") else call.name);
        puts(observation, &header);
        if (call.name != null and same(text(call.name), "view_image")) {
            viewImage(observation, worker, engine, call);
            continue;
        }
        const result = c.ds4_agent_tool_execute_one(worker, call);
        puts(observation, result);
        const bytes = text(result);
        if (bytes.len != 0 and bytes[bytes.len - 1] != '\n') puts(observation, "\n");
        c.free(result);
    }
    if (calls.len == 0) puts(observation, "Tool error: empty tool call block\n");
}

pub fn build(engine: ?*c.ds4_engine, transcript: *const c.ds4_tokens, observation: *const Observation, tokens: *c.ds4_tokens, spans_out: [*c][*c]c.ds4_vision_span, errors: []u8) bool {
    const parts = allocate([*c]const u8, observation.part_count);
    defer c.free(@ptrCast(parts));
    for (0..observation.part_count) |index| parts[index] = if (observation.parts[index].text == null) @as([*c]const u8, "") else observation.parts[index].text;
    var images: [*c]c.ds4_vision_embedding = null;
    var spans: [*c]c.ds4_vision_span = null;
    var success = false;
    defer {
        if (!success) {
            for (items(c.ds4_vision_embedding, images, observation.image_count)) |*embedding| c.ds4_vision_embedding_free(embedding);
            for (items(c.ds4_vision_span, spans, observation.image_count)) |*span| c.ds4_vision_embedding_free(&span.embedding);
            c.free(spans);
            c.ds4_tokens_free(tokens);
        }
        c.free(images);
    }
    if (observation.image_count != 0) {
        images = allocate(c.ds4_vision_embedding, observation.image_count);
        spans = allocate(c.ds4_vision_span, observation.image_count);
        @memset(items(c.ds4_vision_embedding, images, observation.image_count), std.mem.zeroes(c.ds4_vision_embedding));
        @memset(items(c.ds4_vision_span, spans, observation.image_count), std.mem.zeroes(c.ds4_vision_span));
        const width = c.ds4_engine_embd_dim(engine);
        for (items(c.ds4_vision_embedding, observation.images, observation.image_count), 0..) |source, index| {
            if (source.data == null or source.token_count == 0 or width <= 0 or @as(u64, source.token_count) > std.math.maxInt(usize) / @as(u64, @intCast(width)) / @sizeOf(f32)) {
                errorText(errors, "invalid image observation embedding");
                return false;
            }
            const bytes = @as(usize, source.token_count) * @as(usize, @intCast(width)) * @sizeOf(f32);
            images[index] = source;
            images[index].data = @ptrCast(@alignCast(c.malloc(bytes)));
            if (images[index].data == null) {
                errorText(errors, "unable to copy image observation embedding");
                return false;
            }
            @memcpy(@as([*]u8, @ptrCast(images[index].data))[0..bytes], @as([*]const u8, @ptrCast(source.data))[0..bytes]);
        }
    }
    c.ds4_tokens_copy(tokens, transcript);
    // GLM grounds images in user turns; text-only observations retain the native
    // tool-response role used throughout the rest of the agent protocol.
    if (c.ds4_chat_append_multimodal_message(engine, tokens, if (observation.image_count != 0) @as([*:0]const u8, "user") else "tool", parts, images, observation.image_count, spans, errors.ptr, errors.len) == 0) return false;
    success = true;
    spans_out[0] = spans;
    return true;
}
// -1 is a rendering/embedding error, rather than a request to compact.
pub fn fits(engine: ?*c.ds4_engine, transcript: *const c.ds4_tokens, context_size: c_int, observation: *const Observation, reserve: c_int, tokens_out: [*c]c_int, errors: []u8) c_int {
    var tokens = std.mem.zeroes(c.ds4_tokens);
    var spans: [*c]c.ds4_vision_span = null;
    if (errors.len != 0) errors[0] = 0;
    if (!build(engine, transcript, observation, &tokens, &spans, errors)) return -1;
    const count = tokens.len;
    c.ds4_tokens_free(&tokens);
    for (items(c.ds4_vision_span, spans, observation.image_count)) |*span| c.ds4_vision_embedding_free(&span.embedding);
    c.free(spans);
    if (tokens_out != null) tokens_out[0] = count;
    return @intFromBool(context_size > 0 and count + reserve < context_size);
}
