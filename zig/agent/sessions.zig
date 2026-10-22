//! Saved-session identity, title codec and file catalog. Live payload loading,
//! synchronized saves and Worker/status publication remain outside this owner.
const std = @import("std");
pub const c = @import("agent_sessions_c");
const allocator = std.heap.c_allocator;
pub const Identity = c.ds4_agent_session_identity;
pub const Meta = c.agent_kv_session_meta;
fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn bytes(pointer: [*c]const u8, count: usize) []const u8 {
    return if (count == 0) &.{} else pointer[0..count];
}
fn oom(label: [*:0]const u8) noreturn {
    c.perror(label);
    c.exit(1);
}
fn duplicate(value: []const u8) [*c]u8 {
    return (allocator.dupeSentinel(u8, value, 0) catch oom("ds4-agent: malloc")).ptr;
}
fn message(errors: []u8, value: [*:0]const u8) void {
    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s", value);
}
fn systemError(errors: []u8) void {
    message(errors, c.strerror(c.__error().*));
}
fn trim(value: []const u8) []const u8 {
    var first: usize = 0;
    var last = value.len;
    while (first < last and c.isspace(value[first]) != 0) first += 1;
    while (last > first and c.isspace(value[last - 1]) != 0) last -= 1;
    return value[first..last];
}
fn take(buffer: *std.ArrayList(u8)) [*c]u8 {
    return (buffer.toOwnedSliceSentinel(allocator, 0) catch oom("ds4-agent: realloc")).ptr;
}
fn append(buffer: *std.ArrayList(u8), value: []const u8) void {
    buffer.appendSlice(allocator, value) catch oom("ds4-agent: realloc");
}

pub fn clearIdentity(identity: *Identity) void {
    identity.sha[0] = 0;
    c.free(identity.title);
    identity.title = null;
    identity.created_at = 0;
    c.free(identity.legacy_path);
    identity.legacy_path = null;
}
pub fn freeMeta(meta: *Meta) void {
    c.free(meta.title);
    meta.* = std.mem.zeroes(Meta);
}
// Session IDs are intentionally independent of the transcript. Once title and
// creation time exist, resaving keeps one filename as text and KV evolve.
pub fn identitySha(title: [*c]const u8, created_at: u64, output: [*c]u8) void {
    var input: std.ArrayList(u8) = .empty;
    defer input.deinit(allocator);
    append(&input, text(title));
    var timestamp: [8]u8 = undefined;
    std.mem.writeInt(u64, &timestamp, created_at, .little);
    append(&input, &timestamp);
    c.ds4_kvstore_sha1_bytes_hex(input.items.ptr, input.items.len, output);
}
// Cap disk-declared lengths against the actual remaining file before allocating.
// Seeking restores the payload cursor; nonseekable/truncated files fail plainly.
fn remaining(stream: *c.FILE) ?u64 {
    const position = c.ftello(stream);
    if (position < 0 or c.fseeko(stream, 0, c.SEEK_END) != 0) return null;
    const end = c.ftello(stream);
    if (end < 0 or c.fseeko(stream, position, c.SEEK_SET) != 0) return null;
    return if (end > position) @intCast(end - position) else 0;
}
pub fn readText(stream: *c.FILE, length: u32, output: *[*c]u8, errors: []u8) bool {
    const available = remaining(stream) orelse {
        message(errors, "truncated cached text");
        return false;
    };
    if (length > available) {
        message(errors, "truncated cached text");
        return false;
    }
    const result = allocator.allocSentinel(u8, length, 0) catch oom("ds4-agent: malloc");
    if (c.fread(result.ptr, 1, length, stream) != length) {
        message(errors, "truncated cached text");
        allocator.free(result);
        return false;
    }
    output.* = result.ptr; // Failed reads leave the caller's owner untouched.
    return true;
}
pub fn writeTitle(stream: *c.FILE, title: [*c]const u8, errors: []u8) bool {
    const value = text(title);
    if (value.len > std.math.maxInt(u32)) {
        message(errors, "agent session title is too large");
        return false;
    }
    var length: [4]u8 = undefined;
    std.mem.writeInt(u32, &length, @intCast(value.len), .little);
    return c.fwrite(&length, 1, length.len, stream) == length.len and
        c.fwrite(if (title == null) @as([*c]const u8, "") else title, 1, value.len, stream) == value.len;
}
// Read the optional title after the payload, then restore the payload-start
// cursor expected by the existing live ds4_session_load_payload transaction.
pub fn readTitle(stream: *c.FILE, header: *const c.ds4_kvstore_entry, output: *[*c]u8, errors: []u8) bool {
    const payload_position = c.ftello(stream);
    if (payload_position < 0) {
        systemError(errors);
        return false;
    }
    if (header.payload_bytes > std.math.maxInt(c_longlong) or c.fseeko(stream, @intCast(header.payload_bytes), c.SEEK_CUR) != 0) {
        systemError(errors);
        return false;
    }
    var encoded: [4]u8 = undefined;
    if (c.fread(&encoded, 1, encoded.len, stream) != encoded.len) {
        message(errors, "missing agent session title trailer");
        _ = c.fseeko(stream, payload_position, c.SEEK_SET);
        return false;
    }
    const length = std.mem.readInt(u32, &encoded, .little);
    const available = remaining(stream);
    if (available == null or length > available.?) {
        message(errors, "truncated agent session title trailer");
        _ = c.fseeko(stream, payload_position, c.SEEK_SET);
        return false;
    }
    const result = allocator.allocSentinel(u8, length, 0) catch oom("ds4-agent: malloc");
    if (c.fread(result.ptr, 1, length, stream) != length) {
        message(errors, "truncated agent session title trailer");
        allocator.free(result);
        _ = c.fseeko(stream, payload_position, c.SEEK_SET);
        return false;
    }
    if (c.fseeko(stream, payload_position, c.SEEK_SET) != 0) {
        systemError(errors);
        allocator.free(result);
        return false;
    }
    output.* = result.ptr;
    return true;
}
pub fn fileSha(header: *const c.ds4_kvstore_entry, rendered: [*c]const u8, length: u32, title: [*c]const u8, output: [*c]u8) void {
    if (header.ext_flags & c.DS4_KVSTORE_EXT_SESSION_TITLE != 0) identitySha(title, header.created_at, output) else c.ds4_kvstore_sha1_bytes_hex(rendered, length, output);
}

fn normalizedTitle(value: []const u8, budget: usize, empty: []const u8) [*c]u8 {
    const limited = budget != 0;
    const maximum = if (limited) @max(budget, 4) else 0;
    var output: std.ArrayList(u8) = .empty;
    defer output.deinit(allocator);
    var space = false;
    var truncated = false;
    for (trim(value)) |byte| {
        if (c.isspace(byte) != 0) {
            space = output.items.len != 0;
            continue;
        }
        if (space and (!limited or output.items.len + 4 < maximum)) {
            append(&output, " ");
            space = false;
        }
        if (limited and output.items.len + 4 > maximum) {
            truncated = true;
            break;
        }
        append(&output, &.{byte});
    }
    if (truncated) append(&output, "...");
    return if (output.items.len == 0) duplicate(empty) else take(&output);
}
pub fn titlePrompt(prompt: [*c]const u8, budget: usize) [*c]u8 {
    return normalizedTitle(text(prompt), budget, "(empty user prompt)");
}
// The first rendered user turn supplies the title. Zero budget means the full
// normalized title; terminal callers provide an explicit display budget.
pub fn titleText(rendered: [*c]const u8, length: usize, budget: usize) [*c]u8 {
    const user = "<｜User｜>";
    const assistant = "<｜Assistant｜>";
    const first = std.mem.find(u8, text(rendered), user) orelse return duplicate("(no user prompt)");
    const start = first + user.len;
    if (start >= length) return duplicate("(empty user prompt)");
    var end = length;
    const tail = text(rendered + start);
    if (std.mem.find(u8, tail, assistant)) |offset| end = @min(end, start + offset);
    if (std.mem.find(u8, tail, user)) |offset| end = @min(end, start + offset);
    return normalizedTitle(bytes(rendered, length)[start..end], budget, "(empty user prompt)");
}
fn clipTitle(title: [*c]const u8, budget: usize) [*c]u8 {
    if (title == null) return duplicate("(no user prompt)");
    const value = text(title);
    if (budget == 0 or value.len <= budget) return duplicate(value);
    var output: std.ArrayList(u8) = .empty;
    defer output.deinit(allocator);
    append(&output, value[0 .. @max(budget, 4) - 3]);
    append(&output, "...");
    return take(&output);
}
fn titleFile(path: [*c]const u8, budget: usize) [*c]u8 {
    const stream = c.fopen(path, "rb");
    if (stream == null) return duplicate("(unreadable session)");
    var header = std.mem.zeroes(c.ds4_kvstore_entry);
    var length: u32 = 0;
    var rendered: [*c]u8 = null;
    var title: [*c]u8 = null;
    defer {
        c.free(rendered);
        c.free(title);
    }
    var ok = c.ds4_kvstore_read_header(stream, &header, &length) and readText(stream, length, &rendered, &.{});
    if (ok and header.ext_flags & c.DS4_KVSTORE_EXT_SESSION_TITLE != 0) ok = readTitle(stream, &header, &title, &.{});
    _ = c.fclose(stream);
    if (!ok) return duplicate("(unreadable session)");
    return if (title != null) clipTitle(title, budget) else titleText(rendered, length, budget);
}

const Item = struct { entry: c.ds4_kvstore_entry, title: [*c]u8 };
fn recent(_: void, a: Item, b: Item) bool {
    const a_time = if (a.entry.last_used != 0) a.entry.last_used else a.entry.created_at;
    const b_time = if (b.entry.last_used != 0) b.entry.last_used else b.entry.created_at;
    return if (a_time == b_time) c.strcmp(&a.entry.sha, &b.entry.sha) < 0 else a_time > b_time;
}
fn age(when: u64, output: []u8) void {
    const now: u64 = @bitCast(@as(i64, c.time(null)));
    const seconds = if (when != 0 and now > when) now - when else 0;
    const scale: u64 = if (seconds < 60) 1 else if (seconds < 3600) 60 else if (seconds < 86400) 3600 else 86400;
    const unit: [*:0]const u8 = if (seconds < 60) "s" else if (seconds < 3600) "m" else if (seconds < 86400) "h" else "d";
    _ = c.snprintf(output.ptr, output.len, "%llu%s ago", @as(c_ulonglong, seconds / scale), unit);
}
// sysprompt.kv is an implementation cache, never a resumable user session.
pub fn list(directory: [*c]const u8, model_id: u8, columns: c_int) void {
    const handle = c.opendir(directory);
    if (handle == null) {
        _ = c.printf("no sessions: %s\n", c.strerror(c.__error().*));
        return;
    }
    const budget = @min(if (columns > 16) @as(usize, @intCast(columns - 12)) else 20, 160);
    var entries: std.ArrayList(Item) = .empty;
    defer {
        for (entries.items) |*item| {
            c.ds4_kvstore_entry_free(&item.entry);
            c.free(item.title);
        }
        entries.deinit(allocator);
    }
    while (true) {
        const entry = c.readdir(handle);
        if (entry == null) break;
        var sha: [41]u8 = undefined;
        if (!c.ds4_kvstore_sha_hex_name(&entry.*.d_name, &sha)) continue;
        const path = c.ds4_kvstore_path_join(directory, &entry.*.d_name);
        defer c.free(path);
        var file = std.mem.zeroes(c.ds4_kvstore_entry);
        if (!c.ds4_kvstore_read_entry_file(path, &sha, &file)) continue;
        if (file.model_id != model_id) {
            c.ds4_kvstore_entry_free(&file);
            continue;
        }
        entries.append(allocator, .{ .entry = file, .title = titleFile(path, budget) }) catch oom("ds4-agent: realloc");
    }
    _ = c.closedir(handle);
    if (entries.items.len == 0) {
        _ = c.printf("no saved sessions\n");
        return;
    }
    std.mem.sortUnstable(Item, entries.items, {}, recent);
    const color = c.isatty(c.STDOUT_FILENO) != 0;
    const sha_on: [*:0]const u8 = if (color) "\x1b[1;96m" else "";
    const title_on: [*:0]const u8 = if (color) "\x1b[1;97m" else "";
    const help_on: [*:0]const u8 = if (color) "\x1b[97m" else "";
    const dim: [*:0]const u8 = if (color) "\x1b[90m" else "";
    const reset: [*:0]const u8 = if (color) "\x1b[0m" else "";
    for (entries.items) |item| {
        var formatted_age: [32]u8 = undefined;
        age(if (item.entry.last_used != 0) item.entry.last_used else item.entry.created_at, &formatted_age);
        _ = c.printf("%s%.8s%s %s>%s %s%s%s\n", sha_on, &item.entry.sha, reset, dim, reset, title_on, item.title, reset);
        _ = c.printf("         %s> %s, %u tokens, %.2f MB%s%s\n\n", dim, &formatted_age, item.entry.tokens, @as(f64, @floatFromInt(item.entry.file_size)) / (1024.0 * 1024.0), @as([*:0]const u8, if (item.entry.payload_bytes == 0) ", stripped" else ""), reset);
    }
    _ = c.printf("%sUse /switch <id> to select a session, /del <id> to remove, /strip <id> to strip KV cache.%s\n", help_on, reset);
}
const Completion = struct { sha: [41]u8, last_used: u64 };
fn completionRecent(_: void, a: Completion, b: Completion) bool {
    return if (a.last_used == b.last_used) c.strcmp(&a.sha, &b.sha) < 0 else a.last_used > b.last_used;
}
// /switch accepts an empty or hexadecimal prefix. Completion order is recent
// use then full SHA; a prefix longer than eight characters displays full IDs.
pub fn complete(directory: [*c]const u8, model_id: u8, input: [*c]const u8, output: *c.linenoiseCompletions) void {
    if (input == null or !std.mem.startsWith(u8, text(input), "/switch")) return;
    var prefix = text(input)["/switch".len..];
    if (prefix.len != 0 and prefix[0] != ' ' and prefix[0] != '\t') return;
    prefix = std.mem.trimStart(u8, prefix, " \t");
    for (prefix) |byte| if (c.isxdigit(byte) == 0) return;
    if (prefix.len > 40) return;
    const handle = c.opendir(directory);
    if (handle == null) return;
    var entries: std.ArrayList(Completion) = .empty;
    defer entries.deinit(allocator);
    while (true) {
        const entry = c.readdir(handle);
        if (entry == null) break;
        var sha: [41]u8 = undefined;
        if (!c.ds4_kvstore_sha_hex_name(&entry.*.d_name, &sha)) continue;
        if (prefix.len != 0 and c.strncasecmp(&sha, prefix.ptr, prefix.len) != 0) continue;
        const path = c.ds4_kvstore_path_join(directory, &entry.*.d_name);
        defer c.free(path);
        var file = std.mem.zeroes(c.ds4_kvstore_entry);
        if (!c.ds4_kvstore_read_entry_file(path, &sha, &file)) continue;
        const admitted = file.model_id == model_id and file.last_used != std.math.maxInt(u64);
        const last_used = file.last_used;
        c.ds4_kvstore_entry_free(&file);
        if (!admitted) continue;
        entries.append(allocator, .{ .sha = sha, .last_used = last_used }) catch oom("ds4-agent: realloc");
    }
    _ = c.closedir(handle);
    std.mem.sortUnstable(Completion, entries.items, {}, completionRecent);
    for (entries.items) |entry| {
        var line: [64]u8 = undefined;
        _ = c.snprintf(&line, line.len, "/switch %.*s", @as(c_int, if (prefix.len > 8) 40 else 8), &entry.sha);
        c.linenoiseAddCompletion(output, &line);
    }
}
const Match = struct { sha: [41]u8, path: [*c]u8 };
fn resolve(directory: [*c]const u8, model_id: u8, prefix: [*c]const u8, errors: []u8) ?Match {
    const value = text(prefix);
    if (value.len == 0 or value.len > 40) {
        message(errors, "invalid session SHA prefix");
        return null;
    }
    for (value) |byte| if (c.isxdigit(byte) == 0) {
        message(errors, "invalid session SHA prefix");
        return null;
    };
    const handle = c.opendir(directory);
    if (handle == null) {
        systemError(errors);
        return null;
    }
    var matches: usize = 0;
    var match: Match = undefined;
    while (true) {
        const entry = c.readdir(handle);
        if (entry == null) break;
        var sha: [41]u8 = undefined;
        if (!c.ds4_kvstore_sha_hex_name(&entry.*.d_name, &sha) or c.strncasecmp(&sha, prefix, value.len) != 0) continue;
        const path = c.ds4_kvstore_path_join(directory, &entry.*.d_name);
        var file = std.mem.zeroes(c.ds4_kvstore_entry);
        const same_model = c.ds4_kvstore_read_entry_file(path, &sha, &file) and file.model_id == model_id;
        c.ds4_kvstore_entry_free(&file);
        if (!same_model) {
            c.free(path);
            continue;
        }
        matches += 1;
        if (matches == 1) match = .{ .sha = sha, .path = path } else c.free(path);
    }
    _ = c.closedir(handle);
    if (matches == 0) {
        if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "no saved session matches %.40s", prefix);
        return null;
    }
    if (matches > 1) {
        if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "session prefix %.40s is ambiguous", prefix);
        c.free(match.path);
        return null;
    }
    return match;
}
pub fn find(directory: [*c]const u8, model_id: u8, prefix: [*c]const u8, sha: [*c]u8, path: *[*c]u8, errors: []u8) bool {
    const match = resolve(directory, model_id, prefix, errors) orelse return false;
    @memcpy(sha[0..41], &match.sha);
    path.* = match.path;
    return true;
}
pub fn delete(directory: [*c]const u8, model_id: u8, prefix: [*c]const u8, sha: [*c]u8, errors: []u8) bool {
    const match = resolve(directory, model_id, prefix, errors) orelse return false;
    defer c.free(match.path);
    if (c.unlink(match.path) != 0) {
        systemError(errors);
        return false;
    }
    if (sha != null) @memcpy(sha[0..41], &match.sha);
    return true;
}

// Strip heavy backend payloads without changing rendered text or title identity.
// Reopening then tokenizes the text and rebuilds live KV through ordinary sync.
pub fn strip(directory: [*c]const u8, model_id: u8, engine: ?*c.ds4_engine, prefix: [*c]const u8, sha_out: [*c]u8, tokens_out: [*c]u32, errors: []u8) bool {
    if (errors.len != 0) errors[0] = 0;
    const match = resolve(directory, model_id, prefix, errors) orelse return false;
    defer c.free(match.path);
    const source = c.fopen(match.path, "rb");
    if (source == null) {
        systemError(errors);
        return false;
    }
    var header = std.mem.zeroes(c.ds4_kvstore_entry);
    var length: u32 = 0;
    var rendered: [*c]u8 = null;
    var title: [*c]u8 = null;
    defer {
        c.free(title);
        c.free(rendered);
    }
    var ok = c.ds4_kvstore_read_header(source, &header, &length) and readText(source, length, &rendered, errors);
    if (ok and header.ext_flags & c.DS4_KVSTORE_EXT_SESSION_TITLE != 0) ok = readTitle(source, &header, &title, errors);
    _ = c.fclose(source);
    if (!ok) {
        if (errors.len != 0 and errors[0] == 0) message(errors, "failed to read session");
        return false;
    }
    var identity: [41]u8 = undefined;
    fileSha(&header, rendered, length, title, &identity);
    if (!std.mem.eql(u8, text(&identity), text(&match.sha))) {
        message(errors, "cached session identity does not match file name");
        return false;
    }
    var tokenized = std.mem.zeroes(c.ds4_tokens);
    c.ds4_tokenize_rendered_chat(engine, rendered, &tokenized);
    const token_count: u32 = @bitCast(tokenized.len);
    c.ds4_tokens_free(&tokenized);
    var template: std.ArrayList(u8) = .empty;
    defer template.deinit(allocator);
    append(&template, text(match.path));
    append(&template, ".tmp.XXXXXX");
    const temporary = take(&template);
    defer c.free(temporary);
    const descriptor = c.mkstemp(temporary);
    if (descriptor < 0) {
        systemError(errors);
        return false;
    }
    const destination = c.fdopen(descriptor, "wb");
    if (destination == null) {
        systemError(errors);
        _ = c.close(descriptor);
        _ = c.unlink(temporary);
        return false;
    }
    var encoded: [c.DS4_KVSTORE_FIXED_HEADER]u8 = undefined;
    const now: u64 = @bitCast(@as(i64, c.time(null)));
    c.ds4_kvstore_fill_header(&encoded, header.model_id, header.quant_bits, header.reason, header.ext_flags, token_count, header.hits, header.ctx_size, header.created_at, now, 0);
    var text_length: [4]u8 = undefined;
    std.mem.writeInt(u32, &text_length, length, .little);
    c.__error().* = 0;
    ok = c.fwrite(&encoded, 1, encoded.len, destination) == encoded.len and
        c.fwrite(&text_length, 1, text_length.len, destination) == text_length.len and
        c.fwrite(rendered, 1, length, destination) == length and
        (header.ext_flags & c.DS4_KVSTORE_EXT_SESSION_TITLE == 0 or writeTitle(destination, title, errors)) and c.fflush(destination) == 0;
    var saved_errno = c.__error().*;
    if (c.fclose(destination) != 0) {
        if (saved_errno == 0) saved_errno = c.__error().*;
        ok = false;
    }
    if (ok and c.rename(temporary, match.path) != 0) {
        saved_errno = c.__error().*;
        ok = false;
    }
    if (!ok) {
        message(errors, if (saved_errno != 0) c.strerror(saved_errno) else "failed to write stripped session");
        _ = c.unlink(temporary);
        return false;
    }
    if (sha_out != null) @memcpy(sha_out[0..41], &match.sha);
    if (tokens_out != null) tokens_out[0] = token_count;
    return true;
}
