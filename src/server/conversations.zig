//! Conversation records outlive model slots and response files. Manual item
//! edits discard private replay; visible items can build a new rendered context.
//! ResponseStore remains the atomic file publisher; this owner holds the one
//! process mutex across each version check and read/modify/write transaction.
const std = @import("std");
const c = @import("server_conversations_c");
const allocator = std.heap.c_allocator;
const Record = c.ds4_conversation;
const Invalid = error{InvalidItems};
var mutex: c.pthread_mutex_t = c.ds4_conversation_mutex_init;

fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
    c.exit(1);
}
fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn duplicate(value: []const u8) [*c]u8 {
    return (allocator.dupeSentinel(u8, value, 0) catch fatal()).ptr;
}
fn equal(pointer: [*c]const u8, expected: []const u8) bool {
    return std.mem.eql(u8, text(pointer), expected);
}
fn fail(errors: []u8, status: c_int, message: []const u8) c_int {
    if (errors.len != 0) {
        const count = @min(errors.len - 1, message.len);
        @memcpy(errors[0..count], message[0..count]);
        errors[count] = 0;
    }
    return status;
}
fn errorPointer(errors: []u8) [*c]u8 {
    return if (errors.len == 0) null else errors.ptr;
}
const Text = struct {
    list: std.ArrayList(u8) = .empty,
    fn deinit(self: *Text) void {
        self.list.deinit(allocator);
    }
    fn append(self: *Text, value: []const u8) void {
        self.list.appendSlice(allocator, value) catch fatal();
    }
    fn byte(self: *Text, value: u8) void {
        self.list.append(allocator, value) catch fatal();
    }
    fn number(self: *Text, value: anytype) void {
        self.list.print(allocator, "{d}", .{value}) catch fatal();
    }
    fn quoted(self: *Text, value: [*c]const u8) void {
        const hex = "0123456789abcdef";
        self.byte('"');
        for (text(value)) |character| switch (character) {
            '"', '\\' => self.append(&.{ '\\', character }),
            '\n' => self.append("\\n"),
            '\r' => self.append("\\r"),
            '\t' => self.append("\\t"),
            0...8, 11, 12, 14...31 => self.append(&.{ '\\', 'u', '0', '0', hex[character >> 4], hex[character & 15] }),
            else => self.byte(character),
        };
        self.byte('"');
    }
    fn take(self: *Text) [*c]u8 {
        return (self.list.toOwnedSliceSentinel(allocator, 0) catch fatal()).ptr;
    }
};
fn whitespace(cursor: *[*c]const u8) void {
    c.ds4_server_json_ws(cursor);
}
fn string(cursor: *[*c]const u8) Invalid![*c]u8 {
    var value: [*c]u8 = null;
    if (!c.ds4_server_json_string(cursor, &value)) return error.InvalidItems;
    return value;
}
fn rawValue(cursor: *[*c]const u8) Invalid![*c]u8 {
    var value: [*c]u8 = null;
    if (!c.ds4_server_json_raw(cursor, &value)) return error.InvalidItems;
    return value;
}

/// A raw owned value, including quoting/brackets. The last duplicate wins and
/// the whole outer object must finish; unlike stringField this is not a probe.
pub fn jsonField(object: [*c]const u8, wanted: [*c]const u8) [*c]u8 {
    if (object == null) return null;
    var cursor = object;
    whitespace(&cursor);
    if (cursor[0] != '{') return null;
    cursor += 1;
    var found: [*c]u8 = null;
    var success = false;
    defer if (!success) c.free(found);
    while (true) {
        whitespace(&cursor);
        if (cursor[0] == '}') {
            cursor += 1;
            break;
        }
        const key = string(&cursor) catch return null;
        defer c.free(key);
        whitespace(&cursor);
        if (cursor[0] != ':') return null;
        cursor += 1;
        const value = rawValue(&cursor) catch return null;
        if (equal(key, text(wanted))) {
            c.free(found);
            found = value;
        } else c.free(value);
        whitespace(&cursor);
        if (cursor[0] == ',') {
            cursor += 1;
            continue;
        }
        if (cursor[0] == '}') {
            cursor += 1;
            break;
        }
        return null;
    }
    whitespace(&cursor);
    if (cursor[0] != 0) return null;
    success = true;
    return found;
}
/// The existing Responses probe returns the FIRST matching string immediately;
/// it intentionally does not validate a later duplicate or trailing suffix.
fn stringField(object: [*c]const u8, wanted: []const u8) [*c]u8 {
    var cursor = object;
    whitespace(&cursor);
    if (cursor[0] != '{') return null;
    cursor += 1;
    while (cursor[0] != 0) {
        whitespace(&cursor);
        if (cursor[0] == '}') break;
        const key = string(&cursor) catch return null;
        defer c.free(key);
        whitespace(&cursor);
        if (cursor[0] != ':') return null;
        cursor += 1;
        if (equal(key, wanted)) return string(&cursor) catch null;
        if (!c.ds4_server_json_skip(&cursor)) return null;
        whitespace(&cursor);
        if (cursor[0] == ',') cursor += 1;
    }
    return null;
}
fn identifier(value: []const u8, maximum: usize) bool {
    if (value.len == 0 or value.len > maximum) return false;
    for (value) |character| switch (character) {
        'a'...'z', 'A'...'Z', '0'...'9', '_' => {},
        else => return false,
    };
    return true;
}
fn validId(value: []const u8) bool {
    return value.len > 5 and value.len <= 35 and std.mem.startsWith(u8, value, "conv_") and identifier(value[5..], 30);
}
fn unsigned(raw: [*c]const u8) ?c_ulonglong {
    if (raw == null or raw[0] < '0' or raw[0] > '9') return null;
    c.__error().* = 0;
    var end: [*c]u8 = null;
    const value = c.strtoull(raw, &end, 10);
    return if (c.__error().* != 0 or end == null or end[0] != 0) null else value;
}

const Items = struct {
    list: std.ArrayList([*c]u8) = .empty,
    fn deinit(self: *Items) void {
        for (self.list.items) |item| c.free(item);
        self.list.deinit(allocator);
    }
    fn append(self: *Items, item: [*c]u8) void {
        self.list.append(allocator, item) catch fatal();
    }
    fn json(self: *const Items) [*c]u8 {
        var out: Text = .{};
        defer out.deinit();
        out.byte('[');
        for (self.list.items, 0..) |item, position| {
            if (position != 0) out.byte(',');
            out.append(text(item));
        }
        out.byte(']');
        return out.take();
    }
};
fn array(raw: [*c]const u8) Invalid!Items {
    if (raw == null) return error.InvalidItems;
    var cursor = raw;
    whitespace(&cursor);
    if (cursor[0] != '[') return error.InvalidItems;
    cursor += 1;
    var result: Items = .{};
    errdefer result.deinit();
    while (true) {
        whitespace(&cursor);
        if (cursor[0] == ']') {
            cursor += 1;
            break;
        }
        if (cursor[0] != '{') return error.InvalidItems;
        result.append(try rawValue(&cursor));
        whitespace(&cursor);
        if (cursor[0] == ',') {
            cursor += 1;
            whitespace(&cursor);
            if (cursor[0] == ']') return error.InvalidItems;
            continue;
        }
        if (cursor[0] == ']') {
            cursor += 1;
            break;
        }
        return error.InvalidItems;
    }
    whitespace(&cursor);
    if (cursor[0] != 0) return error.InvalidItems;
    return result;
}
pub fn deinit(record: ?*Record) void {
    const value = record orelse return;
    c.free(value.object);
    c.free(value.items);
    c.free(value.replay);
    value.* = std.mem.zeroes(Record);
}
fn location(root: [*c]const u8, id: [*c]const u8, directory: *[c.PATH_MAX]u8, key: *[41]u8, errors: []u8) c_int {
    if (text(root).len == 0) return fail(errors, 500, "conversation store is not configured");
    if (!validId(text(id))) return fail(errors, 404, "conversation not found");
    const count = c.snprintf(directory, directory.len, "%s/conversations", root);
    if (count < 0 or count >= c.PATH_MAX) return fail(errors, 500, "conversation store path is too long");
    _ = c.snprintf(key, key.len, "resp_%s", id);
    return 0;
}
fn loadUnlocked(root: [*c]const u8, id: [*c]const u8, record: *Record, errors: []u8) c_int {
    record.* = std.mem.zeroes(Record);
    var directory: [c.PATH_MAX]u8 = undefined;
    var key: [41]u8 = undefined;
    const status = location(root, id, &directory, &key, errors);
    if (status != 0) return status;
    var raw: [*c]u8 = null;
    var length: usize = 0;
    const result = c.ds4_response_store_read(&directory, &key, &raw, &length, errorPointer(errors), errors.len);
    if (result != 0) return if (result == 1) fail(errors, 404, "conversation was not found") else 500;
    defer c.free(raw);
    record.object = jsonField(raw, "conversation");
    record.items = jsonField(raw, "items");
    record.replay = jsonField(raw, "replay");
    const version = jsonField(raw, "version");
    defer c.free(version);
    const deleted = jsonField(raw, "deleted");
    defer c.free(deleted);
    const object_id = if (record.object == null) null else stringField(record.object, "id");
    defer c.free(object_id);
    var ok = record.object != null and record.items != null and record.replay != null and object_id != null and equal(object_id, text(id));
    if (ok) {
        const parsed = unsigned(version);
        ok = parsed != null;
        if (parsed) |value| record.version = value;
        ok = ok and record.version > 0 and deleted != null and (equal(deleted, "true") or equal(deleted, "false"));
        if (ok) {
            if (array(record.items)) |values| {
                var owned = values;
                owned.deinit();
            } else |_| ok = false;
        }
    }
    record.deleted = deleted != null and equal(deleted, "true");
    if (record.replay != null and equal(record.replay, "null")) {
        c.free(record.replay);
        record.replay = null;
    }
    if (!ok) {
        deinit(record);
        return fail(errors, 500, "invalid conversation record");
    }
    return 0;
}
pub fn load(root: [*c]const u8, id: [*c]const u8, record: *Record, errors: []u8) c_int {
    _ = c.pthread_mutex_lock(&mutex);
    defer _ = c.pthread_mutex_unlock(&mutex);
    const status = loadUnlocked(root, id, record, errors);
    if (status == 0 and record.deleted) {
        deinit(record);
        return fail(errors, 404, "conversation was deleted");
    }
    return status;
}
fn writeUnlocked(root: [*c]const u8, id: [*c]const u8, record: *const Record, errors: []u8) c_int {
    var directory: [c.PATH_MAX]u8 = undefined;
    var key: [41]u8 = undefined;
    const status = location(root, id, &directory, &key, errors);
    if (status != 0) return status;
    if (c.ds4_response_store_init(&directory, errorPointer(errors), errors.len) != 0) return 500;
    var out: Text = .{};
    defer out.deinit();
    out.append("{\"conversation\":");
    out.append(text(record.object));
    out.append(",\"items\":");
    out.append(text(record.items));
    out.append(",\"replay\":");
    out.append(if (record.replay == null) "null" else text(record.replay));
    out.append(",\"version\":");
    out.number(record.version);
    out.append(",\"deleted\":");
    out.append(if (record.deleted) "true}" else "false}");
    const json = out.take();
    defer c.free(json);
    return if (c.ds4_response_store_write(&directory, &key, json, text(json).len, errorPointer(errors), errors.len) != 0) 500 else 0;
}

// Normalize easy strings and assign public item IDs without changing tool call
// IDs. The real Responses parser remains the replay validator; images reject.
fn normalize(raw: [*c]const u8) Invalid![*c]u8 {
    var easy: [*c]u8 = null;
    defer c.free(easy);
    var input = raw;
    var cursor = input;
    whitespace(&cursor);
    if (cursor[0] == '"') {
        const value = try string(&cursor);
        defer c.free(value);
        whitespace(&cursor);
        if (cursor[0] != 0) return error.InvalidItems;
        var out: Text = .{};
        defer out.deinit();
        out.append("[{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":");
        out.quoted(value);
        out.append("}]}]");
        easy = out.take();
        input = easy;
    }
    var values = try array(input);
    defer values.deinit();
    var messages = std.mem.zeroes(c.chat_msgs);
    defer c.ds4_server_input_messages_free(&messages);
    cursor = input;
    var ok = c.ds4_server_json_responses_input(&cursor, &messages, null, null, false);
    whitespace(&cursor);
    ok = ok and cursor[0] == 0;
    if (messages.len != 0) for (@as([*]c.chat_msg, @ptrCast(messages.v))[0..@intCast(messages.len)]) |message| {
        if (message.images.len != 0) ok = false;
    };
    if (!ok) return error.InvalidItems;
    for (values.list.items, 0..) |*item, position| {
        const supplied_kind = stringField(item.*, "type");
        const kind = if (supplied_kind == null) duplicate("message") else supplied_kind;
        defer c.free(kind);
        var id = stringField(item.*, "id");
        defer c.free(id);
        const role = stringField(item.*, "role");
        defer c.free(role);
        const message = equal(kind, "message");
        const supported = message or equal(kind, "function_call") or equal(kind, "function_call_output") or equal(kind, "reasoning");
        if (!supported or (role != null and !equal(role, "user") and !equal(role, "assistant") and !equal(role, "system") and !equal(role, "developer"))) return error.InvalidItems;
        if (id != null and !identifier(text(id), 128)) return error.InvalidItems;
        if (id == null) {
            var generated: [48]u8 = undefined;
            const prefix: [*c]const u8 = if (message) "msg_" else if (equal(kind, "function_call")) "fc_" else if (equal(kind, "function_call_output")) "fco_" else "rs_";
            c.ds4_server_output_random_id(&generated, generated.len, prefix);
            id = duplicate(text(&generated));
        }
        for (values.list.items[0..position]) |prior| {
            const old_id = stringField(prior, "id");
            defer c.free(old_id);
            if (old_id != null and equal(old_id, text(id))) return error.InvalidItems;
        }
        var out: Text = .{};
        defer out.deinit();
        out.append("{\"id\":");
        out.quoted(id);
        out.append(",\"type\":");
        out.quoted(kind);
        if (message) {
            out.append(",\"role\":");
            out.quoted(if (role == null) "user" else role);
        }
        const status = jsonField(item.*, "status");
        defer c.free(status);
        if (message and status == null) out.append(",\"status\":\"completed\"");
        var field: [*c]const u8 = item.*;
        whitespace(&field);
        field += 1;
        while (true) {
            whitespace(&field);
            if (field[0] == '}') break;
            const key = try string(&field);
            defer c.free(key);
            whitespace(&field);
            field += 1; // The object was validated above.
            const value = try rawValue(&field);
            defer c.free(value);
            if (!equal(key, "id") and !equal(key, "type") and !(message and equal(key, "role"))) {
                out.byte(',');
                out.quoted(key);
                out.byte(':');
                if (message and equal(key, "content") and value[0] == '"') {
                    var value_cursor: [*c]const u8 = value;
                    const decoded = string(&value_cursor) catch null;
                    defer c.free(decoded);
                    out.append(if (role != null and equal(role, "assistant")) "[{\"type\":\"output_text\",\"annotations\":[],\"text\":" else "[{\"type\":\"input_text\",\"text\":");
                    out.quoted(decoded);
                    out.append("}]");
                } else out.append(text(value));
            }
            whitespace(&field);
            if (field[0] == ',') {
                field += 1;
                continue;
            }
            break;
        }
        out.byte('}');
        c.free(item.*);
        item.* = out.take();
    }
    return values.json();
}
pub fn inputItems(raw: [*c]const u8, normalized: *[*c]u8, errors: []u8) bool {
    normalized.* = null;
    if (raw == null) return false;
    normalized.* = normalize(raw) catch {
        _ = fail(errors, 400, "conversation items require supported text/function items");
        return false;
    };
    return true;
}
fn concat(first: [*c]const u8, second: [*c]const u8, third: [*c]const u8) Invalid![*c]u8 {
    var ids: Items = .{};
    defer ids.deinit();
    var out: Text = .{};
    defer out.deinit();
    out.byte('[');
    var any = false;
    for ([_][*c]const u8{ first, second, third }) |raw| {
        var values = try array(if (raw == null) "[]" else raw);
        defer values.deinit();
        for (values.list.items) |item| {
            const id = stringField(item, "id");
            if (id == null) return error.InvalidItems;
            for (ids.list.items) |prior| {
                if (equal(prior, text(id))) {
                    c.free(id);
                    return error.InvalidItems;
                }
            }
            ids.append(id);
            if (any) out.byte(',');
            out.append(text(item));
            any = true;
        }
    }
    out.byte(']');
    return out.take();
}
pub fn publish(root: [*c]const u8, id: [*c]const u8, expected: c_ulonglong, input: [*c]const u8, output: [*c]const u8, replay: [*c]const u8, errors: []u8) c_int {
    var normalized_input: [*c]u8 = null;
    var normalized_output: [*c]u8 = null;
    defer c.free(normalized_input);
    defer c.free(normalized_output);
    if (!inputItems(if (input == null) "[]" else input, &normalized_input, errors) or !inputItems(if (output == null) "[]" else output, &normalized_output, errors))
        return fail(errors, 500, "invalid completed conversation publication");
    _ = c.pthread_mutex_lock(&mutex);
    defer _ = c.pthread_mutex_unlock(&mutex);
    var record = std.mem.zeroes(Record);
    defer deinit(&record);
    const status = loadUnlocked(root, id, &record, errors);
    if (status != 0) return status;
    if (record.deleted) return fail(errors, 404, "conversation was deleted");
    if (record.version != expected) return fail(errors, 409, "conversation changed during response generation");
    if (record.version == std.math.maxInt(c_ulonglong)) return fail(errors, 500, "conversation version exhausted");
    const joined = concat(record.items, normalized_input, normalized_output) catch return fail(errors, 500, "invalid conversation items");
    c.free(record.items);
    record.items = joined;
    c.free(record.replay);
    record.replay = if (replay == null) null else duplicate(text(replay));
    record.version += 1;
    return writeUnlocked(root, id, &record, errors);
}

fn listing(values: *const Items, ascending: bool, limit: usize, after: ?[]const u8, found: *bool) [*c]u8 {
    const items = values.list.items;
    var start: usize = 0;
    found.* = after == null;
    if (after) |cursor| for (0..items.len) |position| {
        const at = if (ascending) position else items.len - 1 - position;
        const id = stringField(items[at], "id");
        defer c.free(id);
        if (id != null and equal(id, cursor)) {
            start = position + 1;
            found.* = true;
            break;
        }
    };
    const count = @min(if (start <= items.len) items.len - start else 0, limit);
    var out: Text = .{};
    defer out.deinit();
    out.append("{\"object\":\"list\",\"data\":[");
    for (0..count) |position| {
        if (position != 0) out.byte(',');
        out.append(text(items[if (ascending) start + position else items.len - 1 - start - position]));
    }
    out.append("],\"first_id\":");
    const first = if (count == 0) null else stringField(items[if (ascending) start else items.len - 1 - start], "id");
    defer c.free(first);
    const last = if (count == 0) null else stringField(items[if (ascending) start + count - 1 else items.len - start - count], "id");
    defer c.free(last);
    if (first == null) out.append("null") else out.quoted(first);
    out.append(",\"last_id\":");
    if (last == null) out.append("null") else out.quoted(last);
    out.append(",\"has_more\":");
    out.append(if (start + count < items.len) "true}" else "false}");
    return out.take();
}
fn create(root: [*c]const u8, body: [*c]const u8, public: *[*c]u8, errors: []u8) c_int {
    const source: [*c]const u8 = if (body == null) "{}" else body;
    var cursor = source;
    whitespace(&cursor);
    if (cursor[0] != '{' or !c.ds4_server_json_skip(&cursor)) return 400;
    whitespace(&cursor);
    if (cursor[0] != 0) return 400;
    const raw_items = jsonField(source, "items");
    defer c.free(raw_items);
    const metadata = jsonField(source, "metadata");
    defer c.free(metadata);
    var normalized: [*c]u8 = null;
    if (!inputItems(if (raw_items == null) "[]" else raw_items, &normalized, errors)) return 400;
    var record = std.mem.zeroes(Record);
    record.items = normalized;
    defer deinit(&record);
    var values = array(normalized) catch unreachable;
    defer values.deinit();
    if (values.list.items.len > 20 or (metadata != null and metadata[0] != '{' and !equal(metadata, "null"))) return fail(errors, 400, "invalid initial items or metadata");
    var id: [40]u8 = undefined;
    c.ds4_server_output_random_id(&id, id.len, "conv_");
    var out: Text = .{};
    defer out.deinit();
    out.append("{\"id\":");
    out.quoted(&id);
    out.append(",\"object\":\"conversation\",\"created_at\":");
    out.number(@as(c_longlong, @intCast(c.time(null))));
    out.append(",\"metadata\":");
    out.append(if (metadata != null and !equal(metadata, "null")) text(metadata) else "{}");
    out.byte('}');
    record.object = out.take();
    record.version = 1;
    _ = c.pthread_mutex_lock(&mutex);
    const status = writeUnlocked(root, &id, &record, errors);
    _ = c.pthread_mutex_unlock(&mutex);
    if (status == 0) public.* = duplicate(text(record.object));
    return if (status != 0) status else 200;
}
fn appendItems(root: [*c]const u8, id: [*c]const u8, body: [*c]const u8, record: *Record, public: *[*c]u8, errors: []u8) c_int {
    const raw = jsonField(if (body == null) "{}" else body, "items");
    defer c.free(raw);
    var normalized: [*c]u8 = null;
    defer c.free(normalized);
    if (raw == null or !inputItems(raw, &normalized, errors)) return 400;
    const joined = concat(record.items, normalized, null) catch return 400;
    defer c.free(joined);
    var validated: [*c]u8 = null;
    if (!inputItems(joined, &validated, errors)) return 400;
    c.free(record.items);
    record.items = validated;
    c.free(record.replay);
    record.replay = null;
    record.version +%= 1; // Route edits wrap, unlike the guarded publish CAS.
    const status = writeUnlocked(root, id, record, errors);
    if (status == 0) {
        var added = array(normalized) catch unreachable;
        defer added.deinit();
        var ignored = false;
        public.* = listing(&added, true, added.list.items.len, null, &ignored);
    }
    return status;
}
/// 0 means another route; otherwise the HTTP status is authoritative. Whole
/// deletion leaves a tombstone with item bytes intact, including GET item access.
pub fn route(root: [*c]const u8, method: [*c]const u8, path: [*c]const u8, body: [*c]const u8, public: *[*c]u8, errors: []u8) c_int {
    public.* = null;
    var selected = text(path);
    if (std.mem.startsWith(u8, selected, "/v1/")) selected = selected[3..];
    if (!std.mem.startsWith(u8, selected, "/conversations") or
        (selected.len > 14 and selected[14] != '/' and selected[14] != '?')) return 0;
    if (text(root).len == 0) return fail(errors, 500, "conversation store is not configured");
    const suffix = selected[14..];
    const question = std.mem.findScalar(u8, suffix, '?');
    const address = suffix[0..(question orelse suffix.len)];
    var ascending = false;
    var limit: usize = 20;
    var after: ?[]const u8 = null;
    if (question) |position| {
        var parts = std.mem.tokenizeScalar(u8, suffix[position + 1 ..], '&');
        while (parts.next()) |part| {
            const equals = std.mem.findScalar(u8, part, '=') orelse continue;
            const key = part[0..equals];
            const value = part[equals + 1 ..];
            if (std.mem.eql(u8, key, "order")) {
                if (!std.mem.eql(u8, value, "asc") and !std.mem.eql(u8, value, "desc")) return fail(errors, 400, "invalid conversation list query");
                ascending = std.mem.eql(u8, value, "asc");
            } else if (std.mem.eql(u8, key, "limit")) {
                const terminated = duplicate(value);
                defer c.free(terminated);
                const parsed = unsigned(terminated);
                if (parsed == null or parsed.? < 1 or parsed.? > 100) return fail(errors, 400, "invalid conversation list query");
                limit = @intCast(parsed.?);
            } else if (std.mem.eql(u8, key, "after")) {
                if (!identifier(value, 128)) return fail(errors, 400, "invalid conversation list query");
                after = value;
            }
        }
    }
    if (address.len == 0) return if (equal(method, "POST")) create(root, body, public, errors) else 405;
    if (address[0] != '/') return 404;
    const slash = std.mem.findScalar(u8, address[1..], '/');
    const id = address[1..if (slash) |position| position + 1 else address.len];
    if (!validId(id)) return 404;
    const tail: ?[]const u8 = if (slash) |position| address[position + 2 ..] else null;
    var item_id: ?[]const u8 = null;
    if (tail) |value| {
        if (!std.mem.startsWith(u8, value, "items") or (value.len > 5 and value[5] != '/')) return 404;
        if (value.len > 5) {
            item_id = value[6..];
            if (!identifier(item_id.?, 128)) return 404;
        }
    }
    const terminated_id = duplicate(id);
    defer c.free(terminated_id);
    _ = c.pthread_mutex_lock(&mutex);
    defer _ = c.pthread_mutex_unlock(&mutex);
    var record = std.mem.zeroes(Record);
    defer deinit(&record);
    const status = loadUnlocked(root, terminated_id, &record, errors);
    if (status != 0) return status;
    if (record.deleted and (tail == null or !equal(method, "GET"))) return fail(errors, 404, "conversation was deleted");
    if (tail == null) {
        if (equal(method, "GET")) {
            public.* = duplicate(text(record.object));
            return 200;
        }
        if (!equal(method, "DELETE")) return 405;
        record.deleted = true;
        record.version +%= 1;
        const written = writeUnlocked(root, terminated_id, &record, errors);
        if (written != 0) return written;
        var out: Text = .{};
        defer out.deinit();
        out.append("{\"id\":");
        out.quoted(terminated_id);
        out.append(",\"object\":\"conversation.deleted\",\"deleted\":true}");
        public.* = out.take();
        return 200;
    }
    var values = array(record.items) catch return 500;
    defer values.deinit();
    if (item_id) |wanted| {
        var found: ?usize = null;
        for (values.list.items, 0..) |item, position| {
            const candidate = stringField(item, "id");
            defer c.free(candidate);
            if (candidate != null and equal(candidate, wanted)) {
                found = position;
                break;
            }
        }
        const position = found orelse return 404;
        if (equal(method, "GET")) {
            public.* = duplicate(text(values.list.items[position]));
            return 200;
        }
        if (!equal(method, "DELETE")) return 405;
        c.free(values.list.orderedRemove(position));
        c.free(record.items);
        record.items = values.json();
        c.free(record.replay);
        record.replay = null;
        record.version +%= 1;
        const written = writeUnlocked(root, terminated_id, &record, errors);
        if (written != 0) return written;
        public.* = duplicate(text(record.object));
        return 200;
    }
    if (equal(method, "GET")) {
        var found = false;
        public.* = listing(&values, ascending, limit, after, &found);
        if (!found) {
            c.free(public.*);
            public.* = null;
            return 404;
        }
        return 200;
    }
    if (equal(method, "POST")) {
        const written = appendItems(root, terminated_id, body, &record, public, errors);
        return if (written != 0) written else 200;
    }
    return 405;
}
