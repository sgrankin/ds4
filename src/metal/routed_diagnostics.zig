//! Optional selected-route binary replay/recording and merged hotlist files.
//! Process state lives only in RoutedSingle's sole entry. These diagnostics do
//! not change expert selection, cache ownership or shader arithmetic.
const std = @import("std");
const c = @import("metal_c");
const max_layers = c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER;
const max_experts = c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT;
const max_selected = c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED;
const max_u64 = std.math.maxInt(u64);
const allocator = std.heap.c_allocator;
const Entry = struct { layer: u32, expert: u32, hits: u64 };

fn pathFor(name: [*:0]const u8) ?[*:0]const u8 {
    const value = c.getenv(name);
    return if (value == null or value[0] == 0) null else @ptrCast(value);
}
fn earlier(_: void, a: Entry, b: Entry) bool {
    if (a.hits != b.hits) return a.hits > b.hits;
    if (a.layer != b.layer) return a.layer < b.layer;
    return a.expert < b.expert;
}
fn skipSpace(text: [*c]u8) [*c]u8 {
    var cursor = text;
    while (cursor[0] != 0 and c.isspace(cursor[0]) != 0) cursor += 1;
    return cursor;
}
const Columns = struct { layer: c_ulong, expert: c_ulong, hits: c_ulonglong };
fn columns(text: [*c]u8) ?Columns {
    var cursor = text;
    var end: [*c]u8 = null;
    c.__error().* = 0;
    const layer = c.strtoul(cursor, &end, 10);
    if (end == cursor or c.__error().* != 0) return null;
    cursor = skipSpace(end);
    c.__error().* = 0;
    const expert = c.strtoul(cursor, &end, 10);
    if (end == cursor or c.__error().* != 0) return null;
    cursor = skipSpace(end);
    c.__error().* = 0;
    const hits = c.strtoull(cursor, &end, 10);
    if (end == cursor or c.__error().* != 0) return null;
    return .{ .layer = layer, .expert = expert, .hits = hits };
}

pub const State = struct {
    record_initialized: bool = false,
    record_file: ?*c.FILE = null,
    record_count: u64 = 0,
    replay_initialized: bool = false,
    replay_ids: []i32 = &.{},
    replay_count: u64 = 0,
    replay_position: u64 = 0,
    hotlist_initialized: bool = false,
    hotlist_counts: [max_layers][max_experts]u64 = @splat(@splat(0)),
    hotlist_records: u64 = 0,
    hotlist_selections: u64 = 0,

    fn merge(self: *State, path: [*:0]const u8) bool {
        if (c.getenv("DS4_MOE_RECORD_SELECTED_HOTLIST_MERGE") == null or
            c.getenv("DS4_MOE_RECORD_SELECTED_HOTLIST_FRESH") != null) return true;
        const file = c.fopen(path, "rb") orelse {
            if (c.__error().* == c.ENOENT) return true;
            _ = c.fprintf(c.__stderrp, "ds4: failed to open selected hotlist merge file %s\n", path);
            return false;
        };
        var file_open = true;
        defer if (file_open) {
            _ = c.fclose(file);
        };
        var line: [256]u8 = undefined;
        var line_number: u64 = 0;
        var loaded_entries: u64 = 0;
        var loaded_hits: u64 = 0;
        var header_records: u64 = max_u64;
        var header_selections: u64 = max_u64;
        while (c.fgets(&line, line.len, file) != null) {
            line_number +%= 1;
            const cursor = skipSpace(&line);
            if (cursor[0] == 0) continue;
            if (cursor[0] == '#') {
                var value: c_ulonglong = 0;
                if (c.sscanf(cursor, "# layer_records %llu", &value) == 1) {
                    header_records = value;
                } else if (c.sscanf(cursor, "# selections %llu", &value) == 1) {
                    header_selections = value;
                }
                continue;
            }
            // Retain the accepted libc grammar, including ignored weight and
            // trailing fields. Only the first three numeric columns are read.
            const parsed = columns(cursor) orelse {
                _ = c.fprintf(c.__stderrp, "ds4: invalid selected hotlist merge line %llu in %s\n", @as(c_ulonglong, line_number), path);
                return false;
            };
            if (parsed.layer < max_layers and parsed.expert < max_experts and parsed.hits != 0) {
                self.hotlist_counts[parsed.layer][parsed.expert] +|= parsed.hits;
                loaded_entries +%= 1;
                loaded_hits +|= parsed.hits;
            }
        }
        if (c.ferror(file) != 0) {
            _ = c.fprintf(c.__stderrp, "ds4: failed to read selected hotlist merge file %s\n", path);
            return false;
        }
        // C closes before publishing totals and printing the merge summary.
        _ = c.fclose(file);
        file_open = false;
        self.hotlist_records = if (header_records != max_u64) header_records else loaded_hits / 6;
        self.hotlist_selections = if (header_selections != max_u64) header_selections else loaded_hits;
        _ = c.fprintf(c.__stderrp, "ds4: merged selected-id hotlist %s (%llu entries, %llu hits)\n", path, @as(c_ulonglong, loaded_entries), @as(c_ulonglong, loaded_hits));
        return true;
    }

    pub fn closeHotlist(self: *State) void {
        const path = pathFor("DS4_MOE_RECORD_SELECTED_HOTLIST") orelse return;
        if (!self.hotlist_initialized) return;
        var entries = allocator.alloc(Entry, max_layers * max_experts) catch {
            _ = c.fprintf(c.__stderrp, "ds4: failed to allocate selected hotlist entries\n");
            return;
        };
        defer allocator.free(entries);
        var count: usize = 0;
        for (&self.hotlist_counts, 0..) |*row, layer| {
            for (row, 0..) |hits, expert| {
                if (hits == 0) continue;
                entries[count] = .{ .layer = @intCast(layer), .expert = @intCast(expert), .hits = hits };
                count += 1;
            }
        }
        std.mem.sortUnstable(Entry, entries[0..count], {}, earlier);
        const file = c.fopen(path, "wb") orelse {
            _ = c.fprintf(c.__stderrp, "ds4: failed to open selected hotlist file %s\n", path);
            return;
        };
        _ = c.fprintf(file, "# ds4 selected-id hotlist v1\n# layer_records %llu\n# selections %llu\n# columns: layer expert hits weight\n", @as(c_ulonglong, self.hotlist_records), @as(c_ulonglong, self.hotlist_selections));
        for (entries[0..count]) |entry| {
            _ = c.fprintf(file, "%u %u %llu 0\n", entry.layer, entry.expert, @as(c_ulonglong, entry.hits));
        }
        allocator.free(entries);
        entries = &.{};
        if (c.fclose(file) != 0) {
            _ = c.fprintf(c.__stderrp, "ds4: failed to close selected hotlist file %s\n", path);
        } else {
            _ = c.fprintf(c.__stderrp, "ds4: wrote selected-id hotlist to %s (%llu layer records, %llu selections)\n", path, @as(c_ulonglong, self.hotlist_records), @as(c_ulonglong, self.hotlist_selections));
        }
    }

    pub fn recordHotlist(self: *State, layer: u32, ids: [*c]const i32, count: u32, total: u32) bool {
        const path = pathFor("DS4_MOE_RECORD_SELECTED_HOTLIST") orelse return true;
        if (ids == null or count == 0 or count > max_selected) return false;
        if (!self.hotlist_initialized) {
            self.hotlist_initialized = true;
            if (!self.merge(path)) return false;
            _ = c.atexit(c.ds4_native_metal_selected_hotlist_close);
        }
        if (layer >= max_layers) return true;
        self.hotlist_records +%= 1;
        for (ids[0..count]) |id| {
            if (id < 0) continue;
            const expert: u32 = @intCast(id);
            if (expert >= total or expert >= max_experts) continue;
            self.hotlist_counts[layer][expert] +%= 1;
            self.hotlist_selections +%= 1;
        }
        return true;
    }

    pub fn closeTrace(self: *State) void {
        const file = self.record_file orelse return;
        const path = pathFor("DS4_MOE_RECORD_SELECTED_IDS") orelse "(unknown)";
        _ = c.fclose(file);
        self.record_file = null;
        _ = c.fprintf(c.__stderrp, "ds4: recorded %llu routed-MoE selected-id entries to %s\n", @as(c_ulonglong, self.record_count), path);
    }

    pub fn recordTrace(self: *State, ids: [*c]const i32, count: u32) bool {
        const path = pathFor("DS4_MOE_RECORD_SELECTED_IDS") orelse return true;
        if (count != 6) {
            _ = c.fprintf(c.__stderrp, "ds4: selected-id recording expects exactly 6 selected experts\n");
            return false;
        }
        if (!self.record_initialized) {
            self.record_initialized = true;
            self.record_file = c.fopen(path, "wb");
            if (self.record_file == null) {
                _ = c.fprintf(c.__stderrp, "ds4: failed to open selected-id record file %s\n", path);
                return false;
            }
            _ = c.setvbuf(self.record_file, null, c._IOFBF, 1 << 20);
            _ = c.atexit(c.ds4_native_metal_selected_trace_close);
        }
        if (c.fwrite(ids, @sizeOf(i32), count, self.record_file) != count) {
            _ = c.fprintf(c.__stderrp, "ds4: failed to write selected-id record file %s\n", path);
            return false;
        }
        if (c.fflush(self.record_file) != 0) {
            _ = c.fprintf(c.__stderrp, "ds4: failed to flush selected-id record file %s\n", path);
            return false;
        }
        self.record_count +%= 1;
        return true;
    }

    pub fn replay(self: *State, ids: [*c]i32, count: u32) c_int {
        const path = pathFor("DS4_MOE_REPLAY_SELECTED_IDS") orelse return 0;
        if (count != 6) {
            _ = c.fprintf(c.__stderrp, "ds4: selected-id replay expects exactly 6 selected experts\n");
            return -1;
        }
        if (!self.replay_initialized) {
            self.replay_initialized = true;
            const file = c.fopen(path, "rb") orelse {
                _ = c.fprintf(c.__stderrp, "ds4: failed to open selected-id replay file %s\n", path);
                return -1;
            };
            var file_open = true;
            defer if (file_open) {
                _ = c.fclose(file);
            };
            if (c.fseeko(file, 0, c.SEEK_END) != 0) {
                _ = c.fprintf(c.__stderrp, "ds4: failed to seek selected-id replay file %s\n", path);
                return -1;
            }
            const end = c.ftello(file);
            if (end < 0) {
                _ = c.fprintf(c.__stderrp, "ds4: failed to size selected-id replay file %s\n", path);
                return -1;
            }
            if (c.fseeko(file, 0, c.SEEK_SET) != 0) {
                _ = c.fprintf(c.__stderrp, "ds4: failed to rewind selected-id replay file %s\n", path);
                return -1;
            }
            const size: u64 = @intCast(end);
            const entry_bytes = @as(u64, count) * @sizeOf(i32);
            if (size == 0 or size % entry_bytes != 0) {
                _ = c.fprintf(c.__stderrp, "ds4: selected-id replay file %s has invalid size %llu\n", path, @as(c_ulonglong, size));
                return -1;
            }
            if (size > std.math.maxInt(usize)) {
                _ = c.fprintf(c.__stderrp, "ds4: selected-id replay file %s is too large\n", path);
                return -1;
            }
            self.replay_count = size / entry_bytes;
            self.replay_ids = allocator.alloc(i32, @intCast(size / @sizeOf(i32))) catch {
                _ = c.fprintf(c.__stderrp, "ds4: failed to allocate selected-id replay buffer\n");
                return -1;
            };
            if (c.fread(self.replay_ids.ptr, 1, @intCast(size), file) != size) {
                _ = c.fprintf(c.__stderrp, "ds4: failed to read selected-id replay file %s\n", path);
                allocator.free(self.replay_ids);
                self.replay_ids = &.{};
                return -1;
            }
            _ = c.fclose(file);
            file_open = false;
            _ = c.fprintf(c.__stderrp, "ds4: loaded %llu routed-MoE selected-id entries from %s\n", @as(c_ulonglong, self.replay_count), path);
        }
        if (self.replay_position >= self.replay_count) {
            _ = c.fprintf(c.__stderrp, "ds4: selected-id replay exhausted after %llu entries\n", @as(c_ulonglong, self.replay_position));
            return -1;
        }
        const offset: usize = @intCast(self.replay_position *% count);
        @memcpy(ids[0..count], self.replay_ids[offset..][0..count]);
        self.replay_position +%= 1;
        return 1;
    }
};
