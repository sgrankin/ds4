//! Exact sampled tool-call text, shared by random protocol IDs.
//! Clients may reorder/rebuild JSON keys; replaying the sampled DSML verbatim
//! keeps their next tool result aligned with the live model's KV frontier.
//! This owner borrows the actual cache record and mutex. Server request policy,
//! model-scoped checkpoint discovery and KV state remain outside this module.
const std = @import("std");
const c = @import("server_tool_memory_c");
const allocator = std.heap.c_allocator;
const Record = c.tool_memory;
const Entry = c.tool_memory_entry;
const Block = c.tool_memory_block;
const Source = c.tool_memory_source;

fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
    c.exit(1);
}
fn duplicate(text: []const u8) [:0]u8 {
    return allocator.dupeSentinel(u8, text, 0) catch fatal();
}
fn string(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn maxEntries(state: *const Record) c_int {
    return if (state.max_entries > 0) state.max_entries else c.DS4_TOOL_MEMORY_DEFAULT_MAX_IDS;
}
fn maxBytes(state: *const Record) usize {
    return if (state.max_bytes > 0) state.max_bytes else c.DS4_TOOL_MEMORY_MAX_BYTES;
}
fn callSlice(calls: *const c.tool_calls) []const c.tool_call {
    return if (calls.len == 0) &.{} else @as([*]const c.tool_call, @ptrCast(calls.v))[0..@intCast(calls.len)];
}

fn sameRanges(entry: *const Entry, ranges: []const c.ds4_prompt_range) bool {
    if (entry.special_count != ranges.len) return false;
    for (ranges, 0..) |range, index| {
        const old = entry.special_ranges[index];
        if (range.begin != old.begin or range.end != old.end) return false;
    }
    return true;
}

pub const Memory = struct {
    state: *Record,
    // Only quiescent deinit lacks a mutex; all live actions borrow tool_mu.
    mutex: ?*c.pthread_mutex_t,

    fn lock(self: Memory) void {
        _ = c.pthread_mutex_lock(self.mutex.?);
    }
    fn unlock(self: Memory) void {
        _ = c.pthread_mutex_unlock(self.mutex.?);
    }
    fn initLocked(self: Memory) void {
        if (self.state.by_id != null and self.state.by_block != null) return;
        self.state.by_id = c.raxNew();
        self.state.by_block = c.raxNew();
        if (self.state.by_id == null or self.state.by_block == null) fatal();
    }
    fn linkHead(self: Memory, entry: *Entry) void {
        const state = self.state;
        entry.prev = null;
        entry.next = state.head;
        if (state.head != null) state.head.*.prev = entry else state.tail = entry;
        state.head = entry;
    }
    fn unlink(self: Memory, entry: *Entry) void {
        if (entry.prev != null) entry.prev.*.next = entry.next else self.state.head = entry.next;
        if (entry.next != null) entry.next.*.prev = entry.prev else self.state.tail = entry.prev;
        entry.prev = null;
        entry.next = null;
    }
    fn touch(self: Memory, entry: *Entry) void {
        self.state.clock +%= 1;
        entry.stamp = self.state.clock;
        if (self.state.head == entry) return;
        self.unlink(entry);
        self.linkHead(entry);
    }
    fn findEntry(self: Memory, id: []const u8) ?*Entry {
        if (self.state.by_id == null or id.len == 0) return null;
        const value = c.raxFind(self.state.by_id, @constCast(id.ptr), id.len);
        return if (value == c.raxNotFound) null else @ptrCast(@alignCast(value));
    }
    fn findBlock(self: Memory, text: []const u8) ?*Block {
        if (self.state.by_block == null or text.len == 0) return null;
        const value = c.raxFind(self.state.by_block, @constCast(text.ptr), text.len);
        return if (value == c.raxNotFound) null else @ptrCast(@alignCast(value));
    }
    fn getBlock(self: Memory, text: []const u8) *Block {
        if (self.findBlock(text)) |existing| return existing;
        const block = allocator.create(Block) catch fatal();
        block.* = std.mem.zeroes(Block);
        block.dsml = duplicate(text).ptr;
        block.len = text.len;
        block.bytes = text.len + 1 + @sizeOf(Block);
        if (c.raxInsert(self.state.by_block, block.dsml, block.len, block, null) == 0) {
            c.free(block.dsml);
            allocator.destroy(block);
            fatal();
        }
        self.state.bytes += block.bytes;
        return block;
    }
    fn releaseBlock(self: Memory, block: *Block) void {
        block.refs -= 1;
        if (block.refs > 0) return;
        if (self.state.by_block != null) _ = c.raxRemove(self.state.by_block, block.dsml, block.len, null);
        self.state.bytes -|= block.bytes;
        c.free(block.dsml);
        allocator.destroy(block);
    }
    fn unlinkBlockEntry(block: *Block, entry: *Entry) void {
        var link = &block.entries;
        while (link.* != null) {
            const current: *Entry = @ptrCast(link.*);
            if (current == entry) {
                link.* = entry.block_next;
                entry.block_next = null;
                return;
            }
            link = &current.block_next;
        }
    }
    fn remove(self: Memory, entry: *Entry) void {
        if (self.state.by_id != null and entry.id != null)
            _ = c.raxRemove(self.state.by_id, entry.id, string(entry.id).len, null);
        self.unlink(entry);
        if (entry.block != null) unlinkBlockEntry(@ptrCast(entry.block), entry);
        self.state.bytes -|= entry.bytes;
        if (self.state.entries > 0) self.state.entries -= 1;
        c.free(entry.id);
        c.free(entry.special_ranges);
        if (entry.block != null) self.releaseBlock(@ptrCast(entry.block));
        allocator.destroy(entry);
    }
    fn prune(self: Memory) void {
        while ((self.state.entries > maxEntries(self.state) or self.state.bytes > maxBytes(self.state)) and self.state.tail != null)
            self.remove(@ptrCast(self.state.tail));
    }
    fn putLocked(self: Memory, id: []const u8, text: []const u8, source: Source, ranges: []const c.ds4_prompt_range) void {
        if (id.len == 0 or text.len == 0) return;
        self.initLocked();
        if (self.findEntry(id)) |old| {
            if (old.block != null and std.mem.eql(u8, old.block.*.dsml[0..old.block.*.len], text) and sameRanges(old, ranges)) {
                // Disk replay must not demote an ID already sampled in RAM.
                if (source == c.TOOL_MEMORY_RAM) old.source = c.TOOL_MEMORY_RAM;
                self.touch(old);
                self.prune();
                return;
            }
            self.remove(old);
        }
        const block = self.getBlock(text);
        const entry = allocator.create(Entry) catch fatal();
        entry.* = std.mem.zeroes(Entry);
        entry.id = duplicate(id).ptr;
        entry.block = block;
        entry.special_ranges = if (ranges.len == 0) null else (allocator.dupe(c.ds4_prompt_range, ranges) catch fatal()).ptr;
        entry.special_count = ranges.len;
        entry.bytes = id.len + 1 + @sizeOf(Entry) + ranges.len * @sizeOf(c.ds4_prompt_range);
        self.state.clock +%= 1;
        entry.stamp = self.state.clock;
        entry.source = source;
        entry.block_next = block.entries;
        block.entries = entry;
        block.refs += 1;
        if (c.raxInsert(self.state.by_id, entry.id, id.len, entry, null) == 0) {
            unlinkBlockEntry(block, entry);
            c.free(entry.id);
            c.free(entry.special_ranges);
            allocator.destroy(entry);
            self.releaseBlock(block);
            fatal();
        }
        self.linkHead(entry);
        self.state.entries += 1;
        self.state.bytes += entry.bytes;
        self.prune();
    }

    /// No lock during shutdown: the Server has already stopped its users.
    pub fn deinit(state: *Record) void {
        // Removal itself does not use mutex; avoid inventing an initialized one.
        while (state.tail != null) remove(.{ .state = state, .mutex = null }, @ptrCast(state.tail));
        if (state.by_id != null) c.raxFree(state.by_id);
        if (state.by_block != null) c.raxFree(state.by_block);
        state.* = std.mem.zeroes(Record);
    }
    pub fn contains(self: Memory, id: []const u8) bool {
        if (id.len == 0) return false;
        self.lock();
        defer self.unlock();
        return self.findEntry(id) != null; // Existence checks do not refresh LRU.
    }
    pub fn put(self: Memory, id: []const u8, text: []const u8, source: Source) void {
        if (id.len == 0 or text.len == 0) return;
        self.lock();
        defer self.unlock();
        self.putLocked(id, text, source, &.{});
    }
    pub fn remember(self: Memory, calls: *const c.tool_calls) void {
        const text = string(calls.raw_tool_text);
        if (text.len == 0) return;
        self.lock();
        defer self.unlock();
        const ranges: []const c.ds4_prompt_range = if (calls.raw_special_count == 0) &.{} else calls.raw_special_ranges[0..calls.raw_special_count];
        for (callSlice(calls)) |call| self.putLocked(string(call.id), text, c.TOOL_MEMORY_RAM, ranges);
    }
    pub fn attach(self: Memory, messages: *c.chat_msgs, stats: ?*c.tool_replay_stats) void {
        self.lock();
        defer self.unlock();
        const items: []c.chat_msg = if (messages.len == 0) &.{} else @as([*]c.chat_msg, @ptrCast(messages.v))[0..@intCast(messages.len)];
        for (items) |*message| {
            const calls = &message.calls;
            if (calls.len == 0 or calls.raw_tool_text != null) continue;
            var matched: ?*Block = null;
            var matched_source: Source = c.TOOL_MEMORY_DISK;
            var matched_entry: ?*Entry = null;
            var exact = true;
            var missing: c_int = 0;
            for (callSlice(calls)) |call| {
                const entry = self.findEntry(string(call.id));
                if (entry == null or entry.?.block == null) {
                    exact = false;
                    missing += 1;
                    continue;
                }
                const found = entry.?;
                self.touch(found);
                const block: *Block = @ptrCast(found.block);
                if (matched == null) {
                    matched = block;
                    matched_source = found.source;
                    matched_entry = found;
                } else if (matched != block or !sameRanges(matched_entry.?, if (found.special_count == 0) &.{} else found.special_ranges[0..found.special_count])) exact = false;
                if (found.source == c.TOOL_MEMORY_RAM) matched_source = c.TOOL_MEMORY_RAM;
            }
            // All IDs must identify the same immutable sampled block. Matching
            // each ID separately is insufficient for a multi-call assistant turn.
            if (exact and matched != null) {
                calls.raw_tool_text = duplicate(matched.?.dsml[0..matched.?.len]).ptr;
                const found = matched_entry.?;
                if (found.special_count != 0) {
                    const copied = allocator.dupe(c.ds4_prompt_range, found.special_ranges[0..found.special_count]) catch fatal();
                    calls.raw_special_ranges = copied.ptr;
                    calls.raw_special_count = copied.len;
                }
                if (stats) |value| {
                    if (matched_source == c.TOOL_MEMORY_RAM) value.mem += 1 else value.disk += 1;
                }
            } else if (stats) |value| {
                value.canonical += 1;
                value.missing_ids += missing;
            }
        }
    }

    const Measure = struct { count: u32, bytes: u64 };
    fn measure(self: Memory, text: []const u8) ?Measure {
        var result: Measure = .{ .count = 0, .bytes = c.KV_TOOL_MAP_HEADER };
        var blocks = Blocks.init(self, text);
        while (blocks.next()) |block| {
            var current = block.entries;
            while (current != null) : (current = current.*.block_next) {
                const id_length = string(current.*.id).len;
                if (id_length > std.math.maxInt(u32) or block.len > std.math.maxInt(u32)) continue;
                if (result.count == std.math.maxInt(u32)) return null;
                result.bytes = std.math.add(u64, result.bytes, 12 + @as(u64, id_length) + block.len + 16 * @as(u64, current.*.special_count)) catch return null;
                result.count += 1;
            }
        }
        if (result.count == 0) result.bytes = 0;
        return result;
    }
    pub fn serializedSize(self: Memory, text: []const u8) ?u64 {
        if (text.len == 0) return 0;
        self.lock();
        defer self.unlock();
        return (self.measure(text) orelse return null).bytes;
    }
    pub fn write(self: Memory, file: *c.FILE, text: []const u8, written: ?*u64) bool {
        if (written) |value| value.* = 0;
        if (text.len == 0) return true;
        self.lock();
        defer self.unlock();
        const measured = self.measure(text) orelse return false;
        if (measured.count == 0) return true;
        var header = [_]u8{ c.KV_TOOL_MAP_MAGIC0, c.KV_TOOL_MAP_MAGIC1, c.KV_TOOL_MAP_MAGIC2, c.KV_TOOL_MAP_VERSION, 0, 0, 0, 0 };
        std.mem.writeInt(u32, header[4..8], measured.count, .little);
        var blocks = Blocks.init(self, text);
        if (c.fwrite(&header, 1, header.len, file) != header.len) return false;
        while (blocks.next()) |block| {
            var current = block.entries;
            while (current != null) : (current = current.*.block_next) {
                const id = string(current.*.id);
                if (id.len > std.math.maxInt(u32) or block.len > std.math.maxInt(u32)) continue;
                var lengths: [12]u8 = undefined;
                std.mem.writeInt(u32, lengths[0..4], @intCast(id.len), .little);
                std.mem.writeInt(u32, lengths[4..8], @intCast(block.len), .little);
                std.mem.writeInt(u32, lengths[8..12], @intCast(current.*.special_count), .little);
                if (c.fwrite(&lengths, 1, lengths.len, file) != lengths.len or
                    c.fwrite(id.ptr, 1, id.len, file) != id.len or
                    c.fwrite(block.dsml, 1, block.len, file) != block.len) return false;
                for (0..current.*.special_count) |index| {
                    const range = current.*.special_ranges[index];
                    var encoded_range: [16]u8 = undefined;
                    std.mem.writeInt(u64, encoded_range[0..8], range.begin, .little);
                    std.mem.writeInt(u64, encoded_range[8..16], range.end, .little);
                    if (c.fwrite(&encoded_range, 1, encoded_range.len, file) != encoded_range.len) return false;
                }
            }
        }
        if (written) |value| value.* = measured.bytes;
        return true;
    }
    pub fn load(self: Memory, file: *c.FILE, wanted: ?*const c.stop_list) c_int {
        var header: [c.KV_TOOL_MAP_HEADER]u8 = undefined;
        if (c.fread(&header, 1, header.len, file) != header.len) return 0;
        if (!std.mem.eql(u8, header[0..4], &.{ c.KV_TOOL_MAP_MAGIC0, c.KV_TOOL_MAP_MAGIC1, c.KV_TOOL_MAP_MAGIC2, c.KV_TOOL_MAP_VERSION })) return 0;
        const count = std.mem.readInt(u32, header[4..8], .little);
        if (@as(u64, count) > @as(u64, @intCast(maxEntries(self.state))) * 4) return 0;
        var loaded: c_int = 0;
        for (0..count) |_| {
            var lengths: [12]u8 = undefined;
            if (c.fread(&lengths, 1, lengths.len, file) != lengths.len) return loaded;
            const id_length = std.mem.readInt(u32, lengths[0..4], .little);
            const text_length = std.mem.readInt(u32, lengths[4..8], .little);
            const range_count = std.mem.readInt(u32, lengths[8..12], .little);
            if (range_count > text_length) return loaded;
            if (id_length == 0 or id_length > 256 or text_length == 0 or text_length > c.DS4_TOOL_MEMORY_MAX_BYTES) return loaded;
            if (@as(u64, id_length) + text_length + 16 * @as(u64, range_count) > maxBytes(self.state)) return loaded;
            const id = allocator.allocSentinel(u8, id_length, 0) catch fatal();
            defer allocator.free(id);
            const text = allocator.allocSentinel(u8, text_length, 0) catch fatal();
            defer allocator.free(text);
            if (c.fread(id.ptr, 1, id.len, file) != id.len or c.fread(text.ptr, 1, text.len, file) != text.len) return loaded;
            const ranges = allocator.alloc(c.ds4_prompt_range, range_count) catch fatal();
            defer allocator.free(ranges);
            var previous: usize = 0;
            for (ranges) |*range| {
                var encoded_range: [16]u8 = undefined;
                if (c.fread(&encoded_range, 1, encoded_range.len, file) != encoded_range.len) return loaded;
                const begin = std.mem.readInt(u64, encoded_range[0..8], .little);
                const end = std.mem.readInt(u64, encoded_range[8..16], .little);
                if (begin < previous or end <= begin or end > text_length) return loaded;
                range.* = .{ .begin = @intCast(begin), .end = @intCast(end) };
                previous = range.end;
            }
            // The historical format accepts embedded NULs: insertion and wanted
            // matching use the C string prefix, though fread consumes all bytes.
            const id_string = string(id.ptr);
            if (wanted == null or wantedContains(wanted.?, id_string)) {
                self.lock();
                self.putLocked(id_string, string(text.ptr), c.TOOL_MEMORY_DISK, ranges);
                self.unlock();
                loaded += 1; // Counts accepted records even if LRU immediately prunes.
            }
        }
        return loaded;
    }
};

fn wantedContains(wanted: *const c.stop_list, id: []const u8) bool {
    if (id.len == 0) return false;
    const items: []const [*c]u8 = if (wanted.len == 0) &.{} else @as([*]const [*c]u8, @ptrCast(wanted.v))[0..@intCast(wanted.len)];
    for (items) |item| {
        if (item != null and std.mem.eql(u8, string(item), id)) return true;
    }
    return false;
}

// Do not rediscover a block's closing delimiter from text: literal parameters
// can contain the same spelling. Match the complete immutable sampled block.
const Blocks = struct {
    memory: Memory,
    text: []const u8,
    current: [*c]Entry,
    scan: u64,

    fn init(memory: Memory, text: []const u8) Blocks {
        memory.state.scan_clock +%= 1;
        return .{ .memory = memory, .text = text, .current = memory.state.head, .scan = memory.state.scan_clock };
    }
    fn next(self: *Blocks) ?*Block {
        while (self.current != null) {
            const entry = self.current;
            self.current = entry.*.next;
            if (entry.*.block == null) continue;
            const block: *Block = @ptrCast(entry.*.block);
            if (block.seen == self.scan) continue;
            block.seen = self.scan;
            if (std.mem.indexOf(u8, self.text, block.dsml[0..block.len]) != null) return block;
        }
        return null;
    }
};
