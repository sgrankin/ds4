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
        if (entry.block != null) self.releaseBlock(@ptrCast(entry.block));
        allocator.destroy(entry);
    }
    fn prune(self: Memory) void {
        while ((self.state.entries > maxEntries(self.state) or self.state.bytes > maxBytes(self.state)) and self.state.tail != null)
            self.remove(@ptrCast(self.state.tail));
    }
    fn putLocked(self: Memory, id: []const u8, text: []const u8, source: Source) void {
        if (id.len == 0 or text.len == 0) return;
        self.initLocked();
        if (self.findEntry(id)) |old| {
            if (old.block != null and std.mem.eql(u8, old.block.*.dsml[0..old.block.*.len], text)) {
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
        entry.bytes = id.len + 1 + @sizeOf(Entry);
        self.state.clock +%= 1;
        entry.stamp = self.state.clock;
        entry.source = source;
        entry.block_next = block.entries;
        block.entries = entry;
        block.refs += 1;
        if (c.raxInsert(self.state.by_id, entry.id, id.len, entry, null) == 0) {
            unlinkBlockEntry(block, entry);
            c.free(entry.id);
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
        self.putLocked(id, text, source);
    }
    pub fn remember(self: Memory, calls: *const c.tool_calls) void {
        const text = string(calls.raw_tool_text);
        if (text.len == 0) return;
        self.lock();
        defer self.unlock();
        for (callSlice(calls)) |call| self.putLocked(string(call.id), text, c.TOOL_MEMORY_RAM);
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
                } else if (matched != block) exact = false;
                if (found.source == c.TOOL_MEMORY_RAM) matched_source = c.TOOL_MEMORY_RAM;
            }
            // All IDs must identify the same immutable sampled block. Matching
            // each ID separately is insufficient for a multi-call assistant turn.
            if (exact and matched != null) {
                calls.raw_tool_text = duplicate(matched.?.dsml[0..matched.?.len]).ptr;
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
                result.bytes = std.math.add(u64, result.bytes, 8 + @as(u64, id_length) + block.len) catch return null;
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
                var lengths: [8]u8 = undefined;
                std.mem.writeInt(u32, lengths[0..4], @intCast(id.len), .little);
                std.mem.writeInt(u32, lengths[4..8], @intCast(block.len), .little);
                if (c.fwrite(&lengths, 1, lengths.len, file) != lengths.len or
                    c.fwrite(id.ptr, 1, id.len, file) != id.len or
                    c.fwrite(block.dsml, 1, block.len, file) != block.len) return false;
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
            var lengths: [8]u8 = undefined;
            if (c.fread(&lengths, 1, lengths.len, file) != lengths.len) return loaded;
            const id_length = std.mem.readInt(u32, lengths[0..4], .little);
            const text_length = std.mem.readInt(u32, lengths[4..8], .little);
            if (id_length == 0 or id_length > 256 or text_length == 0 or text_length > c.DS4_TOOL_MEMORY_MAX_BYTES) return loaded;
            const id = allocator.allocSentinel(u8, id_length, 0) catch fatal();
            defer allocator.free(id);
            const text = allocator.allocSentinel(u8, text_length, 0) catch fatal();
            defer allocator.free(text);
            if (c.fread(id.ptr, 1, id.len, file) != id.len or c.fread(text.ptr, 1, text.len, file) != text.len) return loaded;
            // The historical format accepts embedded NULs: insertion and wanted
            // matching use the C string prefix, though fread consumes all bytes.
            const id_string = string(id.ptr);
            if (wanted == null or wantedContains(wanted.?, id_string)) {
                self.put(id_string, string(text.ptr), c.TOOL_MEMORY_DISK);
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

const Forms = [_]struct { start: []const u8, end: []const u8 }{
    .{ .start = "\n\n<｜DSML｜ calls>", .end = "</｜DSML｜ calls>" },
    .{ .start = "<｜DSML｜ calls>", .end = "</｜DSML｜ calls>" },
    .{ .start = "\n\n<｜DSML｜tool_calls>", .end = "</｜DSML｜tool_calls>" },
    .{ .start = "<｜DSML｜tool_calls>", .end = "</｜DSML｜tool_calls>" },
    .{ .start = "\n\n<DSML｜tool_calls>", .end = "</DSML｜tool_calls>" },
    .{ .start = "<DSML｜tool_calls>", .end = "</DSML｜tool_calls>" },
    .{ .start = "\n\n<tool_calls>", .end = "</tool_calls>" },
    .{ .start = "<tool_calls>", .end = "</tool_calls>" },
};

/// Preserve earliest-start and first-form ties, including a preceding blank
/// line when present. Only complete blocks occur in saved trailer selection.
const Blocks = struct {
    memory: Memory,
    text: []const u8,
    offset: usize = 0,
    scan: u64,

    fn init(memory: Memory, text: []const u8) Blocks {
        memory.state.scan_clock +%= 1;
        return .{ .memory = memory, .text = text, .scan = memory.state.scan_clock };
    }
    fn next(self: *Blocks) ?*Block {
        while (true) {
            var best: ?usize = null;
            var end: usize = 0;
            for (Forms) |form| {
                const start = std.mem.findPos(u8, self.text, self.offset, form.start) orelse continue;
                if (best != null and start >= best.?) continue;
                const close = std.mem.findPos(u8, self.text, start, form.end) orelse continue;
                best = start;
                end = close + form.end.len;
            }
            const start = best orelse return null;
            self.offset = end;
            const block = self.memory.findBlock(self.text[start..end]) orelse continue;
            if (block.seen == self.scan) continue;
            block.seen = self.scan;
            return block;
        }
    }
};
