//! Indirect expert tables. The actual cache State and ARC table objects are
//! shared with the load path; this owner selects ranges and publishes addresses.
//! Every wrapped buffer and construction token has one scoped retain. Tables
//! keep all indirect resources alive, so SSD mode must not select Q4 tables.
const std = @import("std");
const c = @import("metal_c");
const Cache = @import("expert_cache.zig");
const State = Cache.State;
const Entry = Cache.Entry;
const Buffer = Cache.Buffer;
const Table = c.ds4_metal_table_ref;
const Residency = c.ds4_metal_residency_ref;
const Key = c.ds4_metal_table_key;
const Facts = c.ds4_metal_table_facts;
const max = std.math.maxInt(u64);
const layers = c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER;
const experts = c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT;
const selected_max = c.DS4_METAL_MAX_ROUTED_EXPERT_USED;
fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn errorLine(text: [*:0]const u8) bool {
    _ = c.fputs(text, c.__stderrp);
    return false;
}
fn words(buffer: Buffer) [*]u64 {
    return @ptrCast(@alignCast(c.ds4_metal_expert_buffer_storage(buffer).data.?));
}
fn bytes(buffer: Buffer) [*]u8 {
    return @ptrCast(c.ds4_metal_expert_buffer_storage(buffer).data.?);
}
pub fn groupSize(total: u32) u32 {
    const text = c.getenv("DS4_METAL_Q4_EXPERT_TABLE_GROUP_SIZE");
    if (text == null or text[0] == 0) return 1;
    var end: [*c]u8 = null;
    const value = c.strtoul(text, &end, 10);
    return if (end == text or end[0] != 0 or value < 2 or value > total) 1 else @intCast(value);
}
pub fn queueRequested() bool {
    return present("DS4_METAL_Q4_TABLE_QUEUE_RESIDENCY_SET");
}
pub fn modelResidency() bool {
    return present("DS4_METAL_Q4_TABLE_MODEL_RESIDENCY_SET");
}
pub fn bindAnchors() bool {
    return present("DS4_METAL_Q4_TABLE_BIND_ANCHORS");
}
pub fn nonStreamingOptIn() bool {
    return present("DS4_METAL_ENABLE_Q4_SELECTED_EXPERT_VIEWS") or present("DS4_METAL_ENABLE_PRO_Q4_SELECTED_EXPERT_VIEWS") or present("DS4_METAL_ENABLE_Q4_EXPERT_TABLE") or present("DS4_METAL_ENABLE_Q4_EXPERT_ADDRESS_TABLE") or present("DS4_METAL_ENABLE_PRO_Q4_EXPERT_TABLE_AUTO") or present("DS4_METAL_ENABLE_PRO_Q4_EXPERT_ADDRESS_AUTO");
}
pub fn selectedAllowed(f: Facts) bool {
    return f.streaming or (!f.glm and nonStreamingOptIn());
}
pub fn autoAvailable(f: Facts) bool {
    // Indirect tables retain every expert and its residency set. They cannot
    // coexist with a bounded SSD cache, even when only six experts are read.
    return !f.streaming and present("DS4_METAL_ENABLE_PRO_Q4_EXPERT_TABLE_AUTO") and !present("DS4_METAL_DISABLE_PRO_Q4_EXPERT_TABLE_AUTO") and !present("DS4_METAL_DISABLE_Q4_EXPERT_TABLE") and f.queue_residency;
}
pub fn queueEnabled(f: Facts, automatic: bool) bool {
    return (automatic or queueRequested()) and f.queue_residency;
}
pub fn indirectShape(total: u32, used: u32, gate: u64, down: u64) bool {
    return total == 384 and used == 6 and gate >= 2 << 30 and down >= 2 << 30;
}
pub fn tableAuto(f: Facts, total: u32, used: u32, gate: u64, down: u64) bool {
    // Automatic argument tables require every indirect expert to be made
    // visible up front. Exact selected slices remain the fallback otherwise.
    return indirectShape(total, used, gate, down) and autoAvailable(f);
}
pub fn addressAuto(f: Facts, total: u32, used: u32, gate: u64, down: u64) bool {
    // GPU addresses are an explicit experiment; active slices remain the
    // correctness/performance baseline until queue residency is available.
    return !f.streaming and present("DS4_METAL_ENABLE_PRO_Q4_EXPERT_ADDRESS_AUTO") and selectedAllowed(f) and indirectShape(total, used, gate, down) and f.address_pipelines and !present("DS4_METAL_DISABLE_PRO_Q4_EXPERT_TABLE_AUTO") and !present("DS4_METAL_DISABLE_PRO_Q4_EXPERT_ADDRESS_AUTO") and !present("DS4_METAL_DISABLE_Q4_EXPERT_ADDRESS_TABLE") and f.queue_residency;
}
pub fn fullRequested(streaming: bool) bool {
    return streaming and present("DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE") and !present("DS4_METAL_DISABLE_STREAMING_FULL_EXPERT_ADDR_TABLE");
}

pub const Tables = struct {
    state: *State,
    facts: Facts,

    fn cacheBuffers(self: Tables, layer: u32) [3]Buffer {
        return .{ self.state.cache_gate_addr_buffers[layer], self.state.cache_up_addr_buffers[layer], self.state.cache_down_addr_buffers[layer] };
    }
    pub fn ensureAddresses(self: Tables, layer: u32) bool {
        if (!self.facts.device or layer >= layers) return false;
        const labels = .{ "ds4_stream_expert_gate_addresses", "ds4_stream_expert_up_addresses", "ds4_stream_expert_down_addresses" };
        inline for (labels, 0..) |label, kind| {
            if (c.ds4_metal_tables_ensure_buffer(self.state, kind, layer, experts * 8, true, label) == null) return errorLine("ds4: Metal streaming expert address table allocation failed\n");
        }
        return true;
    }
    pub fn zeroAddress(self: Tables, layer: u32, expert: u32) void {
        if (layer >= layers or expert >= experts) return;
        for (self.cacheBuffers(layer)) |buffer| if (buffer != null) {
            words(buffer)[expert] = 0;
        };
    }
    pub fn setAddress(self: Tables, layer: u32, expert: u32, gate: Buffer, gate_inner: usize, up: Buffer, up_inner: usize, down: Buffer, down_inner: usize) bool {
        if (!self.ensureAddresses(layer)) return false;
        const values = .{ c.ds4_metal_tables_address(gate, gate_inner), c.ds4_metal_tables_address(up, up_inner), c.ds4_metal_tables_address(down, down_inner) };
        if (values[0] == 0 or values[1] == 0 or values[2] == 0) return errorLine("ds4: Metal streaming expert address table requires GPU addresses\n");
        inline for (self.cacheBuffers(layer), values) |buffer, value| words(buffer)[expert] = value;
        return true;
    }
    pub fn ensureCompact(self: Tables, layer: u32) bool {
        if (!self.facts.device or layer >= layers) return false;
        const labels = .{ "ds4_stream_compact_gate_addresses", "ds4_stream_compact_up_addresses", "ds4_stream_compact_down_addresses", "ds4_stream_compact_selected_ids" };
        inline for (labels, 0..) |label, i| {
            if (c.ds4_metal_tables_ensure_buffer(self.state, c.DS4_TABLE_COMPACT_GATE + i, layer, if (i == 3) 24 else 48, true, label) == null) return errorLine("ds4: Metal streaming compact address buffer allocation failed\n");
        }
        return true;
    }
    pub fn compact(self: Tables, layer: u32, entries: []const [*c]Entry) bool {
        if (layer >= layers or entries.len == 0 or entries.len > 6 or !self.ensureCompact(layer)) return false;
        var values: [3][6]u64 = @splat(@splat(0));
        for (entries, 0..) |pointer, i| {
            if (pointer == null) return false;
            const entry: *const Entry = @ptrCast(pointer);
            if (entry.gate_buffer == null or entry.up_buffer == null or entry.down_buffer == null) return false;
            values[0][i] = c.ds4_metal_tables_address(entry.gate_buffer, entry.gate_inner);
            values[1][i] = c.ds4_metal_tables_address(entry.up_buffer, entry.up_inner);
            values[2][i] = c.ds4_metal_tables_address(entry.down_buffer, entry.down_inner);
            if (values[0][i] == 0 or values[1][i] == 0 or values[2][i] == 0) return errorLine("ds4: Metal streaming compact address path requires GPU addresses\n");
        }
        const buffers = .{ self.state.compact_gate_addr_buffers[layer], self.state.compact_up_addr_buffers[layer], self.state.compact_down_addr_buffers[layer] };
        inline for (buffers, 0..) |buffer, i| @memcpy(words(buffer)[0..6], &values[i]);
        const slot_ids = [_]i32{ 0, 1, 2, 3, 4, 5 };
        @memcpy(bytes(self.state.compact_selected_buffers[layer])[0..24], std.mem.asBytes(&slot_ids));
        return true;
    }
    pub fn hostIds(self: Tables, layer: u32, ids: []const i32) bool {
        if (!self.facts.device or layer >= layers or ids.len == 0 or ids.len > selected_max) return false;
        const buffer = c.ds4_metal_tables_ensure_buffer(self.state, c.DS4_TABLE_HOST_IDS, layer, selected_max * 4, false, "ds4_stream_selected_ids");
        if (buffer == null) return errorLine("ds4: Metal streaming selected-id buffer allocation failed\n");
        var padded: [selected_max]i32 = @splat(0);
        @memcpy(padded[0..ids.len], ids);
        @memcpy(bytes(buffer)[0..@sizeOf(@TypeOf(padded))], std.mem.asBytes(&padded));
        return true;
    }
    pub fn clearFull(self: Tables, layer: u32) void {
        if (layer >= layers) return;
        const entry = &self.state.full_entries[layer];
        c.ds4_metal_expert_entry_retire(entry, null);
        entry.model_map = null;
        entry.model_size = 0;
        entry.gate_abs_offset = 0;
        entry.up_abs_offset = 0;
        entry.down_abs_offset = 0;
        entry.gate_expert_bytes = 0;
        entry.down_expert_bytes = 0;
        entry.logical_bytes = 0;
        entry.last_used = 0;
        entry.use_count = 0;
        entry.gate_inner = 0;
        entry.up_inner = 0;
        entry.down_inner = 0;
        entry.slab_slot = 0;
        entry.valid = 0;
        entry.slab_backed = 0;
    }
    pub fn full(self: Tables, key: Cache.Identity, layer: u32, total: u32) ?*Entry {
        if (!fullRequested(self.facts.streaming) or key.model == null or key.model_bytes == 0 or layer >= layers or total == 0 or total > experts or key.gate_bytes == 0 or key.down_bytes == 0 or total > max / key.gate_bytes or total > max / key.down_bytes) return null;
        const gate_bytes = @as(u64, total) * key.gate_bytes;
        const down_bytes = @as(u64, total) * key.down_bytes;
        if (key.gate > key.model_bytes or key.up > key.model_bytes or key.down > key.model_bytes or gate_bytes > key.model_bytes - key.gate or gate_bytes > key.model_bytes - key.up or down_bytes > key.model_bytes - key.down) return null;
        const entry = &self.state.full_entries[layer];
        if (!Cache.matches(entry, key)) {
            self.clearFull(layer);
            // Attempt all three wraps before testing failure, as the original
            // transaction does; each temporary owns one independent ARC retain.
            const gate = c.ds4_metal_tables_wrap(key.model, key.model_bytes, key.gate, gate_bytes, true);
            defer c.ds4_metal_tables_release_buffer(gate.buffer);
            const up = c.ds4_metal_tables_wrap(key.model, key.model_bytes, key.up, gate_bytes, true);
            defer c.ds4_metal_tables_release_buffer(up.buffer);
            const down = c.ds4_metal_tables_wrap(key.model, key.model_bytes, key.down, down_bytes, true);
            defer c.ds4_metal_tables_release_buffer(down.buffer);
            if (gate.buffer == null or up.buffer == null or down.buffer == null) return null;
            c.ds4_metal_expert_entry_retain(entry, gate.buffer, up.buffer, down.buffer);
            entry.model_map = key.model;
            entry.model_size = key.model_bytes;
            entry.gate_abs_offset = key.gate;
            entry.up_abs_offset = key.up;
            entry.down_abs_offset = key.down;
            entry.gate_expert_bytes = key.gate_bytes;
            entry.down_expert_bytes = key.down_bytes;
            entry.logical_bytes = gate_bytes *% 2 +% down_bytes;
            entry.gate_inner = @intCast(gate.inner);
            entry.up_inner = @intCast(up.inner);
            entry.down_inner = @intCast(down.inner);
            entry.valid = 1;
            if (!self.ensureAddresses(layer)) {
                self.clearFull(layer);
                return null;
            }
            const bases = .{ c.ds4_metal_tables_address(entry.gate_buffer, entry.gate_inner), c.ds4_metal_tables_address(entry.up_buffer, entry.up_inner), c.ds4_metal_tables_address(entry.down_buffer, entry.down_inner) };
            if (bases[0] == 0 or bases[1] == 0 or bases[2] == 0) {
                _ = errorLine("ds4: Metal full streaming expert address table requires GPU addresses\n");
                self.clearFull(layer);
                return null;
            }
            inline for (self.cacheBuffers(layer), bases, .{ key.gate_bytes, key.gate_bytes, key.down_bytes }) |buffer, base, stride| {
                for (words(buffer)[0..experts], 0..) |*address, expert| address.* = if (expert < total) base +% @as(u64, expert) *% stride else 0;
            }
        }
        const buffers = self.cacheBuffers(layer);
        return if (entry.valid != 0 and buffers[0] != null and buffers[1] != null and buffers[2] != null) entry else null;
    }

    pub fn q4(self: Tables, model: ?*const anyopaque, model_bytes: u64, offset: u64, expert_bytes: u64, total: u32, encoder: c.ds4_metal_argument_ref, address: bool) Table {
        if (self.facts.streaming or model == null or !self.facts.device or !self.facts.table_cache or (!address and encoder == null) or model_bytes == 0 or expert_bytes == 0 or total == 0 or total > 384) return null;
        const prefix: [*:0]const u8 = if (address) "ds4: Metal Q4 expert address table" else "ds4: Metal Q4 expert table";
        if (total > max / expert_bytes) {
            _ = c.fprintf(c.__stderrp, "%s byte size overflow\n", prefix);
            return null;
        }
        const tensor_bytes = @as(u64, total) * expert_bytes;
        if (offset > model_bytes or tensor_bytes > model_bytes - offset) {
            _ = c.fprintf(c.__stderrp, "%s is outside the mapped model\n", prefix);
            return null;
        }
        const group = groupSize(total);
        const key = Key{ .model = model, .model_bytes = model_bytes, .offset = offset, .expert_bytes = expert_bytes, .experts = total, .group = group, .encoded_bytes = if (address) 0 else c.ds4_metal_tables_encoded_bytes(encoder), .address = address };
        if (c.ds4_metal_tables_lookup(&key)) |cached| return cached;
        const capacity = if (group > 1) (total + group - 1) / group else total;
        const table = c.ds4_metal_tables_create(&key, encoder, capacity) orelse return null;
        defer c.ds4_metal_tables_drop(table);
        if (address and c.ds4_metal_tables_address_bytes(table) == null) return null;
        if (address and !self.facts.gpu_addresses) {
            _ = errorLine("ds4: Metal GPU addresses require macOS 13 or newer\n");
            return null;
        }
        var first: u32 = 0;
        while (first < total) : (first += group) {
            const count = @min(group, total - first);
            if (group > 1 or address) {
                if (first > max / expert_bytes or count > max / expert_bytes) {
                    _ = c.fprintf(c.__stderrp, "%s group byte overflow\n", prefix);
                    return null;
                }
            }
            const relative = @as(u64, first) * expert_bytes;
            const size = @as(u64, count) * expert_bytes;
            if (relative > max - offset) {
                _ = c.fprintf(c.__stderrp, "%s%s offset overflow\n", prefix, @as([*:0]const u8, if (group > 1 or address) " group" else ""));
                return null;
            }
            const wrapped = c.ds4_metal_tables_wrap(model, model_bytes, offset + relative, size, group == 1);
            defer c.ds4_metal_tables_release_buffer(wrapped.buffer);
            if (wrapped.buffer == null) return null;
            const base = if (address) c.ds4_metal_tables_address(wrapped.buffer, @intCast(wrapped.inner)) else 0;
            var i: u32 = 0;
            while (i < count) : (i += 1) {
                const inner = @as(u64, i) *% expert_bytes;
                if (address) {
                    const addresses: [*]u64 = @ptrCast(@alignCast(c.ds4_metal_tables_address_bytes(table).?));
                    addresses[first + i] = base +% inner;
                } else c.ds4_metal_tables_argument_buffer(encoder, wrapped.buffer, wrapped.inner +% inner, first + i);
            }
            c.ds4_metal_tables_append(table, wrapped.buffer);
        }
        if (present("DS4_METAL_Q4_EXPERT_TABLE_PROFILE")) _ = c.fprintf(c.__stderrp, "%s: experts=%u group=%u buffers=%lu expert_bytes=%.2f MiB\n", prefix, total, group, @as(c_ulong, c.ds4_metal_tables_regions(table).buffers), @as(f64, @floatFromInt(expert_bytes)) / (1024 * 1024));
        if (!address) {
            const residency = self.tensorResidency(table);
            if (residency != null) {
                defer c.ds4_metal_tables_residency_drop(residency);
                c.ds4_metal_tables_tensor_residency(table, residency);
            }
        }
        c.ds4_metal_tables_publish(&key, table);
        return table;
    }
    fn tensorResidency(self: Tables, table: Table) Residency {
        _ = self;
        const count = c.ds4_metal_tables_regions(table).buffers;
        if (count == 0 or !present("DS4_METAL_Q4_TABLE_RESIDENCY_SET") or !present("DS4_METAL_Q4_TABLE_PER_TENSOR_RESIDENCY_SET")) return null;
        const residency = c.ds4_metal_tables_residency_create(count, false) orelse return null;
        var buffers: [384]Buffer = undefined;
        const actual = c.ds4_metal_tables_buffers(table, &buffers);
        for (buffers[0..actual]) |buffer| c.ds4_metal_tables_residency_add(residency, buffer);
        c.ds4_metal_tables_residency_commit(residency);
        return residency;
    }
    pub fn layerResidency(self: Tables, gate: Table, up: Table, down: Table, automatic: bool) Residency {
        const queue = queueEnabled(self.facts, automatic);
        if (gate == null or up == null or down == null or !self.facts.device or !self.facts.layer_cache or (!present("DS4_METAL_Q4_TABLE_RESIDENCY_SET") and !queue) or !self.facts.residency_sets) return null;
        if (c.ds4_metal_tables_layer_lookup(gate, up, down)) |cached| return cached;
        const capacity = c.ds4_metal_tables_regions(gate).buffers + c.ds4_metal_tables_regions(up).buffers + c.ds4_metal_tables_regions(down).buffers + 3;
        const residency = c.ds4_metal_tables_residency_create(capacity, true) orelse return null;
        defer c.ds4_metal_tables_residency_drop(residency);
        for ([_]Table{ gate, up, down }) |table| {
            const regions = c.ds4_metal_tables_regions(table);
            if (regions.argument != null) c.ds4_metal_tables_residency_add(residency, regions.argument);
            if (regions.address != null) c.ds4_metal_tables_residency_add(residency, regions.address);
            var buffers: [384]Buffer = undefined;
            const count = c.ds4_metal_tables_buffers(table, &buffers);
            for (buffers[0..count]) |buffer| c.ds4_metal_tables_residency_add(residency, buffer);
        }
        c.ds4_metal_tables_residency_commit(residency);
        c.ds4_metal_tables_layer_publish(gate, up, down, residency, queue);
        return residency;
    }
    pub fn preload(self: Tables, model: ?*const anyopaque, model_bytes: u64, gate: u64, up: u64, down: u64, gate_bytes: u64, down_bytes: u64, total: u32) bool {
        if (model == null or model_bytes == 0 or gate_bytes == 0 or down_bytes == 0 or total == 0) return false;
        if (total > max / gate_bytes or total > max / down_bytes) return errorLine("ds4: Metal Q4 expert table preload byte size overflow\n");
        const address = addressAuto(self.facts, total, 6, @as(u64, total) * gate_bytes, @as(u64, total) * down_bytes);
        const argument = tableAuto(self.facts, total, 6, @as(u64, total) * gate_bytes, @as(u64, total) * down_bytes);
        if (!address and !argument) return true;
        if (!address and (self.facts.encoders[0] == null or self.facts.encoders[1] == null or self.facts.encoders[2] == null)) return errorLine("ds4: Metal Q4 expert table preload missing argument encoders\n");
        // Construct all three before testing failure, preserving partial cache
        // publication if a later resource cannot be prepared.
        const a = self.q4(model, model_bytes, gate, gate_bytes, total, self.facts.encoders[0], address);
        const b = self.q4(model, model_bytes, up, gate_bytes, total, self.facts.encoders[1], address);
        const d = self.q4(model, model_bytes, down, down_bytes, total, self.facts.encoders[2], address);
        if (a == null or b == null or d == null) return false;
        if (self.layerResidency(a, b, d, true) == null) return errorLine(if (address) "ds4: Metal Q4 expert address table preload failed to create queue residency set\n" else "ds4: Metal Q4 expert table preload failed to create queue residency set\n");
        return true;
    }
};

pub fn anchors(encoder: c.ds4_metal_table_encoder_ref, table: Table, first: usize, limit: usize) bool {
    if (!bindAnchors()) return true;
    if (encoder == null or table == null) return false;
    const regions = c.ds4_metal_tables_regions(table);
    if (!regions.has_buffers) return false;
    const count = regions.buffers;
    if (count > limit) {
        _ = c.fprintf(c.__stderrp, "ds4: Metal Q4 table anchor count %lu exceeds available slots %lu\n", @as(c_ulong, count), @as(c_ulong, limit));
        return false;
    }
    var buffers: [384]Buffer = undefined;
    const actual = c.ds4_metal_tables_buffers(table, &buffers);
    for (buffers[0..actual], 0..) |buffer, i| c.ds4_metal_tables_bind_buffer(encoder, buffer, first + i);
    return true;
}
pub fn useResources(command: c.ds4_metal_table_command_ref, encoder: c.ds4_metal_table_encoder_ref, table: Table, queue: bool) void {
    if (encoder == null or table == null) return;
    const regions = c.ds4_metal_tables_regions(table);
    if (regions.argument != null) c.ds4_metal_tables_use_buffer(encoder, regions.argument);
    if (regions.address != null) c.ds4_metal_tables_use_buffer(encoder, regions.address);
    if (!queue and present("DS4_METAL_Q4_TABLE_RESIDENCY_SET") and regions.residency != null and command != null) c.ds4_metal_tables_use_residency(command, regions.residency);
    if (!present("DS4_METAL_Q4_TABLE_USE_RESOURCES") and !present("DS4_METAL_Q4_ADDR_USE_RESOURCES")) return;
    var buffers: [384]Buffer = undefined;
    const count = c.ds4_metal_tables_buffers(table, &buffers);
    var first: usize = 0;
    while (first < count) : (first += 64) c.ds4_metal_tables_use_buffers(encoder, buffers[first..@min(first + 64, count)].ptr, @intCast(@min(64, count - first)));
}
