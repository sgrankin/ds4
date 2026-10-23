//! Streaming expert entry lifetime and residency policy. All remaining C load
//! transactions borrow these SAME records. Buffer references are opaque and
//! borrowed here: only the ARC actions assign, transfer or release them.
const std = @import("std");
const c = @import("metal_c");
pub const State = c.ds4_metal_expert_cache_state;
pub const Entry = c.ds4_gpu_stream_expert_cache_entry;
pub const Reuse = c.ds4_gpu_stream_expert_reusable_buffers;
pub const Buffer = c.ds4_metal_expert_buffer_ref;
const layers = c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER;
const experts = c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT;
const slots = c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES;
const selected_max = c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_SELECTED;
const invalid = std.math.maxInt(u32);
// Pure imports in Cache and Load borrow one process owner from the sole entry.
const process = &c.ds4_native_expert_policy;
fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
pub fn timingSummary() bool {
    if (!process.timing_checked) {
        process.timing_enabled = (present("DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY") or present("DS4_METAL_STREAMING_EXPERT_PROFILE_SUMMARY")) and !present("DS4_METAL_DISABLE_STREAMING_EXPERT_TIMING_SUMMARY");
        process.timing_checked = true;
    }
    return process.timing_enabled;
}
pub fn decayTokens() u32 {
    if (process.decay_interval == 0) {
        process.decay_interval = c.DS4_METAL_STREAM_EXPERT_HOTNESS_DECAY_TOKENS;
        const value = c.getenv("DS4_METAL_STREAM_CACHE_DECAY_TOKENS");
        if (value != null and value[0] != 0) {
            var end: [*c]u8 = null;
            const parsed = c.strtoul(value, &end, 10);
            if (end != value and end[0] == 0 and parsed >= 1 and parsed <= 4096) process.decay_interval = @intCast(parsed) else _ = c.fprintf(c.__stderrp, "ds4: invalid cache decay interval; using 16 tokens\n");
        }
    }
    return process.decay_interval;
}
fn ids(pointer: [*c]const i32, count: u32) []const i32 {
    return if (pointer == null or count == 0) &.{} else pointer[0..count];
}
pub fn objectCount(gate: Buffer, up: Buffer, down: Buffer) u64 {
    if (gate == null or up == null or down == null) return 0;
    if (gate == up and gate == down) return 1;
    return if (gate == up or gate == down or up == down) 2 else 3;
}
fn buffersReady(reuse: *const Reuse) bool {
    return reuse.gate_buffer != null and reuse.up_buffer != null and reuse.down_buffer != null;
}
pub fn reusable(entry: *const Entry, gate_bytes: u64, down_bytes: u64) bool {
    return entry.valid != 0 and entry.gate_buffer != null and entry.up_buffer != null and entry.down_buffer != null and entry.gate_expert_bytes == gate_bytes and entry.down_expert_bytes == down_bytes;
}
pub const Identity = struct {
    model: ?*const anyopaque,
    model_bytes: u64,
    gate: u64,
    up: u64,
    down: u64,
    gate_bytes: u64,
    down_bytes: u64,
};
pub fn matches(entry: *const Entry, key: Identity) bool {
    return entry.valid != 0 and entry.model_map == key.model and entry.model_size == key.model_bytes and
        entry.gate_abs_offset == key.gate and entry.up_abs_offset == key.up and entry.down_abs_offset == key.down and
        entry.gate_expert_bytes == key.gate_bytes and entry.down_expert_bytes == key.down_bytes and
        entry.gate_buffer != null and entry.up_buffer != null and entry.down_buffer != null;
}
const Victim = struct { layer: u32 = invalid, expert: u32 = invalid, hotness: u32 = invalid, used: u64 = std.math.maxInt(u64) };
fn preferable(hotness: u32, used: u64, previous: Victim) bool {
    return hotness < previous.hotness or (hotness == previous.hotness and used < previous.used);
}

pub const Cache = struct {
    state: *State,
    streaming: bool,

    /// The allocator has one frozen size class. Mixed-precision outlier layers
    /// are rejected and use mapped experts; adopting them would poison reuse.
    pub fn noteSize(self: Cache, gate: u64, down: u64) bool {
        if (gate == 0 or down == 0) return false;
        if (gate > (std.math.maxInt(u64) - down) / 2) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal streaming expert cache byte size overflow\n");
            return false;
        }
        const bytes = gate * 2 + down;
        if (self.state.cache_expert_bytes == 0) {
            self.state.cache_expert_bytes = bytes;
            return true;
        }
        return bytes == self.state.cache_expert_bytes;
    }
    pub fn requested(self: Cache) u32 {
        return if (self.streaming) self.state.cache_budget_override else 0;
    }
    pub fn budget(self: Cache) u32 {
        const wanted = self.requested();
        const locked = self.state.cache_mlock_budget_cap;
        return if (wanted != 0 and locked != 0 and wanted > locked) locked else wanted;
    }
    pub fn configured(self: Cache) u32 {
        return @min(self.budget(), slots);
    }
    pub fn effectiveCap(self: Cache, total: u32, selected: u32) u32 {
        // Residency is GLOBAL: every layer may use any routed slot. A per-layer
        // budget would make capacity depend on model depth instead of bytes.
        if (self.budget() == 0) return 0;
        return @min(@max(total, selected), experts);
    }
    pub fn splitReady(self: Cache) bool {
        if (!self.streaming or self.state.cache_decode_tokens < 4) return false;
        const available = self.budget();
        const minimum = if (available != 0 and available < 2048) available / 2 else 1024;
        return self.state.cache_entry_count >= minimum;
    }
    pub fn decay(self: Cache) void {
        for (&self.state.cache_route_hotness) |*layer| for (layer) |*hotness| {
            hotness.* >>= 1;
        };
    }
    pub fn resetHotness(self: Cache) void {
        for (&self.state.cache_route_hotness) |*layer| @memset(layer, 0);
        self.state.cache_hotness_decay_token = self.state.cache_decode_tokens;
    }
    pub fn maybeDecay(self: Cache) void {
        const interval = decayTokens();
        const state = self.state;
        if (state.cache_decode_tokens == 0) return;
        if (state.cache_hotness_decay_token == 0) {
            state.cache_hotness_decay_token = state.cache_decode_tokens;
            return;
        }
        while (state.cache_decode_tokens -% state.cache_hotness_decay_token >= interval) {
            self.decay();
            state.cache_hotness_decay_token +%= interval;
        }
    }
    pub fn addHotness(self: Cache, layer: u32, expert: u32, amount: u32) void {
        if (layer >= layers or expert >= experts or amount == 0) return;
        self.state.cache_route_hotness[layer][expert] +|= amount;
    }
    pub fn noteSelected(self: Cache, layer: u32, selected: [*c]const i32, count: u32) void {
        if (selected == null or layer >= layers or count == 0) return;
        self.maybeDecay();
        for (selected[0..count]) |expert| if (expert >= 0 and expert < experts) {
            self.addHotness(layer, @intCast(expert), 1);
        };
    }
    pub fn noteFrequency(self: Cache, layer: u32, frequency: [*c]const u32, total: u32) void {
        if (frequency == null or layer >= layers) return;
        self.maybeDecay();
        for (frequency[0..@min(total, experts)], 0..) |amount, expert| self.addHotness(layer, @intCast(expert), amount);
    }
    pub fn noteToken(self: Cache, layer: u32) void {
        const state = self.state;
        if (!self.streaming or state.cache_decode_tokens == std.math.maxInt(u64)) return;
        // GLM's dense prefix and pipeline slices may start at different layers;
        // typed and generic routed paths must advance the SAME aging clock.
        state.cache_first_decode_layer = @min(state.cache_first_decode_layer, layer);
        if (layer != state.cache_first_decode_layer) return;
        state.cache_decode_tokens += 1;
        self.maybeDecay();
    }
    pub fn nextSequence(self: Cache) u64 {
        const state = self.state;
        if (state.cache_cb_seq == std.math.maxInt(u64)) {
            // Never let zero become a valid in-flight marker, even on the
            // astronomically distant command-buffer epoch wrap.
            state.cache_cb_seq = 0;
            state.cache_done_seq = 0;
            state.cache_batch_seq = 0;
            state.cache_owned_seq = 0;
            state.cache_pending_max_seq = 0;
            for (&state.entries) |*layer| for (layer) |*entry| {
                entry.inflight_seq = 0;
            };
        }
        state.cache_cb_seq += 1;
        return state.cache_cb_seq;
    }
    pub fn batchCreated(self: Cache) void {
        self.state.cache_batch_seq = self.nextSequence();
    }
    pub fn batchCommitted(self: Cache) void {
        self.state.cache_pending_max_seq = @max(self.state.cache_pending_max_seq, self.state.cache_batch_seq);
        self.state.cache_batch_seq = 0;
    }
    pub fn ownedCreated(self: Cache) void {
        self.state.cache_owned_seq = self.nextSequence();
    }
    pub fn pendingCompleted(self: Cache) void {
        self.state.cache_done_seq = @max(self.state.cache_done_seq, self.state.cache_pending_max_seq);
        self.state.cache_pending_max_seq = 0;
    }
    pub fn ownedCompleted(self: Cache) void {
        self.state.cache_done_seq = @max(self.state.cache_done_seq, self.state.cache_owned_seq);
        self.state.cache_owned_seq = 0;
    }
    pub fn inflight(self: Cache, entry: ?*const Entry) bool {
        return entry != null and entry.?.valid != 0 and entry.?.inflight_seq > self.state.cache_done_seq;
    }
    pub fn mark(self: Cache, maybe_entry: ?*Entry) bool {
        const entry = maybe_entry orelse return false;
        if (entry.valid == 0) return false;
        const sequence = if (self.state.cache_batch_seq != 0) self.state.cache_batch_seq else self.state.cache_owned_seq;
        if (sequence == 0) return false;
        entry.inflight_seq = sequence;
        return true;
    }
    pub fn markMany(self: Cache, entries: [*c]const [*c]Entry, count: u32, mask: u32) bool {
        if (entries == null or count == 0) return false;
        for (entries[0..count], 0..) |entry, index| {
            if (mask != 0 and mask & (@as(u32, 1) << @as(u5, @truncate(index))) == 0) continue;
            if (!self.mark(if (entry == null) null else @ptrCast(entry))) return false;
        }
        return true;
    }
    pub fn protected(self: Cache, layer: u32, expert: u32, protect_layer: u32, selected: []const i32) bool {
        if (layer < layers and expert < experts and self.inflight(&self.state.entries[layer][expert])) return true;
        return layer == protect_layer and protectedExpert(expert, selected);
    }
    pub fn slotRange(self: Cache, slot: u32) ?struct { slab: u32, base: u64 } {
        const state = self.state;
        for (0..state.cache_slab_count) |index| {
            const start = state.cache_slab_start_slot[index];
            const count = state.cache_slab_slot_count[index];
            if (slot < start or slot >= start +% count) continue;
            return .{ .slab = @intCast(index), .base = @as(u64, slot - start) *% state.cache_slab_slot_bytes };
        }
        return null;
    }
    pub fn slotForBuffer(self: Cache, buffer: Buffer, inner: usize) ?u32 {
        const state = self.state;
        if (buffer == null or state.cache_slab_slot_bytes == 0) return null;
        for (0..state.cache_slab_count) |index| {
            if (state.cache_slabs[index] != buffer) continue;
            const bytes = state.cache_slab_slot_bytes;
            if (bytes == 0 or inner % bytes != 0) return null;
            const local = inner / bytes;
            if (local >= state.cache_slab_slot_count[index]) return null;
            const slot = @as(u64, state.cache_slab_start_slot[index]) + local;
            return if (slot > std.math.maxInt(u32)) null else @intCast(slot);
        }
        return null;
    }
    pub fn recycle(self: Cache, slot: u32) void {
        const state = self.state;
        if (slot >= slots or state.cache_free_slot_count >= slots) return;
        state.cache_free_slots[state.cache_free_slot_count] = slot;
        state.cache_free_slot_count += 1;
    }
    pub fn unlockSlot(self: Cache, slot: u32) bool {
        const state = self.state;
        if (slot >= slots or state.cache_slab_slot_locked[slot] == 0) return true;
        const range = self.slotRange(slot) orelse return false;
        if (range.slab >= state.cache_slab_count or state.cache_slabs[range.slab] == null or state.cache_slab_slot_bytes == 0 or range.base > std.math.maxInt(usize)) return false;
        const memory = c.ds4_metal_expert_buffer_storage(state.cache_slabs[range.slab]);
        if (memory.data == null) return false;
        const pointer: [*]u8 = @ptrCast(memory.data.?);
        if (c.munlock(pointer + @as(usize, @intCast(range.base)), @intCast(state.cache_slab_slot_bytes)) != 0) return false;
        state.cache_slab_slot_locked[slot] = 0;
        state.cache_mlock_bytes -|= state.cache_slab_slot_bytes;
        return true;
    }
    pub fn unlockExplicit(self: Cache, buffer: Buffer) void {
        if (!self.streaming or buffer == null) return;
        const memory = c.ds4_metal_expert_buffer_storage(buffer);
        if (memory.data == null or memory.bytes == 0 or c.munlock(memory.data, @intCast(memory.bytes)) != 0) return;
        self.state.cache_mlock_bytes -|= memory.bytes;
    }
    pub fn evictEnabled(self: Cache) bool {
        return self.streaming and present("DS4_METAL_ENABLE_STREAMING_EXPERT_EVICT_DONTNEED") and !present("DS4_METAL_DISABLE_STREAMING_EXPERT_EVICT_DONTNEED");
    }
    pub fn dontNeed(self: Cache, model: ?*const anyopaque, size: u64, offset: u64, length: u64) void {
        if (!self.streaming or !present("DS4_METAL_ENABLE_STREAMING_EXPERT_EVICT_DONTNEED") or present("DS4_METAL_DISABLE_STREAMING_EXPERT_EVICT_DONTNEED") or model == null or size == 0 or offset > size or length == 0 or length > size - offset) return;
        if (!@hasDecl(c, "POSIX_MADV_DONTNEED")) return;
        const page: u64 = @intCast(c.sysconf(c._SC_PAGESIZE));
        const aligned = offset & ~(page - 1);
        const leading = offset - aligned;
        if (length > std.math.maxInt(u64) - leading or leading + length > std.math.maxInt(u64) - (page - 1)) return;
        const count = @min((leading + length + page - 1) & ~(page - 1), size - aligned);
        if (count == 0 or count > std.math.maxInt(usize)) return;
        const base = @intFromPtr(model.?);
        if (aligned > std.math.maxInt(usize) - base) return;
        const pointer: ?*anyopaque = @ptrFromInt(base + @as(usize, @intCast(aligned)));
        const result = c.posix_madvise(pointer, @intCast(count), c.POSIX_MADV_DONTNEED);
        if (result == 0) self.state.cache_evict_advise_bytes +|= count else if (present("DS4_METAL_STREAMING_EXPERT_EVICT_DONTNEED_PROFILE")) _ = c.fprintf(c.__stderrp, "ds4: Metal streaming expert evict DONTNEED failed offset=%.2f GiB len=%.2f MiB: %s\n", @as(f64, @floatFromInt(offset)) / 1073741824.0, @as(f64, @floatFromInt(length)) / 1048576.0, c.strerror(result));
    }
    pub fn clear(self: Cache, layer: u32, expert: u32, eviction: bool, recycle_slot: bool, reuse: ?*Reuse) void {
        if (layer >= layers or expert >= experts) return;
        if (reuse) |target| c.ds4_metal_expert_reuse_reset(target);
        const entry = &self.state.entries[layer][expert];
        if (entry.valid == 0 or self.inflight(entry)) return;
        const bytes = entry.logical_bytes;
        self.dontNeed(entry.model_map, entry.model_size, entry.gate_abs_offset, entry.gate_expert_bytes);
        self.dontNeed(entry.model_map, entry.model_size, entry.up_abs_offset, entry.gate_expert_bytes);
        self.dontNeed(entry.model_map, entry.model_size, entry.down_abs_offset, entry.down_expert_bytes);
        c.ds4_metal_expert_zero_address(layer, expert);
        if (reuse == null) {
            if (entry.slab_backed != 0 and recycle_slot) self.recycle(entry.slab_slot) else if (entry.slab_backed == 0) {
                self.unlockExplicit(entry.gate_buffer);
                if (entry.up_buffer != entry.gate_buffer) self.unlockExplicit(entry.up_buffer);
                if (entry.down_buffer != entry.gate_buffer and entry.down_buffer != entry.up_buffer) self.unlockExplicit(entry.down_buffer);
            }
        }
        // The ARC move retains reuse's owners BEFORE clearing entry's refs.
        c.ds4_metal_expert_entry_retire(entry, reuse);
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
        entry.inflight_seq = 0;
        entry.slab_slot = 0;
        entry.valid = 0;
        entry.slab_backed = 0;
        self.state.cache_layer_count[layer] -|= 1;
        self.state.cache_entry_count -|= 1;
        self.state.cache_bytes -|= bytes;
        if (eviction) {
            self.state.cache_evictions +%= 1;
            self.state.cache_layer_evictions[layer] +%= 1;
        }
    }
    pub fn clearLayer(self: Cache, layer: u32) void {
        if (layer >= layers) return;
        for (0..experts) |expert| self.clear(layer, @intCast(expert), false, true, null);
        self.state.cache_layer_count[layer] = 0;
    }
    pub fn pruneLayer(self: Cache, layer: u32, total: u32, selected: u32, protect: []const i32) void {
        if (layer >= layers) return;
        const cap = self.effectiveCap(total, selected);
        if (cap == 0) return;
        // Route hotness counts misses too. Hit-only LFU penalizes repeatedly
        // selected experts that are evicted before receiving their second hit.
        while (self.state.cache_layer_count[layer] > cap) {
            var victim: Victim = .{};
            for (0..total) |expert| {
                const entry = &self.state.entries[layer][expert];
                if (entry.valid == 0 or self.inflight(entry) or protectedExpert(@intCast(expert), protect)) continue;
                const hotness = self.state.cache_route_hotness[layer][expert];
                if (preferable(hotness, entry.last_used, victim)) victim = .{ .layer = layer, .expert = @intCast(expert), .hotness = hotness, .used = entry.last_used };
            }
            if (victim.expert == invalid) break;
            self.clear(layer, victim.expert, true, true, null);
        }
    }
    pub fn pruneGlobal(self: Cache, protect_layer: u32, protect: []const i32) void {
        const budget_count = self.budget();
        if (budget_count == 0 or self.state.cache_entry_count <= budget_count) return;
        while (self.state.cache_entry_count > budget_count) {
            var victim: Victim = .{};
            for (&self.state.entries, 0..) |*layer_entries, layer| for (layer_entries, 0..) |*entry, expert| {
                if (entry.valid == 0 or self.protected(@intCast(layer), @intCast(expert), protect_layer, protect)) continue;
                const hotness = self.state.cache_route_hotness[layer][expert];
                if (preferable(hotness, entry.last_used, victim)) victim = .{ .layer = @intCast(layer), .expert = @intCast(expert), .hotness = hotness, .used = entry.last_used };
            };
            if (victim.layer == invalid or victim.expert == invalid) break;
            self.clear(victim.layer, victim.expert, true, true, null);
        }
    }
    pub fn peek(self: Cache, key: Identity, layer: u32, expert: u32, total: u32, selected: u32) ?*Entry {
        if (!self.streaming or layer >= layers or expert >= experts or expert >= total or !self.noteSize(key.gate_bytes, key.down_bytes) or self.effectiveCap(total, selected) == 0) return null;
        const entry = &self.state.entries[layer][expert];
        if (!matches(entry, key)) return null;
        self.state.cache_clock +%= 1;
        entry.last_used = self.state.cache_clock;
        entry.use_count +%= 1;
        self.state.cache_hits +%= 1;
        self.state.cache_layer_hits[layer] +%= 1;
        return entry;
    }
    pub fn snapshot(self: Cache, table: ?*const c.ds4_gpu_stream_expert_table, selected: ?[]const i32, output: [*c]u8) bool {
        const input = table orelse return false;
        if (!self.streaming or output == null or input.layer >= layers or input.n_total_expert == 0 or input.n_total_expert > experts or input.gate_expert_bytes == 0 or input.down_expert_bytes == 0) return false;
        if (selected != null and selected.?.len == 0) return false;
        const count = if (selected) |selection| selection.len else input.n_total_expert;
        for (0..count) |index| {
            const expert: u32 = if (selected) |selection| blk: {
                const id = selection[index];
                if (id < 0 or @as(u32, @intCast(id)) >= input.n_total_expert) return false;
                break :blk @intCast(id);
            } else @intCast(index);
            if (expert > (std.math.maxInt(u64) - input.gate_offset) / input.gate_expert_bytes or expert > (std.math.maxInt(u64) - input.up_offset) / input.gate_expert_bytes or expert > (std.math.maxInt(u64) - input.down_offset) / input.down_expert_bytes) return false;
            const key = Identity{ .model = input.model_map, .model_bytes = input.model_size, .gate = input.gate_offset + @as(u64, expert) * input.gate_expert_bytes, .up = input.up_offset + @as(u64, expert) * input.gate_expert_bytes, .down = input.down_offset + @as(u64, expert) * input.down_expert_bytes, .gate_bytes = input.gate_expert_bytes, .down_bytes = input.down_expert_bytes };
            output[index] = @intFromBool(matches(&self.state.entries[input.layer][expert], key));
        }
        return true;
    }
    pub fn install(self: Cache, key: Identity, layer: u32, expert: u32, gate: Buffer, up: Buffer, down: Buffer, gate_inner: usize, up_inner: usize, down_inner: usize) ?*Entry {
        if (gate == null or up == null or down == null or layer >= layers or expert >= experts) return null;
        if (key.gate_bytes > (std.math.maxInt(u64) - key.down_bytes) / 2) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal streaming expert cache byte size overflow\n");
            return null;
        }
        const bytes = key.gate_bytes * 2 + key.down_bytes;
        const entry = &self.state.entries[layer][expert];
        if (entry.valid != 0) {
            if (self.inflight(entry) and c.ds4_metal_expert_on_service_thread()) return null;
            if (self.inflight(entry) and !c.ds4_metal_expert_wait("streaming expert cache install")) return null;
            if (self.inflight(entry)) return null;
            self.clear(layer, expert, false, true, null);
            if (entry.valid != 0) return null;
        }
        if (!c.ds4_metal_expert_install_address(layer, expert, gate, gate_inner, up, up_inner, down, down_inner)) return null;
        c.ds4_metal_expert_entry_retain(entry, gate, up, down);
        entry.model_map = key.model;
        entry.model_size = key.model_bytes;
        entry.gate_abs_offset = key.gate;
        entry.up_abs_offset = key.up;
        entry.down_abs_offset = key.down;
        entry.gate_expert_bytes = key.gate_bytes;
        entry.down_expert_bytes = key.down_bytes;
        entry.logical_bytes = bytes;
        self.state.cache_clock +%= 1;
        entry.last_used = self.state.cache_clock;
        entry.use_count = 1;
        entry.gate_inner = gate_inner;
        entry.up_inner = up_inner;
        entry.down_inner = down_inner;
        entry.inflight_seq = 0;
        const slot = if (gate == up and gate == down) self.slotForBuffer(gate, gate_inner) else null;
        entry.slab_backed = @intFromBool(slot != null);
        entry.slab_slot = slot orelse 0;
        entry.valid = 1;
        self.state.cache_layer_count[layer] +%= 1;
        self.state.cache_entry_count +|= 1;
        self.state.cache_bytes +|= bytes;
        self.state.cache_misses +%= 1;
        self.state.cache_layer_misses[layer] +%= 1;
        self.state.cache_wraps +%= 3;
        return entry;
    }
    fn scanTiming(self: Cache, count: u64, start: f64) void {
        self.state.timing_reuse_scan_calls +%= 1;
        self.state.timing_reuse_scan_entries +%= count;
        self.state.timing_reuse_scan_ms += c.ds4_metal_expert_now_ms() - start;
    }
    fn clearTiming(self: Cache, start: f64) void {
        self.state.timing_reuse_clear_ms += c.ds4_metal_expert_now_ms() - start;
    }
    pub fn takeReusable(self: Cache, force: bool, protect_layer: u32, protect: []const i32, gate_bytes: u64, down_bytes: u64, reuse: ?*Reuse) bool {
        const target = reuse orelse return false;
        c.ds4_metal_expert_reuse_reset(target);
        const capacity = self.budget();
        if (capacity == 0 or (!force and self.state.cache_entry_count < capacity)) return false;
        var waited = false;
        while (true) {
            var victim: Victim = .{};
            var skipped_inflight = false;
            const timing = timingSummary();
            const start = if (timing) c.ds4_metal_expert_now_ms() else 0;
            var scanned: u64 = 0;
            for (&self.state.entries, 0..) |*layer_entries, layer| for (layer_entries, 0..) |*entry, expert| {
                scanned +%= 1;
                if (!reusable(entry, gate_bytes, down_bytes)) continue;
                if (self.inflight(entry)) {
                    skipped_inflight = true;
                    continue;
                }
                if (self.protected(@intCast(layer), @intCast(expert), protect_layer, protect)) continue;
                const hotness = self.state.cache_route_hotness[layer][expert];
                if (preferable(hotness, entry.last_used, victim)) victim = .{ .layer = @intCast(layer), .expert = @intCast(expert), .hotness = hotness, .used = entry.last_used };
            };
            if (timing) self.scanTiming(scanned, start);
            if (victim.layer == invalid or victim.expert == invalid) {
                // The asynchronous load service must never flush/wait against
                // the encoding thread. Its caller retries on the main thread.
                if (skipped_inflight and !waited and !c.ds4_metal_expert_on_service_thread()) {
                    waited = true;
                    if (!c.ds4_metal_expert_wait("streaming expert cache reuse")) return false;
                    continue;
                }
                return false;
            }
            const clear_start = if (timing) c.ds4_metal_expert_now_ms() else 0;
            self.clear(victim.layer, victim.expert, true, true, target);
            if (timing) self.clearTiming(clear_start);
            if (!buffersReady(target)) {
                c.ds4_metal_expert_reuse_reset(target);
                return false;
            }
            self.state.cache_buffer_reuses +%= objectCount(target.gate_buffer, target.up_buffer, target.down_buffer);
            return true;
        }
    }
    /// Preserve the original bounded candidate array order, including strict
    /// hotness/LRU ties. It is not sorted before buffers are returned.
    pub fn takeBatch(self: Cache, needed: u32, protect_layer: u32, protect: []const i32, gate_bytes: u64, down_bytes: u64, reuses: [*c]Reuse) u32 {
        if (reuses == null or needed == 0) return 0;
        const count = @min(needed, selected_max);
        for (0..count) |index| c.ds4_metal_expert_reuse_reset(@ptrCast(&reuses[index]));
        const capacity = self.budget();
        if (capacity == 0 or self.state.cache_entry_count < capacity) return 0;
        var waited = false;
        while (true) {
            var victims: [selected_max]Victim = @splat(.{});
            var used: usize = 0;
            var skipped_inflight = false;
            const timing = timingSummary();
            const start = if (timing) c.ds4_metal_expert_now_ms() else 0;
            var scanned: u64 = 0;
            for (&self.state.entries, 0..) |*layer_entries, layer| for (layer_entries, 0..) |*entry, expert| {
                scanned +%= 1;
                if (!reusable(entry, gate_bytes, down_bytes)) continue;
                if (self.inflight(entry)) {
                    skipped_inflight = true;
                    continue;
                }
                if (self.protected(@intCast(layer), @intCast(expert), protect_layer, protect)) continue;
                const candidate = Victim{ .layer = @intCast(layer), .expert = @intCast(expert), .hotness = self.state.cache_route_hotness[layer][expert], .used = entry.last_used };
                if (used < count) {
                    victims[used] = candidate;
                    used += 1;
                    continue;
                }
                var worst: usize = 0;
                for (1..used) |index| if (victims[index].hotness > victims[worst].hotness or (victims[index].hotness == victims[worst].hotness and victims[index].used > victims[worst].used)) {
                    worst = index;
                };
                if (preferable(candidate.hotness, candidate.used, victims[worst])) victims[worst] = candidate;
            };
            if (timing) self.scanTiming(scanned, start);
            if (used == 0) {
                // This batch action keeps the historical caller contract: it
                // waits once when only in-flight candidates were skipped.
                if (skipped_inflight and !waited) {
                    waited = true;
                    if (!c.ds4_metal_expert_wait("streaming expert cache batch reuse")) return 0;
                    continue;
                }
                return 0;
            }
            var returned: u32 = 0;
            const clear_start = if (timing) c.ds4_metal_expert_now_ms() else 0;
            for (victims[0..used]) |victim| {
                if (victim.layer == invalid or victim.expert == invalid) continue;
                const target: *Reuse = @ptrCast(&reuses[returned]);
                self.clear(victim.layer, victim.expert, true, true, target);
                if (!buffersReady(target)) {
                    c.ds4_metal_expert_reuse_reset(target);
                    continue;
                }
                self.state.cache_buffer_reuses +%= objectCount(target.gate_buffer, target.up_buffer, target.down_buffer);
                returned += 1;
            }
            if (timing) self.clearTiming(clear_start);
            return returned;
        }
    }
    pub fn lockedCount(self: Cache) u32 {
        var count: u32 = 0;
        for (self.state.cache_slab_slot_locked[0..@min(self.state.cache_slab_total_slots, slots)]) |locked| if (locked != 0) {
            count += 1;
        };
        return count;
    }
    pub fn releaseMargin(self: Cache, protect_layer: u32, protect: []const i32) u32 {
        const state = self.state;
        if (state.cache_mlock_relief_applied != 0 or state.cache_slab_slot_bytes == 0) return 0;
        const locked_before = self.lockedCount();
        if (locked_before == 0) return 0;
        const target = (locked_before + 9) / 10;
        var released: u32 = 0;
        while (released < target) {
            var victim: Victim = .{};
            var slot: u32 = invalid;
            for (&state.entries, 0..) |*layer_entries, layer| for (layer_entries, 0..) |*entry, expert| {
                if (entry.valid == 0 or entry.slab_backed == 0 or entry.slab_slot >= slots or state.cache_slab_slot_locked[entry.slab_slot] == 0 or self.inflight(entry) or self.protected(@intCast(layer), @intCast(expert), protect_layer, protect)) continue;
                const hotness = state.cache_route_hotness[layer][expert];
                if (preferable(hotness, entry.last_used, victim)) {
                    victim = .{ .layer = @intCast(layer), .expert = @intCast(expert), .hotness = hotness, .used = entry.last_used };
                    slot = entry.slab_slot;
                }
            };
            if (victim.layer == invalid or victim.expert == invalid or slot == invalid) break;
            self.clear(victim.layer, victim.expert, true, false, null);
            if (state.entries[victim.layer][victim.expert].valid != 0) break;
            if (self.unlockSlot(slot)) released += 1;
        }
        if (released == 0) return 0;
        state.cache_mlock_relief_applied = 1;
        var cap = state.cache_entry_count;
        const locked_after = self.lockedCount();
        if (locked_after != 0 and locked_after < cap) cap = locked_after;
        if (cap != 0 and (state.cache_mlock_budget_cap == 0 or cap < state.cache_mlock_budget_cap)) state.cache_mlock_budget_cap = cap;
        const bytes = @as(u64, released) *% state.cache_slab_slot_bytes;
        _ = c.fprintf(c.__stderrp, "ds4:   released locked cache margin: %u experts / %.2f GiB; runtime cache cap now %u experts\n", released, @as(f64, @floatFromInt(bytes)) / 1073741824.0, state.cache_mlock_budget_cap);
        return released;
    }
    /// Keep the surviving slab prefix warm when workspace growth reduces the
    /// expert budget. The caller has drained GPU users before this action.
    pub fn trim(self: Cache, count: u32) bool {
        const state = self.state;
        if (state.cache_slab_count == 0) return false;
        c.ds4_metal_expert_pending_clear();
        for (&state.entries) |*layer| for (layer) |*entry| {
            if (entry.valid != 0 and (entry.slab_backed == 0 or self.inflight(entry))) return false;
        };
        for (&state.entries, 0..) |*layer_entries, layer| for (layer_entries, 0..) |*entry, expert| {
            if (entry.valid != 0 and entry.slab_slot >= count) self.clear(@intCast(layer), @intCast(expert), true, false, null);
        };
        for (@min(count, state.cache_slab_total_slots)..state.cache_slab_total_slots) |slot| if (!self.unlockSlot(@intCast(slot))) {
            _ = c.fprintf(c.__stderrp, "metal: failed to unlock trimmed expert slot %u\n", @as(c_uint, @intCast(slot)));
            c.abort();
        };
        var free: u32 = 0;
        for (state.cache_free_slots[0..state.cache_free_slot_count]) |slot| if (slot < count) {
            state.cache_free_slots[free] = slot;
            free += 1;
        };
        state.cache_free_slot_count = free;
        var slab_count: u32 = 0;
        for (0..state.cache_slab_count) |index| {
            const start = state.cache_slab_start_slot[index];
            if (start >= count) {
                c.ds4_metal_expert_retire_slab(state, @intCast(index));
                state.cache_slab_start_slot[index] = 0;
                state.cache_slab_slot_count[index] = 0;
                state.cache_slab_slots_used[index] = 0;
            } else {
                const remaining = count - start;
                state.cache_slab_slot_count[index] = @min(state.cache_slab_slot_count[index], remaining);
                state.cache_slab_slots_used[index] = @min(state.cache_slab_slots_used[index], remaining);
                slab_count += 1;
            }
        }
        state.cache_slab_count = slab_count;
        state.cache_slab_total_slots = @min(state.cache_slab_total_slots, count);
        return true;
    }
    pub fn clearAll(self: Cache, reset_statistics: bool) void {
        const state = self.state;
        c.ds4_metal_expert_pending_clear();
        state.cache_done_seq = state.cache_cb_seq;
        state.cache_batch_seq = 0;
        state.cache_owned_seq = 0;
        state.cache_pending_max_seq = 0;
        for (&state.entries, 0..) |*layer_entries, layer| {
            for (layer_entries, 0..) |*entry, expert| {
                entry.inflight_seq = 0;
                self.clear(@intCast(layer), @intCast(expert), false, true, null);
            }
            c.ds4_metal_expert_full_layer_retire(@intCast(layer));
            state.cache_layer_count[layer] = 0;
            c.ds4_metal_expert_zero_layer_address(@intCast(layer));
        }
        state.cache_bytes = 0;
        state.cache_entry_count = 0;
        // Recycled slots stay locked until the whole cache is retired. Unlock
        // BEFORE dropping Metal references: storage can outlive those owners.
        for (0..state.cache_slab_total_slots) |slot| if (!self.unlockSlot(@intCast(slot))) {
            _ = c.fprintf(c.__stderrp, "metal: failed to unlock expert cache slot %u\n", @as(c_uint, @intCast(slot)));
            c.abort();
        };
        for (0..state.cache_slab_count) |index| {
            c.ds4_metal_expert_retire_slab(state, @intCast(index));
            state.cache_slab_start_slot[index] = 0;
            state.cache_slab_slot_count[index] = 0;
            state.cache_slab_slots_used[index] = 0;
        }
        state.cache_slab_count = 0;
        state.cache_slab_total_slots = 0;
        state.cache_free_slot_count = 0;
        state.cache_slab_slot_bytes = 0;
        @memset(&state.cache_slab_slot_locked, 0);
        if (reset_statistics) self.resetStatistics();
    }

    fn resetStatistics(self: Cache) void {
        // Exact original reset set; explicit I/O attribution and prior per-layer
        // reporting snapshots deliberately retain their old lifetime.
        inline for (.{
            "pending_wait_ms",
            "pending_joins",
            "gate_probes",
            "gate_submissions",
            "cache_hits",
            "cache_misses",
            "cache_evictions",
            "cache_wraps",
            "cache_clock",
            "cache_evict_advise_bytes",
            "cache_willneed_advise_bytes",
            "cache_pread_bytes",
            "cache_pread_ms",
            "cache_mlock_bytes",
            "cache_mlock_fail_bytes",
            "cache_mlock_failures",
            "cache_mlock_ms",
            "cache_mlock_warned",
            "cache_mlock_budget_cap",
            "cache_mlock_relief_applied",
            "cache_buffer_allocs",
            "cache_buffer_reuses",
            "cache_decode_tokens",
            "cache_first_decode_layer",
            "cache_hotness_decay_token",
            "cache_route_hotness",
            "timing_selected_calls",
            "timing_selected_read_ms",
            "timing_selected_sync_ms",
            "timing_selected_copy_ms",
            "timing_selected_bind_ms",
            "timing_split_layers",
            "timing_split_resident_experts",
            "timing_split_missing_experts",
            "timing_split_resident_ms",
            "timing_split_missing_ms",
            "timing_split_missing_load_ms",
            "timing_split_missing_slot_ms",
            "timing_split_missing_prune_ms",
            "timing_split_missing_addr_ms",
            "timing_split_missing_wait_ms",
            "timing_load_calls",
            "timing_load_prepare_ms",
            "timing_load_pread_ms",
            "timing_load_install_ms",
            "timing_prepare_batch_reuse_calls",
            "timing_prepare_batch_reuse_ms",
            "timing_prepare_buffer_calls",
            "timing_prepare_buffer_ms",
            "timing_prepare_task_experts",
            "timing_prepare_task_ms",
            "timing_reuse_scan_calls",
            "timing_reuse_scan_entries",
            "timing_reuse_scan_ms",
            "timing_reuse_clear_ms",
            "timing_readahead_calls",
            "timing_readahead_bytes",
            "timing_readahead_ms",
            "timing_cache_all_resident_layers",
            "timing_cache_all_missing_layers",
            "timing_cache_mixed_layers",
            "timing_cache_resident_experts",
            "timing_cache_missing_experts",
            "timing_last_report",
            "cache_layer_hits",
            "cache_layer_misses",
            "cache_layer_evictions",
            "cache_layer_pread_bytes",
            "cache_layer_pread_ms",
        }) |name| {
            if (comptime std.mem.eql(u8, name, "cache_first_decode_layer")) self.state.cache_first_decode_layer = invalid else @field(self.state.*, name) = std.mem.zeroes(@TypeOf(@field(self.state.*, name)));
        }
    }
};
pub fn protectedExpert(expert: u32, selected: []const i32) bool {
    for (selected) |id| if (id >= 0 and @as(u32, @intCast(id)) == expert) return true;
    return false;
}
