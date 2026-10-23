//! Slab/explicit-buffer policy and locked-cache admission. ARC only allocates,
//! retains or releases real Metal handles; native owns budgets and transitions.
const std = @import("std");
const c = @import("metal_c");
const Cache = @import("expert_cache.zig");
const IO = @import("expert_io.zig");
const Buffer = Cache.Buffer;
const Reuse = Cache.Reuse;
const max = std.math.maxInt(u64);
const gib: u64 = 1024 * 1024 * 1024;
fn present(name: [*:0]const u8) bool {
    return c.getenv(name) != null;
}
fn ready(reuse: *const Reuse) bool {
    return reuse.gate_buffer != null and reuse.up_buffer != null and reuse.down_buffer != null;
}
pub const Owned = struct {
    token: ?*c.ds4_metal_expert_owned_buffer = null,
    pub fn buffer(self: Owned) Buffer {
        return if (self.token) |token| @ptrCast(token) else null;
    }
    pub fn deinit(self: *Owned) void {
        if (self.token) |token| c.ds4_metal_expert_buffer_release(token);
        self.token = null;
    }
};
pub const Storage = struct {
    cache: Cache.Cache,
    device: bool,
    pub fn usesCombined(self: Storage) bool {
        return self.cache.streaming and !present("DS4_METAL_DISABLE_STREAMING_EXPERT_COMBINED_BUFFER");
    }
    pub fn slabs(self: Storage) bool {
        return self.usesCombined() and !present("DS4_METAL_DISABLE_STREAMING_EXPERT_SLABS");
    }
    pub fn targetBytes(_: Storage) u64 {
        const mib: u64 = 1024 * 1024;
        var bytes: u64 = 4096 * mib;
        const text = c.getenv("DS4_METAL_STREAMING_EXPERT_SLAB_MB");
        if (text != null and text[0] != 0) {
            var end: [*c]u8 = null;
            const value = c.strtoull(text, &end, 10);
            if (end != text and end[0] == 0 and value != 0) bytes = if (value > max / mib) max else value * mib;
        }
        return bytes;
    }
    pub fn capToLocked(self: Storage) void {
        const state = self.cache.state;
        var count = state.cache_entry_count;
        const locked = self.cache.lockedCount();
        if (locked != 0 and locked < count) count = locked;
        var safe_gib = state.cache_mlock_bytes / gib;
        if (safe_gib > 1) safe_gib -= 1;
        if (safe_gib != 0 and state.cache_expert_bytes != 0) {
            const safe_bytes = if (safe_gib > max / gib) max else safe_gib * gib;
            const safe_count = @min(safe_bytes / state.cache_expert_bytes, std.math.maxInt(u32));
            if (safe_count != 0 and safe_count < count) count = @intCast(safe_count);
        }
        if (count != 0 and (state.cache_mlock_budget_cap == 0 or count < state.cache_mlock_budget_cap)) state.cache_mlock_budget_cap = count;
    }
    pub fn warnMlock(self: Storage, failed: u64, err: c_int) void {
        const state = self.cache.state;
        if (state.cache_mlock_warned != 0) return;
        state.cache_mlock_warned = 1;
        const budget = self.cache.requested();
        const requested = if (budget != 0 and state.cache_expert_bytes != 0) @as(u64, budget) *| state.cache_expert_bytes else 0;
        var suggestion = state.cache_mlock_bytes / gib;
        if (suggestion > 1) suggestion -= 1;
        _ = c.fprintf(c.__stderrp, "ds4: warning: streaming expert cache could not mlock all buffers\n");
        if (requested != 0) _ = c.fprintf(c.__stderrp, "ds4:   requested cache: %u experts / %.2f GiB\n", budget, @as(f64, @floatFromInt(requested)) / gib) else _ = c.fprintf(c.__stderrp, "ds4:   requested cache: %u experts\n", budget);
        _ = c.fprintf(c.__stderrp, "ds4:   locked so far:   %.2f GiB\n", @as(f64, @floatFromInt(state.cache_mlock_bytes)) / gib);
        _ = c.fprintf(c.__stderrp, "ds4:   failed buffer:   %.2f MiB (%s)\n", @as(f64, @floatFromInt(failed)) / 1048576, if (err != 0) c.strerror(err) else @as([*c]const u8, "mlock unavailable"));
        _ = c.fprintf(c.__stderrp, "ds4:   macOS may page unlocked expert buffers, causing poor or unstable speed\n");
        if (state.cache_mlock_budget_cap != 0 and state.cache_expert_bytes != 0) {
            const capped = @as(u64, state.cache_mlock_budget_cap) *| state.cache_expert_bytes;
            _ = c.fprintf(c.__stderrp, "ds4:   using locked cache cap: %u experts / %.2f GiB\n", state.cache_mlock_budget_cap, @as(f64, @floatFromInt(capped)) / gib);
        }
        if (suggestion != 0) _ = c.fprintf(c.__stderrp, "ds4:   try: --ssd-streaming-cache-experts %lluGB\n", @as(c_ulonglong, suggestion)) else _ = c.fprintf(c.__stderrp, "ds4:   try a smaller --ssd-streaming-cache-experts NGB budget\n");
    }
    pub fn allocate(self: Storage, bytes: u64, label: [*c]const u8, slab: bool) Owned {
        if (!self.device or bytes == 0 or bytes > std.math.maxInt(usize)) return .{};
        var owner = Owned{ .token = c.ds4_metal_expert_buffer_new(bytes, label) };
        if (owner.token == null) {
            _ = c.fprintf(c.__stderrp, "ds4: Metal streaming expert %s allocation failed (%.2f MiB)\n", @as([*:0]const u8, if (slab) "slab" else "explicit buffer"), @as(f64, @floatFromInt(bytes)) / 1048576);
            return .{};
        }
        const state = self.cache.state;
        state.cache_buffer_allocs +%= 1;
        if (slab or !self.cache.streaming) return owner;
        const memory = c.ds4_metal_expert_buffer_storage(owner.buffer());
        const start = IO.now();
        const locked = memory.data != null and memory.bytes != 0 and c.mlock(memory.data, @intCast(memory.bytes)) == 0;
        state.cache_mlock_ms += IO.now() - start;
        if (locked) {
            state.cache_mlock_bytes +|= memory.bytes;
            return owner;
        }
        state.cache_mlock_failures +%= 1;
        state.cache_mlock_fail_bytes +|= memory.bytes;
        const err = if (memory.data != null and memory.bytes != 0) c.__error().* else 0;
        self.capToLocked();
        self.warnMlock(memory.bytes, err);
        if (present("DS4_METAL_STREAMING_EXPERT_BUFFER_MLOCK_PROFILE")) _ = c.fprintf(c.__stderrp, "ds4: Metal streaming expert buffer mlock failed len=%.2f MiB: %s\n", @as(f64, @floatFromInt(memory.bytes)) / 1048576, if (err != 0) c.strerror(err) else @as([*c]const u8, "mlock unavailable"));
        owner.deinit();
        return .{};
    }
    pub fn lockSlot(self: Storage, slot: u32) bool {
        const state = self.cache.state;
        if (slot >= c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES or state.cache_slab_slot_locked[slot] != 0) return true;
        const range = self.cache.slotRange(slot) orelse return false;
        if (range.slab >= state.cache_slab_count or
            state.cache_slabs[range.slab] == null or
            state.cache_slab_slot_bytes == 0 or
            range.base > std.math.maxInt(usize)) return false;
        const memory = c.ds4_metal_expert_buffer_storage(state.cache_slabs[range.slab]);
        if (memory.data == null) return false;
        const base: [*]u8 = @ptrCast(memory.data.?);
        const start = IO.now();
        const result = c.mlock(base + @as(usize, @intCast(range.base)), @intCast(state.cache_slab_slot_bytes));
        state.cache_mlock_ms += IO.now() - start;
        if (result == 0) {
            state.cache_mlock_bytes +|= state.cache_slab_slot_bytes;
            state.cache_slab_slot_locked[slot] = 1;
            return true;
        }
        state.cache_mlock_failures +%= 1;
        state.cache_mlock_fail_bytes +|= state.cache_slab_slot_bytes;
        self.capToLocked();
        self.warnMlock(state.cache_slab_slot_bytes, c.__error().*);
        return false;
    }
    pub fn slotBuffers(self: Storage, slot: u32, gate: u64, down: u64, out: *Reuse) bool {
        const state = self.cache.state;
        const range = self.cache.slotRange(slot) orelse return false;
        if (range.slab >= state.cache_slab_count or
            state.cache_slabs[range.slab] == null or
            gate > (max - down) / 2 or
            range.base > max - (gate * 2 + down) or
            range.base + gate * 2 + down > std.math.maxInt(usize)) return false;
        const buffer = state.cache_slabs[range.slab];
        if (!self.lockSlot(slot)) return false;
        c.ds4_metal_expert_reuse_assign(out, buffer, buffer, buffer, @intCast(range.base), @intCast(range.base + gate), @intCast(range.base + gate * 2));
        return true;
    }
    pub fn slabSlot(self: Storage, gate: u64, down: u64, out: *Reuse) bool {
        if (!self.slabs() or gate == 0 or down == 0 or gate > (max - down) / 2) return false;
        var bytes = gate * 2 + down;
        if (bytes == 0 or bytes > std.math.maxInt(usize)) return false;
        const page: u64 = @intCast(c.sysconf(c._SC_PAGESIZE));
        if (page != 0) {
            bytes = (bytes +% (page - 1)) & ~(page - 1);
            if (bytes == 0 or bytes > std.math.maxInt(usize)) return false;
        }
        const state = self.cache.state;
        if (state.cache_slab_slot_bytes != 0 and state.cache_slab_slot_bytes != bytes) return false;
        state.cache_slab_slot_bytes = bytes;
        if (state.cache_free_slot_count != 0) {
            state.cache_free_slot_count -= 1;
            return self.slotBuffers(state.cache_free_slots[state.cache_free_slot_count], gate, down, out);
        }
        var slab = state.cache_slab_count;
        if (slab != 0 and state.cache_slab_slots_used[slab - 1] < state.cache_slab_slot_count[slab - 1]) slab -= 1 else {
            if (state.cache_slab_count >= c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_SLABS) return false;
            const budget = self.cache.budget();
            if (budget != 0 and state.cache_slab_total_slots >= budget) return false;
            var count: u32 = @intCast(@min(@max(self.targetBytes() / bytes, 1), std.math.maxInt(u32)));
            if (budget != 0) count = @min(count, budget - state.cache_slab_total_slots);
            if (count == 0) return false;
            var owner = Owned{};
            defer owner.deinit();
            while (count != 0) : (count /= 2) {
                if (count <= max / bytes and @as(u64, count) * bytes <= std.math.maxInt(usize)) {
                    owner = self.allocate(@as(u64, count) * bytes, "ds4_stream_expert_slab", true);
                    if (owner.token != null) break;
                }
            }
            if (owner.token == null or count == 0) return false;
            slab = state.cache_slab_count;
            state.cache_slab_count +%= 1;
            c.ds4_metal_expert_slab_assign(state, slab, owner.buffer());
            state.cache_slab_start_slot[slab] = state.cache_slab_total_slots;
            state.cache_slab_slot_count[slab] = count;
            state.cache_slab_slots_used[slab] = 0;
            state.cache_slab_total_slots +%= count;
        }
        const local = state.cache_slab_slots_used[slab];
        state.cache_slab_slots_used[slab] +%= 1;
        return self.slotBuffers(state.cache_slab_start_slot[slab] +% local, gate, down, out);
    }
    pub fn cappedReuse(self: Storage, layer: u32, protect: []const i32, gate: u64, down: u64, out: *Reuse) bool {
        if (self.cache.state.cache_mlock_budget_cap == 0) return false;
        _ = self.cache.releaseMargin(layer, protect);
        var reuse = std.mem.zeroes(Reuse);
        defer c.ds4_metal_expert_reuse_reset(&reuse);
        if (!self.cache.takeReusable(true, layer, protect, gate, down, &reuse)) return false;
        c.ds4_metal_expert_reuse_assign(out, reuse.gate_buffer, reuse.up_buffer, reuse.down_buffer, reuse.gate_inner, reuse.up_inner, reuse.down_inner);
        return ready(out);
    }
    pub fn prepare(
        self: Storage,
        layer: u32,
        expert: u32,
        protect_layer: u32,
        protect: []const i32,
        gate: u64,
        down: u64,
        force_reuse: bool,
        out: *Reuse,
    ) bool {
        if (layer >= c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_LAYER or expert >= c.DS4_METAL_STREAM_EXPERT_CACHE_MAX_EXPERT) return false;
        c.ds4_metal_expert_reuse_reset(out);
        var reuse = std.mem.zeroes(Reuse);
        defer c.ds4_metal_expert_reuse_reset(&reuse);
        const entry = &self.cache.state.entries[layer][expert];
        if (entry.valid != 0 and self.cache.inflight(entry)) {
            if (c.ds4_metal_expert_on_service_thread() or !c.ds4_metal_expert_wait("streaming expert cache replacement")) return false;
        }
        if (Cache.reusable(entry, gate, down)) {
            self.cache.clear(layer, expert, false, true, &reuse);
            if (ready(&reuse)) self.cache.state.cache_buffer_reuses +%= Cache.objectCount(reuse.gate_buffer, reuse.up_buffer, reuse.down_buffer);
        } else if (entry.valid != 0) self.cache.clear(layer, expert, false, true, null);
        if (!ready(&reuse) and !self.cache.takeReusable(force_reuse, protect_layer, protect, gate, down, &reuse)) c.ds4_metal_expert_reuse_reset(&reuse);
        if (ready(&reuse)) {
            c.ds4_metal_expert_reuse_assign(out, reuse.gate_buffer, reuse.up_buffer, reuse.down_buffer, reuse.gate_inner, reuse.up_inner, reuse.down_inner);
            return true;
        }
        if (self.usesCombined()) {
            if (gate > max - gate or
                gate * 2 > max - down or
                gate > std.math.maxInt(usize) or
                gate * 2 > std.math.maxInt(usize) or
                gate * 2 + down > std.math.maxInt(usize)) return false;
            if (self.cache.state.cache_mlock_budget_cap != 0) return self.cappedReuse(protect_layer, protect, gate, down, out);
            if (self.slabSlot(gate, down, out)) return true;
            if (self.cache.state.cache_mlock_budget_cap != 0) return self.cappedReuse(protect_layer, protect, gate, down, out);
            var combined = self.allocate(gate * 2 + down, "ds4_stream_expert_combined", false);
            defer combined.deinit();
            if (combined.token == null) return self.cappedReuse(protect_layer, protect, gate, down, out);
            c.ds4_metal_expert_reuse_assign(out, combined.buffer(), combined.buffer(), combined.buffer(), 0, @intCast(gate), @intCast(gate * 2));
            return true;
        }
        // Attempt all three allocations before deciding failure, as in C.
        var gate_owner = self.allocate(gate, "ds4_stream_expert_gate", false);
        defer gate_owner.deinit();
        var up_owner = self.allocate(gate, "ds4_stream_expert_up", false);
        defer up_owner.deinit();
        var down_owner = self.allocate(down, "ds4_stream_expert_down", false);
        defer down_owner.deinit();
        if (gate_owner.token == null or up_owner.token == null or down_owner.token == null) {
            self.cache.unlockExplicit(gate_owner.buffer());
            self.cache.unlockExplicit(up_owner.buffer());
            self.cache.unlockExplicit(down_owner.buffer());
            return self.cappedReuse(protect_layer, protect, gate, down, out);
        }
        c.ds4_metal_expert_reuse_assign(out, gate_owner.buffer(), up_owner.buffer(), down_owner.buffer(), 0, 0, 0);
        return true;
    }
};
