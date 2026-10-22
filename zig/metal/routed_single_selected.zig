//! Selected-ID acquisition and expert-resource preparation. Per-call masks and
//! offsets are native locals; the actual global cache/pending-load owns bytes.
const std = @import("std");
const c = @import("metal_c");
const Batch = @import("routed_batch.zig");
const Dispatch = @import("routed_single_dispatch.zig");
const Execution = Dispatch.Execution;
const present = Batch.present;
const absent = Batch.absent;
var override_ids: [8]i32 = @splat(0);
var override_count: u32 = 0;
pub fn setOverride(ids: [*c]const i32, count: u32) c_int {
    if (count > 8 or (ids == null and count != 0)) return 0;
    for (0..count) |slot| override_ids[slot] = ids[slot];
    override_count = count;
    return 1;
}
pub fn clearOverride() void {
    override_count = 0;
}
fn takeOverride(ids: *[8]i32, count: u32) bool {
    if (override_count != count) return false;
    @memcpy(ids[0..count], override_ids[0..count]);
    override_count = 0;
    return true;
}
fn now() f64 {
    return c.ds4_metal_routed_now_ms();
}
fn maskedRequested(ready: bool) bool {
    return (present("DS4_METAL_ENABLE_STREAMING_EXPERT_MASKED_ADDR") or ready) and absent("DS4_METAL_DISABLE_STREAMING_EXPERT_MASKED_ADDR");
}
fn addrRequested(ready: bool) bool {
    return (present("DS4_METAL_ENABLE_STREAMING_EXPERT_ADDR_TABLE") or present("DS4_METAL_ENABLE_STREAMING_EXPERT_HIT_VALIDATOR") or present("DS4_METAL_ENABLE_STREAMING_EXPERT_MASKED_ADDR") or ready) and
        absent("DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE");
}
pub const State = struct {
    ids: [8]i32 = @splat(0),
    gate: [8]u64 = @splat(0),
    up: [8]u64 = @splat(0),
    down: [8]u64 = @splat(0),
    cache: bool = false,
    address: bool = false,
    masked: bool = false,
    compact: bool = false,
    split_deferred: bool = false,
    split_completed: bool = false,
    staged_gate: bool = false,
    resident: u32 = 0,
    missing: u32 = 0,

    pub fn prepare(self: *State, execution: *Execution) bool {
        @setFloatMode(.optimized);
        const q = execution.q;
        const f = execution.facts;
        const p = execution.policy;
        const resource = execution.resource;
        var stats: c.ds4_metal_single_stats = undefined;
        c.ds4_metal_single_cache_stats(&stats, !p.selected_profile);
        const timing = p.selected_profile or stats.summary;
        var started = if (timing) now() else 0;
        var sync_ms: f64 = 0;
        var copy_ms: f64 = 0;
        var source: [*:0]const u8 = "readback";
        var ids_available = true;
        var ids_from_host = false;
        const size_known = c.ds4_metal_single_cache_note_size(resource);
        const full = p.slots_iq2 and
            present("DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE") and
            absent("DS4_METAL_DISABLE_STREAMING_FULL_EXPERT_ADDR_TABLE") and
            Batch.pairPresent(&f.batch, c.DS4_ROUTED_ADDR_IQ2_SWI, c.DS4_ROUTED_ADDR_Q2_SUM);
        self.cache = !full and (p.slots_iq2 or p.stream_iq2 or p.slots_q4 or p.slots_mx) and size_known and c.ds4_metal_single_cache_available(resource);
        if (p.stream_iq2 and !self.cache) {
            _ = Batch.errorLine("ds4: Metal IQ2/IQ2 streaming decode requires a non-empty expert cache");
            return Dispatch.silent(32831);
        }
        const ready = self.cache and c.ds4_metal_single_split_ready();
        const compact = self.cache and
            p.slots_iq2 and
            present("DS4_METAL_ENABLE_STREAMING_COMPACT_ADDR") and
            absent("DS4_METAL_DISABLE_STREAMING_COMPACT_ADDR") and
            absent("DS4_METAL_DISABLE_STREAMING_EXPERT_ADDR_TABLE") and
            !ready and
            !maskedRequested(ready) and
            Batch.pairPresent(&f.batch, c.DS4_ROUTED_ADDR_IQ2_SWI, c.DS4_ROUTED_ADDR_Q2_SUM);
        const split_candidate = self.cache and p.slots_iq2 and !compact and ready and f.cached[c.DS4_SINGLE_MASK_PAIR] != null and f.cached[c.DS4_SINGLE_MASK_SUM] != null;
        const validator = self.cache and
            p.slots_iq2 and
            present("DS4_METAL_ENABLE_STREAMING_EXPERT_HIT_VALIDATOR") and
            absent("DS4_METAL_DISABLE_STREAMING_EXPERT_HIT_VALIDATOR") and
            f.cached[c.DS4_SINGLE_VALIDATE] != null and
            Batch.pairPresent(&f.batch, c.DS4_ROUTED_ADDR_IQ2_SWI, c.DS4_ROUTED_ADDR_Q2_SUM) and
            c.ds4_metal_single_addresses(resource);
        if (full) {
            source = "gpu-full-addr";
            ids_available = false;
            clearOverride();
            if (!c.ds4_metal_single_full_addresses(resource)) return Dispatch.silent(32880);
            self.address = true;
        } else {
            const replayed = c.ds4_metal_single_trace_replay(&self.ids, q.used_experts);
            if (replayed < 0) return Dispatch.silent(32890);
            if (replayed > 0) {
                source = "replay";
                ids_from_host = true;
                clearOverride();
            } else if (takeOverride(&self.ids, q.used_experts)) {
                source = "override";
                ids_from_host = true;
            } else {
                const prefetched = if (self.cache) c.ds4_metal_single_prefetch_take(resource, &self.ids) else 0;
                if (prefetched != 0) {
                    // GLM started these reads already. Submit shared-expert GPU
                    // work while the actual pending load finishes.
                    source = "prefetched";
                    if (prefetched == 2 and c.ds4_gpu_flush_commands() == 0) return false;
                } else if (validator) {
                    clearOverride();
                    var all: u32 = 0;
                    var miss: u32 = 0;
                    var invalid: u32 = 0;
                    if (!c.ds4_metal_single_validate(resource, &self.ids, &all, &miss, &invalid)) return Dispatch.silent(32917);
                    source = if (invalid != 0) "validator-invalid" else if (miss != 0) "validator-miss" else if (all != 0) "validator-hit" else "validator-miss";
                } else {
                    clearOverride();
                    const had_batch = c.ds4_gpu_commands_active() != 0;
                    if (had_batch) {
                        const began = if (timing) now() else 0;
                        if (p.shared_event) {
                            if (!c.ds4_metal_single_readback_event()) return Dispatch.silent(32942);
                        } else if (c.ds4_gpu_end_commands() == 0) return Dispatch.silent(32931);
                        if (timing) sync_ms += now() - began;
                    }
                    const began = if (timing) now() else 0;
                    if (c.ds4_gpu_tensor_read(q.selected, 0, @ptrCast(&self.ids), @as(u64, q.used_experts) * 4) == 0) return Dispatch.silent(if (had_batch) 32943 else 32965);
                    if (timing) copy_ms += now() - began;
                    if (had_batch and !p.shared_event) {
                        const boundary = if (timing) now() else 0;
                        if (c.ds4_gpu_begin_commands() == 0) return Dispatch.silent(32967);
                        if (timing) sync_ms += now() - boundary;
                    }
                }
            }
        }
        const read_ms = sync_ms + copy_ms;
        if (timing) started = now();
        if (ids_available) {
            for (self.ids[0..q.used_experts]) |id| {
                if (id < 0 or @as(u32, @intCast(id)) >= q.total_experts) {
                    _ = c.fprintf(c.__stderrp, "ds4: Metal routed MoE selected expert id %d is outside 0..%u\n", id, q.total_experts);
                    return Dispatch.silent(32986);
                }
            }
            if (!c.ds4_metal_single_trace_record(q.layer, &self.ids, q.used_experts, q.total_experts)) return Dispatch.silent(32997);
            for (self.ids[0..q.used_experts], 0..) |id, slot| {
                const expert: u64 = @intCast(id);
                if (expert > std.math.maxInt(u64) / q.gate_expert_bytes or expert > std.math.maxInt(u64) / q.down_expert_bytes) return Batch.errorLine("ds4: Metal routed MoE selected expert offset overflow") != 0;
                const gate = expert * q.gate_expert_bytes;
                const down = expert * q.down_expert_bytes;
                if (gate > std.math.maxInt(u64) - q.gate_offset or gate > std.math.maxInt(u64) - q.up_offset or down > std.math.maxInt(u64) - q.down_offset) return Batch.errorLine("ds4: Metal routed MoE selected expert offset overflow") != 0;
                self.gate[slot] = q.gate_offset + gate;
                self.up[slot] = q.up_offset + gate;
                self.down[slot] = q.down_offset + down;
                if (self.cache) {
                    if (c.ds4_metal_single_peek(resource, @intCast(slot), @intCast(id), self.gate[slot], self.up[slot], self.down[slot])) self.resident |= @as(u32, 1) << @as(u5, @intCast(slot)) else self.missing |= @as(u32, 1) << @as(u5, @intCast(slot));
                } else {
                    const mode: u32 = if (p.base_views) 0 else if (p.transient_views) 2 else 1;
                    if (!c.ds4_metal_single_wrap_slot(resource, @intCast(slot), self.gate[slot], self.up[slot], self.down[slot], mode)) return Dispatch.silent(33105);
                }
            }
        }
        if (self.cache) {
            const current_ready = c.ds4_metal_single_split_ready();
            self.address = ((p.slots_iq2 and
                addrRequested(current_ready) and
                Batch.pairPresent(&f.batch, c.DS4_ROUTED_ADDR_IQ2_SWI, c.DS4_ROUTED_ADDR_Q2_SUM)) or p.stream_iq2) and
                c.ds4_metal_single_addresses(resource);
            if (p.stream_iq2 and !self.address) {
                _ = Batch.errorLine("ds4: Metal IQ2/IQ2 streaming decode could not prepare expert address buffers");
                return Dispatch.silent(33123);
            }
            self.masked = self.address and
                p.slots_iq2 and
                maskedRequested(c.ds4_metal_single_split_ready()) and
                f.cached[c.DS4_SINGLE_MASK_PAIR] != null and
                f.cached[c.DS4_SINGLE_MASK_SUM] != null;
            self.split_deferred = split_candidate and
                self.masked and
                self.resident != 0 and
                self.missing != 0 and
                @popCount(self.missing) >= 3 and
                c.ds4_gpu_commands_active() != 0 and
                absent("DS4_METAL_MOE_ONE_STAGE_PROFILE");
            if (self.split_deferred and !c.ds4_metal_single_begin_load(resource, &self.ids)) return Dispatch.silent(33156);
            const staged = present("DS4_METAL_V41_STAGED_EXPERT_LOAD") and
                q.input_width == 5120 and
                q.middle_width == 2304 and
                q.output_width == 5120 and
                q.used_experts == 6 and
                p.slots_iq2 and
                p.fused and
                p.slotPair(f) != null and
                c.ds4_gpu_commands_active() != 0 and
                absent("DS4_METAL_MOE_WRITE_CLAMPED_ACT") and
                !f.parallel_pending and
                !compact and
                !self.split_deferred and
                self.missing != 0 and
                absent("DS4_METAL_MOE_ONE_STAGE_PROFILE");
            if (staged) {
                const gate_ready = c.ds4_metal_single_pending_gate(resource, &self.ids, self.missing);
                if (gate_ready < 0) return false;
                if (gate_ready == 2) {
                    // Pending load owns new buffers until down is ready. Pin
                    // resident slots before submitting the overlapping gate.
                    if (!c.ds4_metal_single_mark_present(resource) or !execution.ops.command.acquire() or !execution.pair(c.DS4_SINGLE_ARRAY_PAIR, p.slotPair(f), 6) or c.ds4_gpu_flush_commands() == 0) return false;
                    self.staged_gate = true;
                    c.ds4_metal_single_gate_submitted();
                }
            }
            if (self.missing != 0 and !self.split_deferred) {
                if (!c.ds4_metal_single_load_missing(resource, &self.ids, &self.gate, &self.up, &self.down, self.missing)) return Dispatch.silent(33175);
                if (!c.ds4_metal_single_bind_loaded(resource, self.missing)) return Dispatch.silent(33202);
            }
            if (p.stream_iq2 and self.address and !c.ds4_metal_single_publish_addresses(resource, &self.ids)) return Dispatch.silent(33202);
            c.ds4_metal_single_prune(resource, &self.ids);
            if (compact) {
                if (!c.ds4_metal_single_compact(resource)) return Dispatch.silent(33223);
                self.address = true;
                self.masked = false;
                self.compact = true;
            }
            if (self.address and !self.compact and ids_from_host and !c.ds4_metal_single_host_ids(resource, &self.ids)) return Dispatch.silent(33239);
        }
        const bind_ms = if (timing) now() - started else 0;
        if (timing) c.ds4_metal_single_note_selected(sync_ms, copy_ms, bind_ms, self.resident, self.missing, self.cache);
        if (p.selected_profile) {
            var current: c.ds4_metal_single_stats = undefined;
            c.ds4_metal_single_cache_stats(&current, false);
            const mode: [*:0]const u8 = if (self.split_deferred) "stream-split" else if (self.masked) "stream-addr-mask" else if (self.compact) "stream-compact-addr" else if (self.address) "stream-addr" else if (self.cache) "stream-cache" else if (p.slots_iq2) "exact-cache" else p.viewMode();
            _ = c.fprintf(c.__stderrp, "ds4: Metal selected views layer=%u path=%s mode=%s ids=%s experts=", q.layer, p.selectedPath(), mode, source);
            for (0..q.used_experts) |slot| _ = c.fprintf(c.__stderrp, "%s%d", @as([*:0]const u8, if (slot != 0) "," else ""), @as(i32, if (ids_available) self.ids[slot] else -1));
            _ = c.fprintf(c.__stderrp, " expert_gate=%.2f MiB expert_down=%.2f MiB read=%.3f ms bind=%.3f ms cache_hits=%llu cache_misses=%llu cache_wraps=%llu cache_evictions=%llu\n", @as(f64, @floatFromInt(q.gate_expert_bytes)) / (1024 * 1024), @as(f64, @floatFromInt(q.down_expert_bytes)) / (1024 * 1024), read_ms, bind_ms, current.hits -% stats.hits, current.misses -% stats.misses, current.wraps -% stats.wraps, current.evictions -% stats.evictions);
        }
        return true;
    }

    /// Resident GPU work overlaps missing-expert I/O, but both stages reuse the
    /// same scratch. Wait for resident completion before the missing stage;
    /// issue one all-six down pass to retain the final accumulation order.
    pub fn split(self: *State, execution: *Execution) bool {
        @setFloatMode(.optimized);
        const resource = execution.resource;
        const profile = present("DS4_METAL_STREAMING_EXPERT_SPLIT_PROFILE");
        var stats: c.ds4_metal_single_stats = undefined;
        c.ds4_metal_single_cache_stats(&stats, !profile);
        const timing = profile or stats.summary;
        var started = if (timing) now() else 0;
        var ok = execution.masked(false, self.resident);
        if (ok) ok = execution.flush();
        const submit_ms = if (timing) now() - started else 0;
        if (timing) started = now();
        const missing_start = started;
        var details: [5]f64 = @splat(0);
        if (ok) {
            ok = c.ds4_metal_single_load_missing(resource, &self.ids, &self.gate, &self.up, &self.down, self.missing);
            if (timing) {
                const next = now();
                details[0] = next - started;
                started = next;
            }
            if (ok) ok = c.ds4_metal_single_bind_loaded(resource, self.missing);
        }
        if (timing) {
            const next = now();
            details[1] = next - started;
            started = next;
        }
        if (ok) {
            c.ds4_metal_single_prune(resource, &self.ids);
            if (timing) {
                const next = now();
                details[2] = next - started;
                started = next;
            }
            ok = c.ds4_metal_single_addresses(resource);
            if (timing) {
                const next = now();
                details[3] = next - started;
                started = next;
            }
        }
        if (ok) {
            ok = c.ds4_metal_single_wait_resident();
            if (timing) details[4] = now() - started;
        }
        const missing_ms = if (timing) now() - missing_start else 0;
        if (ok) ok = execution.masked(false, self.missing);
        if (ok) ok = execution.masked(true, 0x3f);
        self.split_completed = ok;
        if (timing) c.ds4_metal_single_note_split(self.resident, self.missing, submit_ms, missing_ms, &details);
        if (profile) _ = c.fprintf(c.__stderrp, "ds4: Metal streaming expert split layer=%u resident=0x%02x missing=0x%02x resident_submit=%.3f ms missing_bind=%.3f ms\n", execution.q.layer, self.resident, self.missing, submit_ms, missing_ms);
        return ok;
    }
};
