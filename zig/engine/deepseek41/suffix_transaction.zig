//! Session-private suffix cache transactions. GraphSuffix owns numerical
//! transplantation; Checkpoint owns canonical bytes; Coordinator owns fresh
//! request execution. Provenance and publication stay explicit in this owner.
const std = @import("std");
const c = @import("gguf_c");
const Coordinator = @import("coordinator.zig").Coordinator;
const Checkpoint = @import("checkpoint.zig").Checkpoint;
const checkpoint_owner = @import("checkpoint.zig");
const suffix = @import("suffix.zig");
const Payload = @import("state_payload.zig").Payload;
const Frontier = @import("frontier.zig").Frontier;
const Cache = c.ds4_suffix_cache;
const Snapshot = c.ds4_session_snapshot;

fn fail(errors: []u8, message: []const u8) c_int {
    if (errors.len != 0) {
        const count = @min(errors.len - 1, message.len);
        @memcpy(errors[0..count], message[0..count]);
        errors[count] = 0;
    }
    return 1;
}
pub fn freeCache(maybe_cache: ?*Cache) void {
    const cache = maybe_cache orelse return;
    checkpoint_owner.freeSnapshot(&cache.state);
    c.free(cache.prerope);
    cache.* = std.mem.zeroes(Cache);
}

pub const Transaction = struct {
    session: *const Coordinator,
    engine_identity: ?*c.ds4_engine,

    fn graph(self: *const Transaction) suffix.Suffix {
        return .{ .graph = self.session.runtime.prefill.executor.graph, .config = self.session.runtime.prefill.executor.ops.config };
    }
    fn payload(self: *const Transaction) Payload {
        return .{ .graph = self.session.runtime.prefill.executor.graph, .config = self.session.runtime.prefill.executor.ops.config };
    }
    fn checkpoint(self: *const Transaction) Checkpoint {
        return .{ .frontier = self.session.runtime.state, .graph = self.session.runtime.prefill.executor.graph, .config = self.session.runtime.prefill.executor.ops.config, .ready = self.session.runtime.ready };
    }
    fn cancelled(self: *const Transaction) bool {
        return if (self.session.control.cancel) |check| check(self.session.control.cancel_ud) else false;
    }
    fn cancellation(self: *const Transaction) suffix.Cancel {
        return .{ .callback = self.session.control.cancel, .userdata = self.session.control.cancel_ud };
    }
    /// Native DS41/non-distributed dispatch is established by the constructor;
    /// all remaining feature/provenance gates are decisions of this owner.
    fn configurationSupported(self: *const Transaction) bool {
        const session = self.session;
        const core = session.core;
        const state = session.runtime.state;
        const device = session.runtime.prefill.executor.graph;
        const has_mtp = core.backend != c.DS4_BACKEND_CPU and
            core.distributed.role == c.DS4_DISTRIBUTED_NONE and core.mtp_ready;
        return !state.approximate_decoder_replay and !session.transport.active and
            core.directional_steering_dirs == null and !core.dspark and !has_mtp and
            session.runtime.ready and !device.workspace_failed and device.image_count == 0 and
            device.imatrix == null and !device.predict_active and !device.predict_load.active and
            state.sync_image_count == 0 and state.checkpoint_image_count == 0;
    }
    fn supported(self: *const Transaction) bool {
        const state = self.session.runtime.state;
        const device = self.session.runtime.prefill.executor.graph;
        return self.configurationSupported() and device.valid and state.checkpoint_valid and
            device.pos == @as(u32, @bitCast(state.checkpoint.len));
    }
    pub fn enableArchive(self: *const Transaction, errors: []u8) c_int {
        if (!self.configurationSupported()) return fail(errors, "pre-RoPE archive requires text-only native DS4.1 session");
        if (self.graph().enableArchive()) return 0;
        return fail(errors, "cannot allocate pre-RoPE archive");
    }
    pub fn archiveBytes(self: *const Transaction) u64 {
        const device = self.session.runtime.prefill.executor.graph;
        return if (self.session.runtime.ready and device.prerope != null) device.prerope.*.bytes else 0;
    }
    pub fn cacheBytes(self: *const Transaction) u64 {
        if (!self.supported()) return 0;
        const device = self.session.runtime.prefill.executor.graph;
        return c.DS4_SESSION_PAYLOAD_U32_FIELDS * 4 + self.payload().bodyBytes(device.pos) +
            (if (device.prerope != null) suffix.archiveBytes(device.pos) else @as(u64, 0));
    }
    pub fn capture(self: *const Transaction, maybe_cache: ?*Cache, errors: []u8) c_int {
        const cache = maybe_cache orelse return fail(errors, "missing suffix cache");
        // The retained allocation may be reused, but no failed recapture may
        // leave the old endpoint or originals labelled as the new source.
        cache.engine = null;
        cache.state.len = 0;
        c.free(cache.prerope);
        cache.prerope = null;
        cache.prerope_bytes = 0;
        cache.approximate = false;
        const state = self.session.runtime.state;
        const device = self.session.runtime.prefill.executor.graph;
        if (self.supported() and (!state.approximate_suffix or device.prerope != null)) {
            const result = if (state.approximate_suffix) self.diagnostic(&cache.state, errors) else self.checkpoint().saveSnapshot(&cache.state, errors);
            if (result == 0 and (device.prerope == null or self.graph().captureArchive(cache))) {
                cache.engine = self.engine_identity;
                cache.ctx_size = state.ctx_size;
                cache.approximate = state.approximate_suffix;
                return 0;
            }
        }
        return fail(errors, "suffix capture requires text-only non-TP DS4.1 state; approximate sources require original pre-RoPE tape");
    }
    pub fn loadPrefix(self: *const Transaction, maybe_cache: ?*const Cache, errors: []u8) c_int {
        const state = self.session.runtime.state;
        const cache = maybe_cache orelse return fail(errors, "suffix prefix restore requires exact canonical DS4.1 state");
        if (cache.approximate or !self.configurationSupported() or cache.engine != self.engine_identity or
            cache.ctx_size != state.ctx_size or (cache.prerope == null and cache.prerope_bytes != 0))
            return fail(errors, "suffix prefix restore requires exact canonical DS4.1 state");
        const end = self.graph().validatePrefix(cache, errors) catch return 1;
        if (self.cancelled()) return fail(errors, "suffix prefix restore cancelled");
        if (cache.prerope != null and self.enableArchive(errors) != 0) return 1;
        if (self.checkpoint().loadSnapshot(&cache.state, errors) != 0) return self.restoreFailure(errors);
        if (cache.prerope != null) self.graph().restoreArchive(cache, end, self.cancellation()) catch return self.restoreFailure(errors);
        if (self.cancelled()) return self.restoreFailure(errors);
        return 0;
    }
    fn restoreFailure(self: *const Transaction, errors: []u8) c_int {
        self.session.invalidate();
        if (errors.len != 0 and errors[0] != 0) return 1;
        return fail(errors, "suffix prefix or original-tail restore failed");
    }
    pub fn diagnostic(self: *const Transaction, maybe_snapshot: ?*Snapshot, errors: []u8) c_int {
        const snapshot = maybe_snapshot orelse return fail(errors, "no complete DS4.1 state for suffix diagnostics");
        if (!self.supported()) return fail(errors, "no complete DS4.1 state for suffix diagnostics");
        const device = self.session.runtime.prefill.executor.graph;
        const bytes = c.DS4_SESSION_PAYLOAD_U32_FIELDS * 4 + self.payload().bodyBytes(device.pos);
        if (bytes >= std.math.maxInt(usize)) return 1;
        if (snapshot.cap < bytes + 1) {
            const memory = c.realloc(snapshot.ptr, @intCast(bytes + 1)) orelse return fail(errors, "cannot allocate suffix diagnostic");
            snapshot.ptr = @ptrCast(memory);
            snapshot.cap = bytes + 1;
        }
        const stream = c.fmemopen(snapshot.ptr, @intCast(bytes + 1), "wb");
        if (stream == null) return fail(errors, "cannot open suffix diagnostic");
        const state = self.session.runtime.state;
        const tokens: []const c_int = if (state.checkpoint.len == 0) &.{} else state.checkpoint.v[0..@intCast(state.checkpoint.len)];
        const logits = state.logits[0..self.session.runtime.prefill.executor.ops.config.shape.n_vocab];
        var result: c_int = 0;
        self.payload().save(@ptrCast(stream), tokens, logits, errors) catch {
            result = 1;
        };
        if (c.fclose(stream) != 0 and result == 0) result = 1;
        if (result == 0) {
            // Inspection bytes cannot become canonical checkpoints or exact
            // sources. Keep the tensor layout intact and change only magic.
            std.mem.writeInt(u32, snapshot.ptr[0..4], c.DS4_SESSION_SUFFIX_DIAGNOSTIC_MAGIC, .little);
            snapshot.len = bytes;
        }
        return result;
    }
    pub fn graft(self: *const Transaction, maybe_cache: ?*const Cache, maybe_prompt: ?*const c.ds4_tokens, source_start: u32, start: u32, statistics: ?*c.ds4_suffix_reuse_stats, relocation: ?*c.ds4_suffix_relocation_stats, errors: []u8) c_int {
        if (statistics) |output| output.* = std.mem.zeroes(c.ds4_suffix_reuse_stats);
        if (relocation) |output| output.* = std.mem.zeroes(c.ds4_suffix_relocation_stats);
        const message = "suffix reuse requires even-aligned source and destination within the context";
        const cache = maybe_cache orelse return fail(errors, message);
        const prompt = maybe_prompt orelse return fail(errors, message);
        const state = self.session.runtime.state;
        if (!self.supported() or state.approximate_suffix or start != @as(u32, @bitCast(state.checkpoint.len)) or
            prompt.len < 0 or cache.engine != self.engine_identity or cache.ctx_size != state.ctx_size)
            return fail(errors, message);
        const tokens: []const c_int = if (prompt.len == 0) &.{} else prompt.v[0..@intCast(prompt.len)];
        if (!(Frontier{ .state = state }).startsWithCheckpoint(tokens)) return fail(errors, message);
        var mutated = false;
        const observed = self.graph().graft(cache, tokens, source_start, start, self.cancellation(), &mutated, errors) catch {
            if (mutated) {
                state.checkpoint_valid = false;
                return self.graftFailure(errors);
            }
            return 1;
        };
        if (mutated) state.checkpoint_valid = false;
        for (tokens[start..observed.end]) |token| c.ds4_tokens_push(&state.checkpoint, token);
        state.checkpoint_valid = true;
        state.approximate_suffix = true;
        state.mtp_draft_valid = false;
        self.session.clearCapture();
        if (self.session.sync(prompt, errors) != 0) return self.graftFailure(errors);
        if (statistics) |output| output.* = observed.reuse;
        if (relocation) |output| output.* = observed.relocation;
        return 0;
    }
    fn graftFailure(self: *const Transaction, errors: []u8) c_int {
        self.session.invalidate();
        if (errors.len != 0 and errors[0] != 0) return 1;
        return fail(errors, "suffix graft or fresh-tail evaluation failed");
    }
};
