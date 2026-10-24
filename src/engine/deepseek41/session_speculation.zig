//! Complete supported public speculative admission and rewind transactions.
//! Numerical drafting/verification belongs to Cycle/DSpark; ordinary eval and
//! failure invalidation compose Coordinator directly. No Session layout is read.
const c = @import("gguf_c");
const Coordinator = @import("coordinator.zig").Coordinator;
const Selection = @import("../session_sampling.zig").Selection;
const Cycle = @import("speculative.zig").Cycle;
const dspark = @import("dspark.zig");
const batch_owner = @import("batch_transaction.zig");
const View = c.ds4_session_batch_view;

fn fail(errors: []u8, message: []const u8) void {
    if (errors.len != 0) {
        const length = @min(errors.len - 1, message.len);
        @memcpy(errors[0..length], message[0..length]);
        errors[length] = 0;
    }
}

pub const Session = struct {
    coordinator: Coordinator,
    identity: ?*c.ds4_session,
    distributed: ?*c.ds4_dist_session,
    statistics: *c.ds4_dspark_spec_stats,
    verify_transport: ?*c.ds4_tp,

    pub fn eval(self: *const Session, token: c_int, errors: []u8) c_int {
        return self.coordinator.eval(self.identity, self.distributed, token, errors);
    }
    pub fn stop(self: *const Session, token: c_int, mode: c.ds4_think_mode) bool {
        const owner = self.selection();
        return owner.stop(token, mode);
    }
    fn selection(self: *const Session) Selection {
        return .{
            .family = c.DS4_MODEL_FAMILY_DEEPSEEK41,
            .flint = null,
            .state = self.coordinator.runtime.state,
            .policy = &self.coordinator.core.overthink,
            .vocab = &self.coordinator.core.vocab,
            .count = self.coordinator.runtime.prefill.executor.ops.config.shape.n_vocab,
        };
    }
    fn cycle(self: *const Session) Cycle {
        const owner = &self.coordinator;
        const frontier = owner.runtime.state;
        const transport = owner.transport;
        const vocabulary = owner.runtime.prefill.executor.ops.config.shape.n_vocab;
        return .{
            .executor = owner.runtime.prefill.executor,
            .checkpoint = &frontier.checkpoint,
            .checkpoint_valid = &frontier.checkpoint_valid,
            .logits = frontier.logits[0..vocabulary],
            .sample_probs = frontier.sample_probs[0..vocabulary],
            .cancel = owner.control.cancel,
            .cancel_userdata = owner.control.cancel_ud,
            .stats = self.statistics,
            .policy = .{ .quality = owner.core.quality, .strict = owner.core.dspark_strict, .exact_sampling = owner.core.dspark_exact_sampling, .confidence_threshold = owner.core.dspark_confidence_threshold },
            .tp = .{ .handle = if (transport.active) transport.ctx else null, .verify_handle = self.verify_transport, .session_id = owner.control.tp_session_id, .rank = transport.rank, .vocab_split = transport.active and transport.vocab_split },
        };
    }
    fn ordinary(self: *const Session, token: c_int, accepted: [*c]c_int, errors: []u8) c_int {
        if (accepted == null) return 0;
        if (self.eval(token, errors) != 0) return -1;
        accepted[0] = token;
        return 1;
    }

    pub fn argmax(self: *const Session, first: c_int, maximum: c_int, eos: c_int, ignore_eos: bool, mode: c.ds4_think_mode, accepted: [*c]c_int, capacity: c_int, errors: []u8) c_int {
        if (maximum <= 0 or capacity <= 0) return 0;
        if (!self.coordinator.runtime.state.checkpoint_valid) {
            fail(errors, "speculative decode requires a synchronized checkpoint");
            return -1;
        }
        const count = @min(capacity, maximum);
        // A nonzero penalty selects the adjusted one-token path. Speculative
        // acceptance must not silently sample the unadjusted target instead.
        if (self.coordinator.core.overthink.penalty > 0) return self.ordinary(first, accepted, errors);
        if (accepted == null) return -1;
        const output = accepted[0..@intCast(@min(count, c.DS41_VERIFY_ROWS))];
        const action = self.cycle();
        return action.run(self, &self.coordinator.core.mtp_model, first, count, eos, ignore_eos, mode, 0, 0, 1, 0, null, output, errors);
    }
    pub fn sample(self: *const Session, first: c_int, maximum: c_int, eos: c_int, temperature: f32, top_k: c_int, top_p: f32, min_p: f32, rng: ?*u64, accepted: [*c]c_int, capacity: c_int, errors: []u8) c_int {
        // This dispatch intentionally precedes nullable RNG/output admission.
        if (temperature <= 0) return self.argmax(first, maximum, eos, false, c.DS4_THINK_HIGH, accepted, capacity, errors);
        if (accepted == null or rng == null or maximum <= 0 or capacity <= 0) return 0;
        if (!self.coordinator.runtime.state.checkpoint_valid) {
            fail(errors, "speculative decode requires a synchronized checkpoint");
            return -1;
        }
        const count = @min(capacity, maximum);
        if (self.coordinator.core.overthink.penalty > 0) return self.ordinary(first, accepted, errors);
        const action = self.cycle();
        return action.run(self, &self.coordinator.core.mtp_model, first, count, eos, false, c.DS4_THINK_HIGH, temperature, top_k, top_p, min_p, rng, accepted[0..@intCast(@min(count, c.DS41_VERIFY_ROWS))], errors);
    }

    fn canRestore(self: *const Session, position: c_int) bool {
        const state = self.coordinator.runtime.state;
        const graph = self.coordinator.runtime.prefill.executor.graph;
        const draft = graph.dspark;
        if (!state.checkpoint_valid or !graph.valid or draft == null or position < 0) return false;
        const target: u32 = @intCast(position);
        return draft.*.rewind_end == @as(u32, @bitCast(state.checkpoint.len)) and
            target >= draft.*.verify_start and target < draft.*.rewind_end;
    }
    /// State-only restore is a worker action as well as the local half of a
    /// mirrored rewind. The coordinator replays before sampling interior logits.
    pub fn restorePrefix(self: *const Session, position: c_int) c_int {
        if (!self.canRestore(position)) return 1;
        const graph = self.coordinator.runtime.prefill.executor.graph;
        const draft: *c.ds41_dspark = @ptrCast(graph.dspark);
        const state = self.coordinator.runtime.state;
        if (!dspark.commit(self.coordinator.runtime.prefill.executor.ops.config, draft, graph, @as(u32, @intCast(position)) - draft.verify_start, state.logits)) {
            state.checkpoint_valid = false;
            return -1;
        }
        state.checkpoint.len = position;
        return 0;
    }
    pub fn rewindSpeculative(self: *const Session, position: c_int) bool {
        if (!self.canRestore(position)) return false;
        const owner = &self.coordinator;
        const transport = owner.transport;
        const mirror = transport.active and transport.rank == 0;
        if (mirror and c.ds4_tp_send_spec_restore(transport.ctx, owner.control.tp_session_id, position) == 0) return false;
        const local = self.restorePrefix(position);
        var peer: c_int = 0;
        var error_text: [128]u8 = undefined;
        if (mirror and c.ds4_tp_wait_command_status(transport.ctx, owner.control.tp_session_id, &peer, "V4.1 speculative restore", &error_text, error_text.len) == 0) peer = -1;
        if (local != 0 or peer != 0) {
            owner.runtime.state.checkpoint_valid = false;
            if (mirror) owner.invalidate();
            return false;
        }
        return true;
    }
    pub fn rewind(self: *const Session, requested: c_int) void {
        const position = @max(0, requested);
        const owner = &self.coordinator;
        const state = owner.runtime.state;
        if (position >= state.checkpoint.len or self.rewindSpeculative(position)) return;
        const transport = owner.transport;
        if (transport.active and transport.rank == 0 and !c.ds4_tp_failed(transport.ctx)) {
            if (c.ds4_tp_send_rewind(transport.ctx, owner.control.tp_session_id, position) == 0)
                state.checkpoint_valid = false;
        }
        state.checkpoint.len = position;
        // DeepSeek compressors cannot be rolled back by truncating row counts.
        // Without a saved frontier, the caller must rebuild this retained prefix.
        state.checkpoint_valid = false;
        state.mtp_draft_valid = false;
        owner.clearCapture();
        // Legacy GLM/generic capture is unallocated in this supported DS41 owner.
    }
};

pub const Batch = struct {
    ordinary: batch_owner.Batch,
    verify_transport: ?*c.ds4_tp,

    fn session(self: *const Batch, view: View) Session {
        return .{
            .coordinator = self.ordinary.row(view),
            .identity = view.session,
            .distributed = view.distributed,
            .statistics = @ptrCast(view.statistics),
            .verify_transport = self.verify_transport,
        };
    }
    pub fn decode(self: *const Batch, views: []const View, accepted: [*c][2]c_int, counts: [*c]c_int, errors: []u8) batch_owner.Result {
        if (views.len == 0 or accepted == null or counts == null) {
            fail(errors, "empty speculative decode batch");
            return .{ .status = 1 };
        }
        const vocabulary = self.ordinary.session.runtime.prefill.executor.ops.config.shape.n_vocab;
        for (views, 0..) |view, index| {
            const item: c_int = @intCast(index);
            if (view.session == null or view.engine != self.ordinary.engine_identity or self.ordinary.engine_identity == null) {
                if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "decode batch item %d belongs to a different engine", item);
                return .{ .status = 1 };
            }
            if (view.token < 0 or view.token >= @as(c_int, @intCast(vocabulary))) {
                if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "decode batch item %d has an invalid token", item);
                return .{ .status = 1 };
            }
            for (views[0..index], 0..) |previous, before| {
                if (previous.session == view.session) {
                    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "decode batch repeats session at items %d and %d", @as(c_int, @intCast(before)), item);
                    return .{ .status = 1 };
                }
            }
            const frontier: *const c.ds4_session_frontier = @ptrCast(view.frontier);
            if (frontier.checkpoint.len >= frontier.ctx_size) {
                if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "decode batch item %d reached its context limit", item);
                return .{ .status = 1 };
            }
        }
        if (self.ordinary.session.core.overthink.penalty > 0) {
            const result = self.ordinary.decode(views, errors);
            if (result.status != 0) return result;
            for (views, 0..) |view, index| {
                accepted[index][0] = view.token;
                counts[index] = 1;
            }
            return result;
        }
        // The supported DS41 path runs one cycle per Session. Every member is
        // invalidated if a later cycle fails after earlier members advanced.
        for (views, 0..) |view, index| {
            const owner = self.session(view);
            const state = owner.coordinator.runtime.state;
            const room = state.ctx_size - state.checkpoint.len;
            var tokens: [3]c_int = undefined;
            const count = owner.argmax(view.token, @min(room, 2), -1, false, c.DS4_THINK_HIGH, &tokens, 2, errors);
            if (count <= 0) {
                self.ordinary.invalidate(views, null);
                return .{ .status = 1 };
            }
            counts[index] = count;
            accepted[index][0] = tokens[0];
            accepted[index][1] = if (count > 1) tokens[1] else -1;
        }
        return .{ .status = 0 };
    }
};
