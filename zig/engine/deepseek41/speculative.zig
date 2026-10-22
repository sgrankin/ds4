//! One V4.1 speculative cycle over the real target and drafter owners.
//! Sessions stay opaque: this action borrows their checkpoint/frontier buffers,
//! explicit policy and transport identity. It never owns a model or cache.
const std = @import("std");
const c = @import("gguf_c");
const Executor = @import("executor.zig").Executor;
const Drafter = @import("dspark.zig").Drafter;
const dspark = @import("dspark.zig");
const PrefillPolicy = @import("prefill_policy.zig").Policy;
const Model = c.ds4_model;
const Stats = c.ds4_dspark_spec_stats;
const verify_rows = c.DS41_VERIFY_ROWS;

fn nowSeconds() f64 {
    var time: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &time);
    return @as(f64, @floatFromInt(time.tv_sec)) + @as(f64, @floatFromInt(time.tv_nsec)) * 1.0e-9;
}

fn verifyCap() u32 {
    const text = c.getenv("DS4_DSPARK_VERIFY_CAP");
    if (text == null or text[0] == 0) return 0;
    var end: [*c]u8 = null;
    c.__error().* = 0;
    const count = c.strtoul(text, &end, 10);
    // Like the existing environment parser, a numeric prefix is sufficient.
    if (end == text or c.__error().* != 0 or count > std.math.maxInt(u32)) return 0;
    return @intCast(count);
}

fn enabledStats(stats: *Stats) ?*Stats {
    const text = c.getenv("DS4_DSPARK_STATS");
    return if (text != null and text[0] != 0 and c.strcmp(text, "0") != 0) stats else null;
}

fn noteLength(histogram: *[c.DS4_DSPARK_MAX_BLOCK_SIZE + 1]u64, count: u32) void {
    histogram[@min(count, c.DS4_DSPARK_MAX_BLOCK_SIZE)] +%= 1;
}

fn sigmoid(value: f32) f32 {
    @setFloatMode(.optimized); // This C helper belongs to the fast-math engine TU.
    if (value >= 0) {
        const exponential = c.expf(-value);
        return 1.0 / (1.0 + exponential);
    }
    const exponential = c.expf(value);
    return exponential / (1.0 + exponential);
}

/// Native view of the borrowed arguments. Generated C pointers are narrowed
/// once here; subsequent checkpoint and sampling operations use real slices.
pub const Cycle = struct {
    executor: Executor,
    checkpoint: *c.ds4_tokens,
    checkpoint_valid: *bool,
    logits: []f32,
    sample_probs: []f32,
    cancel: c.ds4_session_cancel_fn,
    cancel_userdata: ?*anyopaque,
    stats: *Stats,
    policy: c.ds4_dspark_cycle_policy,
    tp: c.ds4_dspark_cycle_transport,

    pub fn init(executor: Executor, arguments: *const c.ds4_dspark_cycle_args) Cycle {
        const vocabulary = executor.ops.config.shape.n_vocab;
        return .{
            .executor = executor,
            .checkpoint = @ptrCast(arguments.checkpoint),
            .checkpoint_valid = @ptrCast(arguments.checkpoint_valid),
            .logits = arguments.logits[0..vocabulary],
            .sample_probs = arguments.sample_probs[0..vocabulary],
            .cancel = arguments.cancel,
            .cancel_userdata = arguments.cancel_userdata,
            .stats = @ptrCast(arguments.stats),
            .policy = arguments.policy,
            .tp = arguments.tp,
        };
    }

    fn leader(self: *const Cycle) bool {
        return self.tp.handle != null and self.tp.rank == 0;
    }

    fn draftState(self: *const Cycle) ?*c.ds41_dspark {
        const pointer = self.executor.graph.dspark;
        return if (pointer == null) null else @ptrCast(pointer);
    }

    fn drafter(self: *const Cycle, model: *const Model) ?Drafter {
        const state = self.draftState() orelse return null;
        return .{
            .state = state,
            .ops = .{
                .config = self.executor.ops.config,
                .model = model,
                .exact_prefill_rows = self.executor.ops.exact_prefill_rows,
            },
        };
    }

    /// Both ranks retain the small drafter. A sharded target head requires its
    /// complete rows before the leader can apply the sequential Markov bias.
    pub fn draft(self: *const Cycle, model: *const Model, token: c_int) bool {
        const draft_owner = self.drafter(model) orelse return false;
        const graph = self.executor.graph;
        const state = draft_owner.state;
        const split = self.tp.handle != null and self.tp.vocab_split;
        if (split and self.tp.rank == 0 and
            c.ds4_tp_send_dspark_draft(self.tp.handle, self.tp.session_id, token, @intCast(graph.pos), c.DS4_TP_DSPARK_DRAFT) == 0) return false;
        if (!draft_owner.forward(self.executor.ops.model, self.executor.weights, graph.pos, token, graph)) return false;
        if (!split) return true;
        const vocabulary = self.executor.ops.config.shape.n_vocab;
        const half = vocabulary / 2;
        // draft_probs is unused until Markov sampling. Pack the local halves
        // there so expanding a row cannot overwrite the following source row.
        if (c.ds4_gpu_tensor_read(state.logits, 0, state.draft_probs, @as(u64, c.DS41_DRAFT_ROWS) * half * 4) == 0) return false;
        for (0..c.DS41_DRAFT_ROWS) |index| {
            const local = state.draft_probs + index * half;
            const row = state.host_logits + index * vocabulary;
            if (self.tp.rank == 1) {
                if (c.ds4_tp_send_logits_half(self.tp.handle, local, half) == 0) return false;
            } else {
                @memcpy(row[0..half], local[0..half]);
                if (c.ds4_tp_recv_logits_half(self.tp.handle, row + half, half) == 0) return false;
            }
        }
        return self.tp.rank == 1 or c.ds4_gpu_tensor_write(state.logits, 0, state.host_logits, @as(u64, c.DS41_DRAFT_ROWS) * vocabulary * 4) != 0;
    }

    pub fn verify(self: *const Cycle, tokens: []const c_int) bool {
        const graph = self.executor.graph;
        const state = self.draftState() orelse return false;
        const count: u32 = @intCast(tokens.len);
        if (self.leader() and c.ds4_tp_send_verify(self.tp.handle, self.tp.session_id, tokens.ptr, count) == 0) return false;
        // Scalar TP merges its head on the host. Save that complete frontier
        // for keep=0, rather than the device's stale other vocabulary half.
        if (c.ds4_gpu_tensor_write(graph.logits, 0, self.logits.ptr, self.logits.len * 4) == 0 or
            !dspark.verify(state, self.executor, self.tp.verify_handle, tokens)) return false;
        if (self.tp.handle == null or !self.tp.vocab_split) return true;
        const vocabulary = self.executor.ops.config.shape.n_vocab;
        const half = vocabulary / 2;
        for (0..count) |index| {
            const row = state.host_verified + index * vocabulary;
            if (self.tp.rank == 1) {
                if (c.ds4_tp_send_logits_half(self.tp.handle, row + half, half) == 0) return false;
            } else if (c.ds4_tp_recv_logits_half(self.tp.handle, row + half, half) == 0) return false;
        }
        return self.tp.rank == 1 or c.ds4_gpu_tensor_write(state.verify_logits, 0, state.host_verified, @as(u64, count) * vocabulary * 4) != 0;
    }

    /// Publish accepted tokens only after the device frontier and TP worker
    /// acknowledge the same commit. Outbound vectors keep C allocation guards.
    pub fn commit(self: *const Cycle, tokens: []const c_int, keep: u32) bool {
        const graph = self.executor.graph;
        const state = self.draftState() orelse return false;
        if (keep > tokens.len or (self.leader() and c.ds4_tp_send_verify_commit(self.tp.handle, if (keep == tokens.len) c.DS4_TP_VERIFY_COMMIT_FULL else c.DS4_TP_VERIFY_COMMIT_PREFIX, @intCast(keep)) == 0)) return false;
        if (!dspark.commit(self.executor.ops.config, state, graph, keep, self.logits.ptr)) return false;
        var error_text: [128]u8 = undefined;
        if (self.leader() and c.ds4_tp_wait_command_ack(self.tp.handle, self.tp.session_id, "V4.1 verify commit", &error_text, error_text.len) == 0) return false;
        for (tokens[0..keep]) |token| c.ds4_tokens_push(self.checkpoint, token);
        return true;
    }

    fn ordinary(context: anytype, token: c_int, accepted: []c_int, error_text: []u8) c_int {
        if (context.eval(token, error_text) != 0) return -1;
        accepted[0] = token;
        return 1;
    }

    fn fail(self: *const Cycle, stats: ?*Stats, error_text: []u8) c_int {
        if (c.ds4_gpu_commands_active() != 0) _ = c.ds4_gpu_end_commands();
        self.executor.graph.valid = false;
        self.checkpoint_valid.* = false;
        if (stats) |counters| counters.verifier_errors +%= 1;
        _ = c.snprintf(error_text.ptr, error_text.len, "V4.1 DSpark draft/verification failed at position %u", self.executor.graph.pos);
        if (self.tp.handle != null) c.ds4_tp_mark_failed(self.tp.handle);
        return -1;
    }

    /// Greedy and exact sampled acceptance share one verifier transaction.
    /// Each examined exact proposal consumes one acceptance draw; only a
    /// rejection samples max(0,p-q), before commit and correction evaluation.
    /// Context is a typed Session primitive: supported callers compose the
    /// native Coordinator/Selection directly, with no controller callbacks.
    pub fn run(self: *const Cycle, context: anytype, model: *const Model, first_token: c_int, requested_cap: c_int, eos: c_int, ignore_eos: bool, think_mode: c.ds4_think_mode, temperature: f32, top_k: c_int, top_p: f32, min_p: f32, rng: ?*u64, accepted: []c_int, error_text: []u8) c_int {
        const graph = self.executor.graph;
        if (accepted.len == 0 or requested_cap < 1 or !self.checkpoint_valid.* or !graph.valid or
            graph.pos != @as(u32, @bitCast(self.checkpoint.len))) return -1;
        var cap = requested_cap;
        const space: c_int = @bitCast(graph.ctx -% graph.pos);
        if (cap > space) cap = space;
        if (cap < 1) return 0;
        cap = @min(cap, verify_rows);
        const configured_cap = verifyCap();
        if (configured_cap != 0 and configured_cap < @as(u32, @intCast(cap)) - 1)
            cap = @intCast(configured_cap + 1);
        const stats = enabledStats(self.stats);
        if (stats) |counters| {
            counters.cycles +%= 1;
            counters.first_tokens +%= 1;
        }
        const policy = PrefillPolicy{ .config = self.executor.ops.config, .graph = graph, .weights = self.executor.weights };
        var ready = graph.dspark != null and !self.policy.quality and !self.policy.strict and cap > 1 and
            first_token != eos and graph.pos != 0 and graph.pos <= 1048576 - c.DS41_DRAFT_ROWS and policy.tpBatchEnabled();
        // Old session payloads contain target caches, not draft features. Warm
        // their missing window with ordinary decoding; never fabricate features.
        if (ready) {
            const state = self.draftState().?;
            var position = graph.pos -| c.DS41_DRAFT_WINDOW;
            while (position < graph.pos) : (position += 1) {
                for (0..3) |stage| ready = ready and state.positions[stage][position % c.DS41_DRAFT_WINDOW] == position;
            }
        }
        if (!ready) {
            if (stats) |counters| counters.no_draft +%= 1;
            return ordinary(context, first_token, accepted, error_text);
        }
        const state = self.draftState().?;
        const draft_owner = self.drafter(model).?;
        const vocabulary = self.executor.ops.config.shape.n_vocab;
        const exact = temperature > 0 and self.policy.exact_sampling;
        const started = nowSeconds();
        if (!self.draft(model, first_token)) return self.fail(stats, error_text);
        var tokens: [verify_rows]c_int = @splat(0);
        tokens[0] = first_token;
        var count: u32 = 1;
        while (count < @as(u32, @intCast(cap))) : (count += 1) {
            const row = count - 1;
            var confidence: f32 = undefined;
            if (!draft_owner.markov(row, tokens[row], &confidence)) return self.fail(stats, error_text);
            if (self.policy.confidence_threshold > 0 and sigmoid(confidence) < self.policy.confidence_threshold) break;
            const proposal = state.host_logits + @as(u64, row) * vocabulary;
            if (exact) {
                const probabilities = state.draft_probs + @as(u64, row) * vocabulary;
                const random = rng orelse return self.fail(stats, error_text);
                if (!c.ds4_sampling_build_probabilities(proposal, vocabulary, temperature, top_k, top_p, min_p, probabilities))
                    return self.fail(stats, error_text);
                tokens[count] = c.ds4_sampling_probabilities(probabilities, vocabulary, random);
            } else tokens[count] = c.ds4_sampling_argmax(proposal, vocabulary);
            if (ignore_eos and context.stop(tokens[count], think_mode)) break;
            if (tokens[count] == eos) {
                count += 1;
                break;
            }
        }
        if (count == 1) {
            if (stats) |counters| {
                counters.no_draft +%= 1;
                counters.propose_ms += (nowSeconds() - started) * 1000;
            }
            return ordinary(context, first_token, accepted, error_text);
        }
        if (self.cancel) |cancel| {
            if (cancel(self.cancel_userdata)) {
                _ = c.snprintf(error_text.ptr, error_text.len, "interrupted");
                return -1; // No target mutation or invalidation on this boundary.
            }
        }
        const proposed = nowSeconds();
        if (stats) |counters| {
            counters.propose_ms += (proposed - started) * 1000;
            counters.proposed_tokens +%= count - 1;
            noteLength(&counters.draft_len_hist, count - 1);
        }
        if (!self.verify(tokens[0..count])) return self.fail(stats, error_text);
        var keep: u32 = 1;
        var correction: c_int = -1;
        while (keep < count) : (keep += 1) {
            const target = state.host_verified + @as(u64, keep - 1) * vocabulary;
            if (!exact) {
                if (c.ds4_sampling_argmax(target, vocabulary) != tokens[keep]) break;
            } else {
                const proposal = state.draft_probs + @as(u64, keep - 1) * vocabulary;
                if (!c.ds4_sampling_build_probabilities(target, vocabulary, temperature, top_k, top_p, min_p, self.sample_probs.ptr)) return self.fail(stats, error_text);
                const random = rng.?; // Proposal sampling above established it.
                const token: usize = @intCast(tokens[keep]);
                if (c.ds4_sampling_rng_f32(random) * proposal[token] > self.sample_probs[token]) {
                    correction = c.ds4_sampling_residual_probabilities(self.sample_probs.ptr, proposal, vocabulary, random);
                    break;
                }
            }
        }
        if (!self.commit(tokens[0..count], keep)) return self.fail(stats, error_text);
        @memcpy(accepted[0..keep], tokens[0..keep]);
        const accepted_drafts = keep - 1;
        if (stats) |counters| {
            counters.verify_ms += (nowSeconds() - proposed) * 1000;
            counters.accepted_draft_tokens +%= accepted_drafts;
            noteLength(&counters.accepted_len_hist, accepted_drafts);
            if (keep == count) {
                counters.full_accepts +%= 1;
                counters.direct_full_commits +%= 1;
            } else {
                counters.partial_accepts +%= 1;
                counters.direct_partial_commits +%= 1;
            }
        }
        if (correction >= 0) {
            if (context.eval(correction, error_text) != 0) return -1;
            accepted[keep] = correction;
            keep += 1;
            // The correction occupies an already-saved rejected ring slot.
            // Earlier verified prefixes remain available for a later rewind.
            state.rewind_end = graph.pos;
        }
        if (c.getenv("DS4_DSPARK_SPEC_LOG") != null) _ = c.fprintf(c.__stderrp, "ds4: V4.1 DSpark proposed=%u accepted=%u draft=%.3f ms verify=%.3f ms\n", count - 1, accepted_drafts, (proposed - started) * 1000, (nowSeconds() - proposed) * 1000);
        if (stats) |counters| counters.total_ms += (nowSeconds() - started) * 1000;
        return @intCast(keep);
    }
};
