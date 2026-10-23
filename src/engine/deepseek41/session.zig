//! V4.1 session synchronization and committed decode frontiers for Metal.
//! Numerical execution and graph lifetime use the existing native owners.
//! C still dispatches other families/TP transport; native Memory owns admission;
//! no Engine/Session layout or individual-field service is imported here.
const std = @import("std");
const c = @import("gguf_c");
const Prefill = @import("prefill.zig").Prefill;
const Callbacks = @import("prefill.zig").Callbacks;
const scheduling = @import("prefill_policy.zig");
const Policy = scheduling.Policy;
const residency = @import("encoder_residency.zig");
const Frontier = @import("frontier.zig").Frontier;
const resources = @import("resources.zig");
const Memory = @import("memory.zig").Memory;
const Graph = c.ds41_gpu_graph;
const State = c.ds4_session_frontier;

fn cancelled(callbacks: Callbacks) bool {
    return if (callbacks.cancel) |check| check(callbacks.cancel_userdata) else false;
}
fn message(destination: []u8, text: []const u8) void {
    if (destination.len == 0) return;
    const count = @min(destination.len - 1, text.len);
    @memcpy(destination[0..count], text[0..count]);
    destination[count] = 0;
}
fn report(callbacks: Callbacks, current: c_int) void {
    if (callbacks.progress) |progress|
        progress(callbacks.progress_userdata, "prefill_chunk", current, callbacks.total);
}

pub fn invalidate(state: *State, graph: ?*Graph) void {
    const frontier = Frontier{ .state = state };
    frontier.invalidate();
    if (graph) |ready| resources.resetRecord(ready);
}

pub const Runtime = struct {
    state: *State,
    prefill: Prefill,
    // The actual Engine budget remains authoritative, borrowed by the native
    // admission owner. No Engine layout or C workspace controller is needed.
    memory: Memory,
    ready: bool,

    fn policy(self: *const Runtime, graph: *const Graph) Policy {
        return .{ .config = self.prefill.executor.ops.config, .graph = graph, .weights = self.prefill.executor.weights };
    }

    /// Bring live state to exactly this token prefix. Exact-prefix extension
    /// preserves caches; any rewrite behind the live frontier rebuilds them.
    /// Wide chunks finalize compressor/index rows in the established order.
    pub fn sync(self: *const Runtime, maybe_prompt: ?*const c.ds4_tokens, callbacks_: Callbacks, errors: []u8) c_int {
        const state = self.state;
        const prompt = maybe_prompt orelse {
            message(errors, "missing session or prompt");
            return 1;
        };
        if (prompt.len <= 0) {
            message(errors, "empty prompt");
            return 1;
        }
        if (prompt.len >= state.ctx_size) {
            if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "prompt length %d exceeds context %d (one token of generation room is required)", prompt.len, state.ctx_size);
            return 1;
        }
        var callbacks = callbacks_;
        callbacks.total = prompt.len;
        if (cancelled(callbacks)) {
            message(errors, "interrupted");
            return c.DS4_SESSION_SYNC_INTERRUPTED;
        }
        const executor = &self.prefill.executor;
        const graph = executor.graph;
        if (!self.ready or graph.workspace_failed) {
            message(errors, "V4.1 graph is not initialized");
            return 1;
        }
        const frontier = Frontier{ .state = state };
        const tokens = prompt.v[0..@intCast(prompt.len)];
        if (!state.checkpoint_valid or !graph.valid or graph.pos != @as(u32, @intCast(state.checkpoint.len)) or
            !frontier.startsWithCheckpoint(tokens))
        {
            resources.resetRecord(graph);
            state.checkpoint.len = 0;
            state.checkpoint_valid = false;
            frontier.clearApproximation();
        }

        if (graph.prefill_max > graph.prefill_cap) {
            // This is a borrowed shape view only, never an owned graph copy.
            var shape = graph.*;
            shape.prefill_cap = graph.prefill_max;
            const future = self.policy(&shape);
            const need = future.encoderChunk(@intCast(prompt.len - state.checkpoint.len));
            if (need > graph.prefill_cap) {
                const remaining: u32 = @intCast(prompt.len - state.checkpoint.len);
                if (!self.memory.growWorkspace(graph, need, remaining, errors)) {
                    if (graph.workspace_failed) state.checkpoint_valid = false;
                    return 1;
                }
                state.prefill_cap = graph.prefill_cap;
            }
        }

        var lease = std.mem.zeroes(residency.Lease);
        residency.acquire(&lease, graph, executor.ops.model, executor.weights, executor.ops.config, @intCast(prompt.len - state.checkpoint.len), callbacks.cancel, callbacks.cancel_userdata);
        defer residency.release(&lease, graph, executor.ops.model);
        var pending_logits = false;
        var interrupted = false;
        var decoder_pending = false;
        var position = state.checkpoint.len;
        while (position < prompt.len) {
            if (cancelled(callbacks)) {
                interrupted = true;
                break;
            }
            const remaining: u32 = @intCast(prompt.len - position);
            if (graph.encoder_resident and remaining < 768)
                residency.release(&lease, graph, executor.ops.model);
            const schedule = self.policy(graph);
            // The old short_prefill_count is CUDA-only and is always zero on
            // Apple. Ordinary Metal selected tiles remain in prefillCount.
            const count: u32 = if (graph.encoder_resident)
                @min(remaining - 512, graph.prefill_cap)
            else
                schedule.prefillCount(remaining);
            const defer_decoder = !graph.encoder_resident and count >= 16384 and remaining - count >= 8192 and
                c.getenv("DS4_METAL_DISABLE_V41_DECODER_SUFFIX") == null and
                c.getenv("DS4_METAL_DISABLE_V41_DEFER_DECODER") == null;
            const layer_major = count > 1;
            const replay_rows = if (layer_major) schedule.replayRows(count) else 0;
            if (replay_rows != 0) {
                const selected = scheduling.replaySelected();
                if (state.approximate_decoder_replay and
                    (state.decoder_replay_rows != replay_rows or state.decoder_replay_selected != selected))
                {
                    residency.release(&lease, graph, executor.ops.model);
                    message(errors, "decoder replay policy changed; invalidate or restore an exact checkpoint first");
                    return 1;
                }
                // Mark BEFORE approximate writes, including a deferred encoder
                // sweep. The canonical payload/suffix gates use this same state.
                state.approximate_suffix = true;
                state.approximate_decoder_replay = true;
                state.decoder_replay_rows = replay_rows;
                state.decoder_replay_selected = selected;
            }
            const start: usize = @intCast(position);
            const current = tokens[start..][0..count];
            // Never add an encoder sweep solely to defer a decoder. A short
            // remainder finishes the pending decoder before ordinary execution.
            const ok = if (layer_major)
                self.prefill.sweep(current, callbacks, defer_decoder, decoder_pending)
            else
                executor.step(tokens[start], null);
            if (!ok) {
                residency.release(&lease, graph, executor.ops.model);
                state.checkpoint_valid = false;
                if (cancelled(callbacks)) {
                    message(errors, "interrupted");
                    return c.DS4_SESSION_SYNC_INTERRUPTED;
                }
                if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "V4.1 prefill failed at position %d", position);
                return 1;
            }
            for (current) |token| c.ds4_tokens_push(&state.checkpoint, token);
            position += @intCast(count);
            decoder_pending = defer_decoder;
            pending_logits = true;
            state.checkpoint_valid = false;
            report(callbacks, position);
        }
        residency.release(&lease, graph, executor.ops.model);
        if (decoder_pending) {
            state.checkpoint_valid = false;
            graph.valid = false;
            message(errors, "interrupted");
            return c.DS4_SESSION_SYNC_INTERRUPTED;
        }
        // One final head is sufficient for prefill. Even an interrupted prefix
        // needs matching logits before it is saveable or safe to continue.
        if (pending_logits) {
            if (!executor.logits(state.logits)) {
                graph.valid = false;
                message(errors, "V4.1 prefill output head failed");
                return 1;
            }
            state.checkpoint_valid = true;
            report(callbacks, state.checkpoint.len);
        }
        if (interrupted) {
            message(errors, "interrupted");
            return c.DS4_SESSION_SYNC_INTERRUPTED;
        }
        return 0;
    }

    pub fn eval(self: *const Runtime, token: c_int, errors: []u8) c_int {
        const state = self.state;
        const graph = self.prefill.executor.graph;
        const draft = graph.dspark;
        const rewindable = draft != null and draft.*.rewind_end == graph.pos and draft.*.rewind_end != 0 and
            draft.*.rewind_end < draft.*.verify_start + draft.*.verify_count;
        if (!self.ready or (!state.checkpoint_valid and state.checkpoint.len != 0) or
            graph.pos != @as(u32, @intCast(state.checkpoint.len)) or
            !self.prefill.executor.step(token, state.logits))
        {
            state.checkpoint_valid = false;
            if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "V4.1 decode failed at position %d", state.checkpoint.len);
            return 1;
        }
        c.ds4_tokens_push(&state.checkpoint, token);
        state.checkpoint_valid = true;
        if (rewindable) draft.*.rewind_end = graph.pos;
        return 0;
    }
};
