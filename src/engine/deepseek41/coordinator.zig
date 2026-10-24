//! DS41 public request synchronization over actual shared owners. Numerical
//! prefill/decode remains Runtime; this coordinator owns image admission,
//! transport framing, callback scopes and successful request publication.
const std = @import("std");
const Control = @import("../session_control.zig");
const c = @import("gguf_c");
const Runtime = @import("session.zig").Runtime;
const runtime_owner = @import("session.zig");
const Frontier = @import("frontier.zig").Frontier;
const Callbacks = @import("prefill.zig").Callbacks;

fn fail(errors: []u8, message: []const u8) c_int {
    if (errors.len != 0) {
        const count = @min(errors.len - 1, message.len);
        @memcpy(errors[0..count], message[0..count]);
        errors[count] = 0;
    }
    return 1;
}
fn eventMatches(event: [*c]const u8, expected: []const u8) bool {
    return std.mem.eql(u8, std.mem.span(event), expected);
}
/// Correctness isolation: dump the post-prefill logits vector once. Its sole
/// process-wide flag lives in the existing export facade, shared with fallback.
pub fn dumpPrefillLogits(dumped: *bool, logits: [*c]const f32, vocabulary: u32) void {
    const path = c.getenv("DS4_GLM_LOGIT_DUMP");
    if (path == null or path[0] == 0 or dumped.* or logits == null) return;
    const stream = c.fopen(path, "wb");
    if (stream == null) return;
    _ = c.fwrite(logits, @sizeOf(f32), vocabulary, stream);
    _ = c.fclose(stream);
    dumped.* = true;
    _ = c.fprintf(c.__stderrp, "ds4: prefill logits dumped to %s\n", path);
}

pub const Coordinator = struct {
    runtime: Runtime,
    core: *c.ds4_engine_core,
    transport: *c.ds4_engine_tp_state,
    control: *c.ds4_session_control,
    dump_recorded: *bool,

    fn frontier(self: *const Coordinator) Frontier {
        return .{ .state = self.runtime.state };
    }
    fn leader(self: *const Coordinator) bool {
        return self.transport.active and self.transport.rank == 0;
    }
    fn callbacks(self: *const Coordinator) Callbacks {
        return .{ .progress = self.control.progress, .progress_userdata = self.control.progress_ud, .cancel = self.control.cancel, .cancel_userdata = self.control.cancel_ud };
    }
    /// A new public request clears proposal scheduling history. This is the
    /// original count-only reset, not a retuning of any speculative policy.
    fn beginRequest(self: *const Coordinator) void {
        Control.beginRequest(self.control);
    }
    pub fn invalidate(self: *const Coordinator) void {
        if (self.leader() and !c.ds4_tp_failed(self.transport.ctx))
            _ = c.ds4_tp_send_invalidate(self.transport.ctx, self.control.tp_session_id);
        runtime_owner.invalidate(self.runtime.state, if (self.runtime.ready) self.runtime.prefill.executor.graph else null);
        self.clearCapture();
    }
    pub fn clearCapture(self: *const Coordinator) void {
        Control.clearCapture(self.control);
    }
    fn localSync(self: *const Coordinator, prompt: ?*const c.ds4_tokens, errors: []u8) c_int {
        return self.runtime.sync(prompt, self.callbacks(), errors);
    }
    fn lockstepSync(self: *const Coordinator, prompt: *const c.ds4_tokens, errors: []u8) c_int {
        const control = self.control;
        var agreement = Agreement{ .owner = self, .progress = control.progress, .display = control.display_progress, .progress_userdata = control.progress_ud, .display_userdata = control.display_progress_ud, .cancel = control.cancel, .cancel_userdata = control.cancel_ud };
        control.progress = progressAgreed;
        control.display_progress = displayAgreed;
        control.progress_ud = &agreement;
        control.display_progress_ud = &agreement;
        control.cancel = cancelAgreed;
        control.cancel_ud = &agreement;
        defer {
            control.progress = agreement.progress;
            control.display_progress = agreement.display;
            control.progress_ud = agreement.progress_userdata;
            control.display_progress_ud = agreement.display_userdata;
            control.cancel = agreement.cancel;
            control.cancel_ud = agreement.cancel_userdata;
        }
        agreement.checkpoint(0, 0, prompt.len);
        var result = if (agreement.failed) @as(c_int, 1) else self.localSync(prompt, errors);
        agreement.checkpoint(3, result, prompt.len);
        if (agreement.failed) return fail(errors, "tp: prefill synchronization failed");
        if (agreement.cancelled and result == 0) {
            _ = fail(errors, "interrupted");
            result = c.DS4_SESSION_SYNC_INTERRUPTED;
        }
        return result;
    }

    /// The leader mirrors the request before local work. A worker ACK frames
    /// the control stream; even on local failure, successful split logits must
    /// be drained before both peers are invalidated.
    pub fn sync(self: *const Coordinator, prompt: ?*const c.ds4_tokens, errors: []u8) c_int {
        const state = self.runtime.state;
        self.control.dspark_rewind_end = 0;
        if (state.checkpoint_valid and !self.frontier().visionPrefixMatches(images(state.sync_images, state.sync_image_count))) self.invalidate();
        self.beginRequest();
        const mirror = self.leader();
        if (mirror and (prompt == null or prompt.?.len <= 0)) return self.localSync(prompt, errors);
        if (mirror and prompt != null and prompt.?.len > 0) {
            if (state.sync_image_count > std.math.maxInt(u32)) return fail(errors, "tp: too many image spans to synchronize");
            const sent = if (state.sync_image_count != 0)
                c.ds4_tp_send_sync_multimodal(self.transport.ctx, self.control.tp_session_id, prompt.?.v, @intCast(prompt.?.len), state.sync_images, @intCast(state.sync_image_count))
            else
                c.ds4_tp_send_sync(self.transport.ctx, self.control.tp_session_id, prompt.?.v, @intCast(prompt.?.len));
            if (sent == 0) return fail(errors, "tp: worker sync send failed");
        }
        const result = if (self.transport.active and prompt != null and prompt.?.len > 0)
            self.lockstepSync(prompt.?, errors)
        else
            self.localSync(prompt, errors);
        if (result == 0) dumpPrefillLogits(self.dump_recorded, state.logits, self.runtime.prefill.executor.ops.config.shape.n_vocab);
        // The legacy GLM KV dump uses its own graph, which is unallocated in
        // this DS41 owner; other-family dispatch still executes that C action.
        if (mirror) {
            var worker_status: c_int = 1;
            const acknowledged = !c.ds4_tp_failed(self.transport.ctx) and
                c.ds4_tp_wait_command_status(self.transport.ctx, self.control.tp_session_id, &worker_status, "prefill sync", errors.ptr, errors.len) != 0;
            const worker_ok = acknowledged and worker_status == 0;
            var logits_ok = true;
            if (worker_ok and self.transport.vocab_split) {
                const half = self.runtime.prefill.executor.ops.config.shape.n_vocab / 2;
                if (c.ds4_tp_recv_logits_half(self.transport.ctx, state.logits + half, half) == 0) {
                    _ = fail(errors, "tp: worker sync logits half missing");
                    logits_ok = false;
                }
            }
            if (result != 0 or !worker_ok or !logits_ok) {
                self.invalidate();
                if (result == 0 and acknowledged and worker_status != 0 and errors.len != 0)
                    _ = c.snprintf(errors.ptr, errors.len, "tp: worker prefill failed (status %d)", worker_status);
                return if (result != 0) result else 1;
            }
        }
        if (result == 0 and !self.frontier().rememberImages()) {
            self.invalidate();
            return fail(errors, "unable to retain image prompt identity");
        }
        if (result == 0 and prompt != null) state.overthink_prompt_len = prompt.?.len;
        return result;
    }
    /// Mirror public eval before local work, then merge vocabulary halves. All
    /// ordinary and speculative DS41 eval entry points use this same primitive.
    /// Distributed execution remains the existing whole transport library action.
    pub fn eval(self: *const Coordinator, identity: ?*c.ds4_session, distributed: ?*c.ds4_dist_session, token: c_int, errors: []u8) c_int {
        const state = self.runtime.state;
        if (!state.checkpoint_valid) return fail(errors, "decode requires a synchronized checkpoint");
        const mirror = self.leader();
        if (mirror) {
            self.transport.eval_seq +%= 1;
            if (c.ds4_tp_send_eval(self.transport.ctx, self.control.tp_session_id, self.transport.eval_seq, token) == 0)
                return fail(errors, "tp: worker eval send failed");
        }
        self.control.dspark_rewind_end = 0;
        const result = if (distributed) |owner|
            c.ds4_dist_session_eval(owner, identity, &state.checkpoint, token, state.logits, errors.ptr, errors.len)
        else
            self.runtime.eval(token, errors);
        if (result != 0 and mirror) {
            self.invalidate();
            return result;
        }
        if (result == 0 and self.transport.active and c.ds4_gpu_tp_failed() != 0) {
            _ = fail(errors, "tp: gate transport failed");
            if (mirror) self.invalidate();
            return 1;
        }
        if (result == 0 and self.transport.active and self.transport.vocab_split) {
            const half = self.runtime.prefill.executor.ops.config.shape.n_vocab / 2;
            if (self.transport.rank == 0) {
                if (c.ds4_tp_recv_logits_half(self.transport.ctx, state.logits + half, half) == 0) {
                    _ = fail(errors, "tp: worker logits half missing");
                    self.invalidate();
                    return 1;
                }
            } else if (c.ds4_tp_send_logits_half(self.transport.ctx, state.logits + half, half) == 0)
                return fail(errors, "tp: logits half send failed");
        }
        return result;
    }

    /// Image spans are a synchronous borrow. Reject every malformed span
    /// before publishing any borrow, then retire both real records on all exits.
    pub fn syncMultimodal(self: *const Coordinator, prompt: ?*const c.ds4_tokens, raw: [*c]const c.ds4_vision_span, count: usize, errors: []u8) c_int {
        const input = prompt orelse return fail(errors, "invalid multimodal prompt");
        if (count != 0 and raw == null) return fail(errors, "invalid multimodal prompt");
        if (count != 0 and !self.core.vision_ready) return fail(errors, "vision encoder is not loaded");
        var previous_end: u64 = 0;
        for (images(raw, count)) |span| {
            const end = @as(u64, span.token_start) + span.embedding.token_count;
            const prompt_length: u64 = @bitCast(@as(i64, input.len));
            if (span.embedding.data == null or span.embedding.token_count == 0 or span.token_start < previous_end or end > prompt_length)
                return fail(errors, "invalid or overlapping image token span");
            // DS41 represents the complete expanded block with image placeholders.
            // The older DeepSeek-family tagged-token parser remains C dispatch.
            for (span.token_start..end) |token| {
                if (input.v[token] != self.core.vision_image_token)
                    return fail(errors, "image span does not cover image placeholder tokens");
            }
            previous_end = end;
        }
        const state = self.runtime.state;
        const graph = self.runtime.prefill.executor.graph;
        state.sync_images = raw;
        state.sync_image_count = count;
        graph.images = raw;
        graph.image_count = count;
        defer {
            graph.images = null;
            graph.image_count = 0;
            state.sync_images = null;
            state.sync_image_count = 0;
        }
        return self.sync(input, errors);
    }
};
fn images(pointer: [*c]const c.ds4_vision_span, count: usize) []const c.ds4_vision_span {
    return if (count == 0) &.{} else pointer[0..count];
}

/// TP peers poll the real cancellation action only at equal graph positions.
/// Inner loops see the agreed result, including timing-dependent Engram waits.
const Agreement = struct {
    owner: *const Coordinator,
    progress: c.ds4_session_progress_fn,
    display: c.ds4_session_progress_fn,
    progress_userdata: ?*anyopaque,
    display_userdata: ?*anyopaque,
    cancel: c.ds4_session_cancel_fn,
    cancel_userdata: ?*anyopaque,
    cancelled: bool = false,
    failed: bool = false,
    fn checkpoint(self: *Agreement, point: u32, current: c_int, total: c_int) void {
        if (self.failed) return;
        const requested = self.cancelled or if (self.cancel) |check| check(self.cancel_userdata) else false;
        self.failed = c.ds4_tp_sync_checkpoint(self.owner.transport.ctx, point, current, total, requested, &self.cancelled) == 0;
    }
};
fn cancelAgreed(context: ?*anyopaque) callconv(.c) bool {
    const agreement: *const Agreement = @ptrCast(@alignCast(context.?));
    return agreement.cancelled or agreement.failed;
}
fn progressAgreed(context: ?*anyopaque, event: [*c]const u8, current: c_int, total: c_int) callconv(.c) void {
    const agreement: *Agreement = @ptrCast(@alignCast(context.?));
    if (agreement.progress) |callback| callback(agreement.progress_userdata, event, current, total);
    if (!agreement.cancelled) agreement.checkpoint(if (eventMatches(event, "prefill_chunk")) 1 else 2, current, total);
}
fn displayAgreed(context: ?*anyopaque, event: [*c]const u8, current: c_int, total: c_int) callconv(.c) void {
    const agreement: *Agreement = @ptrCast(@alignCast(context.?));
    if (agreement.display) |callback| callback(agreement.display_userdata, event, current, total);
    if (!agreement.cancelled) agreement.checkpoint(2, current, total);
}
