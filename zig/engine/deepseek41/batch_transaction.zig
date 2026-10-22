//! Public DS41 decode and mixed-prefill batches. Each borrowed row names actual
//! Session owners; only graph transients are shared. Admission, TP framing and
//! checkpoint publication stay here, while SessionBatch owns numerical work.
const std = @import("std");
const c = @import("gguf_c");
const Coordinator = @import("coordinator.zig").Coordinator;
const SessionBatch = @import("session_batch.zig").SessionBatch;
const Frontier = @import("frontier.zig").Frontier;
const View = c.ds4_session_batch_view;
const Graph = c.ds41_gpu_graph;
const max_rows = c.DS4_TP_BATCH_MAX_ROWS;

pub const Result = struct {
    status: c_int,
    native_batch: bool = false,
    mixed_batch: bool = false,
};
fn fail(errors: []u8, text: []const u8) c_int {
    if (errors.len != 0) {
        const count = @min(errors.len - 1, text.len);
        @memcpy(errors[0..count], text[0..count]);
        errors[count] = 0;
    }
    return 1;
}
fn state(view: View) *c.ds4_session_frontier {
    return @ptrCast(view.frontier);
}
fn device(view: View) *c.ds4_deepseek41_session_device {
    return @ptrCast(view.device);
}
fn control(view: View) *c.ds4_session_control {
    return @ptrCast(view.control);
}
fn cancelled(view: View) bool {
    const request = control(view);
    return if (request.cancel) |check| check(request.cancel_ud) else false;
}

pub const Batch = struct {
    // This is an ordinary borrowed owner wrapper. Copies below replace only
    // actual owner pointers; no graph/frontier/control record is snapshotted.
    session: Coordinator,
    engine_identity: ?*c.ds4_engine,

    pub fn row(self: *const Batch, view: View) Coordinator {
        var owner = self.session;
        owner.runtime.state = state(view);
        owner.runtime.prefill.executor.graph = &device(view).graph;
        owner.runtime.ready = device(view).ready;
        owner.control = control(view);
        return owner;
    }
    fn eval(self: *const Batch, view: View, errors: []u8) c_int {
        if (view.session == null) return fail(errors, "decode requires a synchronized checkpoint");
        const owner = self.row(view);
        return owner.eval(view.session, view.distributed, view.token, errors);
    }
    pub fn invalidate(self: *const Batch, views: []const View, prefill: ?View) void {
        if (prefill) |view| self.row(view).invalidate();
        for (views) |view| self.row(view).invalidate();
    }
    fn vocabulary(self: *const Batch) u32 {
        return self.session.runtime.prefill.executor.ops.config.shape.n_vocab;
    }

    /// The network crossover deliberately differs from single-host execution.
    /// Diagnostic profiles and support drafts keep ordered one-row execution.
    fn supported(self: *const Batch, views: []const View) bool {
        const core = self.session.core;
        const transport = self.session.transport;
        const tp_batch = c.getenv("DS4_METAL_TP_SESSION_BATCH");
        if (views.len < 2 or views.len > max_rows or core.backend != c.DS4_BACKEND_METAL or
            core.support_kind != c.DS4_SUPPORT_NONE or
            (transport.active and tp_batch != null and c.strcmp(tp_batch, "0") == 0) or
            c.getenv("DS4_METAL_GRAPH_DUMP_PREFIX") != null or
            c.getenv("DS4_METAL_DECODE_STAGE_PROFILE") != null or
            c.getenv("DS4_METAL_DISABLE_V41_SESSION_BATCH") != null) return false;
        var graphs: [max_rows]*Graph = undefined;
        for (views, 0..) |view, index| {
            if (view.session == null or view.engine != self.engine_identity) return false;
            const graph = &device(view).graph;
            const frontier = state(view);
            if (!device(view).ready or !frontier.checkpoint_valid or view.distributed != null or
                graph.streaming or graph.imatrix != null or graph.quality or !graph.valid or
                graph.image_count != 0 or graph.pos != @as(u32, @intCast(frontier.checkpoint.len))) return false;
            // Smaller TP batches lose to ordered decode. Single-host execution
            // starts at two, while the Metal network crossover starts at three.
            if (graph.tp_world == 2 and views.len < 3) return false;
            graphs[index] = graph;
        }
        return SessionBatch.workspace(graphs[0..views.len]) != null;
    }
    fn mixedSupported(self: *const Batch, views: []const View, prefill: View, prompt: *const c.ds4_tokens) bool {
        const count = prompt.len - state(prefill).checkpoint.len;
        if (views.len < 1 or count < 1 or views.len >= max_rows) return false;
        const rows: usize = @intCast(count);
        if (rows > max_rows - views.len or cancelled(prefill)) return false;
        var combined: [max_rows]View = undefined;
        for (combined[0..rows], 0..) |*view, index| {
            view.* = prefill;
            view.token = prompt.v[@as(usize, @intCast(state(prefill).checkpoint.len)) + index];
        }
        for (views, 0..) |view, index| {
            if (cancelled(view)) return false;
            combined[rows + index] = view;
        }
        return self.supported(combined[0 .. rows + views.len]);
    }
    fn send(self: *const Batch, views: []const View, prefill: ?View, prompt: ?*const c.ds4_tokens) bool {
        // Preserve the original calloc failure and zero reserved wire fields.
        const allocation = c.calloc(views.len, @sizeOf(c.ds4_tp_batch_item)) orelse return false;
        defer c.free(allocation);
        const wire: [*]c.ds4_tp_batch_item = @ptrCast(@alignCast(allocation));
        for (views, 0..) |view, index| {
            const id = control(view).tp_session_id;
            if (id == 0) return false;
            wire[index].session_id = id;
            wire[index].token = view.token;
        }
        if (prefill) |view| return c.ds4_tp_send_mixed_batch(self.session.transport.ctx, control(view).tp_session_id, prompt.?.v, @intCast(prompt.?.len), wire, @intCast(views.len)) != 0;
        return c.ds4_tp_send_eval_batch(self.session.transport.ctx, wire, @intCast(views.len)) != 0;
    }
    fn receive(self: *const Batch, views: []const View, prefill: ?View, errors: []u8) bool {
        const transport = self.session.transport;
        if (!transport.active or transport.rank != 0 or !transport.vocab_split) return true;
        const half = self.vocabulary() / 2;
        if (prefill) |view| {
            if (c.ds4_tp_recv_logits_half(transport.ctx, state(view).logits + half, half) == 0) {
                _ = fail(errors, "tp: worker mixed-prefill logits missing");
                return false;
            }
        }
        for (views, 0..) |view, index| {
            if (c.ds4_tp_recv_logits_half(transport.ctx, state(view).logits + half, half) == 0) {
                if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "tp: worker batch logits missing for item %d", @as(c_int, @intCast(index)));
                return false;
            }
        }
        return true;
    }

    fn run(self: *const Batch, views: []const View, prefill: ?View, prompt: ?*const c.ds4_tokens, errors: []u8) c_int {
        const transport = self.session.transport;
        const mirror = transport.active and transport.rank == 0;
        if (mirror and !self.send(views, prefill, prompt)) {
            _ = fail(errors, "tp: worker decode batch send failed");
            self.invalidate(views, prefill);
            return 1;
        }
        if (transport.active) c.ds4_gpu_tp_set_session_batch_mode(1);
        var ok = c.ds4_gpu_begin_commands() != 0;
        const prefix: usize = if (prefill) |view| @intCast(prompt.?.len - state(view).checkpoint.len) else 0;
        var graphs: [max_rows]*Graph = undefined;
        var tokens: [max_rows]c_int = undefined;
        if (prefill) |view| {
            for (0..prefix) |index| {
                graphs[index] = &device(view).graph;
                tokens[index] = prompt.?.v[@as(usize, @intCast(state(view).checkpoint.len)) + index];
            }
        }
        for (views, 0..) |view, index| {
            graphs[prefix + index] = &device(view).graph;
            tokens[prefix + index] = view.token;
        }
        const count = prefix + views.len;
        const batch = SessionBatch{ .executor = self.session.runtime.prefill.executor };
        ok = ok and batch.run(graphs[0..count], tokens[0..count], @intCast(prefix));
        // Finish partial command buffers BEFORE invalidating failed sessions.
        if (c.ds4_gpu_end_commands() == 0) ok = false;
        if (transport.active) c.ds4_gpu_tp_set_session_batch_mode(0);
        if (ok and transport.active and c.ds4_gpu_tp_failed() != 0) {
            _ = fail(errors, "tp: batch gate transport failed");
            ok = false;
        }
        const logit_bytes = @as(u64, self.vocabulary()) * @sizeOf(f32);
        if (ok) if (prefill) |view| {
            ok = c.ds4_gpu_tensor_read(device(view).graph.logits, 0, state(view).logits, logit_bytes) != 0;
        };
        for (views) |view| {
            if (!ok) break;
            ok = c.ds4_gpu_tensor_read(device(view).graph.logits, 0, state(view).logits, logit_bytes) != 0;
        }
        if (ok and mirror) {
            const id = if (prefill) |view| control(view).tp_session_id else 0;
            ok = c.ds4_tp_wait_command_ack(transport.ctx, id, "decode batch", errors.ptr, errors.len) != 0 and
                self.receive(views, prefill, errors);
        }
        if (!ok) {
            self.invalidate(views, prefill);
            if (errors.len != 0 and errors[0] == 0)
                _ = c.snprintf(errors.ptr, errors.len, "%s batched decode failed", c.ds4_backend_name(self.session.core.backend));
            return 1;
        }
        if (prefill) |view| {
            c.ds4_tokens_copy(&state(view).checkpoint, prompt.?);
            state(view).checkpoint_valid = true;
            state(view).mtp_draft_valid = false;
        }
        for (views) |view| {
            c.ds4_tokens_push(&state(view).checkpoint, view.token);
            state(view).checkpoint_valid = true;
            state(view).mtp_draft_valid = false;
        }
        if (prefill) |view| {
            const request = control(view);
            if (request.progress) |report| report(request.progress_ud, "prefill_chunk", prompt.?.len, prompt.?.len);
        }
        if (c.getenv("DS4_METAL_SESSION_BATCH_LOG") != null)
            _ = c.fprintf(c.__stderrp, "ds4: %s session batch rows=%d prefill_rows=%u family=deepseek native_ds41=1 native_glm53=0 native_qwen4=0 native_shared=0 native_qkv=0\n", c.ds4_backend_name(self.session.core.backend), @as(c_int, @intCast(views.len)), @as(u32, @intCast(prefix)));
        return 0;
    }

    pub fn decode(self: *const Batch, views: []const View, errors: []u8) Result {
        if (views.len == 0) return .{ .status = fail(errors, "empty decode batch") };
        // A one-row request intentionally bypasses batch token/context checks.
        if (views.len == 1) return .{ .status = self.eval(views[0], errors) };
        if (views[0].session == null or views[0].engine == null) return .{ .status = fail(errors, "decode batch has no session") };
        for (views, 0..) |view, index| {
            const item: c_int = @intCast(index);
            if (view.session == null or view.engine != self.engine_identity) {
                if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "decode batch item %d belongs to a different engine", item);
                return .{ .status = 1 };
            }
            if (view.token < 0 or view.token >= @as(c_int, @intCast(self.vocabulary()))) {
                if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "decode batch item %d has an invalid token", item);
                return .{ .status = 1 };
            }
            for (views[0..index], 0..) |previous, before| {
                if (previous.session == view.session) {
                    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "decode batch repeats session at items %d and %d", @as(c_int, @intCast(before)), item);
                    return .{ .status = 1 };
                }
            }
            if (state(view).checkpoint.len >= state(view).ctx_size) {
                if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "decode batch item %d reached its context limit", item);
                return .{ .status = 1 };
            }
        }
        if (self.supported(views)) {
            const status = self.run(views, null, null, errors);
            return .{ .status = status, .native_batch = status == 0 };
        }
        // Preserve logical all-or-nothing behavior on serialized execution:
        // earlier rows may have advanced, so every member must rebuild on failure.
        for (views) |view| {
            if (self.eval(view, errors) != 0) {
                self.invalidate(views, null);
                return .{ .status = 1 };
            }
        }
        return .{ .status = 0 };
    }

    pub fn mixed(self: *const Batch, views: []const View, prefill: ?View, prompt: ?*const c.ds4_tokens, errors: []u8) Result {
        if (views.len == 0 or prefill == null or prompt == null or prefill.?.session == null or prefill.?.engine == null)
            return .{ .status = fail(errors, "invalid mixed model batch") };
        const view = prefill.?;
        const input = prompt.?;
        const frontier = state(view);
        if (!frontier.checkpoint_valid or input.len <= frontier.checkpoint.len or input.len > frontier.ctx_size)
            return .{ .status = fail(errors, "mixed prefill must extend a valid session checkpoint") };
        const tokens: []const c_int = if (input.len <= 0) &.{} else input.v[0..@intCast(input.len)];
        if (!(Frontier{ .state = frontier }).startsWithCheckpoint(tokens))
            return .{ .status = fail(errors, "mixed prefill must extend a valid session checkpoint") };
        for (views, 0..) |decode_view, index| {
            const item: c_int = @intCast(index);
            if (decode_view.session == null or decode_view.session == view.session or decode_view.engine != view.engine or
                !state(decode_view).checkpoint_valid or state(decode_view).checkpoint.len >= state(decode_view).ctx_size or
                decode_view.token < 0 or decode_view.token >= @as(c_int, @intCast(self.vocabulary())))
            {
                if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "invalid mixed decode item %d", item);
                return .{ .status = 1 };
            }
            for (views[0..index], 0..) |previous, before| {
                if (previous.session == decode_view.session) {
                    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "mixed decode repeats session at items %d and %d", @as(c_int, @intCast(before)), item);
                    return .{ .status = 1 };
                }
            }
        }
        if (self.mixedSupported(views, view, input)) {
            const status = self.run(views, view, input, errors);
            return .{ .status = status, .native_batch = status == 0, .mixed_batch = status == 0 };
        }
        const owner = self.row(view);
        const status = owner.sync(input, errors);
        if (status != 0) return .{ .status = status };
        const result = self.decode(views, errors);
        if (result.status != 0) owner.invalidate();
        return result;
    }
};
