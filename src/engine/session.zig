//! Ordinary single-Metal session routing over actual component owners. The C
//! shell captures a synchronous view; validation, callback scopes, image
//! publication and payload dispatch live here. Numerical model owners remain
//! independent. Legacy/reference/distributed admission stays outside this path.
const std = @import("std");
const c = @import("gguf_c");
const Frontier = @import("session_frontier.zig").Frontier;
const Control = @import("session_control.zig");
const Qwen = @import("qwen4/session.zig");
const QwenPayload = @import("qwen4/payload.zig");
const Glm = @import("glm53/session.zig");
const GlmPayload = @import("glm53/payload.zig");
const View = c.ds4_session_view;

fn fail(errors: []u8, message: [:0]const u8) c_int {
    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s", message.ptr);
    return 1;
}
fn pointer(errors: []u8) [*c]u8 {
    return if (errors.len == 0) null else errors.ptr;
}
fn images(raw: [*c]const c.ds4_vision_span, count: usize) []const c.ds4_vision_span {
    return if (count == 0) &.{} else raw[0..count];
}
fn commonSync(user: ?*anyopaque, prompt: [*c]const c.ds4_tokens, errors: [*c]u8, capacity: usize) callconv(.c) c_int {
    const view: *View = @ptrCast(@alignCast(user orelse return 1));
    return sync(view, if (prompt == null) null else @ptrCast(prompt), if (capacity == 0) &.{} else errors[0..capacity]);
}
pub fn qwenRuntime(view: *View) c.ds4_qwen4_runtime {
    return .{
        .state = view.qwen,
        .frontier = view.frontier,
        .pending = view.pending,
        .control = view.control,
        .model = &view.core.*.model,
        .weights = &view.core.*.weights,
        .exact_sampling = view.core.*.dspark_exact_sampling,
        .sync = commonSync,
        .user = view,
    };
}
pub fn glmRuntime(view: *View) c.ds4_glm_runtime {
    return .{
        .state = view.glm,
        .frontier = view.frontier,
        .pending = view.pending,
        .control = view.control,
        .core = view.core,
        .ablate_mask = view.ablate_mask,
        .sync = commonSync,
        .user = view,
    };
}
fn flintProgress(user: ?*anyopaque, event: [*c]const u8, current: c_int, total: c_int) callconv(.c) void {
    const view: *View = @ptrCast(@alignCast(user orelse return));
    const control = view.control;
    if (control.*.progress) |callback| callback(control.*.progress_ud, event, current, total);
    if (control.*.display_progress) |callback| callback(control.*.display_progress_ud, event, current, total);
}
pub fn flintCommit(view: *View) void {
    const count: usize = @intCast(view.core.*.vocab.n_vocab);
    @memcpy(view.frontier.*.logits[0..count], c.flint_session_logits(view.flint.*)[0..count]);
    view.frontier.*.checkpoint_valid = true;
}

pub fn invalidate(view: *View) void {
    if (view.kind == c.DS4_SESSION_FLINT) {
        c.flint_session_reset(view.flint.*);
        view.frontier.*.checkpoint.len = 0;
        view.frontier.*.checkpoint_valid = false;
        return;
    }
    if (view.kind == c.DS4_SESSION_DS41) {
        c.ds4_native_session_invalidate41(view.frontier, if (view.ds41.*.ready) &view.ds41.*.graph else null);
    } else {
        (Frontier{ .state = view.frontier }).invalidate();
    }
    Control.clearCapture(view.control);
    if (view.kind == c.DS4_SESSION_GLM53) Glm.resetDense(view.glm, view.pending) else if (view.pending != null) {
        // The previous common GLM reset also cleared this shared pending owner
        // for Qwen/DS41; keep that publication independent of model allocation.
        view.pending.*.have = 0;
        view.pending.*.have2 = false;
        view.pending.*.rollback_valid = false;
        view.pending.*.min_pos = 0;
    }
}

pub fn sync(view: *View, prompt: ?*const c.ds4_tokens, errors: []u8) c_int {
    const frontier = view.frontier;
    if (view.kind == c.DS4_SESSION_DS41) return c.ds4_native_session_request_sync41(frontier, view.ds41, view.core, view.transport, view.control, view.config41, view.memory41, prompt, view.exact41, pointer(errors), errors.len);
    if (view.kind == c.DS4_SESSION_FLINT) {
        const input = prompt orelse return fail(errors, "invalid Flint prompt length");
        if (input.len < 1 or input.len > frontier.*.ctx_size) return fail(errors, "invalid Flint prompt length");
        const result = c.flint_session_sync(view.flint.*, input, flintProgress, view, view.control.*.cancel, view.control.*.cancel_ud, pointer(errors), errors.len);
        if (result != 0) {
            invalidate(view);
            return result;
        }
        c.ds4_tokens_copy(&frontier.*.checkpoint, input);
        frontier.*.overthink_prompt_len = input.len;
        flintCommit(view);
        return 0;
    }
    view.control.*.dspark_rewind_end = 0;
    const owner = Frontier{ .state = frontier };
    if (frontier.*.checkpoint_valid and !owner.visionPrefixMatches(images(frontier.*.sync_images, frontier.*.sync_image_count))) invalidate(view);
    Control.beginRequest(view.control);
    const input = prompt orelse return fail(errors, "missing session or prompt");
    if (input.len <= 0) return fail(errors, "empty prompt");
    if (input.len >= frontier.*.ctx_size) {
        if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "prompt length %d exceeds context %d (one token of generation room is required)", input.len, frontier.*.ctx_size);
        return 1;
    }
    if (Control.cancelled(view.control)) {
        _ = fail(errors, "interrupted");
        return c.DS4_SESSION_SYNC_INTERRUPTED;
    }
    const result = switch (view.kind) {
        c.DS4_SESSION_QWEN => result: {
            var runtime = qwenRuntime(view);
            break :result Qwen.sync(&runtime, input, errors);
        },
        c.DS4_SESSION_GLM53 => result: {
            var runtime = glmRuntime(view);
            break :result Glm.sync(&runtime, input, errors);
        },
        else => return fail(errors, "unsupported native session model"),
    };
    if (result == 0) {
        c.ds4_native_session_prefill_logit_dump(frontier.*.logits, @intCast(view.core.*.vocab.n_vocab));
        if (!owner.rememberImages()) {
            invalidate(view);
            return fail(errors, "unable to retain image prompt identity");
        }
        frontier.*.overthink_prompt_len = input.len;
    }
    return result;
}

pub fn syncMultimodal(view: *View, prompt: ?*const c.ds4_tokens, raw: [*c]const c.ds4_vision_span, count: usize, errors: []u8) c_int {
    if (view.kind == c.DS4_SESSION_DS41) return c.ds4_native_session_multimodal_sync41(view.frontier, view.ds41, view.core, view.transport, view.control, view.config41, view.memory41, prompt, raw, count, view.exact41, pointer(errors), errors.len);
    if (view.kind == c.DS4_SESSION_FLINT) {
        if (count != 0) return fail(errors, "Flint vision is not implemented");
        return sync(view, prompt, errors);
    }
    const input = prompt orelse return fail(errors, "invalid multimodal prompt");
    if (count != 0 and raw == null) return fail(errors, "invalid multimodal prompt");
    if (count != 0 and !view.core.*.vision_ready) return fail(errors, "vision encoder is not loaded");
    var previous_end: u64 = 0;
    for (images(raw, count)) |span| {
        const end = @as(u64, span.token_start) + span.embedding.token_count;
        if (span.embedding.data == null or span.embedding.token_count == 0 or span.token_start < previous_end or end > @as(u64, @bitCast(@as(i64, input.len)))) return fail(errors, "invalid or overlapping image token span");
        for (span.token_start..end) |token| if (input.v[token] != view.core.*.vision_image_token) return fail(errors, "image span does not cover image placeholder tokens");
        previous_end = end;
    }
    view.frontier.*.sync_images = raw;
    view.frontier.*.sync_image_count = count;
    defer {
        view.frontier.*.sync_images = null;
        view.frontier.*.sync_image_count = 0;
    }
    return sync(view, input, errors);
}

pub fn eval(view: *View, token: c_int, errors: []u8) c_int {
    const frontier = view.frontier;
    if (view.kind == c.DS4_SESSION_DS41) return c.ds4_native_session_request_eval41(frontier, view.ds41, view.core, view.transport, view.control, null, null, view.config41, view.memory41, token, view.exact41, pointer(errors), errors.len);
    if (view.kind == c.DS4_SESSION_FLINT) {
        if (!frontier.*.checkpoint_valid or token < 0 or token >= view.core.*.vocab.n_vocab or frontier.*.checkpoint.len >= frontier.*.ctx_size) return fail(errors, "invalid Flint decode frontier/token");
        const result = c.flint_session_eval(view.flint.*, token, pointer(errors), errors.len);
        if (result != 0) {
            invalidate(view);
            return result;
        }
        c.ds4_tokens_push(&frontier.*.checkpoint, token);
        flintCommit(view);
        return 0;
    }
    if (!frontier.*.checkpoint_valid) return fail(errors, "decode requires a synchronized checkpoint");
    return switch (view.kind) {
        c.DS4_SESSION_QWEN => result: {
            var runtime = qwenRuntime(view);
            break :result Qwen.eval(&runtime, token, errors);
        },
        c.DS4_SESSION_GLM53 => result: {
            var runtime = glmRuntime(view);
            break :result Glm.eval(&runtime, token, errors);
        },
        else => fail(errors, "unsupported native session model"),
    };
}

pub fn payloadBytes(view: *View) u64 {
    const frontier = view.frontier;
    if (view.kind == c.DS4_SESSION_DS41) return c.ds4_native_checkpoint_bytes41(frontier, &view.ds41.*.graph, view.config41, view.ds41.*.ready);
    if (view.kind == c.DS4_SESSION_FLINT or !frontier.*.checkpoint_valid or frontier.*.approximate_suffix) return 0;
    if (view.kind == c.DS4_SESSION_QWEN) {
        const state = view.qwen;
        if (!state.*.ready or state.*.rewound or state.*.graph.pos != frontier.*.checkpoint.len) return 0;
        const rows: u32 = @intCast(frontier.*.checkpoint.len);
        return (@as(u64, c.DS4_SESSION_PAYLOAD_U32_FIELDS) + rows + state.*.graph.config.shape.n_vocab) * @sizeOf(u32) +
            QwenPayload.tensorBytes(&state.*.graph.config, rows, @min(state.*.graph.mtp_pos, rows));
    }
    if (view.kind == c.DS4_SESSION_GLM53 and view.glm.*.ready) return GlmPayload.bytes(&view.glm.*.graph, frontier, view.glm.*.dense_cache_len);
    return 0;
}

pub fn savePayload(view: *View, file: ?*c.FILE, errors: []u8) c_int {
    if (view.kind == c.DS4_SESSION_FLINT) return fail(errors, "Flint GPU-state persistence/suffix reuse is not implemented");
    if (view.kind == c.DS4_SESSION_DS41) return c.ds4_native_checkpoint_save_payload41(view.frontier, &view.ds41.*.graph, view.config41, view.ds41.*.ready, file, pointer(errors), errors.len);
    if (view.frontier.*.approximate_suffix) return fail(errors, "private approximate suffix state cannot be exported as canonical KV");
    const stream = file orelse return fail(errors, "session has no valid checkpoint to save");
    if (!view.frontier.*.checkpoint_valid) return fail(errors, "session has no valid checkpoint to save");
    if (view.kind == c.DS4_SESSION_QWEN) return QwenPayload.save(&view.qwen.*.graph, view.frontier, view.qwen.*.ready, view.qwen.*.rewound, stream, errors);
    if (view.kind == c.DS4_SESSION_GLM53) {
        if (!view.glm.*.ready) return fail(errors, "GLM graph is not ready for snapshot");
        GlmPayload.save(&view.glm.*.graph, view.frontier, view.glm.*.dense_cache_len, stream, errors) catch return 1;
        return 0;
    }
    return fail(errors, "unsupported native session model");
}

pub fn loadPayload(view: *View, file: ?*c.FILE, bytes: u64, errors: []u8) c_int {
    if (view.kind == c.DS4_SESSION_FLINT) return fail(errors, "Flint GPU-state persistence/suffix reuse is not implemented");
    if (view.kind == c.DS4_SESSION_DS41) return c.ds4_native_checkpoint_load_payload41(view.frontier, &view.ds41.*.graph, view.config41, view.ds41.*.ready, file, bytes, pointer(errors), errors.len);
    const stream = file orelse return fail(errors, "invalid session payload load");
    var remaining = bytes;
    var header: [c.DS4_SESSION_PAYLOAD_U32_FIELDS]u32 = undefined;
    for (&header) |*value| if (c.ds4_native_payload_read_u32(stream, value, &remaining, pointer(errors), errors.len) != 0) return 1;
    if (header[0] != c.DS4_SESSION_PAYLOAD_MAGIC or header[1] != c.DS4_SESSION_PAYLOAD_VERSION) return fail(errors, "unsupported session payload version");
    const result = switch (view.kind) {
        c.DS4_SESSION_QWEN => result: {
            var mutated = false;
            const result = QwenPayload.load(&view.qwen.*.graph, view.frontier, view.qwen.*.ready, stream, &header, &remaining, &mutated, errors);
            if (mutated) {
                view.pending.*.have = 0;
                view.pending.*.have2 = false;
            }
            if (result == 0) view.qwen.*.rewound = false;
            break :result result;
        },
        c.DS4_SESSION_GLM53 => result: {
            if (!view.glm.*.ready) return fail(errors, "GLM graph is not ready for KV restore");
            var dense = view.glm.*.dense_cache_len;
            const result: c_int = if (GlmPayload.load(&view.glm.*.graph, view.frontier, &dense, stream, &header, remaining, errors)) 0 else |_| 1;
            if (result == 0 or !view.frontier.*.checkpoint_valid) {
                Glm.resetDense(view.glm, view.pending);
                if (result == 0) view.glm.*.dense_cache_len = dense;
            }
            break :result result;
        },
        else => return fail(errors, "unsupported native session model"),
    };
    if (result == 0) {
        view.frontier.*.overthink_prompt_len = view.frontier.*.checkpoint.len;
        (Frontier{ .state = view.frontier }).clearApproximation();
    }
    return result;
}
