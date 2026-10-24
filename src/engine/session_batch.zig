//! Common Qwen/GLM/Flint batch transactions over the same actual SessionView.
//! Numerical batching stays with its model; this owner validates membership,
//! chooses the existing path and publishes or invalidates every frontier.
//! DS41 already has its complete native batch coordinator and remains there.
const std = @import("std");
const c = @import("gguf_c");
const Session = @import("session.zig");
const Qwen = @import("qwen4/batch.zig");
const QwenSpec = @import("qwen4/batch_session.zig");
const QwenSession = @import("qwen4/session.zig");
const Glm = @import("glm53/session.zig");
const GlmBatch = @import("glm53/session_batch.zig");
const GlmExecutor = @import("glm53/executor.zig");
const GlmPlanning = @import("glm53/planning.zig");
const View = c.ds4_session_view;
const Pool = c.ds4_session_resource_pool;
const Policy = c.ds4_qwen4_batch_policy;
const maximum = 64;

fn fail(errors: []u8, message: [:0]const u8) c_int {
    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s", message.ptr);
    return 1;
}
fn indexed(errors: []u8, format: [:0]const u8, index: usize) c_int {
    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, format.ptr, @as(c_int, @intCast(index)));
    return 1;
}
fn invalidate(views: []View) void {
    for (views) |*view| Session.invalidate(view);
}
fn validate(views: []const View, tokens: []const c_int, errors: []u8) c_int {
    if (views.len == 0 or tokens.len != views.len) return fail(errors, "empty decode batch");
    const core = views[0].core;
    if (core == null or views[0].frontier == null) return fail(errors, "decode batch has no session");
    for (views, tokens, 0..) |view, token, index| {
        if (view.core != core or view.frontier == null) return indexed(errors, "decode batch item %d belongs to a different engine", index);
        if (token < 0 or token >= core.*.vocab.n_vocab) return indexed(errors, "decode batch item %d has an invalid token", index);
        for (views[0..index], 0..) |previous, earlier| if (previous.frontier == view.frontier) {
            if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "decode batch repeats session at items %d and %d", @as(c_int, @intCast(earlier)), @as(c_int, @intCast(index)));
            return 1;
        };
        if (view.frontier.*.checkpoint.len >= view.frontier.*.ctx_size) return indexed(errors, "decode batch item %d reached its context limit", index);
    }
    return 0;
}
fn penalty(view: *const View) bool {
    return view.core.*.overthink.penalty > 0 and view.core.*.overthink.count > 0;
}
fn optimized(views: []const View) bool {
    if (views.len < 2 or views[0].core.*.support_kind != c.DS4_SUPPORT_NONE or
        c.getenv("DS4_METAL_GRAPH_DUMP_PREFIX") != null or c.getenv("DS4_METAL_DECODE_STAGE_PROFILE") != null) return false;
    for (views) |view| if (!view.frontier.*.checkpoint_valid) return false;
    return true;
}
fn qwenMembers(views: []const View, tokens: []const c_int, out: []c.ds4_qwen4_batch_member) bool {
    for (views, tokens, out, 0..) |view, token, *member, index| {
        if (view.kind != c.DS4_SESSION_QWEN or !view.qwen.*.ready) return false;
        member.* = .{ .graph = &view.qwen.*.graph, .frontier = view.frontier, .slot = view.qwen.*.slot, .tokens = .{ token, -1 }, .n = 1, .row0 = @intCast(index) };
    }
    return true;
}
fn glmRows(views: []const View, tokens: []const c_int, out: []c.ds4_glm_session_row) bool {
    if (views[0].kind != c.DS4_SESSION_GLM53 or views[0].core.*.glm_mtp or c.getenv("DS4_GLM_MTP_PROBE") != null) return false;
    for (views, tokens, out) |view, token, *row| {
        if (view.kind != c.DS4_SESSION_GLM53 or !view.glm.*.ready or view.glm.*.graph.ssd_streaming) return false;
        row.* = .{ .graph = &view.glm.*.graph, .position = @intCast(view.frontier.*.checkpoint.len), .token = token };
    }
    return true;
}
fn glmDenseBatch(rows: []const c.ds4_glm_session_row) bool {
    const first = rows[0].graph;
    if (first.*.config.shape.n_rot != 0 or !first.*.has_token_embd or !first.*.has_output_head or first.*.compact_cache_cap == 0 or first.*.batch_qk_low == null or rows.len > first.*.ctx_cap or rows.len > first.*.config.prefill_chunk) return false;
    for (rows) |row| {
        const graph = row.graph;
        if (!graph.*.glm53 or graph.*.ssd_streaming or graph.*.placement != null or graph.*.full_kv_cache or
            !GlmBatch.sameSteering(first, graph) or
            graph.*.compact_cache_cap != first.*.compact_cache_cap or graph.*.layer_start != first.*.layer_start or
            graph.*.layer_end != first.*.layer_end or graph.*.tp_world != first.*.tp_world or graph.*.tp_rank != first.*.tp_rank or
            row.position >= GlmPlanning.denseLimit(graph)) return false;
    }
    return true;
}
fn encoded(views: []View, tokens: []const c_int, pool: *Pool, errors: []u8) ?c_int {
    if (!optimized(views)) return null;
    const core = views[0].core;
    var members: [maximum]c.ds4_qwen4_batch_member = undefined;
    const qwen = views.len <= maximum and qwenMembers(views, tokens, members[0..views.len]) and Qwen.supported(pool.qwen4_shared_workspace, members[0..views.len], false);
    var local_rows: [maximum]c.ds4_glm_session_row = undefined;
    const rows = if (views.len <= maximum) local_rows[0..views.len] else std.heap.c_allocator.alloc(c.ds4_glm_session_row, views.len) catch return fail(errors, "unable to allocate decode batch views");
    defer if (views.len > maximum) std.heap.c_allocator.free(rows);
    const glm = !qwen and glmRows(views, tokens, rows);
    if (!qwen and !glm) return null;
    const dense = glm and glmDenseBatch(rows);
    var ok = c.ds4_gpu_begin_commands() != 0;
    if (ok and qwen) {
        ok = Qwen.encode(members[0..views.len], pool.qwen4_shared_workspace, &pool.qwen4_pool, &core.*.model, &core.*.weights);
    } else if (ok and dense) {
        ok = GlmBatch.encode(&core.*.model, &core.*.weights, rows, views[0].ablate_mask);
    } else if (ok) {
        // Preserve the existing one-command-epoch scalar fallback beyond the
        // dense cross-session kernel's range, rather than serial GPU waits.
        for (views, tokens) |view, token| {
            if (!GlmExecutor.forward(&view.glm.*.graph, &core.*.model, &core.*.weights, token, null, @intCast(view.frontier.*.checkpoint.len), null, view.frontier.*.logits, true, view.ablate_mask)) {
                ok = false;
                break;
            }
        }
    }
    if (c.ds4_gpu_end_commands() == 0) ok = false;
    const bytes = @as(u64, @intCast(core.*.vocab.n_vocab)) * @sizeOf(f32);
    for (views, 0..) |*view, index| {
        if (!ok) break;
        ok = if (qwen)
            c.ds4_gpu_tensor_read(pool.qwen4_shared_workspace.*.batch_logits, index * bytes, view.frontier.*.logits, bytes) != 0
        else
            c.ds4_gpu_tensor_read(view.glm.*.graph.logits, 0, view.frontier.*.logits, bytes) != 0;
        if (ok and qwen) {
            view.qwen.*.graph.pos += 1;
            view.qwen.*.rewound = false;
            view.pending.*.have = 0;
            view.pending.*.have2 = false;
        }
    }
    if (!ok) {
        invalidate(views);
        if (errors.len != 0 and errors[0] == 0) _ = fail(errors, "metal batched decode failed");
        return 1;
    }
    for (views, tokens) |*view, token| {
        c.ds4_tokens_push(&view.frontier.*.checkpoint, token);
        view.frontier.*.checkpoint_valid = true;
        view.frontier.*.mtp_draft_valid = false;
    }
    if (c.getenv("DS4_METAL_SESSION_BATCH_LOG") != null)
        _ = c.fprintf(c.__stderrp, "ds4: metal session batch rows=%d prefill_rows=0 family=%s native_ds41=0 native_glm53=%d native_qwen4=%d native_shared=0 native_qkv=0\n", @as(c_int, @intCast(views.len)), @as([*c]const u8, if (qwen) "qwen" else "glm"), @as(c_int, @intFromBool(dense)), @as(c_int, @intFromBool(qwen)));
    return 0;
}

pub fn eval(views: []View, tokens: []const c_int, pool: *Pool, errors: []u8) c_int {
    if (views.len == 1) {
        if (views[0].frontier == null or views[0].core == null) return fail(errors, "decode requires a synchronized checkpoint");
        return Session.eval(&views[0], tokens[0], errors);
    }
    if (validate(views, tokens, errors) != 0) return 1;
    if (encoded(views, tokens, pool, errors)) |result| return result;
    for (views, tokens) |*view, token| if (Session.eval(view, token, errors) != 0) {
        invalidate(views);
        return 1;
    };
    return 0;
}

fn scalarSpeculate(view: *View, token: c_int, accepted: []c_int, errors: []u8) c_int {
    if (view.kind == c.DS4_SESSION_FLINT) {
        if (!view.frontier.*.checkpoint_valid or accepted.len == 0 or token < 0 or token >= view.core.*.vocab.n_vocab) {
            _ = fail(errors, "invalid Flint packet frontier/token");
            return -1;
        }
        const count = c.flint_session_packet(view.flint.*, token, @intCast(accepted.len), 0, 0, 1, 0, null, accepted.ptr, if (errors.len == 0) null else errors.ptr, errors.len);
        if (count < 1 or count > accepted.len) return -1;
        for (accepted[0..@intCast(count)]) |input| c.ds4_tokens_push(&view.frontier.*.checkpoint, input);
        Session.flintCommit(view);
        return count;
    }
    if (!view.frontier.*.checkpoint_valid) {
        _ = fail(errors, "speculative decode requires a synchronized checkpoint");
        return -1;
    }
    if (view.core.*.glm_mtp and !penalty(view)) {
        if (view.kind == c.DS4_SESSION_QWEN and view.qwen.*.ready and view.qwen.*.graph.config.shape.n_nextn_predict != 0) {
            var runtime = Session.qwenRuntime(view);
            return QwenSession.speculate(&runtime, token, 0, 0, 0, 0, null, false, accepted, errors);
        }
        if (view.kind == c.DS4_SESSION_GLM53 and view.glm.*.ready and view.glm.*.graph.config.shape.n_nextn_predict != 0 and c.getenv("DS4_GLM_MTP_PROBE") == null) {
            var runtime = Session.glmRuntime(view);
            return Glm.speculate(&runtime, token, -1, 0, 0, 0, 0, null, false, accepted, errors);
        }
    }
    if (Session.eval(view, token, errors) != 0) return -1;
    accepted[0] = token;
    return 1;
}

pub fn speculate(views: []View, tokens: []const c_int, pool: *Pool, policy: *Policy, accepted: [][2]c_int, counts: []c_int, errors: []u8) c_int {
    if (views.len == 0 or accepted.len != views.len or counts.len != views.len) return fail(errors, "empty speculative decode batch");
    if (validate(views, tokens, errors) != 0) return 1;
    if (penalty(&views[0])) {
        const result = eval(views, tokens, pool, errors);
        if (result == 0) for (accepted, counts, tokens) |*output, *count, token| {
            output[0] = token;
            count.* = 1;
        };
        return result;
    }
    var members: [maximum]c.ds4_qwen4_batch_member = undefined;
    if (views.len >= 2 and views.len <= 16 and views[0].kind == c.DS4_SESSION_QWEN and views[0].core.*.glm_mtp and
        qwenMembers(views, tokens, members[0..views.len]) and Qwen.supported(pool.qwen4_shared_workspace, members[0..views.len], true))
    {
        var runtimes: [16]c.ds4_qwen4_runtime = undefined;
        for (views, runtimes[0..views.len]) |*view, *runtime| runtime.* = Session.qwenRuntime(view);
        const result = QwenSpec.speculate(runtimes[0..views.len], tokens, pool.qwen4_shared_workspace, policy, accepted, counts, errors);
        if (result != 0) invalidate(views);
        return result;
    }
    for (views, tokens, accepted, counts) |*view, token, *output, *count| {
        const room: usize = @intCast(view.frontier.*.ctx_size - view.frontier.*.checkpoint.len);
        var inputs: [2]c_int = undefined;
        const length = scalarSpeculate(view, token, inputs[0..@min(room, 2)], errors);
        if (length <= 0) {
            invalidate(views);
            return 1;
        }
        count.* = length;
        output.* = .{ inputs[0], if (length > 1) inputs[1] else -1 };
    }
    return 0;
}

pub fn mixed(views: []View, tokens: []const c_int, pool: *Pool, prefill: *View, prompt: *const c.ds4_tokens, errors: []u8) c_int {
    if (prefill.kind != c.DS4_SESSION_FLINT) {
        if (views.len == 0) return fail(errors, "invalid mixed model batch");
        const frontier = prefill.frontier;
        if (!frontier.*.checkpoint_valid or prompt.len <= frontier.*.checkpoint.len or prompt.len > frontier.*.ctx_size or !c.ds4_tokens_starts_with(prompt, &frontier.*.checkpoint))
            return fail(errors, "mixed prefill must extend a valid session checkpoint");
        for (views, tokens, 0..) |view, token, index| {
            if (view.frontier == null or view.frontier == frontier or view.core != prefill.core or !view.frontier.*.checkpoint_valid or
                view.frontier.*.checkpoint.len >= view.frontier.*.ctx_size or token < 0 or token >= prefill.core.*.vocab.n_vocab)
                return indexed(errors, "invalid mixed decode item %d", index);
            for (views[0..index], 0..) |previous, earlier| if (previous.frontier == view.frontier) {
                if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "mixed decode repeats session at items %d and %d", @as(c_int, @intCast(earlier)), @as(c_int, @intCast(index)));
                return 1;
            };
        }
    }
    // Qwen has no supported mixed-prefill arena path; GLM and Flint already
    // sequence prefill before independent decode. Preserve that request order.
    const result = Session.sync(prefill, prompt, errors);
    if (result != 0) return result;
    const decoded = eval(views, tokens, pool, errors);
    if (decoded != 0) Session.invalidate(prefill);
    return decoded;
}
