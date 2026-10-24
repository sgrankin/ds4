//! GLM53 session frontiers and point-proposal speculation. Target state is the
//! commit authority: verification backs up KDA recurrence and unfinished DSA
//! pools, then rejection restores that state before replaying retained rows.
const std = @import("std");
const c = @import("gguf_c");
const Config = @import("config.zig");
const Planning = @import("planning.zig");
const Chunk = @import("chunk.zig");
const Executor = @import("executor.zig");
const Resources = @import("resources.zig");
const Speculation = @import("speculation.zig");
const Mtp = @import("mtp.zig");
const Head = @import("head.zig");
pub const State = c.ds4_glm_session_state;
pub const Runtime = c.ds4_glm_runtime;
const Frontier = c.ds4_session_frontier;
const Pending = c.ds4_model_speculation_state;

fn fail(errors: []u8, message: [:0]const u8) c_int {
    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s", message.ptr);
    return 1;
}
fn now() f64 {
    var time: c.struct_timeval = undefined;
    _ = c.gettimeofday(&time, null);
    return @as(f64, @floatFromInt(time.tv_sec)) + @as(f64, @floatFromInt(time.tv_usec)) / 1.0e6;
}
fn append(frontier: *Frontier, token: c_int) void {
    c.ds4_tokens_push(&frontier.checkpoint, token);
}
fn backend(runtime: *const Runtime) [*c]const u8 {
    return c.ds4_backend_name(runtime.core.*.backend);
}

pub fn resetDense(state: *State, pending: *Pending) void {
    state.dense_cache_len = 0;
    pending.have = 0;
    pending.have2 = false;
    pending.rollback_valid = false;
    pending.min_pos = 0;
}
pub fn capDense(state: *State, frontier: *const Frontier) void {
    if (state.ready and !state.graph.full_kv_cache) {
        state.dense_cache_len = 0;
        return;
    }
    var capacity: u32 = if (frontier.checkpoint.len > 0) @intCast(frontier.checkpoint.len) else 0;
    if (state.ready) capacity = @min(capacity, state.graph.ctx_cap);
    state.dense_cache_len = @min(state.dense_cache_len, capacity);
}
pub fn noteDense(state: *State, frontier: *const Frontier, position: u32, count: u32) void {
    if (count == 0 or (state.ready and !state.graph.full_kv_cache) or position > state.dense_cache_len) return;
    const end = std.math.add(u32, position, count) catch std.math.maxInt(u32);
    state.dense_cache_len = if (state.ready) @min(end, state.graph.ctx_cap) else end;
    capDense(state, frontier);
}
fn updatesDense(state: *const State, position: u32) bool {
    const graph = &state.graph;
    const indexed = graph.compact_cache_cap != 0 and
        (!graph.full_kv_cache or position >= graph.ctx_cap or Planning.useIndexedPrefill(graph));
    return position < graph.ctx_cap and !indexed;
}
fn imageOverlap(frontier: *const Frontier, position: u32, count: u32) bool {
    const end = @as(u64, position) + count;
    for (frontier.sync_images[0..frontier.sync_image_count]) |span| {
        const image_end = @as(u64, span.token_start) + span.embedding.token_count;
        if (position < image_end and span.token_start < end) return true;
    }
    return false;
}
fn failed(runtime: *Runtime) void {
    runtime.frontier.*.checkpoint_valid = false;
    runtime.frontier.*.mtp_draft_valid = false;
    capDense(runtime.state, runtime.frontier);
}
fn scalar(runtime: *Runtime, token: c_int, position: u32, logits: ?[*]f32) bool {
    return Executor.forward(&runtime.state.*.graph, &runtime.core.*.model, &runtime.core.*.weights, token, null, position, null, logits, false, runtime.ablate_mask);
}

pub fn sync(runtime: *Runtime, requested: *const c.ds4_tokens, errors: []u8) c_int {
    const state = runtime.state;
    const graph = &state.*.graph;
    const frontier = runtime.frontier;
    const control = runtime.control;
    var prompt = requested.*;
    const truncate = c.getenv("DS4_GLM_PREFILL_TRUNC");
    if (truncate != null and truncate[0] != 0) {
        const length = c.atoi(truncate);
        if (length > 0 and length < prompt.len) prompt.len = length;
    }
    if (!state.*.ready) {
        if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s GLM graph is not initialized", backend(runtime));
        return 1;
    }
    if (prompt.len >= graph.ctx_size) {
        if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "prompt length %d leaves no GLM Metal context room (ctx %u)", prompt.len, graph.ctx_size);
        frontier.*.checkpoint_valid = false;
        resetDense(state, runtime.pending);
        return 1;
    }
    var start: c_int = 0;
    var resumed = false;
    if (frontier.*.checkpoint_valid and prompt.len >= frontier.*.checkpoint.len and c.ds4_tokens_starts_with(&prompt, &frontier.*.checkpoint)) {
        start = frontier.*.checkpoint.len;
        resumed = true;
        frontier.*.mtp_draft_valid = false;
    } else {
        frontier.*.checkpoint.len = 0;
        frontier.*.checkpoint_valid = false;
        frontier.*.mtp_draft_valid = false;
        resetDense(state, runtime.pending);
        if (!Resources.resetKda(graph)) {
            if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s GLM KDA state reset failed", backend(runtime));
            return 1;
        }
    }
    // Very short checkpoint extensions use decode; image rows always require
    // the batch embedding overlay. Longer suffixes amortize layer-major setup.
    const suffix: u32 = @intCast(prompt.len - start);
    const short_resume = resumed and suffix != 0 and suffix < Planning.resumePrefillMin(graph) and
        !imageOverlap(frontier, @intCast(start), suffix);
    var index = start;
    while (index < prompt.len) {
        if (control.*.cancel) |cancel| if (cancel(control.*.cancel_ud)) {
            _ = fail(errors, "interrupted");
            frontier.*.mtp_draft_valid = false;
            capDense(state, frontier);
            return c.DS4_SESSION_SYNC_INTERRUPTED;
        };
        const position: u32 = @intCast(frontier.*.checkpoint.len);
        var count: u32 = @min(@as(u32, @intCast(prompt.len - index)), graph.config.prefill_chunk);
        if (short_resume) count = 1;
        const indexed = !short_resume and (Planning.useIndexedPrefill(graph) or
            (position >= graph.ctx_cap and graph.indexed_prefill_cap != 0));
        if (indexed) {
            count = Planning.limitIndexedChunk(graph, position, @min(count, graph.indexed_prefill_cap));
        } else if (!short_resume and position < graph.ctx_cap) {
            count = @min(count, graph.ctx_cap - position);
        } else if (!short_resume) count = 1;
        const logits: ?[*]f32 = if (index + @as(c_int, @intCast(count)) == prompt.len) frontier.*.logits else null;
        const tokens = prompt.v[@intCast(index)..][0..count];
        // Nonfinal chunks advance recurrent/cache state without a head. Their
        // transcript cannot be sampled or persisted with the previous logits.
        frontier.*.checkpoint_valid = false;
        const ok = if (short_resume or (!indexed and position >= graph.ctx_cap))
            scalar(runtime, tokens[0], position, logits)
        else
            Chunk.forward(graph, &runtime.core.*.model, &runtime.core.*.weights, tokens, null, frontier.*.sync_images[0..frontier.*.sync_image_count], position, null, logits, .{
                .callback = control.*.display_progress,
                .user = control.*.display_progress_ud,
                .absolute_base = @intCast(start),
                .done_base = @intCast(index - start),
                .total = suffix,
            }, runtime.ablate_mask);
        if (!ok) {
            if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s GLM-5.3 prefill failed at token %d", backend(runtime), index);
            failed(runtime);
            return 1;
        }
        for (tokens) |token| append(frontier, token);
        frontier.*.checkpoint_valid = logits != null;
        frontier.*.mtp_draft_valid = false;
        if (position < graph.ctx_cap) noteDense(state, frontier, position, count);
        index += @intCast(count);
        if (control.*.progress) |progress| progress(control.*.progress_ud, "prefill_chunk", index, prompt.len);
    }
    c.ds4_native_session_prefill_logit_dump(frontier.*.logits, graph.config.shape.n_vocab);
    return 0;
}

/// Explicit invalid-frontier replay keeps the transcript alive while the
/// public sync service performs common image and request bookkeeping.
pub fn replay(runtime: *Runtime, errors: []u8) c_int {
    if (runtime.frontier.*.checkpoint_valid) return 0;
    if (runtime.frontier.*.checkpoint_image_count != 0) return fail(errors, "GLM image rewind requires multimodal sync");
    var kept = std.mem.zeroes(c.ds4_tokens);
    c.ds4_tokens_copy(&kept, &runtime.frontier.*.checkpoint);
    defer c.ds4_tokens_free(&kept);
    if (runtime.sync) |common_sync|
        return common_sync(runtime.user, &kept, if (errors.len == 0) null else errors.ptr, errors.len);
    return sync(runtime, &kept, errors);
}

const Probe = struct { hits: u32 = 0, total: u32 = 0, have: bool = false, draft: c_int = -1, minimum: u32 = std.math.maxInt(u32) };
var probe = Probe{};
fn probeDraft(runtime: *Runtime, position: u32) void {
    const next = c.ds4_sampling_argmax(runtime.frontier.*.logits, runtime.state.*.graph.config.shape.n_vocab);
    if (probe.have) {
        probe.total += 1;
        probe.hits += @intFromBool(probe.draft == next);
        if (probe.total % 16 == 0) _ = c.fprintf(c.__stderrp, "ds4: glm mtp probe: %u/%u hits (%.1f%%)\n", probe.hits, probe.total, 100.0 * @as(f64, @floatFromInt(probe.hits)) / @as(f64, @floatFromInt(probe.total)));
    }
    if (probe.minimum == std.math.maxInt(u32)) probe.minimum = position;
    probe.have = Mtp.step(&runtime.state.*.graph, &runtime.core.*.model, &runtime.core.*.weights, next, position, probe.minimum, &probe.draft);
    if (!probe.have) _ = c.fprintf(c.__stderrp, "ds4: glm mtp probe: draft step failed at pos %u\n", position);
}

pub fn eval(runtime: *Runtime, token: c_int, errors: []u8) c_int {
    const state = runtime.state;
    const frontier = runtime.frontier;
    runtime.control.*.dspark_rewind_end = 0;
    if (!state.*.ready) {
        if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s GLM graph is not initialized", backend(runtime));
        return 1;
    }
    if (frontier.*.checkpoint.len >= state.*.graph.ctx_size) {
        if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "GLM Metal context reached (%u)", state.*.graph.ctx_size);
        return 1;
    }
    if (runtime.pending.*.inside == 0) {
        runtime.pending.*.rollback_valid = false;
        runtime.pending.*.have = 0;
    }
    const position: u32 = @intCast(frontier.*.checkpoint.len);
    const updates_dense = updatesDense(state, position);
    if (!scalar(runtime, token, position, frontier.*.logits)) {
        if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s GLM decode failed", backend(runtime));
        failed(runtime);
        return 1;
    }
    append(frontier, token);
    frontier.*.checkpoint_valid = true;
    frontier.*.mtp_draft_valid = false;
    if (updates_dense) noteDense(state, frontier, position, 1);
    if (c.getenv("DS4_GLM_MTP_PROBE") != null and state.*.graph.config.shape.n_nextn_predict != 0) probeDraft(runtime, position);
    return 0;
}

pub fn rewind(runtime: *Runtime, position: u32) bool {
    const state = runtime.state;
    const graph = &state.*.graph;
    const pending = runtime.pending;
    if (!pending.*.rollback_valid or !graph.glm53 or runtime.frontier.*.checkpoint.len != pending.*.rollback_pos + 2 or
        (position != pending.*.rollback_pos and position != pending.*.rollback_pos + 1)) return false;
    const start = pending.*.rollback_pos;
    var ok = Speculation.copy(graph, false);
    state.*.dense_cache_len = pending.*.rollback_dense_len;
    if (ok and position == start + 1) {
        ok = scalar(runtime, pending.*.rollback_first_token, start, runtime.frontier.*.logits);
        if (ok) noteDense(state, runtime.frontier, start, 1);
    }
    pending.*.rollback_valid = false;
    pending.*.have = 0;
    return ok;
}

fn propose(runtime: *Runtime, token: c_int, position: u32) void {
    var draft: c_int = -1;
    const pending = runtime.pending;
    pending.*.have = 0;
    if (Mtp.step(&runtime.state.*.graph, &runtime.core.*.model, &runtime.core.*.weights, token, position, pending.*.min_pos, &draft)) {
        pending.*.draft = draft;
        pending.*.parent = token;
        pending.*.have = 1;
    }
}
fn argmax(runtime: *Runtime, logits: [*c]const f32) c_int {
    return c.ds4_sampling_argmax(logits, runtime.state.*.graph.config.shape.n_vocab);
}
fn rejectFailure(runtime: *Runtime, restore: bool, errors: []u8, message: [:0]const u8) c_int {
    if (restore) _ = Speculation.copy(&runtime.state.*.graph, false);
    runtime.frontier.*.checkpoint_valid = false;
    _ = fail(errors, message);
    return -1;
}
fn pointAccept(probability: f32, rng: *u64) bool {
    if (probability >= 1) return true;
    if (!(probability > 0) or !std.math.isFinite(probability)) return false;
    return c.ds4_sampling_rng_f32(rng) <= probability;
}

/// A deterministic draft is a point proposal. Greedy mode accepts target
/// argmax; exact sampling uses target probability and samples the residual on
/// rejection. Only committed target rows become the public frontier.
pub fn speculate(runtime: *Runtime, first: c_int, eos: c_int, temperature: f32, top_k: c_int, top_p: f32, min_p: f32, rng: [*c]u64, exact: bool, accepted: []c_int, errors: []u8) c_int {
    if (accepted.len == 0) return 0;
    const state = runtime.state;
    const graph = &state.*.graph;
    const frontier = runtime.frontier;
    const pending = runtime.pending;
    const model = &runtime.core.*.model;
    const weights = &runtime.core.*.weights;
    const position: u32 = @intCast(frontier.*.checkpoint.len);
    const dense_before = state.*.dense_cache_len;
    const timing = runtime.core.*.glm_mtp_timing;
    const vocab = graph.config.shape.n_vocab;
    const hidden_values: u64 = Config.hcWidth(&graph.config);
    const hidden_bytes = hidden_values * @sizeOf(f32);
    if (pending.*.hc == null) {
        pending.*.hc = @ptrCast(@alignCast(c.malloc(@intCast(2 * hidden_bytes))));
        pending.*.logits0 = @ptrCast(@alignCast(c.malloc(@as(usize, vocab) * @sizeOf(f32))));
        if (pending.*.hc == null or pending.*.logits0 == null) {
            _ = fail(errors, "glm mtp: out of memory");
            return -1;
        }
    }
    pending.*.rollback_valid = false;
    if (pending.*.have != 0 and first != pending.*.parent) pending.*.have = 0;
    if (pending.*.have == 0 or accepted.len < 2 or position + 2 > graph.ctx_size or
        (graph.compact_cache_cap != 0 and position + 1 >= graph.compact_cache_cap))
    {
        pending.*.inside = 1;
        const status = eval(runtime, first, errors);
        pending.*.inside = 0;
        if (status != 0) return -1;
        if (pending.*.min_pos == 0 or pending.*.min_pos > position) pending.*.min_pos = position;
        propose(runtime, argmax(runtime, frontier.*.logits), position);
        accepted[0] = first;
        return 1;
    }
    const start_time = if (timing) now() else 0;
    pending.*.have = 0;
    const draft = pending.*.draft;
    const saved = Speculation.ensure(graph) and Speculation.copy(graph, true);
    const tokens = [_]c_int{ first, draft };
    const verified = saved and Chunk.forward(graph, model, weights, &tokens, null, &.{}, position, pending.*.hc, frontier.*.logits, .{ .absolute_base = position, .total = 2 }, runtime.ablate_mask);
    if (!verified) return rejectFailure(runtime, saved, errors, "glm mtp: GLM 5.3 verify failed");
    const verified_time = if (timing) now() else 0;
    if (!Speculation.ensure(graph)) return rejectFailure(runtime, saved, errors, "glm mtp: scratch alloc failed");
    // Row0's four HC streams must pass through the shared language head;
    // graph.logits from the two-row pass already represents row1.
    const head_ok = c.ds4_gpu_tensor_write(graph.hc_cur, 0, pending.*.hc, hidden_bytes) != 0 and
        Head.forward(graph, model, weights, graph.hc_cur, pending.*.logits0);
    if (!head_ok) return rejectFailure(runtime, saved, errors, "glm mtp: row0 head failed");
    const target = argmax(runtime, pending.*.logits0);
    var replacement: c_int = -1;
    var accept = target == draft;
    if (exact) {
        if (rng == null or !c.ds4_sampling_build_probabilities(pending.*.logits0, vocab, temperature, top_k, top_p, min_p, frontier.*.sample_probs))
            return rejectFailure(runtime, saved, errors, "glm mtp: target distribution failed");
        accept = pointAccept(frontier.*.sample_probs[@intCast(draft)], rng);
        if (!accept) {
            frontier.*.sample_probs[@intCast(draft)] = 0;
            replacement = c.ds4_sampling_probabilities(frontier.*.sample_probs, vocab, rng);
            if (replacement < 0) return rejectFailure(runtime, saved, errors, "glm mtp: replacement sampling failed");
        }
    }
    append(frontier, first);
    frontier.*.checkpoint_valid = true;
    var committed: c_int = 1;
    if (accept) {
        append(frontier, draft);
        noteDense(state, frontier, position, 2);
        committed = 2;
        // Advance both predictor positions so the next proposal is conditioned
        // on the verified continuation, without changing target recurrence.
        const next = argmax(runtime, frontier.*.logits);
        var ignored: c_int = -1;
        var next_draft: c_int = -1;
        if (c.ds4_gpu_tensor_write(graph.hc_cur, 0, pending.*.hc, hidden_bytes) != 0 and
            Mtp.step(graph, model, weights, draft, position, pending.*.min_pos, &ignored) and
            c.ds4_gpu_tensor_write(graph.hc_cur, 0, pending.*.hc + @as(usize, @intCast(hidden_values)), hidden_bytes) != 0 and
            Mtp.step(graph, model, weights, next, position + 1, pending.*.min_pos, &next_draft))
        {
            pending.*.draft = next_draft;
            pending.*.parent = next;
            pending.*.have = 1;
        }
        accepted[0] = first;
        accepted[1] = draft;
    } else {
        if (!Speculation.copy(graph, false) or !scalar(runtime, first, position, frontier.*.logits))
            return rejectFailure(runtime, false, errors, "glm mtp: rejection replay failed");
        if (exact) {
            if (!scalar(runtime, replacement, position + 1, frontier.*.logits))
                return rejectFailure(runtime, false, errors, "glm mtp: replacement replay failed");
            append(frontier, replacement);
            noteDense(state, frontier, position, 2);
            committed = 2;
            accepted[0] = first;
            accepted[1] = replacement;
            if (replacement != eos) propose(runtime, argmax(runtime, frontier.*.logits), position + 1) else pending.*.have = 0;
        } else {
            noteDense(state, frontier, position, 1);
            propose(runtime, target, position);
            accepted[0] = first;
        }
    }
    if (timing) {
        const end_time = now();
        const draft_text = c.ds4_vocab_decode_token(&runtime.core.*.vocab, draft, null);
        const target_text = c.ds4_vocab_decode_token(&runtime.core.*.vocab, target, null);
        defer c.free(draft_text);
        defer c.free(target_text);
        const status: [*:0]const u8 = if (accept) (if (exact) "EXACT_ACCEPT" else "ACCEPT") else (if (exact) "exact_reject" else "reject");
        _ = c.fprintf(c.__stderrp, "ds4: glm mtp cycle: verify2 %.1f ms, head+draft %.1f ms, %s (draft %d '%s' vs true %d '%s')\n", (verified_time - start_time) * 1000, (end_time - verified_time) * 1000, status, draft, if (draft_text != null) draft_text else @as([*c]const u8, "?"), target, if (target_text != null) target_text else @as([*c]const u8, "?"));
    }
    if (saved and committed == 2) {
        pending.*.rollback_pos = position;
        pending.*.rollback_dense_len = dense_before;
        pending.*.rollback_first_token = first;
        pending.*.rollback_valid = true;
    }
    return committed;
}
