//! Qwen session frontiers and point-proposal speculation. The target graph is
//! the commit authority; predictor recurrence is only a proposal. Rollback
//! snapshots let rejection restore GDN/PLE state without replaying a prefix.
const std = @import("std");
const c = @import("gguf_c");
const resources = @import("resources.zig");
const forward = @import("forward.zig");
const snapshots = @import("snapshot.zig");
const mtp = @import("mtp.zig");
const ops = @import("ops.zig");
pub const Runtime = c.ds4_qwen4_runtime;
pub const State = c.ds4_qwen4_session_state;
const Frontier = c.ds4_session_frontier;

pub fn fail(errors: []u8, message: [:0]const u8) c_int {
    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s", message.ptr);
    return 1;
}

pub fn trace() bool {
    return c.getenv("DS4_QWEN4_SPEC_TRACE") != null;
}

pub fn forceAccept() bool {
    return c.getenv("DS4_QWEN4_SPEC_FORCE_ACCEPT") != null;
}

pub fn noteFirst(state: *State, accepted: bool) void {
    state.depth_window = (state.depth_window << 1) | @intFromBool(accepted);
    state.depth_cycles += 1;
    if (accepted) state.depth_accepted += 1;
}

fn depth(state: *State) u32 {
    const raw = c.getenv("DS4_QWEN4_MTP_DEPTH");
    const fixed = if (raw != null and raw[0] != 0) c.atoi(raw) else 0;
    if (fixed == 2 or fixed == 3) return @intCast(fixed);
    const recent = @popCount(state.depth_window & 255);
    // Engage chained proposals only on a perfect comparable first-draft
    // window, then disengage quickly when the extra row stops paying.
    if (!state.depth3_engaged) {
        if (recent >= 8 and state.depth_cycles >= 8 and state.reject2_streak == 0) state.depth3_engaged = true;
    } else if (recent < 6 or state.reject2_streak >= 2) {
        state.depth3_engaged = false;
    }
    return if (state.depth3_engaged) 3 else 2;
}

fn append(frontier: *Frontier, token: c_int) void {
    c.ds4_tokens_push(&frontier.checkpoint, token);
}

pub fn logits(state: *State) [*]f32 {
    if (state.verify_logits == null) state.verify_logits = @ptrCast(@alignCast(c.ds4_sampling_alloc(4 * @as(usize, state.graph.config.shape.n_vocab) * @sizeOf(f32))));
    return state.verify_logits;
}

fn copyLogits(frontier: *Frontier, source: [*c]const f32, count: u32) void {
    @memcpy(frontier.logits[0..count], source[0..count]);
}

pub fn sync(runtime: *Runtime, prompt: *const c.ds4_tokens, errors: []u8) c_int {
    const state = runtime.state;
    const graph = &state.*.graph;
    const frontier = runtime.frontier;
    runtime.pending.*.have = 0;
    runtime.pending.*.have2 = false;
    var start: c_int = 0;
    if (frontier.*.checkpoint_valid and graph.pos == frontier.*.checkpoint.len and
        (!state.*.rewound or prompt.len > frontier.*.checkpoint.len) and
        prompt.len >= frontier.*.checkpoint.len and c.ds4_tokens_starts_with(prompt, &frontier.*.checkpoint))
    {
        start = frontier.*.checkpoint.len;
    } else {
        resources.reset(graph);
        frontier.*.checkpoint.len = 0;
        frontier.*.checkpoint_valid = false;
    }
    var index = start;
    while (index < prompt.len) : (index += 1) {
        const token = prompt.v[@intCast(index)];
        if (token < 0 or token >= graph.config.shape.n_vocab) {
            if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "token id %d at position %d is outside the vocabulary", token, index);
            return 1;
        }
    }
    graph.vis_spans = frontier.*.sync_images;
    graph.vis_span_count = frontier.*.sync_image_count;
    // A one-token prefill tail is a prompt row. Verifier rows are not.
    graph.dump_prompt_rows = true;
    defer {
        graph.vis_spans = null;
        graph.vis_span_count = 0;
        graph.dump_prompt_rows = false;
    }
    index = start;
    while (index < prompt.len) {
        if (runtime.control.*.cancel) |cancel| {
            if (cancel(runtime.control.*.cancel_ud)) {
                _ = fail(errors, "interrupted");
                frontier.*.checkpoint_valid = frontier.*.checkpoint.len > 0;
                return c.DS4_SESSION_SYNC_INTERRUPTED;
            }
        }
        const count = @min(@as(u32, @intCast(prompt.len - index)), graph.cap_tokens);
        frontier.*.checkpoint_valid = false;
        const tokens = prompt.v[@intCast(index)..][0..count];
        if (!forward.run(graph, runtime.model, runtime.weights, tokens, frontier.*.logits, false)) {
            if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "Qwen3.8 prefill failed at token %d", index);
            return 1;
        }
        for (tokens) |token| append(frontier, token);
        index += @intCast(count);
        frontier.*.checkpoint_valid = true;
        state.*.rewound = false;
        frontier.*.mtp_draft_valid = false;
        if (runtime.control.*.progress) |progress| progress(runtime.control.*.progress_ud, "prefill_chunk", index, prompt.len);
    }
    frontier.*.checkpoint_valid = true;
    frontier.*.mtp_draft_valid = false;
    return 0;
}

/// A rewind older than the three verifier snapshots resets recurrence. Keep
/// transcript ownership independent of sync, which may replace its frontier.
pub fn replay(runtime: *Runtime, errors: []u8) c_int {
    const state = runtime.state;
    const frontier = runtime.frontier;
    if (!state.*.ready or state.*.graph.pos == frontier.*.checkpoint.len) return 0;
    if (frontier.*.checkpoint_image_count != 0) {
        frontier.*.checkpoint_valid = false;
        _ = fail(errors, "Qwen3.8 image rewind requires multimodal sync");
        return -1;
    }
    var kept = std.mem.zeroes(c.ds4_tokens);
    c.ds4_tokens_copy(&kept, &frontier.*.checkpoint);
    defer c.ds4_tokens_free(&kept);
    frontier.*.checkpoint_valid = false;
    if (kept.len == 0) {
        resources.reset(&state.*.graph);
        return 0;
    }
    if (runtime.sync) |common_sync| return common_sync(runtime.user, &kept, if (errors.len == 0) null else errors.ptr, errors.len);
    return sync(runtime, &kept, errors);
}

pub fn eval(runtime: *Runtime, token: c_int, errors: []u8) c_int {
    const state = runtime.state;
    const frontier = runtime.frontier;
    runtime.control.*.dspark_rewind_end = 0;
    if (!state.*.ready) return fail(errors, "Qwen3.8 graph is not initialized");
    if (replay(runtime, errors) != 0) return 1;
    if (state.*.graph.pos >= state.*.graph.ctx_cap) return fail(errors, "context is full");
    if (runtime.pending.*.inside == 0) {
        runtime.pending.*.have = 0;
        runtime.pending.*.have2 = false;
    }
    if (!forward.run(&state.*.graph, runtime.model, runtime.weights, &.{token}, frontier.*.logits, false)) {
        frontier.*.checkpoint_valid = false;
        return fail(errors, "Qwen3.8 decode failed");
    }
    append(frontier, token);
    frontier.*.checkpoint_valid = true;
    state.*.rewound = false;
    frontier.*.mtp_draft_valid = false;
    return 0;
}

pub fn rewind(state: *State, frontier: *Frontier, position: u32) void {
    const graph = &state.graph;
    var row: ?usize = null;
    if (state.verify_logits != null) {
        if (graph.snap_valid and graph.snap_pos == position and snapshots.copy(graph, false, .first)) {
            row = 0;
        } else if (graph.snap2_valid and graph.snap2_pos == position and snapshots.copy(graph, false, .second)) {
            row = 1;
        } else if (graph.snap0_valid and graph.snap0_pos == position and snapshots.copy(graph, false, .boundary)) {
            row = 3;
        }
    }
    if (row) |index| {
        const vocab = graph.config.shape.n_vocab;
        copyLogits(frontier, state.verify_logits + index * vocab, vocab);
        graph.snap_valid = false;
        graph.snap2_valid = false;
        graph.snap0_valid = false;
        graph.mtp_pos = @min(graph.mtp_pos, position);
    } else resources.reset(graph);
    state.rewound = row == null;
}

fn draft(runtime: *Runtime, row: u32, parent: c_int, position: u32) void {
    const pending = runtime.pending;
    pending.*.have = 0;
    var token: c_int = -1;
    if (mtp.steps(&runtime.state.*.graph, runtime.model, runtime.weights, row, &.{parent}, position, true, null, &token)) {
        pending.*.draft = token;
        pending.*.parent = parent;
        pending.*.have = 1;
    }
}

fn chain(runtime: *Runtime, target_depth: u32, position: u32) void {
    if (target_depth != 3) return;
    var token: c_int = -1;
    if (mtp.chain(&runtime.state.*.graph, runtime.model, runtime.weights, runtime.pending.*.draft, position, &token)) {
        runtime.pending.*.draft2 = token;
        runtime.pending.*.have2 = true;
    }
}

fn propose(runtime: *Runtime, tokens: []const c_int, position: u32, target_depth: u32, chain_position: u32, can_chain: bool) void {
    const pending = runtime.pending;
    var token: c_int = -1;
    if (mtp.steps(&runtime.state.*.graph, runtime.model, runtime.weights, 0, tokens, position, true, null, &token)) {
        pending.*.draft = token;
        pending.*.parent = tokens[tokens.len - 1];
        pending.*.have = 1;
        if (can_chain) chain(runtime, target_depth, chain_position);
    }
}

fn pointAccept(probability: f32, rng: *u64) bool {
    if (probability >= 1) return true;
    if (!(probability > 0) or !std.math.isFinite(probability)) return false;
    return c.ds4_sampling_rng_f32(rng) <= probability;
}

pub fn speculate(runtime: *Runtime, first: c_int, temperature: f32, top_k: c_int, top_p: f32, min_p: f32, rng: [*c]u64, exact: bool, accepted: []c_int, errors: []u8) c_int {
    const state = runtime.state;
    const graph = &state.*.graph;
    const frontier = runtime.frontier;
    const pending = runtime.pending;
    if (replay(runtime, errors) != 0) return -1;
    const position = graph.pos;
    const vocab = graph.config.shape.n_vocab;
    const target_depth: u32 = if ((exact and temperature > 0) or c.getenv("DS4_QWEN4_NO_MTP_BATCH") != null) 2 else depth(state);
    if (pending.*.have != 0 and first != pending.*.parent) pending.*.have = 0;
    if (target_depth == 3 and pending.*.have != 0 and pending.*.have2 and graph.snap2_ple_hist == null and accepted.len >= 3 and position + 3 <= graph.ctx_cap and graph.cap_tokens >= 3) _ = snapshots.ensure(graph, .second);
    const deep = target_depth == 3 and pending.*.have != 0 and pending.*.have2 and accepted.len >= 3 and position + 3 <= graph.ctx_cap and graph.cap_tokens >= 3 and graph.mtp_R != null and graph.snap2_ple_hist != null;
    if (pending.*.have == 0 or accepted.len < 2 or position + 2 > graph.ctx_cap or graph.cap_tokens < 2 or graph.mtp_R == null or !ops.fused(graph, 2)) {
        pending.*.inside = 1;
        const status = eval(runtime, first, errors);
        pending.*.inside = 0;
        if (status != 0) return -1;
        const parent = c.ds4_sampling_argmax(frontier.*.logits, vocab);
        pending.*.have2 = false;
        propose(runtime, &.{parent}, position, target_depth, position + 1, position + 2 <= graph.ctx_cap);
        if (trace()) _ = c.fprintf(c.__stderrp, "ds4: spec pos %u token %d plain\n", position, first);
        accepted[0] = first;
        return 1;
    }
    pending.*.have = 0;
    pending.*.have2 = false;
    const first_draft = pending.*.draft;
    const second_draft = if (deep) pending.*.draft2 else -1;
    const tokens = [_]c_int{ first, first_draft, second_draft };
    const count: u32 = if (deep) 3 else 2;
    const rows = logits(state);
    // Exact-mode caller may discard a whole block at a tool-syntax boundary.
    // Preserve its boundary logits and recurrence independently of row0.
    graph.snap0_valid = runtime.exact_sampling and snapshots.ensure(graph, .boundary) and snapshots.copy(graph, true, .boundary);
    if (graph.snap0_valid) @memcpy(rows[3 * vocab ..][0..vocab], frontier.*.logits[0..vocab]);
    graph.snap_after_first = graph.snap_ple_hist != null;
    graph.snap_valid = false;
    graph.snap_after_second = deep and graph.snap2_ple_hist != null;
    graph.snap2_valid = false;
    graph.verify_rows_exact = deep;
    c.ds4_gpu_qwen4_set_verify_rows_exact(deep);
    const ok = forward.run(graph, runtime.model, runtime.weights, tokens[0..count], rows, true);
    c.ds4_gpu_qwen4_set_verify_rows_exact(false);
    graph.verify_rows_exact = false;
    graph.snap_after_second = false;
    if (!ok) {
        graph.snap_after_first = false;
        frontier.*.checkpoint_valid = false;
        _ = fail(errors, "Qwen3.8 mtp: verify failed");
        return -1;
    }
    state.*.spec_cycles += 1;
    state.*.rewound = false;
    append(frontier, first);
    frontier.*.checkpoint_valid = true;
    frontier.*.mtp_draft_valid = false;
    var accept = c.ds4_sampling_argmax(rows, vocab) == first_draft or forceAccept();
    const accept_second = deep and accept and (c.ds4_sampling_argmax(rows + vocab, vocab) == second_draft or forceAccept());
    var replacement: c_int = -1;
    if (exact and temperature > 0) {
        if (frontier.*.sample_probs == null) frontier.*.sample_probs = @ptrCast(@alignCast(c.ds4_sampling_alloc(@as(usize, vocab) * @sizeOf(f32))));
        if (rng == null or !c.ds4_sampling_build_probabilities(rows, vocab, temperature, top_k, top_p, min_p, frontier.*.sample_probs)) {
            frontier.*.checkpoint_valid = false;
            _ = fail(errors, "Qwen3.8 mtp: target distribution failed");
            return -1;
        }
        accept = pointAccept(frontier.*.sample_probs[@intCast(first_draft)], rng);
        if (!accept) {
            frontier.*.sample_probs[@intCast(first_draft)] = 0;
            replacement = c.ds4_sampling_probabilities(frontier.*.sample_probs, vocab, rng);
            if (replacement < 0) {
                frontier.*.checkpoint_valid = false;
                _ = fail(errors, "Qwen3.8 mtp: replacement sampling failed");
                return -1;
            }
        }
    }
    noteFirst(state, accept);
    if (deep) {
        if (accept_second) state.*.reject2_streak = 0 else if (accept) {
            state.*.reject2_streak += 1;
        }
    }
    if (trace()) _ = c.fprintf(c.__stderrp, "ds4: spec pos %u token %d draft %d %s%s\n", position, first, first_draft, @as([*:0]const u8, if (accept) "accept" else "reject"), @as([*:0]const u8, if (deep) (if (accept_second) " +accept2" else " +reject2") else ""));
    if (accept and (!deep or accept_second)) {
        append(frontier, first_draft);
        if (deep) append(frontier, second_draft);
        copyLogits(frontier, rows + (count - 1) * vocab, vocab);
        const parent = c.ds4_sampling_argmax(frontier.*.logits, vocab);
        const next = [_]c_int{ first_draft, second_draft, parent };
        const short = [_]c_int{ first_draft, parent };
        propose(runtime, if (deep) &next else &short, position, target_depth, graph.pos, graph.pos + 1 < graph.ctx_cap);
        state.*.spec_accepted += 1;
        accepted[0] = first;
        accepted[1] = first_draft;
        if (deep) accepted[2] = second_draft;
        return if (deep) 3 else 2;
    }
    if (accept and deep) {
        if (!graph.snap2_valid or !snapshots.swap(graph, .second)) {
            frontier.*.checkpoint_valid = false;
            _ = fail(errors, "Qwen3.8 mtp: rejection restore failed");
            return -1;
        }
        frontier.*.checkpoint.len = @intCast(position + 1);
        append(frontier, first_draft);
        copyLogits(frontier, rows + vocab, vocab);
        const parent = c.ds4_sampling_argmax(frontier.*.logits, vocab);
        propose(runtime, &.{ first_draft, parent }, position, target_depth, graph.pos, graph.pos + 1 < graph.ctx_cap);
        state.*.spec_accepted += 1;
        accepted[0] = first;
        accepted[1] = first_draft;
        return 2;
    }
    if (!graph.snap_valid or !snapshots.swap(graph, .first)) {
        frontier.*.checkpoint_valid = false;
        _ = fail(errors, "Qwen3.8 mtp: rejection restore failed");
        return -1;
    }
    if (replacement >= 0) {
        if (!forward.run(graph, runtime.model, runtime.weights, &.{replacement}, frontier.*.logits, false)) {
            frontier.*.checkpoint_valid = false;
            _ = fail(errors, "Qwen3.8 mtp: replacement replay failed");
            return -1;
        }
        append(frontier, replacement);
        draft(runtime, 0, c.ds4_sampling_argmax(frontier.*.logits, vocab), position + 1);
        accepted[0] = first;
        accepted[1] = replacement;
        return 2;
    }
    copyLogits(frontier, rows, vocab);
    const parent = c.ds4_sampling_argmax(frontier.*.logits, vocab);
    propose(runtime, &.{parent}, position, target_depth, position + 1, position + 2 <= graph.ctx_cap);
    accepted[0] = first;
    return 1;
}
