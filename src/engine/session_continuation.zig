//! Ordinary Qwen/GLM/Flint continuation policy. Numerical verification stays
//! with each native model; this owner admits packets, publishes their outer
//! frontier and clears common proposal state after rewind. DS41 has its own
//! complete native coordinator and is dispatched before this component.
const std = @import("std");
const c = @import("gguf_c");
const Runtime = @import("session_runtime.zig");
const Control = @import("session_control.zig");
const View = c.ds4_session_view;

fn fail(errors: []u8, text: [:0]const u8) c_int {
    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s", text.ptr);
    return -1;
}
fn pointer(errors: []u8) [*c]u8 {
    return if (errors.len == 0) null else errors.ptr;
}
fn plain(view: *View, first: c_int, accepted: [*c]c_int, errors: []u8) c_int {
    if (accepted == null) return 0;
    if (c.ds4_native_session_eval(view, first, pointer(errors), errors.len) != 0) return -1;
    accepted[0] = first;
    return 1;
}
fn packet(view: *View, first: c_int, maximum: c_int, temperature: f32, top_k: c_int, top_p: f32, min_p: f32, rng: [*c]u64, accepted: [*c]c_int, capacity: c_int, errors: []u8) c_int {
    if (accepted == null or maximum <= 0 or capacity <= 0) return 0;
    const frontier = view.frontier;
    var limit = @min(@min(maximum, capacity), @min(frontier.*.ctx_size - frontier.*.checkpoint.len, 8));
    if (view.core.*.overthink.penalty > 0) limit = @min(limit, 1);
    if (!frontier.*.checkpoint_valid or limit < 1 or first < 0 or first >= view.core.*.vocab.n_vocab)
        return fail(errors, "invalid Flint packet frontier/token");
    const count = c.flint_session_packet(view.flint.*, first, @intCast(limit), temperature, top_k, top_p, min_p, rng, accepted, pointer(errors), errors.len);
    if (count < 1 or count > limit) {
        c.ds4_native_session_invalidate(view);
        return -1;
    }
    for (accepted[0..@intCast(count)]) |token| c.ds4_tokens_push(&frontier.*.checkpoint, token);
    const vocabulary: usize = @intCast(view.core.*.vocab.n_vocab);
    @memcpy(frontier.*.logits[0..vocabulary], c.flint_session_logits(view.flint.*)[0..vocabulary]);
    frontier.*.checkpoint_valid = true;
    return count;
}

pub fn speculate(view: *View, first: c_int, maximum: c_int, eos: c_int, temperature: f32, top_k: c_int, top_p: f32, min_p: f32, rng: [*c]u64, ignore_eos: bool, accepted: [*c]c_int, capacity: c_int, errors: []u8) c_int {
    if (view.kind == c.DS4_SESSION_FLINT) {
        if (ignore_eos) return if (maximum > 0 and capacity > 0) plain(view, first, accepted, errors) else 0;
        return packet(view, first, maximum, temperature, top_k, top_p, min_p, rng, accepted, capacity, errors);
    }
    if (maximum <= 0 or capacity <= 0) return 0;
    const greedy = temperature <= 0;
    if (!greedy and (accepted == null or rng == null)) return 0;
    if (!view.frontier.*.checkpoint_valid) return fail(errors, "speculative decode requires a synchronized checkpoint");
    if (view.core.*.overthink.penalty > 0) return plain(view, first, accepted, errors);
    if (accepted == null) return 0;
    const limit = @min(maximum, capacity);
    const exact = !greedy and view.core.*.dspark_exact_sampling;
    const t: f32 = if (greedy) 0 else temperature;
    const k: c_int = if (greedy) 0 else top_k;
    const p: f32 = if (greedy) 0 else top_p;
    const m: f32 = if (greedy) 0 else min_p;
    const random: [*c]u64 = if (greedy) null else rng;
    if (view.kind == c.DS4_SESSION_QWEN and view.core.*.glm_mtp and
        view.qwen.*.graph.config.shape.n_nextn_predict != 0 and view.qwen.*.ready)
    {
        var runtime = Runtime.qwen(view);
        return c.ds4_native_qwen4_session_speculate(&runtime, first, t, k, p, m, random, exact, accepted, limit, pointer(errors), errors.len);
    }
    if (view.kind == c.DS4_SESSION_GLM53 and view.core.*.glm_mtp and
        view.glm.*.graph.config.shape.n_nextn_predict != 0 and view.glm.*.ready and
        !(greedy and c.getenv("DS4_GLM_MTP_PROBE") != null))
    {
        var runtime = Runtime.glm(view);
        return c.ds4_native_glm_session_speculate(&runtime, first, if (greedy) -1 else eos, t, k, if (greedy) 1 else p, m, random, exact, accepted, limit, pointer(errors), errors.len);
    }
    return plain(view, first, accepted, errors);
}

pub fn rewind(view: *View, requested: c_int) void {
    const frontier = view.frontier;
    const position = @max(requested, 0);
    if (position >= frontier.*.checkpoint.len) return;
    if (view.kind == c.DS4_SESSION_FLINT) {
        if (position == 0) {
            c.ds4_native_session_invalidate(view);
            return;
        }
        var prefix = std.mem.zeroes(c.ds4_tokens);
        defer c.ds4_tokens_free(&prefix);
        for (frontier.*.checkpoint.v[0..@intCast(position)]) |token| c.ds4_tokens_push(&prefix, token);
        var errors: [256]u8 = @splat(0);
        if (c.ds4_native_session_sync(view, &prefix, &errors, errors.len) != 0)
            _ = c.fprintf(c.__stderrp, "ds4: Flint rewind: %s\n", &errors);
        return;
    }
    var state_ok = false;
    if (frontier.*.checkpoint_valid) switch (view.kind) {
        c.DS4_SESSION_QWEN => {
            c.ds4_native_qwen4_session_rewind(view.qwen, frontier, @intCast(position));
            state_ok = true;
        },
        c.DS4_SESSION_GLM53 => {
            var runtime = Runtime.glm(view);
            state_ok = !view.glm.*.graph.glm53 or c.ds4_native_glm_session_rewind(&runtime, @intCast(position));
        },
        else => {},
    };
    frontier.*.checkpoint.len = position;
    if (!state_ok) frontier.*.checkpoint_valid = false;
    frontier.*.mtp_draft_valid = false;
    Control.clearCapture(view.control);
    view.pending.*.have = 0;
    view.pending.*.rollback_valid = false;
    // The previous common cleanup also capped GLM's dormant zero-state record
    // in other families. Only GLM has a real dense-cache owner in SessionView.
    if (view.kind == c.DS4_SESSION_GLM53) c.ds4_native_glm_session_dense_cap(view.glm, frontier);
}
