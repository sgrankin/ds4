//! Ragged speculative batches and their measured schedule. Target verification
//! owns token acceptance; predictor work is scheduled only when it earns its
//! cost. Common session validation/invalidation remains in the public façade.
const std = @import("std");
const c = @import("gguf_c");
const batch = @import("batch.zig");
const session = @import("session.zig");
const snapshots = @import("snapshot.zig");
const ops = @import("ops.zig");
const Runtime = session.Runtime;
const Policy = c.ds4_qwen4_batch_policy;
const maximum = 16;

fn ema(previous: f32, sample: f32, count: u32) f32 {
    const denominator: f32 = if (count < 15) @floatFromInt(count + 1) else 16;
    return previous + (sample - previous) / denominator;
}

fn next(policy: *const Policy) bool {
    // Periodic probes let a change in text or GPU load reverse the decision.
    if (policy.ms[0] <= 0 or policy.ms[1] <= 0) return policy.cycles % 32 < 8;
    return if ((1 + policy.p) * policy.ms[0] > policy.ms[1]) policy.cycles % 128 >= 2 else policy.cycles % 32 < 2;
}

pub fn speculate(runtimes: []Runtime, tokens: []const c_int, arena: *c.ds4_qwen4_gpu_graph, policy: *Policy, accepted: [][2]c_int, counts: []c_int, errors: []u8) c_int {
    const count = runtimes.len;
    if (count == 0 or count > maximum or tokens.len != count or accepted.len != count or counts.len != count) return session.fail(errors, "invalid Qwen3.8 speculative batch");
    const vocab = arena.config.shape.n_vocab;
    const started = ops.now();
    if (policy.width != count) {
        policy.* = std.mem.zeroes(Policy);
        policy.width = @intCast(count);
    }
    var members: [maximum]c.ds4_qwen4_batch_member = undefined;
    var positions: [maximum]u32 = undefined;
    var committed: [maximum]u32 = undefined;
    var parents: [maximum]c_int = undefined;
    var drafts: [maximum]c_int = undefined;
    var rows: u32 = 0;
    var drafted: u32 = 0;
    var accepted_drafts: u32 = 0;
    for (runtimes, tokens, 0..) |*runtime, token, index| {
        const graph = &runtime.state.*.graph;
        const pending = runtime.pending;
        const frontier = runtime.frontier;
        members[index] = .{ .graph = graph, .frontier = frontier, .tokens = .{ token, -1 }, .n = 1, .row0 = rows, .slot = runtime.state.*.slot };
        positions[index] = graph.pos;
        // The rows read below replace those paired with older snapshots.
        graph.snap_valid = false;
        graph.snap2_valid = false;
        graph.snap0_valid = false;
        if (pending.*.have != 0 and pending.*.parent == token and frontier.*.ctx_size - frontier.*.checkpoint.len >= 2 and graph.pos + 2 <= graph.ctx_cap and graph.snap_ple_hist != null and graph.mtp_R != null) {
            members[index].tokens[1] = pending.*.draft;
            members[index].n = 2;
            graph.snap_after_first = true;
        }
        pending.*.have = 0;
        pending.*.have2 = false;
        rows += members[index].n;
    }
    var ok = c.ds4_gpu_begin_commands() != 0 and batch.encodeRagged(members[0..count], rows, arena, runtimes[0].model, runtimes[0].weights);
    if (c.ds4_gpu_end_commands() == 0) ok = false;
    for (runtimes) |*runtime| {
        const graph = &runtime.state.*.graph;
        if (graph.snap_after_first) {
            graph.snap_valid = ok;
            graph.snap_after_first = false;
        }
    }
    if (!ok) return session.fail(errors, "metal speculative batch failed");
    for (runtimes, 0..) |*runtime, index| {
        const state = runtime.state;
        const graph = &state.*.graph;
        const frontier = runtime.frontier;
        const member = &members[index];
        const output = session.logits(state);
        if (c.ds4_gpu_tensor_read(arena.batch_logits, @as(u64, member.row0) * vocab * @sizeOf(f32), output, @as(u64, member.n) * vocab * @sizeOf(f32)) == 0) return session.fail(errors, "metal speculative batch: logits read failed");
        c.ds4_tokens_push(&frontier.*.checkpoint, member.tokens[0]);
        committed[index] = 1;
        if (member.n == 2) {
            state.*.spec_cycles += 1;
            const accept = c.ds4_sampling_argmax(output, vocab) == member.tokens[1] or session.forceAccept();
            session.noteFirst(state, accept);
            drafted += 1;
            accepted_drafts += @intFromBool(accept);
            if (accept) {
                c.ds4_tokens_push(&frontier.*.checkpoint, member.tokens[1]);
                @memcpy(frontier.*.logits[0..vocab], output[vocab..][0..vocab]);
                graph.pos = positions[index] + 2;
                state.*.spec_accepted += 1;
                committed[index] = 2;
            } else {
                if (!graph.snap_valid or !snapshots.swap(graph, .first)) return session.fail(errors, "Qwen3.8 mtp: rejection restore failed");
                @memcpy(frontier.*.logits[0..vocab], output[0..vocab]);
            }
            if (session.trace()) _ = c.fprintf(c.__stderrp, "ds4: spec batch pos %u token %d draft %d %s\n", positions[index], member.tokens[0], member.tokens[1], @as([*:0]const u8, if (accept) "accept" else "reject"));
        } else {
            @memcpy(frontier.*.logits[0..vocab], output[0..vocab]);
            graph.pos = positions[index] + 1;
        }
        frontier.*.checkpoint_valid = true;
        frontier.*.mtp_draft_valid = false;
        state.*.rewound = false;
        parents[index] = c.ds4_sampling_argmax(frontier.*.logits, vocab);
        counts[index] = @intCast(committed[index]);
        accepted[index] = .{ member.tokens[0], if (committed[index] == 2) member.tokens[1] else -1 };
    }
    if (drafted != 0) {
        policy.p = ema(policy.p, @as(f32, @floatFromInt(accepted_drafts)) / @as(f32, @floatFromInt(drafted)), policy.spec_cycles);
        policy.spec_cycles += 1;
    }
    var room = next(policy) and c.getenv("DS4_QWEN4_NO_BATCH_DRAFT") == null;
    for (runtimes) |*runtime| {
        const graph = &runtime.state.*.graph;
        if (graph.pos + 2 > graph.ctx_cap) room = false;
    }
    if (room) {
        if (!batch.drafts(members[0..count], committed[0..count], parents[0..count], positions[0..count], rows, arena, runtimes[0].model, runtimes[0].weights, drafts[0..count])) {
            _ = c.ds4_gpu_end_commands();
            return session.fail(errors, "metal speculative batch: predictor failed");
        }
        for (runtimes, 0..) |*runtime, index| {
            runtime.pending.*.draft = drafts[index];
            runtime.pending.*.parent = parents[index];
            runtime.pending.*.have = 1;
        }
    }
    // Transition cycles price neither schedule: only plain/plain or
    // verify+predictor together are comparable steady-cycle costs.
    if ((drafted != 0) == room) {
        const cost = &policy.ms[@intFromBool(drafted != 0)];
        cost.* = ema(cost.*, @floatCast((ops.now() - started) * 1e3), if (cost.* == 0) 0 else 16);
    }
    if (session.trace()) _ = c.fprintf(c.__stderrp, "ds4: spec batch policy: cycle %u %s, p %.2f plain %.1f ms spec %.1f ms, next %s\n", policy.cycles, @as([*:0]const u8, if (drafted != 0) "speculative" else "plain"), @as(f64, policy.p), @as(f64, policy.ms[0]), @as(f64, policy.ms[1]), @as([*:0]const u8, if (room) "drafts" else "plain"));
    policy.cycles += 1;
    return 0;
}
