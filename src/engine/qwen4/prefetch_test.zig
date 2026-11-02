//! Exact frontier/continuation parity for native PLE prefetch. The model-free
//! mode checks that invalid chunks cannot touch embedding or recurrent state.
const std = @import("std");
const c = @import("gguf_c");
const input = @import("input.zig");
const chunk = 256;
const prompt_count = 2 * chunk + 17;
const frontiers = 3;

fn check(ok: bool) void {
    if (!ok) c.abort();
}

const Progress = struct {
    session: *c.ds4_session,
    expected: []f32,
    scratch: []f32,
    calls: usize = 0,
    control: bool,
    cancel_after_first: bool = false,
    cancelled: bool = false,
};
fn report(context: ?*anyopaque, event: [*c]const u8, current: c_int, total: c_int) callconv(.c) void {
    if (c.strcmp(event, "prefill_chunk") != 0) return;
    const state: *Progress = @ptrCast(@alignCast(context.?));
    const positions = [_]c_int{ chunk, 2 * chunk, prompt_count };
    check(state.calls < positions.len and current == positions[state.calls] and total == prompt_count);
    check(c.ds4_session_pos(state.session) == current);
    check(c.ds4_session_copy_logits(state.session, state.scratch.ptr, @intCast(state.scratch.len)) == state.scratch.len);
    for (state.scratch) |value| check(std.math.isFinite(value));
    const target = state.expected[state.calls * state.scratch.len ..][0..state.scratch.len];
    if (state.control) @memcpy(target, state.scratch) else check(std.mem.eql(u8, std.mem.sliceAsBytes(target), std.mem.sliceAsBytes(state.scratch)));
    state.calls += 1;
    if (state.cancel_after_first and state.calls == 1) state.cancelled = true;
}
fn cancelled(context: ?*anyopaque) callconv(.c) bool {
    const state: *Progress = @ptrCast(@alignCast(context.?));
    return state.cancelled;
}
fn invalidStaging() void {
    var graph = std.mem.zeroes(c.ds4_qwen4_gpu_graph);
    graph.config.shape.n_vocab = 100;
    var model = std.mem.zeroes(c.ds4_model);
    var weights = std.mem.zeroes(c.ds4_weights);
    const previous = graph.ple_prev;
    const prefetched = [_]f32{0};
    for ([_]c_int{ -1, 100 }) |invalid| {
        check(!input.stageWithPle(&graph, &model, &weights, &.{ 0, invalid }, null));
        check(!input.stageWithPle(&graph, &model, &weights, &.{ 0, invalid }, &prefetched));
        check(graph.pos == 0 and graph.mrope_delta == 0 and std.mem.eql(c_int, &previous, &graph.ple_prev));
    }
    _ = c.puts("Qwen invalid tokens rejected before input staging: OK");
}

pub export fn main(argc: c_int, argv: [*c][*c]u8) c_int {
    invalidStaging();
    if (argc == 2 and c.strcmp(argv[1], "--invalid-tokens") == 0) return 0;
    if (argc != 2) {
        _ = c.fprintf(c.__stderrp, "usage: test-qwen4-ple-prefetch QWEN_GGUF | --invalid-tokens\n");
        return 2;
    }
    var options = std.mem.zeroes(c.ds4_engine_options);
    options.model_path = argv[1];
    options.context_size = prompt_count + 8;
    options.prefill_chunk = chunk;
    options.backend = c.DS4_BACKEND_METAL;
    var engine: ?*c.ds4_engine = null;
    check(c.ds4_engine_open(&engine, &options) == 0);
    defer c.ds4_engine_close(engine);
    check(c.ds4_engine_is_qwen4(engine));
    var tokens = std.mem.zeroes(c.ds4_tokens);
    c.ds4_tokenize_text(engine, "The quick brown fox counts 17 green apples, then 29 blue boats.\n", &tokens);
    defer c.ds4_tokens_free(&tokens);
    const seed = tokens.len;
    check(seed > 0 and seed < chunk);
    while (tokens.len < prompt_count) c.ds4_tokens_push(&tokens, tokens.v[@intCast(@mod(tokens.len, seed))]);
    const vocabulary: usize = @intCast(c.ds4_engine_vocab_size(engine));
    const expected = std.heap.c_allocator.alloc(f32, (frontiers + 1) * vocabulary) catch c.abort();
    defer std.heap.c_allocator.free(expected);
    const scratch = std.heap.c_allocator.alloc(f32, vocabulary) catch c.abort();
    defer std.heap.c_allocator.free(scratch);
    var next: c_int = -1;
    for (0..3) |mode| {
        if (mode == 0) check(c.setenv("DS4_QWEN4_NO_PLE_PREFETCH", "1", 1) == 0) else check(c.unsetenv("DS4_QWEN4_NO_PLE_PREFETCH") == 0);
        var session: ?*c.ds4_session = null;
        check(c.ds4_session_create(&session, engine, prompt_count + 8) == 0);
        defer c.ds4_session_free(session);
        var progress = Progress{ .session = session.?, .expected = expected, .scratch = scratch, .control = mode == 0, .cancel_after_first = mode == 2 };
        c.ds4_session_set_progress(session, report, &progress);
        c.ds4_session_set_cancel(session, cancelled, &progress);
        var errors: [256]u8 = @splat(0);
        var result = c.ds4_session_sync(session, &tokens, &errors, errors.len);
        if (mode == 2) {
            check(result == c.DS4_SESSION_SYNC_INTERRUPTED and progress.calls == 1);
            check(c.ds4_session_pos(session) == chunk);
            progress.cancelled = false;
            result = c.ds4_session_sync(session, &tokens, &errors, errors.len);
        }
        if (result != 0) _ = c.fprintf(c.__stderrp, "%s\n", &errors);
        check(result == 0 and progress.calls == frontiers);
        c.ds4_session_set_progress(session, null, null);
        c.ds4_session_set_cancel(session, null, null);
        if (mode == 0) next = c.ds4_session_argmax(session);
        check(c.ds4_session_eval(session, next, &errors, errors.len) == 0);
        check(c.ds4_session_copy_logits(session, scratch.ptr, @intCast(vocabulary)) == vocabulary);
        const decode = expected[frontiers * vocabulary ..];
        if (mode == 0) @memcpy(decode, scratch) else check(std.mem.eql(u8, std.mem.sliceAsBytes(decode), std.mem.sliceAsBytes(scratch)));
        _ = c.printf("Qwen PLE mode %zu: exact chunk and decode logits OK\n", mode);
    }
    for ([_]usize{ 0, chunk, prompt_count - 1 }, 0..) |position, index| {
        var session: ?*c.ds4_session = null;
        check(c.ds4_session_create(&session, engine, prompt_count + 8) == 0);
        defer c.ds4_session_free(session);
        const original = tokens.v[position];
        tokens.v[position] = if (index == 1) @intCast(vocabulary) else -1;
        var errors: [256]u8 = @splat(0);
        check(c.ds4_session_sync(session, &tokens, &errors, errors.len) == 1);
        tokens.v[position] = original;
        check(c.ds4_session_sync(session, &tokens, &errors, errors.len) == 0);
        check(c.ds4_session_eval(session, next, &errors, errors.len) == 0);
        check(c.ds4_session_copy_logits(session, scratch.ptr, @intCast(vocabulary)) == vocabulary);
        check(std.mem.eql(u8, std.mem.sliceAsBytes(expected[frontiers * vocabulary ..]), std.mem.sliceAsBytes(scratch)));
    }
    _ = c.puts("Qwen PLE invalid-token recovery: exact logits OK");
    return 0;
}
