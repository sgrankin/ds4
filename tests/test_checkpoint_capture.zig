//! Live oracle for checkpoints assembled inside one DeepSeek prefill sweep.
//! Run beneath the existing pressure guard with installed weights. Numerical
//! differences from changing batch shapes are reported rather than rejected.
const std = @import("std");
const c = @import("checkpoint_capture_c");
const allocator = std.heap.c_allocator;
const endpoints = [_]c_int{ 127, 128, 129, 390, 985, 2247 };
const teacher_steps = 2;
var errors: [512]u8 = @splat(0);

fn require(ok: bool, message: []const u8) !void {
    if (!ok) {
        std.debug.print("checkpoint capture: {s}: {s}\n", .{ message, std.mem.sliceTo(&errors, 0) });
        return error.CheckFailed;
    }
}

fn copyLogits(session: *c.ds4_session, out: []f32) !void {
    try require(c.ds4_session_checkpoint_valid(session), "valid logit frontier");
    try require(c.ds4_session_copy_logits(session, out.ptr, @intCast(out.len)) == @as(c_int, @intCast(out.len)), "complete logit row");
    for (out) |value| try require(std.math.isFinite(value), "finite logits");
}

fn logZ(logits: []const f32) f64 {
    var maximum: f64 = -std.math.inf(f64);
    for (logits) |value| maximum = @max(maximum, value);
    var sum: f64 = 0;
    for (logits) |value| sum += @exp(@as(f64, value) - maximum);
    return maximum + @log(sum);
}

fn compare(reference: []const f32, actual: []const f32, token: c_int, position: c_int, step: usize) !void {
    const reference_z = logZ(reference);
    const actual_z = logZ(actual);
    var maximum: f64 = 0;
    var kl: f64 = 0;
    for (reference, actual) |left, right| {
        maximum = @max(maximum, @abs(@as(f64, left) - right));
        const p = @as(f64, left) - reference_z;
        const q = @as(f64, right) - actual_z;
        kl += @exp(p) * (p - q);
    }
    const target: usize = @intCast(token);
    const delta_nll = (actual_z - actual[target]) - (reference_z - reference[target]);
    try require(std.math.isFinite(kl) and std.math.isFinite(delta_nll), "finite teacher metrics");
    std.debug.print("checkpoint_row position={d} step={d} max_delta={d:.9} delta_nll={d:.9} kl={d:.9} byte_equal={d}\n", .{
        position, step, maximum, delta_nll, @max(kl, 0), @intFromBool(std.mem.eql(u8, std.mem.sliceAsBytes(reference), std.mem.sliceAsBytes(actual))),
    });
}

fn sync(session: *c.ds4_session, prompt: *const c.ds4_tokens, length: c_int) !void {
    var prefix = prompt.*;
    prefix.len = length;
    try require(c.ds4_session_sync(session, &prefix, &errors, errors.len) == 0, "ordinary prefix sync");
    try require(c.ds4_session_pos(session) == length, "ordinary prefix position");
}

fn restore(session: *c.ds4_session, snapshot: *const c.ds4_suffix_cache, prompt: *const c.ds4_tokens, position: c_int) !void {
    try require(c.ds4_session_load_suffix_prefix(session, snapshot, &errors, errors.len) == 0, "restore collected prefix and originals");
    try require(c.ds4_session_pos(session) == position and c.ds4_session_checkpoint_valid(session) and !c.ds4_session_is_approximate(session), "restored exact frontier");
    const history = c.ds4_session_tokens(session);
    try require(history != null and history.*.len == position, "restored token count");
    const length: usize = @intCast(position);
    try require(std.mem.eql(c_int, history.*.v[0..length], prompt.v[0..length]), "restored prefix tokens");
}

fn bytes(snapshot: c.ds4_session_snapshot) []const u8 {
    return snapshot.ptr[0..@intCast(snapshot.len)];
}

fn originals(snapshot: c.ds4_suffix_cache) []const u8 {
    if (snapshot.prerope_bytes == 0) return &.{};
    const ptr: [*]const u8 = @ptrCast(snapshot.prerope.?);
    return ptr[0..@intCast(snapshot.prerope_bytes)];
}

fn roundtrip(session: *c.ds4_session, source: *const c.ds4_suffix_cache) !void {
    var recaptured = std.mem.zeroes(c.ds4_suffix_cache);
    defer c.ds4_suffix_cache_free(&recaptured);
    try require(c.ds4_session_capture_suffix_cache(session, &recaptured, &errors, errors.len) == 0, "recapture restored prefix");
    try require(!recaptured.approximate and std.mem.eql(u8, bytes(source.state), bytes(recaptured.state)), "canonical bytes survive restore/recapture");
    try require(originals(source.*).len > 0 and std.mem.eql(u8, originals(source.*), originals(recaptured)), "original rotary coverage survives restore/recapture");
}

fn append(tokens: *c.ds4_tokens, values: []const c_int) void {
    for (values) |token| c.ds4_tokens_push(tokens, token);
}

fn relocate(session: *c.ds4_session, source: *const c.ds4_suffix_cache, anchor: *const c.ds4_suffix_cache, prompt: *const c.ds4_tokens, source_start: c_int, destination_start: c_int) !void {
    const source_end = endpoints[endpoints.len - 1];
    const state_hash = std.hash.Wyhash.hash(0, bytes(source.state));
    const original_hash = std.hash.Wyhash.hash(0, originals(source.*));
    var shifted = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&shifted);
    append(&shifted, prompt.v[0..@intCast(destination_start)]);
    append(&shifted, prompt.v[@intCast(source_start)..@intCast(source_end)]);
    append(&shifted, prompt.v[@intCast(source_end)..@intCast(source_end + teacher_steps)]);
    try restore(session, anchor, prompt, destination_start);
    var stats = std.mem.zeroes(c.ds4_suffix_relocation_stats);
    try require(c.ds4_session_reuse_suffix_relocated(session, source, &shifted, @intCast(source_start), &stats, &errors, errors.len) == 0, "relocate collected source with original rotary values");
    try require(c.ds4_session_checkpoint_valid(session) and c.ds4_session_pos(session) == shifted.len and c.ds4_session_is_approximate(session), "relocated live frontier");
    try require(stats.source_start == source_start and stats.source_end == source_end and stats.destination_start == destination_start and stats.destination_end == destination_start + source_end - source_start, "relocation geometry");
    try require(stats.reuse.reused_tokens == source_end - source_start and stats.reuse.fresh_tail_tokens == teacher_steps and stats.reuse.pending_pairs == 3, "odd-endpoint pending compressors and fresh tail");
    var recaptured = std.mem.zeroes(c.ds4_suffix_cache);
    defer c.ds4_suffix_cache_free(&recaptured);
    try require(c.ds4_session_capture_suffix_cache(session, &recaptured, &errors, errors.len) == 0 and recaptured.approximate and recaptured.prerope_bytes > 0, "relocated private source remains capturable");
    try require(std.hash.Wyhash.hash(0, bytes(source.state)) == state_hash and std.hash.Wyhash.hash(0, originals(source.*)) == original_hash, "repeated relocations leave original source immutable");
    std.debug.print("checkpoint_relocation source={d}:{d} destination={d}:{d} pending_pairs={d} originals_immutable=1\n", .{
        stats.source_start, stats.source_end, stats.destination_start, stats.destination_end, stats.reuse.pending_pairs,
    });
}

fn usage(program: []const u8) noreturn {
    std.debug.print("usage: {s} MODEL [--prefill-chunk N] [--cache-gb N]\nRun under pressure_run.py; use main/index FP4 environment settings to qualify packed storage.\n", .{program});
    std.process.exit(2);
}

fn run(args: std.process.Args) !void {
    var iterator = args.iterate();
    const program = iterator.next() orelse "test-checkpoint-capture";
    const model = iterator.next() orelse usage(program);
    if (std.mem.eql(u8, model, "--help")) usage(program);
    var options = std.mem.zeroes(c.ds4_engine_options);
    options.model_path = model.ptr;
    options.backend = c.DS4_BACKEND_METAL;
    options.context_size = 8192;
    options.prefill_chunk = 2048;
    options.power_percent = 100;
    options.ssd_streaming = true;
    options.ssd_streaming_cache_bytes = @as(u64, 32) << 30;
    while (iterator.next()) |flag| {
        const value = iterator.next() orelse usage(program);
        if (std.mem.eql(u8, flag, "--prefill-chunk")) {
            options.prefill_chunk = std.fmt.parseInt(u32, value, 10) catch usage(program);
        } else if (std.mem.eql(u8, flag, "--cache-gb")) {
            const gb = std.fmt.parseInt(u64, value, 10) catch usage(program);
            options.ssd_streaming_cache_bytes = std.math.mul(u64, gb, 1 << 30) catch usage(program);
        } else usage(program);
    }
    var engine: ?*c.ds4_engine = null;
    try require(c.ds4_engine_open(&engine, &options) == 0 and engine != null, "open installed model");
    defer c.ds4_engine_close(engine);
    try require(c.ds4_engine_is_deepseek41(engine), "DeepSeek4.1 model");
    var session: ?*c.ds4_session = null;
    try require(c.ds4_session_create(&session, engine, options.context_size) == 0 and session != null, "create one mutable session");
    defer c.ds4_session_free(session);
    const live = session.?;
    try require(c.ds4_session_enable_suffix_prerope(live, &errors, errors.len) == 0, "enable original rotary archive");

    var filler = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&filler);
    c.ds4_tokenize_text(engine, "Parser investigation: retained records explain each decision, preserve the code CEDAR-4826, and remain available for later comparison.\n", &filler);
    try require(filler.len > 0, "filler tokenization");
    var prompt = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&prompt);
    c.ds4_chat_begin(engine, &prompt);
    c.ds4_tokens_push(&prompt, c.ds4_token_user(engine));
    var index: usize = 0;
    while (prompt.len < 3072) : (index += 1) c.ds4_tokens_push(&prompt, filler.v[index % @as(usize, @intCast(filler.len))]);
    c.ds4_chat_append_assistant_prefix(engine, &prompt, c.DS4_THINK_HIGH);
    try require(prompt.len > endpoints[endpoints.len - 1] + teacher_steps and prompt.len < options.context_size, "bounded reasoning-enabled prompt");

    const vocabulary: usize = @intCast(c.ds4_engine_vocab_size(engine));
    try require(vocabulary > 0, "vocabulary");
    const ordinary_final = try allocator.alloc(f32, vocabulary);
    defer allocator.free(ordinary_final);
    const actual = try allocator.alloc(f32, vocabulary);
    defer allocator.free(actual);
    const reference = try allocator.alloc(f32, endpoints.len * (teacher_steps + 1) * vocabulary);
    defer allocator.free(reference);

    // First retain an ordinary uninterrupted sync's final logits. Then build
    // independent endpoint teachers monotonically, restoring before advancing
    // so none of those checks recomputes a complete prefix from token zero.
    try sync(live, &prompt, prompt.len);
    try copyLogits(live, ordinary_final);
    c.ds4_session_invalidate(live);
    for (endpoints, 0..) |position, endpoint| {
        try sync(live, &prompt, position);
        var saved = std.mem.zeroes(c.ds4_suffix_cache);
        defer c.ds4_suffix_cache_free(&saved);
        try require(c.ds4_session_capture_suffix_cache(live, &saved, &errors, errors.len) == 0, "capture independent endpoint teacher");
        for (0..teacher_steps + 1) |step| {
            const offset = (endpoint * (teacher_steps + 1) + step) * vocabulary;
            try copyLogits(live, reference[offset..][0..vocabulary]);
            if (step < teacher_steps) try require(c.ds4_session_eval(live, prompt.v[@as(usize, @intCast(position)) + step], &errors, errors.len) == 0, "teacher continuation");
        }
        try restore(live, &saved, &prompt, position);
    }
    c.ds4_session_invalidate(live);
    var captured: [endpoints.len]c.ds4_suffix_cache = @splat(std.mem.zeroes(c.ds4_suffix_cache));
    defer for (&captured) |*snapshot| c.ds4_suffix_cache_free(snapshot);
    try require(c.ds4_session_sync_checkpoints(live, &prompt, &endpoints, endpoints.len, &captured, &errors, errors.len) == 0, "collect all endpoints inside prefill");
    try require(c.ds4_session_pos(live) == prompt.len and !c.ds4_session_is_approximate(live), "collection preserves final frontier");
    try copyLogits(live, actual);
    try compare(ordinary_final, actual, c.ds4_session_argmax(live), prompt.len, 0);
    try require(std.mem.eql(u8, std.mem.sliceAsBytes(ordinary_final), std.mem.sliceAsBytes(actual)), "collection leaves ordinary final logits byte-identical");
    for (endpoints, &captured, 0..) |position, *snapshot, endpoint| {
        try require(snapshot.state.len > 52 and snapshot.engine == engine and snapshot.ctx_size == options.context_size and !snapshot.approximate, "complete canonical endpoint ownership");
        try restore(live, snapshot, &prompt, position);
        try roundtrip(live, snapshot);
        for (0..teacher_steps + 1) |step| {
            try copyLogits(live, actual);
            const offset = (endpoint * (teacher_steps + 1) + step) * vocabulary;
            const target = prompt.v[@as(usize, @intCast(position)) + step];
            try compare(reference[offset..][0..vocabulary], actual, target, position, step);
            if (step < teacher_steps) try require(c.ds4_session_eval(live, target, &errors, errors.len) == 0, "restored continuation");
        }
        std.debug.print("checkpoint_restore position={d} state_bytes={d} original_bytes={d} canonical_roundtrip=1 originals_roundtrip=1\n", .{ position, snapshot.state.len, snapshot.prerope_bytes });
    }
    // Reuse the same immutable odd endpoint for a left shift and a right
    // shift. Existing collected anchors avoid unrelated prefix prefill work.
    try relocate(live, &captured[5], &captured[1], &prompt, 390, 128);
    try relocate(live, &captured[5], &captured[3], &prompt, 128, 390);
    // Approximate state uses the server's existing segmented fallback. Supply
    // a genuine prefix extension so that rejection protects this live state,
    // rather than merely rejecting an unrelated or malformed token history.
    var extension = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&extension);
    c.ds4_tokens_copy(&extension, c.ds4_session_tokens(live));
    c.ds4_tokens_push(&extension, prompt.v[0]);
    const before = c.ds4_session_pos(live);
    try require(c.ds4_session_is_approximate(live), "relocation creates unsupported approximate source");
    const approximate_logits = try allocator.alloc(f32, vocabulary);
    defer allocator.free(approximate_logits);
    try copyLogits(live, approximate_logits);
    const unsupported_end = [_]c_int{extension.len};
    var unsupported = [_]c.ds4_suffix_cache{std.mem.zeroes(c.ds4_suffix_cache)};
    defer c.ds4_suffix_cache_free(&unsupported[0]);
    try require(c.ds4_session_sync_checkpoints(live, &extension, &unsupported_end, 1, &unsupported, &errors, errors.len) == -1, "approximate state requests ordinary fallback");
    try require(c.ds4_session_pos(live) == before and c.ds4_session_checkpoint_valid(live) and c.ds4_session_is_approximate(live), "unsupported collection preserves live frontier");
    try copyLogits(live, actual);
    try require(std.mem.eql(u8, std.mem.sliceAsBytes(approximate_logits), std.mem.sliceAsBytes(actual)), "unsupported collection preserves logits");
    const rejected = unsupported[0];
    try require(rejected.state.ptr == null and rejected.state.len == 0 and rejected.state.cap == 0 and rejected.engine == null and rejected.ctx_size == 0 and rejected.prerope == null and rejected.prerope_bytes == 0 and !rejected.approximate, "unsupported collection publishes no endpoint");
    std.debug.print("checkpoint_fallback approximate=1 rejected_before_mutation=1 outputs=0 logits_identical=1\n", .{});
    // Replacing that approximate transcript has no live prefix to preserve.
    // Rebuild in the normal sweep while collecting raw-ring and odd boundaries.
    const rewrite_ends = [_]c_int{ 128, 2247 };
    var rewritten: [rewrite_ends.len]c.ds4_suffix_cache = @splat(std.mem.zeroes(c.ds4_suffix_cache));
    defer for (&rewritten) |*snapshot| c.ds4_suffix_cache_free(snapshot);
    try require(c.ds4_session_sync_checkpoints(live, &prompt, &rewrite_ends, rewrite_ends.len, &rewritten, &errors, errors.len) == 0, "collect fresh rewrite over approximate live state");
    try require(c.ds4_session_pos(live) == prompt.len and c.ds4_session_checkpoint_valid(live) and !c.ds4_session_is_approximate(live), "rewrite restores ordinary exact frontier");
    try copyLogits(live, actual);
    try require(std.mem.eql(u8, std.mem.sliceAsBytes(ordinary_final), std.mem.sliceAsBytes(actual)), "rewrite preserves original ordinary final logits");
    for (rewrite_ends, &rewritten) |position, *snapshot| {
        try require(snapshot.state.len > 52 and snapshot.engine == engine and !snapshot.approximate, "rewrite publishes canonical endpoint");
        try restore(live, snapshot, &prompt, position);
        try roundtrip(live, snapshot);
    }
    std.debug.print("checkpoint_rewrite approximate=1 reset=1 collected=2 logits_identical=1 canonical_roundtrip=1\n", .{});
    std.debug.print("checkpoint capture: PASS endpoints={d} teacher_rows={d} relocations=2 reasoning=high\n", .{ endpoints.len, endpoints.len * (teacher_steps + 1) });
}

pub fn main(init: std.process.Init.Minimal) !void {
    run(init.args) catch |failure| {
        if (failure != error.CheckFailed) std.debug.print("checkpoint capture: {s}\n", .{@errorName(failure)});
        std.process.exit(1);
    };
}
