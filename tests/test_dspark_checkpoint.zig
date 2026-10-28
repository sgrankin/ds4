//! Installed-model numerical oracle. No downloads, no second engine, at most
//! two sessions. Public checkpoint APIs prove old/new framing; an EOS-excluding
//! matcher proves the first restored cycle really drafts at positive temperature.
const std = @import("std");
const c = @import("checkpoint_c");
extern "c" fn setenv([*:0]const u8, [*:0]const u8, c_int) c_int;
const a = std.heap.c_allocator;
const flag = "DS4_V41_CHECKPOINT_DRAFT_FEATURES";
var errors: [512]u8 = @splat(0);
const profile = "dspark-checkpoint-numerical-v1";

fn require(ok: bool, message: []const u8) !void {
    if (!ok) {
        std.debug.print("draft checkpoint: {s}: {s}\n", .{ message, std.mem.sliceTo(&errors, 0) });
        return error.CheckFailed;
    }
}

const Walk = struct {
    masks: usize = 0,
    count: usize = 1, // Caller already accepted the first token.
    eos: u32,

    fn from(context: ?*anyopaque) *Walk {
        return @ptrCast(@alignCast(context.?));
    }
    fn mask(context: ?*anyopaque, words: [*c]u32, count: usize) callconv(.c) c_int {
        const self = from(context);
        @memset(words[0..count], std.math.maxInt(u32));
        words[self.eos / 32] &= ~(@as(u32, 1) << @intCast(self.eos % 32));
        self.masks += 1;
        return 0;
    }
    fn accept(context: ?*anyopaque, token: c_int) callconv(.c) c_int {
        const self = from(context);
        if (token < 0 or token == @as(c_int, @intCast(self.eos))) return -1;
        self.count += 1;
        return 0;
    }
    fn rollback(context: ?*anyopaque, count: usize) callconv(.c) c_int {
        const self = from(context);
        if (count >= self.count) return -1;
        self.count -= count;
        return 0;
    }
};

fn cycle(session: *c.ds4_session, engine: *c.ds4_engine, should_draft: bool, reference: ?*c.ds4_session) !void {
    const vocab: usize = @intCast(c.ds4_engine_vocab_size(engine));
    var walk = Walk{ .eos = @intCast(c.ds4_token_eos(engine)) };
    const seed_logits = try a.alloc(f32, vocab);
    defer a.free(seed_logits);
    try require(c.ds4_session_copy_sampling_logits(session, seed_logits.ptr, @intCast(vocab)) == @as(c_int, @intCast(vocab)), "sampling frontier");
    seed_logits[walk.eos] = -std.math.inf(f32);
    var rng: u64 = 0x1234ab56cd;
    const first = c.ds4_sample_logits(seed_logits.ptr, @intCast(vocab), 1, 0, 0.95, 0.05, &rng);
    try require(first >= 0, "positive-temperature masked seed");
    const callbacks = c.ds4_token_constraint{ .context = &walk, .mask = Walk.mask, .accept = Walk.accept, .rollback = Walk.rollback };
    var accepted: [6]c_int = undefined;
    const count = c.ds4_session_eval_speculative_constrained(session, first, 6, @intCast(walk.eos), 1, 0, 0.95, 0.05, &rng, &callbacks, &accepted, accepted.len, &errors, errors.len);
    try require(count > 0, "masked cycle succeeds");
    try require((walk.masks > 0) == should_draft, "actual first-cycle draft readiness");
    try require(walk.count == @as(usize, @intCast(count)), "matcher tracks exactly committed tokens");
    if (reference) |teacher| {
        for (accepted[0..@intCast(count)]) |token|
            try require(c.ds4_session_eval(teacher, token, &errors, errors.len) == 0, "scalar teacher-forcing accepted tokens");
        const left = try a.alloc(f32, vocab);
        defer a.free(left);
        const right = try a.alloc(f32, vocab);
        defer a.free(right);
        try require(c.ds4_session_copy_logits(session, left.ptr, @intCast(vocab)) == @as(c_int, @intCast(vocab)), "restored frontier logits");
        try require(c.ds4_session_copy_logits(teacher, right.ptr, @intCast(vocab)) == @as(c_int, @intCast(vocab)), "teacher frontier logits");
        var maximum: f64 = 0;
        for (left, right) |x, y| {
            try require(std.math.isFinite(x) and std.math.isFinite(y), "finite logits");
            maximum = @max(maximum, @abs(@as(f64, x) - y));
        }
        std.debug.print("draft checkpoint: first-cycle masks={d} committed={d} teacher_max={d}\n", .{ walk.masks, count, maximum });
        try require(maximum < 0.01, "restored MTP teacher agreement");
    }
}

fn preferred(cache: *c.ds4_state_cache, tokens: *const c.ds4_tokens, saved: *const c.ds4_session_snapshot) !c.ds4_state_cache_blob {
    var matched: u32 = 0;
    const ref = c.ds4_state_cache_find_prefix(cache, profile, 2048, tokens.v, @intCast(tokens.len), &matched);
    defer if (ref != null) c.ds4_state_cache_release(ref);
    try require(ref != null and matched == tokens.len, "find complete SSD prefix");
    var blob = std.mem.zeroes(c.ds4_state_cache_blob);
    try require(c.ds4_state_cache_materialize(ref, &blob, &errors, errors.len) == 0, "materialize preferred endpoint");
    errdefer c.ds4_state_cache_blob_free(&blob);
    try require(blob.state_bytes == saved.len, "DSP1 wins equal-prefix tie against base");
    try require(std.mem.eql(u8, blob.state[0..@intCast(blob.state_bytes)], saved.ptr[0..@intCast(saved.len)]), "SSD checkpoint bytes exactly round-trip");
    var stats = std.mem.zeroes(c.ds4_state_cache_stats);
    c.ds4_state_cache_get_stats(cache, &stats);
    try require(stats.hot_bytes == 0 and stats.hot_reads == 0 and stats.disk_reads > 0, "HOT0 actually reads SSD");
    return blob;
}

pub fn main(init: std.process.Init.Minimal) !void {
    var args = init.args.iterate();
    const program = args.next() orelse "test-dspark-checkpoint";
    const model = args.next() orelse {
        std.debug.print("usage: {s} TARGET SUPPORT EMPTY_CACHE_DIR [--packed-cross|--main-packed-cross|--all-packed-cross]\n", .{program});
        return error.MissingModel;
    };
    const support = args.next() orelse return error.MissingSupport;
    const cache_dir = args.next() orelse return error.MissingCacheDirectory;
    const option = args.next();
    const cross_index = option != null and (std.mem.eql(u8, option.?, "--packed-cross") or std.mem.eql(u8, option.?, "--all-packed-cross"));
    const cross_main = option != null and (std.mem.eql(u8, option.?, "--main-packed-cross") or std.mem.eql(u8, option.?, "--all-packed-cross"));
    try require(option == null or cross_index or cross_main, "unknown option");
    const packed_cross = option != null;
    const packed_version: u32 = 2 + 2 * @as(u32, @intFromBool(cross_main)) + @as(u32, @intFromBool(cross_index));
    try require(args.next() == null, "unexpected extra argument");
    // Presence enables the existing actual GPU tensor-byte log; physical main
    // and index layouts are selected per graph below for cross-import cases.
    try require(setenv("DS4_V41_COMPRESSED_KV_F16", "0", 1) == 0, "request actual allocation log");
    var options = std.mem.zeroes(c.ds4_engine_options);
    options.model_path = model.ptr;
    options.mtp_path = support.ptr;
    options.backend = c.DS4_BACKEND_METAL;
    options.context_size = 2048;
    options.prefill_chunk = 512;
    options.power_percent = 100;
    options.ssd_streaming = true;
    options.ssd_streaming_cache_bytes = 40 << 30;
    options.dspark = true;
    options.dspark_exact_sampling = true;
    options.dspark_confidence_threshold_set = true;
    options.dspark_confidence_threshold = 0;
    options.mtp_draft_tokens = 6;
    options.placement_session_count_hint = 2;
    options.share_session_prefill_workspace = true;
    var engine: ?*c.ds4_engine = null;
    try require(c.ds4_engine_open(&engine, &options) == 0, "open target and drafter");
    defer c.ds4_engine_close(engine);
    try require(c.ds4_engine_decode_packet_size(engine) > 1, "support enables actual packets");
    var source: ?*c.ds4_session = null;
    var restored: ?*c.ds4_session = null;
    if (packed_cross) {
        try require(setenv("DS4_V41_INDEX_FP4", "0", 1) == 0, "allocate F32 source index");
        try require(setenv("DS4_V41_COMPRESSED_KV_FP4", "0", 1) == 0, "allocate F32 source main");
    }
    try require(c.ds4_session_create(&source, engine, 2048) == 0, "create source");
    defer c.ds4_session_free(source);
    if (packed_cross) {
        try require(setenv("DS4_V41_INDEX_FP4", if (cross_index) "1" else "0", 1) == 0, "select restore index layout");
        try require(setenv("DS4_V41_COMPRESSED_KV_FP4", if (cross_main) "1" else "0", 1) == 0, "select restore main layout");
    }
    try require(c.ds4_session_create(&restored, engine, 2048) == 0, "create restore target");
    defer c.ds4_session_free(restored);
    var tokens = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&tokens);
    c.ds4_encode_chat_prompt(engine, null, "Count carefully from one to one hundred.", c.DS4_THINK_HIGH, &tokens);
    // Numerical ring-wrap case, not a behavioral-quality prompt.
    while (tokens.len < 257) c.ds4_tokens_push(&tokens, 300 + @mod(tokens.len, 23));
    try require(c.ds4_session_sync(source, &tokens, &errors, errors.len) == 0, "prefill complete wrapped feature window");
    var old = std.mem.zeroes(c.ds4_session_snapshot);
    defer c.ds4_session_snapshot_free(&old);
    var saved = std.mem.zeroes(c.ds4_session_snapshot);
    defer c.ds4_session_snapshot_free(&saved);
    try require(setenv(flag, "0", 1) == 0, "disable draft persistence");
    try require(c.ds4_session_save_snapshot(source, &old, &errors, errors.len) == 0, "save unchanged v2 base");
    try require(setenv(flag, "1", 1) == 0, "enable draft persistence");
    try require(c.ds4_session_save_snapshot(source, &saved, &errors, errors.len) == 0, "save DSP1");
    try require(saved.len == old.len + 32 + 3 * 128 * 5120 * 2, "bounded DSP1 wire size");
    try require(c.ds4_session_load_snapshot(restored, &old, &errors, errors.len) == 0, "old v2 accepted with flag on");
    try cycle(restored.?, engine.?, false, null);
    try require(setenv(flag, "0", 1) == 0, "disable restore optimization");
    try require(c.ds4_session_load_snapshot(restored, &saved, &errors, errors.len) == 0, "DSP1 consumed with flag off");
    try cycle(restored.?, engine.?, false, null);
    try require(setenv(flag, "1", 1) == 0, "reenable restore optimization");

    var cache = c.ds4_state_cache_open(cache_dir.ptr, 0, 512 << 20, &errors, errors.len);
    try require(cache != null, "open HOT0 SSD cache");
    defer if (cache != null) {
        _ = c.ds4_state_cache_close(cache, &errors, errors.len);
    };
    var stats = std.mem.zeroes(c.ds4_state_cache_stats);
    c.ds4_state_cache_get_stats(cache, &stats);
    try require(stats.endpoints == 0, "use an empty cache directory so DSP1-then-base forces traversal tie");
    var dsp_key: [41]u8 = undefined;
    var base_key: [41]u8 = undefined;
    try require(c.ds4_state_cache_put(cache, profile, saved.ptr, saved.len, null, 0, false, &dsp_key, &errors, errors.len) == 0, "publish DSP1 first");
    try require(c.ds4_state_cache_put(cache, profile, old.ptr, old.len, null, 0, false, &base_key, &errors, errors.len) == 0, "publish base at list head");
    try require(!std.mem.eql(u8, &dsp_key, &base_key), "two immutable manifests share one prefix");
    var before_restart = try preferred(cache.?, &tokens, &saved);
    c.ds4_state_cache_blob_free(&before_restart);
    try require(c.ds4_state_cache_close(cache, &errors, errors.len) == 0, "close with all references released");
    cache = null;
    cache = c.ds4_state_cache_open(cache_dir.ptr, 0, 512 << 20, &errors, errors.len);
    try require(cache != null, "reopen and discover both native manifests");
    c.ds4_state_cache_get_stats(cache, &stats);
    try require(stats.endpoints == 2, "both base and DSP1 survive restart");
    var from_ssd = try preferred(cache.?, &tokens, &saved);
    defer c.ds4_state_cache_blob_free(&from_ssd);
    const snapshot = c.ds4_session_snapshot{ .ptr = from_ssd.state, .len = from_ssd.state_bytes, .cap = from_ssd.state_bytes };
    try require(c.ds4_session_load_snapshot(restored, &snapshot, &errors, errors.len) == 0, "restore SSD DSP1 and wrapped ring tags");
    if (packed_cross) {
        // Isolate physical layout conversion before any restored inference. The
        // scalar teacher must resume from its original state after this probe.
        var packed_before_cycle = std.mem.zeroes(c.ds4_session_snapshot);
        defer c.ds4_session_snapshot_free(&packed_before_cycle);
        var expanded_before_cycle = std.mem.zeroes(c.ds4_session_snapshot);
        defer c.ds4_session_snapshot_free(&expanded_before_cycle);
        try require(c.ds4_session_save_snapshot(restored, &packed_before_cycle, &errors, errors.len) == 0, "save pre-cycle packed checkpoint");
        try require(c.ds4_session_load_snapshot(source, &packed_before_cycle, &errors, errors.len) == 0, "expand pre-cycle checkpoint into F32 source");
        try require(c.ds4_session_save_snapshot(source, &expanded_before_cycle, &errors, errors.len) == 0, "resave pre-cycle F32 checkpoint with DSP1");
        const original = saved.ptr[0..@intCast(saved.len)];
        const expanded_wire = expanded_before_cycle.ptr[0..@intCast(expanded_before_cycle.len)];
        var first_difference: ?usize = null;
        for (original[0..@min(original.len, expanded_wire.len)], expanded_wire[0..@min(original.len, expanded_wire.len)], 0..) |x, y, i| {
            if (x != y) {
                first_difference = i;
                break;
            }
        }
        std.debug.print("draft checkpoint: pre-cycle roundtrip v2->v{d}->v2 original_bytes={d} packed_bytes={d} expanded_bytes={d}", .{ packed_version, saved.len, packed_before_cycle.len, expanded_before_cycle.len });
        if (first_difference) |offset| {
            std.debug.print(" first_difference={d} original=0x{x:0>2} expanded=0x{x:0>2}\n", .{ offset, original[offset], expanded_wire[offset] });
        } else if (original.len != expanded_wire.len) {
            std.debug.print(" first_difference={d} (length)\n", .{@min(original.len, expanded_wire.len)});
        } else {
            std.debug.print(" exact\n", .{});
        }
        try require(c.ds4_session_load_snapshot(source, &saved, &errors, errors.len) == 0, "restore original scalar teacher after roundtrip probe");
        try require(original.len == expanded_wire.len and first_difference == null, "pre-cycle canonical roundtrip preserves every byte");

        // Run the same seed once without MTP, then restore both frontiers. A
        // scalar mismatch isolates publication/readers from batch verification.
        const vocab: usize = @intCast(c.ds4_engine_vocab_size(engine));
        const scalar_packed = try a.alloc(f32, vocab);
        defer a.free(scalar_packed);
        const scalar_plain = try a.alloc(f32, vocab);
        defer a.free(scalar_plain);
        try require(c.ds4_session_copy_sampling_logits(restored, scalar_packed.ptr, @intCast(vocab)) == @as(c_int, @intCast(vocab)), "pre-cycle scalar seed logits");
        scalar_packed[@intCast(c.ds4_token_eos(engine))] = -std.math.inf(f32);
        var scalar_rng: u64 = 0x1234ab56cd;
        const scalar_token = c.ds4_sample_logits(scalar_packed.ptr, @intCast(vocab), 1, 0, 0.95, 0.05, &scalar_rng);
        try require(scalar_token >= 0, "pre-cycle scalar sampled seed");
        try require(c.ds4_session_eval(restored, scalar_token, &errors, errors.len) == 0, "pre-cycle scalar FP4 decode");
        try require(c.ds4_session_eval(source, scalar_token, &errors, errors.len) == 0, "pre-cycle scalar F32 decode");
        try require(c.ds4_session_copy_logits(restored, scalar_packed.ptr, @intCast(vocab)) == @as(c_int, @intCast(vocab)), "pre-cycle scalar FP4 logits");
        try require(c.ds4_session_copy_logits(source, scalar_plain.ptr, @intCast(vocab)) == @as(c_int, @intCast(vocab)), "pre-cycle scalar F32 logits");
        var scalar_maximum: f64 = 0;
        var scalar_maximum_index: usize = 0;
        var scalar_finite = true;
        for (scalar_packed, scalar_plain, 0..) |x, y, i| {
            scalar_finite = scalar_finite and std.math.isFinite(x) and std.math.isFinite(y);
            const difference = @abs(@as(f64, x) - y);
            if (difference > scalar_maximum) {
                scalar_maximum = difference;
                scalar_maximum_index = i;
            }
        }
        std.debug.print("draft checkpoint: pre-cycle scalar token={d} max={d} logit={d} packed={d} plain={d} finite={}\n", .{ scalar_token, scalar_maximum, scalar_maximum_index, scalar_packed[scalar_maximum_index], scalar_plain[scalar_maximum_index], scalar_finite });
        try require(c.ds4_session_load_snapshot(source, &saved, &errors, errors.len) == 0, "restore scalar teacher after one-token probe");
        try require(c.ds4_session_load_snapshot(restored, &packed_before_cycle, &errors, errors.len) == 0, "restore packed frontier after one-token probe");
        try require(scalar_finite and scalar_maximum < 0.01, "pre-cycle scalar cache-layout agreement");
    }
    try cycle(restored.?, engine.?, true, source);
    if (packed_cross) {
        var packed_snapshot = std.mem.zeroes(c.ds4_session_snapshot);
        defer c.ds4_session_snapshot_free(&packed_snapshot);
        try require(c.ds4_session_save_snapshot(restored, &packed_snapshot, &errors, errors.len) == 0, "save physical FP4 canonical checkpoint");
        try require(std.mem.readInt(u32, packed_snapshot.ptr[4..8], .little) == packed_version, "physical cache exports selected v3/v4/v5");
        const live_tokens = c.ds4_session_tokens(restored);
        try require(live_tokens != null, "borrow actual committed token history");
        var packed_key: [41]u8 = undefined;
        try require(c.ds4_state_cache_put(cache, profile, packed_snapshot.ptr, packed_snapshot.len, null, 0, false, &packed_key, &errors, errors.len) == 0, "publish complete packed state plus DSP1");
        try require(c.ds4_state_cache_close(cache, &errors, errors.len) == 0, "close after packed publication");
        cache = null;
        cache = c.ds4_state_cache_open(cache_dir.ptr, 0, 512 << 20, &errors, errors.len);
        try require(cache != null, "reopen packed SSD manifest");
        var packed_ssd = try preferred(cache.?, live_tokens, &packed_snapshot);
        defer c.ds4_state_cache_blob_free(&packed_ssd);
        const packed_wire = c.ds4_session_snapshot{ .ptr = packed_ssd.state, .len = packed_ssd.state_bytes, .cap = packed_ssd.state_bytes };
        try require(c.ds4_session_load_snapshot(source, &packed_wire, &errors, errors.len) == 0, "import packed SSD into F32 graph");
        var twin = std.mem.zeroes(c.ds4_session_snapshot);
        defer c.ds4_session_snapshot_free(&twin);
        try require(c.ds4_session_save_snapshot(source, &twin, &errors, errors.len) == 0, "save same-prefix F32 v2 twin");
        try require(std.mem.readInt(u32, twin.ptr[4..8], .little) == 2, "twin uses canonical v2");
        const position = std.mem.readInt(u32, packed_snapshot.ptr[28..32], .little);
        const saved_row_bytes: u64 = @as(u64, if (cross_main) 1760 else 0) + @as(u64, if (cross_index) 444 else 0);
        const saved_cache_bytes = (@as(u64, 3) * (position / 2) + position) * saved_row_bytes;
        try require(twin.len == packed_snapshot.len + saved_cache_bytes, "same-prefix wire difference is exactly packed main/index rows");
        try require(c.ds4_state_cache_put(cache, profile, twin.ptr, twin.len, null, 0, false, &base_key, &errors, errors.len) == 0, "publish same-prefix v2 SSD twin");
        std.debug.print("draft checkpoint: paired prefix={d} v2_bytes={d} packed_version={d} packed_bytes={d} cache_saved_bytes={d}\n", .{ position, twin.len, packed_version, packed_snapshot.len, saved_cache_bytes });
        try cycle(restored.?, engine.?, true, source);

        var expanded = std.mem.zeroes(c.ds4_session_snapshot);
        defer c.ds4_session_snapshot_free(&expanded);
        try require(c.ds4_session_save_snapshot(source, &expanded, &errors, errors.len) == 0, "resave converted F32 checkpoint");
        try require(std.mem.readInt(u32, expanded.ptr[4..8], .little) == 2, "F32 graph retains canonical v2");
        try require(c.ds4_session_load_snapshot(restored, &expanded, &errors, errors.len) == 0, "import canonical v2 back into physical FP4 graph");
        try cycle(restored.?, engine.?, true, source);
        std.debug.print("draft checkpoint: v2/v{d} cross-import, both-direction MTP teacher agreement and HOT0 SSD restart PASS\n", .{packed_version});
    }
    std.debug.print("draft checkpoint: PASS old_base={d} DSP1={d} HOT0 SSD restart exact\n", .{ old.len, saved.len });
}
