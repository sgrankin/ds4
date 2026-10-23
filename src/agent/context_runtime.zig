//! Live context graph transactions and bounded immutable prefix/source owners.
//! The actual Worker record is shared with C; Engine/Session remain opaque.
//! Text edits are prepared before any graph write and committed only after a
//! successful suffix graft (or selection of the ordinary exact rebuild path).
const std = @import("std");
const c = @import("agent_runtime_c");
const suffix_match = @import("suffix_match");
const Trace = @import("trace.zig").Trace;
const scratch_bytes: u64 = 8 << 20;

const LegacyTail = struct { source: c_int, destination: c_int };
fn zero(comptime T: type) T {
    return std.mem.zeroes(T);
}
fn text(pointer: [*c]const u8) []const u8 {
    return if (pointer == null) &.{} else std.mem.span(pointer);
}
fn ids(tokens: *const c.ds4_tokens) []const c_int {
    return if (tokens.len == 0) &.{} else tokens.v[0..@intCast(tokens.len)];
}
fn same(left: [*c]const c.ds4_tokens, right: *const c.ds4_tokens) bool {
    return left != null and std.mem.eql(c_int, ids(@ptrCast(left)), ids(right));
}
fn report(errors: []u8, message: [*c]const u8) void {
    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s", message);
}
fn integer(name: [*:0]const u8, fallback: c_int, minimum: c_int, maximum: c_int) c_int {
    const value = c.getenv(name);
    if (value == null or value[0] == 0) return fallback;
    var end: [*c]u8 = null;
    const parsed = c.strtol(value, &end, 10);
    if (end == value) return fallback;
    while (end[0] == ' ' or end[0] == '\t' or end[0] == '\r' or end[0] == '\n') end += 1;
    if (end[0] != 0) return fallback;
    return @intCast(@max(minimum, @min(maximum, parsed)));
}
fn boolean(name: [*:0]const u8, fallback: bool) bool {
    const value = c.getenv(name);
    if (value == null or value[0] == 0) return fallback;
    if (c.strcasecmp(value, "true") == 0 or c.strcasecmp(value, "yes") == 0 or c.strcmp(value, "1") == 0) return true;
    if (c.strcasecmp(value, "false") == 0 or c.strcasecmp(value, "no") == 0 or c.strcmp(value, "0") == 0) return false;
    return fallback;
}
fn suffixRequested() bool {
    return boolean("DS4_AGENT_CONTEXT_SUFFIX_REUSE", false);
}
fn preropeRequested() bool {
    return suffixRequested() and boolean("DS4_AGENT_CONTEXT_SUFFIX_PREROPE", false);
}
fn rightRequested() bool {
    return preropeRequested() and boolean("DS4_AGENT_CONTEXT_SUFFIX_RIGHT", false);
}
fn residentConflict() bool {
    if (preropeRequested()) return false;
    const value = c.getenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP");
    if (value == null or value[0] == 0) return false;
    var end: [*c]u8 = null;
    const gap = c.strtod(value, &end);
    return end == null or end[0] != 0 or !std.math.isFinite(gap) or gap != 0;
}
fn now() f64 {
    var stamp: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &stamp);
    return @as(f64, @floatFromInt(stamp.tv_sec)) + @as(f64, @floatFromInt(stamp.tv_nsec)) * 1e-9;
}
fn cancel(context: ?*anyopaque) callconv(.c) bool {
    const worker: *c.agent_worker = @ptrCast(@alignCast(context.?));
    _ = c.pthread_mutex_lock(&worker.mu);
    defer _ = c.pthread_mutex_unlock(&worker.mu);
    return worker.interrupt or worker.stop;
}

pub const Runtime = struct {
    worker: *c.agent_worker,

    fn config(self: Runtime) *c.agent_config {
        return @ptrCast(self.worker.cfg);
    }
    pub fn contextSize(self: Runtime) c_int {
        var current = if (self.worker.session != null) c.ds4_session_ctx(self.worker.session) else 0;
        if (current <= 0 and self.worker.cfg != null) current = self.config().gen.ctx_size;
        return current;
    }
    fn interrupted(self: Runtime) bool {
        return cancel(self.worker);
    }
    fn trace(self: Runtime, format: [*:0]const u8, arguments: anytype) void {
        (Trace{ .worker = self.worker }).log(format, arguments);
    }
    fn budget() u64 {
        return @as(u64, @intCast(integer("DS4_AGENT_CONTEXT_FRONTIER_MB", 128, 1, 1024))) << 20;
    }
    fn fits(self: Runtime, staged: u64, limit: u64) bool {
        return staged <= limit and self.worker.state_cache_staged_bytes <= limit - staged and
            self.worker.context_frontier_bytes <= limit - staged - self.worker.state_cache_staged_bytes;
    }
    pub fn clearImages(self: Runtime) void {
        const worker = self.worker;
        for (0..worker.image_count) |index| c.ds4_vision_embedding_free(&worker.images[index].embedding);
        c.free(worker.images);
        worker.images = null;
        worker.image_count = 0;
        worker.image_cap = 0;
    }
    pub fn appendImages(self: Runtime, images: [*c]c.ds4_vision_span, count: usize) void {
        if (count == 0) return;
        const worker = self.worker;
        if (worker.image_count + count > worker.image_cap) {
            var capacity = if (worker.image_cap != 0) worker.image_cap else 4;
            while (capacity < worker.image_count + count) capacity *%= 2;
            const storage = c.realloc(worker.images, capacity *% @sizeOf(c.ds4_vision_span));
            if (storage == null) {
                c.perror("ds4-agent: realloc");
                c.exit(1);
            }
            worker.images = @ptrCast(@alignCast(storage));
            worker.image_cap = capacity;
        }
        @memcpy(worker.images[worker.image_count..][0..count], images[0..count]);
        worker.image_count += count;
        @memset(images[0..count], zero(c.ds4_vision_span));
    }
    pub fn imagesFit(self: Runtime, tokens: *const c.ds4_tokens) bool {
        for (0..self.worker.image_count) |index| {
            const image = self.worker.images[index];
            const end = @as(u64, image.token_start) +% image.embedding.token_count;
            if (end > @as(u64, @bitCast(@as(i64, tokens.len)))) return false;
        }
        return true;
    }
    pub fn cacheAllowed(self: Runtime) bool {
        const worker = self.worker;
        return worker.state_cache != null and worker.image_count == 0 and
            c.ds4_engine_is_deepseek41(worker.engine) and !c.ds4_session_is_distributed(worker.session);
    }
    pub fn materialize(self: Runtime, reference: ?*c.ds4_state_cache_ref, output: *c.ds4_suffix_cache, kind: [*c]c.ds4_state_cache_kind, errors: []u8) bool {
        var blob = zero(c.ds4_state_cache_blob);
        if (c.ds4_state_cache_materialize(reference, &blob, errors.ptr, errors.len) != 0) return false;
        if (blob.ctx_size != @as(u32, @bitCast(self.contextSize()))) {
            c.ds4_state_cache_blob_free(&blob);
            return false;
        }
        if (kind != null) kind.* = blob.kind;
        output.* = .{ .state = .{ .ptr = blob.state, .len = blob.state_bytes, .cap = blob.state_bytes }, .prerope = blob.prerope, .prerope_bytes = blob.prerope_bytes, .engine = self.worker.engine, .ctx_size = @bitCast(blob.ctx_size), .approximate = blob.approximate };
        return true; // Blob ownership moves into output; do not free it here.
    }
    pub fn putSource(self: Runtime, source: *const c.ds4_suffix_cache, key: [*c]u8) bool {
        if (!self.cacheAllowed()) return false;
        var errors: [160]u8 = @splat(0);
        const worker = self.worker;
        const result = c.ds4_state_cache_put(worker.state_cache, &worker.state_cache_profile, source.state.ptr, source.state.len, source.prerope, source.prerope_bytes, source.approximate, key, &errors, errors.len);
        self.trace("state cache store %s tokens=%d approximate=%d%s%s", .{ @as([*:0]const u8, if (result != 0) "miss" else "ok"), c.ds4_session_pos(worker.session), @as(c_int, @intFromBool(source.approximate)), @as([*:0]const u8, if (errors[0] != 0) " error=" else ""), &errors });
        return result == 0;
    }
    pub fn storeEndpoint(self: Runtime) void {
        const worker = self.worker;
        const replay = c.ds4_session_is_decoder_replay(worker.session);
        if (!self.cacheAllowed() or self.interrupted() or
            (c.ds4_session_is_approximate(worker.session) and !replay and !preropeRequested())) return;
        const count = if (replay) c.ds4_session_replay_snapshot_bytes(worker.session) else if (preropeRequested()) c.ds4_session_suffix_cache_bytes(worker.session) else c.ds4_session_payload_bytes(worker.session);
        const staged = count +% 1 +% @as(u64, if (worker.state_cache_staged_bytes != 0) 0 else scratch_bytes);
        if (count == 0 or !self.fits(staged, budget())) {
            self.trace("state cache endpoint skipped: staging budget tokens=%d", .{c.ds4_session_pos(worker.session)});
            return;
        }
        var source = zero(c.ds4_suffix_cache);
        defer c.ds4_suffix_cache_free(&source);
        var errors: [160]u8 = @splat(0);
        var key: [41]u8 = undefined;
        if (replay) {
            var result = c.ds4_session_save_replay_snapshot(worker.session, &source.state, &errors, errors.len);
            if (result == 0) result = c.ds4_state_cache_put_kind(worker.state_cache, &worker.state_cache_profile, source.state.ptr, source.state.len, null, 0, c.DS4_STATE_CACHE_REPLAY, &key, &errors, errors.len);
            self.trace("state cache replay store %s tokens=%d%s%s", .{ @as([*:0]const u8, if (result != 0) "miss" else "ok"), c.ds4_session_pos(worker.session), @as([*:0]const u8, if (errors[0] != 0) " error=" else ""), &errors });
            return;
        }
        const result = if (preropeRequested()) c.ds4_session_capture_suffix_cache(worker.session, &source, &errors, errors.len) else c.ds4_session_save_snapshot(worker.session, &source.state, &errors, errors.len);
        if (result == 0) {
            _ = self.putSource(&source, &key);
        } else self.trace("state cache capture failed error=%s", .{&errors});
    }
    pub fn restoreCachePrefix(self: Runtime, tokens: *const c.ds4_tokens) bool {
        const worker = self.worker;
        if (!self.cacheAllowed() or c.ds4_session_is_approximate(worker.session) or self.interrupted()) return false;
        const old = c.ds4_session_pos(worker.session);
        const common = c.ds4_session_common_prefix(worker.session, tokens);
        const have: u32 = if (common == old and tokens.len >= old) @bitCast(common) else 0;
        var matched: u32 = 0;
        const reference = c.ds4_state_cache_find_prefix(worker.state_cache, &worker.state_cache_profile, @bitCast(self.contextSize()), tokens.v, @bitCast(tokens.len), &matched) orelse return false;
        defer c.ds4_state_cache_release(reference);
        var source = zero(c.ds4_suffix_cache);
        defer c.ds4_suffix_cache_free(&source);
        var errors: [160]u8 = @splat(0);
        var kind: c.ds4_state_cache_kind = c.DS4_STATE_CACHE_NATIVE;
        var state_bytes: u64 = undefined;
        var pre_bytes: u64 = undefined;
        c.ds4_state_cache_ref_sizes(reference, &state_bytes, &pre_bytes);
        const staged = state_bytes +% pre_bytes +% @as(u64, if (worker.state_cache_staged_bytes != 0) 0 else scratch_bytes);
        if (matched <= have or !self.fits(staged, budget()) or
            !self.materialize(reference, &source, &kind, &errors) or
            (source.approximate and kind != c.DS4_STATE_CACHE_REPLAY)) return false;
        c.ds4_session_set_cancel(worker.session, cancel, worker);
        const result = if (kind == c.DS4_STATE_CACHE_REPLAY)
            c.ds4_session_load_replay_snapshot(worker.session, &source.state, &errors, errors.len)
        else if (preropeRequested()) c.ds4_session_load_suffix_prefix(worker.session, &source, &errors, errors.len) else c.ds4_session_load_snapshot(worker.session, &source.state, &errors, errors.len);
        c.ds4_session_set_cancel(worker.session, null, null);
        const live = c.ds4_session_tokens(worker.session);
        const ok = result == 0 and live != null and live.*.len == @as(c_int, @bitCast(matched)) and
            std.mem.eql(c_int, ids(@ptrCast(live)), ids(tokens)[0..matched]);
        if (!ok) c.ds4_session_invalidate(worker.session);
        self.trace("state cache %s prefix %s tokens=%u%s%s", .{ @as([*:0]const u8, if (kind == c.DS4_STATE_CACHE_REPLAY) "replay" else "exact"), @as([*:0]const u8, if (ok) "hit" else "miss"), matched, @as([*:0]const u8, if (errors[0] != 0) " error=" else ""), &errors });
        return ok;
    }
    /// Opt-in exact system-prefix restoration. Discarded conversation state is
    /// never retained; the changed suffix still follows ordinary sync.
    pub fn restoreCompactPrefix(self: Runtime, system: *const c.ds4_tokens) bool {
        const worker = self.worker;
        if (!boolean("DS4_AGENT_COMPACT_PREFIX_CACHE", false) or worker.sysprompt_path == null or
            worker.image_count != 0 or !c.ds4_engine_is_deepseek41(worker.engine) or
            c.ds4_tp_enabled(&self.config().engine.tp) or c.ds4_session_is_distributed(worker.session) or
            self.config().engine.directional_steering_file != null) return false;
        var length: usize = 0;
        const rendered = c.ds4_kvstore_render_tokens_text(worker.engine, system, &length);
        if (rendered == null) return false;
        defer c.free(rendered);
        var errors: [160]u8 = @splat(0);
        const started = now();
        var rebuilt = zero(c.ds4_tokens);
        defer c.ds4_tokens_free(&rebuilt);
        var load: ?*c.ds4_agent_checkpoint_load = null;
        defer c.ds4_agent_checkpoint_free_load(load);
        const prepared = c.ds4_agent_checkpoint_prepare_load(worker.engine, worker.session, worker.sysprompt_path, null, rendered, length, c.ds4_tp_enabled(&self.config().engine.tp), &rebuilt, &load, &errors, errors.len);
        var loaded = false;
        if (prepared >= 0) {
            if (prepared != 0 and c.ds4_session_is_approximate(worker.session)) c.ds4_session_invalidate(worker.session);
            const synchronized = prepared == 0 or c.ds4_agent_runtime_sync(worker, &rebuilt, true, &errors, errors.len) == 0;
            if (!synchronized) c.ds4_session_invalidate(worker.session);
            c.ds4_tokens_free(&rebuilt);
            loaded = c.ds4_agent_checkpoint_finish_load(load, synchronized, null, null, &errors, errors.len);
        }
        if (loaded and !same(c.ds4_session_tokens(worker.session), system)) {
            c.ds4_session_invalidate(worker.session);
            loaded = false;
            report(&errors, "system payload tokens do not match prefix");
        }
        self.trace("compaction prefix cache %s tokens=%d %.3f ms%s%s", .{ @as([*:0]const u8, if (loaded) "hit" else "miss"), system.len, (now() - started) * 1000, @as([*:0]const u8, if (errors[0] != 0) " error=" else ""), &errors });
        return loaded;
    }
    pub fn dropFrontier(self: Runtime, index: usize) void {
        const worker = self.worker;
        const frontier = &worker.context_frontiers[index];
        worker.context_frontier_bytes -%= frontier.cache.state.cap +% frontier.cache.prerope_bytes +%
            @as(u64, @intCast(frontier.prefix.cap)) *% @sizeOf(c_int);
        c.ds4_suffix_cache_free(&frontier.cache);
        c.ds4_tokens_free(&frontier.prefix);
        const count: usize = @intCast(worker.context_frontier_count);
        std.mem.copyForwards(c.agent_context_frontier, worker.context_frontiers[index .. count - 1], worker.context_frontiers[index + 1 .. count]);
        worker.context_frontier_count -= 1;
        worker.context_frontiers[count - 1] = zero(c.agent_context_frontier);
    }
    pub fn clearFrontiers(self: Runtime) void {
        while (self.worker.context_frontier_count != 0) self.dropFrontier(0);
    }
    pub fn frontierSlots(self: Runtime, limit: *u64) c_int {
        const worker = self.worker;
        const slots = integer("DS4_AGENT_CONTEXT_FRONTIERS", if (suffixRequested()) 2 else 0, 0, 4);
        limit.* = budget();
        if (slots == 0 or (self.config().gen.live_context_path == null and !preropeRequested()) or
            worker.image_count != 0 or !c.ds4_engine_is_deepseek41(worker.engine) or
            c.ds4_tp_enabled(&self.config().engine.tp) or c.ds4_session_is_distributed(worker.session) or
            self.config().engine.directional_steering_file != null)
        {
            self.clearFrontiers();
            return 0;
        }
        var index: usize = 0;
        while (index < @as(usize, @intCast(worker.context_frontier_count))) {
            const frontier = &worker.context_frontiers[index];
            if (frontier.engine != worker.engine or frontier.ctx != self.contextSize()) self.dropFrontier(index) else index += 1;
        }
        while (worker.context_frontier_count > slots or worker.context_frontier_bytes > limit.*) self.dropFrontier(0);
        return slots;
    }
    /// Approximate checkpoints are private pre-RoPE relocation sources only.
    /// Serialized size (including its final NUL) is known before allocation;
    /// retained entries and staged captures share one configured RAM budget.
    pub fn captureFrontier(self: Runtime) bool {
        const worker = self.worker;
        if (c.ds4_session_is_approximate(worker.session) and !preropeRequested()) return false;
        var limit: u64 = undefined;
        const slots = self.frontierSlots(&limit);
        if (slots == 0) {
            self.storeEndpoint();
            return false;
        }
        const live = c.ds4_session_tokens(worker.session);
        const position = c.ds4_session_pos(worker.session);
        if (live == null or live.*.len != position or position <= 0 or !c.ds4_tokens_starts_with(&worker.transcript, live)) return false;
        var index: usize = 0;
        while (index < @as(usize, @intCast(worker.context_frontier_count))) : (index += 1) {
            if (!same(live, &worker.context_frontiers[index].prefix)) continue;
            if (worker.state_cache == null) return true;
            self.dropFrontier(index);
            break;
        }
        const payload = if (preropeRequested()) c.ds4_session_suffix_cache_bytes(worker.session) else c.ds4_session_payload_bytes(worker.session);
        var count = payload +% 1 +% @as(u64, @intCast(position)) *% @sizeOf(c_int);
        const staged = count +% @as(u64, if (worker.state_cache != null) scratch_bytes else 0);
        if (payload == 0 or staged > limit) {
            self.trace("context frontier skipped tokens=%d bytes=%llu budget=%llu", .{ position, count, limit });
            return false;
        }
        while (worker.context_frontier_count >= slots or worker.context_frontier_bytes +% staged > limit) self.dropFrontier(0);
        const frontier = &worker.context_frontiers[@intCast(worker.context_frontier_count)];
        const storage = c.malloc(@as(usize, @intCast(position)) * @sizeOf(c_int));
        if (storage == null) return false;
        frontier.prefix = .{ .v = @ptrCast(@alignCast(storage)), .len = position, .cap = position };
        @memcpy(frontier.prefix.v[0..@intCast(position)], live.*.v[0..@intCast(position)]);
        var errors: [160]u8 = @splat(0);
        const started = now();
        const result = if (preropeRequested()) c.ds4_session_capture_suffix_cache(worker.session, &frontier.cache, &errors, errors.len) else c.ds4_session_save_snapshot(worker.session, &frontier.cache.state, &errors, errors.len);
        if (result != 0) {
            c.ds4_suffix_cache_free(&frontier.cache);
            c.ds4_tokens_free(&frontier.prefix);
            self.trace("context frontier capture failed error=%s", .{&errors});
            return false;
        }
        frontier.cache.engine = worker.engine;
        frontier.cache.ctx_size = self.contextSize();
        frontier.engine = worker.engine;
        frontier.ctx = self.contextSize();
        if (self.putSource(&frontier.cache, &frontier.disk_key)) {
            frontier.disk_state_bytes = frontier.cache.state.len;
            frontier.disk_prerope_bytes = frontier.cache.prerope_bytes;
            frontier.disk_approximate = frontier.cache.approximate;
            c.ds4_suffix_cache_free(&frontier.cache);
            count = @as(u64, @intCast(frontier.prefix.cap)) * @sizeOf(c_int);
        }
        worker.context_frontier_count += 1;
        worker.context_frontier_bytes +%= count;
        self.trace("context frontier captured tokens=%d bytes=%llu retained=%d retained_bytes=%llu %.3f ms", .{ position, count, worker.context_frontier_count, worker.context_frontier_bytes, (now() - started) * 1000 });
        return true;
    }
    pub fn restoreFrontier(self: Runtime, edited: *const c.ds4_tokens) bool {
        const worker = self.worker;
        var limit: u64 = undefined;
        if (self.frontierSlots(&limit) == 0) return false;
        while (true) {
            var best: ?usize = null;
            for (worker.context_frontiers[0..@intCast(worker.context_frontier_count)], 0..) |*frontier, index| {
                if (!frontier.cache.approximate and !frontier.disk_approximate and frontier.engine == worker.engine and
                    frontier.ctx == self.contextSize() and c.ds4_tokens_starts_with(edited, &frontier.prefix) and
                    (best == null or frontier.prefix.len > worker.context_frontiers[best.?].prefix.len)) best = index;
            }
            const index = best orelse return false;
            const frontier = &worker.context_frontiers[index];
            var errors: [160]u8 = @splat(0);
            const started = now();
            var source = zero(c.ds4_suffix_cache);
            const reference = if (frontier.disk_key[0] != 0 and worker.state_cache != null)
                c.ds4_state_cache_get(worker.state_cache, &frontier.disk_key)
            else
                null;
            const staged = frontier.disk_state_bytes +% frontier.disk_prerope_bytes +%
                @as(u64, if (worker.state_cache_staged_bytes != 0) 0 else scratch_bytes);
            var result: c_int = 1;
            if (frontier.disk_key[0] == 0 or (reference != null and self.fits(staged, limit) and
                self.materialize(reference, &source, null, &errors)))
            {
                const selected = if (frontier.disk_key[0] != 0) &source else &frontier.cache;
                if (!selected.approximate) result = if (preropeRequested())
                    c.ds4_session_load_suffix_prefix(worker.session, selected, &errors, errors.len)
                else
                    c.ds4_session_load_snapshot(worker.session, &selected.state, &errors, errors.len);
            }
            c.ds4_suffix_cache_free(&source);
            c.ds4_state_cache_release(reference);
            const ok = result == 0 and same(c.ds4_session_tokens(worker.session), &frontier.prefix);
            self.trace("context frontier restore %s tokens=%d %.3f ms%s%s", .{ @as([*:0]const u8, if (ok) "hit" else "miss"), frontier.prefix.len, (now() - started) * 1000, @as([*:0]const u8, if (errors[0] != 0) " error=" else ""), &errors });
            if (ok) return true;
            c.ds4_session_invalidate(worker.session);
            self.dropFrontier(index);
        }
    }
    fn legacyTail(self: Runtime, edited: *const c.ds4_tokens) ?LegacyTail {
        const old = ids(&self.worker.transcript);
        const new = ids(edited);
        var common: usize = 0;
        while (common < old.len and common < new.len and old[common] == new[common]) common += 1;
        var tail: usize = 0;
        while (tail < old.len - common and tail < new.len - common and
            old[old.len - tail - 1] == new[new.len - tail - 1]) tail += 1;
        var source = old.len - tail;
        var destination = new.len - tail;
        if (source & 1 != 0) {
            source += 1;
            destination += 1;
        }
        if (destination & 1 != 0) return null;
        return .{ .source = @intCast(source), .destination = @intCast(destination) };
    }
    pub fn selectDisk(self: Runtime, edited: *const c.ds4_tokens, fresh: c_int, plan: *c.agent_context_suffix_plan) bool {
        if (!self.cacheAllowed()) return false;
        const worker = self.worker;
        const prerope = preropeRequested();
        const displacement = worker.transcript.len - edited.len;
        const tail: ?LegacyTail = if (prerope) null else (self.legacyTail(edited) orelse return false);
        var cursor: [41]u8 = @splat(0);
        var candidate_ids: [*c]const c_int = undefined;
        var count: u32 = undefined;
        var approximate: bool = undefined;
        while (c.ds4_state_cache_next_candidate(worker.state_cache, &worker.state_cache_profile, @bitCast(self.contextSize()), &cursor, &candidate_ids, &count, &approximate)) |reference| {
            var state_bytes: u64 = undefined;
            var pre_bytes: u64 = undefined;
            c.ds4_state_cache_ref_sizes(reference, &state_bytes, &pre_bytes);
            var chosen = false;
            if (count >= 64 and (if (prerope) pre_bytes != 0 else !approximate)) {
                if (prerope) {
                    var match: c.ds4_suffix_match_plan = undefined;
                    if (suffix_match.shift(c.ds4_suffix_match_plan, candidate_ids, count, edited.v, @bitCast(edited.len), @intCast(edited.len - fresh), rightRequested(), &match) and match.reused_tokens > plan.reused_tokens) {
                        setMatch(plan, &match);
                        chosen = true;
                    }
                } else {
                    const end = @as(c_int, @bitCast(count)) - displacement;
                    if (tail.?.destination > 0 and count <= @as(u32, @bitCast(worker.transcript.len)) and
                        std.mem.eql(c_int, candidate_ids[0..count], ids(&worker.transcript)[0..count]) and
                        @as(c_int, @bitCast(count)) - tail.?.source >= 64 and end >= tail.?.destination and
                        edited.len - end >= fresh and count > plan.source_end)
                    {
                        plan.source_start = @intCast(tail.?.source);
                        plan.destination_start = @intCast(tail.?.destination);
                        plan.source_end = count;
                        plan.destination_end = @intCast(end);
                        plan.reused_tokens = count - @as(u32, @intCast(tail.?.source));
                        plan.fresh_tokens = @intCast(edited.len - end);
                        chosen = true;
                    }
                }
            }
            if (chosen) {
                c.ds4_state_cache_release(plan.disk_ref);
                plan.disk_ref = reference;
                plan.disk_state_bytes = state_bytes;
                plan.disk_prerope_bytes = pre_bytes;
            } else c.ds4_state_cache_release(reference);
        }
        return plan.disk_ref != null;
    }
    fn setMatch(plan: *c.agent_context_suffix_plan, match: *const c.ds4_suffix_match_plan) void {
        plan.source_start = match.source_start;
        plan.destination_start = match.destination_start;
        plan.source_end = match.source_end;
        plan.destination_end = match.destination_end;
        plan.reused_tokens = match.reused_tokens;
        plan.fresh_tokens = match.fresh_tokens;
    }
    /// Select an immutable source ending inside the unchanged tail. Its
    /// endpoint cannot be truncated: at least the latest user marker and its
    /// complete following message/history must be evaluated freshly. Unknown
    /// role layouts rebuild. Approximate sources are private pre-RoPE only;
    /// canonical prefix restoration always excludes them.
    pub fn selectSuffix(self: Runtime, edited: *const c.ds4_tokens, requested: c_int, plan: *c.agent_context_suffix_plan) bool {
        plan.* = zero(c.agent_context_suffix_plan);
        plan.source_frontier = -1;
        const worker = self.worker;
        const displacement = worker.transcript.len - edited.len;
        if (requested < 1 or (!preropeRequested() and (displacement < 0 or displacement & 1 != 0))) return false;
        const user = c.ds4_token_user(worker.engine);
        if (user < 0) return false;
        const latest = std.mem.lastIndexOfScalar(c_int, ids(edited), user) orelse return false;
        const fresh = @max(requested, edited.len - @as(c_int, @intCast(latest)));
        if (fresh >= edited.len) return false;
        // Preserve the existing disk-first policy, including its first-tie rule.
        if (self.selectDisk(edited, fresh, plan)) return true;
        if (preropeRequested()) {
            for (worker.context_frontiers[0..@intCast(worker.context_frontier_count)], 0..) |*frontier, index| {
                if (frontier.engine != worker.engine or frontier.ctx != self.contextSize() or
                    frontier.cache.state.ptr == null or frontier.cache.prerope == null or frontier.prefix.len < 64) continue;
                var match: c.ds4_suffix_match_plan = undefined;
                if (suffix_match.shift(c.ds4_suffix_match_plan, frontier.prefix.v, @bitCast(frontier.prefix.len), edited.v, @bitCast(edited.len), @intCast(edited.len - fresh), rightRequested(), &match) and
                    match.reused_tokens > plan.reused_tokens)
                {
                    plan.* = zero(c.agent_context_suffix_plan);
                    plan.source_frontier = @intCast(index);
                    setMatch(plan, &match);
                }
            }
            return plan.source_frontier >= 0;
        }
        const tail = self.legacyTail(edited) orelse return false;
        for (worker.context_frontiers[0..@intCast(worker.context_frontier_count)], 0..) |*frontier, index| {
            const end = frontier.prefix.len - displacement;
            if (frontier.engine != worker.engine or frontier.ctx != self.contextSize() or
                frontier.cache.state.ptr == null or frontier.cache.state.len == 0 or
                frontier.cache.state.len > frontier.cache.state.cap or !c.ds4_tokens_starts_with(&worker.transcript, &frontier.prefix) or
                frontier.prefix.len - tail.source < 64 or end < tail.destination or edited.len - end < fresh or
                (plan.source_frontier >= 0 and frontier.prefix.len <= @as(c_int, @bitCast(plan.source_end)))) continue;
            plan.* = zero(c.agent_context_suffix_plan);
            plan.source_frontier = @intCast(index);
            plan.source_start = @intCast(tail.source);
            plan.destination_start = @intCast(tail.destination);
            plan.source_end = @intCast(frontier.prefix.len);
            plan.destination_end = @intCast(end);
            plan.reused_tokens = @intCast(frontier.prefix.len - tail.source);
            plan.fresh_tokens = @intCast(edited.len - end);
        }
        return plan.source_frontier >= 0;
    }
    /// Restoring a prefix can evict a frontier. Own its bytes first.
    fn cloneSource(output: *c.ds4_suffix_cache, source: *const c.ds4_suffix_cache) bool {
        output.* = zero(c.ds4_suffix_cache);
        output.engine = source.engine;
        output.ctx_size = source.ctx_size;
        output.approximate = source.approximate;
        output.state.ptr = @ptrCast(c.malloc(@as(usize, @intCast(source.state.len)) +% 1));
        if (output.state.ptr == null) return false;
        output.state.len = source.state.len;
        output.state.cap = source.state.len +% 1;
        if (source.state.len != 0) @memcpy(output.state.ptr[0..@intCast(source.state.len)], source.state.ptr[0..@intCast(source.state.len)]);
        if (source.prerope_bytes != 0) {
            output.prerope = c.malloc(@intCast(source.prerope_bytes));
            if (output.prerope == null) {
                c.ds4_suffix_cache_free(output);
                return false;
            }
            output.prerope_bytes = source.prerope_bytes;
            const destination: [*]u8 = @ptrCast(output.prerope.?);
            const original: [*]const u8 = @ptrCast(source.prerope.?);
            @memcpy(destination[0..@intCast(source.prerope_bytes)], original[0..@intCast(source.prerope_bytes)]);
        }
        return true;
    }
    /// Return1 committed private approximate graph,0 exact fallback,-1
    /// interruption/failed rollback. Transcript ownership stays with the caller.
    /// Source and exact undo share the RAM cap; a prior approximate graph has
    /// no canonical undo and must invalidate for fresh rebuild on failure.
    pub fn trySuffix(self: Runtime, edited: *const c.ds4_tokens, system: *const c.ds4_tokens, errors: []u8) c_int {
        const worker = self.worker;
        if (!suffixRequested() or residentConflict() or
            (c.ds4_session_is_approximate(worker.session) and !preropeRequested()) or worker.image_count != 0 or
            self.config().gen.raw_prompt or !c.ds4_engine_is_deepseek41(worker.engine) or
            c.ds4_tp_enabled(&self.config().engine.tp) or c.ds4_session_is_distributed(worker.session) or
            self.config().engine.directional_steering_file != null) return 0;
        var limit: u64 = undefined;
        if (self.frontierSlots(&limit) == 0) return 0;
        var plan: c.agent_context_suffix_plan = undefined;
        if (!self.selectSuffix(edited, integer("DS4_AGENT_CONTEXT_SUFFIX_FRESH_TAIL", 128, 64, 4096), &plan)) return 0;
        defer c.ds4_state_cache_release(plan.disk_ref);
        const source: ?*const c.agent_context_frontier = if (plan.disk_ref != null) null else &worker.context_frontiers[@intCast(plan.source_frontier)];
        const source_bytes = if (plan.disk_ref != null) plan.disk_state_bytes else source.?.cache.state.len;
        const pre_bytes = if (plan.disk_ref != null) plan.disk_prerope_bytes else source.?.cache.prerope_bytes;
        const approximate = c.ds4_session_is_approximate(worker.session);
        const undo_bytes = if (approximate) 0 else c.ds4_session_payload_bytes(worker.session);
        // Native payload I/O stages8MiB. Snapshot loads also build temporary
        // tokens: reserve twice the context plus stdio/actual packet overhead.
        const scratch = scratch_bytes +% (64 << 10) +% @as(u64, @intCast(self.contextSize())) *% 2 *% @sizeOf(c_int) +%
            @sizeOf(c.ds4_tokens) +% @sizeOf(c.ds4_session_snapshot) +% @sizeOf(c.ds4_suffix_cache) +%
            @sizeOf(c.agent_context_suffix_plan) +% @sizeOf(c.ds4_suffix_relocation_stats);
        const staged = undo_bytes +% 1 +% source_bytes +% 1 +% pre_bytes +%
            @as(u64, plan.destination_start) *% @sizeOf(c_int) +% scratch;
        if ((undo_bytes == 0 and !approximate) or staged > limit or worker.context_frontier_bytes > limit - staged) {
            self.trace("context suffix exact fallback: staged_bytes=%llu retained_bytes=%llu budget=%llu", .{ staged, worker.context_frontier_bytes, limit });
            return 0;
        }
        if (self.interrupted()) {
            report(errors, "interrupted");
            return -1;
        }
        var prefix = zero(c.ds4_tokens);
        var undo = zero(c.ds4_session_snapshot);
        var cache = zero(c.ds4_suffix_cache);
        defer {
            c.ds4_tokens_free(&prefix);
            c.ds4_session_snapshot_free(&undo);
            c.ds4_suffix_cache_free(&cache);
        }
        var statistics = zero(c.ds4_suffix_relocation_stats);
        var attempt_error: [160]u8 = @splat(0);
        var changed = false;
        var rollback = false;
        var result: c_int = 0;
        const started = now();
        attempt: {
            const storage = c.malloc(@as(usize, plan.destination_start) * @sizeOf(c_int));
            if (storage == null) break :attempt;
            prefix = .{ .v = @ptrCast(@alignCast(storage)), .len = @bitCast(plan.destination_start), .cap = @bitCast(plan.destination_start) };
            if (plan.destination_start != 0) @memcpy(prefix.v[0..plan.destination_start], edited.v[0..plan.destination_start]);
            if (!approximate and c.ds4_session_save_snapshot(worker.session, &undo, &attempt_error, attempt_error.len) != 0) break :attempt;
            if (plan.disk_ref != null) {
                if (!self.materialize(plan.disk_ref, &cache, null, &attempt_error)) break :attempt;
                if (!preropeRequested()) {
                    c.free(cache.prerope);
                    cache.prerope = null;
                    cache.prerope_bytes = 0;
                }
                changed = true;
            } else if (preropeRequested()) {
                if (!cloneSource(&cache, &source.?.cache)) break :attempt;
                changed = true;
            } else {
                changed = true;
                // Legacy post-RoPE capture requires an exact source graph.
                if (c.ds4_session_load_snapshot(worker.session, &source.?.cache.state, &attempt_error, attempt_error.len) != 0 or
                    !same(c.ds4_session_tokens(worker.session), &source.?.prefix) or self.interrupted() or
                    c.ds4_session_capture_suffix_cache(worker.session, &cache, &attempt_error, attempt_error.len) != 0)
                {
                    rollback = true;
                    break :attempt;
                }
            }
            const owned = undo.cap +% cache.state.cap +% cache.prerope_bytes +%
                @as(u64, @intCast(prefix.cap)) *% @sizeOf(c_int) +% scratch;
            if (owned > limit or worker.context_frontier_bytes > limit - owned) {
                report(&attempt_error, "owned suffix state exceeds its staging budget");
                rollback = true;
                break :attempt;
            }
            if (worker.state_cache != null) worker.state_cache_staged_bytes = owned;
            if (!self.restoreFrontier(&prefix)) {
                c.ds4_session_invalidate(worker.session);
                _ = self.restoreCompactPrefix(system);
            }
            if (self.interrupted() or c.ds4_agent_runtime_sync(worker, &prefix, true, &attempt_error, attempt_error.len) != 0) {
                rollback = true;
                break :attempt;
            }
            c.ds4_session_set_cancel(worker.session, cancel, worker);
            const reused = c.ds4_session_reuse_suffix_relocated(worker.session, &cache, edited, plan.source_start, &statistics, &attempt_error, attempt_error.len);
            c.ds4_session_set_cancel(worker.session, null, null);
            if (reused != 0 or self.interrupted()) {
                rollback = true;
                break :attempt;
            }
            const latest = std.mem.lastIndexOfScalar(c_int, ids(edited), c.ds4_token_user(worker.engine));
            const latest_user: c_int = if (latest) |index| @intCast(index) else -1;
            self.trace("context suffix APPROXIMATE reused=%u fresh=%u source=%u:%u destination=%u:%u copied_bytes=%llu latest_user_start=%d latest_user_fresh=%d pre_rope=%d prior_approximate=%d source_ssd=%d right_shift=%d source_approximate=%d %.3f ms", .{ statistics.reuse.reused_tokens, statistics.reuse.fresh_tail_tokens, statistics.source_start, statistics.source_end, statistics.destination_start, statistics.destination_end, statistics.reuse.copied_bytes, latest_user, @as(c_int, @intFromBool(latest_user >= 0 and statistics.destination_end <= @as(u32, @intCast(latest_user)))), @as(c_int, @intFromBool(cache.prerope != null)), @as(c_int, @intFromBool(approximate)), @as(c_int, @intFromBool(plan.disk_ref != null)), @as(c_int, @intFromBool(statistics.destination_start > statistics.source_start)), @as(c_int, @intFromBool(cache.approximate)), (now() - started) * 1000 });
            result = 1;
        }
        if (rollback) {
            const was_interrupted = self.interrupted();
            c.ds4_session_set_cancel(worker.session, null, null);
            var restore_error: [160]u8 = @splat(0);
            if (changed and approximate) c.ds4_session_invalidate(worker.session);
            if (changed and !approximate and c.ds4_session_load_snapshot(worker.session, &undo, &restore_error, restore_error.len) != 0) {
                c.ds4_session_invalidate(worker.session);
                if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "context suffix rollback failed: %.120s", &restore_error);
                result = -1;
            } else if (was_interrupted) {
                report(errors, "interrupted");
                result = -1;
            }
            self.trace("context suffix exact fallback rollback=%s interrupted=%d error=%s", .{ @as([*:0]const u8, if (restore_error[0] != 0) "failed" else if (approximate) "fresh-rebuild" else "ok"), @as(c_int, @intFromBool(was_interrupted)), @as([*c]const u8, if (attempt_error[0] != 0) &attempt_error else "source capture/sync was interrupted") });
        }
        worker.state_cache_staged_bytes = 0;
        if (result == 0 and self.interrupted()) {
            report(errors, "interrupted");
            result = -1;
        }
        return result;
    }
    pub fn clearContext(self: Runtime) void {
        c.ds4_agent_context_clear(&self.worker.live_context);
    }
    pub fn render(self: Runtime, length: [*c]usize) [*c]u8 {
        var system = zero(c.ds4_tokens);
        c.ds4_agent_runtime_build_system(self.worker, &system);
        defer c.ds4_tokens_free(&system);
        return c.ds4_agent_context_render(self.worker.engine, &self.worker.transcript, &system, &self.worker.live_context, length);
    }
    /// Mirroring is text-only. Without suffix reuse, edits restore an exact
    /// frontier and the next ordinary sync evaluates the changed suffix.
    pub fn exportContext(self: Runtime, errors: []u8) bool {
        const path = self.config().gen.live_context_path;
        if (path == null) return true;
        if (self.worker.image_count != 0) {
            report(errors, "live context files do not support image-bearing contexts");
            return false;
        }
        var length: usize = 0;
        const rendered = self.render(&length);
        if (rendered == null) {
            report(errors, "failed to render live context");
            return false;
        }
        defer c.free(rendered);
        return c.ds4_agent_file_replace(path, rendered, length, null, 0, errors.ptr, errors.len) == 0;
    }
    /// File read failure must precede any Engine/Session access. A prepared
    /// rejection leaves transcript and Context metadata unchanged; a structured
    /// no-op still commits its metadata, matching the existing Context contract.
    pub fn importContext(self: Runtime, errors: []u8) c_int {
        const worker = self.worker;
        const path = self.config().gen.live_context_path;
        if (path == null) return 0;
        var rendered: [*c]u8 = null;
        var length: usize = 0;
        if (c.ds4_agent_file_read_bytes(path, &rendered, &length, errors.ptr, errors.len) != 0) return -1;
        var system = zero(c.ds4_tokens);
        c.ds4_agent_runtime_build_system(worker, &system);
        defer c.ds4_tokens_free(&system);
        var edit = zero(c.ds4_agent_context_edit);
        defer c.ds4_agent_context_edit_free(&edit);
        const prepared = c.ds4_agent_context_prepare(worker.engine, &worker.transcript, &system, &worker.live_context, rendered, length, worker.image_count, self.contextSize(), &edit, errors.ptr, errors.len);
        c.free(rendered);
        if (prepared <= 0) {
            if (prepared == 0) c.ds4_agent_context_commit(&worker.transcript, &worker.live_context, &edit);
            return prepared;
        }
        const before = worker.transcript.len;
        const approximate = c.ds4_session_is_approximate(worker.session);
        const reused = self.trySuffix(&edit.tokens, &system, errors);
        if (reused < 0) return -1;
        c.ds4_agent_context_commit(&worker.transcript, &worker.live_context, &edit);
        // Release transferred/unused edit allocations before any graph restore.
        c.ds4_agent_context_edit_free(&edit);
        if (reused == 0 and approximate) c.ds4_session_invalidate(worker.session);
        if (reused == 0 and (approximate or c.ds4_session_common_prefix(worker.session, &worker.transcript) !=
            c.ds4_session_pos(worker.session)) and !self.restoreFrontier(&worker.transcript)) _ = self.restoreCompactPrefix(&system);
        self.trace("live context edit committed tokens=%d->%d file=%s", .{ before, worker.transcript.len, path });
        return if (reused != 0) 2 else 1;
    }
};
