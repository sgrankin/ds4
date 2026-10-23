//! Live session publication over the actual Worker: fixed bootstrap cache,
//! explicit stable saves/switches, history and numeric effort replacement.
//! File codecs and graph payloads remain selected production owners.
const std = @import("std");
const c = @import("agent_runtime_c");
const System = @import("system.zig");
const Prompts = @import("prompts.zig");
const Trace = @import("trace.zig").Trace;
const allocator = std.heap.c_allocator;

fn zero(comptime T: type) T {
    return std.mem.zeroes(T);
}
pub fn text(value: [*c]const u8) []const u8 {
    return if (value == null) &.{} else std.mem.span(value);
}
pub fn report(errors: []u8, comptime pattern: [*:0]const u8, arguments: anytype) void {
    if (errors.len != 0) _ = @call(.auto, c.snprintf, .{ errors.ptr, errors.len, pattern } ++ arguments);
}
pub fn fatal() noreturn {
    c.perror("ds4-agent: malloc");
    c.exit(1);
}
pub fn duplicate(value: []const u8) [*c]u8 {
    return (allocator.dupeSentinel(u8, value, 0) catch fatal()).ptr;
}
pub fn emit(worker: *c.agent_worker, value: []const u8) void {
    c.ds4_agent_comms_publish(worker, value.ptr, value.len);
}
pub fn print(worker: *c.agent_worker, comptime pattern: [*:0]const u8, arguments: anytype) void {
    const length = @call(.auto, c.snprintf, .{ @as([*c]u8, null), @as(usize, 0), pattern } ++ arguments);
    if (length <= 0) return;
    const line = allocator.allocSentinel(u8, @intCast(length), 0) catch fatal();
    defer allocator.free(line);
    _ = @call(.auto, c.snprintf, .{ line.ptr, line.len + 1, pattern } ++ arguments);
    emit(worker, line);
}
pub fn contextSize(worker: *const c.agent_worker) c_int {
    var current = if (worker.session != null) c.ds4_session_ctx(worker.session) else 0;
    if (current <= 0 and worker.cfg != null) current = worker.cfg.*.gen.ctx_size;
    return current;
}
pub fn makeDirectory(path: [*c]const u8) bool {
    if (text(path).len == 0) return false;
    const temporary = duplicate(text(path));
    defer c.free(temporary);
    for (1..text(path).len) |index| {
        if (temporary[index] != '/') continue;
        temporary[index] = 0;
        if (c.mkdir(temporary, 0o700) != 0 and c.__error().* != c.EEXIST) return false;
        temporary[index] = '/';
    }
    return c.mkdir(temporary, 0o700) == 0 or c.__error().* == c.EEXIST;
}
fn historyEmit(context: ?*anyopaque, value: [*c]const u8, length: usize) callconv(.c) void {
    c.ds4_agent_comms_publish(@ptrCast(@alignCast(context.?)), value, length);
}
pub fn numericPrefix(engine: ?*c.ds4_engine, mode: c.ds4_think_mode, output: *c.ds4_tokens) void {
    c.ds4_chat_begin(engine, output);
    c.ds4_chat_append_think_prefix(engine, output, mode);
    c.ds4_chat_append_message(engine, output, "system", "");
}

pub const Sessions = struct {
    worker: *c.agent_worker,
    fn trace(self: Sessions) Trace {
        return .{ .worker = self.worker };
    }
    fn textOnly(self: Sessions, tokens: *const c.ds4_tokens) bool {
        return tokens.len >= contextSize(self.worker) or c.ds4_session_is_approximate(self.worker.session);
    }
    pub fn hasUser(self: Sessions) bool {
        _ = c.pthread_mutex_lock(&self.worker.mu);
        defer _ = c.pthread_mutex_unlock(&self.worker.mu);
        return self.worker.user_activity;
    }
    pub fn needsSave(self: Sessions) bool {
        _ = c.pthread_mutex_lock(&self.worker.mu);
        defer _ = c.pthread_mutex_unlock(&self.worker.mu);
        return self.worker.user_activity and self.worker.session_dirty;
    }
    /// The held checkpoint FILE survives ordinary sync. Private approximate
    /// text must reopen through fresh evaluation even if its prefix matches.
    pub fn load(self: Sessions, path: [*c]const u8, sha: [*c]const u8, expected: [*c]const u8, expected_length: usize, output: *c.ds4_tokens, metadata: [*c]c.agent_kv_session_meta, errors: []u8) bool {
        const worker = self.worker;
        var owner: ?*c.ds4_agent_checkpoint_load = null;
        defer c.ds4_agent_checkpoint_free_load(owner);
        var rebuilt = zero(c.ds4_tokens);
        defer c.ds4_tokens_free(&rebuilt);
        const tensor_parallel = worker.cfg != null and c.ds4_tp_enabled(&worker.cfg.*.engine.tp);
        const mode = c.ds4_agent_checkpoint_prepare_load(worker.engine, worker.session, path, sha, expected, expected_length, tensor_parallel, &rebuilt, &owner, errors.ptr, errors.len);
        if (mode < 0) return false;
        var synchronized = true;
        if (mode > 0) {
            if (c.ds4_session_is_approximate(worker.session)) c.ds4_session_invalidate(worker.session);
            if (c.ds4_agent_turn_sync(worker, &rebuilt, true, errors.ptr, errors.len) != 0) {
                c.ds4_session_invalidate(worker.session);
                synchronized = false;
            }
        }
        c.ds4_tokens_free(&rebuilt);
        return c.ds4_agent_checkpoint_finish_load(owner, synchronized, output, metadata, errors.ptr, errors.len);
    }
    /// Fixed sysprompt.kv is shared by models, so its exact rendered bytes and
    /// the checkpoint's model ID must agree before it may skip initial prefill.
    pub fn reset(self: Sessions, errors: []u8) bool {
        const worker = self.worker;
        c.ds4_agent_runtime_context_clear(worker);
        c.ds4_agent_runtime_frontiers_clear(worker);
        c.ds4_agent_runtime_images_clear(worker);
        var system = zero(c.ds4_tokens);
        defer c.ds4_tokens_free(&system);
        System.build(worker, &system);
        var length: usize = 0;
        const rendered = c.ds4_kvstore_render_tokens_text(worker.engine, &system, &length);
        if (rendered == null) {
            report(errors, "failed to render system prompt", .{});
            return false;
        }
        defer c.free(rendered);
        var loaded = false;
        var load_errors: [160]u8 = @splat(0);
        if (worker.sysprompt_path != null) {
            loaded = self.load(worker.sysprompt_path, null, rendered, length, &worker.transcript, null, &load_errors);
            if (loaded) self.trace().log("sysprompt kv hit file=%s tokens=%d", .{ worker.sysprompt_path, worker.transcript.len });
        }
        if (!loaded) {
            if (worker.sysprompt_path != null) c.ds4_agent_turn_system_status(worker, "Updating system prompt cache...");
            c.ds4_tokens_free(&worker.transcript);
            c.ds4_tokens_copy(&worker.transcript, &system);
            if (c.ds4_agent_turn_sync(worker, &worker.transcript, true, errors.ptr, errors.len) != 0) return false;
            if (worker.sysprompt_path != null) {
                var save_errors: [160]u8 = @splat(0);
                var ignored_sha: [41]u8 = undefined;
                if (!c.ds4_agent_checkpoint_save(worker.engine, worker.session, worker.sysprompt_path, &worker.transcript, "agent-system", &ignored_sha, null, 0, false, &save_errors, save_errors.len)) {
                    if (worker.cfg.*.non_interactive) _ = c.fprintf(c.__stderrp, "ds4-agent: failed to save system prompt KV: %s\n", &save_errors) else print(worker, "\nds4-agent: failed to save system prompt KV: %s\n", .{&save_errors});
                } else self.trace().log("sysprompt kv stored file=%s tokens=%d", .{ worker.sysprompt_path, worker.transcript.len });
            }
        }
        worker.last_system_prompt_reminder_at = worker.transcript.len;
        _ = c.pthread_mutex_lock(&worker.mu);
        worker.user_activity = false;
        worker.session_dirty = false;
        worker.status.state = c.AGENT_WORKER_IDLE;
        worker.progress_direct = false;
        worker.status.prefill_done = 0;
        worker.status.prefill_total = 0;
        worker.status.prefill_tps = 0;
        worker.status.generated = 0;
        worker.status.gen_tps = 0;
        worker.status.greedy_sampling = false;
        worker.status.@"error"[0] = 0;
        if (worker.initialized) worker.hints = zero(c.agent_hints);
        c.ds4_agent_comms_wake_locked(worker);
        _ = c.pthread_mutex_unlock(&worker.mu);
        worker.datetime_context_injected = false;
        c.ds4_agent_sessions_identity_clear(&worker.identity);
        c.ds4_tokens_free(&system);
        const mirrored = c.ds4_agent_runtime_export(worker, errors.ptr, errors.len);
        if (mirrored) _ = c.ds4_agent_runtime_capture(worker);
        return mirrored;
    }
    /// Busy saves are deferred by Lifecycle until a stable append-only point.
    /// Identity remains title+creation-time based as the transcript evolves.
    pub fn saveNow(self: Sessions, sha_output: [*c]u8, count_output: [*c]c_int, errors: []u8) bool {
        const worker = self.worker;
        if (!self.hasUser()) {
            report(errors, "nothing to save", .{});
            return false;
        }
        if (worker.image_count != 0) {
            report(errors, "sessions containing images cannot be saved yet", .{});
            return false;
        }
        const approximate = c.ds4_session_is_approximate(worker.session);
        const text_only = self.textOnly(&worker.transcript);
        if (!text_only and c.ds4_agent_turn_sync(worker, &worker.transcript, false, errors.ptr, errors.len) != 0) return false;
        if (!makeDirectory(worker.cache_dir)) {
            report(errors, "failed to create %s", .{worker.cache_dir});
            return false;
        }
        var length: usize = 0;
        const rendered = c.ds4_kvstore_render_tokens_text(worker.engine, &worker.transcript, &length);
        if (rendered == null) {
            report(errors, "failed to render session text", .{});
            return false;
        }
        defer c.free(rendered);
        if (worker.identity.title == null) worker.identity.title = c.ds4_agent_sessions_title_text(rendered, length, 0);
        if (worker.identity.created_at == 0) worker.identity.created_at = @bitCast(@as(i64, c.time(null)));
        var sha: [41]u8 = undefined;
        c.ds4_agent_sessions_identity_sha(worker.identity.title, worker.identity.created_at, &sha);
        var filename: [44]u8 = undefined;
        @memcpy(filename[0..40], sha[0..40]);
        @memcpy(filename[40..44], ".kv\x00");
        const path = c.ds4_kvstore_path_join(worker.cache_dir, &filename);
        defer c.free(path);
        if (!c.ds4_agent_checkpoint_save(worker.engine, worker.session, path, &worker.transcript, "agent-session", sha_output, worker.identity.title, worker.identity.created_at, self.textOnly(&worker.transcript), errors.ptr, errors.len)) return false;
        worker.identity.sha = sha;
        if (approximate) emit(worker, "Saved approximate-context transcript without KV; reopening evaluates the saved text.\n") else if (text_only) print(worker, "Saved full transcript without KV; restart with a larger --ctx, then /switch %.8s and /compact.\n", .{&sha});
        if (worker.identity.legacy_path != null and !std.mem.eql(u8, text(worker.identity.legacy_path), text(path))) _ = c.unlink(worker.identity.legacy_path);
        c.free(worker.identity.legacy_path);
        worker.identity.legacy_path = null;
        _ = c.pthread_mutex_lock(&worker.mu);
        worker.session_dirty = false;
        c.ds4_agent_comms_wake_locked(worker);
        _ = c.pthread_mutex_unlock(&worker.mu);
        if (count_output != null) count_output.* = worker.transcript.len;
        return true;
    }
    pub fn save(self: Sessions, errors: []u8) bool {
        if (!c.ds4_agent_comms_idle(self.worker)) {
            report(errors, "model is busy", .{});
            return false;
        }
        var sha: [41]u8 = undefined;
        var count: c_int = 0;
        const ok = self.saveNow(&sha, &count, errors);
        if (ok) _ = c.printf("saved session %.8s (%d tokens)\n", &sha, count);
        return ok;
    }
    pub fn showHistory(self: Sessions, turns: c_int, errors: []u8) bool {
        if (!c.ds4_agent_comms_idle(self.worker)) {
            report(errors, "model is busy", .{});
            return false;
        }
        var length: usize = 0;
        const rendered = c.ds4_kvstore_render_tokens_text(self.worker.engine, &self.worker.transcript, &length);
        if (rendered == null) {
            report(errors, "failed to render session text", .{});
            return false;
        }
        defer c.free(rendered);
        c.ds4_agent_history_render(self.worker.engine, Prompts.syntax(self.worker.engine), self.worker, historyEmit, c.isatty(c.STDOUT_FILENO) != 0, rendered, length, turns);
        return true;
    }
    /// Load before replacing the actual transcript/identity. Rebuilding stripped
    /// saves remains visible, and optional history never changes the load result.
    pub fn switchTo(self: Sessions, prefix: [*c]const u8, turns: c_int, errors: []u8) bool {
        const worker = self.worker;
        if (!c.ds4_agent_comms_idle(worker)) {
            report(errors, "model is busy", .{});
            return false;
        }
        var sha: [41]u8 = undefined;
        var path: [*c]u8 = null;
        if (!c.ds4_agent_sessions_find(worker.cache_dir, @intCast(c.ds4_engine_model_id(worker.engine)), prefix, &sha, &path, errors.ptr, errors.len)) return false;
        defer c.free(path);
        var stripped = false;
        var entry = zero(c.ds4_kvstore_entry);
        if (c.ds4_kvstore_read_entry_file(path, &sha, &entry)) {
            stripped = entry.payload_bytes == 0;
            c.ds4_kvstore_entry_free(&entry);
        }
        if (stripped) {
            _ = c.printf("rebuilding stripped session %.8s from rendered text...\n", &sha);
            _ = c.fflush(c.__stdoutp);
        }
        var loaded = zero(c.ds4_tokens);
        defer c.ds4_tokens_free(&loaded);
        var metadata = zero(c.agent_kv_session_meta);
        defer c.ds4_agent_sessions_meta_free(&metadata);
        if (!self.load(path, &sha, null, 0, &loaded, &metadata, errors)) return false;
        c.ds4_agent_runtime_images_clear(worker);
        c.ds4_tokens_free(&worker.transcript);
        worker.transcript = loaded;
        loaded = zero(c.ds4_tokens);
        c.free(worker.identity.title);
        worker.identity.title = duplicate(if (metadata.title != null) text(metadata.title) else "(no user prompt)");
        worker.identity.created_at = if (metadata.created_at != 0) metadata.created_at else @bitCast(@as(i64, c.time(null)));
        worker.identity.sha = sha;
        c.free(worker.identity.legacy_path);
        worker.identity.legacy_path = if (metadata.legacy_identity) duplicate(text(path)) else null;
        worker.datetime_context_injected = true;
        _ = c.pthread_mutex_lock(&worker.mu);
        worker.hints = zero(c.agent_hints);
        worker.hints.applied = c.AGENT_HINTS_UNKNOWN;
        worker.user_activity = true;
        worker.session_dirty = false;
        worker.status.state = c.AGENT_WORKER_IDLE;
        worker.status.ctx_used = worker.transcript.len;
        worker.status.ctx_size = contextSize(worker);
        worker.status.prefill_tps = 0;
        worker.status.greedy_sampling = false;
        worker.status.@"error"[0] = 0;
        c.ds4_agent_comms_wake_locked(worker);
        _ = c.pthread_mutex_unlock(&worker.mu);
        _ = c.printf("switched to session %.8s (%d tokens%s)\n", &sha, worker.transcript.len, @as([*:0]const u8, if (stripped) ", rebuilt from text" else ""));
        if (turns > 0) _ = self.showHistory(turns, errors);
        return true;
    }
    pub fn applyThinking(self: Sessions) void {
        const worker = self.worker;
        _ = c.pthread_mutex_lock(&worker.mu);
        const mode = worker.requested_think;
        worker.think_requested = false;
        _ = c.pthread_mutex_unlock(&worker.mu);
        if (!c.ds4_engine_is_deepseek41(worker.engine) or worker.cfg.*.gen.raw_prompt) {
            emit(worker, "\n/think requires a V4.1 chat session\n");
            return;
        }
        // Restored history can have a different effort from the CLI setting;
        // match the actual prefix, not the current preference.
        var old_length: c_int = 0;
        for (0..101) |level| {
            var candidate = zero(c.ds4_tokens);
            const effort: c.ds4_think_mode = @intCast(c.DS4_THINK_LEVEL_BASE + @as(c_int, @intCast(level)));
            numericPrefix(worker.engine, effort, &candidate);
            if (candidate.len > old_length and c.ds4_tokens_starts_with(&worker.transcript, &candidate)) old_length = candidate.len;
            c.ds4_tokens_free(&candidate);
        }
        var next = zero(c.ds4_tokens);
        defer c.ds4_tokens_free(&next);
        numericPrefix(worker.engine, mode, &next);
        const delta = next.len - old_length;
        if (old_length == 0 or @as(i64, worker.transcript.len) + delta >= contextSize(worker)) {
            emit(worker, "\ncannot change thinking prefix: incompatible session or no context room\n");
            return;
        }
        const changed = delta != 0 or !std.mem.eql(c_int, next.v[0..@intCast(old_length)], worker.transcript.v[0..@intCast(old_length)]);
        if (changed) {
            for (@as(usize, @intCast(old_length))..@intCast(worker.transcript.len)) |index| c.ds4_tokens_push(&next, worker.transcript.v[index]);
            _ = c.pthread_mutex_lock(&worker.mu);
            c.ds4_tokens_free(&worker.transcript);
            worker.transcript = next;
            next = zero(c.ds4_tokens);
            for (0..worker.image_count) |index| worker.images[index].token_start = @truncate(@as(u64, @bitCast(@as(i64, worker.images[index].token_start) + delta)));
            _ = c.pthread_mutex_unlock(&worker.mu);
            c.ds4_session_invalidate(worker.session);
        }
        c.ds4_tokens_free(&next);
        c.ds4_agent_runtime_frontiers_clear(worker);
        _ = c.pthread_mutex_lock(&worker.mu);
        worker.cfg.*.gen.think_mode = mode;
        worker.session_dirty = worker.session_dirty or changed;
        worker.status.ctx_used = worker.transcript.len;
        c.ds4_agent_comms_wake_locked(worker);
        _ = c.pthread_mutex_unlock(&worker.mu);
        print(worker, "\nThinking mode: %s.\n", .{c.ds4_think_mode_name(mode)});
    }
};
