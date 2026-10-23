//! Private summary inference and transactional publication of the compacted
//! transcript. Worker is the actual shared owner, not a snapshot. Until the
//! final rebuild succeeds, every failure invalidates private live KV and keeps
//! the real conversation and its image embeddings available for the next turn.
const std = @import("std");
const c = @import("agent_runtime_c");
const System = @import("system.zig");
const Prompts = @import("prompts.zig");
const Trace = @import("trace.zig").Trace;
const allocator = std.heap.c_allocator;

fn zero(comptime T: type) T {
    return std.mem.zeroes(T);
}
fn text(value: [*c]const u8) []const u8 {
    return if (value == null) &.{} else std.mem.span(value);
}
fn report(errors: []u8, comptime pattern: [*:0]const u8, arguments: anytype) void {
    if (errors.len != 0) _ = @call(.auto, c.snprintf, .{ errors.ptr, errors.len, pattern } ++ arguments);
}
fn fatal() noreturn {
    c.perror("ds4-agent: malloc");
    c.exit(1);
}
fn now() f64 {
    var stamp: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &stamp);
    return @as(f64, @floatFromInt(stamp.tv_sec)) + @as(f64, @floatFromInt(stamp.tv_nsec)) * 1e-9;
}
fn prerope() bool {
    for ([_][*:0]const u8{ "DS4_AGENT_CONTEXT_SUFFIX_REUSE", "DS4_AGENT_CONTEXT_SUFFIX_PREROPE" }) |name| {
        const value = c.getenv(name);
        if (value == null or !(c.strcasecmp(value, "true") == 0 or c.strcasecmp(value, "yes") == 0 or c.strcmp(value, "1") == 0)) return false;
    }
    return true;
}
fn special(engine: ?*c.ds4_engine, rendered: [*:0]const u8) c_int {
    var tokens = zero(c.ds4_tokens);
    defer c.ds4_tokens_free(&tokens);
    c.ds4_tokenize_rendered_chat(engine, rendered, &tokens);
    return if (tokens.len == 1) tokens.v[0] else -1;
}
fn emit(worker: *c.agent_worker, value: []const u8) void {
    c.ds4_agent_comms_publish(worker, value.ptr, value.len);
}
fn print(worker: *c.agent_worker, comptime pattern: [*:0]const u8, arguments: anytype) void {
    const length = @call(.auto, c.snprintf, .{ @as([*c]u8, null), @as(usize, 0), pattern } ++ arguments);
    if (length <= 0) return;
    const output = allocator.allocSentinel(u8, @intCast(length), 0) catch fatal();
    defer allocator.free(output);
    _ = @call(.auto, c.snprintf, .{ output.ptr, output.len + 1, pattern } ++ arguments);
    emit(worker, output);
}
fn append(output: *std.ArrayList(u8), bytes: []const u8) void {
    output.appendSlice(allocator, bytes) catch fatal();
    output.append(allocator, 0) catch fatal();
    output.items.len -= 1;
}

pub const Summary = struct {
    worker: *c.agent_worker,

    fn trace(self: Summary) Trace {
        return .{ .worker = self.worker };
    }
    fn contextSize(self: Summary) c_int {
        var current = if (self.worker.session != null) c.ds4_session_ctx(self.worker.session) else 0;
        if (current <= 0 and self.worker.cfg != null) current = self.worker.cfg.*.gen.ctx_size;
        return current;
    }
    fn begin(self: Summary) void {
        const worker = self.worker;
        _ = c.pthread_mutex_lock(&worker.mu);
        defer _ = c.pthread_mutex_unlock(&worker.mu);
        worker.status.state = c.AGENT_WORKER_COMPACTING;
        worker.progress_direct = false;
        worker.progress_started_at = now();
        worker.status.prefill_done = 0;
        worker.status.prefill_total = 0;
        worker.status.prefill_tps = 0;
        worker.status.generated = 0;
        worker.status.gen_tps = 0;
        worker.status.greedy_sampling = false;
        c.ds4_agent_comms_wake_locked(worker);
    }
    fn syncPrivate(self: Summary, prompt: *const c.ds4_tokens, image_count: usize, errors: []u8) c_int {
        const worker = self.worker;
        c.ds4_session_set_progress(worker.session, c.ds4_agent_comms_progress, worker);
        c.ds4_session_set_display_progress(worker.session, c.ds4_agent_comms_progress, worker);
        c.ds4_session_set_cancel(worker.session, c.ds4_agent_comms_cancel, worker);
        defer {
            c.ds4_session_set_cancel(worker.session, null, null);
            c.ds4_session_set_progress(worker.session, null, null);
            c.ds4_session_set_display_progress(worker.session, null, null);
        }
        return if (image_count != 0)
            c.ds4_session_sync_multimodal(worker.session, prompt, worker.images, image_count, errors.ptr, errors.len)
        else
            c.ds4_session_sync(worker.session, prompt, errors.ptr, errors.len);
    }
    fn interrupted(self: Summary, errors: []u8) void {
        c.ds4_session_invalidate(self.worker.session);
        report(errors, "interrupted", .{});
        c.ds4_agent_turn_system_status(self.worker, "Compaction interrupted; keeping the previous conversation state.");
    }
    fn generated(self: Summary, count: c_int, started: f64) void {
        const elapsed = now() - started;
        _ = c.pthread_mutex_lock(&self.worker.mu);
        defer _ = c.pthread_mutex_unlock(&self.worker.mu);
        self.worker.status.generated = count;
        self.worker.status.gen_tps = if (elapsed > 0) @as(f64, @floatFromInt(count)) / elapsed else 0;
        self.worker.status.greedy_sampling = false;
        c.ds4_agent_comms_wake_locked(self.worker);
    }

    /// Perform the full compaction exchange. Historical RAM/SSD endpoints stay
    /// immutable while a private prompt temporarily replaces the live graph.
    pub fn compact(self: Summary, reason: [*c]const u8, open_assistant: [*c]c_int, errors: []u8) bool {
        const worker = self.worker;
        const bottom = worker.transcript.len;
        if (bottom <= 0) return true;
        var system = zero(c.ds4_tokens);
        defer c.ds4_tokens_free(&system);
        System.build(worker, &system);
        if (bottom <= system.len) return true;

        // Preserve the real endpoint before the private summary changes KV.
        // Historical endpoints may precede the latest user in the retained tail.
        if (prerope()) _ = c.ds4_agent_runtime_capture(worker);
        print(worker, "\n\x1b[1;95mCOMPACTING\x1b[0m %s: summarizing durable task state\n\x1b[38;5;245m", .{if (text(reason).len != 0) reason else @as([*c]const u8, "context")});
        var showing_summary = true;
        defer if (showing_summary) emit(worker, "\x1b[0m\n");

        const context = self.contextSize();
        var plan = zero(c.ds4_agent_compaction_plan);
        defer c.ds4_agent_compaction_free(&plan);
        if (!c.ds4_agent_compaction_prepare(&plan, worker.engine, &worker.transcript, &system, worker.images, worker.image_count, context, reason, errors.ptr, errors.len)) return false;
        self.trace().log("compaction summary prefix=%d total=%d retained_unsummarized=%d", .{ plan.summary_bottom, bottom, bottom - plan.summary_bottom });
        self.begin();
        if (plan.summary_room < 256) {
            report(errors, "not enough context left to request compaction summary", .{});
            return false;
        }
        const synchronized = self.syncPrivate(&plan.prompt, plan.summary_images, errors);
        if (synchronized == c.DS4_SESSION_SYNC_INTERRUPTED) {
            self.interrupted(errors);
            emit(worker, "\x1b[0m\n");
            showing_summary = false;
            c.ds4_agent_comms_clear_interrupt(worker);
            return false;
        }
        if (synchronized != 0) {
            c.ds4_session_invalidate(worker.session);
            return false;
        }

        // Until rebuild succeeds the graph contains private instructions and
        // summary tokens, while Worker.transcript remains the real history.
        var summary: std.ArrayList(u8) = .empty;
        defer summary.deinit(allocator);
        var eval_error: [160]u8 = @splat(0);
        const dsml_id = special(worker.engine, "｜DSML｜");
        const started = now();
        var count: c_int = 0;
        while (count < plan.summary_max) : (count += 1) {
            if (c.ds4_agent_comms_interrupted(worker)) {
                report(errors, "interrupted", .{});
                c.ds4_session_invalidate(worker.session);
                emit(worker, "\x1b[0m\n");
                showing_summary = false;
                c.ds4_agent_turn_system_status(worker, "Compaction interrupted; keeping the previous conversation state.");
                c.ds4_agent_comms_clear_interrupt(worker);
                return false;
            }
            const token = c.ds4_session_argmax(worker.session);
            if (c.ds4_token_is_stop_for_think_mode(worker.engine, token, c.DS4_THINK_NONE) or token == dsml_id) {
                if (token == dsml_id and summary.items.len != 0 and summary.items[summary.items.len - 1] == '<') {
                    summary.items.len -= 1;
                    summary.items.ptr[summary.items.len] = 0;
                }
                self.trace().log("compaction summary stopped before control token id=%d", .{token});
                break;
            }
            if (c.ds4_session_eval(worker.session, token, &eval_error, eval_error.len) != 0) {
                report(errors, "%s", .{&eval_error});
                c.ds4_session_invalidate(worker.session);
                return false;
            }
            var length: usize = 0;
            const decoded = c.ds4_token_text(worker.engine, token, &length);
            const bytes = if (length == 0) &.{} else decoded[0..length];
            append(&summary, bytes);
            emit(worker, bytes);
            c.free(decoded);
            self.generated(count + 1, started);
        }
        emit(worker, "\x1b[0m\n");
        showing_summary = false;
        c.ds4_tokens_free(&plan.prompt);
        if (summary.items.len == 0 or summary.items[0] == 0) {
            report(errors, "compaction summary was empty", .{});
            c.ds4_session_invalidate(worker.session);
            return false;
        }
        self.trace().text("compaction-summary", summary.items);
        const rebuilt = c.ds4_agent_compaction_rebuild(&plan, worker.engine, &worker.transcript, &system, worker.images, worker.image_count, context, prerope(), summary.items.ptr, open_assistant, errors.ptr, errors.len);
        self.trace().tokens("compacted_transcript", &plan.compacted, 0);
        if (!rebuilt) {
            c.ds4_session_invalidate(worker.session);
            return false;
        }
        const tail_start = plan.tail_start;
        const tail_destination = plan.tail_dst_start;
        const resumed_start = plan.resumed_start;

        const old_images = worker.images;
        const old_count = worker.image_count;
        const old_capacity = worker.image_cap;
        const new_images: []c.ds4_vision_span = if (old_count == 0) &.{} else allocator.alloc(c.ds4_vision_span, old_count) catch fatal();
        var own_new_images = true;
        defer if (own_new_images and old_count != 0) allocator.free(new_images);
        const kept_storage = if (old_count == 0) null else c.calloc(old_count, @sizeOf(bool));
        defer c.free(kept_storage);
        if (old_count != 0 and kept_storage == null) {
            c.ds4_session_invalidate(worker.session);
            report(errors, "out of memory retaining compacted images", .{});
            return false;
        }
        const kept: []bool = if (old_count == 0) &.{} else @as([*]bool, @ptrCast(kept_storage.?))[0..old_count];
        var new_count: usize = 0;
        for (0..old_count) |index| {
            const image = old_images[index];
            const end = @as(u64, image.token_start) + image.embedding.token_count;
            if (image.token_start >= @as(u32, @bitCast(tail_start)) and end <= @as(u64, @intCast(bottom))) {
                new_images[new_count] = image;
                new_images[new_count].token_start = @intCast(@as(i64, tail_destination) + image.token_start - tail_start);
                new_count += 1;
                kept[index] = true;
            }
        }
        var reused: c_int = 0;
        if (prerope() and old_count == 0)
            reused = c.ds4_agent_runtime_try_suffix(worker, &plan.compacted, &system, errors.ptr, errors.len);
        if (reused < 0) {
            c.ds4_session_invalidate(worker.session);
            return false;
        }
        print(worker, "\x1b[1;95mCOMPACTING\x1b[0m rebuilding context: old=%d summary+tail=%d tail=%d\n", .{ bottom, plan.compacted.len, bottom - tail_start });

        var old_transcript = zero(c.ds4_tokens);
        defer c.ds4_tokens_free(&old_transcript);
        c.ds4_tokens_copy(&old_transcript, &worker.transcript);
        c.ds4_tokens_free(&worker.transcript);
        worker.transcript = plan.compacted;
        plan.compacted = zero(c.ds4_tokens);
        worker.images = if (old_count == 0) null else new_images.ptr;
        worker.image_count = new_count;
        worker.image_cap = old_count;
        if (prerope() and reused == 0) c.ds4_session_invalidate(worker.session);
        if (reused == 0 and old_count == 0) _ = c.ds4_agent_runtime_compact_prefix(worker, &system);
        // Reuse the existing whole native sync ABI; importing Turn here would
        // create a cycle when Turn composes this private transaction.
        if (c.ds4_agent_turn_sync(worker, &worker.transcript, true, errors.ptr, errors.len) != 0) {
            c.ds4_session_invalidate(worker.session);
            c.ds4_tokens_free(&worker.transcript);
            worker.transcript = old_transcript;
            old_transcript = zero(c.ds4_tokens);
            worker.images = old_images;
            worker.image_count = old_count;
            worker.image_cap = old_capacity;
            return false;
        }
        own_new_images = false;
        for (0..old_count) |index| {
            if (!kept[index]) c.ds4_vision_embedding_free(&old_images[index].embedding);
        }
        c.free(old_images);
        _ = c.pthread_mutex_lock(&worker.mu);
        Prompts.hintsCompacted(&worker.hints);
        _ = c.pthread_mutex_unlock(&worker.mu);
        worker.last_system_prompt_reminder_at = worker.transcript.len;
        c.ds4_tokens_free(&old_transcript);
        c.ds4_tokens_free(&system);

        // Never insert a tool observation inside an assistant still emitting.
        const job_update = if (open_assistant == null) c.ds4_agent_executor_compaction_observation(&worker.tools) else null;
        if (job_update != null) {
            defer c.free(job_update);
            c.ds4_chat_append_message(worker.engine, &worker.transcript, "tool", job_update);
            worker.session_dirty = true;
            self.trace().text("tool-after-compaction", text(job_update));
            emit(worker, "\x1b[90mCOMPACTING added bash job update after rebuild\x1b[0m\n");
        }
        self.trace().log("compacted reason=\"%s\" old=%d new=%d tail_start=%d tail=%d suffix_reused=%d", .{ if (reason != null) reason else @as([*c]const u8, ""), bottom, worker.transcript.len, tail_start, bottom - tail_start, @as(c_int, @intFromBool(reused > 0)) });
        if (open_assistant != null) open_assistant.* = resumed_start;
        return true;
    }

    pub fn compactAndExport(self: Summary, reason: [*c]const u8, errors: []u8) bool {
        return self.compact(reason, null, errors) and c.ds4_agent_runtime_export(self.worker, errors.ptr, errors.len);
    }
};
