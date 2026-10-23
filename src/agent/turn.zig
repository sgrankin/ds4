//! Complete chat/raw generation turn over the actual Worker and selected
//! native Stream/Tools/Result/Context owners. No artificial tool-round ceiling:
//! transcript, pressure/compaction, cancellation and model stop decide progress.
const std = @import("std");
const c = @import("agent_runtime_c");
const Prompts = @import("prompts.zig");
const Trace = @import("trace.zig").Trace;
fn zero(comptime T: type) T {
    return std.mem.zeroes(T);
}
fn text(value: [*c]const u8) []const u8 {
    return if (value == null) &.{} else std.mem.span(value);
}
fn report(errors: []u8, value: [*c]const u8) void {
    if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "%s", value);
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
fn emit(context: ?*anyopaque, value: [*c]const u8, length: usize) callconv(.c) void {
    c.ds4_agent_comms_publish(@ptrCast(@alignCast(context.?)), value, length);
}
pub const Turn = struct {
    worker: *c.agent_worker,
    fn config(self: Turn) *c.agent_config {
        return @ptrCast(self.worker.cfg);
    }
    fn trace(self: Turn) Trace {
        return .{ .worker = self.worker };
    }
    fn contextSize(self: Turn) c_int {
        var current = if (self.worker.session != null) c.ds4_session_ctx(self.worker.session) else 0;
        if (current <= 0 and self.worker.cfg != null) current = self.config().gen.ctx_size;
        return current;
    }
    fn idle(self: Turn) c_int {
        c.ds4_agent_comms_status(self.worker, c.AGENT_WORKER_IDLE);
        return 0;
    }
    fn fail(self: Turn, message: [*c]const u8) c_int {
        c.ds4_agent_comms_error(self.worker, message);
        return 1;
    }
    fn settle(self: Turn, errors: [*c]const u8, fallback: [*:0]const u8) c_int {
        if (c.strcmp(errors, "interrupted") == 0) {
            c.ds4_agent_comms_clear_interrupt(self.worker);
            return self.idle();
        }
        return self.fail(if (text(errors).len != 0) errors else fallback);
    }
    fn cancelled(self: Turn) bool {
        return c.ds4_agent_comms_interrupted(self.worker);
    }
    fn begin(self: Turn) void {
        _ = c.pthread_mutex_lock(&self.worker.mu);
        self.worker.interrupt = false;
        self.worker.status.@"error"[0] = 0;
        c.ds4_agent_comms_wake_locked(self.worker);
        _ = c.pthread_mutex_unlock(&self.worker.mu);
    }
    fn setGreedy(self: Turn, enabled: bool) void {
        _ = c.pthread_mutex_lock(&self.worker.mu);
        defer _ = c.pthread_mutex_unlock(&self.worker.mu);
        if (self.worker.status.greedy_sampling != enabled) {
            self.worker.status.greedy_sampling = enabled;
            c.ds4_agent_comms_wake_locked(self.worker);
        }
    }
    fn generated(self: Turn, count: c_int, started: f64) void {
        const elapsed = now() - started;
        _ = c.pthread_mutex_lock(&self.worker.mu);
        defer _ = c.pthread_mutex_unlock(&self.worker.mu);
        self.worker.status.generated = count;
        self.worker.status.gen_tps = if (elapsed > 0) @as(f64, @floatFromInt(count)) / elapsed else 0;
        c.ds4_agent_comms_wake_locked(self.worker);
    }
    fn endAssistant(self: Turn) void {
        if (Prompts.syntax(self.worker.engine) != c.AGENT_TOOL_SYNTAX_GLM)
            c.ds4_tokens_push(&self.worker.transcript, c.ds4_token_eos(self.worker.engine));
    }
    fn reserve(self: Turn) c_int {
        return c.ds4_agent_compaction_reserve(self.worker.engine, self.contextSize());
    }
    fn compact(self: Turn, reason: [*:0]const u8, errors: []u8) bool {
        return c.ds4_agent_turn_compact_private(self.worker, reason, null, errors.ptr, errors.len) and
            c.ds4_agent_runtime_export(self.worker, errors.ptr, errors.len);
    }
    fn compactIfNeeded(self: Turn, reason: [*:0]const u8, errors: []u8) bool {
        return !c.ds4_agent_compaction_should_compact(self.worker.engine, self.contextSize(), self.worker.transcript.len) or
            self.compact(reason, errors);
    }
    pub fn appendUser(self: Turn, value: [*c]const u8, errors: []u8) bool {
        var message = zero(c.ds4_tokens);
        var system = zero(c.ds4_tokens);
        defer c.ds4_tokens_free(&message);
        c.ds4_chat_append_message(self.worker.engine, &message, "user", value);
        c.ds4_agent_runtime_build_system(self.worker, &system);
        const context = self.contextSize();
        var accepted = message.len < context - system.len - 4;
        c.ds4_tokens_free(&system);
        if (!accepted) {
            if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "user message is too large for --ctx %d; use a larger context or a smaller message", context);
        } else if (message.len >= context - self.worker.transcript.len - 4) {
            accepted = self.compact("make room for incoming user message", errors);
            if (accepted and message.len >= context - self.worker.transcript.len - 4) {
                report(errors, "user message does not fit after compaction; use a larger --ctx");
                accepted = false;
            }
        }
        if (accepted) for (0..@intCast(message.len)) |index| c.ds4_tokens_push(&self.worker.transcript, message.v[index]);
        return accepted;
    }
    fn prefillStatus(self: Turn, cached: c_int, remaining: c_int, clear_greedy: bool) void {
        const worker = self.worker;
        _ = c.pthread_mutex_lock(&worker.mu);
        defer _ = c.pthread_mutex_unlock(&worker.mu);
        const label = if (worker.status.state == c.AGENT_WORKER_PREFILL) worker.status.prefill_label else c.ds4_agent_comms_next_label();
        worker.status.state = c.AGENT_WORKER_PREFILL;
        worker.progress_base = cached;
        worker.progress_direct = false;
        worker.progress_started_at = now();
        worker.status.prefill_done = 0;
        worker.status.prefill_total = remaining;
        worker.status.prefill_label = label;
        worker.status.prefill_tps = 0;
        worker.status.generated = 0;
        worker.status.gen_tps = 0;
        if (clear_greedy) worker.status.greedy_sampling = false;
        c.ds4_agent_comms_wake_locked(worker);
    }
    fn installCallbacks(self: Turn, progress: bool) void {
        const worker = self.worker;
        const callback: c.ds4_session_progress_fn = if (progress) c.ds4_agent_comms_progress else null;
        const context: ?*anyopaque = if (progress) worker else null;
        c.ds4_session_set_progress(worker.session, callback, context);
        c.ds4_session_set_display_progress(worker.session, callback, context);
        c.ds4_session_set_cancel(worker.session, c.ds4_agent_comms_cancel, worker);
    }
    /// The cache-saving synchronization action now owns direct native progress
    /// callbacks. Session/model algorithms remain public selected owners.
    pub fn sync(self: Turn, tokens: *const c.ds4_tokens, publish_progress: bool, errors: []u8) c_int {
        const worker = self.worker;
        _ = c.ds4_agent_runtime_cache_restore(worker, tokens);
        const old = c.ds4_session_pos(worker.session);
        const common = c.ds4_session_common_prefix(worker.session, tokens);
        const cached = if (common == old and tokens.len >= old) common else 0;
        var remaining = tokens.len - cached;
        if (remaining < 0) remaining = tokens.len;
        if (publish_progress) self.prefillStatus(cached, remaining, false);
        self.installCallbacks(publish_progress);
        const started = now();
        const result = if (worker.image_count != 0) multimodal: {
            if (!c.ds4_agent_runtime_images_fit(worker, tokens)) {
                report(errors, "image spans do not fit the agent transcript");
                break :multimodal @as(c_int, 1);
            }
            break :multimodal c.ds4_session_sync_multimodal(worker.session, tokens, worker.images, worker.image_count, errors.ptr, errors.len);
        } else c.ds4_session_sync(worker.session, tokens, errors.ptr, errors.len);
        const finished = now();
        // Preserve the original callback retirement order.
        c.ds4_session_set_cancel(worker.session, null, null);
        c.ds4_session_set_progress(worker.session, null, null);
        c.ds4_session_set_display_progress(worker.session, null, null);
        self.trace().log("prefill sync done prompt=%d cached=%d suffix=%d rc=%d %.3f ms", .{ tokens.len, cached, remaining, result, (finished - started) * 1000 });
        if (result == 0) c.ds4_agent_runtime_cache_endpoint(worker);
        return result;
    }
    fn rewind(self: Turn, position: c_int, errors: []u8) c_int {
        c.ds4_session_rewind(self.worker.session, position);
        var prefix = zero(c.ds4_tokens);
        defer c.ds4_tokens_free(&prefix);
        c.ds4_tokens_copy(&prefix, c.ds4_session_tokens(self.worker.session));
        return if (c.ds4_session_common_prefix(self.worker.session, &prefix) != prefix.len) self.sync(&prefix, false, errors) else 0;
    }
    fn finishToken(self: Turn, token: c_int, count: *c_int, started: f64, stream: *c.agent_stream_renderer, evaluate: bool, errors: []u8) bool {
        const worker = self.worker;
        c.ds4_tokens_push(&worker.transcript, token);
        var length: usize = 0;
        const decoded = c.ds4_token_text(worker.engine, token, &length);
        self.trace().token(token, if (length == 0) &.{} else decoded[0..length], count.* + 1);
        c.ds4_agent_stream_text(stream, decoded, length, false);
        c.free(decoded);
        count.* += 1;
        if (evaluate and c.ds4_session_eval(worker.session, token, errors.ptr, errors.len) != 0) {
            c.ds4_session_invalidate(worker.session);
            return false;
        }
        self.generated(count.*, started);
        return true;
    }
    fn forceText(self: Turn, value: [*:0]const u8, maximum: c_int, count: *c_int, started: f64, stream: *c.agent_stream_renderer, errors: []u8) bool {
        var tokens = zero(c.ds4_tokens);
        defer c.ds4_tokens_free(&tokens);
        c.ds4_tokenize_text(self.worker.engine, value, &tokens);
        if (tokens.len > maximum - count.*) {
            if (errors.len != 0) _ = c.snprintf(errors.ptr, errors.len, "not enough generation room to force %s", value);
            return false;
        }
        var index: c_int = 0;
        while (index < tokens.len and count.* < maximum) : (index += 1)
            if (!self.finishToken(tokens.v[@intCast(index)], count, started, stream, true, errors)) return false;
        return true;
    }
    fn seed(self: Turn) u64 {
        if (self.config().gen.seed != 0) return self.config().gen.seed;
        return @as(u64, @bitCast(@as(i64, c.time(null)))) ^ (@as(u64, @intCast(c.getpid())) << 32) ^ @as(u64, c.clock());
    }
    fn decodeBlock(self: Turn, token: c_int, remaining: c_int, mode_greedy: bool, rng: *u64, output: *[17]c_int, errors: []u8) c_int {
        const worker = self.worker;
        const generation = &self.config().gen;
        if (c.ds4_engine_decode_packet_size(worker.engine) > 1 and c.getenv("DS4_MTP_SPEC_DISABLE") == null)
            return c.ds4_session_eval_speculative(worker.session, token, remaining, c.ds4_token_eos(worker.engine), if (mode_greedy) 0 else generation.temperature, 0, if (mode_greedy) 1 else generation.top_p, if (mode_greedy) 0 else generation.min_p, rng, output, output.len, errors.ptr, errors.len);
        if (c.ds4_session_eval(worker.session, token, errors.ptr, errors.len) != 0) return -1;
        output[0] = token;
        return 1;
    }
    pub fn raw(self: Turn, user: [*c]const u8) c_int {
        const worker = self.worker;
        const generation = &self.config().gen;
        self.begin();
        var prompt = zero(c.ds4_tokens);
        defer c.ds4_tokens_free(&prompt);
        c.ds4_agent_runtime_images_clear(worker);
        c.ds4_tokenize_text(worker.engine, if (user != null) user else "", &prompt);
        if (prompt.len <= 0) return self.fail("raw prompt tokenized to no tokens");
        self.trace().text("raw-prompt", text(user));
        self.trace().tokens("raw_prompt_tokens", &prompt, 0);
        var errors: [160]u8 = @splat(0);
        const synced = self.sync(&prompt, true, &errors);
        if (synced == c.DS4_SESSION_SYNC_INTERRUPTED) {
            c.ds4_agent_comms_clear_interrupt(worker);
            return self.idle();
        }
        if (synced != 0) return self.fail(if (errors[0] != 0) &errors else "raw prompt processing failed");
        c.ds4_tokens_free(&worker.transcript);
        c.ds4_tokens_copy(&worker.transcript, &prompt);
        const room = c.ds4_session_ctx(worker.session) - c.ds4_session_pos(worker.session);
        var maximum = generation.n_predict;
        if (room <= 1) maximum = 0 else if (maximum > room - 1) maximum = room - 1;
        var rng = self.seed();
        var count: c_int = 0;
        const started = now();
        _ = c.pthread_mutex_lock(&worker.mu);
        worker.status.state = c.AGENT_WORKER_GENERATING;
        worker.status.prefill_tps = 0;
        worker.status.generated = 0;
        worker.status.gen_tps = 0;
        worker.status.greedy_sampling = generation.temperature <= 0;
        c.ds4_agent_comms_wake_locked(worker);
        _ = c.pthread_mutex_unlock(&worker.mu);
        while (count < maximum and !self.cancelled()) {
            var token = c.ds4_session_sample(worker.session, generation.temperature, 0, generation.top_p, generation.min_p, &rng);
            if (c.ds4_token_is_stop(worker.engine, token)) break;
            const start = c.ds4_session_pos(worker.session);
            var block: [17]c_int = undefined;
            const length = self.decodeBlock(token, maximum - count, false, &rng, &block, &errors);
            if (length < 0) return self.fail(if (errors[0] != 0) &errors else "raw decode failed");
            if (length == 0) return self.fail("raw decode returned no tokens");
            var stop = false;
            var index: c_int = 0;
            while (index < length and count < maximum) : (index += 1) {
                token = block[@intCast(index)];
                if (c.ds4_token_is_stop(worker.engine, token)) {
                    c.ds4_session_rewind(worker.session, start + index);
                    stop = true;
                    break;
                }
                var bytes: usize = 0;
                const decoded = c.ds4_token_text(worker.engine, token, &bytes);
                self.trace().token(token, if (bytes == 0) &.{} else decoded[0..bytes], count + 1);
                c.ds4_tokens_push(&worker.transcript, token);
                c.ds4_agent_comms_publish(worker, decoded, bytes);
                c.free(decoded);
                count += 1;
            }
            self.generated(count, started);
            if (stop) break;
        }
        if (self.cancelled()) c.ds4_agent_comms_clear_interrupt(worker);
        return self.idle();
    }
    pub fn run(self: Turn, user: [*c]const u8) c_int {
        const worker = self.worker;
        const options = self.config();
        const think = c.ds4_think_mode_for_context(options.gen.think_mode, options.gen.ctx_size);
        self.begin();
        var compact_error: [160]u8 = @splat(0);
        if (prerope() and c.ds4_session_enable_suffix_prerope(worker.session, &compact_error, compact_error.len) != 0)
            return self.fail(&compact_error);
        // Accept idle edits BEFORE compaction/mirroring can replace the file
        // and BEFORE appending the newly submitted user message.
        if (options.gen.live_context_path != null) {
            var errors: [160]u8 = @splat(0);
            if (c.ds4_agent_runtime_import(worker, &errors, errors.len) < 0) {
                if (c.strcmp(&errors, "interrupted") == 0) {
                    c.ds4_agent_comms_clear_interrupt(worker);
                    return self.idle();
                }
                _ = c.snprintf(&compact_error, compact_error.len, "live context edit rejected: %.120s", &errors);
                return self.fail(&compact_error);
            }
        }
        if (!self.compactIfNeeded("soft limit before user turn", &compact_error)) return self.settle(&compact_error, "context compaction failed");
        Prompts.datetimeContext(worker);
        Prompts.hintsContext(worker);
        self.trace().text("user", text(user));
        if (!self.appendUser(user, &compact_error)) return self.settle(&compact_error, "");
        if (worker.identity.title == null) {
            worker.identity.title = c.ds4_agent_sessions_title_prompt(user, 0);
            worker.identity.created_at = @bitCast(@as(i64, c.time(null)));
            c.ds4_agent_sessions_identity_sha(worker.identity.title, worker.identity.created_at, &worker.identity.sha);
        }
        var rng = self.seed();
        _ = c.pthread_mutex_lock(&worker.mu);
        worker.user_activity = true;
        worker.session_dirty = true;
        c.ds4_agent_comms_wake_locked(worker);
        _ = c.pthread_mutex_unlock(&worker.mu);
        // There is no artificial tool-call ceiling. Each completed assistant
        // stanza gets its terminator, tool observation and another model round.
        var carried: c_int = 0;
        var continue_assistant = false;
        var resume_think = false;
        var resume_start: c_int = 0;
        var round: c_int = 0;
        while (true) : (round += 1) {
            if (!continue_assistant and !self.compactIfNeeded("soft limit before assistant continuation", &compact_error))
                return self.settle(&compact_error, "context compaction failed");
            if (!continue_assistant) {
                Prompts.systemReminder(worker);
                Prompts.hintsContext(worker);
            }
            if (!c.ds4_agent_runtime_export(worker, &compact_error, compact_error.len)) return self.fail(&compact_error);
            const assistant_start = if (continue_assistant) resume_start else worker.transcript.len;
            if (!continue_assistant) c.ds4_chat_append_assistant_prefix(worker.engine, &worker.transcript, think);
            const old = c.ds4_session_pos(worker.session);
            const common = c.ds4_session_common_prefix(worker.session, &worker.transcript);
            const cached = if (common == old and worker.transcript.len >= old) common else 0;
            const remaining = worker.transcript.len - cached;
            self.trace().log("prefill tool_round=%d transcript=%d prompt=%d cached=%d suffix=%d think=%s", .{ round, worker.transcript.len, worker.transcript.len, cached, remaining, c.ds4_think_mode_name(think) });
            self.trace().tokens("prefill_suffix", &worker.transcript, cached);
            self.prefillStatus(cached, remaining, true);
            var errors: [160]u8 = undefined;
            self.installCallbacks(true);
            const sync_start = now();
            const synced = if (worker.image_count != 0)
                c.ds4_session_sync_multimodal(worker.session, &worker.transcript, worker.images, worker.image_count, &errors, errors.len)
            else
                c.ds4_session_sync(worker.session, &worker.transcript, &errors, errors.len);
            const sync_end = now();
            c.ds4_session_set_cancel(worker.session, null, null);
            c.ds4_session_set_progress(worker.session, null, null);
            c.ds4_session_set_display_progress(worker.session, null, null);
            self.trace().log("prefill sync done tool_round=%d prompt=%d cached=%d suffix=%d rc=%d %.3f ms", .{ round, worker.transcript.len, cached, remaining, synced, (sync_end - sync_start) * 1000 });
            if (synced == c.DS4_SESSION_SYNC_INTERRUPTED) {
                Prompts.publishSystemStatus(worker, "Model reading interrupted; the model may only be aware of the prefix processed so far.");
                self.endAssistant();
                c.ds4_agent_comms_clear_interrupt(worker);
                return self.idle();
            }
            if (synced != 0) return self.fail(&errors);
            var maximum = options.gen.n_predict - carried;
            const room = c.ds4_session_ctx(worker.session) - c.ds4_session_pos(worker.session) - self.reserve();
            var context_limited = room < maximum;
            if (context_limited) maximum = room;
            if (maximum <= 0 and options.gen.n_predict > 0)
                return self.fail("context has no generation room after compaction; use a larger --ctx");
            const color = c.isatty(c.STDOUT_FILENO) != 0;
            const in_think = if (continue_assistant) resume_think else c.ds4_think_mode_enabled(think);
            continue_assistant = false;
            var renderer = zero(c.agent_token_renderer);
            renderer.engine = worker.engine;
            renderer.worker = worker;
            renderer.emit = emit;
            renderer.format_thinking = c.ds4_think_mode_enabled(think);
            renderer.format_markdown = color;
            renderer.in_think = in_think;
            renderer.use_color = color;
            renderer.last_output_newline = true;
            const kind = Prompts.syntax(worker.engine);
            var parser = zero(c.agent_dsml_parser);
            parser.syntax = kind;
            parser.state = c.AGENT_DSML_SEARCH;
            var parser_alive = true;
            defer if (parser_alive) c.ds4_agent_tool_parser_free(&parser);
            var stream = zero(c.agent_stream_renderer);
            stream.renderer = &renderer;
            stream.parser = &parser;
            stream.syntax = kind;
            stream.in_think = in_think;
            var forcer = zero(c.agent_edit_upto_forcer);
            var got_tool = false;
            var malformed = false;
            var preflight_error = false;
            var model_stopped = false;
            var count: c_int = 0;
            var lookahead: c_int = 0;
            const started = now();
            c.ds4_agent_comms_status(worker, c.AGENT_WORKER_GENERATING);
            var status_greedy = false;
            while (count < maximum and !self.cancelled()) {
                c.ds4_agent_comms_apply_power(worker);
                const greedy = c.ds4_agent_stream_greedy(&stream);
                if (greedy != status_greedy) {
                    self.setGreedy(greedy);
                    status_greedy = greedy;
                }
                var token = c.ds4_session_sample(worker.session, if (greedy) 0 else options.gen.temperature, 0, if (greedy) 1 else options.gen.top_p, if (greedy) 0 else options.gen.min_p, &rng);
                if (c.ds4_token_is_stop_for_think_mode(worker.engine, token, think)) {
                    model_stopped = true;
                    if (kind == c.AGENT_TOOL_SYNTAX_GLM and token != c.ds4_token_eos(worker.engine))
                        self.trace().log("glm assistant generation stopped before control token id=%d", .{token});
                    break;
                }
                const block_start = c.ds4_session_pos(worker.session);
                var block: [17]c_int = undefined;
                const length = self.decodeBlock(token, maximum - count, greedy, &rng, &block, &errors);
                if (length < 0) return self.fail(&errors);
                if (length == 0) return self.fail("decode returned no tokens");
                var stop_block = false;
                var restart_sampling = false;
                var index: c_int = 0;
                while (index < length and count < maximum) : (index += 1) {
                    token = block[@intCast(index)];
                    if (c.ds4_token_is_stop_for_think_mode(worker.engine, token, think)) {
                        model_stopped = true;
                        c.ds4_session_rewind(worker.session, block_start + index);
                        stop_block = true;
                        break;
                    }
                    var bytes: usize = 0;
                    const decoded = c.ds4_token_text(worker.engine, token, &bytes);
                    if (options.edit_upto and c.ds4_agent_edit_force_upto(&forcer, &parser, decoded, bytes)) {
                        self.trace().log("edit old auto-upto replaced token=%d text=%.*s", .{ token, @as(c_int, @intCast(@min(bytes, 80))), decoded });
                        c.free(decoded);
                        if (self.rewind(block_start + index, &errors) != 0 or
                            !self.forceText("[upto]\n", maximum, &count, started, &stream, &errors)) return self.fail(&errors);
                        restart_sampling = true;
                        break;
                    }
                    c.free(decoded);
                    if (!self.finishToken(token, &count, started, &stream, false, &errors)) return self.fail(&errors);
                    const next_greedy = c.ds4_agent_stream_greedy(&stream);
                    if (next_greedy != status_greedy) {
                        self.setGreedy(next_greedy);
                        status_greedy = next_greedy;
                    }
                    if (parser.state == c.AGENT_DSML_DONE) {
                        got_tool = true;
                        stop_block = true;
                    } else if (stream.tool_preflight_error) {
                        preflight_error = true;
                        stop_block = true;
                    } else if (parser.state == c.AGENT_DSML_ERROR or stream.dsml_in_think) {
                        malformed = true;
                        stop_block = true;
                    }
                    if (stop_block) {
                        if (index + 1 < length) c.ds4_session_rewind(worker.session, block_start + index + 1);
                        break;
                    }
                    // Opportunistic drafts already match greedy continuation;
                    // only exact sampling needs the new parser distribution.
                    if (index + 1 < length and c.ds4_engine_mtp_exact_sampling(worker.engine) and
                        options.gen.temperature > 0 and next_greedy != greedy)
                    {
                        // Reevaluate this boundary token to restore logits;
                        // later draft tokens used the old parser mode.
                        if (self.rewind(block_start + index, &errors) != 0 or c.ds4_session_eval(worker.session, token, &errors, errors.len) != 0)
                            return self.fail(&errors);
                        restart_sampling = true;
                        break;
                    }
                }
                if (stop_block) break;
                if (restart_sampling) continue;
                // Resolve a split delimiter before rebuilding the parser. At
                // most half the64-token safety reserve, never beyond -n.
                if (context_limited and count == maximum and lookahead < 32 and c.ds4_agent_stream_lookahead(&stream)) {
                    maximum += 1;
                    lookahead += 1;
                    if (maximum == options.gen.n_predict - carried) context_limited = false;
                }
            }
            self.trace().log("generation finished tool_round=%d generated=%d carried=%d context_limited=%d", .{ round, count, carried, @as(c_int, @intFromBool(context_limited)) });
            const interrupted = self.cancelled();
            const partial_tool = stream.dsml_active or stream.dsml_start_len > 0 or parser.state != c.AGENT_DSML_SEARCH;
            c.ds4_agent_stream_text(&stream, null, 0, true);
            c.ds4_agent_presentation_finish(&renderer);
            self.setGreedy(false);
            if (!got_tool and parser.state == c.AGENT_DSML_DONE and parser.calls.len > 0) got_tool = true;
            if (interrupted) {
                self.endAssistant();
                c.ds4_agent_tool_parser_free(&parser);
                parser_alive = false;
                Prompts.publishSystemStatus(worker, "Stopped by user");
                c.ds4_agent_comms_clear_interrupt(worker);
                return self.idle();
            }
            if (context_limited and count == maximum and !model_stopped and !got_tool and !malformed and !preflight_error) {
                carried += count;
                if (partial_tool) {
                    // No tool from this round ran. Drop its partial arguments
                    // before asking the model to retry with smaller arguments.
                    worker.transcript.len = assistant_start;
                    c.ds4_session_invalidate(worker.session);
                }
                c.ds4_agent_tool_parser_free(&parser);
                parser_alive = false;
                self.trace().log("generation context boundary: generated=%d partial_tool=%d", .{ count, @as(c_int, @intFromBool(partial_tool)) });
                var open_assistant = assistant_start;
                if (!c.ds4_agent_turn_compact_private(worker, "generation reached compaction reserve", if (partial_tool) null else &open_assistant, &compact_error, compact_error.len)) return self.settle(&compact_error, "");
                if (partial_tool) c.ds4_chat_append_message(worker.engine, &worker.transcript, "user", "The last tool call was interrupted by context compaction and was NOT executed. " ++
                    "Continue the user's task, respecting its constraints. If a tool is needed, retry with smaller arguments or smaller edits.\n") else {
                    continue_assistant = true;
                    resume_start = open_assistant;
                    resume_think = stream.in_think;
                }
                continue;
            }
            if (stream.dsml_in_think) {
                got_tool = false;
                malformed = true;
                preflight_error = false;
                _ = c.snprintf(&parser.@"error", parser.@"error".len, "tool calling is not allowed inside <think></think>");
            } else if (!malformed and parser.state == c.AGENT_DSML_ERROR) malformed = true else if (!got_tool and !malformed and !preflight_error and !interrupted and
                (parser.state == c.AGENT_DSML_STRUCTURAL or parser.state == c.AGENT_DSML_PARAM_VALUE))
            {
                malformed = true;
                _ = c.snprintf(&parser.@"error", parser.@"error".len, "%s", @as([*:0]const u8, if (kind == c.AGENT_TOOL_SYNTAX_QWEN) "incomplete Qwen tool call" else if (kind == c.AGENT_TOOL_SYNTAX_GLM) "incomplete GLM tool call" else "incomplete DSML tool call"));
            }
            self.endAssistant();
            _ = c.ds4_agent_runtime_capture(worker);
            if (!c.ds4_agent_runtime_export(worker, &compact_error, compact_error.len)) return self.fail(&compact_error);
            if (!got_tool and !malformed and !preflight_error) {
                c.ds4_agent_tool_parser_free(&parser);
                parser_alive = false;
                return self.idle();
            }
            var observation = zero(c.agent_tool_observation);
            c.ds4_agent_observation_init(&observation);
            var observation_alive = true;
            defer if (observation_alive) c.ds4_agent_observation_free(&observation);
            if (preflight_error) {
                c.ds4_agent_observation_puts(&observation, "Tool error: ");
                c.ds4_agent_observation_puts(&observation, if (stream.tool_preflight_error_msg[0] != 0)
                    &stream.tool_preflight_error_msg
                else
                    "edit old selector failed before new was generated");
                c.ds4_agent_observation_puts(&observation, "\n");
            } else if (malformed) {
                c.ds4_agent_observation_puts(&observation, if (kind == c.AGENT_TOOL_SYNTAX_QWEN) "Tool error: invalid Qwen tool call: " else if (kind == c.AGENT_TOOL_SYNTAX_GLM) "Tool error: invalid GLM tool call: " else "Tool error: invalid DSML tool call: ");
                c.ds4_agent_observation_puts(&observation, if (parser.@"error"[0] != 0) &parser.@"error" else "parse error");
                c.ds4_agent_observation_puts(&observation, "\n");
                const reminder = switch (kind) {
                    c.AGENT_TOOL_SYNTAX_QWEN => c.agent_qwen_syntax_reminder,
                    c.AGENT_TOOL_SYNTAX_GLM => c.agent_glm_syntax_reminder,
                    c.AGENT_TOOL_SYNTAX_DSML41 => c.agent_dsml41_syntax_reminder,
                    else => c.agent_dsml_syntax_reminder,
                };
                c.ds4_agent_observation_puts(&observation, @ptrCast(reminder));
            } else {
                c.ds4_agent_observation_free(&observation);
                c.ds4_agent_observation_execute(&observation, worker, worker.engine, &parser.calls);
                if (c.ds4_agent_result_after_tools(worker, &observation) == c.DS4_AGENT_RESULT_INTERRUPTED) {
                    c.ds4_agent_observation_free(&observation);
                    observation_alive = false;
                    c.ds4_agent_tool_parser_free(&parser);
                    parser_alive = false;
                    c.ds4_agent_comms_clear_interrupt(worker);
                    return self.idle();
                }
            }
            var append_error: [160]u8 = @splat(0);
            const appended = c.ds4_agent_result_append(worker, &observation, &append_error, append_error.len);
            // Retire both local owners before the queued-input wait, as C did.
            c.ds4_agent_observation_free(&observation);
            observation_alive = false;
            c.ds4_agent_tool_parser_free(&parser);
            parser_alive = false;
            if (appended != c.DS4_AGENT_RESULT_OK) {
                if (appended == c.DS4_AGENT_RESULT_INTERRUPTED) {
                    c.ds4_agent_comms_clear_interrupt(worker);
                    return self.idle();
                }
                return self.fail(if (append_error[0] != 0) &append_error else "unable to append tool observation");
            }
            carried = 0;
            const queued = c.ds4_agent_comms_request_drain(worker);
            defer c.free(queued);
            if (text(queued).len != 0) {
                self.trace().text("queued_user", text(queued));
                if (!self.appendUser(queued, &compact_error)) return self.settle(&compact_error, "");
                _ = c.pthread_mutex_lock(&worker.mu);
                worker.user_activity = true;
                worker.session_dirty = true;
                c.ds4_agent_comms_wake_locked(worker);
                _ = c.pthread_mutex_unlock(&worker.mu);
            }
        }
    }
};
