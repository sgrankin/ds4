//! CLI logits/logprob dumps, live-vs-fresh decode checks and perplexity scoring.
//! These use the real public session API; no engine records or copied test
//! implementations are needed. Generated C declarations stay behind cli_c.
const std = @import("std");
const options = @import("cli_options.zig");
const c = options.c;
const Options = options.Options;
const Engine = c.ds4_engine;
const Session = c.ds4_session;
const Tokens = c.ds4_tokens;
const File = c.FILE;
const allocator = options.allocator;
const report = options.report;

fn jsonString(file: [*c]File, bytes: []const u8) void {
    const valid_utf8 = std.unicode.utf8ValidateSlice(bytes);
    _ = c.fputc('"', file);
    for (bytes) |byte| {
        switch (byte) {
            '"', '\\' => {
                _ = c.fputc('\\', file);
                _ = c.fputc(byte, file);
            },
            '\n' => _ = c.fputs("\\n", file),
            '\r' => _ = c.fputs("\\r", file),
            '\t' => _ = c.fputs("\\t", file),
            else => if (byte < 0x20 or (!valid_utf8 and byte >= 0x80)) {
                // A token may be an arbitrary byte fragment. Its bytes array is
                // authoritative; escapes keep this display string valid JSON.
                _ = c.fprintf(file, "\\u%04x", @as(c_uint, byte));
            } else {
                _ = c.fputc(byte, file);
            },
        }
    }
    _ = c.fputc('"', file);
}

fn jsonToken(file: [*c]File, engine: *Engine, token: c_int) void {
    var length: usize = 0;
    const pointer = c.ds4_token_text(engine, token, &length);
    defer c.free(pointer);
    const text: []const u8 = if (length == 0) "" else pointer[0..length];
    _ = c.fprintf(file, "{\"id\":%d,\"text\":", token);
    jsonString(file, text);
    _ = c.fputs(",\"bytes\":[", file);
    for (text, 0..) |byte, index| {
        if (index != 0) _ = c.fputc(',', file);
        _ = c.fprintf(file, "%u", @as(c_uint, byte));
    }
    _ = c.fputs("]}", file);
}

// Every installed callback borrows a stack Progress. Clear BOTH registrations
// before returning, whether sync succeeds or fails.
fn syncPrompt(session: *Session, prompt: *const Tokens, progress: *options.Progress, message: []u8) bool {
    progress.install(session);
    defer options.clearProgress(session);
    return c.ds4_session_sync(session, prompt, message.ptr, message.len) == 0;
}

pub fn runLogits(engine: *Engine, cfg: *const Options, prompt: *const Tokens) c_int {
    var maybe_session: ?*Session = null;
    if (c.ds4_session_create(&maybe_session, engine, cfg.gen.ctx_size) != 0) {
        report("ds4: --dump-logits requires a graph session backend\n", .{});
        return 1;
    }
    const session = maybe_session orelse return 1;
    defer c.ds4_session_free(session);
    if (!options.waitRoute(cfg, session)) return 1;
    var message: [160]u8 = undefined;
    var progress = options.Progress{ .input_tokens = prompt.len, .use_color = c.ds4_log_is_tty(c.__stderrp) };
    if (!syncPrompt(session, prompt, &progress, &message)) {
        report("ds4: prompt processing failed: %s\n", .{&message});
        return 1;
    }
    const vocab = c.ds4_engine_vocab_size(engine);
    const logits = allocator.alloc(f32, @intCast(vocab)) catch return 1;
    defer allocator.free(logits);
    if (c.ds4_session_copy_logits(session, logits.ptr, vocab) != vocab) {
        report("ds4: failed to copy session logits\n", .{});
        return 1;
    }
    const path = cfg.gen.dump_logits_path.?;
    const file = c.fopen(path.ptr, "wb");
    if (file == null) {
        report("ds4: failed to open --dump-logits file: %s\n", .{path.ptr});
        return 1;
    }
    _ = c.fprintf(file, "{\n  \"source\":\"ds4\",\n  \"model\":");
    jsonString(file, std.mem.span(cfg.engine.model_path));
    _ = c.fprintf(
        file,
        ",\n  \"backend\":\"%s\",\n  \"quant_bits\":%d,\n" ++
            "  \"prompt_tokens\":%d,\n  \"ctx\":%d,\n  \"vocab\":%d,\n",
        c.ds4_backend_name(cfg.engine.backend),
        c.ds4_engine_routed_quant_bits(engine),
        prompt.len,
        cfg.gen.ctx_size,
        vocab,
    );
    const argmax = c.ds4_session_argmax(session);
    _ = c.fputs("  \"argmax_token\":", file);
    jsonToken(file, engine, argmax);
    _ = c.fprintf(file, ",\n  \"argmax_logit\":%.9g,\n  \"logits\":[", @as(f64, logits[@intCast(argmax)]));
    for (logits, 0..) |value, index| {
        if (index != 0) _ = c.fputc(',', file);
        if (index % 8 == 0) _ = c.fputs("\n    ", file);
        if (std.math.isFinite(value)) {
            _ = c.fprintf(file, "%.9g", @as(f64, value));
        } else {
            _ = c.fputs("null", file);
        }
    }
    _ = c.fputs("\n  ]\n}\n", file);
    if (c.fclose(file) != 0) {
        report("ds4: failed to close --dump-logits file: %s\n", .{path.ptr});
        return 1;
    }
    return 0;
}

pub fn runLogprobs(engine: *Engine, cfg: *const Options, prompt: *const Tokens) c_int {
    var maybe_session: ?*Session = null;
    if (c.ds4_session_create(&maybe_session, engine, cfg.gen.ctx_size) != 0) {
        report("ds4: --dump-logprobs requires a graph session backend\n", .{});
        return 1;
    }
    const session = maybe_session orelse return 1;
    defer c.ds4_session_free(session);
    if (!options.waitRoute(cfg, session)) return 1;
    var message: [160]u8 = undefined;
    var progress = options.Progress{ .input_tokens = prompt.len, .use_color = c.ds4_log_is_tty(c.__stderrp) };
    if (!syncPrompt(session, prompt, &progress, &message)) {
        report("ds4: prompt processing failed: %s\n", .{&message});
        return 1;
    }
    const path = cfg.gen.dump_logprobs_path.?;
    const file = c.fopen(path.ptr, "wb");
    if (file == null) {
        report("ds4: failed to open --dump-logprobs file: %s\n", .{path.ptr});
        return 1;
    }
    var closed = false;
    defer if (!closed) {
        _ = c.fclose(file);
    };
    const top_k: c_int = @min(if (cfg.gen.dump_logprobs_top_k > 0) cfg.gen.dump_logprobs_top_k else 20, 128);
    const scores = allocator.alloc(c.ds4_token_score, @intCast(top_k)) catch return 1;
    defer allocator.free(scores);
    @memset(scores, std.mem.zeroes(c.ds4_token_score));
    _ = c.fprintf(file, "{\n  \"source\":\"ds4\",\n  \"prompt_tokens\":%d,\n  \"ctx\":%d,\n  \"top_k\":%d,\n  \"steps\":[\n", prompt.len, cfg.gen.ctx_size, top_k);
    const room = c.ds4_session_ctx(session) - c.ds4_session_pos(session);
    const maximum: c_int = if (room <= 1) 0 else @min(cfg.gen.n_predict, room - 1);
    var step: c_int = 0;
    while (step < maximum) : (step += 1) {
        const count = c.ds4_session_top_logprobs(session, scores.ptr, top_k);
        const token = c.ds4_session_argmax(session);
        if (step != 0) _ = c.fputs(",\n", file);
        _ = c.fprintf(file, "    {\"step\":%d,\"selected\":", step);
        jsonToken(file, engine, token);
        _ = c.fputs(",\"top_logprobs\":[", file);
        var index: c_int = 0;
        while (index < count and scores[@intCast(index)].id >= 0) : (index += 1) {
            const score = scores[@intCast(index)];
            if (index != 0) _ = c.fputc(',', file);
            _ = c.fputs("{\"token\":", file);
            jsonToken(file, engine, score.id);
            _ = c.fprintf(file, ",\"logit\":%.9g,\"logprob\":%.9g}", @as(f64, score.logit), @as(f64, score.logprob));
        }
        _ = c.fputs("]}", file);
        if (c.ds4_token_is_stop(engine, token)) break;
        if (c.ds4_session_eval(session, token, &message, message.len) != 0) {
            report("ds4: decode failed while dumping logprobs: %s\n", .{&message});
            return 1;
        }
    }
    _ = c.fputs("\n  ]\n}\n", file);
    const close_result = c.fclose(file);
    closed = true;
    if (close_result != 0) {
        report("ds4: failed to close --dump-logprobs file: %s\n", .{path.ptr});
        return 1;
    }
    return 0;
}

fn diagnosticToken(engine: *Engine, token: c_int) void {
    _ = c.fputs("{\"token\":", c.__stderrp);
    jsonToken(c.__stderrp, engine, token);
    _ = c.fputc('}', c.__stderrp);
}

fn diagnosticTop(engine: *Engine, label: [*:0]const u8, scores: []const c.ds4_token_score, count: c_int) void {
    _ = c.fprintf(c.__stderrp, "%s:", label);
    var index: c_int = 0;
    while (index < count and scores[@intCast(index)].id >= 0) : (index += 1) {
        const score = scores[@intCast(index)];
        _ = c.fputc(' ', c.__stderrp);
        jsonToken(c.__stderrp, engine, score.id);
        _ = c.fprintf(c.__stderrp, "@%.6g", @as(f64, score.logit));
    }
    _ = c.fputc('\n', c.__stderrp);
}

pub fn runDecodeConsistency(engine: *Engine, cfg: *const Options, prompt: *const Tokens) c_int {
    @setFloatMode(.optimized);
    if (cfg.dist) |dist| {
        if (dist.role != c.DS4_DISTRIBUTED_NONE) {
            report("ds4: --decode-consistency is local-session only\n", .{});
            return 1;
        }
    }
    if (cfg.gen.decode_consistency_tokens < 0) {
        report("ds4: --decode-consistency requires a non-negative token count\n", .{});
        return 1;
    }
    const vocab = c.ds4_engine_vocab_size(engine);
    const live_logits = allocator.alloc(f32, @intCast(vocab)) catch {
        report("ds4: out of memory allocating diagnostic logits\n", .{});
        return 1;
    };
    defer allocator.free(live_logits);
    const fresh_logits = allocator.alloc(f32, @intCast(vocab)) catch {
        report("ds4: out of memory allocating diagnostic logits\n", .{});
        return 1;
    };
    defer allocator.free(fresh_logits);
    var prefix = std.mem.zeroes(Tokens);
    c.ds4_tokens_copy(&prefix, prompt);
    defer c.ds4_tokens_free(&prefix);
    var live: ?*Session = null;
    defer if (live) |session| c.ds4_session_free(session);
    if (c.ds4_session_create(&live, engine, cfg.gen.ctx_size) != 0) {
        report("ds4: --decode-consistency requires a graph session backend\n", .{});
        return 1;
    }
    var message: [160]u8 = undefined;
    var progress = options.Progress{ .input_tokens = prompt.len, .use_color = c.ds4_log_is_tty(c.__stderrp) };
    if (!syncPrompt(live.?, prompt, &progress, &message)) {
        report("ds4: prompt processing failed: %s\n", .{&message});
        return 1;
    }
    report("ds4: decode-consistency prompt_tokens=%d check_after=%d\n", .{ prompt.len, cfg.gen.decode_consistency_tokens });
    var step: c_int = 0;
    while (step < cfg.gen.decode_consistency_tokens) : (step += 1) {
        const token = c.ds4_session_argmax(live.?);
        report("ds4: decode-consistency selected[%d]=", .{step});
        diagnosticToken(engine, token);
        _ = c.fputc('\n', c.__stderrp);
        c.ds4_tokens_push(&prefix, token);
        if (c.ds4_session_eval(live.?, token, &message, message.len) != 0) {
            report("ds4: decode failed during consistency check: %s\n", .{&message});
            return 1;
        }
        if (c.ds4_token_is_stop(engine, token)) break;
    }
    var live_top: [10]c.ds4_token_score = undefined;
    const live_count = c.ds4_session_top_logprobs(live.?, &live_top, live_top.len);
    if (c.ds4_session_copy_logits(live.?, live_logits.ptr, vocab) != vocab) {
        report("ds4: failed to copy live logits\n", .{});
        return 1;
    }
    // The fresh comparison is sequential: release the live graph BEFORE creating
    // another graph so this diagnostic does not double its KV/graph residency.
    c.ds4_session_free(live.?);
    live = null;
    var maybe_fresh: ?*Session = null;
    if (c.ds4_session_create(&maybe_fresh, engine, cfg.gen.ctx_size) != 0) {
        report("ds4: failed to create fresh diagnostic session\n", .{});
        return 1;
    }
    const fresh = maybe_fresh orelse return 1;
    defer c.ds4_session_free(fresh);
    progress.input_tokens = prefix.len;
    if (!syncPrompt(fresh, &prefix, &progress, &message)) {
        report("ds4: fresh prompt processing failed: %s\n", .{&message});
        return 1;
    }
    var fresh_top: [10]c.ds4_token_score = undefined;
    const fresh_count = c.ds4_session_top_logprobs(fresh, &fresh_top, fresh_top.len);
    if (c.ds4_session_copy_logits(fresh, fresh_logits.ptr, vocab) != vocab) {
        report("ds4: failed to copy fresh logits\n", .{});
        return 1;
    }
    var sum_squares: f64 = 0;
    var maximum: f32 = 0;
    var maximum_index: usize = 0;
    for (live_logits, fresh_logits, 0..) |left, right, index| {
        const difference = @abs(left - right);
        if (difference > maximum) {
            maximum = difference;
            maximum_index = index;
        }
        sum_squares += @as(f64, difference) * @as(f64, difference);
    }
    report("ds4: decode-consistency compared prefix_tokens=%d vocab=%d max_abs=%.9g at token=%d live=%.9g fresh=%.9g rms=%.9g\n", .{
        prefix.len,                           vocab,                                 @as(f64, maximum),                                   @as(c_int, @intCast(maximum_index)),
        @as(f64, live_logits[maximum_index]), @as(f64, fresh_logits[maximum_index]), @sqrt(sum_squares / @as(f64, @floatFromInt(vocab))),
    });
    diagnosticTop(engine, "ds4: live_top", &live_top, live_count);
    diagnosticTop(engine, "ds4: fresh_top", &fresh_top, fresh_count);
    return if (live_count > 0 and fresh_count > 0 and live_top[0].id == fresh_top[0].id) 0 else 1;
}

pub fn runPerplexity(engine: *Engine, cfg: *const Options) c_int {
    @setFloatMode(.optimized);
    var tokens = std.mem.zeroes(Tokens);
    defer c.ds4_tokens_free(&tokens);
    {
        const text = options.readPromptFile(cfg.gen.perplexity_file_path.?, true) orelse return 2;
        defer allocator.free(text);
        c.ds4_tokenize_text(engine, text.ptr, &tokens);
    }
    // Seed with real context to stay on the normal Metal prefill path; score
    // immediately after the same fixed 32-token prefix as the original CLI.
    const prefix_length: c_int = 32;
    if (tokens.len <= prefix_length) {
        report("ds4: --perplexity-file needs more than %d tokens\n", .{prefix_length});
        return 1;
    }
    var scored = tokens.len - prefix_length;
    if (cfg.gen.n_predict > 0) scored = @min(scored, cfg.gen.n_predict);
    scored = @min(scored, cfg.gen.ctx_size - prefix_length);
    if (scored <= 0) {
        report("ds4: context too small for perplexity scoring\n", .{});
        return 1;
    }
    var maybe_session: ?*Session = null;
    if (c.ds4_session_create(&maybe_session, engine, cfg.gen.ctx_size) != 0) {
        report("ds4: --perplexity-file requires a graph session backend\n", .{});
        return 1;
    }
    const session = maybe_session orelse return 1;
    defer c.ds4_session_free(session);
    if (!options.waitRoute(cfg, session)) return 1;
    // sync borrows its input only for the call; the owning tokens allocation
    // stays live throughout scoring, so no temporary token-vector copy is needed.
    const prefix = Tokens{ .v = tokens.v, .len = prefix_length, .cap = prefix_length };
    var message: [160]u8 = undefined;
    if (c.ds4_session_sync(session, &prefix, &message, message.len) != 0) {
        report("ds4: perplexity initial token failed: %s\n", .{&message});
        return 1;
    }
    const ids = tokens.v[0..@as(usize, @intCast(tokens.len))];
    var nll: f64 = 0;
    var step: c_int = 0;
    while (step < scored) : (step += 1) {
        const index = prefix_length + step;
        var score: c.ds4_token_score = undefined;
        const token = ids[@intCast(index)];
        if (c.ds4_session_token_logprob(session, token, &score) == 0) {
            report("ds4: failed to score token %d\n", .{index});
            return 1;
        }
        nll -= @as(f64, score.logprob);
        if (@rem(step + 1, 256) == 0 or step + 1 == scored) {
            report("ds4: perplexity scored %d/%d\r", .{ step + 1, scored });
            _ = c.fflush(c.__stderrp);
        }
        if (step + 1 < scored and c.ds4_session_eval(session, token, &message, message.len) != 0) {
            report("\nds4: perplexity decode failed at token %d: %s\n", .{ index, &message });
            return 1;
        }
    }
    _ = c.fputc('\n', c.__stderrp);
    const average = nll / @as(f64, @floatFromInt(scored));
    _ = c.printf("tokens=%d scored=%d nll=%.9f avg_nll=%.9f ppl=%.9f\n", tokens.len, scored, nll, average, @exp(average));
    return 0;
}
