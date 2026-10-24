//! Metal CLI policy: one-shot generation or a persistent session/transcript.
//! Graph/cache mechanics remain behind public C APIs; terminal and SDK
//! declarations are generated from cli_api.h, never copied into this source.
const std = @import("std");
const Prefix = @import("prompt_prefix_append.zig");
const shared = @import("cli_options.zig");
const diagnostics = @import("cli_diagnostics.zig");
const c = shared.c;
const Options = shared.Options;
const Generation = shared.Generation;
const Engine = c.ds4_engine;
const Session = c.ds4_session;
const allocator = shared.allocator;
const report = shared.report;
const fatal = shared.fatal;
const cString = shared.cString;
const now = shared.nowSeconds;

fn same(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
fn rawBytes(ptr: [*c]const u8, len: usize) []const u8 {
    return if (len == 0) "" else ptr[0..len];
}
fn usage(file: [*c]c.FILE, topic: ?[:0]const u8) void {
    c.ds4_help_print(file, c.DS4_HELP_DS4, cString(topic));
}

var interrupted: c.sig_atomic_t = 0;
var distributed_busy: c.sig_atomic_t = 0;
var drain_notice: c.sig_atomic_t = 0;
fn signalFlag(flag: *c.sig_atomic_t) *volatile c.sig_atomic_t {
    return flag;
}
fn interruptRequested() bool {
    return signalFlag(&interrupted).* != 0;
}
fn clearInterrupt() void {
    signalFlag(&interrupted).* = 0;
    signalFlag(&drain_notice).* = 0;
}
fn sigintHandler(sig: c_int) callconv(.c) void {
    _ = sig;
    signalFlag(&interrupted).* = 1;
    if (signalFlag(&distributed_busy).* != 0 and signalFlag(&drain_notice).* == 0) {
        signalFlag(&drain_notice).* = 1;
        const text = "\nds4: stopping after the distributed cluster finishes the current token/chunk...\n";
        // write is async-signal-safe; do not allocate or format in this handler.
        _ = c.write(c.STDERR_FILENO, text.ptr, text.len);
    }
}
fn busySet(cfg: *const Options, busy: bool) void {
    if (!shared.coordinator(cfg)) return;
    signalFlag(&distributed_busy).* = @intFromBool(busy);
    if (!busy) signalFlag(&drain_notice).* = 0;
}

fn integer(text: [:0]const u8, option: [:0]const u8, allow_zero: bool) c_int {
    var end: [*c]u8 = null;
    const value = c.strtol(text.ptr, &end, 10);
    if (text.len == 0 or end[0] != 0 or value < 0 or (!allow_zero and value == 0) or value > std.math.maxInt(i32))
        fatal(2, "ds4: invalid value for %s: %s\n", .{ option.ptr, text.ptr });
    return @intCast(value);
}
fn unsigned(text: [:0]const u8, option: [:0]const u8) u64 {
    var end: [*c]u8 = null;
    const value = c.strtoull(text.ptr, &end, 10);
    if (text.len == 0 or end[0] != 0 or value == 0) fatal(2, "ds4: invalid value for %s: %s\n", .{ option.ptr, text.ptr });
    return value;
}
fn decimal(text: [:0]const u8, option: [:0]const u8, minimum: f32, maximum: f32) f32 {
    var end: [*c]u8 = null;
    const value = c.strtof(text.ptr, &end);
    if (text.len == 0 or end[0] != 0 or !std.math.isFinite(value) or value < minimum or value > maximum)
        fatal(2, "ds4: invalid value for %s: %s\n", .{ option.ptr, text.ptr });
    return value;
}
fn backend(text: [:0]const u8) c.ds4_backend {
    if (same(text, "metal")) return c.DS4_BACKEND_METAL;
    if (same(text, "cuda")) return c.DS4_BACKEND_CUDA;
    if (same(text, "cpu")) return c.DS4_BACKEND_CPU;
    fatal(2, "ds4: invalid backend: %s\nds4: valid backends are: metal, cuda, cpu\n", .{text.ptr});
}
fn warnThink(gen: *const Generation, name: [*:0]const u8) void {
    if (gen.think_mode == c.DS4_THINK_MAX and shared.effectiveThink(gen) != c.DS4_THINK_MAX)
        c.ds4_log(c.__stderrp, c.DS4_LOG_WARNING, "ds4: warning: %s needs --ctx >= %u; ctx=%d uses normal thinking instead\n", name, c.ds4_think_max_min_context(), gen.ctx_size);
}
fn logMemory(cfg: *const Options, chunk: u32) void {
    @setFloatMode(.optimized);
    const memory = c.ds4_context_memory_estimate_with_prefill_mode(cfg.engine.backend, cfg.gen.ctx_size, chunk, cfg.engine.ssd_streaming);
    const kv = memory.raw_bytes +% memory.compressed_bytes;
    const total = kv +% memory.scratch_bytes;
    const color = c.ds4_log_is_tty(c.__stderrp);
    const green: [*:0]const u8 = if (color) "\x1b[32m" else "";
    const bright: [*:0]const u8 = if (color) "\x1b[1;32m" else "";
    const reset: [*:0]const u8 = if (color) "\x1b[0m" else "";
    report("%sds4: memory: KV %.2f GiB (raw %.2f + compressed %.2f) + buffers %.2f GiB = %s%.2f GiB context%s\n", .{
        green,                                                         @as(f64, @floatFromInt(kv)) / 1073741824,                   @as(f64, @floatFromInt(memory.raw_bytes)) / 1073741824,
        @as(f64, @floatFromInt(memory.compressed_bytes)) / 1073741824, @as(f64, @floatFromInt(memory.scratch_bytes)) / 1073741824, bright,
        @as(f64, @floatFromInt(total)) / 1073741824,                   reset,
    });
    report("%sds4: memory detail: ctx=%d prefill_cap=%u raw_kv_rows=%u compressed_kv_rows=%u backend=%s%s\n", .{ green, cfg.gen.ctx_size, memory.prefill_cap, memory.raw_cap, memory.comp_cap, c.ds4_backend_name(cfg.engine.backend), reset });
}
fn renderedPrompt(prompt: ?[:0]const u8) bool {
    const text = prompt orelse return false;
    for ([_][]const u8{ "<｜begin▁of▁sentence｜>", "<｜User｜>", "[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>" }) |prefix|
        if (std.mem.startsWith(u8, text, prefix)) return true;
    return false;
}
fn buildPrompt(engine: *Engine, gen: *const Generation, out: *c.ds4_tokens) void {
    const mode = shared.effectiveThink(gen);
    if (gen.raw_prompt) {
        c.ds4_tokenize_text(engine, cString(gen.prompt) orelse "", out);
    } else if (renderedPrompt(gen.prompt)) {
        c.ds4_tokenize_rendered_chat(engine, cString(gen.prompt), out);
    } else if (gen.prefix.count == 0) {
        c.ds4_encode_chat_prompt(engine, cString(gen.system), cString(gen.prompt), mode, out);
    } else {
        c.ds4_chat_begin(engine, out);
        c.ds4_chat_append_think_prefix(engine, out, mode);
        if (gen.system) |system| {
            if (system.len != 0) {
                c.ds4_chat_append_message(engine, out, "system", system.ptr);
            }
        }
        Prefix.append(c, engine, out, &gen.prefix);
        c.ds4_chat_append_message(engine, out, "user", cString(gen.prompt) orelse "");
        c.ds4_chat_append_assistant_prefix(engine, out, mode);
    }
}

const Printer = struct {
    engine: *Engine,
    file: [*c]c.FILE,
    format_thinking: bool,
    in_think: bool,
    use_color: bool,
    color_open: bool = false,
    last_newline: bool = true,
    pending: [16]u8 = undefined,
    pending_len: usize = 0,

    fn init(engine: *Engine, mode: c.ds4_think_mode) Printer {
        return .{ .engine = engine, .file = c.__stdoutp, .format_thinking = c.ds4_think_mode_enabled(mode), .in_think = c.ds4_think_mode_enabled(mode), .use_color = c.isatty(c.fileno(c.__stdoutp)) != 0 };
    }
    fn resetColor(self: *Printer) void {
        if (self.use_color and self.color_open) {
            _ = c.fputs("\x1b[0m", self.file);
            self.color_open = false;
        }
    }
    fn character(self: *Printer, byte: u8) void {
        if (self.in_think and self.use_color and !self.color_open) {
            _ = c.fputs("\x1b[90m", self.file);
            self.color_open = true;
        }
        _ = c.fputc(byte, self.file);
        self.last_newline = byte == '\n';
    }
    fn process(self: *Printer, text: []const u8, finish: bool) void {
        const bytes = allocator.alloc(u8, self.pending_len + text.len) catch return;
        defer allocator.free(bytes);
        @memcpy(bytes[0..self.pending_len], self.pending[0..self.pending_len]);
        @memcpy(bytes[self.pending_len..], text);
        self.pending_len = 0;
        var index: usize = 0;
        while (index < bytes.len) {
            const rest = bytes[index..];
            if (std.mem.startsWith(u8, rest, "<think>")) {
                self.in_think = true;
                index += "<think>".len;
            } else if (std.mem.startsWith(u8, rest, "</think>")) {
                self.in_think = false;
                self.resetColor();
                if (!self.last_newline) {
                    _ = c.fputc('\n', self.file);
                    self.last_newline = true;
                }
                index += "</think>".len;
            } else if (!finish and rest[0] == '<' and
                ((rest.len < "<think>".len and std.mem.startsWith(u8, "<think>", rest)) or
                    (rest.len < "</think>".len and std.mem.startsWith(u8, "</think>", rest))))
            {
                if (rest.len < self.pending.len) {
                    @memcpy(self.pending[0..rest.len], rest);
                    self.pending_len = rest.len;
                }
                break;
            } else {
                self.character(rest[0]);
                index += 1;
            }
        }
    }
    fn write(self: *Printer, text: []const u8) void {
        if (self.format_thinking) self.process(text, false) else if (text.len != 0) {
            _ = c.fwrite(text.ptr, 1, text.len, self.file);
            self.last_newline = text[text.len - 1] == '\n';
        }
    }
    fn token(self: *Printer, id: c_int) void {
        var length: usize = 0;
        const piece = c.ds4_token_text(self.engine, id, &length);
        self.write(rawBytes(piece, length));
        _ = c.fflush(self.file);
        c.free(piece);
    }
    fn done(self: *Printer) void {
        if (self.format_thinking) {
            self.process("", true);
            self.resetColor();
        }
        _ = c.fflush(self.file);
        if (!self.last_newline) {
            _ = c.fputc('\n', self.file);
            self.last_newline = true;
        }
        _ = c.fflush(self.file);
    }
};
fn printToken(context: ?*anyopaque, token: c_int) callconv(.c) void {
    const printer: *Printer = @ptrCast(@alignCast(context orelse return));
    printer.token(token);
}
fn generationDone(context: ?*anyopaque) callconv(.c) void {
    const printer: *Printer = @ptrCast(@alignCast(context orelse return));
    printer.done();
}

fn randomSeed(seed: u64) u64 {
    return if (seed != 0) seed else @as(u64, @bitCast(@as(i64, c.time(null)))) ^
        (@as(u64, @intCast(c.getpid())) << 32) ^ @as(u64, c.clock());
}
// Chat consumes every emitted token into KV, including the last. One-shot
// output may stop immediately after printing its final token, as in C.
fn maxTokens(cfg: *const Options, session: *Session) c_int {
    const room = c.ds4_session_ctx(session) - c.ds4_session_pos(session);
    return if (room <= 1) 0 else @min(cfg.gen.n_predict, room - 1);
}
fn decode(engine: *Engine, cfg: *const Options, session: *Session, printer: *Printer, transcript: ?*c.ds4_tokens, mode: c.ds4_think_mode, maximum: c_int, rng: *u64) ?c_int {
    var generated: c_int = 0;
    var message: [160]u8 = undefined;
    while (generated < maximum and !interruptRequested()) {
        const token = c.ds4_session_sample(session, cfg.gen.temperature, cfg.gen.top_k, cfg.gen.top_p, cfg.gen.min_p, rng);
        if (c.ds4_token_is_stop_for_think_mode(engine, token, mode)) break;
        if (c.ds4_engine_decode_packet_size(engine) > 1 and c.getenv("DS4_MTP_SPEC_DISABLE") == null) {
            var tokens: [17]c_int = undefined;
            busySet(cfg, true);
            const count = c.ds4_session_eval_speculative(session, token, maximum - generated, c.ds4_token_eos(engine), cfg.gen.temperature, cfg.gen.top_k, cfg.gen.top_p, cfg.gen.min_p, rng, &tokens, tokens.len, &message, message.len);
            busySet(cfg, false);
            if (count < 0) {
                report("ds4: decode failed: %s\n", .{&message});
                return null;
            }
            var stop = false;
            for (tokens[0..@as(usize, @intCast(count))]) |id| {
                if (c.ds4_token_is_stop_for_think_mode(engine, id, mode)) {
                    stop = true;
                    break;
                }
                if (transcript) |out| c.ds4_tokens_push(out, id);
                printer.token(id);
                generated += 1;
                if (generated >= maximum) break;
            }
            if (stop) break;
        } else {
            if (transcript) |out| c.ds4_tokens_push(out, token);
            printer.token(token);
            generated += 1;
            if (transcript == null and (generated >= maximum or interruptRequested())) continue;
            busySet(cfg, true);
            const result = c.ds4_session_eval(session, token, &message, message.len);
            busySet(cfg, false);
            if (result != 0) {
                report("ds4: decode failed: %s\n", .{&message});
                if (transcript != null) c.ds4_session_invalidate(session);
                return null;
            }
        }
    }
    return generated;
}
fn timing(input: c_int, generated: c_int, prefill: f64, decode_seconds: f64) void {
    @setFloatMode(.optimized);
    c.ds4_log(c.__stderrp, c.DS4_LOG_TIMING, "ds4: prefill: %.2f t/s, generation: %.2f t/s\n", if (prefill > 0) @as(f64, @floatFromInt(input)) / prefill else @as(f64, 0), if (decode_seconds > 0) @as(f64, @floatFromInt(generated)) / decode_seconds else @as(f64, 0));
}
fn sampledGeneration(engine: *Engine, cfg: *const Options, prompt: *const c.ds4_tokens) c_int {
    var session: ?*Session = null;
    if (c.ds4_session_create(&session, engine, cfg.gen.ctx_size) != 0) {
        report("ds4: sampled CLI generation requires a session backend\n", .{});
        return 1;
    }
    defer c.ds4_session_free(session);
    if (!shared.waitRoute(cfg, session.?)) return 1;
    // Pay first-submission GPU cost before the prefill timer, matching workers.
    c.ds4_session_gpu_warmup(session);
    const mode = shared.effectiveThink(&cfg.gen);
    var printer = Printer.init(engine, mode);
    var progress = shared.Progress{ .input_tokens = prompt.len, .use_color = c.ds4_log_is_tty(c.__stderrp) };
    var message: [160]u8 = undefined;
    const prefill_start = now();
    progress.install(session.?);
    busySet(cfg, true);
    const result = c.ds4_session_sync(session, prompt, &message, message.len);
    busySet(cfg, false);
    shared.clearProgress(session.?);
    if (result != 0) {
        report("ds4: prompt processing failed: %s\n", .{&message});
        return 1;
    }
    const prefill_end = now();
    const maximum = maxTokens(cfg, session.?);
    var rng = randomSeed(cfg.gen.seed);
    const start = now();
    const generated = decode(engine, cfg, session.?, &printer, null, mode, maximum, &rng) orelse return 1;
    const end = now();
    printer.done();
    if (interruptRequested()) clearInterrupt();
    timing(prompt.len, generated, prefill_end - prefill_start, end - start);
    return 0;
}

fn generation(engine: *Engine, cfg: *const Options) c_int {
    var prompt = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&prompt);
    buildPrompt(engine, &cfg.gen, &prompt);
    if (cfg.gen.metal_graph_test) return c.ds4_engine_metal_graph_test(engine, &prompt);
    if (cfg.gen.metal_graph_full_test) return c.ds4_engine_metal_graph_full_test(engine, &prompt);
    if (cfg.gen.metal_graph_prompt_test) return c.ds4_engine_metal_graph_prompt_test(engine, &prompt, cfg.gen.ctx_size);
    if (cfg.gen.dump_logits_path != null) return diagnostics.runLogits(engine, cfg, &prompt);
    if (cfg.gen.dump_logprobs_path != null) return diagnostics.runLogprobs(engine, cfg, &prompt);
    if (cfg.gen.decode_consistency_tokens > 0) return diagnostics.runDecodeConsistency(engine, cfg, &prompt);
    var result: c_int = 0;
    if (cfg.gen.head_test) result = c.ds4_engine_head_test(engine, &prompt);
    if (result == 0 and cfg.gen.first_token_test) result = c.ds4_engine_first_token_test(engine, &prompt);
    if (cfg.gen.dump_tokens) c.ds4_engine_dump_tokens(engine, &prompt);
    if (cfg.gen.dump_tokens or cfg.gen.head_test or cfg.gen.first_token_test) {
        if (result == 0) report("ds4: diagnostic run completed on the native %s path.\n", .{c.ds4_backend_name(cfg.engine.backend)});
        return result;
    }
    // TP leaders must use session sync/eval so workers remain in lockstep. The
    // override permits like-for-like single-node versus TP validation.
    if (shared.coordinator(cfg) or cfg.engine.tp.role == c.DS4_TP_LEADER or
        c.getenv("DS4_CLI_FORCE_SESSION") != null or c.ds4_engine_overthinking_penalty_enabled(engine) or
        cfg.gen.temperature > 0 or c.ds4_engine_decode_packet_size(engine) > 1)
        return sampledGeneration(engine, cfg, &prompt);
    var printer = Printer.init(engine, shared.effectiveThink(&cfg.gen));
    var progress = shared.Progress{ .input_tokens = prompt.len, .use_color = c.ds4_log_is_tty(c.__stderrp) };
    return c.ds4_engine_generate_argmax(engine, &prompt, cfg.gen.n_predict, cfg.gen.ctx_size, printToken, generationDone, &printer, shared.progressCallback, &progress);
}
fn trim(text: [*c]u8) [:0]u8 {
    var start: usize = 0;
    const length = c.strlen(text);
    while (start < length and c.isspace(text[start]) != 0) start += 1;
    var end = length;
    while (end > start and c.isspace(text[end - 1]) != 0) end -= 1;
    text[end] = 0;
    return text[start..end :0];
}
fn replHelp() void {
    for ([_][*:0]const u8{
        "Commands:",
        "  /help          Show this help.",
        "  /think [N]     Use normal thinking, or V4.1 effort 0..100 (0 disables thinking).",
        "  /think-max     Use maximum thinking (V4.1: 100; V4: requires ctx >= 393216).",
        "  /nothink       Disable thinking mode.",
        "  /ctx N         Set context size for following prompts.",
        "  /power N       Set GPU duty cycle percentage, 1..100.",
        "  /steer F       Set FFN steering for subsequent tokens; no value shows it.",
        "  /read FILE     Submit a text file, PNG, or JPEG.",
        "  /quit, /exit   Leave the prompt.",
        "  Ctrl+C         Stop generation and return to the prompt.",
    }) |line| _ = c.puts(line);
}
fn chatInit(engine: *Engine, chat: *c.ds4_cli_chat, cfg: *const Options) bool {
    chat.* = std.mem.zeroes(c.ds4_cli_chat);
    chat.initial_system_text = if (cfg.gen.system) |system| system.len != 0 else false;
    c.ds4_chat_begin(engine, &chat.transcript);
    chat.think_prefix_pos = chat.transcript.len;
    _ = c.ds4_cli_chat_apply_think_prefix(engine, chat, shared.effectiveThink(&cfg.gen));
    if (cfg.gen.system) |system| if (system.len != 0) {
        c.ds4_chat_append_message(engine, &chat.transcript, "system", system.ptr);
    };
    Prefix.append(c, engine, &chat.transcript, &cfg.gen.prefix);
    if (c.ds4_cli_chat_create_session(engine, chat, cfg.gen.ctx_size) != 0) {
        c.ds4_tokens_free(&chat.transcript);
        return false;
    }
    if (cfg.gen.prefix.count == 0) return true;
    if (!shared.waitRoute(cfg, chat.session.?)) {
        c.ds4_cli_chat_free(chat);
        return false;
    }
    var message: [160]u8 = @splat(0);
    var progress = shared.Progress{ .input_tokens = chat.transcript.len, .use_color = c.ds4_log_is_tty(c.__stderrp) };
    progress.install(chat.session.?);
    busySet(cfg, true);
    const result = c.ds4_session_sync(chat.session, &chat.transcript, &message, message.len);
    busySet(cfg, false);
    shared.clearProgress(chat.session.?);
    if (result != 0) {
        report("ds4: prefix prefill failed: %s\n", .{if (message[0] != 0) message[0..].ptr else "unknown error"});
        c.ds4_cli_chat_free(chat);
        return false;
    }
    return true;
}
// Tentative user/assistant markers roll back on prompt failure. Interrupted
// turns with no emitted answer also roll back images and invalidate live KV.
fn chatTurn(engine: *Engine, cfg: *Options, chat: *c.ds4_cli_chat, user: [:0]const u8, image: ?*c.ds4_vision_embedding) c_int {
    const session = chat.session orelse {
        report("ds4: no active interactive KV cache\n", .{});
        return 1;
    };
    const mode = c.ds4_think_mode_for_context(cfg.gen.think_mode, chat.ctx_size);
    if (!c.ds4_cli_chat_apply_think_prefix(engine, chat, mode)) return 1;
    const rollback_len = chat.transcript.len;
    const rollback_images = chat.image_count;
    if (image) |embedding| {
        const span = c.ds4_cli_chat_add_image(chat);
        const parts = [_][*c]const u8{ "", user.ptr };
        var message: [160]u8 = @splat(0);
        if (span == null or c.ds4_chat_append_multimodal_message(engine, &chat.transcript, "user", &parts, embedding, 1, span, &message, message.len) == 0) {
            c.ds4_cli_chat_trim_images(chat, rollback_images);
            chat.transcript.len = rollback_len;
            report("ds4: failed to add image: %s\n", .{if (message[0] != 0) message[0..].ptr else "out of memory"});
            return 1;
        }
    } else c.ds4_chat_append_message(engine, &chat.transcript, "user", user.ptr);
    c.ds4_chat_append_assistant_prefix(engine, &chat.transcript, mode);
    const old_pos = c.ds4_session_pos(session);
    const common = c.ds4_session_common_prefix(session, &chat.transcript);
    const cached = if (common == old_pos and chat.transcript.len >= old_pos) common else 0;
    const suffix = chat.transcript.len - cached;
    var message: [160]u8 = undefined;
    var progress = shared.Progress{ .base_tokens = cached, .input_tokens = suffix, .use_color = c.ds4_log_is_tty(c.__stderrp) };
    const prefill_start = now();
    progress.install(session);
    busySet(cfg, true);
    const result = c.ds4_session_sync_multimodal(session, &chat.transcript, chat.images, chat.image_count, &message, message.len);
    busySet(cfg, false);
    shared.clearProgress(session);
    if (result != 0) {
        chat.transcript.len = rollback_len;
        c.ds4_cli_chat_trim_images(chat, rollback_images);
        report("ds4: prompt processing failed: %s\n", .{&message});
        return 1;
    }
    const prefill_end = now();
    var printer = Printer.init(engine, mode);
    const maximum = maxTokens(cfg, session);
    var rng = randomSeed(cfg.gen.seed);
    const start = now();
    const generated = decode(engine, cfg, session, &printer, &chat.transcript, mode, maximum, &rng) orelse return 1;
    const end = now();
    printer.done();
    const was_interrupted = interruptRequested();
    if (was_interrupted and generated == 0) {
        chat.transcript.len = rollback_len;
        c.ds4_cli_chat_trim_images(chat, rollback_images);
        c.ds4_session_invalidate(session);
    } else if (!c.ds4_engine_is_glm_dsa(engine)) c.ds4_tokens_push(&chat.transcript, c.ds4_token_eos(engine));
    if (was_interrupted) clearInterrupt();
    timing(suffix, generated, prefill_end - prefill_start, end - start);
    return 0;
}
fn imageMagic(path: [:0]const u8) bool {
    const file = c.fopen(path.ptr, "rb");
    if (file == null) return false;
    defer _ = c.fclose(file);
    var bytes: [8]u8 = @splat(0);
    const length = c.fread(&bytes, 1, bytes.len, file);
    return (length >= 8 and same(&bytes, "\x89PNG\r\n\x1a\n")) or (length >= 2 and bytes[0] == 0xff and bytes[1] == 0xd8);
}
fn commandArg(command: [:0]u8, name: []const u8) ?[:0]u8 {
    if (!std.mem.startsWith(u8, command, name)) return null;
    if (command.len > name.len and c.isspace(command[name.len]) == 0) return null;
    return trim(command.ptr + name.len);
}
fn repl(engine: *Engine, cfg: *Options) c_int {
    var chat: c.ds4_cli_chat = undefined;
    if (!chatInit(engine, &chat, cfg)) return 1;
    defer c.ds4_cli_chat_free(&chat);
    var old_int: c.struct_sigaction = undefined;
    var action = std.mem.zeroes(c.struct_sigaction);
    _ = c.sigemptyset(&action.sa_mask);
    action.__sigaction_u.__sa_handler = sigintHandler;
    const installed = c.sigaction(c.SIGINT, &action, &old_int) == 0;
    defer if (installed) {
        _ = c.sigaction(c.SIGINT, &old_int, null);
    };
    clearInterrupt();
    var history: [c.PATH_MAX]u8 = undefined;
    const home = c.getenv("HOME");
    _ = c.snprintf(&history, history.len, "%s/.ds4_history", if (home != null and home[0] != 0) home else @as([*c]const u8, "."));
    c.linenoiseSetMultiLine(1);
    _ = c.linenoiseHistorySetMaxLen(512);
    _ = c.linenoiseHistoryLoad(&history);
    replHelp();
    var result: c_int = 0;
    while (true) {
        c.__error().* = 0;
        const line = c.linenoise("ds4> ");
        if (line == null) {
            if (c.__error().* == c.EAGAIN or interruptRequested()) {
                clearInterrupt();
                continue;
            }
            break;
        }
        defer c.linenoiseFree(line);
        const command = trim(line);
        if (command.len == 0) continue;
        _ = c.linenoiseHistoryAdd(command.ptr);
        _ = c.linenoiseHistorySave(&history);
        if (same(command, "/help")) {
            replHelp();
        } else if (commandArg(command, "/think")) |arg| {
            if (c.ds4_engine_is_flint(engine)) {
                report("ds4: Flint chooses when to reason; thinking controls are unsupported\n", .{});
                continue;
            }
            var mode: c.ds4_think_mode = c.DS4_THINK_HIGH;
            if (arg.len != 0 and (!c.ds4_engine_is_deepseek41(engine) or !c.ds4_think_mode_parse_level(arg.ptr, &mode))) {
                report("ds4: /think N requires V4.1 and a level from 0 to 100\n", .{});
            } else if (c.ds4_cli_chat_apply_think_prefix(engine, &chat, mode)) {
                cfg.gen.think_mode = mode;
                _ = c.printf("Thinking mode: %s.\n", c.ds4_think_mode_name(mode));
            }
        } else if (same(command, "/think-max")) {
            if (c.ds4_engine_is_flint(engine)) {
                report("ds4: Flint chooses when to reason; thinking controls are unsupported\n", .{});
                continue;
            }
            const active = c.ds4_think_mode_for_context(c.DS4_THINK_MAX, chat.ctx_size) == c.DS4_THINK_MAX;
            if (c.ds4_cli_chat_apply_think_prefix(engine, &chat, if (active) c.DS4_THINK_MAX else c.DS4_THINK_HIGH)) {
                cfg.gen.think_mode = c.DS4_THINK_MAX;
                warnThink(&cfg.gen, "/think-max");
                _ = c.printf("Thinking mode: %s.\n", @as([*:0]const u8, if (active) "max" else "high (ctx below 393216)"));
            }
        } else if (same(command, "/nothink")) {
            if (c.ds4_engine_is_flint(engine)) {
                report("ds4: Flint chooses when to reason; thinking controls are unsupported\n", .{});
                continue;
            }
            if (c.ds4_cli_chat_apply_think_prefix(engine, &chat, c.DS4_THINK_NONE)) {
                cfg.gen.think_mode = c.DS4_THINK_NONE;
                _ = c.puts("Thinking mode: none.");
            }
        } else if (commandArg(command, "/power")) |arg| {
            if (arg.len == 0) {
                _ = c.printf("Power: %d%%.\n", c.ds4_session_power(chat.session));
            } else {
                var end: [*c]u8 = null;
                const power = c.strtol(arg.ptr, &end, 10);
                if (end[0] != 0 or power < 1 or power > 100) {
                    report("ds4: /power must be between 1 and 100\n", .{});
                } else if (c.ds4_session_set_power(chat.session, @intCast(power)) != 0) {
                    report("ds4: failed to set /power\n", .{});
                } else {
                    cfg.engine.power_percent = @intCast(power);
                    _ = c.printf("Power: %d%%.\n", @as(c_int, @intCast(power)));
                }
            }
        } else if (commandArg(command, "/steer")) |arg| {
            if (arg.len == 0) {
                _ = c.printf("Steering FFN: %g.\n", @as(f64, c.ds4_session_directional_steering_ffn(chat.session)));
            } else {
                var end: [*c]u8 = null;
                c.__error().* = 0;
                const scale = c.strtof(arg.ptr, &end);
                if (end[0] != 0 or c.__error().* == c.ERANGE or !std.math.isFinite(scale) or scale < -100 or scale > 100) {
                    report("ds4: /steer must be between -100 and 100\n", .{});
                } else if (c.ds4_session_set_directional_steering_ffn(chat.session, scale) == 0) {
                    cfg.engine.directional_steering_ffn = scale;
                    _ = c.printf("Steering FFN: %g.\n", @as(f64, scale));
                }
            }
        } else if (commandArg(command, "/ctx")) |arg| {
            if (arg.len == 0) {
                report("ds4: /ctx needs a positive integer\n", .{});
            } else {
                cfg.gen.ctx_size = integer(arg, "/ctx", false);
                logMemory(cfg, c.ds4_engine_prefill_chunk(engine));
                result = c.ds4_cli_chat_set_ctx(engine, &chat, cfg.gen.ctx_size);
                if (result != 0) break;
                _ = c.ds4_cli_chat_apply_think_prefix(engine, &chat, c.ds4_think_mode_for_context(cfg.gen.think_mode, chat.ctx_size));
                warnThink(&cfg.gen, "/ctx");
            }
        } else if (same(command, "/quit") or same(command, "/exit")) {
            break;
        } else if (commandArg(command, "/read")) |path| {
            if (path.len == 0) {
                report("ds4: /read needs a file path\n", .{});
            } else if (imageMagic(path)) {
                var message: [256]u8 = @splat(0);
                var image = std.mem.zeroes(c.ds4_vision_embedding);
                if (c.ds4_engine_vision_encode_file(engine, path.ptr, &image, &message, message.len) == 0) {
                    report("ds4: /read image failed: %s\n", .{&message});
                } else {
                    defer c.ds4_vision_embedding_free(&image);
                    report("ds4: image %ux%u, %u image tokens\n", .{ image.width, image.height, image.token_count });
                    result = chatTurn(engine, cfg, &chat, "", &image);
                }
            } else if (shared.readPromptFile(path, false)) |text| {
                defer allocator.free(text);
                result = chatTurn(engine, cfg, &chat, text, null);
            }
        } else if (command[0] == '/') {
            report("ds4: unknown command: %s\nds4: type /help for commands\n", .{command.ptr});
        } else {
            result = chatTurn(engine, cfg, &chat, command, null);
        }
    }
    return result;
}

const Args = struct {
    argv: [][*c]u8,
    index: c_int = 1,
    fn current(self: *const Args) ?[:0]const u8 {
        const index: usize = @intCast(self.index);
        return if (index < self.argv.len) std.mem.span(self.argv[index]) else null;
    }
    fn value(self: *Args, option: [:0]const u8) [:0]const u8 {
        self.index += 1;
        return self.current() orelse fatal(2, "ds4: missing value for %s\n", .{option.ptr});
    }
    fn helpTopic(self: *const Args) ?[:0]const u8 {
        const next: usize = @intCast(self.index + 1);
        return if (next < self.argv.len and self.argv[next][0] != '-') std.mem.span(self.argv[next]) else null;
    }
};
fn parseOptions(argv: [][*c]u8) Options {
    var cfg = Options{};
    cfg.dist = c.ds4_dist_options_create();
    if (cfg.dist == null) fatal(1, "ds4: out of memory creating distributed options\n", .{});
    var steering_set = false;
    var args = Args{ .argv = argv };
    while (args.current()) |arg| : (args.index += 1) {
        if (same(arg, "-h") or same(arg, "--help")) {
            usage(c.__stdoutp, args.helpTopic());
            c.exit(0);
        }
        var message: [256]u8 = @splat(0);
        const dist = c.ds4_dist_parse_cli_arg(arg.ptr, &args.index, @intCast(argv.len), argv.ptr, cfg.dist, &message, message.len);
        if (dist == c.DS4_DIST_CLI_ERROR) fatal(2, "ds4: %s\n", .{if (message[0] != 0) message[0..].ptr else "invalid distributed option"});
        if (dist == c.DS4_DIST_CLI_MATCHED) continue;
        @memset(&message, 0);
        const tp = c.ds4_tp_parse_cli_arg(arg.ptr, &args.index, @intCast(argv.len), argv.ptr, &cfg.engine.tp, &message, message.len);
        if (tp == c.DS4_TP_CLI_ERROR) fatal(2, "ds4: %s\n", .{if (message[0] != 0) message[0..].ptr else "invalid tensor-parallel option"});
        if (tp == c.DS4_TP_CLI_MATCHED) continue;
        if (same(arg, "-p") or same(arg, "--prompt")) {
            if (cfg.gen.prompt != null) fatal(2, "ds4: specify only one prompt source\n", .{});
            cfg.gen.prompt = args.value(arg);
        } else if (same(arg, "--prompt-file")) {
            if (cfg.gen.prompt != null) fatal(2, "ds4: specify only one prompt source\n", .{});
            cfg.prompt_owned = shared.readPromptFile(args.value(arg), true);
            cfg.gen.prompt = cfg.prompt_owned;
        } else if (same(arg, "--prefix-file")) {
            if (cfg.gen.prefix.count != 0) fatal(2, "ds4: specify --prefix-file only once\n", .{});
            if (c.ds4_prompt_prefix_load(&cfg.gen.prefix, args.value(arg).ptr, &message, message.len) != 0)
                fatal(2, "ds4: %s\n", .{if (message[0] != 0) message[0..].ptr else "invalid prefix file"});
        } else if (same(arg, "-sys") or same(arg, "--system")) {
            cfg.gen.system = args.value(arg);
        } else if (same(arg, "--raw") or same(arg, "--raw-prompt")) {
            cfg.gen.raw_prompt = true;
        } else if (same(arg, "-m") or same(arg, "--model")) {
            cfg.engine.model_path = args.value(arg).ptr;
        } else if (same(arg, "--vision")) {
            cfg.engine.vision_path = args.value(arg).ptr;
        } else if (same(arg, "--mtp")) {
            cfg.engine.glm_mtp = true;
        } else if (same(arg, "--mtp-model")) {
            cfg.engine.mtp_path = args.value(arg).ptr;
        } else if (same(arg, "--mtp-draft")) {
            cfg.engine.mtp_draft_tokens = integer(args.value(arg), arg, false);
        } else if (same(arg, "--mtp-margin")) {
            cfg.engine.mtp_margin = decimal(args.value(arg), arg, 0, 1000);
        } else if (same(arg, "--mtp-timing")) {
            cfg.engine.glm_mtp = true;
            cfg.engine.glm_mtp_timing = true;
        } else if (same(arg, "--dspark")) {
            cfg.engine.dspark = true;
        } else if (same(arg, "--dspark-confidence")) {
            cfg.engine.dspark = true;
            cfg.engine.dspark_confidence_threshold = decimal(args.value(arg), arg, 0, 1);
            cfg.engine.dspark_confidence_threshold_set = true;
        } else if (same(arg, "--dspark-strict")) {
            cfg.engine.dspark = true;
            cfg.engine.dspark_strict = true;
        } else if (same(arg, "--mtp-exact-sampling")) {
            cfg.engine.dspark_exact_sampling = true;
        } else if (same(arg, "-n") or same(arg, "--tokens")) {
            cfg.gen.n_predict = integer(args.value(arg), arg, false);
        } else if (same(arg, "-c") or same(arg, "--ctx")) {
            cfg.gen.ctx_size = integer(args.value(arg), arg, false);
        } else if (same(arg, "--temp")) {
            cfg.gen.temperature = decimal(args.value(arg), arg, 0, 100);
            cfg.gen.temperature_set = true;
        } else if (same(arg, "--top-p")) {
            cfg.gen.top_p = decimal(args.value(arg), arg, 0, 1);
            cfg.gen.top_p_set = true;
        } else if (same(arg, "--top-k")) {
            cfg.gen.top_k = integer(args.value(arg), arg, true);
            cfg.gen.top_k_set = true;
        } else if (same(arg, "--min-p")) {
            cfg.gen.min_p = decimal(args.value(arg), arg, 0, 1);
            cfg.gen.min_p_set = true;
        } else if (same(arg, "--seed")) {
            cfg.gen.seed = unsigned(args.value(arg), arg);
        } else if (same(arg, "--quality")) {
            cfg.engine.quality = true;
        } else if (same(arg, "--ssd-streaming")) {
            cfg.engine.ssd_streaming = true;
        } else if (same(arg, "--ssd-streaming-cold")) {
            cfg.engine.ssd_streaming_cold = true;
        } else if (same(arg, "--ssd-streaming-cache-experts")) {
            if (!c.ds4_parse_streaming_cache_experts_arg(args.value(arg).ptr, &cfg.engine.ssd_streaming_cache_experts, &cfg.engine.ssd_streaming_cache_bytes))
                fatal(2, "ds4: --ssd-streaming-cache-experts must be a positive count or <number>GB\n", .{});
        } else if (same(arg, "--ssd-streaming-full-layers")) {
            cfg.engine.ssd_streaming_full_layers = @intCast(integer(args.value(arg), arg, true));
            cfg.engine.ssd_streaming_full_layers_set = true;
        } else if (same(arg, "--ssd-streaming-preload-experts")) {
            cfg.engine.ssd_streaming_preload_experts = @intCast(integer(args.value(arg), arg, false));
        } else if (same(arg, "--simulate-used-memory")) {
            if (!c.ds4_parse_gib_arg(args.value(arg).ptr, &cfg.engine.simulate_used_memory_bytes))
                fatal(2, "ds4: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n", .{});
        } else if (same(arg, "--prefill-chunk")) {
            cfg.engine.prefill_chunk = @intCast(integer(args.value(arg), arg, false));
        } else if (same(arg, "--flint-profile")) {
            cfg.engine.flint_profile = args.value(arg);
        } else if (same(arg, "--power")) {
            cfg.engine.power_percent = integer(args.value(arg), arg, false);
            if (cfg.engine.power_percent > 100) fatal(2, "ds4: --power must be between 1 and 100\n", .{});
        } else if (same(arg, "--dir-steering-file")) {
            cfg.engine.directional_steering_file = args.value(arg).ptr;
        } else if (same(arg, "--dir-steering-mode")) {
            const value = args.value(arg);
            cfg.engine.directional_steering_mode = if (same(value, "project")) c.DS4_STEERING_PROJECT else if (same(value, "add")) c.DS4_STEERING_ADD else fatal(2, "ds4: --dir-steering-mode must be project or add\n", .{});
        } else if (same(arg, "--expert-profile")) {
            cfg.engine.expert_profile_path = args.value(arg).ptr;
        } else if (same(arg, "--dir-steering-ffn")) {
            cfg.engine.directional_steering_ffn = decimal(args.value(arg), arg, -100, 100);
            steering_set = true;
        } else if (same(arg, "--dir-steering-attn")) {
            cfg.engine.directional_steering_attn = decimal(args.value(arg), arg, -100, 100);
            steering_set = true;
        } else if (same(arg, "-t") or same(arg, "--threads")) {
            cfg.engine.n_threads = integer(args.value(arg), arg, false);
        } else if (same(arg, "--backend")) {
            cfg.engine.backend = backend(args.value(arg));
        } else if (same(arg, "--cpu")) {
            cfg.engine.backend = c.DS4_BACKEND_CPU;
        } else if (same(arg, "--metal")) {
            cfg.engine.backend = c.DS4_BACKEND_METAL;
        } else if (same(arg, "--cuda")) {
            cfg.engine.backend = c.DS4_BACKEND_CUDA;
        } else if (same(arg, "--gpu-vram")) {
            cfg.gpu_vram_arg = args.value(arg);
        } else if (same(arg, "--gpu-devices")) {
            cfg.gpu_devices_arg = args.value(arg);
        } else if (same(arg, "--cuda-tensor-parallel")) {
            cfg.engine.cuda_tensor_parallel = true;
        } else if (same(arg, "--dump-tokens")) {
            cfg.gen.dump_tokens = true;
        } else if (same(arg, "--dump-logits")) {
            cfg.gen.dump_logits_path = args.value(arg);
        } else if (same(arg, "--dump-logprobs")) {
            cfg.gen.dump_logprobs_path = args.value(arg);
        } else if (same(arg, "--logprobs-top-k")) {
            cfg.gen.dump_logprobs_top_k = integer(args.value(arg), arg, false);
        } else if (same(arg, "--decode-consistency")) {
            cfg.gen.decode_consistency_tokens = integer(args.value(arg), arg, false);
        } else if (same(arg, "--perplexity-file")) {
            cfg.gen.perplexity_file_path = args.value(arg);
        } else if (same(arg, "--imatrix-dataset")) {
            cfg.gen.imatrix_dataset_path = args.value(arg);
        } else if (same(arg, "--imatrix-out")) {
            cfg.gen.imatrix_output_path = args.value(arg);
            cfg.engine.backend = c.DS4_BACKEND_METAL;
        } else if (same(arg, "--imatrix-max-prompts")) {
            cfg.gen.imatrix_max_prompts = integer(args.value(arg), arg, false);
        } else if (same(arg, "--imatrix-max-tokens")) {
            cfg.gen.imatrix_max_tokens = integer(args.value(arg), arg, false);
        } else if (same(arg, "--imatrix-min-expert-samples")) {
            cfg.gen.imatrix_min_expert_samples = integer(args.value(arg), arg, false);
        } else if (same(arg, "--think-level")) {
            if (!c.ds4_think_mode_parse_level(args.value(arg).ptr, &cfg.gen.think_mode)) fatal(2, "ds4: --think-level requires an integer from 0 to 100\n", .{});
        } else if (same(arg, "--think")) {
            cfg.gen.think_mode = c.DS4_THINK_HIGH;
        } else if (same(arg, "--think-max")) {
            cfg.gen.think_mode = c.DS4_THINK_MAX;
        } else if (same(arg, "--nothink")) {
            cfg.gen.think_mode = c.DS4_THINK_NONE;
        } else if (same(arg, "--head-test")) {
            cfg.gen.head_test = true;
        } else if (same(arg, "--first-token-test")) {
            cfg.gen.first_token_test = true;
        } else if (same(arg, "--metal-graph-test")) {
            cfg.gen.metal_graph_test = true;
            cfg.engine.backend = c.DS4_BACKEND_METAL;
        } else if (same(arg, "--metal-graph-full-test")) {
            cfg.gen.metal_graph_full_test = true;
            cfg.engine.backend = c.DS4_BACKEND_METAL;
        } else if (same(arg, "--metal-graph-prompt-test")) {
            cfg.gen.metal_graph_prompt_test = true;
            cfg.engine.backend = c.DS4_BACKEND_METAL;
        } else if (same(arg, "--metal-graph-generate")) {
            fatal(2, "ds4: --metal-graph-generate was removed; --metal is the graph path\n", .{});
        } else if (same(arg, "--inspect")) {
            cfg.inspect = true;
        } else if (same(arg, "--warm-weights")) {
            cfg.engine.warm_weights = true;
        } else if (same(arg, "--server")) {
            fatal(2, "ds4: use ds4-server for the HTTP server\n", .{});
        } else {
            report("ds4: unknown option: %s\n", .{arg.ptr});
            usage(c.__stderrp, null);
            c.exit(2);
        }
    }
    if (cfg.engine.directional_steering_file != null and !steering_set) cfg.engine.directional_steering_ffn = 1;
    if (cfg.gen.imatrix_output_path != null and cfg.gen.imatrix_dataset_path == null) fatal(2, "ds4: --imatrix-out requires --imatrix-dataset\n", .{});
    if (cfg.gen.imatrix_dataset_path != null and cfg.gen.imatrix_output_path == null) fatal(2, "ds4: --imatrix-dataset requires --imatrix-out\n", .{});
    if (cfg.gen.imatrix_min_expert_samples < 0) fatal(2, "ds4: --imatrix-min-expert-samples must not be negative\n", .{});
    if (cfg.gen.perplexity_file_path != null and cfg.gen.prompt != null) fatal(2, "ds4: --perplexity-file does not use -p/--prompt-file\n", .{});
    if (cfg.gen.prefix.count != 0 and cfg.gen.raw_prompt) fatal(2, "ds4: --prefix-file cannot be combined with --raw-prompt\n", .{});
    if (cfg.gen.prefix.count != 0 and renderedPrompt(cfg.gen.prompt)) fatal(2, "ds4: --prefix-file cannot be combined with an already-rendered prompt\n", .{});
    var message: [256]u8 = undefined;
    if (c.ds4_tp_adopt_distributed_options(&cfg.engine.tp, cfg.dist, &message, message.len) == 0) fatal(2, "ds4: %s\n", .{&message});
    if (c.ds4_dist_prepare_engine_options(cfg.dist, &cfg.engine, &message, message.len) != 0) fatal(2, "ds4: %s\n", .{&message});
    if (c.ds4_tp_validate_engine_options(&cfg.engine, &message, message.len) == 0) fatal(2, "ds4: %s\n", .{&message});
    return cfg;
}
fn bindLeader(engine: *Engine, cfg: *const Options, leader: *?*c.ds4_tp) bool {
    var identity = std.mem.zeroes(c.ds4_tp_identity);
    identity.gguf_bytes = c.ds4_engine_model_bytes(engine);
    identity.model_id = @intCast(c.ds4_engine_model_id(engine));
    identity.n_layer = @intCast(c.ds4_engine_layer_count(engine));
    identity.n_embd = @intCast(c.ds4_engine_embd_dim(engine));
    identity.n_vocab = @intCast(c.ds4_engine_vocab_size(engine));
    identity.quant_bits = @intCast(c.ds4_engine_routed_quant_bits(engine));
    identity.ctx_size = @intCast(cfg.gen.ctx_size);
    c.ds4_engine_tp_gate_schedule(engine, &identity.gate_slot_start, &identity.gate_slot_step, &identity.gates_per_token, &identity.gate_slot_mask);
    var message: [256]u8 = @splat(0);
    if (c.ds4_tp_create(leader, &cfg.engine.tp, &identity, &message, message.len) == 0 or
        c.ds4_engine_tp_bind(engine, leader.*, &message, message.len) == 0)
    {
        report("ds4: %s\n", .{&message});
        c.ds4_tp_free(leader.*);
        leader.* = null;
        return false;
    }
    return true;
}
fn run(cfg: *Options) c_int {
    if (cfg.gen.dump_tokens) {
        if (cfg.gen.prefix.count != 0) {
            report("ds4: --dump-tokens does not support --prefix-file\n", .{});
            return 2;
        }
        if (cfg.gen.prompt == null) {
            report("ds4: --dump-tokens requires -p or --prompt-file\n", .{});
            return 2;
        }
        if (cfg.gen.raw_prompt or renderedPrompt(cfg.gen.prompt))
            return c.ds4_dump_text_tokenization(cfg.engine.model_path, cString(cfg.gen.prompt), c.__stdoutp);
        return c.ds4_dump_chat_tokenization(cfg.engine.model_path, cString(cfg.gen.system), cString(cfg.gen.prompt), cfg.gen.think_mode, cfg.gen.ctx_size, c.__stdoutp);
    }
    cfg.engine.inspect_only = cfg.inspect;
    cfg.engine.first_token_test = cfg.gen.first_token_test;
    cfg.engine.metal_graph_test = cfg.gen.metal_graph_test;
    cfg.engine.context_size = cfg.gen.ctx_size;
    cfg.engine.placement_ctx_hint = cfg.gen.ctx_size;
    var maybe_engine: ?*Engine = null;
    if (cfg.gpu_vram_arg != null or cfg.gpu_devices_arg != null) {
        var gpu = std.mem.zeroes(c.ds4_gpu_config);
        var skip_cuda = false;
        var message: [256]u8 = undefined;
        if (c.parse_gpu_vram_arg(cString(cfg.gpu_vram_arg), cString(cfg.gpu_devices_arg), &gpu, &skip_cuda, &message, message.len) != 0) {
            report("ds4: %s\n", .{&message});
            return 2;
        }
        cfg.engine.backend = if (skip_cuda) c.DS4_BACKEND_CPU else c.DS4_BACKEND_CUDA;
        if (skip_cuda) {
            if (c.ds4_engine_open(&maybe_engine, &cfg.engine) != 0) return 1;
        } else {
            const was_auto = if (cfg.gpu_vram_arg) |value| same(value, "auto") else cfg.gpu_devices_arg != null;
            var layout: [256]u8 = undefined;
            if (c.format_gpu_layout_line(&gpu, was_auto, &layout, layout.len) > 0) {
                _ = c.fprintf(c.__stdoutp, "%s\n", &layout);
                _ = c.fflush(c.__stdoutp);
            }
            if (c.ds4_engine_create_with_gpu_config(&maybe_engine, &cfg.engine, &gpu) != 0) return 1;
        }
    } else if (c.ds4_engine_open(&maybe_engine, &cfg.engine) != 0) return 1;
    const engine = maybe_engine orelse return 1;
    var leader: ?*c.ds4_tp = null;
    defer {
        if (leader) |tp| _ = c.ds4_tp_send_stop(tp);
        c.ds4_engine_close(engine);
        c.ds4_tp_free(leader);
    }
    if (c.ds4_think_mode_level(cfg.gen.think_mode) >= 0 and !c.ds4_engine_is_deepseek41(engine)) {
        report("ds4: --think-level requires a DeepSeek V4.1 model\n", .{});
        return 2;
    }
    if (c.ds4_engine_is_flint(engine) and cfg.gen.think_mode != c.DS4_THINK_HIGH) {
        report("ds4: Flint chooses when to reason; --nothink, --think-max and --think-level are unsupported\n", .{});
        return 2;
    }
    if (c.ds4_engine_is_glm_dsa(engine) or c.ds4_engine_is_flint(engine)) {
        if (!cfg.gen.temperature_set) cfg.gen.temperature = 1;
        if (!cfg.gen.top_p_set) cfg.gen.top_p = 0.95;
        if (!cfg.gen.min_p_set) cfg.gen.min_p = 0;
        if (c.ds4_engine_is_flint(engine) and !cfg.gen.top_k_set) cfg.gen.top_k = 20;
    }
    if (cfg.engine.tp.role == c.DS4_TP_WORKER) return c.ds4_tp_worker_run(engine, &cfg.engine.tp);
    if (cfg.engine.tp.role == c.DS4_TP_LEADER and !bindLeader(engine, cfg, &leader)) return 1;
    if (cfg.dist) |dist| if (dist.role == c.DS4_DISTRIBUTED_WORKER) {
        var gen = std.mem.zeroes(c.ds4_dist_generation_options);
        gen.prompt = cString(cfg.gen.prompt);
        gen.system = cString(cfg.gen.system);
        gen.dump_logits_path = cString(cfg.gen.dump_logits_path);
        gen.dump_logprobs_path = cString(cfg.gen.dump_logprobs_path);
        gen.dump_logprobs_top_k = cfg.gen.dump_logprobs_top_k;
        gen.n_predict = cfg.gen.n_predict;
        gen.ctx_size = cfg.gen.ctx_size;
        gen.temperature = cfg.gen.temperature;
        gen.top_p = cfg.gen.top_p;
        gen.min_p = cfg.gen.min_p;
        gen.seed = cfg.gen.seed;
        gen.think_mode = cfg.gen.think_mode;
        return c.ds4_dist_run(engine, dist, &gen);
    };
    if (!cfg.inspect) {
        logMemory(cfg, c.ds4_engine_prefill_chunk(engine));
        warnThink(&cfg.gen, "--think-max");
    }
    if (cfg.inspect) {
        c.ds4_engine_summary(engine);
        return 0;
    }
    if (cfg.gen.imatrix_output_path != null)
        return c.ds4_engine_collect_imatrix(engine, cString(cfg.gen.imatrix_dataset_path), cString(cfg.gen.imatrix_output_path), cfg.gen.ctx_size, cfg.gen.imatrix_max_prompts, cfg.gen.imatrix_max_tokens, cfg.gen.imatrix_min_expert_samples);
    if (cfg.gen.perplexity_file_path != null) return diagnostics.runPerplexity(engine, cfg);
    return if (cfg.gen.prompt == null) repl(engine, cfg) else generation(engine, cfg);
}

pub export fn main(argc: c_int, argv: [*c][*c]u8) callconv(.c) c_int {
    var cfg = parseOptions(argv[0..@as(usize, @intCast(argc))]);
    defer cfg.deinit();
    return run(&cfg);
}
