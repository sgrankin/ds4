//! Purpose-built Metal throughput benchmark.
//!
//! Walk one fixed token sequence to successive context frontiers, timing only
//! the newest prefill interval. Decode a fixed number of non-EOS greedy tokens,
//! then restore a small in-memory snapshot or replay the prefix before moving
//! on. Snapshot save/restore is outside both measurement windows.
//!
//! Public engine/TP/distributed APIs remain C ABI calls. Options and control
//! flow live here; generated declarations stay behind the bench_c namespace.
const std = @import("std");
const c = @import("bench_c");
const allocator = std.heap.c_allocator;
const File = c.FILE;
const Engine = c.ds4_engine;
const Session = c.ds4_session;
const default_snapshot_limit: u64 = 1 << 30;

const Options = struct {
    model_path: [:0]const u8 = "ds4flash.gguf",
    vision_path: ?[:0]const u8 = null,
    mtp_path: ?[:0]const u8 = null,
    prompt_path: ?[:0]const u8 = null,
    chat_prompt_path: ?[:0]const u8 = null,
    system: [:0]const u8 = "You are a helpful assistant.",
    csv_path: ?[:0]const u8 = null,
    expert_profile_path: ?[:0]const u8 = null,
    gpu_vram_arg: ?[:0]const u8 = null,
    gpu_devices_arg: ?[:0]const u8 = null,
    backend: c.ds4_backend = c.DS4_BACKEND_METAL,
    threads: c_int = 0,
    ctx_start: c_int = 2048,
    ctx_max: c_int = 32768,
    ctx_alloc: c_int = 0,
    step_incr: c_int = 2048,
    gen_tokens: c_int = 128,
    power_percent: c_int = 0,
    prefill_chunk: u32 = 0,
    ssd_streaming_cache_experts: u32 = 0,
    ssd_streaming_cache_bytes: u64 = 0,
    ssd_streaming_full_layers: u32 = 0,
    ssd_streaming_preload_experts: u32 = 0,
    simulate_used_memory_bytes: u64 = 0,
    step_mul: f64 = 1,
    dump_frontier_logits_dir: ?[:0]const u8 = null,
    dist: c.ds4_dist_options = std.mem.zeroes(c.ds4_dist_options),
    tp: c.ds4_tp_options = std.mem.zeroes(c.ds4_tp_options),
    warm_weights: bool = false,
    quality: bool = false,
    ssd_streaming: bool = false,
    ssd_streaming_cold: bool = false,
    ssd_streaming_full_layers_set: bool = false,
    cuda_tensor_parallel: bool = false,
    show_output: bool = false,
    teacher_forced_decode: bool = false,
    dspark: bool = false,
    dspark_confidence_threshold_set: bool = false,
    dspark_confidence_threshold: f32 = 0,

    fn engineOptions(self: *const Options) c.ds4_engine_options {
        var out = std.mem.zeroes(c.ds4_engine_options);
        out.model_path = self.model_path.ptr;
        out.vision_path = cString(self.vision_path);
        out.mtp_path = cString(self.mtp_path);
        out.backend = self.backend;
        out.n_threads = self.threads;
        out.context_size = self.ctx_alloc;
        out.prefill_chunk = self.prefill_chunk;
        out.ssd_streaming_cache_experts = self.ssd_streaming_cache_experts;
        out.ssd_streaming_cache_bytes = self.ssd_streaming_cache_bytes;
        out.ssd_streaming_full_layers = self.ssd_streaming_full_layers;
        out.ssd_streaming_preload_experts = self.ssd_streaming_preload_experts;
        out.simulate_used_memory_bytes = self.simulate_used_memory_bytes;
        out.power_percent = self.power_percent;
        out.warm_weights = self.warm_weights;
        out.quality = self.quality;
        out.dspark = self.dspark;
        out.dspark_confidence_threshold = self.dspark_confidence_threshold;
        out.dspark_confidence_threshold_set = self.dspark_confidence_threshold_set;
        out.cuda_tensor_parallel = self.cuda_tensor_parallel;
        out.ssd_streaming = self.ssd_streaming;
        out.ssd_streaming_cold = self.ssd_streaming_cold;
        out.ssd_streaming_full_layers_set = self.ssd_streaming_full_layers_set;
        out.expert_profile_path = cString(self.expert_profile_path);
        out.distributed = self.dist;
        out.tp = self.tp;
        return out;
    }
};

// argv strings are borrowed for the program lifetime. Keep the original vector
// and signed index available to the shared public TP/distributed parsers.
const Args = struct {
    argv: [][*c]u8,
    index: c_int = 1,

    fn current(self: *const Args) ?[:0]const u8 {
        const index: usize = @intCast(self.index);
        return if (index < self.argv.len) std.mem.span(self.argv[index]) else null;
    }

    fn value(self: *Args, option: [:0]const u8) [:0]const u8 {
        self.index += 1;
        return self.current() orelse fatal(2, "ds4-bench: %s requires an argument\n", .{option.ptr});
    }

    fn helpTopic(self: *const Args) ?[:0]const u8 {
        const next: usize = @intCast(self.index + 1);
        if (next >= self.argv.len or self.argv[next][0] == '-') return null;
        return std.mem.span(self.argv[next]);
    }
};

fn cString(text: ?[:0]const u8) ?[*:0]const u8 {
    return if (text) |value| value.ptr else null;
}

fn same(left: []const u8, right: []const u8) bool {
    return std.mem.eql(u8, left, right);
}

// Preserve the benchmark's public diagnostics and printf precision. Formatting
// stays at the libc boundary rather than changing CSV/JSON during this port.
fn report(comptime format: [*:0]const u8, args: anytype) void {
    _ = @call(.auto, c.fprintf, .{ c.__stderrp, format } ++ args);
}

fn fatal(status: c_int, comptime format: [*:0]const u8, args: anytype) noreturn {
    report(format, args);
    c.exit(status);
}

fn systemError() [*c]u8 {
    return c.strerror(c.__error().*);
}

fn nowSeconds() f64 {
    @setFloatMode(.optimized);
    var time: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &time);
    return @as(f64, @floatFromInt(time.tv_sec)) + @as(f64, @floatFromInt(time.tv_nsec)) / 1_000_000_000.0;
}

fn bytesToGiB(bytes: u64) f64 {
    @setFloatMode(.optimized);
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0);
}

fn snapshotLimit() u64 {
    const env = c.getenv("DS4_BENCH_SNAPSHOT_MAX_BYTES");
    if (env == null or env[0] == 0) return default_snapshot_limit;
    const text = std.mem.span(env);
    if (same(text, "unlimited") or same(text, "UNLIMITED") or same(text, "inf") or same(text, "INF")) return std.math.maxInt(u64);
    var end: [*c]u8 = null;
    const value = c.strtoull(env, &end, 10);
    if (end == null or end[0] != 0) {
        report("ds4-bench: invalid DS4_BENCH_SNAPSHOT_MAX_BYTES=%s; using default %llu\n", .{ env, @as(c_ulonglong, default_snapshot_limit) });
        return default_snapshot_limit;
    }
    return value;
}

fn usage(file: [*c]File, topic: ?[:0]const u8) void {
    c.ds4_help_print(file, c.DS4_HELP_BENCH, cString(topic));
}

// strtol/strtod intentionally preserve the C CLI's whitespace/sign/overflow
// grammar and errors. These are not replaced with a stricter Zig number parser.
fn integer(text: [:0]const u8, option: [:0]const u8, allow_zero: bool) c_int {
    var end: [*c]u8 = null;
    const value = c.strtol(text.ptr, &end, 10);
    if (text.len == 0 or end[0] != 0 or value < 0 or (!allow_zero and value == 0) or value > std.math.maxInt(c_int)) {
        fatal(2, "ds4-bench: invalid value for %s: %s\n", .{ option.ptr, text.ptr });
    }
    return @intCast(value);
}

fn decimal(text: [:0]const u8, option: [:0]const u8) f64 {
    var end: [*c]u8 = null;
    const value = c.strtod(text.ptr, &end);
    if (text.len == 0 or end[0] != 0 or !std.math.isFinite(value)) {
        fatal(2, "ds4-bench: invalid value for %s: %s\n", .{ option.ptr, text.ptr });
    }
    return value;
}

fn backend(text: [:0]const u8, option: [:0]const u8) c.ds4_backend {
    if (same(text, "metal")) return c.DS4_BACKEND_METAL;
    if (same(text, "cuda")) return c.DS4_BACKEND_CUDA;
    if (same(text, "cpu")) return c.DS4_BACKEND_CPU;
    report("ds4-bench: invalid value for %s: %s\n", .{ option.ptr, text.ptr });
    fatal(2, "ds4-bench: valid backends are: metal, cuda, cpu\n", .{});
}

fn parseOptions(argv: [][*c]u8) Options {
    var options = Options{};
    var args = Args{ .argv = argv };
    while (args.current()) |arg| : (args.index += 1) {
        if (same(arg, "-h") or same(arg, "--help")) {
            usage(c.__stdoutp, args.helpTopic());
            c.exit(0);
        }

        var message: [256]u8 = @splat(0);
        const distributed = c.ds4_dist_parse_cli_arg(arg.ptr, &args.index, @intCast(argv.len), argv.ptr, &options.dist, &message, message.len);
        if (distributed == c.DS4_DIST_CLI_ERROR) fatal(2, "ds4-bench: %s\n", .{if (message[0] != 0) message[0..].ptr else "invalid distributed option"});
        if (distributed == c.DS4_DIST_CLI_MATCHED) continue;
        @memset(&message, 0);
        const tp = c.ds4_tp_parse_cli_arg(arg.ptr, &args.index, @intCast(argv.len), argv.ptr, &options.tp, &message, message.len);
        if (tp == c.DS4_TP_CLI_ERROR) fatal(2, "ds4-bench: %s\n", .{if (message[0] != 0) message[0..].ptr else "invalid tensor-parallel option"});
        if (tp == c.DS4_TP_CLI_MATCHED) continue;

        if (same(arg, "-m") or same(arg, "--model")) {
            options.model_path = args.value(arg);
        } else if (same(arg, "--mtp-model")) {
            options.mtp_path = args.value(arg);
        } else if (same(arg, "--dspark")) {
            options.dspark = true;
        } else if (same(arg, "--dspark-confidence")) {
            const value = decimal(args.value(arg), arg);
            if (value < 0 or value > 1) fatal(2, "ds4-bench: --dspark-confidence must be between 0 and 1\n", .{});
            options.dspark = true;
            options.dspark_confidence_threshold = @floatCast(value);
            options.dspark_confidence_threshold_set = true;
        } else if (same(arg, "--prompt-file")) {
            options.prompt_path = args.value(arg);
        } else if (same(arg, "--vision")) {
            options.vision_path = args.value(arg);
        } else if (same(arg, "--chat-prompt-file")) {
            options.chat_prompt_path = args.value(arg);
        } else if (same(arg, "-sys") or same(arg, "--system")) {
            options.system = args.value(arg);
        } else if (same(arg, "--ctx-start")) {
            options.ctx_start = integer(args.value(arg), arg, false);
        } else if (same(arg, "--ctx-max")) {
            options.ctx_max = integer(args.value(arg), arg, false);
        } else if (same(arg, "--ctx-alloc")) {
            options.ctx_alloc = integer(args.value(arg), arg, false);
        } else if (same(arg, "--step-incr")) {
            options.step_incr = integer(args.value(arg), arg, false);
        } else if (same(arg, "--step-mul")) {
            options.step_mul = decimal(args.value(arg), arg);
        } else if (same(arg, "--gen-tokens") or same(arg, "--tokens") or same(arg, "-n")) {
            options.gen_tokens = integer(args.value(arg), arg, true);
        } else if (same(arg, "--csv")) {
            options.csv_path = args.value(arg);
        } else if (same(arg, "--dump-frontier-logits-dir")) {
            options.dump_frontier_logits_dir = args.value(arg);
        } else if (same(arg, "--expert-profile")) {
            options.expert_profile_path = args.value(arg);
        } else if (same(arg, "-t") or same(arg, "--threads")) {
            options.threads = integer(args.value(arg), arg, false);
        } else if (same(arg, "--backend")) {
            options.backend = backend(args.value(arg), arg);
        } else if (same(arg, "--metal")) {
            options.backend = c.DS4_BACKEND_METAL;
        } else if (same(arg, "--cuda")) {
            options.backend = c.DS4_BACKEND_CUDA;
        } else if (same(arg, "--gpu-vram")) {
            options.gpu_vram_arg = args.value(arg);
        } else if (same(arg, "--gpu-devices")) {
            options.gpu_devices_arg = args.value(arg);
        } else if (same(arg, "--cuda-tensor-parallel")) {
            options.cuda_tensor_parallel = true;
        } else if (same(arg, "--cpu")) {
            options.backend = c.DS4_BACKEND_CPU;
        } else if (same(arg, "--quality")) {
            options.quality = true;
        } else if (same(arg, "--ssd-streaming")) {
            options.ssd_streaming = true;
        } else if (same(arg, "--ssd-streaming-cold")) {
            options.ssd_streaming_cold = true;
        } else if (same(arg, "--ssd-streaming-cache-experts")) {
            if (!c.ds4_parse_streaming_cache_experts_arg(args.value(arg).ptr, &options.ssd_streaming_cache_experts, &options.ssd_streaming_cache_bytes)) {
                fatal(2, "ds4-bench: --ssd-streaming-cache-experts must be a positive count or <number>GB\n", .{});
            }
        } else if (same(arg, "--ssd-streaming-full-layers")) {
            options.ssd_streaming_full_layers = @intCast(integer(args.value(arg), arg, true));
            options.ssd_streaming_full_layers_set = true;
        } else if (same(arg, "--ssd-streaming-preload-experts")) {
            options.ssd_streaming_preload_experts = @intCast(integer(args.value(arg), arg, false));
        } else if (same(arg, "--simulate-used-memory")) {
            if (!c.ds4_parse_gib_arg(args.value(arg).ptr, &options.simulate_used_memory_bytes)) fatal(2, "ds4-bench: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n", .{});
        } else if (same(arg, "--prefill-chunk")) {
            options.prefill_chunk = @intCast(integer(args.value(arg), arg, false));
        } else if (same(arg, "--power")) {
            options.power_percent = integer(args.value(arg), arg, false);
            if (options.power_percent > 100) fatal(2, "ds4-bench: --power must be between 1 and 100\n", .{});
        } else if (same(arg, "--warm-weights")) {
            options.warm_weights = true;
        } else if (same(arg, "--show-output")) {
            options.show_output = true;
        } else if (same(arg, "--teacher-forced-decode")) {
            options.teacher_forced_decode = true;
        } else {
            report("ds4-bench: unknown option: %s\n", .{arg.ptr});
            usage(c.__stderrp, null);
            c.exit(2);
        }
    }

    if ((options.prompt_path == null) == (options.chat_prompt_path == null)) fatal(2, "ds4-bench: specify exactly one of --prompt-file or --chat-prompt-file\n", .{});
    if (options.dspark and options.mtp_path == null) fatal(2, "ds4-bench: --dspark requires --mtp-model FILE\n", .{});
    if (options.dspark and options.teacher_forced_decode) fatal(2, "ds4-bench: --dspark cannot be combined with --teacher-forced-decode\n", .{});
    if (options.ctx_start > options.ctx_max) fatal(2, "ds4-bench: --ctx-start must be <= --ctx-max\n", .{});
    if (options.step_mul < 1) fatal(2, "ds4-bench: --step-mul must be >= 1\n", .{});
    if (options.step_mul == 1 and options.step_incr <= 0) fatal(2, "ds4-bench: --step-incr must be positive when --step-mul is 1\n", .{});
    if (options.ctx_max > std.math.maxInt(c_int) - options.gen_tokens - 1) fatal(2, "ds4-bench: requested context is too large\n", .{});
    if (options.ctx_alloc == 0) options.ctx_alloc = options.ctx_max + options.gen_tokens + 1;
    if (options.ctx_alloc <= options.ctx_max + options.gen_tokens) fatal(2, "ds4-bench: --ctx-alloc must be greater than ctx-max + gen-tokens\n", .{});
    var message: [256]u8 = undefined;
    if (c.ds4_tp_adopt_distributed_options(&options.tp, &options.dist, &message, message.len) == 0) fatal(2, "ds4-bench: %s\n", .{&message});
    if (c.ds4_dist_prepare_engine_options(&options.dist, null, &message, message.len) != 0) fatal(2, "ds4-bench: %s\n", .{&message});
    if (options.dist.role == c.DS4_DISTRIBUTED_WORKER or options.tp.role == c.DS4_TP_WORKER) fatal(2, "ds4-bench: --role worker is a serving mode; start workers with ./ds4\n", .{});
    return options;
}

fn readFile(path: [:0]const u8) ?[:0]u8 {
    const file = c.fopen(path.ptr, "rb");
    if (file == null) {
        report("ds4-bench: failed to open %s: %s\n", .{ path.ptr, systemError() });
        return null;
    }
    defer _ = c.fclose(file);
    if (c.fseek(file, 0, c.SEEK_END) != 0) {
        report("ds4-bench: failed to seek %s\n", .{path.ptr});
        return null;
    }
    const length = c.ftell(file);
    if (length < 0) {
        report("ds4-bench: failed to tell %s\n", .{path.ptr});
        return null;
    }
    if (c.fseek(file, 0, c.SEEK_SET) != 0) {
        report("ds4-bench: failed to rewind %s\n", .{path.ptr});
        return null;
    }
    const bytes = allocator.allocSentinel(u8, @intCast(length), 0) catch {
        report("ds4-bench: out of memory reading %s\n", .{path.ptr});
        return null;
    };
    if (c.fread(bytes.ptr, 1, bytes.len, file) != bytes.len) {
        report("ds4-bench: failed to read %s\n", .{path.ptr});
        allocator.free(bytes);
        return null;
    }
    return bytes;
}

fn jsonString(file: [*c]File, text: []const u8) void {
    _ = c.fputc('"', file);
    for (text) |byte| {
        switch (byte) {
            '"' => _ = c.fputs("\\\"", file),
            '\\' => _ = c.fputs("\\\\", file),
            '\x08' => _ = c.fputs("\\b", file),
            '\x0c' => _ = c.fputs("\\f", file),
            '\n' => _ = c.fputs("\\n", file),
            '\r' => _ = c.fputs("\\r", file),
            '\t' => _ = c.fputs("\\t", file),
            else => if (byte < 0x20) {
                _ = c.fprintf(file, "\\u%04x", @as(c_uint, byte));
            } else {
                _ = c.fputc(byte, file);
            },
        }
    }
    _ = c.fputc('"', file);
}

// This is the PREFILL frontier distribution, before any decode evaluation.
// Keep the original nine-significant-digit JSON representation for comparisons.
fn writeFrontier(cfg: *const Options, engine: *Engine, session: *Session, frontier: c_int, previous: c_int) bool {
    const directory = cfg.dump_frontier_logits_dir orelse return true;
    const vocab = c.ds4_engine_vocab_size(engine);
    const logits = allocator.alloc(f32, @intCast(vocab)) catch {
        report("ds4-bench: out of memory copying frontier logits\n", .{});
        return false;
    };
    defer allocator.free(logits);
    if (c.ds4_session_copy_logits(session, logits.ptr, vocab) != vocab) {
        report("ds4-bench: failed to copy frontier logits at %d\n", .{frontier});
        return false;
    }
    var path: [c.PATH_MAX]u8 = undefined;
    const count = c.snprintf(&path, path.len, "%s/frontier_%06d.logits.json", directory.ptr, frontier);
    if (count <= 0 or @as(usize, @intCast(count)) >= path.len) {
        report("ds4-bench: frontier logits path is too long\n", .{});
        return false;
    }
    const file = c.fopen(&path, "wb");
    if (file == null) {
        report("ds4-bench: failed to open %s: %s\n", .{ &path, systemError() });
        return false;
    }
    const argmax = c.ds4_session_argmax(session);
    _ = c.fprintf(file, "{\n  \"source\":\"ds4-bench\",\n  \"model\":");
    jsonString(file, cfg.model_path);
    _ = c.fprintf(
        file,
        ",\n  \"backend\":\"%s\",\n  \"quality\":%s,\n" ++
            "  \"quant_bits\":%d,\n  \"prompt_tokens\":%d,\n" ++
            "  \"frontier_tokens\":%d,\n  \"prefill_tokens\":%d,\n" ++
            "  \"ctx\":%d,\n  \"vocab\":%d,\n" ++
            "  \"argmax_id\":%d,\n  \"argmax_logit\":%.9g,\n  \"logits\":[",
        c.ds4_backend_name(cfg.backend),
        @as([*:0]const u8, if (cfg.quality) "true" else "false"),
        c.ds4_engine_routed_quant_bits(engine),
        frontier,
        frontier,
        frontier - previous,
        cfg.ctx_alloc,
        vocab,
        argmax,
        @as(f64, logits[@intCast(argmax)]),
    );
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
        report("ds4-bench: failed to close %s\n", .{&path});
        return false;
    }
    return true;
}

fn nextFrontier(cfg: *const Options, current: c_int) c_int {
    @setFloatMode(.optimized);
    if (current >= cfg.ctx_max) return cfg.ctx_max;
    var next: c_int = undefined;
    if (cfg.step_mul == 1) {
        next = if (current > std.math.maxInt(c_int) - cfg.step_incr) cfg.ctx_max else current + cfg.step_incr;
    } else {
        const value = @ceil(@as(f64, @floatFromInt(current)) * cfg.step_mul);
        next = if (value > std.math.maxInt(c_int)) cfg.ctx_max else @intFromFloat(value);
        if (next <= current) next = current + 1;
    }
    return @min(next, cfg.ctx_max);
}

fn logContextMemory(cfg: *const Options, prefill_chunk: u32) void {
    @setFloatMode(.optimized);
    const memory = c.ds4_context_memory_estimate_with_prefill_mode(cfg.backend, cfg.ctx_alloc, prefill_chunk, cfg.ssd_streaming);
    report("ds4-bench: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n", .{
        @as(f64, @floatFromInt(memory.total_bytes)) / (1024.0 * 1024.0),
        cfg.ctx_alloc,
        c.ds4_backend_name(cfg.backend),
        memory.prefill_cap,
        memory.raw_cap,
        memory.comp_cap,
    });
}

fn waitDistributedRoute(session: *Session) bool {
    var message: [256]u8 = @splat(0);
    var last: [256]u8 = @splat(0);
    var ticks: c_uint = 0;
    const delay = c.struct_timespec{ .tv_sec = 0, .tv_nsec = 250_000_000 };
    while (true) {
        const ready = c.ds4_session_distributed_route_ready(session, &message, message.len);
        if (ready > 0) {
            if (ticks != 0) report("ds4-bench: distributed route ready\n", .{});
            return true;
        }
        if (ready < 0) {
            report("ds4-bench: distributed route readiness failed: %s\n", .{if (message[0] != 0) message[0..].ptr else "unknown error"});
            return false;
        }
        const why: [*:0]const u8 = if (message[0] != 0) @ptrCast(&message) else "route incomplete";
        if (c.strcmp(&last, why) != 0 or ticks % 20 == 0) {
            report("ds4-bench: waiting for distributed route: %s\n", .{why});
            _ = c.snprintf(&last, last.len, "%s", why);
        }
        _ = c.nanosleep(&delay, null);
        ticks +%= 1;
    }
}

fn warnDistributedShape(cfg: *const Options, session: *Session) void {
    if (cfg.dist.role != c.DS4_DISTRIBUTED_COORDINATOR) return;
    var chunk = cfg.dist.prefill_chunk;
    if (chunk == 0) {
        const capacity = c.ds4_session_prefill_cap(session);
        if (capacity > 0) chunk = @intCast(capacity);
    }
    if (chunk != 0 and cfg.step_mul == 1 and cfg.step_incr > 0 and @as(u32, @intCast(cfg.step_incr)) < chunk and cfg.ctx_start < cfg.ctx_max) {
        report("ds4-bench: note: --step-incr=%d is smaller than distributed prefill chunk %u; suffix rows will not show multi-chunk pipeline overlap\n", .{ cfg.step_incr, chunk });
    }
}

fn bindTensorParallel(engine: *Engine, cfg: *const Options, tp: *?*c.ds4_tp) bool {
    var identity = std.mem.zeroes(c.ds4_tp_identity);
    identity.gguf_bytes = c.ds4_engine_model_bytes(engine);
    identity.model_id = @intCast(c.ds4_engine_model_id(engine));
    identity.n_layer = @intCast(c.ds4_engine_layer_count(engine));
    identity.n_embd = @intCast(c.ds4_engine_embd_dim(engine));
    identity.n_vocab = @intCast(c.ds4_engine_vocab_size(engine));
    identity.quant_bits = @intCast(c.ds4_engine_routed_quant_bits(engine));
    identity.ctx_size = @intCast(cfg.ctx_alloc);
    c.ds4_engine_tp_gate_schedule(engine, &identity.gate_slot_start, &identity.gate_slot_step, &identity.gates_per_token, &identity.gate_slot_mask);
    var message: [256]u8 = undefined;
    if (c.ds4_tp_create(tp, &cfg.tp, &identity, &message, message.len) == 0 or c.ds4_engine_tp_bind(engine, tp.*, &message, message.len) == 0) {
        report("ds4-bench: %s\n", .{&message});
        return false;
    }
    return true;
}

const SpecTrace = struct {
    milliseconds: [17]f64 = @splat(0),
    calls: [17]c_uint = @splat(0),
    total: c_uint = 0,

    fn record(self: *SpecTrace, count: c_int, start: f64) void {
        @setFloatMode(.optimized);
        if (c.getenv("DS4_BENCH_SPEC_TRACE") == null or count <= 0 or count >= 17) return;
        const index: usize = @intCast(count);
        self.milliseconds[index] += (nowSeconds() - start) * 1e3;
        self.calls[index] +%= 1;
        self.total +%= 1;
        if (self.total % 32 != 0) return;
        report("ds4-bench: spec calls %u:", .{self.total});
        for (1..17) |width| {
            if (self.calls[width] != 0) report(" %d-tok x%u avg %.1f ms", .{ @as(c_int, @intCast(width)), self.calls[width], self.milliseconds[width] / @as(f64, @floatFromInt(self.calls[width])) });
        }
        report("\n", .{});
    }
};

fn showGenerated(engine: *Engine, frontier: c_int, tokens: []const c_int) void {
    if (tokens.len == 0) return;
    report("ds4-bench: gen[ctx=%d] decoded text: \"", .{frontier});
    for (tokens) |token| {
        var length: usize = 0;
        const text = c.ds4_token_text(engine, token, &length);
        if (text != null) {
            _ = c.fwrite(text, 1, length, c.__stderrp);
            c.free(text);
        }
    }
    report("\"\n", .{});
    _ = c.fflush(c.__stderrp);
}

fn checkCudaProfileRange() void {
    // This setting has no profiler calls on Metal, but keep the existing CLI
    // diagnostic for malformed values. Only the CUDA weak-symbol hooks are absent.
    const range = c.getenv("DS4_BENCH_CUDA_PROFILE_RANGE");
    if (range == null) return;
    var start: c_int = -1;
    var count: c_int = 0;
    if (c.sscanf(range, "%d:%d", &start, &count) != 2 or start < 0 or count <= 0) {
        report("ds4-bench: invalid DS4_BENCH_CUDA_PROFILE_RANGE=%s (expected START_TOKEN:TOKEN_COUNT)\n", .{range});
    }
}

fn run(cfg: *Options) c_int {
    @setFloatMode(.optimized);
    var gpu = std.mem.zeroes(c.ds4_gpu_config);
    var skip_cuda = false;
    const have_gpu_config = cfg.gpu_vram_arg != null or cfg.gpu_devices_arg != null;
    var message: [256]u8 = undefined;
    if (have_gpu_config) {
        if (c.parse_gpu_vram_arg(cString(cfg.gpu_vram_arg), cString(cfg.gpu_devices_arg), &gpu, &skip_cuda, &message, message.len) != 0) {
            report("ds4-bench: %s\n", .{&message});
            return 2;
        }
        cfg.backend = if (skip_cuda) c.DS4_BACKEND_CPU else c.DS4_BACKEND_CUDA;
    }

    var engine_options = cfg.engineOptions();
    if (c.ds4_dist_prepare_engine_options(&cfg.dist, &engine_options, &message, message.len) != 0 or
        c.ds4_tp_validate_engine_options(&engine_options, &message, message.len) == 0)
    {
        report("ds4-bench: %s\n", .{&message});
        return 2;
    }
    var maybe_engine: ?*Engine = null;
    if (have_gpu_config and !skip_cuda) {
        const was_auto = if (cfg.gpu_vram_arg) |value| same(value, "auto") else cfg.gpu_devices_arg != null;
        var layout: [256]u8 = undefined;
        if (c.format_gpu_layout_line(&gpu, was_auto, &layout, layout.len) > 0) {
            _ = c.fprintf(c.__stdoutp, "%s\n", &layout);
            _ = c.fflush(c.__stdoutp);
        }
        if (c.ds4_engine_create_with_gpu_config(&maybe_engine, &engine_options, &gpu) != 0) return 1;
    } else if (c.ds4_engine_open(&maybe_engine, &engine_options) != 0) return 1;
    const engine = maybe_engine orelse return 1;
    var tp_leader: ?*c.ds4_tp = null;
    defer {
        if (tp_leader) |leader| _ = c.ds4_tp_send_stop(leader);
        c.ds4_engine_close(engine);
        c.ds4_tp_free(tp_leader);
    }
    if (cfg.tp.role == c.DS4_TP_LEADER and !bindTensorParallel(engine, cfg, &tp_leader)) return 1;
    logContextMemory(cfg, c.ds4_engine_prefill_chunk(engine));

    var prompt = std.mem.zeroes(c.ds4_tokens);
    defer c.ds4_tokens_free(&prompt);
    {
        const text = readFile(cfg.prompt_path orelse cfg.chat_prompt_path.?) orelse return 1;
        defer allocator.free(text);
        if (cfg.chat_prompt_path != null) {
            c.ds4_encode_chat_prompt(engine, cfg.system.ptr, text.ptr, c.DS4_THINK_NONE, &prompt);
        } else {
            c.ds4_tokenize_text(engine, text.ptr, &prompt);
        }
    }

    // Forced decode consumes the fixed prompt's continuation, not model choices.
    const needed = cfg.ctx_max + (if (cfg.teacher_forced_decode) cfg.gen_tokens else @as(c_int, 0));
    if (prompt.len < needed) {
        report("ds4-bench: prompt has %d tokens, need at least %d%s\n", .{
            prompt.len, needed, @as([*:0]const u8, if (cfg.teacher_forced_decode) " for teacher-forced decode" else ""),
        });
        return 1;
    }
    // The C engine owns this allocation; slices below only borrow its tokens.
    const prompt_ids = prompt.v[0..@as(usize, @intCast(prompt.len))];
    var maybe_session: ?*Session = null;
    if (c.ds4_session_create(&maybe_session, engine, cfg.ctx_alloc) != 0) {
        report("ds4-bench: failed to create session\n", .{});
        return 1;
    }
    const session = maybe_session orelse return 1;
    defer c.ds4_session_free(session);
    if (cfg.dist.role == c.DS4_DISTRIBUTED_COORDINATOR and !waitDistributedRoute(session)) return 1;
    warnDistributedShape(cfg, session);

    var snapshot = std.mem.zeroes(c.ds4_session_snapshot);
    defer c.ds4_session_snapshot_free(&snapshot);
    const output = if (cfg.csv_path) |path| c.fopen(path.ptr, "wb") else c.__stdoutp;
    if (output == null) {
        report("ds4-bench: failed to open %s: %s\n", .{ cfg.csv_path.?.ptr, systemError() });
        return 1;
    }
    defer if (output != c.__stdoutp) {
        _ = c.fclose(output);
    };
    _ = c.fprintf(output, "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes\n");
    _ = c.fflush(output);

    const eos = c.ds4_token_eos(engine);
    const distributed = cfg.dist.role == c.DS4_DISTRIBUTED_COORDINATOR or cfg.tp.role == c.DS4_TP_LEADER;
    const speculative = cfg.dspark and c.ds4_engine_mtp_draft_tokens(engine) > 1;
    if (cfg.dspark and !speculative) {
        report("ds4-bench: DSpark support model did not enable speculative decoding\n", .{});
        return 1;
    }
    if (speculative) report("ds4-bench: DSpark enabled with draft width %d; frontier restoration uses session snapshots\n", .{c.ds4_engine_mtp_draft_tokens(engine)});
    const snapshot_max_bytes = snapshotLimit();
    var warned_large_snapshot = false;
    var previous: c_int = 0;
    var frontier = cfg.ctx_start;
    var trace = SpecTrace{};

    while (true) : (frontier = nextFrontier(cfg, frontier)) {
        const prefix = c.ds4_tokens{ .v = prompt.v, .len = frontier, .cap = frontier };
        const prefill_start = nowSeconds();
        if (c.ds4_session_sync(session, &prefix, &message, message.len) != 0) {
            report("ds4-bench: prefill to %d failed: %s\n", .{ frontier, &message });
            return 1;
        }
        const prefill_end = nowSeconds();
        const prefill_seconds = prefill_end - prefill_start;
        if (c.getenv("DS4_METAL_CB_TIMES") != null) report("ds4-bench: prefill window mono %.1f .. %.1f ms\n", .{ prefill_start * 1e3, prefill_end * 1e3 });
        const prefill_tokens: c_int = frontier - previous;
        if (!writeFrontier(cfg, engine, session, frontier, previous)) return 1;

        // Saving/restoring must never enter either benchmark timing window.
        // Approximate/replay sessions report payload_bytes==0 and use re-sync.
        const need_restore = cfg.gen_tokens > 0 and frontier < cfg.ctx_max;
        var have_snapshot = false;
        if (need_restore and !distributed and c.getenv("DS4_BENCH_DISABLE_SNAPSHOT") == null) {
            const payload_bytes = c.ds4_session_payload_bytes(session);
            const forced = c.getenv("DS4_BENCH_FORCE_SNAPSHOT") != null;
            if (payload_bytes > snapshot_max_bytes and !forced) {
                if (!warned_large_snapshot) {
                    report("ds4-bench: session payload snapshot is %.2f GiB, above the %.2f GiB benchmark limit; replaying prefixes instead (set DS4_BENCH_FORCE_SNAPSHOT=1 to force snapshots)\n", .{ bytesToGiB(payload_bytes), bytesToGiB(snapshot_max_bytes) });
                    warned_large_snapshot = true;
                }
            } else if (payload_bytes > 0) {
                if (c.ds4_session_save_snapshot(session, &snapshot, &message, message.len) != 0) {
                    report("ds4-bench: snapshot at %d failed: %s\n", .{ frontier, &message });
                    return 1;
                }
                have_snapshot = true;
            }
        }

        const generation_start = nowSeconds();
        var first_seconds: f64 = 0;
        var steady_seconds: f64 = 0;
        var first_tokens: c_int = 0;
        var generated_count: c_int = 0;
        const generated: ?[]c_int = if (cfg.show_output and cfg.gen_tokens > 0)
            allocator.alloc(c_int, @intCast(cfg.gen_tokens)) catch null
        else
            null;
        defer if (generated) |buffer| allocator.free(buffer);
        var display_count: usize = 0;
        checkCudaProfileRange();
        var stopped = false;
        var decode_ok = true;

        while (generated_count < cfg.gen_tokens and !stopped) {
            if (c.ds4_session_pos(session) + 1 >= c.ds4_session_ctx(session)) {
                report("ds4-bench: generation would exceed allocated context at frontier %d\n", .{frontier});
                decode_ok = false;
                break;
            }
            const token = if (cfg.teacher_forced_decode)
                prompt_ids[@intCast(frontier + generated_count)]
            else
                c.ds4_session_argmax_excluding(session, eos);
            if (token < 0) {
                report("ds4-bench: failed to choose non-EOS token at frontier %d\n", .{frontier});
                decode_ok = false;
                break;
            }
            const token_start = nowSeconds();
            var cycle: [17]c_int = undefined;
            var count: c_int = 1;
            if (speculative) {
                const spec_start = nowSeconds();
                count = c.ds4_session_eval_speculative_argmax(session, token, cfg.gen_tokens - generated_count, eos, &cycle, cycle.len, &message, message.len);
                trace.record(count, spec_start);
                if (count < 0) {
                    report("ds4-bench: DSpark decode at frontier %d failed: %s\n", .{ frontier, &message });
                    decode_ok = false;
                    break;
                }
                if (count == 0) {
                    report("ds4-bench: DSpark decode at frontier %d accepted no tokens\n", .{frontier});
                    decode_ok = false;
                    break;
                }
            } else {
                cycle[0] = token;
                if (c.ds4_session_eval(session, token, &message, message.len) != 0) {
                    report("ds4-bench: decode at frontier %d failed: %s\n", .{ frontier, &message });
                    decode_ok = false;
                    break;
                }
            }
            const token_end = nowSeconds();
            var cycle_tokens: c_int = 0;
            for (cycle[0..@as(usize, @intCast(count))]) |id| {
                if (generated_count >= cfg.gen_tokens) break;
                if (id == eos) {
                    stopped = true;
                    break;
                }
                if (generated) |buffer| {
                    buffer[display_count] = id;
                    display_count += 1;
                }
                generated_count += 1;
                cycle_tokens += 1;
            }
            // A speculative first cycle may emit several tokens. Exclude that
            // entire cycle (not just one token) from the steady-state columns.
            if (first_tokens == 0) {
                first_seconds = token_end - token_start;
                first_tokens = cycle_tokens;
            } else {
                steady_seconds += token_end - token_start;
            }
        }
        const generation_end = nowSeconds();
        if (generated) |buffer| showGenerated(engine, frontier, buffer[0..display_count]);
        if (!decode_ok) return 1;

        if (!need_restore) {
            // Nothing later depends on this frontier state.
        } else if (distributed or !have_snapshot) {
            if (c.ds4_session_sync(session, &prefix, &message, message.len) != 0) {
                report("ds4-bench: replay restore at %d failed: %s\n", .{ frontier, &message });
                return 1;
            }
        } else if (c.ds4_session_load_snapshot(session, &snapshot, &message, message.len) != 0) {
            report("ds4-bench: restore at %d failed: %s\n", .{ frontier, &message });
            return 1;
        }

        const generation_seconds = generation_end - generation_start;
        const steady_tokens: c_int = @max(generated_count - first_tokens, 0);
        _ = c.fprintf(
            output,
            "%d,%d,%.2f,%d,%.2f,%.3f,%d,%.2f,%llu\n",
            frontier,
            prefill_tokens,
            if (prefill_seconds > 0) @as(f64, @floatFromInt(prefill_tokens)) / prefill_seconds else @as(f64, 0),
            generated_count,
            if (generation_seconds > 0) @as(f64, @floatFromInt(generated_count)) / generation_seconds else @as(f64, 0),
            first_seconds * 1000.0,
            steady_tokens,
            if (steady_seconds > 0) @as(f64, @floatFromInt(steady_tokens)) / steady_seconds else @as(f64, 0),
            @as(c_ulonglong, if (have_snapshot) snapshot.len else 0),
        );
        _ = c.fflush(output);
        previous = frontier;
        if (frontier >= cfg.ctx_max) break;
    }
    return 0;
}

pub export fn main(argc: c_int, argv: [*c][*c]u8) callconv(.c) c_int {
    var options = parseOptions(argv[0..@as(usize, @intCast(argc))]);
    return run(&options);
}
