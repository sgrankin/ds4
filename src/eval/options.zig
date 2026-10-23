//! Eval options borrow argv. Keep libc numeric grammar and the public TP/dist
//! parsers; self-tests/list/regrade deliberately bypass model-option checks.
const std = @import("std");
const c = @import("eval_c");
const Common = @import("common.zig");
pub const Config = c.eval_config;
fn same(value: [*c]const u8, expected: []const u8) bool {
    return std.mem.eql(u8, Common.text(value), expected);
}
fn fail(comptime pattern: [*:0]const u8, values: anytype) noreturn {
    _ = @call(.auto, c.fprintf, .{ c.__stderrp, pattern } ++ values);
    c.exit(2);
}
fn invalid(option: [*c]const u8, value: [*c]const u8) noreturn {
    fail("ds4-eval: invalid value for %s: %s\n", .{ option, value });
}
fn integer(value: [*c]const u8, option: [*c]const u8, allow_zero: bool) c_int {
    var end: [*c]u8 = null;
    const parsed = c.strtol(value, &end, 10);
    if (value[0] == 0 or end[0] != 0 or parsed < (if (allow_zero) @as(c_long, 0) else 1) or parsed > std.math.maxInt(c_int)) invalid(option, value);
    return @intCast(parsed);
}
fn decimal(value: [*c]const u8, option: [*c]const u8, minimum: f32, maximum: f32) f32 {
    var end: [*c]u8 = null;
    const parsed = c.strtof(value, &end);
    if (value[0] == 0 or end[0] != 0 or !std.math.isFinite(parsed) or parsed < minimum or parsed > maximum) invalid(option, value);
    return parsed;
}
fn seed(value: [*c]const u8, option: [*c]const u8) u64 {
    var end: [*c]u8 = null;
    const parsed = c.strtoull(value, &end, 10);
    if (value[0] == 0 or end[0] != 0 or parsed == 0) invalid(option, value);
    return parsed;
}
pub fn suite(value: [*c]const u8, option: [*c]const u8) u32 {
    if (same(value, "core")) return c.EVAL_SUITE_CORE;
    if (same(value, "hard")) return c.EVAL_SUITE_HARD;
    if (same(value, "all")) return c.EVAL_SUITE_CORE | c.EVAL_SUITE_HARD;
    if (same(value, "hard-smoke")) return c.EVAL_SUITE_HARD_SMOKE;
    fail("ds4-eval: invalid value for %s: %s (expected core, hard, all, or hard-smoke)\n", .{ option, value });
}
fn backend(value: [*c]const u8, option: [*c]const u8) c.ds4_backend {
    if (same(value, "metal")) return c.DS4_BACKEND_METAL;
    if (same(value, if (c.DS4_EVAL_ROCM != 0) "rocm" else "cuda")) return c.DS4_BACKEND_CUDA;
    if (same(value, "cpu")) return c.DS4_BACKEND_CPU;
    _ = c.fprintf(c.__stderrp, "ds4-eval: invalid value for %s: %s\n", option, value);
    fail(if (c.DS4_EVAL_ROCM != 0) "ds4-eval: valid backends are: metal, rocm, cpu\n" else "ds4-eval: valid backends are: metal, cuda, cpu\n", .{});
}
const Args = struct {
    index: c_int = 1,
    argc: c_int,
    argv: [*c][*c]u8,
    fn value(self: *Args, option: [*c]const u8) [*c]const u8 {
        if (self.index + 1 >= self.argc) fail("ds4-eval: %s requires an argument\n", .{option});
        self.index += 1;
        return self.argv[@intCast(self.index)];
    }
};
pub fn parse(argc: c_int, argv: [*c][*c]u8, default_backend: c.ds4_backend) Config {
    var config = std.mem.zeroes(Config);
    config.model_path = "ds4flash.gguf";
    config.backend = default_backend;
    config.max_tokens = 16000;
    config.suite_mask = c.EVAL_SUITE_CORE;
    config.top_p = c.DS4_DEFAULT_TOP_P;
    config.min_p = c.DS4_DEFAULT_MIN_P;
    config.pause_ms = 350;
    config.soft_limit_reply_budget = 1024;
    config.hard_limit_reply_budget = 512;
    config.soft_limit_think_close_rank = 3;
    config.think_mode = c.DS4_THINK_HIGH;
    var args = Args{ .argc = argc, .argv = argv };
    while (args.index < argc) : (args.index += 1) {
        const option = argv[@intCast(args.index)];
        if (same(option, "-h") or same(option, "--help")) {
            const topic = if (args.index + 1 < argc and argv[@intCast(args.index + 1)][0] != '-') argv[@intCast(args.index + 1)] else null;
            c.ds4_help_print(c.__stdoutp, c.DS4_HELP_EVAL, topic);
            c.exit(0);
        }
        var errors: [256]u8 = @splat(0);
        const distributed = c.ds4_dist_parse_cli_arg(option, &args.index, argc, argv, &config.dist, &errors, errors.len);
        if (distributed == c.DS4_DIST_CLI_ERROR) fail("ds4-eval: %s\n", .{if (errors[0] != 0) @as([*c]const u8, &errors) else @as([*c]const u8, "invalid distributed option")});
        if (distributed == c.DS4_DIST_CLI_MATCHED) continue;
        @memset(&errors, 0);
        const tp = c.ds4_tp_parse_cli_arg(option, &args.index, argc, argv, &config.tp, &errors, errors.len);
        if (tp == c.DS4_TP_CLI_ERROR) fail("ds4-eval: %s\n", .{if (errors[0] != 0) @as([*c]const u8, &errors) else @as([*c]const u8, "invalid tensor-parallel option")});
        if (tp == c.DS4_TP_CLI_MATCHED) continue;
        if (same(option, "-m") or same(option, "--model")) {
            config.model_path = args.value(option);
        } else if (same(option, "--mtp-model")) {
            config.mtp_path = args.value(option);
        } else if (same(option, "-c") or same(option, "--ctx")) {
            config.ctx_size = integer(args.value(option), option, false);
        } else if (same(option, "-n") or same(option, "--tokens")) {
            config.max_tokens = integer(args.value(option), option, false);
            config.max_tokens_set = true;
        } else if (same(option, "--questions")) {
            config.question_limit = integer(args.value(option), option, false);
        } else if (same(option, "--case-sequence")) {
            config.case_sequence = args.value(option);
        } else if (same(option, "--suite")) {
            config.suite_mask = suite(args.value(option), option);
        } else if (same(option, "--source")) {
            config.source_filter = args.value(option);
        } else if (same(option, "--domain")) {
            config.domain_filter = args.value(option);
        } else if (same(option, "--case-id")) {
            config.case_id_filter = args.value(option);
        } else if (same(option, "--list-cases")) {
            config.list_cases = true;
        } else if (same(option, "--validate-cases")) {
            config.validate_cases = true;
        } else if (same(option, "--retry-incomplete")) {
            config.retry_incomplete = true;
        } else if (same(option, "--temp")) {
            config.temperature = decimal(args.value(option), option, 0, 100);
        } else if (same(option, "--top-p")) {
            config.top_p = decimal(args.value(option), option, 0, 1);
        } else if (same(option, "--min-p")) {
            config.min_p = decimal(args.value(option), option, 0, 1);
        } else if (same(option, "--seed")) {
            config.seed = seed(args.value(option), option);
        } else if (same(option, "--trace")) {
            config.trace_path = args.value(option);
        } else if (same(option, "--regrade-trace")) {
            config.regrade_trace_path = args.value(option);
        } else if (same(option, "--soft-limit-reply-budget")) {
            config.soft_limit_reply_budget = integer(args.value(option), option, false);
        } else if (same(option, "--hard-limit-reply-budget")) {
            config.hard_limit_reply_budget = integer(args.value(option), option, false);
        } else if (same(option, "--soft-limit-think-close-rank")) {
            config.soft_limit_think_close_rank = integer(args.value(option), option, false);
        } else if (same(option, "--pause-ms")) {
            config.pause_ms = integer(args.value(option), option, false);
        } else if (same(option, "-t") or same(option, "--threads")) {
            config.threads = integer(args.value(option), option, false);
        } else if (same(option, "--backend")) {
            config.backend = backend(args.value(option), option);
        } else if (same(option, "--metal")) {
            config.backend = c.DS4_BACKEND_METAL;
        } else if (same(option, if (c.DS4_EVAL_ROCM != 0) "--rocm" else "--cuda")) {
            config.backend = c.DS4_BACKEND_CUDA;
        } else if (same(option, "--cpu")) {
            config.backend = c.DS4_BACKEND_CPU;
        } else if (same(option, "--quality")) {
            config.quality = true;
        } else if (same(option, "--ssd-streaming")) {
            config.ssd_streaming = true;
        } else if (same(option, "--ssd-streaming-cold")) {
            config.ssd_streaming_cold = true;
        } else if (same(option, "--ssd-streaming-cache-experts")) {
            if (!c.ds4_parse_streaming_cache_experts_arg(args.value(option), &config.ssd_streaming_cache_experts, &config.ssd_streaming_cache_bytes)) fail("ds4-eval: --ssd-streaming-cache-experts must be a positive count or <number>GB\n", .{});
        } else if (same(option, "--ssd-streaming-full-layers")) {
            config.ssd_streaming_full_layers = @intCast(integer(args.value(option), option, true));
            config.ssd_streaming_full_layers_set = true;
        } else if (same(option, "--ssd-streaming-preload-experts")) {
            config.ssd_streaming_preload_experts = @intCast(integer(args.value(option), option, false));
        } else if (same(option, "--simulate-used-memory")) {
            if (!c.ds4_parse_gib_arg(args.value(option), &config.simulate_used_memory_bytes)) fail("ds4-eval: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n", .{});
        } else if (same(option, "--prefill-chunk")) {
            config.prefill_chunk = @intCast(integer(args.value(option), option, false));
        } else if (same(option, "--power")) {
            config.power_percent = integer(args.value(option), option, false);
            if (config.power_percent > 100) fail("ds4-eval: --power must be between 1 and 100\n", .{});
        } else if (same(option, "--warm-weights")) {
            config.warm_weights = true;
        } else if (same(option, "--think")) {
            config.think_mode = c.DS4_THINK_HIGH;
        } else if (same(option, "--think-max")) {
            config.think_mode = c.DS4_THINK_MAX;
        } else if (same(option, "--nothink")) {
            config.think_mode = c.DS4_THINK_NONE;
        } else if (same(option, "--plain")) {
            config.plain = true;
        } else if (same(option, "--self-test-extractors")) {
            config.self_test_extractors = true;
        } else {
            _ = c.fprintf(c.__stderrp, "ds4-eval: unknown option: %s\n", option);
            c.ds4_help_print(c.__stderrp, c.DS4_HELP_EVAL, null);
            c.exit(2);
        }
    }
    if (config.self_test_extractors or config.validate_cases or config.list_cases or config.regrade_trace_path != null) return config;
    var errors: [256]u8 = undefined;
    if (c.ds4_tp_adopt_distributed_options(&config.tp, &config.dist, &errors, errors.len) == 0) fail("ds4-eval: %s\n", .{&errors});
    if (c.ds4_dist_prepare_engine_options(&config.dist, null, &errors, errors.len) != 0) fail("ds4-eval: %s\n", .{&errors});
    if (config.dist.role == c.DS4_DISTRIBUTED_WORKER or config.tp.role == c.DS4_TP_WORKER) fail("ds4-eval: --role worker is a serving mode; start workers with ./ds4\n", .{});
    if (config.max_tokens > c.EVAL_MAX_CONTEXT) fail("ds4-eval: --tokens (%d) exceeds the %d token context cap\n", .{ config.max_tokens, @as(c_int, c.EVAL_MAX_CONTEXT) });
    if (config.ctx_size > c.EVAL_MAX_CONTEXT) fail("ds4-eval: --ctx (%d) exceeds the %d token context cap\n", .{ config.ctx_size, @as(c_int, c.EVAL_MAX_CONTEXT) });
    if (c.ds4_think_mode_enabled(config.think_mode) and config.hard_limit_reply_budget >= config.max_tokens) fail("ds4-eval: --hard-limit-reply-budget (%d) must be smaller than --tokens (%d)\n", .{ config.hard_limit_reply_budget, config.max_tokens });
    if (c.ds4_think_mode_enabled(config.think_mode) and config.soft_limit_reply_budget < config.hard_limit_reply_budget) fail("ds4-eval: --soft-limit-reply-budget (%d) must be >= --hard-limit-reply-budget (%d)\n", .{ config.soft_limit_reply_budget, config.hard_limit_reply_budget });
    return config;
}
