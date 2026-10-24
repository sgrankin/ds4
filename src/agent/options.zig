//! Agent options borrow argv; only prompt-file text and parsed prefix turns
//! own allocations. Keep libc's accepted numeric spelling and public TP/dist
//! parsers, while every Agent option/compatibility decision is native.
const std = @import("std");
const c = @import("agent_runtime_c");
const Ui = @import("ui_text.zig");
fn same(value: [*c]const u8, expected: []const u8) bool {
    return std.mem.eql(u8, Ui.text(value), expected);
}
pub fn fail(comptime pattern: [*:0]const u8, arguments: anytype) noreturn {
    _ = @call(.auto, c.fprintf, .{ c.__stderrp, pattern } ++ arguments);
    c.exit(2);
}
fn invalid(option: [*c]const u8, value: [*c]const u8) noreturn {
    fail("ds4-agent: invalid value for %s: %s\n", .{ option, value });
}
fn integer(value: [*c]const u8, option: [*c]const u8, nonnegative: bool) c_int {
    var end: [*c]u8 = null;
    const result = c.strtol(value, &end, 10);
    if (value[0] == 0 or end[0] != 0 or result < (if (nonnegative) @as(c_long, 0) else 1) or result > std.math.maxInt(c_int)) invalid(option, value);
    return @intCast(result);
}
fn decimal(value: [*c]const u8, option: [*c]const u8, lower: f32, upper: f32) f32 {
    var end: [*c]u8 = null;
    const result = c.strtof(value, &end);
    if (value[0] == 0 or end[0] != 0 or !std.math.isFinite(result) or result < lower or result > upper) invalid(option, value);
    return result;
}
fn seed(value: [*c]const u8, option: [*c]const u8) u64 {
    var end: [*c]u8 = null;
    const result = c.strtoull(value, &end, 10);
    if (value[0] == 0 or end[0] != 0 or result == 0) invalid(option, value);
    return result;
}
fn backend(value: [*c]const u8) c.ds4_backend {
    if (same(value, "metal")) return c.DS4_BACKEND_METAL;
    if (same(value, if (c.DS4_AGENT_FRONTEND_ROCM != 0) "rocm" else "cuda")) return c.DS4_BACKEND_CUDA;
    if (same(value, "cpu")) return c.DS4_BACKEND_CPU;
    _ = c.fprintf(c.__stderrp, "ds4-agent: invalid backend: %s\n", value);
    fail(if (c.DS4_AGENT_FRONTEND_ROCM != 0) "ds4-agent: valid backends are: metal, rocm, cpu\n" else "ds4-agent: valid backends are: metal, cuda, cpu\n", .{});
}
const Arguments = struct {
    argc: c_int,
    argv: [*c][*c]u8,
    index: c_int = 1,
    fn value(self: *Arguments, option: [*c]const u8) [*c]const u8 {
        if (self.index + 1 >= self.argc) fail("ds4-agent: missing value for %s\n", .{option});
        self.index += 1;
        return self.argv[@intCast(self.index)];
    }
};
pub fn enabled(name: [*:0]const u8) bool {
    const value = c.getenv(name);
    return value != null and (c.strcasecmp(value, "true") == 0 or c.strcasecmp(value, "yes") == 0 or same(value, "1"));
}
pub fn parse(argc: c_int, argv: [*c][*c]u8, default_backend: c.ds4_backend) c.agent_config {
    var config = std.mem.zeroes(c.agent_config);
    config.engine.model_path = "ds4flash.gguf";
    config.engine.backend = default_backend;
    config.engine.mtp_draft_tokens = 1;
    config.engine.mtp_margin = 3;
    config.gen.system = "You are a helpful coding assistant running inside ds4-agent.";
    config.gen.n_predict = 50000;
    config.gen.ctx_size = 100000;
    config.gen.temperature = c.DS4_DEFAULT_TEMPERATURE;
    config.gen.top_p = c.DS4_DEFAULT_TOP_P;
    config.gen.min_p = c.DS4_DEFAULT_MIN_P;
    config.gen.think_mode = c.DS4_THINK_HIGH;
    var steering_set = false;
    var args = Arguments{ .argc = argc, .argv = argv };
    while (args.index < argc) : (args.index += 1) {
        const option = argv[@intCast(args.index)];
        if (same(option, "-h") or same(option, "--help")) {
            const topic = if (args.index + 1 < argc and argv[@intCast(args.index + 1)][0] != '-') argv[@intCast(args.index + 1)] else null;
            c.ds4_help_print(c.__stdoutp, c.DS4_HELP_AGENT, topic);
            c.exit(0);
        }
        var errors: [256]u8 = @splat(0);
        const dist = c.ds4_dist_parse_cli_arg(option, &args.index, argc, argv, &config.engine.distributed, &errors, errors.len);
        if (dist == c.DS4_DIST_CLI_ERROR) fail("ds4-agent: %s\n", .{if (errors[0] != 0) @as([*c]const u8, &errors) else @as([*c]const u8, "invalid distributed option")});
        if (dist == c.DS4_DIST_CLI_MATCHED) continue;
        @memset(&errors, 0);
        const tp = c.ds4_tp_parse_cli_arg(option, &args.index, argc, argv, &config.engine.tp, &errors, errors.len);
        if (tp == c.DS4_TP_CLI_ERROR) fail("ds4-agent: %s\n", .{if (errors[0] != 0) @as([*c]const u8, &errors) else @as([*c]const u8, "invalid tensor-parallel option")});
        if (tp == c.DS4_TP_CLI_MATCHED) continue;
        if (same(option, "-p") or same(option, "--prompt")) {
            if (config.gen.prompt != null) fail("ds4-agent: specify only one of -p/--prompt and --prompt-file\n", .{});
            config.gen.prompt = args.value(option);
        } else if (same(option, "--prompt-file")) {
            if (config.gen.prompt != null) fail("ds4-agent: specify only one of -p/--prompt and --prompt-file\n", .{});
            const path = args.value(option);
            var length: usize = 0;
            if (c.ds4_agent_file_read_bytes(path, &config.gen.prompt_owned, &length, &errors, errors.len) != 0) fail("ds4-agent: %s\n", .{&errors});
            config.gen.prompt = config.gen.prompt_owned;
        } else if (same(option, "--prefix-file")) {
            if (config.gen.prefix.count != 0) fail("ds4-agent: specify --prefix-file only once\n", .{});
            if (c.ds4_prompt_prefix_load(&config.gen.prefix, args.value(option), &errors, errors.len) != 0) fail("ds4-agent: %s\n", .{if (errors[0] != 0) @as([*c]const u8, &errors) else @as([*c]const u8, "invalid prefix file")});
        } else if (same(option, "--non-interactive")) {
            config.non_interactive = true;
        } else if (same(option, "--raw") or same(option, "--raw-prompt")) {
            config.gen.raw_prompt = true;
        } else if (same(option, "--edit-upto")) {
            config.edit_upto = true;
        } else if (same(option, "-sys") or same(option, "--system")) {
            config.gen.system = args.value(option);
        } else if (same(option, "--trace")) {
            config.gen.trace_path = args.value(option);
        } else if (same(option, "--live-context-file")) {
            config.gen.live_context_path = args.value(option);
            if (config.gen.live_context_path[0] == 0) fail("ds4-agent: --live-context-file requires a nonempty path\n", .{});
        } else if (same(option, "-m") or same(option, "--model")) {
            config.engine.model_path = args.value(option);
        } else if (same(option, "--vision")) {
            config.engine.vision_path = args.value(option);
        } else if (same(option, "--mtp")) {
            config.engine.glm_mtp = true;
        } else if (same(option, "--mtp-model")) {
            config.engine.mtp_path = args.value(option);
        } else if (same(option, "--mtp-draft")) {
            config.engine.mtp_draft_tokens = integer(args.value(option), option, false);
        } else if (same(option, "--mtp-margin")) {
            config.engine.mtp_margin = decimal(args.value(option), option, 0, 1000);
        } else if (same(option, "--mtp-timing")) {
            config.engine.glm_mtp = true;
            config.engine.glm_mtp_timing = true;
        } else if (same(option, "--dspark")) {
            config.engine.dspark = true;
        } else if (same(option, "--dspark-confidence")) {
            config.engine.dspark = true;
            config.engine.dspark_confidence_threshold = decimal(args.value(option), option, 0, 1);
            config.engine.dspark_confidence_threshold_set = true;
        } else if (same(option, "--dspark-strict")) {
            config.engine.dspark = true;
            config.engine.dspark_strict = true;
        } else if (same(option, "--mtp-exact-sampling")) {
            config.engine.dspark_exact_sampling = true;
        } else if (same(option, "-c") or same(option, "--ctx")) {
            config.gen.ctx_size = integer(args.value(option), option, false);
        } else if (same(option, "-n") or same(option, "--tokens")) {
            config.gen.n_predict = integer(args.value(option), option, false);
        } else if (same(option, "--temp")) {
            config.gen.temperature = decimal(args.value(option), option, 0, 100);
            config.gen.temperature_set = true;
        } else if (same(option, "--top-p")) {
            config.gen.top_p = decimal(args.value(option), option, 0, 1);
            config.gen.top_p_set = true;
        } else if (same(option, "--min-p")) {
            config.gen.min_p = decimal(args.value(option), option, 0, 1);
            config.gen.min_p_set = true;
        } else if (same(option, "--seed")) {
            config.gen.seed = seed(args.value(option), option);
        } else if (same(option, "--think-level")) {
            if (!c.ds4_think_mode_parse_level(args.value(option), &config.gen.think_mode)) fail("ds4-agent: --think-level requires an integer from 0 to 100\n", .{});
        } else if (same(option, "--think")) {
            config.gen.think_mode = c.DS4_THINK_HIGH;
        } else if (same(option, "--think-max")) {
            config.gen.think_mode = c.DS4_THINK_MAX;
        } else if (same(option, "--nothink")) {
            config.gen.think_mode = c.DS4_THINK_NONE;
        } else if (same(option, "--backend")) {
            config.engine.backend = backend(args.value(option));
        } else if (same(option, "--metal")) {
            config.engine.backend = c.DS4_BACKEND_METAL;
        } else if (same(option, if (c.DS4_AGENT_FRONTEND_ROCM != 0) "--rocm" else "--cuda")) {
            config.engine.backend = c.DS4_BACKEND_CUDA;
        } else if (same(option, "--gpu-vram")) {
            config.gpu_vram_arg = args.value(option);
        } else if (same(option, "--gpu-devices")) {
            config.gpu_devices_arg = args.value(option);
        } else if (same(option, "--cuda-tensor-parallel")) {
            config.engine.cuda_tensor_parallel = true;
        } else if (same(option, "--cpu")) {
            config.engine.backend = c.DS4_BACKEND_CPU;
        } else if (same(option, "-t") or same(option, "--threads")) {
            config.engine.n_threads = integer(args.value(option), option, false);
        } else if (same(option, "--chdir")) {
            config.chdir_path = args.value(option);
        } else if (same(option, "--quality")) {
            config.engine.quality = true;
        } else if (same(option, "--ssd-streaming")) {
            config.engine.ssd_streaming = true;
        } else if (same(option, "--ssd-streaming-cold")) {
            config.engine.ssd_streaming_cold = true;
        } else if (same(option, "--ssd-streaming-cache-experts")) {
            if (!c.ds4_parse_streaming_cache_experts_arg(args.value(option), &config.engine.ssd_streaming_cache_experts, &config.engine.ssd_streaming_cache_bytes)) fail("ds4-agent: --ssd-streaming-cache-experts must be a positive count or <number>GB\n", .{});
        } else if (same(option, "--ssd-streaming-full-layers")) {
            config.engine.ssd_streaming_full_layers = @intCast(integer(args.value(option), option, true));
            config.engine.ssd_streaming_full_layers_set = true;
        } else if (same(option, "--ssd-streaming-preload-experts")) {
            config.engine.ssd_streaming_preload_experts = @intCast(integer(args.value(option), option, false));
        } else if (same(option, "--simulate-used-memory")) {
            if (!c.ds4_parse_gib_arg(args.value(option), &config.engine.simulate_used_memory_bytes)) fail("ds4-agent: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n", .{});
        } else if (same(option, "--prefill-chunk")) {
            config.engine.prefill_chunk = @intCast(integer(args.value(option), option, false));
        } else if (same(option, "--power")) {
            config.engine.power_percent = integer(args.value(option), option, false);
            if (config.engine.power_percent < 1 or config.engine.power_percent > 100) fail("ds4-agent: --power must be between 1 and 100\n", .{});
        } else if (same(option, "--warm-weights")) {
            config.engine.warm_weights = true;
        } else if (same(option, "--dir-steering-file")) {
            config.engine.directional_steering_file = args.value(option);
        } else if (same(option, "--dir-steering-mode")) {
            const value = args.value(option);
            config.engine.directional_steering_mode = if (same(value, "project")) c.DS4_STEERING_PROJECT else if (same(value, "add")) c.DS4_STEERING_ADD else {
                invalid(option, value);
            };
        } else if (same(option, "--dir-steering-ffn")) {
            config.engine.directional_steering_ffn = decimal(args.value(option), option, -100, 100);
            steering_set = true;
        } else if (same(option, "--dir-steering-attn")) {
            config.engine.directional_steering_attn = decimal(args.value(option), option, -100, 100);
            steering_set = true;
        } else {
            _ = c.fprintf(c.__stderrp, "ds4-agent: unknown option: %s\n", option);
            c.ds4_help_print(c.__stderrp, c.DS4_HELP_AGENT, null);
            c.exit(2);
        }
    }
    if (config.engine.directional_steering_file != null and !steering_set) config.engine.directional_steering_ffn = 1;
    var errors: [256]u8 = undefined;
    if (c.ds4_tp_adopt_distributed_options(&config.engine.tp, &config.engine.distributed, &errors, errors.len) == 0) fail("ds4-agent: %s\n", .{&errors});
    if (c.ds4_dist_prepare_engine_options(&config.engine.distributed, &config.engine, &errors, errors.len) != 0) fail("ds4-agent: %s\n", .{&errors});
    if (c.ds4_tp_validate_engine_options(&config.engine, &errors, errors.len) == 0) fail("ds4-agent: %s\n", .{&errors});
    if (config.engine.distributed.role == c.DS4_DISTRIBUTED_WORKER or config.engine.tp.role == c.DS4_TP_WORKER) fail("ds4-agent: --role worker is a serving mode; start workers with ./ds4\n", .{});
    if (config.gen.raw_prompt and (!config.non_interactive or config.gen.prompt == null)) fail("ds4-agent: --raw-prompt requires --non-interactive and an initial prompt\n", .{});
    if (config.gen.raw_prompt and config.gen.prefix.count != 0) fail("ds4-agent: --prefix-file cannot be combined with --raw-prompt\n", .{});
    if (config.gen.live_context_path != null and (config.gen.raw_prompt or config.engine.vision_path != null)) fail("ds4-agent: --live-context-file requires text-only agent chat (no --raw-prompt or --vision)\n", .{});
    const suffix = enabled("DS4_AGENT_CONTEXT_SUFFIX_REUSE");
    if (suffix and !enabled("DS4_AGENT_CONTEXT_SUFFIX_PREROPE") and (config.gen.live_context_path == null or residentConflict())) fail("ds4-agent: legacy DS4_AGENT_CONTEXT_SUFFIX_REUSE requires --live-context-file and resident routing disabled; enable DS4_AGENT_CONTEXT_SUFFIX_PREROPE=1 for built-in compaction and resident routing\n", .{});
    return config;
}
fn residentConflict() bool {
    const value = c.getenv("DS4_METAL_V41_RESIDENT_ROUTE_GAP");
    if (Ui.text(value).len == 0) return false;
    var end: [*c]u8 = null;
    const gap = c.strtod(value, &end);
    return end == null or end[0] != 0 or !std.math.isFinite(gap) or gap != 0;
}
pub fn modelDefaults(engine: ?*c.ds4_engine, generation: ?*c.agent_generation_options) void {
    const options = generation orelse return;
    if (engine == null or !c.ds4_engine_is_glm_dsa(engine)) return;
    if (!options.temperature_set) options.temperature = 1;
    if (!options.top_p_set) options.top_p = 0.95;
    if (!options.min_p_set) options.min_p = 0;
}
