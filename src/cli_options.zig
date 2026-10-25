//! Shared native CLI configuration and actual progress/file helpers.
//! Strings borrow argv unless prompt_owned is present; prefix/dist use their
//! public C destructors. Engine and sessions are owned by their calling scope.
const std = @import("std");
pub const c = @import("cli_c");
pub const allocator = std.heap.c_allocator;
pub const Generation = struct {
    prompt: ?[:0]const u8 = null,
    system: ?[:0]const u8 = "You are a helpful assistant",
    prefix: c.ds4_prompt_prefix = std.mem.zeroes(c.ds4_prompt_prefix),
    raw_prompt: bool = false,
    n_predict: c_int = 50000,
    ctx_size: c_int = 32768,
    temperature: f32 = c.DS4_DEFAULT_TEMPERATURE,
    top_k: c_int = 0,
    top_p: f32 = c.DS4_DEFAULT_TOP_P,
    min_p: f32 = c.DS4_DEFAULT_MIN_P,
    temperature_set: bool = false,
    top_k_set: bool = false,
    top_p_set: bool = false,
    min_p_set: bool = false,
    seed: u64 = 0,
    dump_tokens: bool = false,
    dump_logits_path: ?[:0]const u8 = null,
    dump_logprobs_path: ?[:0]const u8 = null,
    dump_logprobs_top_k: c_int = 20,
    decode_consistency_tokens: c_int = 0,
    perplexity_file_path: ?[:0]const u8 = null,
    imatrix_dataset_path: ?[:0]const u8 = null,
    imatrix_output_path: ?[:0]const u8 = null,
    imatrix_max_prompts: c_int = 0,
    imatrix_max_tokens: c_int = 0,
    imatrix_min_expert_samples: c_int = 0,
    think_mode: c.ds4_think_mode = c.DS4_THINK_HIGH,
    head_test: bool = false,
    first_token_test: bool = false,
    metal_graph_test: bool = false,
    metal_graph_full_test: bool = false,
    metal_graph_prompt_test: bool = false,
};
pub const Options = struct {
    engine: c.ds4_engine_options = defaults: {
        var value = std.mem.zeroes(c.ds4_engine_options);
        value.model_path = "ds4flash.gguf";
        value.backend = c.DS4_BACKEND_METAL;
        value.mtp_draft_tokens = 1;
        value.mtp_margin = 3;
        break :defaults value;
    },
    dist: ?*c.ds4_dist_options = null,
    gen: Generation = .{},
    prompt_owned: ?[:0]u8 = null,
    inspect: bool = false,
    gpu_vram_arg: ?[:0]const u8 = null,
    gpu_devices_arg: ?[:0]const u8 = null,

    pub fn deinit(self: *Options) void {
        c.ds4_prompt_prefix_free(&self.gen.prefix);
        c.ds4_dist_options_free(self.dist);
        if (self.prompt_owned) |text| allocator.free(text);
    }
};
pub fn cString(text: ?[:0]const u8) ?[*:0]const u8 {
    return if (text) |value| value.ptr else null;
}
pub fn report(comptime format: [*:0]const u8, args: anytype) void {
    _ = @call(.auto, c.fprintf, .{ c.__stderrp, format } ++ args);
}
pub fn fatal(status: c_int, comptime format: [*:0]const u8, args: anytype) noreturn {
    report(format, args);
    c.exit(status);
}
pub fn effectiveThink(engine: *c.ds4_engine, gen: *const Generation) c.ds4_think_mode {
    return c.ds4_engine_think_mode_for_context(engine, gen.think_mode, gen.ctx_size);
}
pub fn nowSeconds() f64 {
    @setFloatMode(.optimized);
    var value: c.struct_timespec = undefined;
    _ = c.clock_gettime(c.CLOCK_MONOTONIC, &value);
    return @as(f64, @floatFromInt(value.tv_sec)) + @as(f64, @floatFromInt(value.tv_nsec)) * 1e-9;
}
pub fn coordinator(cfg: *const Options) bool {
    return cfg.engine.distributed.role == c.DS4_DISTRIBUTED_COORDINATOR;
}
pub fn waitRoute(cfg: *const Options, session: *c.ds4_session) bool {
    if (!coordinator(cfg)) return true;
    var message: [256]u8 = @splat(0);
    var last: [256]u8 = @splat(0);
    var ticks: c_uint = 0;
    const delay = c.struct_timespec{ .tv_sec = 0, .tv_nsec = 250000000 };
    while (true) {
        const ready = c.ds4_session_distributed_route_ready(session, &message, message.len);
        if (ready > 0) {
            if (ticks != 0) report("ds4: distributed route ready\n", .{});
            return true;
        }
        if (ready < 0) {
            report("ds4: distributed route readiness failed: %s\n", .{if (message[0] != 0) message[0..].ptr else "unknown error"});
            return false;
        }
        const why: [*:0]const u8 = if (message[0] != 0) @ptrCast(&message) else "route incomplete";
        if (c.strcmp(&last, why) != 0 or ticks % 20 == 0) {
            report("ds4: waiting for distributed route: %s\n", .{why});
            _ = c.snprintf(&last, last.len, "%s", why);
        }
        _ = c.nanosleep(&delay, null);
        ticks +%= 1;
    }
}
pub const Progress = struct {
    base_tokens: c_int = 0,
    input_tokens: c_int = 0,
    use_color: bool = false,
    finished: bool = false,

    pub fn install(self: *Progress, session: *c.ds4_session) void {
        c.ds4_session_set_progress(session, progressCallback, self);
        c.ds4_session_set_display_progress(session, if (self.use_color) progressCallback else null, if (self.use_color) self else null);
    }
};
pub fn clearProgress(session: *c.ds4_session) void {
    c.ds4_session_set_progress(session, null, null);
    c.ds4_session_set_display_progress(session, null, null);
}
pub fn progressCallback(context: ?*anyopaque, event: [*c]const u8, current: c_int, total: c_int) callconv(.c) void {
    @setFloatMode(.optimized);
    _ = total;
    const progress: *Progress = @ptrCast(@alignCast(context orelse return));
    if (event == null or progress.input_tokens <= 0) return;
    const display = c.strcmp(event, "prefill_display") == 0;
    if (c.strcmp(event, "prefill_chunk") != 0 and !display) return;
    if (display and !progress.use_color) return;
    const processed: c_int = @max(0, @min(current - progress.base_tokens, progress.input_tokens));
    const percent = @min(100, 100 * @as(f64, @floatFromInt(processed)) / @as(f64, @floatFromInt(progress.input_tokens)));
    const complete = processed >= progress.input_tokens;
    if (complete and progress.finished) return;
    if (progress.use_color) {
        _ = c.fputc('\r', c.__stderrp);
        c.ds4_log(c.__stderrp, c.DS4_LOG_PREFILL, "processing %d input tokens: %d/%d (%.1f%%)", progress.input_tokens, processed, progress.input_tokens, percent);
        _ = c.fputs("\x1b[K", c.__stderrp);
        if (complete) _ = c.fputc('\n', c.__stderrp);
    } else report("processing %d input tokens: %d/%d (%.1f%%)\n", .{ progress.input_tokens, processed, progress.input_tokens, percent });
    if (complete) progress.finished = true;
    _ = c.fflush(c.__stderrp);
}
pub fn readPromptFile(path: [:0]const u8, fatal_error: bool) ?[:0]u8 {
    const result = readFileChecked(path);
    if (result == null and fatal_error) c.exit(2);
    return result;
}
fn readFileChecked(path: [:0]const u8) ?[:0]u8 {
    const file = c.fopen(path.ptr, "rb");
    if (file == null) {
        report("ds4: failed to open prompt file: %s\n", .{path.ptr});
        return null;
    }
    var closed = false;
    defer if (!closed) {
        _ = c.fclose(file);
    };
    if (c.fseek(file, 0, c.SEEK_END) != 0) {
        report("ds4: failed to seek prompt file: %s\n", .{path.ptr});
        return null;
    }
    const length = c.ftell(file);
    if (length < 0) {
        report("ds4: failed to size prompt file: %s\n", .{path.ptr});
        return null;
    }
    c.rewind(file);
    const bytes = allocator.allocSentinel(u8, @intCast(length), 0) catch {
        report("ds4: out of memory reading prompt file: %s\n", .{path.ptr});
        return null;
    };
    if (c.fread(bytes.ptr, 1, bytes.len, file) != bytes.len) {
        report("ds4: failed to read prompt file: %s\n", .{path.ptr});
        allocator.free(bytes);
        return null;
    }
    const result = c.fclose(file);
    closed = true;
    if (result != 0) {
        report("ds4: failed to close prompt file: %s\n", .{path.ptr});
        allocator.free(bytes);
        return null;
    }
    return bytes;
}
