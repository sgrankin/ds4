//! Complete server option, startup, thread/accept/drain and resource lifetime.
//! Actual runtime/config records are shared once. Engine and Session are opaque.
const std = @import("std");
const c = @import("server_boot_c");
const allocator = std.heap.c_allocator;

fn zero(comptime T: type) T {
    return std.mem.zeroes(T);
}
fn text(value: [*c]const u8) []const u8 {
    return if (value == null) &.{} else std.mem.span(value);
}
fn is(value: [*c]const u8, expected: []const u8) bool {
    return std.mem.eql(u8, text(value), expected);
}
fn flag(name: [*:0]const u8) bool {
    return is(c.getenv(name), "1");
}
fn fatal() noreturn {
    _ = c.fprintf(c.__stderrp, "ds4-server: out of memory\n");
    c.exit(1);
}
fn allocate(comptime T: type, count: usize) []T {
    const values = allocator.alloc(T, count) catch fatal();
    @memset(values, zero(T));
    return values;
}
fn log(kind: c.ds4_log_type, comptime format: [*:0]const u8, args: anytype) void {
    const size = @call(.auto, c.snprintf, .{ @as([*c]u8, null), @as(usize, 0), format } ++ args);
    if (size < 0) return;
    const message = allocator.allocSentinel(u8, @intCast(size), 0) catch fatal();
    defer allocator.free(message);
    _ = @call(.auto, c.snprintf, .{ message.ptr, message.len + 1, format } ++ args);
    c.ds4_server_persist_log(kind, message.ptr);
}
fn invalid(option: [*c]const u8, value: [*c]const u8) noreturn {
    log(c.DS4_LOG_DEFAULT, "ds4-server: invalid value for %s: %s", .{ option, value });
    c.exit(2);
}
fn integer(value: [*c]const u8, option: [*c]const u8, nonnegative: bool) c_int {
    var end: [*c]u8 = null;
    const result = c.strtol(value, &end, 10);
    if (value[0] == 0 or end[0] != 0 or result < (if (nonnegative) @as(c_long, 0) else 1) or result > std.math.maxInt(c_int)) invalid(option, value);
    return @intCast(result);
}
fn real(value: [*c]const u8, option: [*c]const u8, minimum: f32, maximum: f32) f32 {
    @setFloatMode(.strict);
    var end: [*c]u8 = null;
    const result = c.strtof(value, &end);
    // Preserve libc parsing and the C comparison policy, including NaN.
    if (value[0] == 0 or end[0] != 0 or result < minimum or result > maximum) invalid(option, value);
    return result;
}
const Arguments = struct {
    argc: c_int,
    argv: [*c][*c]u8,
    index: c_int = 1,
    fn value(self: *Arguments, option: [*c]const u8) [*c]const u8 {
        if (self.index + 1 >= self.argc) {
            log(c.DS4_LOG_DEFAULT, "ds4-server: missing value for %s", .{option});
            c.exit(2);
        }
        self.index += 1;
        return self.argv[@intCast(self.index)];
    }
};
fn backend(value: [*c]const u8, option: [*c]const u8) c.ds4_backend {
    if (is(value, "metal")) return c.DS4_BACKEND_METAL;
    if (is(value, if (c.DS4_SERVER_BOOT_ROCM != 0) "rocm" else "cuda")) return c.DS4_BACKEND_CUDA;
    if (is(value, "cpu")) return c.DS4_BACKEND_CPU;
    log(c.DS4_LOG_DEFAULT, "ds4-server: invalid %s value: %s", .{ option, value });
    log(c.DS4_LOG_DEFAULT, if (c.DS4_SERVER_BOOT_ROCM != 0) "ds4-server: valid server backends are: metal, rocm, cpu" else "ds4-server: valid server backends are: metal, cuda, cpu", .{});
    c.exit(2);
}
pub fn parse(argc: c_int, argv: [*c][*c]u8, default_backend: c.ds4_backend) c.server_config {
    var cfg = zero(c.server_config);
    cfg.engine.model_path = "ds4flash.gguf";
    cfg.engine.backend = default_backend;
    cfg.engine.mtp_draft_tokens = 1;
    cfg.engine.mtp_margin = 3.0;
    cfg.host = "127.0.0.1";
    cfg.port = 8000;
    cfg.ctx_size = 32768;
    cfg.default_tokens = 393216;
    cfg.tool_memory_max_ids = c.DS4_TOOL_MEMORY_DEFAULT_MAX_IDS;
    cfg.mixed_prefill_quantum = 128;
    cfg.kv_cache = c.ds4_kvstore_default_options();
    var steering_scale_set = false;
    var args = Arguments{ .argc = argc, .argv = argv };
    while (args.index < argc) : (args.index += 1) {
        const option = argv[@intCast(args.index)];
        if (is(option, "-h") or is(option, "--help")) {
            const topic = if (args.index + 1 < argc and argv[@intCast(args.index + 1)][0] != '-') argv[@intCast(args.index + 1)] else null;
            c.ds4_help_print(c.__stdoutp, c.DS4_HELP_SERVER, topic);
            c.exit(0);
        }
        var errors: [256]u8 = @splat(0);
        const distributed = c.ds4_dist_parse_cli_arg(option, &args.index, argc, argv, &cfg.engine.distributed, &errors, errors.len);
        if (distributed == c.DS4_DIST_CLI_ERROR) {
            log(c.DS4_LOG_DEFAULT, "ds4-server: %s", .{@as([*c]const u8, if (errors[0] != 0) &errors else "invalid distributed option")});
            c.exit(2);
        }
        if (distributed == c.DS4_DIST_CLI_MATCHED) continue;
        @memset(&errors, 0);
        const tensor_parallel = c.ds4_tp_parse_cli_arg(option, &args.index, argc, argv, &cfg.engine.tp, &errors, errors.len);
        if (tensor_parallel == c.DS4_TP_CLI_ERROR) {
            log(c.DS4_LOG_DEFAULT, "ds4-server: %s", .{@as([*c]const u8, if (errors[0] != 0) &errors else "invalid tensor-parallel option")});
            c.exit(2);
        }
        if (tensor_parallel == c.DS4_TP_CLI_MATCHED) continue;

        if (is(option, "-m") or is(option, "--model")) {
            cfg.engine.model_path = args.value(option);
        } else if (is(option, "--vision")) {
            cfg.engine.vision_path = args.value(option);
        } else if (is(option, "--mtp")) {
            cfg.engine.glm_mtp = true;
        } else if (is(option, "--mtp-model")) {
            cfg.engine.mtp_path = args.value(option);
        } else if (is(option, "--mtp-draft")) {
            cfg.engine.mtp_draft_tokens = integer(args.value(option), option, false);
        } else if (is(option, "--mtp-margin")) {
            cfg.engine.mtp_margin = real(args.value(option), option, 0, 1000);
        } else if (is(option, "--mtp-timing")) {
            cfg.engine.glm_mtp = true;
            cfg.engine.glm_mtp_timing = true;
        } else if (is(option, "--dspark")) {
            cfg.engine.dspark = true;
        } else if (is(option, "--dspark-confidence")) {
            cfg.engine.dspark = true;
            cfg.engine.dspark_confidence_threshold = real(args.value(option), option, 0, 1);
            cfg.engine.dspark_confidence_threshold_set = true;
        } else if (is(option, "--dspark-strict")) {
            cfg.engine.dspark = true;
            cfg.engine.dspark_strict = true;
        } else if (is(option, "--mtp-exact-sampling")) {
            cfg.engine.dspark_exact_sampling = true;
        } else if (is(option, "-c") or is(option, "--ctx")) {
            cfg.ctx_size = integer(args.value(option), option, false);
        } else if (is(option, "-n") or is(option, "--tokens")) {
            cfg.default_tokens = integer(args.value(option), option, false);
        } else if (is(option, "-t") or is(option, "--threads")) {
            cfg.engine.n_threads = integer(args.value(option), option, false);
        } else if (is(option, "--chdir")) {
            cfg.chdir_path = args.value(option);
        } else if (is(option, "--host")) {
            cfg.host = args.value(option);
        } else if (is(option, "--port")) {
            cfg.port = integer(args.value(option), option, false);
        } else if (is(option, "--cors")) {
            cfg.enable_cors = true;
        } else if (is(option, "--trace")) {
            cfg.trace_path = args.value(option);
        } else if (is(option, "--batched-session")) {
            cfg.batched_sessions = integer(args.value(option), option, false);
        } else if (is(option, "--mixed-prefill-quantum")) {
            cfg.mixed_prefill_quantum = integer(args.value(option), option, false);
            cfg.mixed_prefill_quantum_set = true;
        } else if (is(option, "--kv-disk-dir")) {
            cfg.kv_disk_dir = args.value(option);
        } else if (is(option, "--kv-disk-space-mb")) {
            cfg.kv_disk_space_mb = @intCast(integer(args.value(option), option, false));
        } else if (is(option, "--kv-cache-min-tokens")) {
            cfg.kv_cache.min_tokens = integer(args.value(option), option, false);
        } else if (is(option, "--kv-cache-cold-max-tokens")) {
            cfg.kv_cache.cold_max_tokens = integer(args.value(option), option, true);
        } else if (is(option, "--kv-cache-continued-interval-tokens")) {
            cfg.kv_cache.continued_interval_tokens = integer(args.value(option), option, true);
        } else if (is(option, "--kv-cache-boundary-trim-tokens")) {
            cfg.kv_cache.boundary_trim_tokens = integer(args.value(option), option, true);
        } else if (is(option, "--kv-cache-boundary-align-tokens")) {
            cfg.kv_cache.boundary_align_tokens = integer(args.value(option), option, true);
        } else if (is(option, "--kv-cache-reject-different-quant")) {
            cfg.kv_cache_reject_different_quant = true;
        } else if (is(option, "--disable-exact-dsml-tool-replay")) {
            cfg.disable_exact_dsml_tool_replay = true;
        } else if (is(option, "--tool-memory-max-ids")) {
            cfg.tool_memory_max_ids = integer(args.value(option), option, false);
        } else if (is(option, "--quality")) {
            cfg.engine.quality = true;
        } else if (is(option, "--ssd-streaming")) {
            cfg.engine.ssd_streaming = true;
        } else if (is(option, "--ssd-streaming-cold")) {
            cfg.engine.ssd_streaming_cold = true;
        } else if (is(option, "--ssd-streaming-cache-experts")) {
            if (!c.ds4_parse_streaming_cache_experts_arg(args.value(option), &cfg.engine.ssd_streaming_cache_experts, &cfg.engine.ssd_streaming_cache_bytes)) {
                log(c.DS4_LOG_DEFAULT, "ds4-server: --ssd-streaming-cache-experts must be a positive count or <number>GB", .{});
                c.exit(2);
            }
        } else if (is(option, "--ssd-streaming-full-layers")) {
            cfg.engine.ssd_streaming_full_layers = @intCast(integer(args.value(option), option, true));
            cfg.engine.ssd_streaming_full_layers_set = true;
        } else if (is(option, "--ssd-streaming-preload-experts")) {
            cfg.engine.ssd_streaming_preload_experts = @intCast(integer(args.value(option), option, false));
        } else if (is(option, "--simulate-used-memory")) {
            if (!c.ds4_parse_gib_arg(args.value(option), &cfg.engine.simulate_used_memory_bytes)) {
                log(c.DS4_LOG_DEFAULT, "ds4-server: --simulate-used-memory must be a positive GiB value, e.g. 64GB", .{});
                c.exit(2);
            }
        } else if (is(option, "--prefill-chunk")) {
            cfg.engine.prefill_chunk = @intCast(integer(args.value(option), option, false));
        } else if (is(option, "--flint-profile")) {
            cfg.engine.flint_profile = args.value(option);
        } else if (is(option, "--power")) {
            cfg.engine.power_percent = integer(args.value(option), option, false);
            if (cfg.engine.power_percent < 1 or cfg.engine.power_percent > 100) {
                log(c.DS4_LOG_DEFAULT, "ds4-server: --power must be between 1 and 100", .{});
                c.exit(2);
            }
        } else if (is(option, "--dir-steering-file")) {
            cfg.engine.directional_steering_file = args.value(option);
        } else if (is(option, "--dir-steering-mode")) {
            const value = args.value(option);
            cfg.engine.directional_steering_mode = if (is(value, "project")) c.DS4_STEERING_PROJECT else if (is(value, "add")) c.DS4_STEERING_ADD else {
                log(c.DS4_LOG_DEFAULT, "ds4-server: --dir-steering-mode must be project or add", .{});
                c.exit(2);
            };
        } else if (is(option, "--dir-steering-ffn")) {
            cfg.engine.directional_steering_ffn = real(args.value(option), option, -100, 100);
            steering_scale_set = true;
        } else if (is(option, "--dir-steering-attn")) {
            cfg.engine.directional_steering_attn = real(args.value(option), option, -100, 100);
            steering_scale_set = true;
        } else if (is(option, "--warm-weights")) {
            cfg.engine.warm_weights = true;
        } else if (is(option, "--metal")) {
            cfg.engine.backend = c.DS4_BACKEND_METAL;
        } else if (is(option, if (c.DS4_SERVER_BOOT_ROCM != 0) "--rocm" else "--cuda")) {
            cfg.engine.backend = c.DS4_BACKEND_CUDA;
        } else if (is(option, "--gpu-vram")) {
            cfg.gpu_vram_arg = args.value(option);
        } else if (is(option, "--gpu-devices")) {
            cfg.gpu_devices_arg = args.value(option);
        } else if (is(option, "--cuda-tensor-parallel")) {
            cfg.engine.cuda_tensor_parallel = true;
        } else if (is(option, "--backend")) {
            cfg.engine.backend = backend(args.value(option), option);
        } else if (is(option, "--cpu")) {
            cfg.engine.backend = c.DS4_BACKEND_CPU;
        } else {
            log(c.DS4_LOG_DEFAULT, "ds4-server: unknown option: %s", .{option});
            c.ds4_help_print(c.__stderrp, c.DS4_HELP_SERVER, null);
            c.exit(2);
        }
    }
    if (cfg.kv_cache.cold_max_tokens > 0 and cfg.kv_cache.cold_max_tokens < cfg.kv_cache.min_tokens) {
        log(c.DS4_LOG_DEFAULT, "ds4-server: --kv-cache-cold-max-tokens must be 0 or >= --kv-cache-min-tokens", .{});
        c.exit(2);
    }
    if (cfg.engine.directional_steering_file != null and !steering_scale_set) cfg.engine.directional_steering_ffn = 1;
    var errors: [256]u8 = undefined;
    if (c.ds4_tp_adopt_distributed_options(&cfg.engine.tp, &cfg.engine.distributed, &errors, errors.len) == 0) {
        log(c.DS4_LOG_DEFAULT, "ds4-server: %s", .{@as([*c]const u8, &errors)});
        c.exit(2);
    }
    if (c.ds4_dist_prepare_engine_options(&cfg.engine.distributed, &cfg.engine, &errors, errors.len) != 0) {
        log(c.DS4_LOG_DEFAULT, "ds4-server: %s", .{@as([*c]const u8, &errors)});
        c.exit(2);
    }
    if (c.ds4_tp_validate_engine_options(&cfg.engine, &errors, errors.len) == 0) {
        log(c.DS4_LOG_DEFAULT, "ds4-server: %s", .{@as([*c]const u8, &errors)});
        c.exit(2);
    }
    if (cfg.engine.tp.role == c.DS4_TP_WORKER) {
        log(c.DS4_LOG_DEFAULT, "ds4-server: --role worker is a serving mode; start tensor-parallel workers with ./ds4", .{});
        c.exit(2);
    }
    return cfg;
}

pub fn close(srv: *c.server) void {
    if (srv.trace != null) {
        _ = c.fclose(srv.trace);
        srv.trace = null;
    }
    if (srv.state_cache != null) {
        var stats: c.ds4_state_cache_stats = undefined;
        c.ds4_state_cache_get_stats(srv.state_cache, &stats);
        log(c.DS4_LOG_KVCACHE, "ds4-server: state_cache_summary {\"hot_bytes\":%llu,\"ssd_bytes\":%llu,\"metadata_bytes\":%llu,\"endpoints\":%llu,\"chunks\":%llu,\"pinned_endpoints\":%llu,\"hot_reads\":%llu,\"disk_reads\":%llu,\"endpoint_evictions\":%llu}", .{
            @as(c_ulonglong, stats.hot_bytes),
            @as(c_ulonglong, stats.ssd_bytes),
            @as(c_ulonglong, stats.metadata_bytes),
            @as(c_ulonglong, stats.endpoints),
            @as(c_ulonglong, stats.chunks),
            @as(c_ulonglong, stats.pinned_endpoints),
            @as(c_ulonglong, stats.hot_reads),
            @as(c_ulonglong, stats.disk_reads),
            @as(c_ulonglong, stats.endpoint_evictions),
        });
        var errors: [160]u8 = @splat(0);
        if (c.ds4_state_cache_close(srv.state_cache, &errors, errors.len) != 0) log(c.DS4_LOG_WARNING, "ds4-server: state cache close failed: %s", .{@as([*c]const u8, &errors)});
        srv.state_cache = null;
    }
    c.ds4_kvstore_close(&srv.kv);
    c.ds4_server_tool_memory_free(&srv.tool_mem);
    c.ds4_server_images_cache_clear(&srv.image_cache);
    for (0..@intCast(srv.slot_count)) |index| {
        const slot = &srv.slots[index];
        c.ds4_server_reuse_tool_free(&slot.responses_live);
        c.ds4_server_reuse_tool_free(&slot.anthropic_live);
        c.ds4_server_reuse_visible_free(&slot.thinking_live);
        c.free(slot.live_text);
        if (slot.session != null) c.ds4_session_free(slot.session);
    }
    c.free(@ptrCast(srv.slot_threads));
    c.free(srv.slots);
    _ = c.pthread_mutex_destroy(&srv.tool_mu);
    _ = c.pthread_mutex_destroy(&srv.kv_mu);
    _ = c.pthread_mutex_destroy(&srv.inference_mu);
    _ = c.pthread_mutex_destroy(&srv.model_mu);
    _ = c.pthread_mutex_destroy(&srv.trace_mu);
    _ = c.pthread_cond_destroy(&srv.model_cv);
    _ = c.pthread_cond_destroy(&srv.clients_cv);
    _ = c.pthread_cond_destroy(&srv.cv);
    _ = c.pthread_mutex_destroy(&srv.mu);
    if (srv.tp_leader != null) _ = c.ds4_tp_send_stop(srv.tp_leader);
    c.ds4_engine_close(srv.engine);
    c.ds4_tp_free(srv.tp_leader);
    srv.* = zero(c.server);
}
fn contextMemory(cfg: *const c.server_config, engine: ?*c.ds4_engine, slots: c_int) void {
    const memory = c.ds4_engine_context_memory_estimate(engine, cfg.ctx_size);
    if (c.ds4_engine_is_flint(engine)) {
        const mib: f64 = 1024 * 1024;
        const gib = mib * 1024;
        log(c.DS4_LOG_DEFAULT, "ds4-server: Flint configured context buffers up to %.2f MiB/session (ctx=%d, BF16 KV bank capacity=%u, prefill_chunk=%u, KV=%.2f MiB, scratch/snapshots=%.2f MiB); %d independent sessions up to %.2f GiB total", .{
            @as(f64, @floatFromInt(memory.total_bytes)) / mib,
            cfg.ctx_size,
            memory.raw_cap,
            memory.prefill_cap,
            @as(f64, @floatFromInt(memory.raw_bytes)) / mib,
            @as(f64, @floatFromInt(memory.scratch_bytes)) / mib,
            slots,
            @as(f64, @floatFromInt(memory.total_bytes)) * @as(f64, @floatFromInt(slots)) / gib,
        });
        return;
    }
    log(c.DS4_LOG_DEFAULT, "ds4-server: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)", .{
        @as(f64, @floatFromInt(memory.total_bytes)) / (1024 * 1024),
        cfg.ctx_size,
        c.ds4_backend_name(cfg.engine.backend),
        memory.prefill_cap,
        memory.raw_cap,
        memory.comp_cap,
    });
    if (slots > 1) {
        // Only caches repeat per slot; shared-prefill transients are paid once.
        const gib: f64 = 1024 * 1024 * 1024;
        const caches: f64 = @floatFromInt(memory.raw_bytes + memory.compressed_bytes);
        log(c.DS4_LOG_DEFAULT, "ds4-server: %d resident sessions request at least %.2f GiB of context buffers, or %.2f GiB when the model shares one prefill workspace", .{
            slots,
            @as(f64, @floatFromInt(memory.total_bytes)) * @as(f64, @floatFromInt(slots)) / gib,
            (caches * @as(f64, @floatFromInt(slots)) + @as(f64, @floatFromInt(memory.scratch_bytes))) / gib,
        });
    }
}
fn cacheLog(_: ?*anyopaque, kind: c.ds4_kvstore_log_type, message: [*c]const u8) callconv(.c) void {
    c.ds4_server_persist_log(if (kind == c.DS4_KVSTORE_LOG_DEFAULT) c.DS4_LOG_DEFAULT else if (kind == c.DS4_KVSTORE_LOG_WARNING) c.DS4_LOG_WARNING else c.DS4_LOG_KVCACHE, message);
}
const WorkerContext = struct { runtime: *c.server, stop: [*c]const volatile c.sig_atomic_t };
const SlotContext = struct { slot: *c.server_slot, stop: [*c]const volatile c.sig_atomic_t };
const ClientContext = struct { packet: c.client_arg, stop: [*c]const volatile c.sig_atomic_t };
fn workerEntry(pointer: ?*anyopaque) callconv(.c) ?*anyopaque {
    const context: *const WorkerContext = @ptrCast(@alignCast(pointer.?));
    c.ds4_server_serving_worker(context.runtime, context.stop);
    return null;
}
fn slotEntry(pointer: ?*anyopaque) callconv(.c) ?*anyopaque {
    const context: *const SlotContext = @ptrCast(@alignCast(pointer.?));
    c.ds4_server_serving_slot_worker(context.slot, context.stop);
    return null;
}
fn clientEntry(pointer: ?*anyopaque) callconv(.c) ?*anyopaque {
    const owned: *ClientContext = @ptrCast(@alignCast(pointer.?));
    const context = owned.*;
    allocator.destroy(owned);
    c.ds4_server_serving_client(@ptrCast(context.packet.srv), context.packet.fd, context.stop);
    return null;
}
fn decodeEntry(pointer: ?*anyopaque) callconv(.c) ?*anyopaque {
    c.ds4_server_coordinator_run(@ptrCast(@alignCast(pointer.?)));
    return null;
}
fn joinWorkers(srv: *c.server, worker: c.pthread_t, started: usize, decoder_started: bool) void {
    c.ds4_server_serving_stop(srv);
    if (srv.batched_mode) {
        for (0..started) |index| _ = c.pthread_join(srv.slot_threads[index], null);
        c.ds4_server_coordinator_stop(srv);
        if (decoder_started) _ = c.pthread_join(srv.decode_thread, null);
    } else _ = c.pthread_join(worker, null);
}

pub fn run(argc: c_int, argv: [*c][*c]u8, stop: [*c]volatile c.sig_atomic_t, listen_fd: [*c]volatile c.sig_atomic_t, default_backend: c.ds4_backend) c_int {
    c.ds4_server_boot_install_signals();
    var cfg = parse(argc, argv, default_backend);
    if (cfg.chdir_path != null and c.chdir(cfg.chdir_path) != 0) {
        log(c.DS4_LOG_DEFAULT, "ds4-server: failed to chdir to %s: %s", .{ cfg.chdir_path, c.strerror(c.__error().*) });
        return 1;
    }
    cfg.engine.context_size = cfg.ctx_size;
    cfg.engine.placement_ctx_hint = cfg.ctx_size;
    cfg.engine.placement_session_count_hint = if (cfg.batched_sessions > 0) cfg.batched_sessions else 1;
    cfg.engine.share_session_prefill_workspace = cfg.batched_sessions > 0;
    var engine: ?*c.ds4_engine = null;
    if (cfg.gpu_vram_arg != null or cfg.gpu_devices_arg != null) {
        var gpu = zero(c.ds4_gpu_config);
        var skip_cuda = false;
        var errors: [256]u8 = undefined;
        if (c.parse_gpu_vram_arg(cfg.gpu_vram_arg, cfg.gpu_devices_arg, &gpu, &skip_cuda, &errors, errors.len) != 0) {
            _ = c.fprintf(c.__stderrp, "ds4-server: %s\n", @as([*c]const u8, &errors));
            return 2;
        }
        cfg.engine.backend = if (skip_cuda) c.DS4_BACKEND_CPU else c.DS4_BACKEND_CUDA;
        if (skip_cuda) {
            if (c.ds4_engine_open(&engine, &cfg.engine) != 0) return 1;
        } else {
            const automatic = (cfg.gpu_vram_arg != null and is(cfg.gpu_vram_arg, "auto")) or (cfg.gpu_vram_arg == null and cfg.gpu_devices_arg != null);
            var layout: [256]u8 = undefined;
            if (c.format_gpu_layout_line(&gpu, automatic, &layout, layout.len) > 0) {
                _ = c.fprintf(c.__stdoutp, "%s\n", @as([*c]const u8, &layout));
                _ = c.fflush(c.__stdoutp);
            }
            if (c.ds4_engine_create_with_gpu_config(&engine, &cfg.engine, &gpu) != 0) return 1;
        }
    } else if (c.ds4_engine_open(&engine, &cfg.engine) != 0) return 1;
    if (cfg.engine.distributed.role == c.DS4_DISTRIBUTED_WORKER) {
        var gen = zero(c.ds4_dist_generation_options);
        gen.ctx_size = cfg.ctx_size;
        const rc = c.ds4_dist_run(engine, &cfg.engine.distributed, &gen);
        c.ds4_engine_close(engine);
        return rc;
    }
    var leader: ?*c.ds4_tp = null;
    if (cfg.engine.tp.role == c.DS4_TP_LEADER) {
        var errors: [256]u8 = @splat(0);
        var identity = zero(c.ds4_tp_identity);
        identity.gguf_bytes = c.ds4_engine_model_bytes(engine);
        identity.model_id = @bitCast(c.ds4_engine_model_id(engine));
        identity.n_layer = @bitCast(c.ds4_engine_layer_count(engine));
        identity.n_embd = @bitCast(c.ds4_engine_embd_dim(engine));
        identity.n_vocab = @bitCast(c.ds4_engine_vocab_size(engine));
        identity.quant_bits = @bitCast(c.ds4_engine_routed_quant_bits(engine));
        identity.ctx_size = @bitCast(cfg.ctx_size);
        c.ds4_engine_tp_gate_schedule(engine, &identity.gate_slot_start, &identity.gate_slot_step, &identity.gates_per_token, &identity.gate_slot_mask);
        if (c.ds4_tp_create(&leader, &cfg.engine.tp, &identity, &errors, errors.len) == 0 or c.ds4_engine_tp_bind(engine, leader, &errors, errors.len) == 0) {
            log(c.DS4_LOG_DEFAULT, "ds4-server: %s", .{@as([*c]const u8, &errors)});
            c.ds4_tp_free(leader);
            c.ds4_engine_close(engine);
            return 1;
        }
    }
    if (c.ds4_engine_is_flint(engine) and !cfg.mixed_prefill_quantum_set) cfg.mixed_prefill_quantum = 1024;
    const slot_count = if (cfg.batched_sessions > 0) cfg.batched_sessions else 1;
    contextMemory(&cfg, engine, slot_count);
    var srv = zero(c.server);
    srv.engine = engine;
    srv.tp_leader = leader;
    srv.ctx_size = cfg.ctx_size;
    srv.slot_count = slot_count;
    srv.batched_mode = cfg.batched_sessions > 0;
    srv.qwen4_batch_mtp = cfg.engine.backend == c.DS4_BACKEND_METAL and c.ds4_engine_is_qwen4(engine) and c.ds4_engine_mtp_draft_tokens(engine) > 1;
    srv.mixed_prefill_quantum = cfg.mixed_prefill_quantum;
    srv.last_prefill_slot = slot_count - 1;
    srv.default_tokens = cfg.default_tokens;
    srv.disable_exact_dsml_tool_replay = cfg.disable_exact_dsml_tool_replay;
    srv.tool_mem.max_entries = cfg.tool_memory_max_ids;
    srv.enable_cors = cfg.enable_cors;
    srv.slots = allocate(c.server_slot, @intCast(slot_count)).ptr;
    if (srv.batched_mode) srv.slot_threads = allocate(c.pthread_t, @intCast(slot_count)).ptr;
    _ = c.pthread_mutex_init(&srv.mu, null);
    _ = c.pthread_cond_init(&srv.cv, null);
    _ = c.pthread_cond_init(&srv.clients_cv, null);
    _ = c.pthread_mutex_init(&srv.tool_mu, null);
    _ = c.pthread_mutex_init(&srv.kv_mu, null);
    var attr: c.pthread_mutexattr_t = undefined;
    _ = c.pthread_mutexattr_init(&attr);
    _ = c.pthread_mutexattr_settype(&attr, c.PTHREAD_MUTEX_RECURSIVE);
    _ = c.pthread_mutex_init(&srv.inference_mu, &attr);
    _ = c.pthread_mutexattr_destroy(&attr);
    _ = c.pthread_mutex_init(&srv.model_mu, null);
    _ = c.pthread_cond_init(&srv.model_cv, null);
    _ = c.pthread_mutex_init(&srv.trace_mu, null);
    defer close(&srv);
    for (0..@intCast(slot_count)) |index| {
        const slot = &srv.slots[index];
        slot.srv = &srv;
        slot.id = @intCast(index);
        if (c.ds4_session_create(&slot.session, engine, cfg.ctx_size) != 0) {
            log(c.DS4_LOG_DEFAULT, "ds4-server: failed to create %s session %d/%d", .{
                c.ds4_backend_name(cfg.engine.backend),
                @as(c_int, @intCast(index + 1)),
                slot_count,
            });
            return 1;
        }
    }
    const state_dir = c.getenv("DS4_STATE_CACHE_DIR");
    if (state_dir != null and state_dir[0] != 0) {
        var errors: [160]u8 = @splat(0);
        srv.state_cache = c.ds4_state_cache_runtime_open(engine, &cfg.engine, cfg.ctx_size, &srv.state_cache_profile, &errors, errors.len);
        if (srv.state_cache == null) {
            log(c.DS4_LOG_DEFAULT, "ds4-server: state cache startup failed: %s", .{@as([*c]const u8, &errors)});
            return 1;
        }
        log(c.DS4_LOG_KVCACHE, "ds4-server: state cache enabled dir=%s profile=%s", .{ state_dir, @as([*c]const u8, &srv.state_cache_profile) });
    }
    const response_dir = c.getenv("DS4_SERVER_RESPONSE_STORE_DIR");
    if (response_dir != null and response_dir[0] != 0) {
        var errors: [160]u8 = @splat(0);
        if (c.ds4_response_store_init(response_dir, &errors, errors.len) != 0) {
            log(c.DS4_LOG_DEFAULT, "ds4-server: response store startup failed: %s", .{@as([*c]const u8, &errors)});
            return 1;
        }
        srv.responses_store_dir = response_dir;
        log(c.DS4_LOG_DEFAULT, "ds4-server: durable text/tool responses enabled dir=%s", .{response_dir});
    }
    srv.context_suffix_reuse = flag("DS4_SERVER_CONTEXT_SUFFIX_REUSE");
    srv.context_suffix_canonical = flag("DS4_SERVER_CONTEXT_SUFFIX_CANONICAL");
    srv.context_suffix_pad = flag("DS4_SERVER_CONTEXT_SUFFIX_PAD");
    srv.context_suffix_right = flag("DS4_SERVER_CONTEXT_SUFFIX_RIGHT");
    srv.context_suffix_fresh = 128;
    const fresh = c.getenv("DS4_SERVER_CONTEXT_SUFFIX_FRESH_TAIL");
    if (fresh != null and fresh[0] != 0) srv.context_suffix_fresh = c.atoi(fresh);
    if (srv.context_suffix_reuse) {
        if (srv.state_cache == null or cfg.engine.backend != c.DS4_BACKEND_METAL or !c.ds4_engine_is_deepseek41(engine) or leader != null or cfg.engine.directional_steering_file != null or srv.context_suffix_fresh < 64 or srv.context_suffix_fresh > 4096) {
            log(c.DS4_LOG_DEFAULT, "ds4-server: suffix reuse requires native text V4.1 Metal, DS4_STATE_CACHE_DIR, and fresh tail 64..4096", .{});
            return 1;
        }
        var errors: [160]u8 = @splat(0);
        if (srv.context_suffix_pad) {
            var newline = zero(c.ds4_tokens);
            c.ds4_tokenize_rendered_chat(engine, "\n", &newline);
            var length: usize = 0;
            const piece = if (newline.len == 1) c.ds4_token_text(engine, newline.v[0], &length) else null;
            const valid = piece != null and length == 1 and piece[0] == '\n';
            if (valid) srv.context_suffix_pad_token = newline.v[0];
            if (!valid) log(c.DS4_LOG_DEFAULT, "ds4-server: suffix padding requires one literal newline token", .{});
            c.free(piece);
            c.ds4_tokens_free(&newline);
            if (!valid) return 1;
        }
        for (0..@intCast(slot_count)) |index| if (c.ds4_session_enable_suffix_prerope(srv.slots[index].session, &errors, errors.len) != 0) {
            log(c.DS4_LOG_DEFAULT, "ds4-server: suffix originals startup failed: %s", .{@as([*c]const u8, &errors)});
            return 1;
        };
        log(c.DS4_LOG_KVCACHE, "ds4-server: experimental suffix reuse enabled fresh_tail=%d canonical_full_input=%d prefix_pad=%d right_shift=%d requires=prompt_cache_key", .{
            srv.context_suffix_fresh,
            @as(c_int, @intFromBool(srv.context_suffix_canonical)),
            @as(c_int, @intFromBool(srv.context_suffix_pad)),
            @as(c_int, @intFromBool(srv.context_suffix_right)),
        });
    }
    if (cfg.kv_disk_dir != null) _ = c.ds4_kvstore_open(&srv.kv, cfg.kv_disk_dir, cfg.kv_disk_space_mb, cfg.kv_cache_reject_different_quant, cfg.kv_cache, "ds4-server", cacheLog, null);
    if (srv.disable_exact_dsml_tool_replay) log(c.DS4_LOG_DEFAULT, "ds4-server: exact DSML tool replay disabled; tool history uses canonical JSON rendering", .{});
    if (srv.batched_mode) {
        log(c.DS4_LOG_DEFAULT, "ds4-server: batched mode enabled resident_sessions=%d prefill_quantum=%d mixed_prefill_quantum=%d decode_coalesce_us=%ld", .{
            srv.slot_count,
            c.ds4_server_coordinator_quantum_for(&srv, false),
            c.ds4_server_coordinator_quantum_for(&srv, true),
            c.ds4_server_coordinator_coalesce_us(),
        });
        if (c.ds4_engine_mtp_draft_tokens(engine) > 1 and !srv.qwen4_batch_mtp) log(c.DS4_LOG_DEFAULT, "ds4-server: MTP speculative decoding is disabled while native session batching is active", .{});
    }
    if (cfg.trace_path != null) {
        srv.trace = c.fopen(cfg.trace_path, "w");
        if (srv.trace == null) {
            log(c.DS4_LOG_DEFAULT, "ds4-server: failed to open trace file %s: %s", .{ cfg.trace_path, c.strerror(c.__error().*) });
            return 1;
        }
        _ = c.setvbuf(srv.trace, null, c._IONBF, 0);
        log(c.DS4_LOG_DEFAULT, "ds4-server: tracing session to %s", .{cfg.trace_path});
    }

    // These private envelopes borrow actual records and the existing signal
    // flag. They add no runtime fields and outlive every joined callback.
    var worker_context = WorkerContext{ .runtime = &srv, .stop = stop };
    const slot_contexts = allocator.alloc(SlotContext, if (srv.batched_mode) @as(usize, @intCast(slot_count)) else 0) catch fatal();
    defer allocator.free(slot_contexts);
    // Debug inference frames exceed Darwin's default small pthread stack.
    var inference_attributes: c.pthread_attr_t = undefined;
    if (c.pthread_attr_init(&inference_attributes) != 0) return 1;
    defer _ = c.pthread_attr_destroy(&inference_attributes);
    if (c.pthread_attr_setstacksize(&inference_attributes, 8 * 1024 * 1024) != 0) return 1;
    var worker: c.pthread_t = zero(c.pthread_t);
    var started: usize = 0;
    var decoder_started = false;
    if (srv.batched_mode) {
        if (c.pthread_create(&srv.decode_thread, &inference_attributes, decodeEntry, &srv) != 0) {
            log(c.DS4_LOG_DEFAULT, "ds4-server: failed to start decode coordinator", .{});
            return 1;
        }
        decoder_started = true;
        for (slot_contexts, 0..) |*context, index| {
            context.* = .{ .slot = @ptrCast(&srv.slots[index]), .stop = stop };
            if (c.pthread_create(&srv.slot_threads[index], &inference_attributes, slotEntry, context) != 0) {
                log(c.DS4_LOG_DEFAULT, "ds4-server: failed to start session worker %d/%d", .{ @as(c_int, @intCast(index + 1)), srv.slot_count });
                joinWorkers(&srv, worker, started, decoder_started);
                return 1;
            }
            started += 1;
        }
    } else if (c.pthread_create(&worker, &inference_attributes, workerEntry, &worker_context) != 0) {
        log(c.DS4_LOG_DEFAULT, "ds4-server: failed to start worker", .{});
        return 1;
    }
    const fd = c.ds4_server_http_listen(cfg.host, cfg.port);
    if (fd < 0) {
        log(c.DS4_LOG_DEFAULT, "ds4-server: failed to listen on %s:%d: %s", .{ cfg.host, cfg.port, c.strerror(c.__error().*) });
        joinWorkers(&srv, worker, started, decoder_started);
        return 1;
    }
    listen_fd.* = fd;
    log(c.DS4_LOG_DEFAULT, "ds4-server: listening on http://%s:%d", .{ cfg.host, cfg.port });
    while (stop.* == 0) {
        const accepted = c.accept(fd, null, null);
        if (accepted < 0) {
            if (stop.* != 0) break;
            if (c.__error().* == c.EINTR) continue;
            log(c.DS4_LOG_DEFAULT, "ds4-server: accept failed: %s", .{c.strerror(c.__error().*)});
            continue;
        }
        if (stop.* != 0) {
            _ = c.close(accepted);
            break;
        }
        c.ds4_server_http_configure(accepted);
        const context = allocator.create(ClientContext) catch fatal();
        context.* = .{ .packet = .{ .srv = &srv, .fd = accepted }, .stop = stop };
        _ = c.pthread_mutex_lock(&srv.mu);
        srv.clients += 1;
        _ = c.pthread_mutex_unlock(&srv.mu);
        var thread: c.pthread_t = undefined;
        if (c.pthread_create(&thread, null, clientEntry, context) != 0) {
            _ = c.pthread_mutex_lock(&srv.mu);
            srv.clients -= 1;
            _ = c.pthread_cond_broadcast(&srv.clients_cv);
            _ = c.pthread_mutex_unlock(&srv.mu);
            allocator.destroy(context);
            _ = c.close(accepted);
            continue;
        }
        _ = c.pthread_detach(thread);
    }
    if (listen_fd.* >= 0) {
        _ = c.close(fd);
        listen_fd.* = -1;
    }
    log(c.DS4_LOG_DEFAULT, "ds4-server: shutdown requested, draining requests", .{});
    joinWorkers(&srv, worker, started, decoder_started);
    _ = c.pthread_mutex_lock(&srv.mu);
    while (srv.clients > 0) _ = c.pthread_cond_wait(&srv.clients_cv, &srv.mu);
    _ = c.pthread_mutex_unlock(&srv.mu);
    for (0..@intCast(srv.slot_count)) |index| {
        if (!srv.kv.enabled) break;
        const slot = &srv.slots[index];
        const tokens = c.ds4_session_tokens(slot.session);
        if (tokens == null or tokens.*.len < srv.kv.opt.min_tokens) continue;
        log(c.DS4_LOG_KVCACHE, "ds4-server: persisting resident KV cache before shutdown slot=%d tokens=%d", .{ @as(c_int, @intCast(index)), tokens.*.len });
        c.ds4_server_persist_store_current(&srv, @ptrCast(slot), "shutdown");
    }
    return 0;
}
