//! Eval startup and run selection. Core/hard questions are borrowed immutable
//! C data; all dynamic prompt/UI/session/transport owners retire in scope.
const std = @import("std");
const c = @import("eval_c");
const Options = @import("options.zig");
const Cases = @import("cases.zig");
const Prompts = @import("prompts.zig");
const Trace = @import("trace.zig");
const Terminal = @import("ui.zig");
const Runtime = @import("runtime.zig");
const allocator = std.heap.c_allocator;
fn logMemory(config: *const c.eval_config) void {
    const memory = c.ds4_context_memory_estimate_with_prefill_mode(config.backend, config.ctx_size, config.prefill_chunk, config.ssd_streaming);
    _ = c.fprintf(
        c.__stderrp,
        "ds4-eval: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
        @as(f64, @floatFromInt(memory.total_bytes)) / (1024.0 * 1024.0),
        config.ctx_size,
        c.ds4_backend_name(config.backend),
        memory.prefill_cap,
        memory.raw_cap,
        memory.comp_cap,
    );
}
const EngineResources = struct {
    engine: *c.ds4_engine,
    transport: ?*c.ds4_tp = null,
    session: ?*c.ds4_session = null,
    stop_transport: bool = false,
    fn deinit(self: *EngineResources) void {
        if (self.session) |session| c.ds4_session_free(session);
        if (self.stop_transport) _ = c.ds4_tp_send_stop(self.transport);
        // Engine close unbinds borrowed TP views; the transport outlives it.
        c.ds4_engine_close(self.engine);
        c.ds4_tp_free(self.transport);
    }
};
pub fn main(argc: c_int, argv: [*c][*c]u8) c_int {
    var config = Options.parse(argc, argv, c.DS4_EVAL_DEFAULT_BACKEND);
    if (config.self_test_extractors) return c.ds4_eval_run_extractor_self_tests();
    if (config.validate_cases) return Cases.validate(true);
    if (config.regrade_trace_path != null) return Trace.regrade(config.regrade_trace_path);
    var count: c_int = 0;
    const selected = Cases.select(&config, &count);
    if (selected == null) return 2;
    defer c.free(selected);
    const cases: []const c.eval_case = selected[0..@intCast(count)];
    if (config.list_cases) {
        Cases.list(cases);
        return 0;
    }
    const sequence = if (config.case_sequence != null) Runtime.sequence(config.case_sequence, count) orelse return 2 else null;
    defer if (sequence) |values| allocator.free(values);
    if (config.seed == 0) config.seed = @as(u64, @intCast(c.time(null))) ^ (@as(u64, @intCast(c.getpid())) << 32) ^ @as(u64, @intCast(c.clock()));
    const trace: ?*c.FILE = if (config.trace_path != null) c.fopen(config.trace_path, "w") else null;
    if (config.trace_path != null and trace == null) {
        _ = c.fprintf(c.__stderrp, "ds4-eval: cannot open trace '%s': %s\n", config.trace_path, c.strerror(c.__error().*));
        return 2;
    }
    defer if (trace) |file| {
        _ = c.fclose(file);
    };
    var options = std.mem.zeroes(c.ds4_engine_options);
    options.model_path = config.model_path;
    options.mtp_path = config.mtp_path;
    options.backend = config.backend;
    options.n_threads = config.threads;
    options.context_size = @max(config.ctx_size, 0);
    options.mtp_draft_tokens = 1;
    options.mtp_margin = 3.0;
    options.power_percent = config.power_percent;
    options.prefill_chunk = config.prefill_chunk;
    options.ssd_streaming_cache_experts = config.ssd_streaming_cache_experts;
    options.ssd_streaming_cache_bytes = config.ssd_streaming_cache_bytes;
    options.ssd_streaming_full_layers = config.ssd_streaming_full_layers;
    options.ssd_streaming_preload_experts = config.ssd_streaming_preload_experts;
    options.simulate_used_memory_bytes = config.simulate_used_memory_bytes;
    options.warm_weights = config.warm_weights;
    options.quality = config.quality;
    options.ssd_streaming = config.ssd_streaming;
    options.ssd_streaming_cold = config.ssd_streaming_cold;
    options.ssd_streaming_full_layers_set = config.ssd_streaming_full_layers_set;
    options.distributed = config.dist;
    options.tp = config.tp;
    var errors: [256]u8 = undefined;
    if (c.ds4_dist_prepare_engine_options(&config.dist, &options, &errors, errors.len) != 0) {
        _ = c.fprintf(c.__stderrp, "ds4-eval: %s\n", &errors);
        return 2;
    }
    if (c.ds4_tp_validate_engine_options(&options, &errors, errors.len) == 0) {
        _ = c.fprintf(c.__stderrp, "ds4-eval: %s\n", &errors);
        return 2;
    }
    var engine: ?*c.ds4_engine = null;
    if (c.ds4_engine_open(&engine, &options) != 0) return 1;
    var owners = EngineResources{ .engine = engine.? };
    defer owners.deinit();
    const max_generation = Prompts.maxBudget(&config, cases);
    var largest: Prompts.Largest = undefined;
    if (config.ctx_size <= 0) {
        const sized = Prompts.autoContext(engine.?, &config, cases, max_generation);
        config.ctx_size = sized.context;
        largest = sized.largest;
        _ = c.fprintf(
            c.__stderrp,
            "ds4-eval: context auto-sized to %d tokens (largest prompt=%d tokens, case=%d, generation budget=%d)\n",
            config.ctx_size,
            largest.tokens,
            largest.index + 1,
            max_generation,
        );
    } else {
        largest = Prompts.maxPrompt(engine.?, &config, cases, config.ctx_size);
        _ = c.fprintf(
            c.__stderrp,
            "ds4-eval: context set to %d tokens (largest prompt=%d tokens, case=%d, generation budget=%d)\n",
            config.ctx_size,
            largest.tokens,
            largest.index + 1,
            max_generation,
        );
        Prompts.warn(&config, largest, max_generation);
    }
    _ = c.fprintf(c.__stderrp, "ds4-eval: model shape %s\n", c.ds4_engine_model_name(engine.?));
    Prompts.warnThink(&config);
    Trace.header(trace, &config, c.ds4_engine_model_name(engine.?), count, largest.tokens, max_generation);
    logMemory(&config);
    if (config.tp.role == c.DS4_TP_LEADER) {
        var identity = std.mem.zeroes(c.ds4_tp_identity);
        identity.gguf_bytes = c.ds4_engine_model_bytes(engine.?);
        identity.model_id = @intCast(c.ds4_engine_model_id(engine.?));
        identity.n_layer = @intCast(c.ds4_engine_layer_count(engine.?));
        identity.n_embd = @intCast(c.ds4_engine_embd_dim(engine.?));
        identity.n_vocab = @intCast(c.ds4_engine_vocab_size(engine.?));
        identity.quant_bits = @intCast(c.ds4_engine_routed_quant_bits(engine.?));
        identity.ctx_size = @intCast(config.ctx_size);
        c.ds4_engine_tp_gate_schedule(engine.?, &identity.gate_slot_start, &identity.gate_slot_step, &identity.gates_per_token, &identity.gate_slot_mask);
        if (c.ds4_tp_create(&owners.transport, &config.tp, &identity, &errors, errors.len) == 0 or c.ds4_engine_tp_bind(engine.?, owners.transport, &errors, errors.len) == 0) {
            _ = c.fprintf(c.__stderrp, "ds4-eval: %s\n", &errors);
            c.ds4_tp_free(owners.transport);
            owners.transport = null;
            return 1;
        }
        owners.stop_transport = true;
    }
    if (c.ds4_session_create(&owners.session, engine.?, config.ctx_size) != 0) {
        _ = c.fprintf(c.__stderrp, "ds4-eval: failed to create session\n");
        return 1;
    }
    const session = owners.session.?;
    if (config.dist.role == c.DS4_DISTRIBUTED_COORDINATOR and !Runtime.waitRoute(session)) return 1;
    var ui: Terminal.Ui = .{};
    ui.startRun(cases, max_generation, !config.plain and c.isatty(c.STDOUT_FILENO) != 0);
    defer ui.deinit();
    var rng = config.seed;
    var failed_run = false;
    var next: c_int = 0;
    var sequence_position: usize = 0;
    while (next >= 0) {
        if (sequence) |values| {
            if (sequence_position >= values.len) break;
            next = values[sequence_position];
            sequence_position += 1;
            ui.selected_case = next;
            ui.selection_active = false;
        }
        ui.consume();
        _ = Runtime.waitPaused(&ui, "idle");
        if (ui.quit_requested) break;
        if (sequence == null and ui.requested_case >= 0) {
            next = ui.requested_case;
            ui.requested_case = -1;
            ui.selection_active = false;
            ui.selected_case = next;
        }
        const index: usize = @intCast(next);
        const budget = Prompts.budget(&config, &cases[index], false);
        var result = Runtime.runOne(engine.?, session, &config, &ui, trace, index, budget, &rng);
        if (result == .ok and ui.status[index] == c.EVAL_INCOMPLETE and config.retry_incomplete) {
            const retry = Prompts.budget(&config, &cases[index], true);
            _ = c.fprintf(c.__stderrp, "ds4-eval: retrying incomplete case %s/%s with %d tokens\n", cases[index].source, cases[index].id, retry);
            result = Runtime.runOne(engine.?, session, &config, &ui, trace, index, retry, &rng);
        }
        if (result == .failure) {
            failed_run = true;
            break;
        }
        if (result == .quit) break;
        // A completed case advances. Real q either returned quit above or is
        // consumed at the next idle boundary; do not keep a stale quit flag.
        ui.quit_requested = false;
        if (sequence != null) continue;
        if (ui.requested_case >= 0) {
            next = ui.requested_case;
            ui.requested_case = -1;
            ui.selection_active = false;
            ui.selected_case = next;
            continue;
        }
        next = Runtime.nextPending(&ui, next + 1);
        if (next >= 0) {
            ui.selected_case = next;
            ui.selection_active = false;
        }
    }
    var passed: c_int = 0;
    var failed: c_int = 0;
    var incomplete: c_int = 0;
    for (ui.status) |status| {
        if (status == c.EVAL_PASSED) passed += 1 else if (status == c.EVAL_FAILED) failed += 1 else if (status == c.EVAL_INCOMPLETE) incomplete += 1;
    }
    if (ui.active) Terminal.restoreTerminal();
    Runtime.report(&ui, passed, failed, incomplete);
    if (trace) |file| {
        _ = c.fprintf(file, "===== SUMMARY =====\npassed: %d\nfailed: %d\nincomplete: %d\ntotal: %d\n", passed, failed, incomplete, count);
        _ = c.fflush(file);
    }
    return @intFromBool(failed_run or failed != 0 or incomplete != 0);
}
