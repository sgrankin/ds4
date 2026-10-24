//! C session facade for the native graph owner. Algorithms and lifetime rules
//! live in deepseek41/; the actual graph record remains shared during migration.
const std = @import("std");
const c = @import("gguf_c");
const Config = @import("config.zig").ModelConfig;
const resources = @import("deepseek41/resources.zig");
const Layout = @import("deepseek41/layout.zig").Layout;
const Attention = @import("deepseek41/attention.zig").Attention;
const BatchAttention = @import("deepseek41/batch_attention.zig").BatchAttention;
const Executor = @import("deepseek41/executor.zig").Executor;
const Prefill = @import("deepseek41/prefill.zig").Prefill;
const SessionBatch = @import("deepseek41/session_batch.zig").SessionBatch;
const Policy = @import("deepseek41/prefill_policy.zig").Policy;

fn configuration(record: *const c.ds4_model_config) Config {
    return .{ .shape = record.shape, .compress_ratios = record.compress_ratios };
}

export fn ds4_native_graph_alloc41(record: *c.ds41_gpu_graph, model: *const c.ds4_model, weights: *const c.ds4_weights, config: *const c.ds4_model_config, path: [*:0]const u8, context: u32, chunk: u32, streaming: bool) bool {
    // Match C's failed-output state, including context admission failures.
    record.* = std.mem.zeroes(c.ds41_gpu_graph);
    record.table[0].fd = -1;
    record.table[1].fd = -1;
    var owner = resources.Graph.init(configuration(config), model, weights, path, context, chunk, streaming) catch return false;
    record.* = owner.take();
    return true;
}
export fn ds4_native_graph_free41(record: ?*c.ds41_gpu_graph) void {
    if (record) |graph| resources.freeRecord(graph);
}
export fn ds4_native_graph_reset41(record: *c.ds41_gpu_graph) void {
    resources.resetRecord(record);
}
export fn ds4_native_graph_workspace_grow41(record: *c.ds41_gpu_graph, config: *const c.ds4_model_config, capacity: u32) bool {
    const value = configuration(config);
    return resources.growRecord(record, &value, capacity);
}
export fn ds4_native_graph_bytes41(config: *const c.ds4_model_config, context: u32, chunk: u32) u64 {
    const value = configuration(config);
    const plan = Layout.init(&value, context, chunk);
    return plan.bytes();
}
export fn ds4_native_graph_memory41(config: *const c.ds4_model_config, context: u32, chunk: u32) c.ds4_context_memory {
    const value = configuration(config);
    const plan = Layout.init(&value, context, chunk);
    return plan.memory();
}

export fn ds4_native_graph_attention41(graph: *c.ds41_gpu_graph, model: *const c.ds4_model, layer: *const c.ds4_layer_weights, config: *const c.ds4_model_config, index: u32, projected: bool) bool {
    const value = configuration(config);
    const attention = Attention{ .config = &value, .model = model, .graph = graph };
    return attention.run(layer, index, projected);
}
export fn ds4_native_graph_attention_batch41(graph: *c.ds41_gpu_graph, model: *const c.ds4_model, layer: *const c.ds4_layer_weights, config: *const c.ds4_model_config, index: u32, count: u32, exact_prefill_rows: bool) bool {
    const value = configuration(config);
    const attention = BatchAttention{ .ops = .{ .config = &value, .model = model, .exact_prefill_rows = exact_prefill_rows }, .graph = graph };
    return attention.run(layer, index, count);
}

fn execution(graph: *c.ds41_gpu_graph, model: *const c.ds4_model, weights: *const c.ds4_weights, config: *const Config, exact: bool) Executor {
    return .{
        .ops = .{ .config = config, .model = model, .exact_prefill_rows = exact },
        .graph = graph,
        .weights = weights,
        .services = .{
            .trace = c.ds4_native_graph_trace_row,
            .observe_imatrix = c.ds4_native_executor_observe_imatrix,
        },
    };
}

export fn ds4_native_graph_step41(maybe_graph: ?*c.ds41_gpu_graph, model: *const c.ds4_model, weights: *const c.ds4_weights, config: *const c.ds4_model_config, token: c_int, output: [*c]f32, exact: bool) bool {
    const graph = maybe_graph orelse return false;
    const value = configuration(config);
    const executor = execution(graph, model, weights, &value, exact);
    return executor.step(token, output);
}
export fn ds4_native_graph_logits41(graph: *c.ds41_gpu_graph, model: *const c.ds4_model, weights: *const c.ds4_weights, config: *const c.ds4_model_config, output: [*c]f32, exact: bool) bool {
    const value = configuration(config);
    const executor = execution(graph, model, weights, &value, exact);
    return executor.logits(output);
}
export fn ds4_native_graph_prefill_count41(graph: *const c.ds41_gpu_graph, weights: *const c.ds4_weights, config: *const c.ds4_model_config, remaining: u32) u32 {
    const value = configuration(config);
    const policy = Policy{ .config = &value, .graph = graph, .weights = weights };
    return policy.prefillCount(remaining);
}
export fn ds4_native_graph_prefill41(graph: *c.ds41_gpu_graph, model: *const c.ds4_model, weights: *const c.ds4_weights, config: *const c.ds4_model_config, tokens: [*c]const c_int, count: u32, progress: c.ds4_session_progress_fn, progress_userdata: ?*anyopaque, total: c_int, cancel: c.ds4_session_cancel_fn, cancel_userdata: ?*anyopaque, encoder_only: bool, resume_encoder: bool, exact: bool) bool {
    if (tokens == null or count == 0) return false;
    const value = configuration(config);
    const prefill = Prefill{ .executor = execution(graph, model, weights, &value, exact) };
    const callbacks = @import("deepseek41/prefill.zig").Callbacks{ .progress = progress, .progress_userdata = progress_userdata, .total = total, .cancel = cancel, .cancel_userdata = cancel_userdata };
    return prefill.sweep(tokens[0..count], callbacks, encoder_only, resume_encoder);
}

fn borrowedGraphs(raw: [*c]const ?*c.ds41_gpu_graph, count: c_int, storage: *[c.DS4_TP_BATCH_MAX_ROWS]*c.ds41_gpu_graph) ?[]const *c.ds41_gpu_graph {
    if (raw == null or count < 2 or count > storage.len) return null;
    const length: usize = @intCast(count);
    for (storage[0..length], 0..) |*entry, index| entry.* = raw[index] orelse return null;
    return storage[0..length];
}
export fn ds4_native_graph_batch_workspace41(raw: [*c]const ?*c.ds41_gpu_graph, count: c_int) ?*c.ds41_gpu_graph {
    var storage: [c.DS4_TP_BATCH_MAX_ROWS]*c.ds41_gpu_graph = undefined;
    const graphs = borrowedGraphs(raw, count, &storage) orelse return null;
    return SessionBatch.workspace(graphs);
}
export fn ds4_native_graph_step_batch41(raw: [*c]const ?*c.ds41_gpu_graph, tokens: [*c]const c_int, count: c_int, prefill_rows: u32, model: *const c.ds4_model, weights: *const c.ds4_weights, config: *const c.ds4_model_config, exact: bool) bool {
    if (tokens == null) return false;
    var storage: [c.DS4_TP_BATCH_MAX_ROWS]*c.ds41_gpu_graph = undefined;
    const graphs = borrowedGraphs(raw, count, &storage) orelse return false;
    const graph = SessionBatch.workspace(graphs) orelse return false;
    const value = configuration(config);
    const batch = SessionBatch{ .executor = execution(graph, model, weights, &value, exact) };
    return batch.run(graphs, tokens[0..graphs.len], prefill_rows);
}
export fn ds4_native_graph_short_prefill41(graph: *c.ds41_gpu_graph, model: *const c.ds4_model, weights: *const c.ds4_weights, config: *const c.ds4_model_config, tokens: [*c]const c_int, count: u32, exact: bool) bool {
    if (tokens == null) return false;
    const value = configuration(config);
    const batch = SessionBatch{ .executor = execution(graph, model, weights, &value, exact) };
    return batch.shortPrefill(tokens[0..count]);
}

const state_payload = @import("deepseek41/state_payload.zig");

export fn ds4_native_graph_payload_body_bytes41(graph: *c.ds41_gpu_graph, raw: *const c.ds4_model_config, position: u32) u64 {
    const config = configuration(raw);
    const payload = state_payload.Payload{ .graph = graph, .config = &config };
    return payload.bodyBytes(position);
}

fn errorBuffer(raw: [*c]u8, count: usize) []u8 {
    return if (raw == null or count == 0) &.{} else raw[0..count];
}

export fn ds4_native_graph_save_payload41(graph: *c.ds41_gpu_graph, raw: *const c.ds4_model_config, stream: *c.FILE, tokens: *const c.ds4_tokens, logits: [*c]const f32, err: [*c]u8, errlen: usize) c_int {
    const config = configuration(raw);
    const payload = state_payload.Payload{ .graph = graph, .config = &config };
    if (tokens.len < 0 or (tokens.len != 0 and tokens.v == null) or logits == null) return 1;
    const count: usize = @intCast(tokens.len);
    const items: []const c_int = if (count == 0) &.{} else tokens.v[0..count];
    payload.save(stream, items, logits[0..config.shape.n_vocab], errorBuffer(err, errlen)) catch return 1;
    return 0;
}

export fn ds4_native_graph_load_payload41(graph: *c.ds41_gpu_graph, raw: *const c.ds4_model_config, stream: *c.FILE, header: *const state_payload.Header, remaining: u64, logits: [*c]f32, out: *c.ds4_tokens, mutated: *bool, err: [*c]u8, errlen: usize) c_int {
    const config = configuration(raw);
    const payload = state_payload.Payload{ .graph = graph, .config = &config };
    mutated.* = false;
    out.* = .{ .v = null, .len = 0, .cap = 0 };
    if (logits == null) return 1;
    var tokens = payload.load(stream, header, remaining, logits[0..config.shape.n_vocab], mutated, errorBuffer(err, errlen)) catch return 1;
    out.* = tokens.take();
    return 0;
}

const suffix_state = @import("deepseek41/suffix.zig");

export fn ds4_native_graph_prerope_bytes41(end: u32) u64 {
    return suffix_state.archiveBytes(end);
}
export fn ds4_native_graph_prerope_enable41(graph: *c.ds41_gpu_graph, raw: *const c.ds4_model_config) bool {
    const config = configuration(raw);
    return (suffix_state.Suffix{ .graph = graph, .config = &config }).enableArchive();
}
export fn ds4_native_graph_prerope_capture41(graph: *c.ds41_gpu_graph, raw: *const c.ds4_model_config, cache: *c.ds4_suffix_cache) bool {
    const config = configuration(raw);
    return (suffix_state.Suffix{ .graph = graph, .config = &config }).captureArchive(cache);
}
export fn ds4_native_graph_suffix_validate_prefix41(graph: *c.ds41_gpu_graph, raw: *const c.ds4_model_config, cache: *const c.ds4_suffix_cache, end: *u32, err: [*c]u8, errlen: usize) c_int {
    const config = configuration(raw);
    const suffix = suffix_state.Suffix{ .graph = graph, .config = &config };
    end.* = suffix.validatePrefix(cache, errorBuffer(err, errlen)) catch return 1;
    return 0;
}
export fn ds4_native_graph_prerope_restore41(graph: *c.ds41_gpu_graph, raw: *const c.ds4_model_config, cache: *const c.ds4_suffix_cache, end: u32, callback: c.ds4_session_cancel_fn, userdata: ?*anyopaque) bool {
    const config = configuration(raw);
    const suffix = suffix_state.Suffix{ .graph = graph, .config = &config };
    suffix.restoreArchive(cache, end, .{ .callback = callback, .userdata = userdata }) catch return false;
    return true;
}
export fn ds4_native_graph_graft_suffix41(graph: *c.ds41_gpu_graph, raw: *const c.ds4_model_config, cache: *const c.ds4_suffix_cache, prompt: *const c.ds4_tokens, source_start: u32, start: u32, out: *c.ds4_suffix_relocation_stats, mutated: *bool, callback: c.ds4_session_cancel_fn, userdata: ?*anyopaque, err: [*c]u8, errlen: usize) c_int {
    mutated.* = false;
    if (prompt.len < 0 or (prompt.len != 0 and prompt.v == null)) return 1;
    const tokens: []const c_int = if (prompt.len == 0) &.{} else prompt.v[0..@as(usize, @intCast(prompt.len))];
    const config = configuration(raw);
    const suffix = suffix_state.Suffix{ .graph = graph, .config = &config };
    const result = suffix.graft(cache, tokens, source_start, start, .{ .callback = callback, .userdata = userdata }, mutated, errorBuffer(err, errlen)) catch return 1;
    out.* = result.relocation;
    return 0;
}

const dspark = @import("deepseek41/dspark.zig");

export fn ds4_native_dspark_alloc41(raw: *const c.ds4_model_config, weights: ?*const c.ds4_dspark_weights) ?*c.ds41_dspark {
    const config = configuration(raw);
    return dspark.allocate(&config, weights);
}
export fn ds4_native_dspark_free41(state: ?*c.ds41_dspark) void {
    dspark.free(state);
}
export fn ds4_native_dspark_reset41(state: ?*c.ds41_dspark) void {
    dspark.reset(state);
}
export fn ds4_native_dspark_capture41(raw: *const c.ds4_model_config, state: ?*c.ds41_dspark, layer: u32, start: u32, residual: ?*const c.ds4_gpu_tensor, count: u32) bool {
    const config = configuration(raw);
    return dspark.capture(&config, state, layer, start, residual, count);
}
export fn ds4_native_dspark_seed41(state: ?*c.ds41_dspark, model: *const c.ds4_model, raw: *const c.ds4_model_config, position: u32, exact: bool) bool {
    const config = configuration(raw);
    const drafter = dspark.Drafter{ .state = state orelse return false, .ops = .{ .config = &config, .model = model, .exact_prefill_rows = exact } };
    return drafter.seed(position);
}
export fn ds4_native_dspark_stage41(state: ?*c.ds41_dspark, model: *const c.ds4_model, raw: *const c.ds4_model_config, index: u32, position: u32, exact: bool) bool {
    const config = configuration(raw);
    const drafter = dspark.Drafter{ .state = state orelse return false, .ops = .{ .config = &config, .model = model, .exact_prefill_rows = exact } };
    return drafter.stage(index, position);
}
export fn ds4_native_dspark_forward41(state: ?*c.ds41_dspark, target: *const c.ds4_model, weights: *const c.ds4_weights, model: *const c.ds4_model, raw: *const c.ds4_model_config, position: u32, token: c_int, graph: ?*c.ds41_gpu_graph, exact: bool) bool {
    const config = configuration(raw);
    const drafter = dspark.Drafter{ .state = state orelse return false, .ops = .{ .config = &config, .model = model, .exact_prefill_rows = exact } };
    return drafter.forward(target, weights, position, token, graph);
}
export fn ds4_native_dspark_markov41(state: ?*c.ds41_dspark, model: *const c.ds4_model, raw: *const c.ds4_model_config, row: u32, previous: c_int, confidence: ?*f32, exact: bool) bool {
    const config = configuration(raw);
    const drafter = dspark.Drafter{ .state = state orelse return false, .ops = .{ .config = &config, .model = model, .exact_prefill_rows = exact } };
    return drafter.markov(row, previous, confidence orelse return false);
}
export fn ds4_native_dspark_verify41(state: ?*c.ds41_dspark, graph: *c.ds41_gpu_graph, model: *const c.ds4_model, weights: *const c.ds4_weights, raw: *const c.ds4_model_config, transport: ?*c.ds4_tp, tokens: [*c]const c_int, count: u32, exact: bool) bool {
    const record = state orelse return false;
    if (tokens == null or count < 2 or count > c.DS41_VERIFY_ROWS) return false;
    const config = configuration(raw);
    return dspark.verify(record, execution(graph, model, weights, &config, exact), transport, tokens[0..count]);
}
export fn ds4_native_dspark_commit41(state: ?*c.ds41_dspark, graph: *c.ds41_gpu_graph, raw: *const c.ds4_model_config, keep: u32, logits: [*c]f32) bool {
    const record = state orelse return false;
    if (logits == null) return false;
    const config = configuration(raw);
    return dspark.commit(&config, record, graph, keep, @ptrCast(logits));
}

const SpeculativeCycle = @import("deepseek41/speculative.zig").Cycle;

export fn ds4_native_dspark_cycle_draft41(args: *const c.ds4_dspark_cycle_args, graph: *c.ds41_gpu_graph, model: *const c.ds4_model, weights: *const c.ds4_weights, draft: *const c.ds4_model, raw: *const c.ds4_model_config, token: c_int, exact: bool) bool {
    const config = configuration(raw);
    const cycle = SpeculativeCycle.init(execution(graph, model, weights, &config, exact), args);
    return cycle.draft(draft, token);
}
export fn ds4_native_dspark_cycle_verify41(args: *const c.ds4_dspark_cycle_args, graph: *c.ds41_gpu_graph, model: *const c.ds4_model, weights: *const c.ds4_weights, raw: *const c.ds4_model_config, tokens: [*c]const c_int, count: u32, exact: bool) bool {
    if (tokens == null) return false;
    const config = configuration(raw);
    const cycle = SpeculativeCycle.init(execution(graph, model, weights, &config, exact), args);
    return cycle.verify(tokens[0..count]);
}
export fn ds4_native_dspark_cycle_commit41(args: *const c.ds4_dspark_cycle_args, graph: *c.ds41_gpu_graph, model: *const c.ds4_model, weights: *const c.ds4_weights, raw: *const c.ds4_model_config, tokens: [*c]const c_int, count: u32, keep: u32, exact: bool) bool {
    if (tokens == null) return false;
    const config = configuration(raw);
    const cycle = SpeculativeCycle.init(execution(graph, model, weights, &config, exact), args);
    return cycle.commit(tokens[0..count], keep);
}
// Compatibility for the existing low-level C Cycle fixture/API. The supported
// public Session owner uses direct native Coordinator/Selection composition.
const CycleLibraryContext = struct {
    session: ?*c.ds4_session,
    engine: ?*c.ds4_engine,
    pub fn eval(self: *const CycleLibraryContext, token: c_int, errors: []u8) c_int {
        return c.ds4_session_eval(self.session, token, errors.ptr, errors.len);
    }
    pub fn stop(self: *const CycleLibraryContext, token: c_int, mode: c.ds4_think_mode) bool {
        return c.ds4_token_is_stop_for_think_mode(self.engine, token, mode);
    }
};

export fn ds4_native_dspark_cycle_run41(args: *const c.ds4_dspark_cycle_args, graph: *c.ds41_gpu_graph, model: *const c.ds4_model, weights: *const c.ds4_weights, draft: *const c.ds4_model, raw: *const c.ds4_model_config, first_token: c_int, cap: c_int, eos: c_int, ignore_eos: bool, think_mode: c.ds4_think_mode, temperature: f32, top_k: c_int, top_p: f32, min_p: f32, rng: ?*u64, accepted: [*c]c_int, err: [*c]u8, errlen: usize, exact: bool) c_int {
    if (accepted == null or cap < 1) return -1;
    const config = configuration(raw);
    const cycle = SpeculativeCycle.init(execution(graph, model, weights, &config, exact), args);
    const capacity: usize = @intCast(@min(cap, c.DS41_VERIFY_ROWS));
    const context = CycleLibraryContext{ .session = args.session, .engine = args.engine };
    return cycle.run(&context, draft, first_token, cap, eos, ignore_eos, think_mode, temperature, top_k, top_p, min_p, rng, accepted[0..capacity], errorBuffer(err, errlen));
}

// Append to the ONE existing src/engine/graph.zig export facade. The pure owners
// below do not import this file, so no second graph export object is linked.
const SessionRuntime = @import("deepseek41/session.zig").Runtime;

fn sessionRuntime(state: *c.ds4_session_frontier, graph: *c.ds41_gpu_graph, model: *const c.ds4_model, weights: *const c.ds4_weights, config: *const Config, memory: *c.ds4_engine_memory, inputs: *const c.ds4_engine_memory_inputs, ready: bool, exact: bool) SessionRuntime {
    return .{
        .state = state,
        .prefill = .{ .executor = execution(graph, model, weights, config, exact) },
        .memory = .{ .state = memory, .config = config, .weights = weights, .inputs = inputs.* },
        .ready = ready,
    };
}
export fn ds4_native_session_sync41(state: *c.ds4_session_frontier, graph: *c.ds41_gpu_graph, model: *const c.ds4_model, weights: *const c.ds4_weights, raw: *const c.ds4_model_config, memory: *c.ds4_engine_memory, inputs: *const c.ds4_engine_memory_inputs, ready: bool, prompt: ?*const c.ds4_tokens, progress: c.ds4_session_progress_fn, progress_ud: ?*anyopaque, cancel: c.ds4_session_cancel_fn, cancel_ud: ?*anyopaque, exact: bool, errors: [*c]u8, error_size: usize) c_int {
    const config = configuration(raw);
    const runtime = sessionRuntime(state, graph, model, weights, &config, memory, inputs, ready, exact);
    const callbacks = @import("deepseek41/prefill.zig").Callbacks{
        .progress = progress,
        .progress_userdata = progress_ud,
        .cancel = cancel,
        .cancel_userdata = cancel_ud,
    };
    return runtime.sync(prompt, callbacks, errorBuffer(errors, error_size));
}
export fn ds4_native_session_eval41(state: *c.ds4_session_frontier, graph: *c.ds41_gpu_graph, model: *const c.ds4_model, weights: *const c.ds4_weights, raw: *const c.ds4_model_config, memory: *c.ds4_engine_memory, inputs: *const c.ds4_engine_memory_inputs, ready: bool, token: c_int, exact: bool, errors: [*c]u8, error_size: usize) c_int {
    const config = configuration(raw);
    const runtime = sessionRuntime(state, graph, model, weights, &config, memory, inputs, ready, exact);
    return runtime.eval(token, errorBuffer(errors, error_size));
}
export fn ds4_native_session_invalidate41(state: *c.ds4_session_frontier, graph: ?*c.ds41_gpu_graph) void {
    @import("deepseek41/session.zig").invalidate(state, graph);
}
// Existing C fixture helpers delegate these same whole lease actions.
export fn ds4_native_encoder_acquire41(lease: *c.ds41_encoder_residency, graph: *c.ds41_gpu_graph, model: *const c.ds4_model, weights: *const c.ds4_weights, raw: *const c.ds4_model_config, remaining: u32, cancel: c.ds4_session_cancel_fn, userdata: ?*anyopaque) void {
    const config = configuration(raw);
    @import("deepseek41/encoder_residency.zig").acquire(lease, graph, model, weights, &config, remaining, cancel, userdata);
}
export fn ds4_native_encoder_release41(lease: *c.ds41_encoder_residency, graph: *c.ds41_gpu_graph, model: *const c.ds4_model) void {
    @import("deepseek41/encoder_residency.zig").release(lease, graph, model);
}

// Both the production Engine and the existing simulated-host fixture use
// these same complete admission actions over the actual budget record.
export fn ds4_native_memory_admit_for_host41(state: *c.ds4_engine_memory, weights: *const c.ds4_weights, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, graph_bytes: u64, fit_cache: bool, host: u64, recommended: u64) bool {
    const config = configuration(raw);
    const memory = @import("deepseek41/memory.zig").Memory{ .state = state, .config = &config, .weights = weights, .inputs = inputs.* };
    return memory.admitForHost(graph_bytes, fit_cache, host, recommended);
}
export fn ds4_native_memory_admit41(state: *c.ds4_engine_memory, weights: *const c.ds4_weights, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, graph_bytes: u64, fit_cache: bool) bool {
    const config = configuration(raw);
    const memory = @import("deepseek41/memory.zig").Memory{ .state = state, .config = &config, .weights = weights, .inputs = inputs.* };
    return memory.admit(graph_bytes, fit_cache);
}

// Whole checkpoint actions, appended to the ONE graph export facade.
const CheckpointOwner = @import("deepseek41/checkpoint.zig");
fn checkpoint(state: *c.ds4_session_frontier, graph: *c.ds41_gpu_graph, config: *const Config, ready: bool) CheckpointOwner.Checkpoint {
    return .{ .frontier = state, .graph = graph, .config = config, .ready = ready };
}
export fn ds4_native_checkpoint_bytes41(state: *c.ds4_session_frontier, graph: *c.ds41_gpu_graph, raw: *const c.ds4_model_config, ready: bool) u64 {
    const config = configuration(raw);
    const owner = checkpoint(state, graph, &config, ready);
    return owner.canonicalBytes();
}
export fn ds4_native_checkpoint_save_payload41(state: *c.ds4_session_frontier, graph: *c.ds41_gpu_graph, raw: *const c.ds4_model_config, ready: bool, stream: ?*c.FILE, errors: [*c]u8, error_size: usize) c_int {
    const config = configuration(raw);
    const owner = checkpoint(state, graph, &config, ready);
    return owner.saveCanonical(stream, errorBuffer(errors, error_size));
}
export fn ds4_native_checkpoint_load_payload41(state: *c.ds4_session_frontier, graph: *c.ds41_gpu_graph, raw: *const c.ds4_model_config, ready: bool, stream: ?*c.FILE, bytes: u64, errors: [*c]u8, error_size: usize) c_int {
    const config = configuration(raw);
    const owner = checkpoint(state, graph, &config, ready);
    return owner.loadCanonical(stream, bytes, errorBuffer(errors, error_size));
}
export fn ds4_native_checkpoint_replay_bytes41(state: *c.ds4_session_frontier, graph: *c.ds41_gpu_graph, raw: *const c.ds4_model_config, ready: bool) u64 {
    const config = configuration(raw);
    const owner = checkpoint(state, graph, &config, ready);
    return owner.replayBytes();
}
export fn ds4_native_checkpoint_save_replay41(state: *c.ds4_session_frontier, graph: *c.ds41_gpu_graph, raw: *const c.ds4_model_config, ready: bool, snapshot: ?*c.ds4_session_snapshot, errors: [*c]u8, error_size: usize) c_int {
    const config = configuration(raw);
    const owner = checkpoint(state, graph, &config, ready);
    return owner.saveReplay(snapshot, errorBuffer(errors, error_size));
}
export fn ds4_native_checkpoint_load_replay41(state: *c.ds4_session_frontier, graph: *c.ds41_gpu_graph, raw: *const c.ds4_model_config, ready: bool, snapshot: ?*const c.ds4_session_snapshot, invalidate: *bool, errors: [*c]u8, error_size: usize) c_int {
    const config = configuration(raw);
    const owner = checkpoint(state, graph, &config, ready);
    return owner.loadReplay(snapshot, invalidate, errorBuffer(errors, error_size));
}
// Complete Session resource lifetime; Session shell/transport registration stay C.
const SessionResources = @import("deepseek41/session_resources.zig");
export fn ds4_native_session_resources_init41(device: *c.ds4_deepseek41_session_device, frontier: *c.ds4_session_frontier, memory_state: *c.ds4_engine_memory, inputs: *const c.ds4_engine_memory_inputs, model: *const c.ds4_model, weights: *const c.ds4_weights, raw: *const c.ds4_model_config, path: [*:0]const u8, context: u32, chunk: u32, quality: bool, streaming: bool, drafter: ?*const c.ds4_dspark_weights, transport: *const c.ds4_engine_tp_state) bool {
    const config = configuration(raw);
    const owner = SessionResources.SessionResources{
        .device = device,
        .frontier = frontier,
        .memory = .{ .state = memory_state, .config = &config, .weights = weights, .inputs = inputs.* },
        .model = model,
        .path = path,
        .transport = transport,
    };
    return owner.init(.{ .context = context, .chunk = chunk, .quality = quality, .streaming = streaming, .drafter = drafter });
}
export fn ds4_native_session_resources_free41(device: *c.ds4_deepseek41_session_device, frontier: *c.ds4_session_frontier, memory_state: *c.ds4_engine_memory) void {
    SessionResources.deinit(device, frontier, memory_state);
}

// Complete V4.1 Metal startup cache policies. No Engine layout crosses.
export fn ds4_native_memory_cap_requested41(state: *c.ds4_engine_memory, weights: *const c.ds4_weights, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, context: c_int, chunk: u32) void {
    const config = configuration(raw);
    const memory = @import("deepseek41/memory.zig").Memory{ .state = state, .config = &config, .weights = weights, .inputs = inputs.* };
    memory.capRequestedCache(context, chunk);
}
export fn ds4_native_memory_automatic_cache41(state: *c.ds4_engine_memory, weights: *const c.ds4_weights, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, context: c_int, chunk: u32) bool {
    const config = configuration(raw);
    const memory = @import("deepseek41/memory.zig").Memory{ .state = state, .config = &config, .weights = weights, .inputs = inputs.* };
    return memory.automaticCache(context, chunk);
}
export fn ds4_native_memory_reserve_prefill41(state: *c.ds4_engine_memory, weights: *const c.ds4_weights, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, full_layers_set: bool) bool {
    const config = configuration(raw);
    const memory = @import("deepseek41/memory.zig").Memory{ .state = state, .config = &config, .weights = weights, .inputs = inputs.* };
    return memory.reservePrefill(full_layers_set);
}

// Target views/locks/cache policy and the corresponding temporary span owner.
export fn ds4_native_target_residency_install41(state: *c.ds4_engine_memory, weights: *const c.ds4_weights, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, model: *const c.ds4_model, placement: *const c.ds4_model_residency_inputs, output: *c.ds4_model_map_selection) bool {
    const config = configuration(raw);
    const owner = @import("deepseek41/target_residency.zig").TargetResidency{
        .memory = .{ .state = state, .config = &config, .weights = weights, .inputs = inputs.* },
        .model = model,
        .placement = placement.*,
    };
    return owner.install(output);
}
export fn ds4_native_target_selection_free41(selection: *c.ds4_model_map_selection) void {
    @import("deepseek41/target_residency.zig").freeSelection(selection);
}

// Append to the ONE existing engine graph facade. Same generated gguf_c types.
const core_owner = @import("core.zig");

export fn ds4_native_core_finish41(record: *c.ds4_engine_core, config: *const c.ds4_model_config, options: *const c.ds4_engine_options, shard_model_bytes: *u64) bool {
    var owner = core_owner.Core{
        .state = record,
        .config = configuration(config),
    };
    return owner.finishOpen(options, shard_model_bytes);
}
export fn ds4_native_core_free41(record: *c.ds4_engine_core) void {
    // Config is not read during retirement. Do not reconstruct it from process
    // globals after another family has been selected by a diagnostic fixture.
    core_owner.freeRecord(record);
}

const RouteServices = @import("deepseek41/route_services.zig");
export fn ds4_resident_route_metrics_read(output: ?*c.ds4_resident_route_metrics) void {
    RouteServices.metricsRead(output);
}

// Append to the ONE existing graph facade. Runtime/execution constructors and
// configuration() are reused; pure owner imports instantiate no C exports.
const SessionCoordinator = @import("deepseek41/coordinator.zig");
var prefill_logit_dumped = false;
export fn ds4_native_session_prefill_logit_dump(logits: [*c]const f32, vocabulary: u32) void {
    SessionCoordinator.dumpPrefillLogits(&prefill_logit_dumped, logits, vocabulary);
}
fn coordinator(state: *c.ds4_session_frontier, device: *c.ds4_deepseek41_session_device, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, control: *c.ds4_session_control, config: *const Config, inputs: *const c.ds4_engine_memory_inputs, exact: bool) SessionCoordinator.Coordinator {
    return .{
        .runtime = sessionRuntime(state, &device.graph, &core.model, &core.weights, config, &core.memory, inputs, device.ready, exact),
        .core = core,
        .transport = transport,
        .control = control,
        .dump_recorded = &prefill_logit_dumped,
    };
}
export fn ds4_native_session_request_sync41(state: *c.ds4_session_frontier, device: *c.ds4_deepseek41_session_device, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, control: *c.ds4_session_control, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, prompt: ?*const c.ds4_tokens, exact: bool, errors: [*c]u8, capacity: usize) c_int {
    const config = configuration(raw);
    const owner = coordinator(state, device, core, transport, control, &config, inputs, exact);
    return owner.sync(prompt, errorBuffer(errors, capacity));
}
export fn ds4_native_session_multimodal_sync41(state: *c.ds4_session_frontier, device: *c.ds4_deepseek41_session_device, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, control: *c.ds4_session_control, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, prompt: ?*const c.ds4_tokens, images: [*c]const c.ds4_vision_span, count: usize, exact: bool, errors: [*c]u8, capacity: usize) c_int {
    const config = configuration(raw);
    const owner = coordinator(state, device, core, transport, control, &config, inputs, exact);
    return owner.syncMultimodal(prompt, images, count, errorBuffer(errors, capacity));
}

// Append after the Coordinator snippet in the ONE graph facade. Every action
// retains its local Config and Coordinator until the borrowed transaction ends.
const SuffixTransaction = @import("deepseek41/suffix_transaction.zig");
export fn ds4_native_session_suffix_free(cache: ?*c.ds4_suffix_cache) void {
    SuffixTransaction.freeCache(cache);
}
export fn ds4_native_session_suffix_enable41(state: *c.ds4_session_frontier, device: *c.ds4_deepseek41_session_device, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, control: *c.ds4_session_control, engine: ?*c.ds4_engine, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, exact: bool, errors: [*c]u8, capacity: usize) c_int {
    const config = configuration(raw);
    const session = coordinator(state, device, core, transport, control, &config, inputs, exact);
    const owner = SuffixTransaction.Transaction{ .session = &session, .engine_identity = engine };
    return owner.enableArchive(errorBuffer(errors, capacity));
}
export fn ds4_native_session_suffix_archive_bytes41(state: *c.ds4_session_frontier, device: *c.ds4_deepseek41_session_device, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, control: *c.ds4_session_control, engine: ?*c.ds4_engine, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, exact: bool) u64 {
    const config = configuration(raw);
    const session = coordinator(state, device, core, transport, control, &config, inputs, exact);
    const owner = SuffixTransaction.Transaction{ .session = &session, .engine_identity = engine };
    return owner.archiveBytes();
}
export fn ds4_native_session_suffix_bytes41(state: *c.ds4_session_frontier, device: *c.ds4_deepseek41_session_device, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, control: *c.ds4_session_control, engine: ?*c.ds4_engine, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, exact: bool) u64 {
    const config = configuration(raw);
    const session = coordinator(state, device, core, transport, control, &config, inputs, exact);
    const owner = SuffixTransaction.Transaction{ .session = &session, .engine_identity = engine };
    return owner.cacheBytes();
}
export fn ds4_native_session_suffix_capture41(state: *c.ds4_session_frontier, device: *c.ds4_deepseek41_session_device, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, control: *c.ds4_session_control, engine: ?*c.ds4_engine, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, exact: bool, cache: ?*c.ds4_suffix_cache, errors: [*c]u8, capacity: usize) c_int {
    const config = configuration(raw);
    const session = coordinator(state, device, core, transport, control, &config, inputs, exact);
    const owner = SuffixTransaction.Transaction{ .session = &session, .engine_identity = engine };
    return owner.capture(cache, errorBuffer(errors, capacity));
}
export fn ds4_native_session_suffix_load_prefix41(state: *c.ds4_session_frontier, device: *c.ds4_deepseek41_session_device, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, control: *c.ds4_session_control, engine: ?*c.ds4_engine, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, exact: bool, cache: ?*const c.ds4_suffix_cache, errors: [*c]u8, capacity: usize) c_int {
    const config = configuration(raw);
    const session = coordinator(state, device, core, transport, control, &config, inputs, exact);
    const owner = SuffixTransaction.Transaction{ .session = &session, .engine_identity = engine };
    return owner.loadPrefix(cache, errorBuffer(errors, capacity));
}
export fn ds4_native_session_suffix_diagnostic41(state: *c.ds4_session_frontier, device: *c.ds4_deepseek41_session_device, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, control: *c.ds4_session_control, engine: ?*c.ds4_engine, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, exact: bool, snapshot: ?*c.ds4_session_snapshot, errors: [*c]u8, capacity: usize) c_int {
    const config = configuration(raw);
    const session = coordinator(state, device, core, transport, control, &config, inputs, exact);
    const owner = SuffixTransaction.Transaction{ .session = &session, .engine_identity = engine };
    return owner.diagnostic(snapshot, errorBuffer(errors, capacity));
}
export fn ds4_native_session_suffix_graft41(state: *c.ds4_session_frontier, device: *c.ds4_deepseek41_session_device, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, control: *c.ds4_session_control, engine: ?*c.ds4_engine, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, exact: bool, cache: ?*const c.ds4_suffix_cache, prompt: ?*const c.ds4_tokens, source_start: u32, start: u32, statistics: ?*c.ds4_suffix_reuse_stats, relocation: ?*c.ds4_suffix_relocation_stats, errors: [*c]u8, capacity: usize) c_int {
    const config = configuration(raw);
    const session = coordinator(state, device, core, transport, control, &config, inputs, exact);
    const owner = SuffixTransaction.Transaction{ .session = &session, .engine_identity = engine };
    return owner.graft(cache, prompt, source_start, start, statistics, relocation, errorBuffer(errors, capacity));
}

// Append to the ONE graph facade after Coordinator. All rows share the same
// local immutable Config/inputs; they borrow their actual owners synchronously.
const BatchTransaction = @import("deepseek41/batch_transaction.zig");
export fn ds4_native_session_request_eval41(state: *c.ds4_session_frontier, device: *c.ds4_deepseek41_session_device, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, control: *c.ds4_session_control, identity: ?*c.ds4_session, distributed: ?*c.ds4_dist_session, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, token: c_int, exact: bool, errors: [*c]u8, capacity: usize) c_int {
    const config = configuration(raw);
    const owner = coordinator(state, device, core, transport, control, &config, inputs, exact);
    return owner.eval(identity, distributed, token, errorBuffer(errors, capacity));
}
export fn ds4_native_session_batch41(state: *c.ds4_session_frontier, device: *c.ds4_deepseek41_session_device, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, control: *c.ds4_session_control, engine: ?*c.ds4_engine, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, rows: [*c]const c.ds4_session_batch_view, count: c_int, prefill: ?*const c.ds4_session_batch_view, prompt: ?*const c.ds4_tokens, exact: bool, native_batch: *bool, mixed_batch: *bool, errors: [*c]u8, capacity: usize) c_int {
    const config = configuration(raw);
    const owner = BatchTransaction.Batch{
        .session = coordinator(state, device, core, transport, control, &config, inputs, exact),
        .engine_identity = engine,
    };
    const views: []const c.ds4_session_batch_view = if (rows == null or count <= 0) &.{} else rows[0..@intCast(count)];
    const result = if (prefill) |view|
        owner.mixed(views, view.*, prompt, errorBuffer(errors, capacity))
    else
        owner.decode(views, errorBuffer(errors, capacity));
    // Preserve the existing C test observations without a callback into policy.
    native_batch.* = result.native_batch;
    mixed_batch.* = result.mixed_batch;
    return result.status;
}

// Append to the existing ONE graph facade AFTER the Batch snippet.
const SessionSpeculation = @import("deepseek41/session_speculation.zig");
fn sessionSpeculation(view: *const c.ds4_session_batch_view, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, config: *const Config, inputs: *const c.ds4_engine_memory_inputs, verify_transport: ?*c.ds4_tp, exact: bool) SessionSpeculation.Session {
    return .{
        .coordinator = coordinator(@ptrCast(view.frontier), @ptrCast(view.device), core, transport, @ptrCast(view.control), config, inputs, exact),
        .identity = view.session,
        .distributed = view.distributed,
        .statistics = @ptrCast(view.statistics),
        .verify_transport = verify_transport,
    };
}
export fn ds4_native_session_speculate_argmax41(view: *const c.ds4_session_batch_view, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, verify_transport: ?*c.ds4_tp, first: c_int, maximum: c_int, eos: c_int, ignore_eos: bool, mode: c.ds4_think_mode, accepted: [*c]c_int, capacity: c_int, exact: bool, errors: [*c]u8, error_capacity: usize) c_int {
    const config = configuration(raw);
    const owner = sessionSpeculation(view, core, transport, &config, inputs, verify_transport, exact);
    return owner.argmax(first, maximum, eos, ignore_eos, mode, accepted, capacity, errorBuffer(errors, error_capacity));
}
export fn ds4_native_session_speculate_sample41(view: *const c.ds4_session_batch_view, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, verify_transport: ?*c.ds4_tp, first: c_int, maximum: c_int, eos: c_int, temperature: f32, top_k: c_int, top_p: f32, min_p: f32, rng: ?*u64, accepted: [*c]c_int, capacity: c_int, exact: bool, errors: [*c]u8, error_capacity: usize) c_int {
    const config = configuration(raw);
    const owner = sessionSpeculation(view, core, transport, &config, inputs, verify_transport, exact);
    return owner.sample(first, maximum, eos, temperature, top_k, top_p, min_p, rng, accepted, capacity, errorBuffer(errors, error_capacity));
}
export fn ds4_native_session_restore_prefix41(view: *const c.ds4_session_batch_view, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, verify_transport: ?*c.ds4_tp, position: c_int, exact: bool) c_int {
    const config = configuration(raw);
    const owner = sessionSpeculation(view, core, transport, &config, inputs, verify_transport, exact);
    return owner.restorePrefix(position);
}
export fn ds4_native_session_rewind_speculative41(view: *const c.ds4_session_batch_view, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, verify_transport: ?*c.ds4_tp, position: c_int, exact: bool) bool {
    const config = configuration(raw);
    const owner = sessionSpeculation(view, core, transport, &config, inputs, verify_transport, exact);
    return owner.rewindSpeculative(position);
}
export fn ds4_native_session_rewind41(view: *const c.ds4_session_batch_view, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, verify_transport: ?*c.ds4_tp, position: c_int, exact: bool) void {
    const config = configuration(raw);
    const owner = sessionSpeculation(view, core, transport, &config, inputs, verify_transport, exact);
    owner.rewind(position);
}
export fn ds4_native_session_speculative_batch41(base: *const c.ds4_session_batch_view, core: *c.ds4_engine_core, transport: *c.ds4_engine_tp_state, raw: *const c.ds4_model_config, inputs: *const c.ds4_engine_memory_inputs, verify_transport: ?*c.ds4_tp, rows: [*c]const c.ds4_session_batch_view, count: c_int, accepted: [*c][2]c_int, counts: [*c]c_int, exact: bool, native_batch: *bool, mixed_batch: *bool, errors: [*c]u8, capacity: usize) c_int {
    const config = configuration(raw);
    const session = sessionSpeculation(base, core, transport, &config, inputs, verify_transport, exact);
    const owner = SessionSpeculation.Batch{
        .ordinary = .{ .session = session.coordinator, .engine_identity = base.engine },
        .verify_transport = verify_transport,
    };
    const views: []const c.ds4_session_batch_view = if (rows == null or count <= 0) &.{} else rows[0..@intCast(count)];
    const result = owner.decode(views, accepted, counts, errorBuffer(errors, capacity));
    native_batch.* = result.native_batch;
    mixed_batch.* = result.mixed_batch;
    return result.status;
}

// The production batch selector is also reached by the existing image-route fixture.
export fn ds4_native_graph_route_batch41(graph: *c.ds41_gpu_graph, model: *const c.ds4_model, layer: *const c.ds4_layer_weights, raw: *const c.ds4_model_config, count: u32) c_int {
    const config = configuration(raw);
    const owner = @import("deepseek41/moe.zig").Moe{ .graph = graph, .ops = .{ .config = &config, .model = model } };
    return @intFromBool(owner.routeBatch(layer, count));
}
