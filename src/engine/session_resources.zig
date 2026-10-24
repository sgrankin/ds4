//! Ordinary single-Metal session resource transactions. SessionView borrows
//! the actual model states; Pool is the one engine owner of shared scratch,
//! recurrence slots and live GLM reservations. The private C Session shell
//! remains opaque because its legacy CPU/distributed layout is conditional.
const std = @import("std");
const c = @import("session_resources_c");
const View = c.ds4_session_view;
const Pool = c.ds4_session_resource_pool;
const Inputs = c.ds4_session_resource_inputs;
const accounting = @import("session_resource_accounting.zig");

fn floats(count: u32) [*c]f32 {
    const memory = c.ds4_sampling_alloc(@as(usize, count) * @sizeOf(f32)) orelse unreachable;
    return @ptrCast(@alignCast(memory));
}
fn zeroGraph() *c.ds4_qwen4_gpu_graph {
    const memory = c.calloc(1, @sizeOf(c.ds4_qwen4_gpu_graph)) orelse {
        _ = c.fputs("ds4: out of memory\n", c.__stderrp);
        c.exit(1);
    };
    return @ptrCast(@alignCast(memory));
}
fn hostBuffers(view: *View, vocabulary: u32) void {
    view.frontier.*.logits = floats(vocabulary);
    view.frontier.*.sample_probs = floats(vocabulary);
}

fn initQwen(view: *View, pool: *Pool, inputs: *const Inputs, context: u32) bool {
    const core = view.core;
    const state = view.qwen;
    const config = inputs.qwen_config;
    const capacity = if (core.*.prefill_chunk != 0 and core.*.prefill_chunk < context)
        core.*.prefill_chunk
    else
        c.ds4_native_qwen4_prefill_chunk(context);
    const block_capacity = context / 4 + 1;
    const share = core.*.share_session_prefill_workspace;
    const arena_fits = pool.qwen4_shared_workspace != null and
        pool.qwen4_shared_workspace.*.cap_tokens >= capacity and
        pool.qwen4_shared_workspace.*.n_block_cap >= block_capacity;
    if (share and pool.qwen4_shared_workspace != null and !arena_fits and pool.qwen4_arena_users == 0) {
        c.ds4_native_qwen4_graph_free(pool.qwen4_shared_workspace);
        c.free(pool.qwen4_shared_workspace);
        pool.qwen4_shared_workspace = null;
    }
    const arena = if (share and arena_fits) pool.qwen4_shared_workspace else null;
    state.*.slot = -1;
    if (share and !core.*.glm_mtp and c.getenv("DS4_QWEN4_NO_STATE_POOL") == null and
        c.ds4_native_qwen4_pool_ensure(&pool.qwen4_pool, config, @intCast(@max(inputs.session_count_hint, 0))))
        state.*.slot = c.ds4_native_qwen4_pool_take(&pool.qwen4_pool);

    const pooled = state.*.slot >= 0;
    if (!c.ds4_native_qwen4_graph_alloc(&state.*.graph, &core.*.weights, context, capacity, core.*.glm_mtp, arena, if (pooled) &pool.qwen4_pool.state else null, if (pooled) &pool.qwen4_pool.history else null, if (pooled) @intCast(state.*.slot) else 0, config)) {
        if (pooled) c.ds4_native_qwen4_pool_release(&pool.qwen4_pool, @intCast(state.*.slot));
        return false;
    }
    if (share and pool.qwen4_shared_workspace == null) {
        pool.qwen4_shared_workspace = zeroGraph();
        const bytes = c.ds4_native_qwen4_graph_scratch_bytes(&state.*.graph);
        c.ds4_native_qwen4_graph_transfer_scratch(pool.qwen4_shared_workspace, &state.*.graph);
        _ = c.fprintf(c.__stderrp, "ds4: shared Qwen3.8 session transients: cap=%u, %.2f GiB once; each further session costs its caches only\n", capacity, @as(f64, @floatFromInt(bytes)) / (1024 * 1024 * 1024));
    }
    c.ds4_native_qwen4_graph_reset(&state.*.graph);
    if (!c.ds4_native_qwen4_graph_load_steering(&state.*.graph, core.*.directional_steering_file, core.*.directional_steering_attn_scale, core.*.directional_steering_ffn_scale)) {
        if (pooled) c.ds4_native_qwen4_pool_release(&pool.qwen4_pool, @intCast(state.*.slot));
        c.ds4_native_qwen4_graph_free(&state.*.graph);
        return false;
    }
    state.*.ready = true;
    if (!state.*.graph.owns_scratch) pool.qwen4_arena_users += 1;
    view.frontier.*.prefill_cap = context;
    hostBuffers(view, config.*.shape.n_vocab);
    return true;
}

fn readVectors(path: [*c]const u8, output: [*c]f32, count: u64) bool {
    var status: c.struct_stat = undefined;
    if (c.stat(path, &status) != 0) {
        _ = c.fprintf(c.__stderrp, "ds4: failed to stat %s: %s\n", path, c.strerror(c.__error().*));
        return false;
    }
    if (status.st_size < 0 or @as(u64, @intCast(status.st_size)) != count * @sizeOf(f32)) {
        _ = c.fprintf(c.__stderrp, "ds4: %s has size %llu bytes, expected %llu bytes\n", path, @as(c_ulonglong, @bitCast(status.st_size)), @as(c_ulonglong, count * @sizeOf(f32)));
        return false;
    }
    const stream = c.fopen(path, "rb") orelse {
        _ = c.fprintf(c.__stderrp, "ds4: failed to open %s for reading: %s\n", path, c.strerror(c.__error().*));
        return false;
    };
    const read = c.fread(output, @sizeOf(f32), @intCast(count), stream);
    const closed = c.fclose(stream);
    if (read != count or closed != 0) {
        _ = c.fprintf(c.__stderrp, "ds4: failed to read %s\n", path);
        return false;
    }
    return true;
}
fn glmSteering(graph: *c.ds4_glm_gpu_graph, core: *const c.ds4_engine_core) bool {
    const attention = core.directional_steering_attn_scale;
    const ffn = core.directional_steering_ffn_scale;
    if (attention == 0 and ffn == 0) return true;
    const path = core.directional_steering_file;
    if (path == null or path[0] == 0) {
        _ = c.fputs("ds4: directional steering needs --dir-steering-file\n", c.__stderrp);
        return false;
    }
    const shape = &graph.config.shape;
    const layers = shape.n_layer - shape.n_nextn_predict;
    if (layers == 0 or layers != graph.normal_layers) return false;
    const count = @as(u64, layers) * shape.n_embd;
    const memory = c.ds4_sampling_alloc(@intCast(count * @sizeOf(f32))) orelse unreachable;
    defer c.free(memory);
    const vectors: [*c]f32 = @ptrCast(@alignCast(memory));
    var ok = readVectors(path, vectors, count);
    if (ok) {
        graph.directional_steering_dirs_by_tier[0] = c.ds4_gpu_tensor_alloc(count * @sizeOf(f32));
        ok = graph.directional_steering_dirs_by_tier[0] != null and
            c.ds4_gpu_tensor_write(graph.directional_steering_dirs_by_tier[0], 0, vectors, count * @sizeOf(f32)) != 0;
    }
    if (!ok) {
        _ = c.fprintf(c.__stderrp, "ds4: failed to load GLM directional steering vectors from %s\n", path);
        return false;
    }
    graph.directional_steering_attn_scale = attention;
    graph.directional_steering_ffn_scale = ffn;
    _ = c.fprintf(c.__stderrp, "ds4: GLM directional steering enabled: %s attn=%g ffn=%g\n", path, @as(f64, attention), @as(f64, ffn));
    return true;
}

fn initGlm(view: *View, pool: *Pool, inputs: *const Inputs, context: c_int) bool {
    const core = view.core;
    const state = view.glm;
    const shape = inputs.shape;
    const normal_layers = shape.*.n_layer - shape.*.n_nextn_predict;
    if (normal_layers == 0) {
        _ = c.fputs("ds4: invalid GLM layer slice 0:0\n", c.__stderrp);
        return false;
    }
    const last = normal_layers - 1;
    var guard_context: u32 = undefined;
    if (!c.ds4_native_glm_context_request(shape, context, &guard_context)) return false;
    const work_context = c.ds4_native_glm_full_attention_cap(guard_context, core.*.ssd_streaming);
    const planned = c.ds4_native_glm_memory(shape, guard_context, work_context, guard_context, 0, last);
    const all_graphs = accounting.graphBudget(pool.glm_session_graph_bytes, pool.glm_session_count, inputs.session_count_hint, planned.total_bytes);
    const other_graphs = all_graphs - planned.total_bytes;
    var transient = inputs.streaming_transient_guard_bytes;
    if (core.*.ssd_streaming and !core.*.memory.ssd_streaming_budget_finalized) {
        transient = c.ds4_session_resources_fit_glm_cache(inputs.engine, context);
        c.ds4_gpu_set_streaming_expert_cache_budget(core.*.memory.ssd_streaming_cache_experts);
        core.*.memory.ssd_streaming_budget_finalized = true;
    }
    const extra = transient +| other_graphs;
    const graph = &state.*.graph;
    graph.* = std.mem.zeroes(c.ds4_glm_gpu_graph);
    c.ds4_native_glm_graph_configure(graph, shape);
    graph.weights = &core.*.weights;
    graph.ssd_streaming = core.*.ssd_streaming;
    graph.ssd_streaming_cold = core.*.ssd_streaming_cold;
    graph.ctx_size = guard_context;
    if (!c.ds4_session_resources_admit_glm(core, guard_context, last, extra)) return false;
    if (!c.ds4_native_glm_plan_allocation(graph, &core.*.model, &core.*.weights, 0, last, true, true) or
        !c.ds4_native_glm_allocate_tensors(graph)) return false;
    var keep = false;
    defer if (!keep) {
        _ = c.ds4_native_glm_graph_free(graph);
    };
    view.frontier.*.prefill_cap = graph.ctx_cap;
    c.ds4_gpu_enable_q8_dequant_gemm();
    graph.quality = core.*.quality;
    if (!glmSteering(graph, core)) return false;
    if (core.*.ssd_streaming and c.getenv("DS4_METAL_GLM_DISABLE_STREAMING_SEED_BEFORE_PREFILL") == null and
        !c.ds4_session_resources_seed_glm(core)) return false;
    hostBuffers(view, shape.*.n_vocab);
    state.*.ready = true;
    state.*.reserved_graph_bytes = planned.total_bytes;
    pool.glm_session_graph_bytes +|= planned.total_bytes;
    pool.glm_session_count += 1;
    keep = true;
    return true;
}

pub fn init(view: *View, pool: *Pool, inputs: *const Inputs, context: c_int) bool {
    if (context <= 0) return false;
    view.frontier.*.ctx_size = context;
    switch (view.kind) {
        c.DS4_SESSION_DS41 => {
            const core = view.core;
            const drafter = if (core.*.dspark and core.*.dspark_weights.v41) &core.*.dspark_weights else null;
            return c.ds4_native_session_resources_init41(view.ds41, view.frontier, &core.*.memory, view.memory41, &core.*.model, &core.*.weights, view.config41, core.*.model_path, @intCast(context), core.*.prefill_chunk, core.*.quality, core.*.ssd_streaming, drafter, view.transport);
        },
        c.DS4_SESSION_FLINT => {
            view.flint.* = c.flint_session_create(inputs.flint, @intCast(context));
            if (view.flint.* == null) return false;
            view.frontier.*.prefill_cap = view.core.*.prefill_chunk;
            hostBuffers(view, @intCast(view.core.*.vocab.n_vocab));
            return true;
        },
        c.DS4_SESSION_QWEN => return initQwen(view, pool, inputs, @intCast(context)),
        c.DS4_SESSION_GLM53 => return initGlm(view, pool, inputs, context),
        else => return false,
    }
}

pub fn deinit(view: *View, pool: *Pool) void {
    switch (view.kind) {
        c.DS4_SESSION_DS41 => c.ds4_native_session_resources_free41(view.ds41, view.frontier, &view.core.*.memory),
        c.DS4_SESSION_FLINT => {
            c.flint_session_free(view.flint.*);
            view.flint.* = null;
        },
        c.DS4_SESSION_QWEN => {
            const state = view.qwen;
            if (view.core.*.glm_mtp_timing and state.*.spec_cycles != 0)
                _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 mtp: %llu verify cycles, %llu drafts accepted (%.1f%%)\n", @as(c_ulonglong, state.*.spec_cycles), @as(c_ulonglong, state.*.spec_accepted), 100 * @as(f64, @floatFromInt(state.*.spec_accepted)) / @as(f64, @floatFromInt(state.*.spec_cycles)));
            c.free(state.*.verify_logits);
            state.*.verify_logits = null;
            if (state.*.slot >= 0) c.ds4_native_qwen4_pool_release(&pool.qwen4_pool, @intCast(state.*.slot));
            if (state.*.ready and !state.*.graph.owns_scratch) pool.qwen4_arena_users -= 1;
            c.ds4_native_qwen4_graph_free(&state.*.graph);
            state.*.slot = -1;
            state.*.ready = false;
        },
        c.DS4_SESSION_GLM53 => {
            if (view.glm.*.reserved_graph_bytes != 0) {
                pool.glm_session_graph_bytes -= view.glm.*.reserved_graph_bytes;
                pool.glm_session_count -= 1;
                view.glm.*.reserved_graph_bytes = 0;
            }
            _ = c.ds4_native_glm_graph_free(&view.glm.*.graph);
            view.glm.*.ready = false;
        },
        else => {},
    }
    const frontier = view.frontier;
    c.ds4_tokens_free(&frontier.*.checkpoint);
    c.free(frontier.*.checkpoint_images);
    c.free(frontier.*.logits);
    c.free(frontier.*.sampling_logits);
    c.free(frontier.*.sample_probs);
    frontier.* = std.mem.zeroes(c.ds4_session_frontier);
    c.free(view.pending.*.hc);
    c.free(view.pending.*.logits0);
    view.pending.* = std.mem.zeroes(c.ds4_model_speculation_state);
}
