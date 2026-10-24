//! Thin C dispatch seam. Algorithms and resource ownership live in native owners.
const c = @import("gguf_c");
const resources = @import("resources.zig");
const forward = @import("forward.zig");
const mtp_owner = @import("mtp.zig");
const state_pool = @import("state_pool.zig");
const payload_owner = @import("payload.zig");
const batch_owner = @import("batch.zig");
const session_owner = @import("session.zig");
const batch_session = @import("batch_session.zig");

export fn ds4_native_qwen4_graph_alloc(graph: *c.ds4_qwen4_gpu_graph, weights: *const c.ds4_weights, context: u32, capacity: u32, mtp: bool, shared: ?*const c.ds4_qwen4_gpu_graph, states: [*c]const ?*c.ds4_gpu_tensor, histories: [*c]const ?*c.ds4_gpu_tensor, slot: u32, config: *const c.ds4_qwen4_config) bool {
    resources.allocate(graph, weights, context, capacity, mtp, shared, states, histories, slot, config) catch {
        _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 graph allocation failed (ctx %u)\n", context);
        return false;
    };
    return true;
}

export fn ds4_native_qwen4_graph_free(graph: ?*c.ds4_qwen4_gpu_graph) void {
    if (graph) |record| resources.free(record);
}

export fn ds4_native_qwen4_graph_reset(graph: *c.ds4_qwen4_gpu_graph) void {
    resources.reset(graph);
}

export fn ds4_native_qwen4_graph_transfer_scratch(destination: *c.ds4_qwen4_gpu_graph, source: *c.ds4_qwen4_gpu_graph) void {
    resources.transferScratch(destination, source);
}

export fn ds4_native_qwen4_graph_scratch_bytes(graph: *const c.ds4_qwen4_gpu_graph) u64 {
    return resources.scratchBytes(graph);
}

export fn ds4_native_qwen4_graph_load_steering(graph: ?*c.ds4_qwen4_gpu_graph, path: [*c]const u8, attention_scale: f32, ffn_scale: f32) bool {
    return resources.loadSteering(graph orelse return false, path, attention_scale, ffn_scale);
}

export fn ds4_native_qwen4_graph_forward_tokens(graph: *c.ds4_qwen4_gpu_graph, model: *const c.ds4_model, weights: *const c.ds4_weights, tokens: [*c]const c_int, count: u32, output: [*c]f32, all_rows: bool) bool {
    return forward.run(graph, model, weights, tokens[0..count], output, all_rows);
}

export fn ds4_native_qwen4_graph_mtp_steps(graph: *c.ds4_qwen4_gpu_graph, model: *const c.ds4_model, weights: *const c.ds4_weights, row: u32, tokens: [*c]const c_int, count: u32, position: u32, want_logits: bool, output: [*c]f32, draft: [*c]c_int) bool {
    return mtp_owner.steps(graph, model, weights, row, tokens[0..count], position, want_logits, output, draft);
}

export fn ds4_native_qwen4_pool_ensure(pool: *c.ds4_qwen4_state_pool, config: *const c.ds4_qwen4_config, slots: u32) bool {
    return state_pool.ensure(pool, config, slots);
}

export fn ds4_native_qwen4_pool_take(pool: *c.ds4_qwen4_state_pool) c_int {
    return state_pool.take(pool);
}

export fn ds4_native_qwen4_pool_release(pool: *c.ds4_qwen4_state_pool, slot: u32) void {
    state_pool.release(pool, slot);
}

export fn ds4_native_qwen4_pool_free(pool: *c.ds4_qwen4_state_pool) void {
    state_pool.free(pool);
}

fn errors(pointer: [*c]u8, capacity: usize) []u8 {
    return if (pointer == null or capacity == 0) &.{} else pointer[0..capacity];
}

export fn ds4_native_qwen4_payload_tensor_bytes(config: *const c.ds4_qwen4_config, rows: u32, mtp_rows: u32) u64 {
    return payload_owner.tensorBytes(config, rows, mtp_rows);
}

export fn ds4_native_qwen4_payload_save(graph: *c.ds4_qwen4_gpu_graph, state: *const c.ds4_session_frontier, ready: bool, rewound: bool, file: *c.FILE, error_text: [*c]u8, capacity: usize) c_int {
    return payload_owner.save(graph, state, ready, rewound, file, errors(error_text, capacity));
}

export fn ds4_native_qwen4_payload_load(graph: *c.ds4_qwen4_gpu_graph, state: *c.ds4_session_frontier, ready: bool, file: *c.FILE, header: *const [c.DS4_SESSION_PAYLOAD_U32_FIELDS]u32, remaining: *u64, mutated: *bool, error_text: [*c]u8, capacity: usize) c_int {
    return payload_owner.load(graph, state, ready, file, header, remaining, mutated, errors(error_text, capacity));
}

export fn ds4_native_qwen4_batch_supported(arena: ?*const c.ds4_qwen4_gpu_graph, members: [*c]const c.ds4_qwen4_batch_member, count: u32, speculative: bool) bool {
    return batch_owner.supported(arena, members[0..count], speculative);
}

export fn ds4_native_qwen4_batch_encode(members: [*c]const c.ds4_qwen4_batch_member, count: u32, arena: *c.ds4_qwen4_gpu_graph, pool: ?*const c.ds4_qwen4_state_pool, model: *const c.ds4_model, weights: *const c.ds4_weights) bool {
    return batch_owner.encode(members[0..count], arena, pool, model, weights);
}

export fn ds4_native_qwen4_session_sync(runtime: *c.ds4_qwen4_runtime, prompt: *const c.ds4_tokens, error_text: [*c]u8, capacity: usize) c_int {
    return session_owner.sync(runtime, prompt, errors(error_text, capacity));
}

export fn ds4_native_qwen4_session_eval(runtime: *c.ds4_qwen4_runtime, token: c_int, error_text: [*c]u8, capacity: usize) c_int {
    return session_owner.eval(runtime, token, errors(error_text, capacity));
}

export fn ds4_native_qwen4_session_rewind(state: *c.ds4_qwen4_session_state, frontier: *c.ds4_session_frontier, position: u32) void {
    session_owner.rewind(state, frontier, position);
}

export fn ds4_native_qwen4_session_speculate(runtime: *c.ds4_qwen4_runtime, first: c_int, temperature: f32, top_k: c_int, top_p: f32, min_p: f32, rng: [*c]u64, exact_sampling: bool, accepted: [*c]c_int, accepted_capacity: c_int, error_text: [*c]u8, capacity: usize) c_int {
    return session_owner.speculate(runtime, first, temperature, top_k, top_p, min_p, rng, exact_sampling, accepted[0..@intCast(accepted_capacity)], errors(error_text, capacity));
}

export fn ds4_native_qwen4_batch_speculate(runtimes: [*c]c.ds4_qwen4_runtime, tokens: [*c]const c_int, count: u32, arena: *c.ds4_qwen4_gpu_graph, policy: *c.ds4_qwen4_batch_policy, accepted: [*c][2]c_int, counts: [*c]c_int, error_text: [*c]u8, capacity: usize) c_int {
    return batch_session.speculate(runtimes[0..count], tokens[0..count], arena, policy, accepted[0..count], counts[0..count], errors(error_text, capacity));
}
