//! Stateless Qwen inference retains one private graph, no MTP allocations and
//! a head only on the final prefill chunk, matching the specialized C path.
const c = @import("generation_c");
const Generation = @import("../engine_generation.zig");

pub fn generate(engine: *c.ds4_engine, core: *const c.ds4_engine_core, config: *const c.ds4_qwen4_config, prompt: *const c.ds4_tokens, maximum: c_int, context: c_int, output: Generation.Output) c_int {
    if (prompt.len <= 0 or prompt.len >= context) {
        _ = c.fputs("ds4: prompt is empty or exceeds context size\n", c.__stderrp);
        return 1;
    }
    const memory = c.calloc(1, @sizeOf(c.ds4_qwen4_gpu_graph)) orelse {
        _ = c.fputs("ds4: out of memory\n", c.__stderrp);
        c.exit(1);
    };
    const graph: *c.ds4_qwen4_gpu_graph = @ptrCast(@alignCast(memory));
    defer c.free(memory);
    if (!c.ds4_native_qwen4_graph_alloc(graph, &core.weights, @intCast(context), c.ds4_native_qwen4_prefill_chunk(@intCast(context)), false, null, null, null, 0, config)) return 1;
    defer c.ds4_native_qwen4_graph_free(graph);
    if (!c.ds4_native_qwen4_graph_load_steering(graph, core.directional_steering_file, core.directional_steering_attn_scale, core.directional_steering_ffn_scale)) return 1;
    c.ds4_native_qwen4_graph_reset(graph);
    const allocation = c.ds4_sampling_alloc(@as(usize, config.shape.n_vocab) * @sizeOf(f32)) orelse unreachable;
    defer c.free(allocation);
    const logits: [*]f32 = @ptrCast(@alignCast(allocation));
    const prefill_start = Generation.now();
    var index: c_int = 0;
    while (index < prompt.len) {
        const count: u32 = @min(@as(u32, @intCast(prompt.len - index)), graph.cap_tokens);
        const last = index + @as(c_int, @intCast(count)) == prompt.len;
        const ok = c.ds4_native_qwen4_graph_forward_tokens(graph, &core.model, &core.weights, prompt.v + @as(usize, @intCast(index)), count, if (last) logits else null, false);
        index += @intCast(count);
        if (output.progress) |callback| callback(output.progress_user, "prefill_chunk", index, prompt.len);
        if (!ok) {
            _ = c.fputs("ds4: Qwen3.8 prefill failed\n", c.__stderrp);
            return 1;
        }
    }
    const prefill_end = Generation.now();
    var generated: c_int = 0;
    const start = Generation.now();
    while (generated < maximum and graph.pos < graph.ctx_cap) {
        const token = c.ds4_sampling_argmax(logits, config.shape.n_vocab);
        if (c.ds4_token_is_stop(engine, token)) break;
        output.token(token);
        generated += 1;
        if (generated >= maximum or graph.pos + 1 >= graph.ctx_cap) break;
        if (!c.ds4_native_qwen4_graph_forward_tokens(graph, &core.model, &core.weights, &token, 1, logits, false)) {
            _ = c.fprintf(c.__stderrp, "ds4: Qwen3.8 decode failed at position %u\n", graph.pos);
            return 1;
        }
    }
    const end = Generation.now();
    output.finish();
    Generation.timing("Qwen3.8 ", prompt.len, generated, prefill_end - prefill_start, end - start);
    return 0;
}
